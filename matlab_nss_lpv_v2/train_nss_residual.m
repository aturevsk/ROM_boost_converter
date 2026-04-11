function train_nss_residual(seed, opts)
% train_nss_residual  NSS with residual LPV: x' = (A(u)+dA(x,u))*x + B(u)*u + c(u)
%
% Phase 1: A(u), B(u), c(u) frozen as lookup tables in functionLayer closure.
%          Only dA MLP is trained. Preserves known linear dynamics.
% Phase 2: Full LPV network (all trainable), initialized from Phase 1.
% Phase 3: Fine-tune with lower LR.
%
% Architecture enforces physics: model starts with exact linear dynamics
% at all 12 operating points. dA learns only the nonlinear correction.
%
% Usage:
%   train_nss_residual(7)
%   train_nss_residual(0, 'test', true)

arguments
    seed (1,1) double = 7
    opts.test (1,1) logical = false
end

thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fileparts(thisDir);
addpath(genpath(repoRoot));

TS = 5e-6; NX = 2; H = 64;
VAL_IDX = [5, 12, 14];
CHUNK = 10;  % epochs per nlssest call

CP_DIR = fullfile(thisDir, 'checkpoints', sprintf('run_%04d', seed));
if ~isfolder(CP_DIR), mkdir(CP_DIR); end
rng(seed);

fprintf('=== NSS Residual LPV — Seed %d ===\n', seed);
fprintf('Architecture: dx/dt = (A(u) + dA(x,u))*x + B(u)*u + c(u)\n');
fprintf('Phase 1: dA only (A,B,c frozen) | Phase 2-3: full LPV\n\n');

%% ── Load data & compute normalization ──────────────────────────────────────
csvDir = fullfile(repoRoot, 'data', 'neural_ode');
csvFiles = dir(fullfile(csvDir, 'profile_*.csv'));
[~, ord] = sort({csvFiles.name}); csvFiles = csvFiles(ord);
nTotal = numel(csvFiles);

profiles_u = cell(nTotal,1); profiles_y = cell(nTotal,1);
for k = 1:nTotal
    raw = readmatrix(fullfile(csvDir, csvFiles(k).name));
    profiles_u{k} = raw(:,1); profiles_y{k} = raw(:,2:3);
end

TRAIN_IDX = setdiff(1:nTotal, VAL_IDX); nTrain = numel(TRAIN_IDX);
all_u=[]; all_y=[]; all_dv=[]; all_di=[];
for i = TRAIN_IDX
    all_u=[all_u;profiles_u{i}]; all_y=[all_y;profiles_y{i}];
    all_dv=[all_dv;diff(profiles_y{i}(:,1))/TS]; all_di=[all_di;diff(profiles_y{i}(:,2))/TS];
end
ns.u_mean=mean(all_u); ns.u_std=std(all_u);
ns.y_mean=mean(all_y); ns.y_std=std(all_y);
ns.dxdt_scale=[std(all_dv), std(all_di)];
% Make column vectors for consistency
ns.x_mean = ns.y_mean(:); ns.x_std = ns.y_std(:); ns.dxdt_scale = ns.dxdt_scale(:);
save(fullfile(CP_DIR,'norm_stats.mat'),'-struct','ns');
fprintf('  Norm: x_mean=[%.2f,%.2f] x_std=[%.2f,%.2f] dxdt=[%.0f,%.0f]\n', ...
    ns.x_mean, ns.x_std, ns.dxdt_scale);

% Prepare iddata
trainData = cell(nTrain,1);
for i=1:nTrain
    idx=TRAIN_IDX(i);
    un=(profiles_u{idx}-ns.u_mean)/ns.u_std;
    yn=(profiles_y{idx}-ns.y_mean)./ns.y_std;
    trainData{i}=iddata(yn, un, TS);
end
trainDataMerged = merge(trainData{:});

valData = cell(numel(VAL_IDX),1);
for i=1:numel(VAL_IDX)
    idx=VAL_IDX(i);
    un=(profiles_u{idx}-ns.u_mean)/ns.u_std;
    yn=(profiles_y{idx}-ns.y_mean)./ns.y_std;
    valData{i}=iddata(yn, un, TS);
end
fprintf('  Train: %d profiles | Val: %d (profiles %s)\n', nTrain, numel(VAL_IDX), mat2str(VAL_IDX));

%% ── Build lookup tables ────────────────────────────────────────────────────
lut = build_lookup_tables(repoRoot, ns);
fprintf('  Lookup tables: A,B,c at %d duty points in normalized space\n', lut.nPts);

%% ══════════════════════════════════════════════════════════════════════════
%% PHASE 1: Train dA only (A,B,c frozen in closure)
%% ══════════════════════════════════════════════════════════════════════════

% Build Phase 1 StateNetwork: dA MLP + frozen LPV compute
lgraph = layerGraph();
lgraph = addLayers(lgraph, featureInputLayer(NX, 'Name', 'x'));
lgraph = addLayers(lgraph, featureInputLayer(1, 'Name', 'u'));
lgraph = addLayers(lgraph, concatenationLayer(1, 2, 'Name', 'concat'));

% dA MLP: [x_n, u_n] -> 4 values (dA matrix entries)
lgraph = addLayers(lgraph, fullyConnectedLayer(H, 'Name', 'fc1'));
lgraph = addLayers(lgraph, tanhLayer('Name', 'act1'));
lgraph = addLayers(lgraph, fullyConnectedLayer(H, 'Name', 'fc2'));
lgraph = addLayers(lgraph, tanhLayer('Name', 'act2'));
lgraph = addLayers(lgraph, fullyConnectedLayer(4, 'Name', 'fc_dA'));  % 4 outputs for dA(2x2)

% LPV compute: (A(u)+dA)*x + B(u)*u + c(u)
% A,B,c captured in closure — NOT trainable
D_grid = lut.D_grid; A_norm = lut.A_norm; B_norm = lut.B_norm; c_norm = lut.c_norm;
u_mean_c = ns.u_mean; u_std_c = ns.u_std;

lpvFcn = @(dA_flat, xn, un) lpv_phase1_compute(dA_flat, xn, un, ...
    D_grid, A_norm, B_norm, c_norm, u_mean_c, u_std_c);

lgraph = addLayers(lgraph, functionLayer(lpvFcn, ...
    'Name', 'dxdt', 'NumInputs', 3, 'Formattable', true));

% Wire
lgraph = connectLayers(lgraph, 'x', 'concat/in1');
lgraph = connectLayers(lgraph, 'u', 'concat/in2');
lgraph = connectLayers(lgraph, 'concat', 'fc1');
lgraph = connectLayers(lgraph, 'fc1', 'act1');
lgraph = connectLayers(lgraph, 'act1', 'fc2');
lgraph = connectLayers(lgraph, 'fc2', 'act2');
lgraph = connectLayers(lgraph, 'act2', 'fc_dA');
lgraph = connectLayers(lgraph, 'fc_dA', 'dxdt/in1');
lgraph = connectLayers(lgraph, 'x', 'dxdt/in2');
lgraph = connectLayers(lgraph, 'u', 'dxdt/in3');

stateNet1 = dlnetwork(lgraph);

% Initialize fc1 from operating points (same as before)
jacFile = fullfile(repoRoot,'data','neural_ode','jacobian_targets.json');
jd = jsondecode(fileread(jacFile));
x_ops = single(jd.x_ops); u_ops = single(jd.u_ops(:));
nPts = numel(u_ops); neurPerPt = min(4, H/nPts);

fc1_w = find(strcmp(stateNet1.Learnables.Layer,'fc1') & contains(stateNet1.Learnables.Parameter,'Weights'));
fc1_b = find(strcmp(stateNet1.Learnables.Layer,'fc1') & contains(stateNet1.Learnables.Parameter,'Bias'));
W1 = zeros(H, NX+1, 'single'); b1 = zeros(H, 1, 'single');
for i=1:nPts
    xn=(x_ops(i,:)'-ns.x_mean)./ns.x_std;
    un=(u_ops(i)-ns.u_mean)/ns.u_std;
    xu=[xn;un];
    for j=1:neurPerPt
        row=(i-1)*neurPerPt+j; if row>H, break; end
        W1(row,:)=xu'*(0.5+0.5*(j-1));
        b1(row)=-dot(W1(row,:),xu)*(0.3+0.2*(j-1));
    end
end
filled=nPts*neurPerPt;
if filled<H, W1(filled+1:end,:)=0.05*randn(H-filled,NX+1,'single'); b1(filled+1:end)=0.05*randn(H-filled,1,'single'); end
stateNet1.Learnables.Value{fc1_w} = dlarray(W1);
stateNet1.Learnables.Value{fc1_b} = dlarray(b1);

% Initialize fc_dA to ZERO — dA starts at exactly zero
fc_dA_w = find(strcmp(stateNet1.Learnables.Layer,'fc_dA') & contains(stateNet1.Learnables.Parameter,'Weights'));
fc_dA_b = find(strcmp(stateNet1.Learnables.Layer,'fc_dA') & contains(stateNet1.Learnables.Parameter,'Bias'));
stateNet1.Learnables.Value{fc_dA_w} = dlarray(zeros(4, H, 'single'));
stateNet1.Learnables.Value{fc_dA_b} = dlarray(zeros(4, 1, 'single'));

nParams = sum(cellfun(@numel, {stateNet1.Learnables.Value{:}}));
fprintf('  Phase 1 network: %d params (dA MLP only, A/B/c frozen in closure)\n', nParams);
fprintf('  fc_dA initialized to ZERO -> model starts with exact linear dynamics\n');

% Assign to NSS
nss = idNeuralStateSpace(NX, NumInputs=1, NumOutputs=NX, Ts=TS);  % DT mode
nss.StateNetwork = stateNet1;

%% ── Train Phase 1 ──────────────────────────────────────────────────────────
if opts.test
    ph1_epochs = 10; ph1_lr = 1e-3;
else
    ph1_epochs = 1000;  % generous — will early stop if no improvement
    ph1_lr = 1e-3;
end
winSamples = round(5e-3 / TS);  % 5ms windows

tStart = tic;
bestVal = inf; bestNSS = nss; noImprove = 0; currentLR = ph1_lr;
nChunks = ceil(ph1_epochs / CHUNK);

fprintf('\n--- Phase 1: dA only | %d epochs (chunks of %d) | win=5ms | LR=%.1e ---\n', ...
    ph1_epochs, CHUNK, ph1_lr);

epochsDone = 0;
for ch = 1:nChunks
    epThis = min(CHUNK, ph1_epochs - epochsDone);

    trainOpts = nssTrainingOptions('adam');
    trainOpts.MaxEpochs = epThis;
    trainOpts.LearnRate = currentLR;
    trainOpts.LearnRateSchedule = 'piecewise';
    trainOpts.LearnRateDropPeriod = 100;
    trainOpts.LearnRateDropFactor = 0.5;
    trainOpts.WindowSize = winSamples;
    trainOpts.NumWindowFraction = eps;
    trainOpts.LossFcn = 'MeanSquaredError';
    trainOpts.Lambda = 1e-4;

    tChunk = tic;
    try
        nss = nlssest(trainDataMerged, nss, trainOpts);
    catch ME
        if contains(ME.message, 'NaN')
            currentLR = currentLR * 0.5;
            fprintf('  [!] NaN at ep %d. LR->%.1e. Reloading best.\n', epochsDone+epThis, currentLR);
            if isfile(fullfile(CP_DIR,'best.mat'))
                cp=load(fullfile(CP_DIR,'best.mat')); nss=cp.bestNSS;
            end
            if currentLR<1e-6, fprintf('  LR floor, ending phase.\n'); break; end
            continue;
        else, rethrow(ME); end
    end
    chunkTime = toc(tChunk);
    epochsDone = epochsDone + epThis;

    % Validate
    [valLoss, fpVout, fpIL] = validateNSS(nss, valData, profiles_u, profiles_y, VAL_IDX, ns);

    elapsed = toc(tStart)/3600;
    imp = '';
    if valLoss < bestVal
        bestVal = valLoss; bestNSS = nss; noImprove = 0; imp = ' ** BEST **';
        save(fullfile(CP_DIR,'best.mat'),'nss','bestNSS','bestVal','ns','-v7.3');
        fprintf('    [cp] best.mat\n');
    else
        noImprove = noImprove + 1;
    end
    fprintf('  [Ph1] Ep %d/%d | Val: %.4f | FP: V=%.3f iL=%.3f | %.0fs/chunk | %.2fh | noImp=%d%s\n', ...
        epochsDone, ph1_epochs, valLoss, fpVout, fpIL, chunkTime, elapsed, noImprove, imp);

    if noImprove >= 10 && ~opts.test
        fprintf('  Phase 1 early stop: no improvement for %d chunks\n', noImprove);
        break;
    end
end
save(fullfile(CP_DIR,'after_Phase1.mat'),'nss','bestNSS','bestVal','ns','-v7.3');
fprintf('  Phase 1 done. Best val: %.4f\n', bestVal);

if opts.test
    [~, fpV, fpI] = validateNSS(bestNSS, valData, profiles_u, profiles_y, VAL_IDX, ns);
    fprintf('\n=== TEST Complete | FP: V=%.4f iL=%.4f ===\n', fpV, fpI);
    return;
end

%% ══════════════════════════════════════════════════════════════════════════
%% PHASE 2: Full LPV (all trainable), initialized from Phase 1
%% ══════════════════════════════════════════════════════════════════════════
fprintf('\n--- Phase 2: Full LPV | 500 epochs | win=20ms | LR=5e-4 ---\n');
nss = bestNSS;

% Build Phase 2 network: full LPV (8 outputs, same as previous NSS LPV)
lgraph2 = layerGraph();
lgraph2 = addLayers(lgraph2, featureInputLayer(NX, 'Name', 'x'));
lgraph2 = addLayers(lgraph2, featureInputLayer(1, 'Name', 'u'));
lgraph2 = addLayers(lgraph2, concatenationLayer(1, 2, 'Name', 'concat'));
lgraph2 = addLayers(lgraph2, fullyConnectedLayer(H, 'Name', 'fc1'));
lgraph2 = addLayers(lgraph2, tanhLayer('Name', 'act1'));
lgraph2 = addLayers(lgraph2, fullyConnectedLayer(H, 'Name', 'fc2'));
lgraph2 = addLayers(lgraph2, tanhLayer('Name', 'act2'));
lgraph2 = addLayers(lgraph2, fullyConnectedLayer(8, 'Name', 'fc_raw'));  % 8 outputs: A(4)+B(2)+c(2)

lpvFcn2 = @(raw, xn, un) ...
    [raw(1,:).*xn(1,:)+raw(2,:).*xn(2,:)+raw(5,:).*un+raw(7,:); ...
     raw(3,:).*xn(1,:)+raw(4,:).*xn(2,:)+raw(6,:).*un+raw(8,:)];

lgraph2 = addLayers(lgraph2, functionLayer(lpvFcn2, ...
    'Name', 'dxdt', 'NumInputs', 3, 'Formattable', true));

lgraph2 = connectLayers(lgraph2, 'x', 'concat/in1');
lgraph2 = connectLayers(lgraph2, 'u', 'concat/in2');
lgraph2 = connectLayers(lgraph2, 'concat', 'fc1');
lgraph2 = connectLayers(lgraph2, 'fc1', 'act1');
lgraph2 = connectLayers(lgraph2, 'act1', 'fc2');
lgraph2 = connectLayers(lgraph2, 'fc2', 'act2');
lgraph2 = connectLayers(lgraph2, 'act2', 'fc_raw');
lgraph2 = connectLayers(lgraph2, 'fc_raw', 'dxdt/in1');
lgraph2 = connectLayers(lgraph2, 'x', 'dxdt/in2');
lgraph2 = connectLayers(lgraph2, 'u', 'dxdt/in3');

stateNet2 = dlnetwork(lgraph2);

% Initialize from Phase 1: copy fc1, fc2 weights
ph1_net = nss.StateNetwork;
for k = 1:height(stateNet2.Learnables)
    layerName = stateNet2.Learnables.Layer{k};
    paramName = stateNet2.Learnables.Parameter{k};
    if any(strcmp(layerName, {'fc1','fc2'}))
        % Find matching in Phase 1
        idx1 = find(strcmp(ph1_net.Learnables.Layer, layerName) & ...
                    strcmp(ph1_net.Learnables.Parameter, paramName));
        if ~isempty(idx1)
            stateNet2.Learnables.Value{k} = ph1_net.Learnables.Value{idx1};
        end
    end
end

% Initialize fc_raw: first 4 rows (A part) = Phase 1 fc_dA weights + scale for A(u) baseline
% For now, initialize fc_raw small (0.01 scale) to avoid NaN
fc_raw_w = find(strcmp(stateNet2.Learnables.Layer,'fc_raw') & contains(stateNet2.Learnables.Parameter,'Weights'));
fc_raw_b = find(strcmp(stateNet2.Learnables.Layer,'fc_raw') & contains(stateNet2.Learnables.Parameter,'Bias'));
W_raw = extractdata(stateNet2.Learnables.Value{fc_raw_w});
b_raw = extractdata(stateNet2.Learnables.Value{fc_raw_b});

% Copy dA weights to first 4 rows of fc_raw
idx_dA_w = find(strcmp(ph1_net.Learnables.Layer,'fc_dA') & contains(ph1_net.Learnables.Parameter,'Weights'));
idx_dA_b = find(strcmp(ph1_net.Learnables.Layer,'fc_dA') & contains(ph1_net.Learnables.Parameter,'Bias'));
if ~isempty(idx_dA_w)
    W_raw(1:4,:) = extractdata(ph1_net.Learnables.Value{idx_dA_w});
    b_raw(1:4) = extractdata(ph1_net.Learnables.Value{idx_dA_b});
end
% Rows 5-8 (B, c parts): small init
W_raw(5:8,:) = 0.01 * randn(4, H, 'single');
b_raw(5:8) = 0.01 * randn(4, 1, 'single');

stateNet2.Learnables.Value{fc_raw_w} = dlarray(W_raw);
stateNet2.Learnables.Value{fc_raw_b} = dlarray(b_raw);

fprintf('  Phase 2 network: %d params (all trainable), initialized from Phase 1\n', ...
    sum(cellfun(@numel, {stateNet2.Learnables.Value{:}})));

nss2 = idNeuralStateSpace(NX, NumInputs=1, NumOutputs=NX, Ts=TS);
nss2.StateNetwork = stateNet2;

% Train Phase 2
winSamples = round(20e-3 / TS);  % 20ms windows
ph2_epochs = 500; currentLR = 5e-4; noImprove = 0;
nChunks = ceil(ph2_epochs / CHUNK);
epochsDone = 0;

for ch = 1:nChunks
    epThis = min(CHUNK, ph2_epochs - epochsDone);
    trainOpts = nssTrainingOptions('adam');
    trainOpts.MaxEpochs = epThis;
    trainOpts.LearnRate = currentLR;
    trainOpts.LearnRateSchedule = 'piecewise';
    trainOpts.LearnRateDropPeriod = 150;
    trainOpts.LearnRateDropFactor = 0.5;
    trainOpts.WindowSize = winSamples;
    trainOpts.NumWindowFraction = eps;
    trainOpts.Lambda = 1e-4;

    tChunk = tic;
    try
        nss2 = nlssest(trainDataMerged, nss2, trainOpts);
    catch ME
        if contains(ME.message, 'NaN')
            currentLR = currentLR * 0.5;
            fprintf('  [!] NaN. LR->%.1e. Reloading best.\n', currentLR);
            if isfile(fullfile(CP_DIR,'best.mat'))
                cp=load(fullfile(CP_DIR,'best.mat')); nss2=cp.bestNSS;
            end
            if currentLR<1e-6, break; end
            continue;
        else, rethrow(ME); end
    end
    chunkTime = toc(tChunk);
    epochsDone = epochsDone + epThis;

    [valLoss, fpVout, fpIL] = validateNSS(nss2, valData, profiles_u, profiles_y, VAL_IDX, ns);
    elapsed = toc(tStart)/3600;
    imp = '';
    if valLoss < bestVal
        bestVal=valLoss; bestNSS=nss2; noImprove=0; imp=' ** BEST **';
        save(fullfile(CP_DIR,'best.mat'),'nss','bestNSS','bestVal','ns','-v7.3');
        fprintf('    [cp] best.mat\n');
    else, noImprove=noImprove+1; end
    fprintf('  [Ph2] Ep %d/%d | Val: %.4f | FP: V=%.3f iL=%.3f | %.0fs | %.2fh | noImp=%d%s\n', ...
        epochsDone, ph2_epochs, valLoss, fpVout, fpIL, chunkTime, elapsed, noImprove, imp);
    if noImprove>=10, fprintf('  Phase 2 early stop\n'); break; end
end
save(fullfile(CP_DIR,'after_Phase2.mat'),'nss2','bestNSS','bestVal','ns','-v7.3');

%% ══════════════════════════════════════════════════════════════════════════
%% PHASE 3: Fine-tune
%% ══════════════════════════════════════════════════════════════════════════
fprintf('\n--- Phase 3: Fine-tune | 200 epochs | win=20ms | LR=1e-4 ---\n');
nss2 = bestNSS;
ph3_epochs = 200; currentLR = 1e-4; noImprove = 0;
nChunks = ceil(ph3_epochs / CHUNK); epochsDone = 0;

for ch = 1:nChunks
    epThis = min(CHUNK, ph3_epochs - epochsDone);
    trainOpts = nssTrainingOptions('adam');
    trainOpts.MaxEpochs = epThis;
    trainOpts.LearnRate = currentLR;
    trainOpts.LearnRateSchedule = 'piecewise';
    trainOpts.LearnRateDropPeriod = 50;
    trainOpts.LearnRateDropFactor = 0.5;
    trainOpts.WindowSize = winSamples;
    trainOpts.NumWindowFraction = eps;
    trainOpts.Lambda = 1e-4;

    tChunk = tic;
    try
        nss2 = nlssest(trainDataMerged, nss2, trainOpts);
    catch ME
        if contains(ME.message,'NaN')
            currentLR=currentLR*0.5;
            fprintf('  [!] NaN. LR->%.1e\n',currentLR);
            if isfile(fullfile(CP_DIR,'best.mat')),cp=load(fullfile(CP_DIR,'best.mat'));nss2=cp.bestNSS;end
            if currentLR<1e-6,break;end
            continue;
        else,rethrow(ME);end
    end
    chunkTime=toc(tChunk); epochsDone=epochsDone+epThis;

    [valLoss,fpVout,fpIL]=validateNSS(nss2,valData,profiles_u,profiles_y,VAL_IDX,ns);
    elapsed=toc(tStart)/3600; imp='';
    if valLoss<bestVal
        bestVal=valLoss;bestNSS=nss2;noImprove=0;imp=' ** BEST **';
        save(fullfile(CP_DIR,'best.mat'),'nss2','bestNSS','bestVal','ns','-v7.3');
        fprintf('    [cp] best.mat\n');
    else,noImprove=noImprove+1;end
    fprintf('  [Ph3] Ep %d/%d | Val: %.4f | FP: V=%.3f iL=%.3f | %.0fs | %.2fh | noImp=%d%s\n', ...
        epochsDone,ph3_epochs,valLoss,fpVout,fpIL,chunkTime,elapsed,noImprove,imp);
    if noImprove>=10,fprintf('  Phase 3 early stop\n');break;end
end

%% ── Final ──────────────────────────────────────────────────────────────────
totalH = toc(tStart)/3600;
[~,fpV,fpI] = validateNSS(bestNSS,valData,profiles_u,profiles_y,VAL_IDX,ns);
save(fullfile(CP_DIR,'final.mat'),'bestNSS','bestVal','ns','-v7.3');

fprintf('\n=== NSS Residual LPV Seed %d Complete ===\n', seed);
fprintf('  Time: %.2f h | Best val: %.6f\n', totalH, bestVal);
fprintf('  FP RMSE: Vout=%.4fV iL=%.4fA\n', fpV, fpI);
fprintf('  Previous NSS LPV: Vout=0.378V iL=1.696A\n');
fprintf('  DLT LPV target:   Vout=0.019V iL=0.067A\n');
end


%% ========================================================================
%% Phase 1 LPV compute: (A(u)+dA)*x + B(u)*u + c(u)
%% A,B,c from lookup tables (frozen), dA from MLP (trainable)
%% ========================================================================
function dxdt = lpv_phase1_compute(dA_flat, xn, un, D_grid, A_norm, B_norm, c_norm, u_mean, u_std)
% dA_flat: [4 x B] from MLP
% xn: [2 x B] normalized state
% un: [1 x B] normalized input
% D_grid, A_norm, B_norm, c_norm: lookup table data (closure constants)

% Recover physical duty for interpolation
D = un * u_std + u_mean;  % denormalize duty

% Interpolate A, B, c at current duty
% Simple linear interpolation along D_grid
nPts = numel(D_grid);
B = size(xn, 2);  % batch size

dxdt = zeros(2, B, 'like', xn);
for b = 1:B
    d = extractdata(D(1,b));
    % Clamp to grid range
    d = max(D_grid(1), min(D_grid(end), d));
    % Find interpolation interval
    idx = find(D_grid <= d, 1, 'last');
    if idx >= nPts, idx = nPts - 1; end
    alpha = (d - D_grid(idx)) / (D_grid(idx+1) - D_grid(idx) + eps);

    A_u = (1-alpha) * A_norm(:,:,idx) + alpha * A_norm(:,:,idx+1);
    B_u = (1-alpha) * B_norm(:,:,idx) + alpha * B_norm(:,:,idx+1);
    c_u = (1-alpha) * c_norm(:,:,idx) + alpha * c_norm(:,:,idx+1);

    % dA from MLP output
    dA = [dA_flat(1,b), dA_flat(2,b); dA_flat(3,b), dA_flat(4,b)];

    % (A(u) + dA) * x + B(u) * u + c(u)
    dxdt(:,b) = (A_u + dA) * xn(:,b) + B_u * un(:,b) + c_u;
end
end


%% ========================================================================
%% Validation helper
%% ========================================================================
function [valLoss, fpVout, fpIL] = validateNSS(nss, valData, profiles_u, profiles_y, VAL_IDX, ns)
valLoss = 0;
for v = 1:numel(valData)
    yp = sim(nss, valData{v});
    if isa(yp,'iddata'), yp = yp.OutputData; end
    yt = valData{v}.OutputData;
    valLoss = valLoss + mean((yp - yt).^2, 'all');
end
valLoss = valLoss / numel(valData);

fpv = []; fpi = [];
for v = 1:numel(VAL_IDX)
    idx = VAL_IDX(v);
    un = (profiles_u{idx} - ns.u_mean) / ns.u_std;
    yn = (profiles_y{idx} - ns.y_mean) ./ ns.y_std;
    vd = iddata(yn, un, 5e-6);
    yp = sim(nss, vd);
    if isa(yp,'iddata'), yp = yp.OutputData; end
    yp_phys = yp .* ns.y_std' + ns.y_mean';
    y_phys = profiles_y{idx};
    fpv(end+1) = sqrt(mean((yp_phys(:,1)-y_phys(:,1)).^2)); %#ok<AGROW>
    fpi(end+1) = sqrt(mean((yp_phys(:,2)-y_phys(:,2)).^2)); %#ok<AGROW>
end
fpVout = mean(fpv); fpIL = mean(fpi);
end
