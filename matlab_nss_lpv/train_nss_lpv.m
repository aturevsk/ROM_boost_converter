function train_nss_lpv(seed, opts)
% train_nss_lpv  Neural State Space with LPV architecture.
%
% Uses idNeuralStateSpace + nlssest with a custom LPV-structured dlnetwork:
%   dx/dt = A(x,u)*x + B(x,u)*u + c(x,u)
%
% The MLP outputs 8 values [A(4), B(2), c(2)], then a functionLayer
% computes A*x + B*u + c via skip connections from the x and u inputs.
% This is the same architecture as DLT LPV, but trained by nlssest
% (no custom loss function — standard MSE only).
%
% Applies all compatible tricks from DLT LPV:
%   - LPV architecture (same 4,936 params)
%   - Physics-informed init from 12 Branch B A-matrices
%   - Output scaling (dxdt_std/x_std ratio)
%   - NumWindowFraction=eps (stochastic per-window updates)
%   - CT mode (Ts=0, dlode45 solver)
%   - 3-phase curriculum (5ms -> 20ms -> 20ms fine-tune)
%
% Usage:
%   train_nss_lpv(7)                    % seed 7 (matching DLT LPV)
%   train_nss_lpv(7, 'maxEpochs', 500) % custom epochs per phase
%   train_nss_lpv(0, 'test', true)      % smoke test

arguments
    seed (1,1) double = 7
    opts.maxEpochs (1,1) double = 500
    opts.test (1,1) logical = false
end

thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fileparts(thisDir);
addpath(genpath(repoRoot));

TS = 5e-6;
NX = 2;
H = 64;
VAL_IDX = [5, 12, 14];

CP_DIR = fullfile(thisDir, 'checkpoints', sprintf('run_%04d', seed));
if ~isfolder(CP_DIR), mkdir(CP_DIR); end

rng(seed);
fprintf('=== NSS LPV Training — Seed %d ===\n', seed);
fprintf('Architecture: dx/dt = A(x,u)*x + B(x,u)*u + c(x,u)\n');
fprintf('Trained with nlssest (no custom loss)\n\n');

%% ── Load data ──────────────────────────────────────────────────────────────
csvDir = fullfile(repoRoot, 'data', 'neural_ode');
csvFiles = dir(fullfile(csvDir, 'profile_*.csv'));
[~, ord] = sort({csvFiles.name}); csvFiles = csvFiles(ord);
nTotal = numel(csvFiles);

profiles_u = cell(nTotal,1); profiles_y = cell(nTotal,1);
for k = 1:nTotal
    raw = readmatrix(fullfile(csvDir, csvFiles(k).name));
    profiles_u{k} = raw(:,1);
    profiles_y{k} = raw(:,2:3);
end

TRAIN_IDX = setdiff(1:nTotal, VAL_IDX);
nTrain = numel(TRAIN_IDX);

%% ── Compute normalization stats from training data ─────────────────────────
all_u = []; all_y = []; all_dv = []; all_di = [];
for i = TRAIN_IDX
    all_u = [all_u; profiles_u{i}]; %#ok<AGROW>
    all_y = [all_y; profiles_y{i}]; %#ok<AGROW>
    all_dv = [all_dv; diff(profiles_y{i}(:,1))/TS]; %#ok<AGROW>
    all_di = [all_di; diff(profiles_y{i}(:,2))/TS]; %#ok<AGROW>
end
ns.u_mean = mean(all_u); ns.u_std = std(all_u);
ns.y_mean = mean(all_y); ns.y_std = std(all_y);
ns.dxdt_std = [std(all_dv), std(all_di)];
fprintf('  dxdt_std = [%.0f, %.0f]\n', ns.dxdt_std);
fprintf('  y_mean = [%.2f, %.2f], y_std = [%.2f, %.2f]\n', ns.y_mean, ns.y_std);

save(fullfile(CP_DIR, 'norm_stats.mat'), '-struct', 'ns');

%% ── Prepare iddata ─────────────────────────────────────────────────────────
% Normalize data for nlssest
trainData = cell(nTrain, 1);
for i = 1:nTrain
    idx = TRAIN_IDX(i);
    u_n = (profiles_u{idx} - ns.u_mean) / ns.u_std;
    y_n = (profiles_y{idx} - ns.y_mean) ./ ns.y_std;
    trainData{i} = iddata(y_n, u_n, TS);
end
trainDataMerged = merge(trainData{:});
fprintf('  Training data: %d profiles, %d samples\n', nTrain, size(all_y,1));

% Validation data
valData = cell(numel(VAL_IDX), 1);
for i = 1:numel(VAL_IDX)
    idx = VAL_IDX(i);
    u_n = (profiles_u{idx} - ns.u_mean) / ns.u_std;
    y_n = (profiles_y{idx} - ns.y_mean) ./ ns.y_std;
    valData{i} = iddata(y_n, u_n, TS);
end
fprintf('  Validation: profiles %s\n', mat2str(VAL_IDX));

%% ── Load Jacobian targets for initialization ───────────────────────────────
jacFile = fullfile(repoRoot, 'data', 'neural_ode', 'jacobian_targets.json');
hasJac = false;
if isfile(jacFile)
    jd = jsondecode(fileread(jacFile));
    A_targets = jd.A_targets; x_ops = jd.x_ops; u_ops = jd.u_ops(:);
    if ndims(A_targets)==3 && size(A_targets,1)==2
        A_targets = permute(A_targets,[3 1 2]);
    end
    hasJac = true;
    fprintf('  Jacobian targets: %d points\n', size(A_targets,1));
end

%% ── Build LPV dlnetwork ────────────────────────────────────────────────────
lgraph = layerGraph();

% Input layers matching NSS convention (separate x and u)
lgraph = addLayers(lgraph, featureInputLayer(NX, 'Name', 'x'));
lgraph = addLayers(lgraph, featureInputLayer(1, 'Name', 'u'));
lgraph = addLayers(lgraph, concatenationLayer(1, 2, 'Name', 'concat'));

% MLP: concat(3) -> fc(64) -> tanh -> fc(64) -> tanh -> fc(8)
lgraph = addLayers(lgraph, fullyConnectedLayer(H, 'Name', 'fc1'));
lgraph = addLayers(lgraph, tanhLayer('Name', 'act1'));
lgraph = addLayers(lgraph, fullyConnectedLayer(H, 'Name', 'fc2'));
lgraph = addLayers(lgraph, tanhLayer('Name', 'act2'));
lgraph = addLayers(lgraph, fullyConnectedLayer(NX*NX + NX + NX, 'Name', 'fc_raw'));  % 8 outputs

% LPV compute: raw(8) + x(2) + u(1) -> dxdt(2)
lgraph = addLayers(lgraph, functionLayer(@(raw, xn, un) ...
    [raw(1,:).*xn(1,:)+raw(2,:).*xn(2,:)+raw(5,:).*un+raw(7,:); ...
     raw(3,:).*xn(1,:)+raw(4,:).*xn(2,:)+raw(6,:).*un+raw(8,:)], ...
    'Name', 'dxdt', 'NumInputs', 3, 'Formattable', true));

% Wire
lgraph = connectLayers(lgraph, 'x', 'concat/in1');
lgraph = connectLayers(lgraph, 'u', 'concat/in2');
lgraph = connectLayers(lgraph, 'concat', 'fc1');
lgraph = connectLayers(lgraph, 'fc1', 'act1');
lgraph = connectLayers(lgraph, 'act1', 'fc2');
lgraph = connectLayers(lgraph, 'fc2', 'act2');
lgraph = connectLayers(lgraph, 'act2', 'fc_raw');
lgraph = connectLayers(lgraph, 'fc_raw', 'dxdt/in1');
lgraph = connectLayers(lgraph, 'x', 'dxdt/in2');
lgraph = connectLayers(lgraph, 'u', 'dxdt/in3');

stateNet = dlnetwork(lgraph);
nParams = sum(cellfun(@numel, {stateNet.Learnables.Value{:}}));
fprintf('  LPV network: %d layers, %d params\n', numel(stateNet.Layers), nParams);

%% ── Physics-informed initialization ────────────────────────────────────────
if hasJac
    nPts = size(A_targets, 1);
    neurPerPt = min(4, H / nPts);

    % Find fc1 weights in Learnables
    fc1_w_idx = find(strcmp(stateNet.Learnables.Layer, 'fc1') & ...
                     contains(stateNet.Learnables.Parameter, 'Weights'));
    fc1_b_idx = find(strcmp(stateNet.Learnables.Layer, 'fc1') & ...
                     contains(stateNet.Learnables.Parameter, 'Bias'));

    W1 = extractdata(stateNet.Learnables.Value{fc1_w_idx});
    b1 = extractdata(stateNet.Learnables.Value{fc1_b_idx});
    W1(:,:) = 0; b1(:) = 0;

    for i = 1:nPts
        % Normalize operating point
        xn = (x_ops(i,:) - ns.y_mean) ./ ns.y_std;
        un = (u_ops(i) - ns.u_mean) / ns.u_std;
        xu = [xn(:); un];

        for j = 1:neurPerPt
            row = (i-1)*neurPerPt + j;
            if row > H, break; end
            W1(row,:) = xu' * (0.5 + 0.5*(j-1));
            b1(row) = -dot(W1(row,:), xu) * (0.3 + 0.2*(j-1));
        end
    end
    filled = nPts * neurPerPt;
    if filled < H
        W1(filled+1:end,:) = 0.05 * randn(H-filled, NX+1);
        b1(filled+1:end) = 0.05 * randn(H-filled, 1);
    end
    stateNet.Learnables.Value{fc1_w_idx} = dlarray(single(W1));
    stateNet.Learnables.Value{fc1_b_idx} = dlarray(single(b1(:)));
    fprintf('  Init: %dx%d=%d neurons from A-matrices\n', nPts, neurPerPt, min(nPts*neurPerPt,H));
end

%% ── Output scaling ─────────────────────────────────────────────────────────
% Scale fc_raw output layer so initial outputs match dxdt magnitude
% The LPV compute A*x+B*u+c produces normalized dxdt. To match the actual
% dxdt_n = dxdt / (y_std * TS^-1) scale, we adjust the last layer.
output_scale = ns.dxdt_std ./ ns.y_std;  % [1x2] but we have 8 outputs
% Scale A-block (rows 1-4) by output_scale components, B-block (5-6), c-block (7-8)

fc_raw_w_idx = find(strcmp(stateNet.Learnables.Layer, 'fc_raw') & ...
                    contains(stateNet.Learnables.Parameter, 'Weights'));
fc_raw_b_idx = find(strcmp(stateNet.Learnables.Layer, 'fc_raw') & ...
                    contains(stateNet.Learnables.Parameter, 'Bias'));

if ~isempty(fc_raw_w_idx)
    W_raw = extractdata(stateNet.Learnables.Value{fc_raw_w_idx});
    b_raw = extractdata(stateNet.Learnables.Value{fc_raw_b_idx});
    % Scale rows: A rows 1,3 by output_scale(1), rows 2,4 by output_scale(2), etc.
    % Actually, the LPV output is [dxdt1; dxdt2] where:
    %   dxdt1 = A11*x1 + A12*x2 + B1*u + c1  (raw indices 1,2,5,7)
    %   dxdt2 = A21*x1 + A22*x2 + B2*u + c2  (raw indices 3,4,6,8)
    % Scale rows contributing to dxdt1 by output_scale(1), dxdt2 by output_scale(2)
    scale_per_row = [output_scale(1); output_scale(1); output_scale(2); output_scale(2); ...
                     output_scale(1); output_scale(2); output_scale(1); output_scale(2)];
    W_raw = W_raw .* scale_per_row;
    b_raw = b_raw(:) .* scale_per_row;
    stateNet.Learnables.Value{fc_raw_w_idx} = dlarray(single(W_raw));
    stateNet.Learnables.Value{fc_raw_b_idx} = dlarray(single(b_raw));
    fprintf('  Output scaling applied: [%.1f, %.1f]\n', output_scale);
end

%% ── Create idNeuralStateSpace ──────────────────────────────────────────────
nss = idNeuralStateSpace(NX, NumInputs=1, NumOutputs=NX, Ts=0);  % CT mode
nss.StateNetwork = stateNet;
fprintf('  idNeuralStateSpace created (CT, Ts=0)\n');

%% ── Training phases ────────────────────────────────────────────────────────
if opts.test
    phases = {struct('name','TEST','epochs',5,'windowMs',5,'lr',1e-3,'dropPeriod',100)};
else
    phases = {
        struct('name','Phase1_5ms','epochs',300,'windowMs',5,'lr',1e-3,'dropPeriod',100)
        struct('name','Phase2_20ms','epochs',min(opts.maxEpochs,500),'windowMs',20,'lr',5e-4,'dropPeriod',150)
        struct('name','Phase3_finetune','epochs',200,'windowMs',20,'lr',1e-4,'dropPeriod',50)
    };
end

tStart = tic;
bestValLoss = inf;
bestNSS = nss;

for phIdx = 1:numel(phases)
    ph = phases{phIdx};
    winSamples = round(ph.windowMs * 1e-3 / TS);

    fprintf('\n--- %s: %d epochs | win=%dms (%d samples) | LR=%.1e ---\n', ...
        ph.name, ph.epochs, ph.windowMs, winSamples, ph.lr);

    trainOpts = nssTrainingOptions('adam');
    trainOpts.MaxEpochs = ph.epochs;
    trainOpts.LearnRate = ph.lr;
    trainOpts.LearnRateSchedule = 'piecewise';
    trainOpts.LearnRateDropPeriod = ph.dropPeriod;
    trainOpts.LearnRateDropFactor = 0.5;
    trainOpts.WindowSize = winSamples;
    trainOpts.NumWindowFraction = eps;  % stochastic per-window updates
    trainOpts.LossFcn = 'MeanSquaredError';

    % Train
    tPhase = tic;
    nss = nlssest(trainDataMerged, nss, trainOpts);
    phTime = toc(tPhase);
    fprintf('  Phase %s done: %.1f min\n', ph.name, phTime/60);

    % Validate
    valLoss = 0;
    for v = 1:numel(valData)
        yp = sim(nss, valData{v});
        if isa(yp, 'iddata'), yp = yp.OutputData; end
        yt = valData{v}.OutputData;
        valLoss = valLoss + mean((yp - yt).^2, 'all');
    end
    valLoss = valLoss / numel(valData);

    % Full-profile RMSE (denormalized)
    fpv = []; fpi = [];
    for v = 1:numel(VAL_IDX)
        idx = VAL_IDX(v);
        u_n = (profiles_u{idx} - ns.u_mean) / ns.u_std;
        y_n = (profiles_y{idx} - ns.y_mean) ./ ns.y_std;
        vd = iddata(y_n, u_n, TS);
        yp = sim(nss, vd);
        if isa(yp, 'iddata'), yp = yp.OutputData; end
        % Denormalize
        yp_phys = yp .* ns.y_std + ns.y_mean;
        y_phys = profiles_y{idx};
        fpv(end+1) = sqrt(mean((yp_phys(:,1) - y_phys(:,1)).^2)); %#ok<AGROW>
        fpi(end+1) = sqrt(mean((yp_phys(:,2) - y_phys(:,2)).^2)); %#ok<AGROW>
    end
    fpVout = mean(fpv); fpIL = mean(fpi);

    elapsed = toc(tStart)/3600;
    improved = '';
    if valLoss < bestValLoss
        bestValLoss = valLoss;
        bestNSS = nss;
        improved = ' ** BEST **';
        save(fullfile(CP_DIR, 'best.mat'), 'nss', 'bestNSS', 'bestValLoss', 'ns', '-v7.3');
        fprintf('    [cp] best.mat\n');
    end
    fprintf('  Val: %.4f | FP: Vout=%.3fV iL=%.3fA | %.2fh%s\n', ...
        valLoss, fpVout, fpIL, elapsed, improved);

    % Save phase checkpoint
    save(fullfile(CP_DIR, sprintf('after_%s.mat', ph.name)), ...
        'nss', 'bestNSS', 'bestValLoss', 'ns', '-v7.3');
end

%% ── Final ──────────────────────────────────────────────────────────────────
totalH = toc(tStart) / 3600;
nss = bestNSS;

% Final full-profile RMSE
fpv = []; fpi = [];
for v = 1:numel(VAL_IDX)
    idx = VAL_IDX(v);
    u_n = (profiles_u{idx} - ns.u_mean) / ns.u_std;
    y_n = (profiles_y{idx} - ns.y_mean) ./ ns.y_std;
    vd = iddata(y_n, u_n, TS);
    yp = sim(nss, vd);
    if isa(yp, 'iddata'), yp = yp.OutputData; end
    yp_phys = yp .* ns.y_std + ns.y_mean;
    y_phys = profiles_y{idx};
    fpv(end+1) = sqrt(mean((yp_phys(:,1) - y_phys(:,1)).^2)); %#ok<AGROW>
    fpi(end+1) = sqrt(mean((yp_phys(:,2) - y_phys(:,2)).^2)); %#ok<AGROW>
end

save(fullfile(CP_DIR, 'final.mat'), 'nss', 'bestNSS', 'bestValLoss', 'ns', '-v7.3');

fprintf('\n=== NSS LPV Seed %d Complete ===\n', seed);
fprintf('  Time: %.2f h\n', totalH);
fprintf('  Best val: %.6f\n', bestValLoss);
fprintf('  FP RMSE: Vout=%.4fV iL=%.4fA\n', mean(fpv), mean(fpi));
fprintf('  DLT LPV ref: Vout=0.019V iL=0.067A\n');
fprintf('  PyTorch LPV ref: Vout=0.019V iL=0.126A\n');

end
