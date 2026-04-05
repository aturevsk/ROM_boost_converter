function train_mlp_rk4(seed, opts)
% train_mlp_rk4  MLP Neural ODE training in MATLAB DLT — replicates PyTorch exactly.
%
% Replicates pytorch_neural_ode_v2/train_neuralode_full.py feature-for-feature:
%   - Same MLP [3->64->tanh->64->tanh->2] with dxdt_scale
%   - Same RK4 at dt=50us (step_skip=10)
%   - Physics-informed init from Branch B A-matrices
%   - CosineAnnealingLR per phase (lr -> 0.01*lr)
%   - Phase 4 extended: ReduceLROnPlateau + progressive windows (20->40->80ms)
%   - Val split: profiles 5,12,14 (mid-range interpolation)
%   - Norm stats from training data only
%   - Global window shuffling per epoch
%   - dlaccelerate for speed
%
% Usage:
%   train_mlp_rk4(3)                     % seed 3 (PyTorch best MLP)
%   train_mlp_rk4(3, 'maxHours', 10)
%   train_mlp_rk4(0, 'test', true)       % smoke test
%
% Target: match PyTorch seed 3 Vout RMSE 0.036V

arguments
    seed (1,1) double = 3
    opts.maxHours (1,1) double = 10
    opts.test (1,1) logical = false
end

thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fileparts(thisDir);
addpath(genpath(repoRoot));

%% ── Constants ──────────────────────────────────────────────────────────────
TS         = 5e-6;
STEP_SKIP  = 10;
DT_RK4     = TS * STEP_SKIP;
NX         = 2;
HIDDEN     = 64;
GRAD_CLIP  = 1.0;
VAL_FREQ   = 10;
EARLY_STOP = 20;
CHECKPOINT_MIN = 15;
VAL_INDICES = [5, 12, 14];  % 1-based profile indices

CHECKPOINT_DIR = fullfile(thisDir, 'checkpoints_mlp', sprintf('run_%04d', seed));
if ~isfolder(CHECKPOINT_DIR), mkdir(CHECKPOINT_DIR); end

maxNumCompThreads(1);  % one thread per process for parallel runs
rng(seed);             % deterministic

fprintf('=== MLP RK4 Training (DLT V2) — Seed %d ===\n', seed);
fprintf('MATLAB %s | maxHours=%.1f\n', version('-release'), opts.maxHours);

%% ── Load data ──────────────────────────────────────────────────────────────
csvDir = fullfile(repoRoot, 'data', 'neural_ode');
csvFiles = dir(fullfile(csvDir, 'profile_*.csv'));
[~, ord] = sort({csvFiles.name}); csvFiles = csvFiles(ord);
nTotal = numel(csvFiles);

profiles_u = cell(nTotal,1); profiles_y = cell(nTotal,1);
for k = 1:nTotal
    raw = single(readmatrix(fullfile(csvDir, csvFiles(k).name)));
    profiles_u{k} = raw(:,1); profiles_y{k} = raw(:,2:3);
end

TRAIN_IDX = setdiff(1:nTotal, VAL_INDICES);
nTrain = numel(TRAIN_IDX);

%% ── Compute norm stats from TRAINING data only ────────────────────────────
all_u = []; all_y = []; all_dvout = []; all_dil = [];
for i = TRAIN_IDX
    all_u = [all_u; profiles_u{i}]; %#ok<AGROW>
    all_y = [all_y; profiles_y{i}]; %#ok<AGROW>
    all_dvout = [all_dvout; diff(profiles_y{i}(:,1))/TS]; %#ok<AGROW>
    all_dil   = [all_dil;   diff(profiles_y{i}(:,2))/TS]; %#ok<AGROW>
end
ns.u_mean = single(mean(all_u)); ns.u_std = single(std(all_u));
ns.x_mean = single(mean(all_y))'; ns.x_std = single(std(all_y))';  % [2x1]
ns.dxdt_scale = single([std(all_dvout); std(all_dil)]);              % [2x1]
fprintf('  dxdt_scale=[%.0f, %.0f]\n', ns.dxdt_scale);

save(fullfile(CHECKPOINT_DIR, 'norm_stats.mat'), '-struct', 'ns');

train_u = profiles_u(TRAIN_IDX); train_y = profiles_y(TRAIN_IDX);
val_u = profiles_u(VAL_INDICES); val_y = profiles_y(VAL_INDICES);
fprintf('  Train: %d profiles, Val: %d (profiles %s)\n', nTrain, numel(VAL_INDICES), mat2str(VAL_INDICES));

%% ── Load Jacobian targets ──────────────────────────────────────────────────
jacFile = fullfile(repoRoot, 'data', 'neural_ode', 'jacobian_targets.json');
hasJac = false; A_targets = []; x_ops = []; u_ops = [];
if isfile(jacFile)
    jd = jsondecode(fileread(jacFile));
    A_targets = single(jd.A_targets); x_ops = single(jd.x_ops); u_ops = single(jd.u_ops(:));
    if ndims(A_targets)==3 && size(A_targets,1)==2, A_targets = permute(A_targets,[3 1 2]); end
    hasJac = true;
    fprintf('  Jacobian targets: %d points\n', size(A_targets,1));
end

%% ── Build MLP dlnetwork ────────────────────────────────────────────────────
layers = [
    featureInputLayer(NX+1, 'Name','input','Normalization','none')
    fullyConnectedLayer(HIDDEN, 'Name','fc1')
    tanhLayer('Name','tanh1')
    fullyConnectedLayer(HIDDEN, 'Name','fc2')
    tanhLayer('Name','tanh2')
    fullyConnectedLayer(NX, 'Name','fc3')
];
net = dlnetwork(layers);
net = initialize(net);

% Physics-informed init from A-matrix at mid-range duty
if hasJac
    nPts = size(A_targets,1);
    midIdx = ceil(nPts/2);
    A_mid = squeeze(A_targets(midIdx,:,:))';  % [2x2]
    B_mid = zeros(NX,1,'single');
    AB = [A_mid, B_mid];  % [2x3]
    sc = max(abs(AB(:))) + 1e-8;
    AB_scaled = AB / sc * 0.5;
    W1 = extractdata(net.Learnables.Value{1});  % [64x3]
    W1(:,:) = 0;
    W1(1:NX, :) = AB_scaled;
    W1(NX+1:end, :) = 0.01 * randn(HIDDEN-NX, NX+1, 'single');
    net.Learnables.Value{1} = dlarray(W1);
    fprintf('  Init from A-matrix at D=%.2f, scale=%.1f\n', u_ops(midIdx), sc);
end
fprintf('  Parameters: %d\n', sum(cellfun(@numel, {net.Learnables.Value{:}})));

%% ── Butterworth filters ────────────────────────────────────────────────────
f_cut = sqrt(600 * 200e3);
fs_eff = 1 / DT_RK4;
f_norm = min(f_cut / (fs_eff/2), 0.95);
if f_norm > 0.05
    [b_lp,a_lp] = butter(2,f_norm,'low');  [b_hp,a_hp] = butter(2,f_norm,'high');
    b_lp=single(b_lp); a_lp=single(a_lp); b_hp=single(b_hp); a_hp=single(a_hp);
else
    b_lp=[]; a_lp=[]; b_hp=[]; a_hp=[];
end

%% ── Phases ─────────────────────────────────────────────────────────────────
if opts.test
    phases = {struct('name','TEST','epochs',5,'window_ms',5,'lr',1e-3,...
        'lambda_J',0,'lambda_ripple',0,'win_per_prof',1)};
    VAL_FREQ = 1; EARLY_STOP = 999;
    fprintf('[TEST MODE]\n');
else
    phases = {
        struct('name','Phase1_HalfOsc','epochs',300,'window_ms',5,'lr',1e-3,...
            'lambda_J',0.1,'lambda_ripple',0,'win_per_prof',3)
        struct('name','Phase2_FullOsc','epochs',500,'window_ms',20,'lr',5e-4,...
            'lambda_J',0.01,'lambda_ripple',0.1,'win_per_prof',3)
        struct('name','Phase3_Finetune','epochs',200,'window_ms',20,'lr',1e-4,...
            'lambda_J',0,'lambda_ripple',0.1,'win_per_prof',3)
    };
end

%% ── Training state ─────────────────────────────────────────────────────────
bestVal = inf; bestNet = net;
trainHist = []; valHist = [];
totalEpochs = 0;
tStart = tic; tLastCp = tic;

accLoss = dlaccelerate(@computeLossAndGrad);

%% ── Phases 1-3 ─────────────────────────────────────────────────────────────
for phIdx = 1:numel(phases)
    ph = phases{phIdx};
    winSamp = round(ph.window_ms * 1e-3 / TS);
    lr_max = ph.lr; lr_min = lr_max * 0.01;
    nEp = ph.epochs;

    clearCache(accLoss);
    fprintf('\n--- %s: %d ep | win=%dms | LR=%.1e->%.1e ---\n', ...
        ph.name, nEp, ph.window_ms, lr_max, lr_min);

    avgG = []; avgSqG = []; phIter = 0; noImprove = 0;

    for epoch = 1:nEp
        totalEpochs = totalEpochs + 1;
        tEp = tic;
        lr = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(pi*(epoch-1)/nEp));

        % Collect & shuffle all windows globally
        allWin = collectWindows(train_u, train_y, nTrain, winSamp, ph.win_per_prof);

        epochLoss = 0; nWin = 0;
        for w = 1:size(allWin,1)
            u_w = allWin{w,1}; y_w = allWin{w,2};
            x0 = dlarray(y_w(1,:)');

            phIter = phIter + 1;
            [loss, grads] = dlfeval(accLoss, net, x0, ...
                dlarray(u_w), dlarray(y_w), ns, DT_RK4, STEP_SKIP, ...
                ph.lambda_ripple, b_lp, a_lp, b_hp, a_hp);

            if isnan(extractdata(loss)) || isinf(extractdata(loss)), continue; end
            grads = clipGrads(grads, GRAD_CLIP);
            [net, avgG, avgSqG] = adamupdate(net, grads, avgG, avgSqG, phIter, lr);
            epochLoss = epochLoss + extractdata(loss); nWin = nWin + 1;

            % Jacobian loss every 5 windows
            if hasJac && ph.lambda_J > 0 && mod(nWin,5)==0
                phIter = phIter + 1;
                [lJ, gJ] = dlfeval(@computeJacLoss, net, A_targets, x_ops, u_ops, ns, ph.lambda_J);
                if ~isnan(extractdata(lJ))
                    gJ = clipGrads(gJ, GRAD_CLIP);
                    [net, avgG, avgSqG] = adamupdate(net, gJ, avgG, avgSqG, phIter, lr);
                end
            end
        end

        avgLoss = epochLoss / max(nWin,1);
        trainHist(end+1) = avgLoss; %#ok<AGROW>
        epTime = toc(tEp);

        if mod(epoch, VAL_FREQ)==0 || epoch==1 || epoch==nEp
            [vl, rv, ri] = computeVal(net, val_u, val_y, ns, DT_RK4, STEP_SKIP);
            [fpv, fpi] = computeFullProfile(net, val_u, val_y, ns, DT_RK4, STEP_SKIP);
            valHist(end+1) = vl; %#ok<AGROW>
            imp = '';
            if vl < bestVal
                bestVal = vl; bestNet = net; noImprove = 0; imp = ' ** BEST **';
                saveCp(CHECKPOINT_DIR, 'best.mat', net, bestNet, bestVal, trainHist, valHist, totalEpochs, ns);
            else
                noImprove = noImprove + 1;
            end
            elapsed = toc(tStart)/3600;
            fprintf('  [%s] Ep %3d/%d | Train: %.4f | Val: %.4f | FP: V=%.3f iL=%.3f | LR=%.1e | %.1fs | %.2fh%s\n', ...
                ph.name, epoch, nEp, avgLoss, vl, fpv, fpi, lr, epTime, elapsed, imp);
            if noImprove >= EARLY_STOP && ~opts.test
                fprintf('  Early stop\n'); break;
            end
        else
            if mod(epoch,50)==0
                fprintf('  [%s] Ep %3d/%d | Train: %.4f | %.1fs\n', ph.name, epoch, nEp, avgLoss, epTime);
            end
        end

        if toc(tLastCp) > CHECKPOINT_MIN*60
            saveCp(CHECKPOINT_DIR, 'latest.mat', net, bestNet, bestVal, trainHist, valHist, totalEpochs, ns);
            tLastCp = tic;
        end
        if toc(tStart)/3600 >= opts.maxHours, fprintf('  Time limit\n'); break; end
    end
    saveCp(CHECKPOINT_DIR, sprintf('after_%s.mat', ph.name), net, bestNet, bestVal, trainHist, valHist, totalEpochs, ns);
    if toc(tStart)/3600 >= opts.maxHours, break; end
end

%% ── Phase 4: Extended ──────────────────────────────────────────────────────
if ~opts.test && toc(tStart)/3600 < opts.maxHours
    fprintf('\n--- Phase4_Extended: ReduceLROnPlateau, progressive windows ---\n');
    if bestVal < inf, net = bestNet; end
    clearCache(accLoss);

    ext_lr = 2e-4; ext_lr_min = 1e-6;
    avgG = []; avgSqG = []; phIter = 0;
    noImpExt = 0; currentLR = ext_lr;
    patience = 10; patienceCount = 0;
    epCount = 0;

    while true
        epCount = epCount + 1; totalEpochs = totalEpochs + 1;
        tEp = tic;

        if epCount <= 200, winMs = 20;
        elseif epCount <= 400, winMs = 40;
        else, winMs = 80; end
        winSamp = round(winMs * 1e-3 / TS);

        allWin = collectWindows(train_u, train_y, nTrain, winSamp, 3);
        epochLoss = 0; nWin = 0;
        for w = 1:size(allWin,1)
            u_w = allWin{w,1}; y_w = allWin{w,2};
            x0 = dlarray(y_w(1,:)');
            phIter = phIter + 1;
            [loss, grads] = dlfeval(accLoss, net, x0, ...
                dlarray(u_w), dlarray(y_w), ns, DT_RK4, STEP_SKIP, ...
                0.1, b_lp, a_lp, b_hp, a_hp);
            if isnan(extractdata(loss)) || isinf(extractdata(loss)), continue; end
            grads = clipGrads(grads, GRAD_CLIP);
            [net, avgG, avgSqG] = adamupdate(net, grads, avgG, avgSqG, phIter, currentLR);
            epochLoss = epochLoss + extractdata(loss); nWin = nWin + 1;
        end
        trainHist(end+1) = epochLoss/max(nWin,1); %#ok<AGROW>
        elapsed = toc(tStart)/3600;

        if elapsed >= opts.maxHours, fprintf('  [Ext] Time limit\n'); break; end
        if currentLR <= ext_lr_min*1.01, fprintf('  [Ext] LR floor\n'); break; end

        if mod(epCount, VAL_FREQ)==0 || epCount==1
            [vl, rv, ri] = computeVal(net, val_u, val_y, ns, DT_RK4, STEP_SKIP);
            [fpv, fpi] = computeFullProfile(net, val_u, val_y, ns, DT_RK4, STEP_SKIP);
            valHist(end+1) = vl; %#ok<AGROW>

            imp = '';
            if vl < bestVal * (1 - 1e-4)
                bestVal = vl; bestNet = net; noImpExt = 0; imp = ' ** BEST **';
                saveCp(CHECKPOINT_DIR, 'best.mat', net, bestNet, bestVal, trainHist, valHist, totalEpochs, ns);
                patienceCount = 0;
            else
                noImpExt = noImpExt + 1;
                patienceCount = patienceCount + 1;
                if patienceCount >= patience
                    currentLR = max(currentLR * 0.5, ext_lr_min);
                    patienceCount = 0;
                    fprintf('    LR reduced to %.1e\n', currentLR);
                end
            end
            fprintf('  [Ext ep%4d] Train: %.4f | Val: %.4f | FP: V=%.3f iL=%.3f | win=%dms | LR=%.1e | %.2fh | noImp=%d/30%s\n', ...
                epCount, trainHist(end), vl, fpv, fpi, winMs, currentLR, elapsed, noImpExt, imp);
            if noImpExt >= 30, fprintf('  [Ext] Plateau\n'); break; end
        end

        if toc(tLastCp) > CHECKPOINT_MIN*60
            saveCp(CHECKPOINT_DIR, 'latest.mat', net, bestNet, bestVal, trainHist, valHist, totalEpochs, ns);
            tLastCp = tic;
        end
    end
end

%% ── Final ──────────────────────────────────────────────────────────────────
totalH = toc(tStart)/3600;
if bestVal < inf, net = bestNet; end
[fpv, fpi] = computeFullProfile(net, val_u, val_y, ns, DT_RK4, STEP_SKIP);
saveCp(CHECKPOINT_DIR, 'final.mat', net, bestNet, bestVal, trainHist, valHist, totalEpochs, ns);

fprintf('\n=== MLP RK4 Seed %d Complete ===\n', seed);
fprintf('  Time: %.2f h | Epochs: %d | Best val: %.6f\n', totalH, totalEpochs, bestVal);
fprintf('  FP RMSE: Vout=%.4fV iL=%.4fA\n', fpv, fpi);
fprintf('  PyTorch ref: Vout=0.036V iL=0.169A\n');
end


%% ========================================================================
%% Collect & shuffle windows globally
%% ========================================================================
function allWin = collectWindows(train_u, train_y, nTrain, winSamp, winPerProf)
allWin = {};
perm = randperm(nTrain);
for p = perm
    u_p = train_u{p}; y_p = train_y{p};
    T = size(u_p,1);
    nW = max(1, floor(T / winSamp));
    nUse = min(winPerProf, nW);
    starts = randperm(nW, nUse) - 1;
    for s = starts
        si = s*winSamp + 1;
        ei = min(si+winSamp-1, T);
        if (ei-si+1) < winSamp, continue; end
        allWin{end+1,1} = u_p(si:si+winSamp-1); %#ok<AGROW>
        allWin{end,2}   = y_p(si:si+winSamp-1, :);
    end
end
% Global shuffle
allWin = allWin(randperm(size(allWin,1)), :);
end


%% ========================================================================
%% Loss + gradient (called via dlfeval + dlaccelerate)
%% ========================================================================
function [loss, grads] = computeLossAndGrad(net, x0, u_win, y_win, ...
        ns, dt, skip, lam_rip, b_lp, a_lp, b_hp, a_hp)
x0 = stripdims(x0); u_win = stripdims(u_win); y_win = stripdims(y_win);

x_pred = rk4Integrate(net, x0, u_win, ns, dt, skip);
T_raw = size(u_win,1); T_sub = size(x_pred,2);
idx = 1:skip:T_raw; idx = idx(1:T_sub);
y_sub = y_win(idx,:)';

x_pred_n = (x_pred - ns.x_mean) ./ ns.x_std;
y_sub_n  = (y_sub  - ns.x_mean) ./ ns.x_std;

if lam_rip > 0 && ~isempty(b_lp) && T_sub > 20
    sk = max(floor(T_sub/10), 5);
    xpT = x_pred_n'; ysT = y_sub_n';
    x_lp = applyIIR(xpT, b_lp, a_lp); y_lp = applyIIR(ysT, b_lp, a_lp);
    x_hp = applyIIR(xpT, b_hp, a_hp); y_hp = applyIIR(ysT, b_hp, a_hp);
    loss = mean((x_lp(sk+1:end,:)-y_lp(sk+1:end,:)).^2,'all') + ...
           lam_rip * mean((x_hp(sk+1:end,:)-y_hp(sk+1:end,:)).^2,'all');
else
    loss = mean((x_pred_n - y_sub_n).^2, 'all');
end
grads = dlgradient(loss, net.Learnables);
end


%% ── RK4 ────────────────────────────────────────────────────────────────────
function x_traj = rk4Integrate(net, x0, u_win, ns, dt, skip)
T_raw = size(u_win,1); T_sub = floor((T_raw-1)/skip)+1;
x = x0; x_traj = x;
for k = 1:(T_sub-1)
    oi = min((k-1)*skip+1, T_raw);
    d_k = u_win(oi);
    k1 = mlpFwd(net,x,d_k,ns); k2 = mlpFwd(net,x+.5*dt*k1,d_k,ns);
    k3 = mlpFwd(net,x+.5*dt*k2,d_k,ns); k4 = mlpFwd(net,x+dt*k3,d_k,ns);
    x = x + (dt/6)*(k1+2*k2+2*k3+k4);
    x_traj = cat(2, x_traj, x);
end
end

function dxdt = mlpFwd(net, x, d, ns)
x_n = (x - ns.x_mean) ./ ns.x_std;
d_n = (d - ns.u_mean) / ns.u_std;
inp = dlarray([stripdims(x_n); d_n], 'CB');
raw = stripdims(forward(net, inp));
dxdt = raw .* ns.dxdt_scale;
end


%% ── IIR filter ─────────────────────────────────────────────────────────────
function y = applyIIR(x, b, a)
[T,C] = size(x); order = numel(a)-1;
y = zeros(T,C,'like',x); d = zeros(order,C,'like',x);
for t = 1:T
    y(t,:) = b(1)*x(t,:) + d(1,:);
    for i = 1:order-1, d(i,:) = b(i+1)*x(t,:) - a(i+1)*y(t,:) + d(i+1,:); end
    d(order,:) = b(order+1)*x(t,:) - a(order+1)*y(t,:);
end
end


%% ── Jacobian loss (analytical) ─────────────────────────────────────────────
function [lJ, gJ] = computeJacLoss(net, A_tgt, x_ops, u_ops, ns, lam_J)
W1=net.Learnables.Value{1}; b1=net.Learnables.Value{2};
W2=net.Learnables.Value{3}; b2=net.Learnables.Value{4};
W3=net.Learnables.Value{5};
nPts = size(A_tgt,1);
if nPts>6
    fi = round(linspace(1,nPts,4));
    rest = setdiff(1:nPts,fi);
    ri = rest(randperm(numel(rest), min(2,numel(rest))));
    idx = [fi, ri];
else, idx = 1:nPts; end

lJ = dlarray(single(0));
for j = idx
    xn = (x_ops(j,:)' - ns.x_mean) ./ ns.x_std;
    dn = (u_ops(j) - ns.u_mean) / ns.u_std;
    inp = [xn; dn];
    z1 = W1*inp+b1(:); a1 = tanh(z1);
    z2 = W2*a1+b2(:);  a2 = tanh(z2);
    D1 = 1-a1.^2; D2 = 1-a2.^2;
    J_phys = ns.dxdt_scale .* (W3*(D2.*(W2*(D1.*W1(:,1:2))))) ./ ns.x_std';
    A_ref = dlarray(single(squeeze(A_tgt(j,:,:))'));
    A_sc = max(abs(extractdata(A_ref)),[],'all')+1;
    lJ = lJ + mean(((J_phys-A_ref)/A_sc).^2,'all');
end
lJ = lJ*(lam_J/numel(idx));
gJ = dlgradient(lJ, net.Learnables);
end


%% ── Validation ─────────────────────────────────────────────────────────────
function [vl, rv, ri] = computeVal(net, val_u, val_y, ns, dt, skip)
vl=0; tv=0; ti=0; tn=0;
for v=1:numel(val_u)
    x_pred = rk4Integrate(net, dlarray(single(val_y{v}(1,:)')), ...
        dlarray(single(val_u{v})), ns, dt, skip);
    T_sub = size(x_pred,2);
    idx = 1:skip:size(val_u{v},1); idx=idx(1:T_sub);
    ys = val_y{v}(idx,:)';
    se = (extractdata(x_pred)-ys).^2;
    tv=tv+sum(se(1,:)); ti=ti+sum(se(2,:)); tn=tn+T_sub;
end
rv=sqrt(tv/tn); ri=sqrt(ti/tn);
vl = (tv/ns.x_std(1)^2 + ti/ns.x_std(2)^2)/(2*tn);
end

function [fpv, fpi] = computeFullProfile(net, val_u, val_y, ns, dt, skip)
rv=[]; ri=[];
for v=1:numel(val_u)
    x_pred = rk4Integrate(net, dlarray(single(val_y{v}(1,:)')), ...
        dlarray(single(val_u{v})), ns, dt, skip);
    T_sub = size(x_pred,2);
    idx = 1:skip:size(val_u{v},1); idx=idx(1:T_sub);
    ys = val_y{v}(idx,:)';
    rv(end+1) = sqrt(mean((extractdata(x_pred(1,:))-ys(1,:)).^2)); %#ok<AGROW>
    ri(end+1) = sqrt(mean((extractdata(x_pred(2,:))-ys(2,:)).^2)); %#ok<AGROW>
end
fpv = mean(rv); fpi = mean(ri);
end


%% ── Gradient clipping ──────────────────────────────────────────────────────
function g = clipGrads(g, maxN)
sq=0; for i=1:height(g), sq=sq+sum(extractdata(g.Value{i}).^2,'all'); end
n=sqrt(sq);
if n>maxN, s=maxN/(n+1e-8); for i=1:height(g), g.Value{i}=g.Value{i}*s; end; end
end


%% ── Checkpoint save ────────────────────────────────────────────────────────
function saveCp(cpDir, fname, net, bestNet, bestVal, trainHist, valHist, totalEpochs, ns)
f = fullfile(cpDir, fname);
save(f, 'net','bestNet','bestVal','trainHist','valHist','totalEpochs','ns', '-v7.3');
fprintf('    [cp] %s (val=%.4f)\n', fname, bestVal);
end
