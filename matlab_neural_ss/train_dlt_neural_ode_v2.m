function train_dlt_neural_ode_v2(testMode)
% train_dlt_neural_ode_v2  Neural ODE V2: dlode45 adjoint gradients + cosine LR
%
% Usage:
%   train_dlt_neural_ode_v2()           % full training
%   train_dlt_neural_ode_v2('test')     % 2-epoch smoke test
%   train_dlt_neural_ode_v2('testjac')  % 1-epoch with Jacobian loss
%
% V2 improvements over V1:
%   - dlode45 with GradientMode='adjoint' (stable gradients for long windows)
%   - Cosine annealing LR schedule (matching PyTorch CosineAnnealingLR)
%
% Same as V1:
%   - MLP architecture: [Vout_n, iL_n, d_n] -> 64 -> tanh -> 64 -> tanh -> 2
%   - 3-phase curriculum, custom loss, Jacobian loss, Adam optimizer
%   - dlaccelerate, maxNumCompThreads, checkpoints, early stopping

if nargin < 1, testMode = ''; end
isTest    = strcmp(testMode, 'test');
isTestJac = strcmp(testMode, 'testjac');

thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fileparts(thisDir);
run(fullfile(repoRoot, 'startup.m'));

fprintf('=== DLT Neural ODE V2 (Adjoint + Cosine LR) ===\n');
fprintf('MATLAB %s | Deep Learning Toolbox | CPU\n', version('-release'));

%% ── Configuration ──────────────────────────────────────────────────────────
TS         = 5e-6;       % base sample period (5 us)
STEP_SKIP  = 10;         % subsample factor (matches PyTorch)
NX         = 2;          % states: [Vout, iL]
NU         = 1;          % input: duty
HIDDEN     = 64;
N_VAL      = 3;

CHECKPOINT_DIR = fullfile(thisDir, 'checkpoints_dlt_v2');
if ~isfolder(CHECKPOINT_DIR), mkdir(CHECKPOINT_DIR); end

maxNumCompThreads(14);
fprintf('CPU threads: %d\n', maxNumCompThreads);

%% ── Load normalisation stats ───────────────────────────────────────────────
normFile = fullfile(repoRoot, 'model_data', 'neuralode_norm_stats.json');
stats    = jsondecode(fileread(normFile));
normStats.u_mean     = single(stats.u_mean);
normStats.u_std      = single(stats.u_std);
normStats.x_mean     = single(stats.x_mean(:));   % [2x1] column
normStats.x_std      = single(stats.x_std(:));
normStats.dxdt_scale = single(stats.dxdt_std(:));
fprintf('Normalisation: dxdt_scale=[%.1f, %.1f]\n', normStats.dxdt_scale);

%% ── Load training data ─────────────────────────────────────────────────────
csvDir = fullfile(repoRoot, 'data', 'neural_ode');
csvFiles = dir(fullfile(csvDir, 'profile_*.csv'));
[~, ord] = sort({csvFiles.name}); csvFiles = csvFiles(ord);

fprintf('\nLoading %d profiles...\n', numel(csvFiles));
profiles_u = cell(numel(csvFiles), 1);
profiles_y = cell(numel(csvFiles), 1);
for k = 1:numel(csvFiles)
    raw = single(readmatrix(fullfile(csvDir, csvFiles(k).name)));
    profiles_u{k} = raw(:, 1);
    profiles_y{k} = raw(:, 2:3);
end
nTotal = numel(profiles_u);
nTrain = nTotal - N_VAL;
fprintf('  %d profiles: %d train, %d val\n', nTotal, nTrain, N_VAL);

train_u = profiles_u(1:nTrain);
train_y = profiles_y(1:nTrain);
val_u   = profiles_u(nTrain+1:end);
val_y   = profiles_y(nTrain+1:end);

%% ── Load Jacobian targets ──────────────────────────────────────────────────
jacFile = fullfile(repoRoot, 'data', 'neural_ode', 'jacobian_targets.json');
hasJacobian = false;
if isfile(jacFile)
    jdata = jsondecode(fileread(jacFile));
    A_targets = single(jdata.A_targets);
    x_ops     = single(jdata.x_ops);
    u_ops     = single(jdata.u_ops(:));
    if ndims(A_targets) == 3 && size(A_targets,1) == 2
        A_targets = permute(A_targets, [3 1 2]);
    end
    hasJacobian = true;
    fprintf('  Jacobian targets: %d points\n', size(A_targets,1));
end

%% ── Build MLP dlnetwork ────────────────────────────────────────────────────
layers = [
    featureInputLayer(NX + NU, 'Name', 'input', 'Normalization', 'none')
    fullyConnectedLayer(HIDDEN, 'Name', 'fc1')
    tanhLayer('Name', 'tanh1')
    fullyConnectedLayer(HIDDEN, 'Name', 'fc2')
    tanhLayer('Name', 'tanh2')
    fullyConnectedLayer(NX, 'Name', 'fc3')
];
net = dlnetwork(layers);
net = initialize(net);
nParams = sum(cellfun(@numel, {net.Learnables.Value{:}}));
fprintf('MLP parameters: %d\n', nParams);

%% ── Butterworth filters for ripple loss ────────────────────────────────────
f_cut  = sqrt(600 * 200e3);
fs_eff = 1 / (TS * STEP_SKIP);
f_norm = min(f_cut / (fs_eff/2), 0.95);
if f_norm > 0.05
    [b_lp, a_lp] = butter(2, f_norm, 'low');
    [b_hp, a_hp] = butter(2, f_norm, 'high');
    b_lp = single(b_lp); a_lp = single(a_lp);
    b_hp = single(b_hp); a_hp = single(a_hp);
else
    b_lp = []; a_lp = []; b_hp = []; a_hp = [];
end

%% ── Training phases ────────────────────────────────────────────────────────
if isTest
    phases = {struct('name','TEST','epochs',2,'window_ms',5,'lr',1e-3,...
        'lambda_J',0.0,'lambda_ripple',0.0,'win_per_prof',1)};
    VAL_FREQ = 1; EARLY_STOP = 999;
    fprintf('[TEST MODE]\n');
elseif isTestJac
    phases = {struct('name','TESTJAC','epochs',1,'window_ms',5,'lr',1e-3,...
        'lambda_J',0.1,'lambda_ripple',0.0,'win_per_prof',1)};
    VAL_FREQ = 1; EARLY_STOP = 999;
    fprintf('[TESTJAC MODE]\n');
else
    phases = {
      struct('name','Phase1_HalfOsc','epochs',300,'window_ms',5,'lr',1e-3,...
          'lambda_J',0.1,'lambda_ripple',0.0,'win_per_prof',3)
      struct('name','Phase2_FullOsc','epochs',500,'window_ms',20,'lr',5e-4,...
          'lambda_J',0.01,'lambda_ripple',0.1,'win_per_prof',3)
      struct('name','Phase3_Finetune','epochs',200,'window_ms',20,'lr',1e-4,...
          'lambda_J',0.0,'lambda_ripple',0.1,'win_per_prof',3)
    };
    VAL_FREQ      = 10;
    EARLY_STOP    = 20;
end

CHECKPOINT_MIN = 15;
GRAD_CLIP      = 1.0;

%% ── Resume from checkpoint ─────────────────────────────────────────────────
bestValLoss   = inf;
bestNet       = net;
trainHistory  = [];
valHistory    = [];
totalEpochs   = 0;
startPhaseIdx = 1;

latestCp = fullfile(CHECKPOINT_DIR, 'dlt_v2_latest.mat');
if isfile(latestCp)
    fprintf('\n=== Resuming from checkpoint ===\n');
    cp = load(latestCp);
    net          = cp.net;
    bestNet      = cp.bestNet;
    bestValLoss  = cp.bestValLoss;
    trainHistory = cp.trainHistory;
    valHistory   = cp.valHistory;
    totalEpochs  = cp.totalEpochs;
    startPhaseIdx = cp.phaseIdx;
    normStats    = cp.normStats;
    fprintf('  bestVal=%.4f, epochs=%d, phase %d\n', bestValLoss, totalEpochs, startPhaseIdx);
end

%% ── dlaccelerate ────────────────────────────────────────────────────────────
accLossFcn = dlaccelerate(@computeLossAndGrad);

%% ── Main training loop ──────────────────────────────────────────────────────
tStart     = tic;
tLastCp    = tic;
iterGlobal = 0;
epochTimesAll = [];  % for benchmarking

for phIdx = startPhaseIdx:numel(phases)
    ph = phases{phIdx};
    phName     = ph.name;
    nEpochs    = ph.epochs;
    winSamples = round(ph.window_ms * 1e-3 / TS);
    winSub     = floor((winSamples - 1) / STEP_SKIP) + 1;
    lr_max     = ph.lr;
    lr_min     = lr_max * 0.01;  % cosine annealing floor (same as PyTorch)
    lam_J      = ph.lambda_J;
    lam_rip    = ph.lambda_ripple;
    winPerProf = ph.win_per_prof;
    winDuration = ph.window_ms * 1e-3;  % seconds

    clearCache(accLossFcn);

    fprintf('\n--- %s: %d ep | win=%.0fms (%d sub) | LR=%.1e->%.1e | lJ=%.3f | lR=%.2f ---\n', ...
        phName, nEpochs, ph.window_ms, winSub, lr_max, lr_min, lam_J, lam_rip);

    % Per-phase Adam state
    avgGrad   = [];
    avgSqGrad = [];
    phIter    = 0;
    noImprove = 0;

    for epoch = 1:nEpochs
        totalEpochs = totalEpochs + 1;
        tEpoch = tic;

        % Cosine annealing LR (matching PyTorch CosineAnnealingLR)
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * (epoch-1) / nEpochs));

        epochLoss = 0;
        nWindows  = 0;
        perm      = randperm(nTrain);

        for pIdx = perm
            u_p = train_u{pIdx};
            y_p = train_y{pIdx};
            T   = size(u_p, 1);

            nWinTotal = max(1, floor(T / winSamples));
            nUse      = min(winPerProf, nWinTotal);
            winIdxs   = randperm(nWinTotal, nUse) - 1;

            for ww = 1:nUse
                s = winIdxs(ww) * winSamples + 1;
                e = min(s + winSamples - 1, T);
                if (e - s + 1) < 20, continue; end

                % Build duty interpolation data for this window
                u_win_raw = u_p(s:e);
                y_win_raw = y_p(s:e, :);
                x0 = dlarray(y_p(s, :)');   % [2x1] unformatted

                % Time vector for duty interpolation (0-based, in seconds)
                dt_duty = TS;  % duty sampled at TS
                t_duty  = (0:numel(u_win_raw)-1)' * dt_duty;  % [len x 1]

                % tspan for dlode45 output (subsampled points)
                tspan = linspace(0, winDuration, winSub);

                % Subsample ground truth
                idx_sub = 1:STEP_SKIP:(e-s+1);
                idx_sub = idx_sub(1:min(numel(idx_sub), winSub));
                y_sub   = dlarray(y_win_raw(idx_sub, :));  % [T_sub x 2]

                phIter     = phIter + 1;
                iterGlobal = iterGlobal + 1;

                [loss, grads] = dlfeval(accLossFcn, ...
                    net, x0, tspan, t_duty, u_win_raw, y_sub, ...
                    normStats, lam_rip, b_lp, a_lp, b_hp, a_hp);

                if isnan(extractdata(loss)) || isinf(extractdata(loss))
                    continue;
                end

                grads = clipGradients(grads, GRAD_CLIP);
                [net, avgGrad, avgSqGrad] = adamupdate(net, grads, ...
                    avgGrad, avgSqGrad, phIter, lr);

                epochLoss = epochLoss + extractdata(loss);
                nWindows  = nWindows + 1;
            end

            % Jacobian loss
            if hasJacobian && lam_J > 0 && mod(nWindows, 5) == 0 && nWindows > 0
                phIter = phIter + 1;
                [lossJ, gradsJ] = dlfeval(@computeJacobianLoss, ...
                    net, A_targets, x_ops, u_ops, normStats, lam_J);
                if ~isnan(extractdata(lossJ)) && ~isinf(extractdata(lossJ))
                    gradsJ = clipGradients(gradsJ, GRAD_CLIP);
                    [net, avgGrad, avgSqGrad] = adamupdate(net, gradsJ, ...
                        avgGrad, avgSqGrad, phIter, lr);
                end
            end
        end

        avgTrain  = epochLoss / max(nWindows, 1);
        epochTime = toc(tEpoch);
        trainHistory(end+1) = avgTrain; %#ok<AGROW>
        epochTimesAll(end+1) = epochTime; %#ok<AGROW>

        % Validation
        if mod(epoch, VAL_FREQ) == 0 || epoch == 1 || epoch == nEpochs
            valLoss = computeValidationLoss(net, val_u, val_y, normStats, ...
                TS, STEP_SKIP, winSamples);
            valHistory(end+1) = valLoss; %#ok<AGROW>

            improved = '';
            if valLoss < bestValLoss
                bestValLoss = valLoss;
                bestNet     = net;
                noImprove   = 0;
                improved    = ' ***BEST***';
                saveCheckpoint(CHECKPOINT_DIR, 'dlt_v2_best.mat', ...
                    net, bestNet, bestValLoss, trainHistory, valHistory, ...
                    totalEpochs, phIdx, normStats, epochTimesAll);
            else
                noImprove = noImprove + 1;
            end

            elapsed = toc(tStart) / 60;
            fprintf('  [%s] Ep %3d/%d | Train: %.4f | Val: %.4f | LR: %.1e | %.1fs/ep | %.1fmin%s\n', ...
                phName, epoch, nEpochs, avgTrain, valLoss, lr, epochTime, elapsed, improved);
        else
            fprintf('  [%s] Ep %3d/%d | Train: %.4f | LR: %.1e | %.1fs/ep\n', ...
                phName, epoch, nEpochs, avgTrain, lr, epochTime);
        end

        % Periodic checkpoint
        if toc(tLastCp) > CHECKPOINT_MIN * 60
            saveCheckpoint(CHECKPOINT_DIR, 'dlt_v2_latest.mat', ...
                net, bestNet, bestValLoss, trainHistory, valHistory, ...
                totalEpochs, phIdx, normStats, epochTimesAll);
            tLastCp = tic;
        end

        if noImprove >= EARLY_STOP
            fprintf('  Early stopping: %d epochs without improvement\n', noImprove);
            break;
        end
    end

    saveCheckpoint(CHECKPOINT_DIR, sprintf('dlt_v2_after_%s.mat', phName), ...
        net, bestNet, bestValLoss, trainHistory, valHistory, ...
        totalEpochs, phIdx + 1, normStats, epochTimesAll);
    fprintf('  Phase %s complete. Best val: %.4f\n', phName, bestValLoss);
end

%% ── Final summary ───────────────────────────────────────────────────────────
totalMin = toc(tStart) / 60;
fprintf('\n=== V2 Training complete ===\n');
fprintf('  Total time: %.1f min (%.1f h)\n', totalMin, totalMin/60);
fprintf('  Total epochs: %d\n', totalEpochs);
fprintf('  Best val loss: %.4f\n', bestValLoss);
fprintf('  Avg epoch time: %.1f s\n', mean(epochTimesAll));
fprintf('  PyTorch reference: 0.040 val, ~22s/epoch\n');
fprintf('  Ratio vs PyTorch: %.2fx val, %.2fx speed\n', ...
    bestValLoss / 0.040, mean(epochTimesAll) / 22);

saveCheckpoint(CHECKPOINT_DIR, 'dlt_v2_final.mat', ...
    bestNet, bestNet, bestValLoss, trainHistory, valHistory, ...
    totalEpochs, numel(phases)+1, normStats, epochTimesAll);
fprintf('Final model saved.\n');

end  % function train_dlt_neural_ode_v2


%% ========================================================================
%% Loss function — uses dlode45 with adjoint gradients
%% ========================================================================
function [loss, grads] = computeLossAndGrad(net, x0, tspan, t_duty, u_duty, y_sub, ...
        normStats, lambda_ripple, b_lp, a_lp, b_hp, a_hp)
% net:     dlnetwork
% x0:      [2x1] dlarray initial state (physical)
% tspan:   [1 x T_sub] time vector for output points (double)
% t_duty:  [T x 1] time vector for duty interpolation (double)
% u_duty:  [T x 1] duty cycle values (single)
% y_sub:   [T_sub x 2] dlarray ground truth at tspan points
% normStats, lambda_ripple, filter coefficients: same as V1

y_sub = stripdims(y_sub);

% Pack theta: net parameters + duty interpolation data + normStats
theta.net       = net;
theta.normStats = normStats;
theta.t_duty    = t_duty;
theta.u_duty    = u_duty;

% x0 must have 'CB' format — neuralODE_rhs must return 'CB' too
x0 = dlarray(stripdims(x0), 'CB');

% Integrate with dlode45 + adjoint gradients
Y = dlode45(@neuralODE_rhs, tspan, x0, theta, ...
    'GradientMode', 'adjoint');
% Y: [2 x T_sub] — state trajectory at tspan points

x_pred = stripdims(Y);   % [2 x T_sub] unformatted
T_sub  = size(x_pred, 2);

% Ground truth: [2 x T_sub]
y_gt = y_sub(1:T_sub, :)';  % [2 x T_sub]

% Normalise
x_pred_n = (x_pred - normStats.x_mean) ./ normStats.x_std;
y_gt_n   = (y_gt   - normStats.x_mean) ./ normStats.x_std;

% Loss
if lambda_ripple > 0 && ~isempty(b_lp) && T_sub > 20
    skip = max(floor(T_sub / 10), 5);
    xpT  = x_pred_n';  ysT = y_gt_n';  % [T_sub x 2]
    x_lp = applyIIR(xpT, b_lp, a_lp);
    y_lp = applyIIR(ysT, b_lp, a_lp);
    x_hp = applyIIR(xpT, b_hp, a_hp);
    y_hp = applyIIR(ysT, b_hp, a_hp);
    loss_avg    = mean((x_lp(skip+1:end,:) - y_lp(skip+1:end,:)).^2, 'all');
    loss_ripple = mean((x_hp(skip+1:end,:) - y_hp(skip+1:end,:)).^2, 'all');
    loss = loss_avg + lambda_ripple * loss_ripple;
else
    loss = mean((x_pred_n - y_gt_n).^2, 'all');
end

grads = dlgradient(loss, net.Learnables);
end


%% ── Neural ODE right-hand side for dlode45 ──────────────────────────────────
function dxdt = neuralODE_rhs(t, x, theta)
% t:     scalar time (within window, 0-based)
% x:     [2 x 1] state in physical units (dlarray, format 'CB')
% theta: struct with .net, .normStats, .t_duty, .u_duty
% Returns dxdt: [2 x 1] in physical units

ns = theta.normStats;

% Interpolate duty cycle at time t (zero-order hold)
% t_duty and u_duty are regular arrays (not dlarray) — no gradient needed for duty
t_val = extractdata(gather(t));  % scalar double
idx = find(theta.t_duty <= t_val, 1, 'last');
if isempty(idx), idx = 1; end
d = single(theta.u_duty(idx));

% Normalize state and duty (keep 'CB' format throughout)
x_n = (x - ns.x_mean) ./ ns.x_std;   % [2 x 1] 'CB'
d_n = (d - ns.u_mean) / ns.u_std;          % scalar

% MLP forward — output must have 'CB' format (matching x0)
inp  = dlarray([stripdims(x_n); d_n], 'CB');  % [3 x 1]
raw  = forward(theta.net, inp);            % [2 x 1] 'CB' format
dxdt = raw .* ns.dxdt_scale;              % [2 x 1] 'CB' format
end


%% ── IIR filter (Direct Form II Transposed) ──────────────────────────────────
function y = applyIIR(x, b, a)
[T, C] = size(x);
order  = numel(a) - 1;
y      = zeros(T, C, 'like', x);
d      = zeros(order, C, 'like', x);
for t_i = 1:T
    y(t_i,:) = b(1) * x(t_i,:) + d(1,:);
    for i = 1:order-1
        d(i,:) = b(i+1) * x(t_i,:) - a(i+1) * y(t_i,:) + d(i+1,:);
    end
    d(order,:) = b(order+1) * x(t_i,:) - a(order+1) * y(t_i,:);
end
end


%% ── Jacobian loss (analytical, same as V1) ──────────────────────────────────
function [lossJ, gradsJ] = computeJacobianLoss(net, A_targets, x_ops, u_ops, normStats, lambda_J)
W1 = net.Learnables.Value{1};
b1 = net.Learnables.Value{2};
W2 = net.Learnables.Value{3};
b2 = net.Learnables.Value{4};
W3 = net.Learnables.Value{5};

nPts = size(A_targets, 1);
if nPts > 6
    fixed = round(linspace(1, nPts, 4));
    rest  = setdiff(1:nPts, fixed);
    rnd   = rest(randperm(numel(rest), min(2, numel(rest))));
    idx   = [fixed, rnd];
else
    idx = 1:nPts;
end

lossJ = dlarray(single(0));
for j = idx
    x_n = (x_ops(j,:)' - normStats.x_mean) ./ normStats.x_std;
    d_n = (u_ops(j) - normStats.u_mean) / normStats.u_std;
    inp = [x_n; d_n];
    z1  = W1 * inp + b1(:);   a1 = tanh(z1);
    z2  = W2 * a1  + b2(:);   a2 = tanh(z2);
    D1  = 1 - a1.^2;  D2 = 1 - a2.^2;
    W1_x   = W1(:, 1:2);
    J_phys = normStats.dxdt_scale .* (W3 * (D2 .* (W2 * (D1 .* W1_x)))) ./ normStats.x_std';
    A_ref  = dlarray(single(squeeze(A_targets(j,:,:))'));
    A_sc   = max(abs(extractdata(A_ref)), [], 'all') + 1;
    lossJ  = lossJ + mean(((J_phys - A_ref) / A_sc).^2, 'all');
end
lossJ  = lossJ * (lambda_J / numel(idx));
gradsJ = dlgradient(lossJ, net.Learnables);
end


%% ── Validation ───────────────────────────────────────────────────────────────
function valLoss = computeValidationLoss(net, val_u, val_y, normStats, TS, step_skip, winSamples)
valLoss = 0;
nVal    = numel(val_u);
winDur  = winSamples * TS;
winSub  = floor((winSamples-1)/step_skip) + 1;
tspan   = linspace(0, winDur, winSub);

theta.net       = net;
theta.normStats = normStats;

for v = 1:nVal
    u_v = val_u{v};
    y_v = val_y{v};
    T   = size(u_v, 1);
    nWin = min(3, floor(T / winSamples));
    profLoss = 0; nW = 0;

    for w = 0:nWin-1
        s = w * winSamples + 1;
        e = min(s + winSamples - 1, T);
        if (e-s+1) < 20, continue; end

        x0 = dlarray(single(y_v(s,:)'), 'CB');
        theta.t_duty = single((0:(e-s))' * TS);
        theta.u_duty = single(u_v(s:e));

        Y = dlode45(@neuralODE_rhs, tspan, x0, theta);  % no grads needed for val

        x_pred = stripdims(Y);  % [2 x T_sub]
        T_sub  = size(x_pred, 2);
        idx    = 1:step_skip:(e-s+1);
        idx    = idx(1:T_sub);
        y_sub  = y_v(s-1+idx, :)';  % [2 x T_sub]

        x_pred_n = (x_pred - normStats.x_mean) ./ normStats.x_std;
        y_sub_n  = (y_sub  - normStats.x_mean) ./ normStats.x_std;
        profLoss = profLoss + extractdata(mean((x_pred_n - y_sub_n).^2, 'all'));
        nW = nW + 1;
    end
    if nW > 0, valLoss = valLoss + profLoss / nW; end
end
valLoss = valLoss / nVal;
end


%% ── Gradient clipping ────────────────────────────────────────────────────────
function grads = clipGradients(grads, maxNorm)
totalSq = 0;
for i = 1:height(grads)
    g = grads.Value{i};
    totalSq = totalSq + sum(extractdata(g).^2, 'all');
end
norm_g = sqrt(totalSq);
if norm_g > maxNorm
    scale = maxNorm / (norm_g + 1e-8);
    for i = 1:height(grads)
        grads.Value{i} = grads.Value{i} * scale;
    end
end
end


%% ── Checkpoint save ──────────────────────────────────────────────────────────
function saveCheckpoint(cpDir, fname, net, bestNet, bestValLoss, ...
        trainHistory, valHistory, totalEpochs, phaseIdx, normStats, epochTimes)
cpFile = fullfile(cpDir, fname);
save(cpFile, 'net', 'bestNet', 'bestValLoss', 'trainHistory', ...
    'valHistory', 'totalEpochs', 'phaseIdx', 'normStats', 'epochTimes', '-v7.3');
fprintf('    [checkpoint] %s (val=%.4f)\n', fname, bestValLoss);
end
