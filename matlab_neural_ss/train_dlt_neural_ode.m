function train_dlt_neural_ode(testMode)
% train_dlt_neural_ode  Custom Neural ODE training loop in MATLAB Deep Learning Toolbox.
%
% Usage:
%   train_dlt_neural_ode()           % full training
%   train_dlt_neural_ode('test')     % 2-epoch smoke test
%
% Replicates PyTorch train_neural_ode_pytorch.py:
%
% Replicates PyTorch train_neural_ode_pytorch.py as closely as possible:
%   - Same MLP architecture: [Vout_n, iL_n, d_n] -> 64 -> tanh -> 64 -> tanh -> 2
%   - Same RK4 integration at dt=50us (step_skip=10)
%   - Same 3-phase curriculum (5ms / 20ms windows)
%   - Same custom loss: L_traj + λ_J*L_jacobian; L_avg + λ_ripple*L_ripple
%   - Same train/val split (last 3 profiles = val)
%   - dlaccelerate for JIT compilation
%   - maxNumCompThreads for M4 Max CPU
%
% Benchmark goal: compare epoch time and final val loss vs PyTorch.
%
% Key differences from PyTorch:
%   - No GPU (Apple Silicon not supported by MATLAB gpuArray)
%   - Filter implemented as explicit IIR loop (same algorithm as PyTorch)
%   - Jacobian computed via dljacobian (exact autodiff, same as PyTorch)

if nargin < 1, testMode = ''; end
isTest = strcmp(testMode, 'test');

thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fileparts(thisDir);
run(fullfile(repoRoot, 'startup.m'));

fprintf('=== DLT Neural ODE (Custom Training Loop) ===\n');
fprintf('MATLAB %s | Deep Learning Toolbox | CPU\n', version('-release'));

%% ── Configuration ──────────────────────────────────────────────────────────
TS         = 5e-6;       % base sample period (5 us)
STEP_SKIP  = 10;         % subsample factor (matches PyTorch)
DT_RK4     = TS * STEP_SKIP;  % RK4 step = 50 us
NX         = 2;          % states: [Vout, iL]
NU         = 1;          % input: duty
HIDDEN     = 64;         % units per hidden layer
N_VAL      = 3;          % last 3 profiles = validation

CHECKPOINT_DIR = fullfile(thisDir, 'checkpoints_dlt');
if ~isfolder(CHECKPOINT_DIR), mkdir(CHECKPOINT_DIR); end

% Maximise CPU threads on M4 Max
maxNumCompThreads(14);
fprintf('CPU threads: %d\n', maxNumCompThreads);

%% ── Load normalisation stats (computed by PyTorch, reuse for fair comparison) ──
normFile = fullfile(repoRoot, 'model_data', 'neuralode_norm_stats.json');
stats    = jsondecode(fileread(normFile));
% stats fields: u_mean, u_std, x_mean[2], x_std[2], dxdt_std[2]
normStats.u_mean    = single(stats.u_mean);
normStats.u_std     = single(stats.u_std);
normStats.x_mean    = single(stats.x_mean(:)');   % 1x2
normStats.x_std     = single(stats.x_std(:)');    % 1x2
normStats.dxdt_scale = single(stats.dxdt_std(:)'); % 1x2  (output scaling)
fprintf('Normalisation loaded: u=[%.3f±%.3f] Vout=[%.2f±%.2f] iL=[%.2f±%.2f]\n', ...
    normStats.u_mean, normStats.u_std, ...
    normStats.x_mean(1), normStats.x_std(1), ...
    normStats.x_mean(2), normStats.x_std(2));
fprintf('dxdt_scale: [%.1f, %.1f]\n', normStats.dxdt_scale(1), normStats.dxdt_scale(2));

%% ── Load training data ─────────────────────────────────────────────────────
csvDir = fullfile(repoRoot, 'data', 'neural_ode');
csvFiles = dir(fullfile(csvDir, 'profile_*.csv'));
csvFiles = csvFiles(~[csvFiles.isdir]);
[~, ord] = sort({csvFiles.name});
csvFiles = csvFiles(ord);
assert(~isempty(csvFiles), 'No CSV files found in %s', csvDir);

fprintf('\nLoading %d profiles...\n', numel(csvFiles));
profiles_u = cell(numel(csvFiles), 1);
profiles_y = cell(numel(csvFiles), 1);
for k = 1:numel(csvFiles)
    raw = single(readmatrix(fullfile(csvDir, csvFiles(k).name)));
    profiles_u{k} = raw(:, 1);          % duty [T x 1]
    profiles_y{k} = raw(:, 2:3);        % [Vout, iL] [T x 2]
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
    % A_targets: [nPts x 2 x 2], x_ops: [nPts x 2], u_ops: [nPts x 1]
    A_targets = single(jdata.A_targets);   % may be [2 2 nPts] depending on JSON
    x_ops     = single(jdata.x_ops);
    u_ops     = single(jdata.u_ops(:));
    if ndims(A_targets) == 3 && size(A_targets,1) == 2
        % reshape from [2 2 nPts] to [nPts 2 2]
        A_targets = permute(A_targets, [3 1 2]);
    end
    nJac = size(A_targets, 1);
    hasJacobian = true;
    fprintf('  Jacobian targets: %d operating points\n', nJac);
else
    fprintf('  WARNING: No Jacobian targets found at %s\n', jacFile);
end

%% ── Build MLP dlnetwork ────────────────────────────────────────────────────
% Architecture: featureInput(3) -> fc(64) -> tanh -> fc(64) -> tanh -> fc(2)
% dxdt_scale is NOT a trainable layer — applied after forward pass.
layers = [
    featureInputLayer(NX + NU, 'Name', 'input', 'Normalization', 'none')
    fullyConnectedLayer(HIDDEN, 'Name', 'fc1')
    tanhLayer('Name', 'tanh1')
    fullyConnectedLayer(HIDDEN, 'Name', 'fc2')
    tanhLayer('Name', 'tanh2')
    fullyConnectedLayer(NX, 'Name', 'fc3')
];
net = dlnetwork(layers);

% Initialise fc3 weights scaled by dxdt_scale (matches PyTorch dxdt_scale buffer init)
% PyTorch last layer: output ~[-1,1] * dxdt_scale => physical derivatives
% We scale the initial weights so that random outputs map to ~physical scale.
% Use Xavier init (default) scaled so that E[|output|] ~ 1/dxdt_scale initially.
% (This just ensures gradients aren't overwhelmed by the scale factor at epoch 0.)
net = initialize(net);
fprintf('MLP parameters: %d\n', sum(cellfun(@numel, {net.Learnables.Value{:}})));

%% ── Butterworth filter design for ripple loss ──────────────────────────────
% Cutoff at sqrt(f_LC * f_sw) = sqrt(600 * 200e3) ≈ 11 kHz  (same as PyTorch)
f_sw    = 200e3;
f_LC    = 600;
f_cut   = sqrt(f_LC * f_sw);  % ~11 kHz

% Filters designed at the subsampled rate (1/DT_RK4 = 20 kHz)
fs_eff  = 1 / DT_RK4;
f_nyq   = fs_eff / 2;
f_norm  = min(f_cut / f_nyq, 0.95);   % normalised cutoff [0,1]

if f_norm > 0.05
    [b_lp, a_lp] = butter(2, f_norm, 'low');
    [b_hp, a_hp] = butter(2, f_norm, 'high');
    b_lp = single(b_lp); a_lp = single(a_lp);
    b_hp = single(b_hp); a_hp = single(a_hp);
    fprintf('Butterworth filters: cutoff=%.0f Hz, fs_eff=%.0f Hz\n', f_cut, fs_eff);
else
    b_lp = []; a_lp = [];
    b_hp = []; a_hp = [];
    fprintf('Filter cutoff too low for effective sample rate, ripple loss disabled.\n');
end

%% ── Training phases (matching PyTorch exactly) ─────────────────────────────
isTestJac = strcmp(testMode, 'testjac');
if isTest
    phases = {
      struct('name','TEST', 'epochs',2, 'window_ms',5, 'lr',1e-3, 'lambda_J',0.0, 'lambda_ripple',0.0, 'win_per_prof',1)
    };
    VAL_FREQ = 1; EARLY_STOP = 999;
    fprintf('[TEST MODE: 2 epochs only]\n');
elseif isTestJac
    phases = {
      struct('name','TESTJAC', 'epochs',1, 'window_ms',5, 'lr',1e-3, 'lambda_J',0.1, 'lambda_ripple',0.0, 'win_per_prof',1)
    };
    VAL_FREQ = 1; EARLY_STOP = 999;
    fprintf('[TESTJAC MODE: 1 epoch with Jacobian loss]\n');
else
    phases = {
      struct('name','Phase1_HalfOsc', 'epochs',300, 'window_ms',5,  'lr',1e-3,   'lambda_J',0.1,  'lambda_ripple',0.0, 'win_per_prof',3)
      struct('name','Phase2_FullOsc',  'epochs',500, 'window_ms',20, 'lr',5e-4,   'lambda_J',0.01, 'lambda_ripple',0.1, 'win_per_prof',3)
      struct('name','Phase3_Finetune', 'epochs',200, 'window_ms',20, 'lr',1e-4,   'lambda_J',0.0,  'lambda_ripple',0.1, 'win_per_prof',3)
    };
end

VAL_FREQ      = 10;   % validate every N epochs (same as PyTorch)
EARLY_STOP    = 20;   % epochs without improvement before early stop (same as PyTorch)
CHECKPOINT_MIN = 15;  % periodic checkpoint interval (minutes)
GRAD_CLIP     = 1.0;  % gradient clip norm (same as PyTorch)

%% ── State: resume from checkpoint if it exists ─────────────────────────────
bestValLoss   = inf;
bestNet       = net;
trainHistory  = [];
valHistory    = [];
totalEpochs   = 0;
startPhaseIdx = 1;

latestCp = fullfile(CHECKPOINT_DIR, 'dlt_latest.mat');
if isfile(latestCp)
    fprintf('\n=== Resuming from %s ===\n', latestCp);
    cp = load(latestCp);
    net          = cp.net;
    bestNet      = cp.bestNet;
    bestValLoss  = cp.bestValLoss;
    trainHistory = cp.trainHistory;
    valHistory   = cp.valHistory;
    totalEpochs  = cp.totalEpochs;
    startPhaseIdx = cp.phaseIdx;
    normStats    = cp.normStats;
    fprintf('  Loaded: bestVal=%.4f, epochs=%d, resuming phase %d\n', ...
        bestValLoss, totalEpochs, startPhaseIdx);
end

%% ── dlaccelerate — accelerated loss function ────────────────────────────────
% NOTE: dlaccelerate traces on first call per unique input size.
% We create one accelerated function handle; it re-traces for each phase
% when window size changes.
accLossFcn = dlaccelerate(@computeLossAndGrad);

%% ── Main training loop ──────────────────────────────────────────────────────
tStart     = tic;
tLastCp    = tic;
iterGlobal = 0;   % global Adam iteration counter

for phIdx = startPhaseIdx:numel(phases)
    ph = phases{phIdx};
    phName    = ph.name;
    nEpochs   = ph.epochs;
    winSamples = round(ph.window_ms * 1e-3 / TS);  % window length at TS
    winSub     = floor((winSamples - 1) / STEP_SKIP) + 1; % subsampled window
    lr         = ph.lr;
    lam_J      = ph.lambda_J;
    lam_rip    = ph.lambda_ripple;
    winPerProf = ph.win_per_prof;

    % Clear dlaccelerate cache when window size changes (new computation graph)
    clearCache(accLossFcn);

    fprintf('\n--- %s: %d epochs | window=%.0fms (%d sub-steps) | LR=%.1e | λ_J=%.3f | λ_rip=%.2f ---\n', ...
        phName, nEpochs, ph.window_ms, winSub, lr, lam_J, lam_rip);

    % Per-phase Adam state (reset each phase like PyTorch creates new optimizer)
    avgGrad   = [];
    avgSqGrad = [];
    phIter    = 0;

    noImprove = 0;
    phBestVal = bestValLoss;

    for epoch = 1:nEpochs
        totalEpochs = totalEpochs + 1;
        tEpoch = tic;

        epochLoss  = 0;
        nWindows   = 0;
        perm       = randperm(nTrain);

        for pIdx = perm
            u_p = train_u{pIdx};   % [T x 1] single
            y_p = train_y{pIdx};   % [T x 2] single
            T   = size(u_p, 1);

            nWinTotal = max(1, floor(T / winSamples));
            nUse      = min(winPerProf, nWinTotal);
            winIdxs   = randperm(nWinTotal, nUse) - 1;  % 0-based starts

            for ww = 1:nUse
                s = winIdxs(ww) * winSamples + 1;       % 1-based
                e = min(s + winSamples - 1, T);
                if (e - s + 1) < 20, continue; end

                % Extract window as unformatted dlarrays (formats stripped inside loss fn)
                u_win = dlarray(u_p(s:e));          % [len x 1]
                y_win = dlarray(y_p(s:e, :));       % [len x 2]
                x0    = dlarray(y_p(s, :)');        % [2 x 1]

                % Compute loss + gradients
                phIter    = phIter + 1;
                iterGlobal = iterGlobal + 1;

                [loss, grads] = dlfeval(accLossFcn, ...
                    net, x0, u_win, y_win, ...
                    normStats, DT_RK4, STEP_SKIP, ...
                    lam_rip, b_lp, a_lp, b_hp, a_hp);

                if isnan(extractdata(loss)) || isinf(extractdata(loss))
                    continue;
                end

                % Gradient clipping (same as PyTorch clip_grad_norm_)
                grads = clipGradients(grads, GRAD_CLIP);

                % Adam update
                [net, avgGrad, avgSqGrad] = adamupdate(net, grads, ...
                    avgGrad, avgSqGrad, phIter, lr);

                epochLoss = epochLoss + extractdata(loss);
                nWindows  = nWindows + 1;
            end

            % Jacobian loss (every 5th profile processed, like PyTorch every 5th window)
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

        avgTrain   = epochLoss / max(nWindows, 1);
        epochTime  = toc(tEpoch);
        trainHistory(end+1) = avgTrain; %#ok<AGROW>

        % Validation
        if mod(epoch, VAL_FREQ) == 0 || epoch == 1 || epoch == nEpochs
            valLoss = computeValidationLoss(net, val_u, val_y, normStats, ...
                DT_RK4, STEP_SKIP, winSamples);
            valHistory(end+1) = valLoss; %#ok<AGROW>

            improved = '';
            if valLoss < bestValLoss
                bestValLoss = valLoss;
                bestNet     = net;
                noImprove   = 0;
                improved    = ' ***BEST***';
                % Save immediately on new best
                saveCheckpoint(CHECKPOINT_DIR, 'dlt_best.mat', ...
                    net, bestNet, bestValLoss, trainHistory, valHistory, ...
                    totalEpochs, phIdx, normStats);
            else
                noImprove = noImprove + 1;
            end

            elapsed = toc(tStart) / 60;
            fprintf('  [%s] Ep %3d/%d | Train: %.4f | Val: %.4f | %.1fs/ep | %.1fmin tot%s\n', ...
                phName, epoch, nEpochs, avgTrain, valLoss, epochTime, elapsed, improved);
        else
            fprintf('  [%s] Ep %3d/%d | Train: %.4f | %.1fs/ep\n', ...
                phName, epoch, nEpochs, avgTrain, epochTime);
        end

        % Periodic checkpoint (every CHECKPOINT_MIN minutes)
        if toc(tLastCp) > CHECKPOINT_MIN * 60
            saveCheckpoint(CHECKPOINT_DIR, 'dlt_latest.mat', ...
                net, bestNet, bestValLoss, trainHistory, valHistory, ...
                totalEpochs, phIdx, normStats);
            tLastCp = tic;
        end

        % Early stopping
        if noImprove >= EARLY_STOP
            fprintf('  Early stopping: %d epochs without improvement\n', noImprove);
            break;
        end
    end

    % Save at end of each phase
    saveCheckpoint(CHECKPOINT_DIR, sprintf('dlt_after_%s.mat', phName), ...
        net, bestNet, bestValLoss, trainHistory, valHistory, ...
        totalEpochs, phIdx + 1, normStats);
    fprintf('  Phase %s complete. Best val: %.4f\n', phName, bestValLoss);
end

%% ── Final summary ───────────────────────────────────────────────────────────
totalMin = toc(tStart) / 60;
fprintf('\n=== Training complete ===\n');
fprintf('  Total time: %.1f min (%.1f h)\n', totalMin, totalMin/60);
fprintf('  Total epochs: %d\n', totalEpochs);
fprintf('  Best val loss: %.4f\n', bestValLoss);
fprintf('  PyTorch reference: 0.040\n');
fprintf('  Ratio vs PyTorch: %.2fx\n', bestValLoss / 0.040);

% Save final model
saveCheckpoint(CHECKPOINT_DIR, 'dlt_final.mat', ...
    bestNet, bestNet, bestValLoss, trainHistory, valHistory, ...
    totalEpochs, numel(phases)+1, normStats);
fprintf('Final model saved.\n');

end  % function train_dlt_neural_ode


%% ========================================================================
%% Loss function (called via dlfeval + dlaccelerate)
%% ========================================================================
function [loss, grads] = computeLossAndGrad(net, x0, u_win, y_win, ...
        normStats, dt, step_skip, lambda_ripple, b_lp, a_lp, b_hp, a_hp)
% All dlarray inputs — strip format labels so arithmetic is format-agnostic.
% x0:    [2 x 1] initial state (physical)
% u_win: [T x 1] duty cycle
% y_win: [T x 2] ground truth states

x0    = stripdims(x0);
u_win = stripdims(u_win);
y_win = stripdims(y_win);        % [T x 2] unformatted

% Integrate with RK4 → x_pred: [2 x T_sub] unformatted
x_pred = rk4Integrate(net, x0, u_win, normStats, dt, step_skip);

% Subsample ground truth  →  y_sub: [2 x T_sub]
T_raw  = size(u_win, 1);
T_sub  = size(x_pred, 2);
idx    = 1 : step_skip : T_raw;
idx    = idx(1:T_sub);
y_sub  = y_win(idx, :)';         % [2 x T_sub]

% Normalise in [2 x T_sub] space
x_mean = normStats.x_mean(:);    % [2 x 1]
x_std  = normStats.x_std(:);     % [2 x 1]
x_pred_n = (x_pred - x_mean) ./ x_std;   % [2 x T_sub]
y_sub_n  = (y_sub  - x_mean) ./ x_std;   % [2 x T_sub]

% Trajectory loss (with optional LP/HP filter)
if lambda_ripple > 0 && ~isempty(b_lp) && T_sub > 20
    skip = max(floor(T_sub / 10), 5);
    % Filter expects [T x 2] — transpose, filter, compare
    xpT = x_pred_n';   % [T_sub x 2]
    ysT = y_sub_n';    % [T_sub x 2]

    x_lp = applyIIR(xpT, b_lp, a_lp);
    y_lp = applyIIR(ysT, b_lp, a_lp);
    x_hp = applyIIR(xpT, b_hp, a_hp);
    y_hp = applyIIR(ysT, b_hp, a_hp);

    loss_avg    = mean((x_lp(skip+1:end,:) - y_lp(skip+1:end,:)).^2, 'all');
    loss_ripple = mean((x_hp(skip+1:end,:) - y_hp(skip+1:end,:)).^2, 'all');
    loss = loss_avg + lambda_ripple * loss_ripple;
else
    loss = mean((x_pred_n - y_sub_n).^2, 'all');
end

grads = dlgradient(loss, net.Learnables);
end


%% ── RK4 integrator ──────────────────────────────────────────────────────────
function x_traj = rk4Integrate(net, x0, u_win, normStats, dt, step_skip)
% x0:    [2 x 1] unformatted dlarray (physical)
% u_win: [T x 1] unformatted dlarray
% Returns x_traj: [2 x T_sub] unformatted

T_raw  = size(u_win, 1);
T_sub  = floor((T_raw - 1) / step_skip) + 1;
x      = x0;
x_traj = x;

for k = 1 : (T_sub - 1)
    orig_idx = min((k-1) * step_skip + 1, T_raw);
    d_k = u_win(orig_idx);   % scalar unformatted

    k1 = mlpEval(net, x,               d_k, normStats);
    k2 = mlpEval(net, x + 0.5*dt*k1,  d_k, normStats);
    k3 = mlpEval(net, x + 0.5*dt*k2,  d_k, normStats);
    k4 = mlpEval(net, x + dt*k3,       d_k, normStats);

    x = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4);
    x_traj = cat(2, x_traj, x);   % [2 x k+1]
end
end


%% ── MLP evaluation ──────────────────────────────────────────────────────────
function dxdt = mlpEval(net, x_phys, d_phys, normStats)
% x_phys: [2 x 1] unformatted, d_phys: scalar unformatted
% Returns dxdt: [2 x 1] unformatted (physical units V/s, A/s)
x_n = (x_phys - normStats.x_mean(:)) ./ normStats.x_std(:);  % [2 x 1]
d_n = (d_phys - normStats.u_mean)    ./ normStats.u_std;       % scalar

% Wrap as 'CB' (channels x batch=1) for featureInputLayer
inp  = dlarray([x_n; d_n], 'CB');       % [3 x 1]
raw  = stripdims(forward(net, inp));    % [2 x 1] unformatted, ~[-1,1]
dxdt = raw .* normStats.dxdt_scale(:); % [2 x 1] physical
end


%% ── Normalisation helper ─────────────────────────────────────────────────────
function x_n = normX(x_phys, ns)
% x_phys: [2 x N] physical → [2 x N] normalised
x_n = (x_phys - ns.x_mean(:)) ./ ns.x_std(:);
end


%% ── IIR filter (Direct Form II Transposed) — differentiable ─────────────────
function y = applyIIR(x, b, a)
% x: [T x C] dlarray, b/a: filter coefficients (single)
% Returns y: [T x C] filtered (same as scipy.signal.lfilter)
[T, C] = size(x);
order  = numel(a) - 1;
y      = zeros(T, C, 'like', x);
d      = zeros(order, C, 'like', x);   % delay line

for t = 1:T
    y(t,:) = b(1) * x(t,:) + d(1,:);
    for i = 1:order-1
        d(i,:) = b(i+1) * x(t,:) - a(i+1) * y(t,:) + d(i+1,:);
    end
    d(order,:) = b(order+1) * x(t,:) - a(order+1) * y(t,:);
end
end


%% ── Jacobian loss ────────────────────────────────────────────────────────────
function [lossJ, gradsJ] = computeJacobianLoss(net, A_targets, x_ops, u_ops, normStats, lambda_J)
% Analytical Jacobian ∂f/∂x via chain rule through MLP weights.
% dljacobian cannot trace data inputs — use explicit chain rule instead.
%
% Architecture: f(x) = dxdt_scale .* W3*tanh(W2*tanh(W1*[x_n;u_n]+b1)+b2)+b3
% ∂f/∂x_phys = diag(dxdt_scale) * W3 * diag(1-a2^2) * W2 * diag(1-a1^2) * W1[:,1:2] / x_std
%
% W1,b1,W2,b2,W3 are in net.Learnables at positions 1-5 (fc1,fc2,fc3).
% This formulation is fully differentiable w.r.t. net.Learnables.

% Extract weights (order: fc1/W, fc1/b, fc2/W, fc2/b, fc3/W, fc3/b)
W1 = net.Learnables.Value{1};   % [64 x 3]
b1 = net.Learnables.Value{2};   % [64 x 1]
W2 = net.Learnables.Value{3};   % [64 x 64]
b2 = net.Learnables.Value{4};   % [64 x 1]
W3 = net.Learnables.Value{5};   % [2 x 64]

% Subset of operating points (same as PyTorch: 4 fixed + 2 random)
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
    % Normalize operating point
    x_n = (x_ops(j,:)' - normStats.x_mean(:)) ./ normStats.x_std(:);  % [2 x 1]
    d_n = (u_ops(j)    - normStats.u_mean)     ./ normStats.u_std;      % scalar
    inp = [x_n; d_n];   % [3 x 1]

    % Forward pass (needed to get activations for Jacobian)
    z1 = W1 * inp + b1;       % [64 x 1]
    a1 = tanh(z1);             % [64 x 1]
    z2 = W2 * a1 + b2;        % [64 x 1]
    a2 = tanh(z2);             % [64 x 1]

    % Activation derivatives: tanh'(z) = 1 - tanh^2(z)
    D1 = 1 - a1.^2;   % [64 x 1]
    D2 = 1 - a2.^2;   % [64 x 1]

    % Chain rule: ∂f/∂x_phys = dxdt_scale .* W3 * D2 .* W2 * D1 .* W1[:,1:2] / x_std
    W1_x  = W1(:, 1:2);                   % [64 x 2] (x columns only)
    J_phys = normStats.dxdt_scale(:) .* (W3 * (D2 .* (W2 * (D1 .* W1_x)))) ...
             ./ normStats.x_std(:);        % [2 x 2]

    A_ref = dlarray(single(squeeze(A_targets(j,:,:))'));  % [2 x 2]
    A_sc  = max(abs(extractdata(A_ref)), [], 'all') + 1;
    lossJ = lossJ + mean(((J_phys - A_ref) / A_sc).^2, 'all');
end
lossJ  = lossJ * (lambda_J / numel(idx));
gradsJ = dlgradient(lossJ, net.Learnables);
end


%% ── Validation ───────────────────────────────────────────────────────────────
function valLoss = computeValidationLoss(net, val_u, val_y, normStats, dt, step_skip, winSamples)
valLoss = 0;
nVal    = numel(val_u);
for v = 1:nVal
    u_v = val_u{v};
    y_v = val_y{v};
    T   = size(u_v, 1);

    % Use up to 3 non-overlapping windows from each val profile
    nWin = min(3, floor(T / winSamples));
    profLoss = 0;
    nW = 0;
    for w = 0:nWin-1
        s = w * winSamples + 1;
        e = min(s + winSamples - 1, T);
        if (e - s + 1) < 20, continue; end

        u_win = dlarray(single(u_v(s:e)));
        y_win = dlarray(single(y_v(s:e,:)));
        x0    = dlarray(single(y_v(s,:)'));

        x_pred   = rk4Integrate(net, x0, u_win, normStats, dt, step_skip);
        T_sub    = size(x_pred, 2);
        idx      = 1:step_skip:size(y_win,1);
        idx      = idx(1:T_sub);
        y_sub    = stripdims(y_win(idx,:))';   % [2 x T_sub]

        x_pred_n = normX(x_pred, normStats);   % [2 x T_sub]
        y_sub_n  = normX(y_sub,  normStats);   % [2 x T_sub]
        profLoss = profLoss + extractdata(mean((x_pred_n - y_sub_n).^2, 'all'));
        nW = nW + 1;
    end
    if nW > 0
        valLoss = valLoss + profLoss / nW;
    end
end
valLoss = valLoss / nVal;
end


%% ── Gradient clipping ────────────────────────────────────────────────────────
function grads = clipGradients(grads, maxNorm)
% Clip by global gradient norm (same as PyTorch clip_grad_norm_)
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
        trainHistory, valHistory, totalEpochs, phaseIdx, normStats)
cpFile = fullfile(cpDir, fname);
save(cpFile, 'net', 'bestNet', 'bestValLoss', 'trainHistory', ...
    'valHistory', 'totalEpochs', 'phaseIdx', 'normStats', '-v7.3');
fprintf('    [checkpoint] %s (val=%.4f)\n', fname, bestValLoss);
end
