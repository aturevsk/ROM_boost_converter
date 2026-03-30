% train_neural_ss_normalized.m
% Continuous-time Neural State-Space with output scaling via last-layer init.
%
% Key insight: The MLP outputs dx_n/dt, which has std ~[250, 1757] in
% normalized space. A tanh-based MLP can't produce these magnitudes.
% PyTorch solves this with dxdt_scale buffer. We solve it by initializing
% the last FC layer weights proportional to dxdt_std/x_std, so the MLP
% output starts at the right magnitude and adapts from there.
%
% Architecture: dx_n/dt = MLP([x_n; u_n]) — continuous time, dlode45
% Curriculum: Phase 1 (5ms windows), Phase 2 (20ms windows)

function train_neural_ss_normalized(mode)
if nargin < 1, mode = 'full'; end
isTest = strcmp(mode, 'test');

baseDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(baseDir);
run(fullfile(repoRoot, 'startup.m'));
dataDir = fullfile(repoRoot, 'data');
modelDataDir = fullfile(repoRoot, 'model_data');
checkpointDir = fullfile(baseDir, 'checkpoints_nss');
if ~exist(checkpointDir, 'dir'), mkdir(checkpointDir); end

%% 1. Load training data (same as PyTorch)
fprintf('=== Loading training data ===\n');
load(fullfile(dataDir, 'boost_nss_training_data.mat'), 'allU', 'allY');
fprintf('Loaded %d profiles\n', numel(allU));

Ts = 5e-6;
nx = 2;

%% 2. Compute normalization statistics from training data
nTotal = numel(allU);
nVal = 3;
nTrainFull = nTotal - nVal;
if isTest
    nTrain = min(3, nTrainFull);
else
    nTrain = nTrainFull;
end

allDuty = []; allVout = []; allIL = []; allDvout = []; allDil = [];
for i = 1:nTrain
    allDuty = [allDuty; allU{i}.duty]; %#ok<AGROW>
    allVout = [allVout; allY{i}.Vout]; %#ok<AGROW>
    allIL   = [allIL;   allY{i}.iL];   %#ok<AGROW>
    allDvout = [allDvout; diff(allY{i}.Vout)/Ts]; %#ok<AGROW>
    allDil   = [allDil;   diff(allY{i}.iL)/Ts];   %#ok<AGROW>
end

normStats.u_mean = mean(allDuty);
normStats.u_std  = std(allDuty);
normStats.y_mean = [mean(allVout), mean(allIL)];
normStats.y_std  = [std(allVout),  std(allIL)];

% Output scaling: ratio of derivative scale to state scale
% This is what PyTorch handles via dxdt_scale buffer
x_std = [std(allVout); std(allIL)];
dxdt_std = [std(allDvout); std(allDil)];
output_scale = dxdt_std ./ x_std;

fprintf('Normalization stats:\n');
fprintf('  duty:  mean=%.3f, std=%.3f\n', normStats.u_mean, normStats.u_std);
fprintf('  Vout:  mean=%.2f, std=%.2f\n', normStats.y_mean(1), normStats.y_std(1));
fprintf('  iL:    mean=%.2f, std=%.2f\n', normStats.y_mean(2), normStats.y_std(2));
fprintf('  dxdt_std: [%.0f, %.0f]\n', dxdt_std);
fprintf('  output_scale (dxdt_std/x_std): [%.1f, %.1f]\n', output_scale);

%% 3. Normalize data and create iddata objects
fprintf('\n=== Normalizing data ===\n');
data_norm = cell(nTotal, 1);

for i = 1:nTotal
    duty_n = (allU{i}.duty - normStats.u_mean) / normStats.u_std;
    vout_n = (allY{i}.Vout - normStats.y_mean(1)) / normStats.y_std(1);
    iL_n   = (allY{i}.iL   - normStats.y_mean(2)) / normStats.y_std(2);
    data_norm{i} = iddata([vout_n, iL_n], duty_n, Ts);
end

trainData = merge(data_norm{1:nTrain});
trainData.InputName = {'duty_n'};
trainData.OutputName = {'Vout_n', 'iL_n'};

valData = cell(nVal, 1);
for v = 1:nVal
    valData{v} = data_norm{nTrainFull + v};
    valData{v}.InputName = {'duty_n'};
    valData{v}.OutputName = {'Vout_n', 'iL_n'};
end
fprintf('Train: %d experiments, Val: %d\n', nTrain, nVal);

%% 4. Create Continuous-Time NSS with output-scaled last layer
fprintf('\n=== Creating Neural State-Space ===\n');

nss = idNeuralStateSpace(nx, NumInputs=1, NumOutputs=2, Ts=0);
nss.InputName = {'duty_n'};
nss.OutputName = {'Vout_n', 'iL_n'};
nss.StateName = {'x1_n', 'x2_n'};

stateNet = createMLPNetwork(nss, 'state', ...
    LayerSizes=[64 64], ...
    Activations='tanh', ...
    WeightsInitializer='glorot');

% Scale last layer (dxdt) weights by output_scale so the MLP can
% produce derivatives of the right magnitude from the start.
% Without this, tanh output ~[-1,1] gets mapped to ~[-0.1, 0.1] by glorot,
% but dx_n/dt needs to be ~O(250-1757).
for k = 1:size(stateNet.Learnables, 1)
    if strcmp(stateNet.Learnables.Layer{k}, 'dxdt')
        if contains(stateNet.Learnables.Parameter{k}, 'Weights')
            W = extractdata(stateNet.Learnables.Value{k});
            W(1,:) = W(1,:) * output_scale(1);
            W(2,:) = W(2,:) * output_scale(2);
            stateNet.Learnables.Value{k} = dlarray(W);
            fprintf('Scaled dxdt weights: row1 * %.1f, row2 * %.1f\n', output_scale);
        elseif contains(stateNet.Learnables.Parameter{k}, 'Bias')
            b = extractdata(stateNet.Learnables.Value{k});
            b(1) = b(1) * output_scale(1);
            b(2) = b(2) * output_scale(2);
            stateNet.Learnables.Value{k} = dlarray(b);
        end
    end
end
nss.StateNetwork = stateNet;

fprintf('Continuous-time NSS: nx=%d, nu=1, ny=2\n', nx);
fprintf('Architecture: dx_n/dt = MLP([x_n; u_n]) with scaled output layer\n');
nParams = sum(cellfun(@numel, nss.StateNetwork.Learnables.Value));
fprintf('Parameters: %d\n', nParams);

%% 5. Training
fprintf('\n=== Training ===\n');

% ODE solver: MaxStepSize=50us matches PyTorch RK4 step (step_skip=10*5us)
odeSolverOpts = nssTrainingOptions('adam').ODESolverOptions;
odeSolverOpts.MaxStepSize = 50e-6;
fprintf('ODE solver: dlode45, MaxStep=50us\n');

if isTest
    opts = nssTrainingOptions('adam');
    opts.MaxEpochs = 50;
    opts.LearnRate = 0.001;
    opts.LossFcn = 'MeanSquaredError';
    opts.WindowSize = 1000;
    opts.InputInterSample = 'zoh';
    opts.ODESolverOptions = odeSolverOpts;
    opts.PlotLossFcn = false;

    fprintf('TEST mode: 50 epochs, window=1000 (5ms)\n');
    tic;
    nssEst = nlssest(trainData, nss, opts, ...
        UseLastExperimentForValidation=true);
    trainTime = toc;
    fprintf('Training: %.1f min\n', trainTime/60);
else
    % Phase 1: 5ms windows — learn basic dynamics
    opts1 = nssTrainingOptions('adam');
    opts1.MaxEpochs = 300;
    opts1.LearnRate = 0.001;
    opts1.LearnRateSchedule = 'piecewise';
    opts1.LearnRateDropPeriod = 100;
    opts1.LearnRateDropFactor = 0.5;
    opts1.LossFcn = 'MeanSquaredError';
    opts1.WindowSize = 1000;  % 5ms
    opts1.InputInterSample = 'zoh';
    opts1.ODESolverOptions = odeSolverOpts;
    opts1.PlotLossFcn = false;

    fprintf('Phase 1: 300 epochs, window=1000 (5ms), LR=0.001\n');
    tic;
    nss1 = nlssest(trainData, nss, opts1, ...
        UseLastExperimentForValidation=true);
    t1 = toc;
    fprintf('Phase 1 done: %.1f min\n', t1/60);

    save(fullfile(checkpointDir, 'nss_phase1.mat'), 'nss1', 'normStats');
    fprintf('Phase 1 checkpoint saved.\n');

    % Phase 2: 20ms windows — learn longer-horizon dynamics
    opts2 = nssTrainingOptions('adam');
    opts2.MaxEpochs = 500;
    opts2.LearnRate = 5e-4;  % match PyTorch Phase 2
    opts2.LearnRateSchedule = 'piecewise';
    opts2.LearnRateDropPeriod = 150;
    opts2.LearnRateDropFactor = 0.5;
    opts2.LossFcn = 'MeanSquaredError';
    opts2.WindowSize = 4000;  % 20ms
    opts2.InputInterSample = 'zoh';
    opts2.ODESolverOptions = odeSolverOpts;
    opts2.PlotLossFcn = false;

    fprintf('Phase 2: 500 epochs, window=4000 (20ms), LR=5e-4\n');
    tic;
    nssEst = nlssest(trainData, nss1, opts2, ...
        UseLastExperimentForValidation=true);
    t2 = toc;
    trainTime = t1 + t2;
    fprintf('Phase 2 done: %.1f min\n', t2/60);
    fprintf('Total training: %.1f min\n', trainTime/60);

    save(fullfile(checkpointDir, 'nss_phase2.mat'), 'nssEst', 'normStats');
end

%% 6. Validation (denormalize predictions, correct ICs)
fprintf('\n=== Validation (denormalized, correct ICs) ===\n');

Y_val_raw = allY(nTrainFull+1:nTotal);

fig = figure('Position', [100 100 900 600], 'Name', 'NSS Validation');

for v = 1:min(3, numel(valData))
    y_raw = [Y_val_raw{v}.Vout, Y_val_raw{v}.iL];
    x0_n = [(y_raw(1,1) - normStats.y_mean(1)) / normStats.y_std(1);
            (y_raw(1,2) - normStats.y_mean(2)) / normStats.y_std(2)];

    u_val_only = iddata([], valData{v}.InputData, Ts);
    u_val_only.InputName = {'duty_n'};
    opt = simOptions('InitialCondition', x0_n);
    yPred_n = sim(nssEst, u_val_only, opt);

    yPred_data = yPred_n.OutputData;
    vout_pred = yPred_data(:,1) * normStats.y_std(1) + normStats.y_mean(1);
    iL_pred   = yPred_data(:,2) * normStats.y_std(2) + normStats.y_mean(2);

    vout_true = Y_val_raw{v}.Vout;
    iL_true   = Y_val_raw{v}.iL;
    T_val = min(numel(vout_pred), numel(vout_true));
    vout_pred = vout_pred(1:T_val);
    iL_pred = iL_pred(1:T_val);
    vout_true = vout_true(1:T_val);
    iL_true = iL_true(1:T_val);

    t_plot = (0:T_val-1)' * Ts * 1e3;

    vout_rmse = sqrt(mean((vout_pred - vout_true).^2));
    iL_rmse   = sqrt(mean((iL_pred - iL_true).^2));
    fprintf('  Val %d: Vout RMSE=%.3fV, iL RMSE=%.3fA\n', v, vout_rmse, iL_rmse);

    subplot(3, 2, (v-1)*2+1);
    plot(t_plot, vout_true, 'b-', 'LineWidth', 1); hold on;
    plot(t_plot, vout_pred, 'r--', 'LineWidth', 1.5);
    ylabel('Vout (V)'); grid on;
    if v == 1, legend('Simscape', 'NSS'); title('Vout'); end

    subplot(3, 2, (v-1)*2+2);
    plot(t_plot, iL_true, 'b-', 'LineWidth', 1); hold on;
    plot(t_plot, iL_pred, 'r--', 'LineWidth', 1.5);
    ylabel('iL (A)'); grid on;
    if v == 1, title('iL'); end
end

sgtitle('Neural State-Space (CT, output-scaled init)');
saveas(fig, fullfile(modelDataDir, 'nss_normalized_validation.png'));
close(fig);

%% 7. Save final model
save(fullfile(modelDataDir, 'boost_nss_normalized.mat'), 'nssEst', 'normStats');
fprintf('\nModel saved to boost_nss_normalized.mat\n');
fprintf('Done.\n');
end
