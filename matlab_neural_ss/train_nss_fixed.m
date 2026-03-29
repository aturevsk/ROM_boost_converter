% train_nss_fixed.m
% Fixed Neural State-Space training addressing all identified gaps vs PyTorch:
%
% FIX 1: Discrete-time (Ts=5us) instead of continuous-time
%         - Eliminates adaptive dlode45 solver (was AbsTol=RelTol=0.01)
%         - Training/validation/deployment all use same x[k+1] = x[k] + f(x,u)
%         - Skip connection built-in by default in DT NSS
%
% FIX 2: No output scaling needed
%         - DT increments dx_n ~ O(0.001-0.01), within glorot init range
%         - CT needed scaling because dx_n/dt ~ O(250-1760)
%
% FIX 3: 3-phase curriculum matching PyTorch exactly
%         - Phase 1: 300 epochs, 5ms windows, LR=1e-3
%         - Phase 2: 500 epochs, 20ms windows, LR=5e-4
%         - Phase 3: 200 epochs, 20ms windows, LR=1e-4
%
% FIX 4: Gradient clipping via Lambda regularization
%
% FIX 5: 15-minute checkpoints and plateau detection for long runs
%
% Usage:
%   train_nss_fixed('test')    % quick 5-minute test
%   train_nss_fixed('full')    % full 3-phase training
%   train_nss_fixed('extend')  % resume from best checkpoint, 8hr extended run

function train_nss_fixed(mode, maxHours)
if nargin < 1, mode = 'full'; end
if nargin < 2
    if strcmp(mode, 'extend'), maxHours = 8.0;
    else, maxHours = Inf; end
end

isTest = strcmp(mode, 'test');
isExtend = strcmp(mode, 'extend');

SAVE_INTERVAL_MIN = 15;
PLATEAU_ROUNDS = 12;
PLATEAU_REL_TOL = 1e-4;

Ts = 5e-6;
nx = 2;

baseDir = fileparts(mfilename('fullpath'));
testsDir = fileparts(baseDir);
modelsDir = fullfile(testsDir, 'models');
checkpointDir = fullfile(baseDir, 'checkpoints');
if ~exist(checkpointDir, 'dir'), mkdir(checkpointDir); end

logFile = fullfile(baseDir, 'training_log.txt');

logMsg(logFile, '============================================================');
logMsg(logFile, 'Fixed NSS Training (Discrete-Time)');
logMsg(logFile, sprintf('  Mode:          %s', mode));
logMsg(logFile, sprintf('  Ts:            %g (discrete-time)', Ts));
logMsg(logFile, sprintf('  Architecture:  MLP [64,64] tanh + skip connection'));
logMsg(logFile, sprintf('  Max hours:     %.1f', maxHours));
logMsg(logFile, '============================================================');

%% 1. Load training data (same as PyTorch)
logMsg(logFile, 'Loading training data...');
load(fullfile(modelsDir, 'boost_nss_training_data.mat'), 'allU', 'allY');

nTotal = numel(allU);
nVal = 3;
nTrainFull = nTotal - nVal;
if isTest
    nTrain = min(3, nTrainFull);
else
    nTrain = nTrainFull;
end

%% 2. Compute normalization (same stats as PyTorch)
allDuty = []; allVout = []; allIL = [];
for i = 1:nTrain
    allDuty = [allDuty; allU{i}.duty]; %#ok<AGROW>
    allVout = [allVout; allY{i}.Vout]; %#ok<AGROW>
    allIL   = [allIL;   allY{i}.iL];   %#ok<AGROW>
end

normStats.u_mean = mean(allDuty);
normStats.u_std  = std(allDuty);
normStats.y_mean = [mean(allVout), mean(allIL)];
normStats.y_std  = [std(allVout),  std(allIL)];

logMsg(logFile, sprintf('  duty:  mean=%.3f, std=%.3f', normStats.u_mean, normStats.u_std));
logMsg(logFile, sprintf('  Vout:  mean=%.2f, std=%.2f', normStats.y_mean(1), normStats.y_std(1)));
logMsg(logFile, sprintf('  iL:    mean=%.2f, std=%.2f', normStats.y_mean(2), normStats.y_std(2)));

%% 3. Normalize data
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

valDataCells = cell(nVal, 1);
Y_val_raw = allY(nTrainFull+1:nTotal);
for v = 1:nVal
    valDataCells{v} = data_norm{nTrainFull + v};
    valDataCells{v}.InputName = {'duty_n'};
    valDataCells{v}.OutputName = {'Vout_n', 'iL_n'};
end
logMsg(logFile, sprintf('  Train: %d, Val: %d', nTrain, nVal));

%% 4. Create DISCRETE-TIME NSS (FIX #1 + #2)
logMsg(logFile, '');
logMsg(logFile, '=== Creating Discrete-Time NSS ===');

nss = idNeuralStateSpace(nx, NumInputs=1, NumOutputs=2, Ts=Ts);
nss.InputName = {'duty_n'};
nss.OutputName = {'Vout_n', 'iL_n'};
nss.StateName = {'x1_n', 'x2_n'};

stateNet = createMLPNetwork(nss, 'state', ...
    LayerSizes=[64 64], ...
    Activations='tanh', ...
    WeightsInitializer='glorot');
nss.StateNetwork = stateNet;

nParams = sum(cellfun(@numel, nss.StateNetwork.Learnables.Value));
logMsg(logFile, sprintf('  Discrete-time NSS: nx=%d, Ts=%g', nx, Ts));
logMsg(logFile, sprintf('  Architecture: x[k+1] = x[k] + MLP([x_n; u_n]) (skip built-in)'));
logMsg(logFile, sprintf('  Parameters: %d', nParams));
logMsg(logFile, sprintf('  No output scaling needed (dx_n ~ O(0.001-0.01))'));

%% 5. Training phases (FIX #3: 3-phase curriculum matching PyTorch)
if isTest
    phases = {
        struct('name', 'TEST', 'epochs', 30, 'window', 1000, 'lr', 1e-3, ...
               'lr_drop_period', 10, 'lr_drop_factor', 0.5, 'lambda', 0)
    };
elseif isExtend
    phases = {};  % Skip to extended training loop below
else
    phases = {
        struct('name', 'Phase1_5ms', 'epochs', 300, 'window', 1000, ...
               'lr', 1e-3, 'lr_drop_period', 100, 'lr_drop_factor', 0.5, ...
               'lambda', 1e-5)
        struct('name', 'Phase2_20ms', 'epochs', 500, 'window', 4000, ...
               'lr', 5e-4, 'lr_drop_period', 150, 'lr_drop_factor', 0.5, ...
               'lambda', 1e-5)
        struct('name', 'Phase3_Finetune', 'epochs', 200, 'window', 4000, ...
               'lr', 1e-4, 'lr_drop_period', 100, 'lr_drop_factor', 0.5, ...
               'lambda', 0)
    };
end

%% 6. Resume from checkpoint if extending
bestValLoss = Inf;
bestModel = [];
tStart = tic;

if isExtend
    cpFile = fullfile(checkpointDir, 'nss_fixed_best.mat');
    if ~isfile(cpFile)
        cpFile = fullfile(checkpointDir, 'nss_fixed_phase3.mat');
    end
    if ~isfile(cpFile)
        cpFile = fullfile(checkpointDir, 'nss_fixed_phase2.mat');
    end
    if isfile(cpFile)
        logMsg(logFile, sprintf('Resuming from: %s', cpFile));
        cp = load(cpFile);
        nss = cp.nssEst;
        normStats = cp.normStats;
        [bestValLoss, vout_rmses, iL_rmses] = computeValidation(nss, valDataCells, Y_val_raw, normStats, Ts);
        logMsg(logFile, sprintf('  Initial val loss: %.6f', bestValLoss));
        for v = 1:numel(vout_rmses)
            logMsg(logFile, sprintf('  Val %d: Vout RMSE=%.3fV, iL RMSE=%.3fA', v, vout_rmses(v), iL_rmses(v)));
        end
        bestModel = nss;
    else
        logMsg(logFile, 'No checkpoint found — running full training first');
        isExtend = false;
        phases = {
            struct('name', 'Phase1_5ms', 'epochs', 300, 'window', 1000, ...
                   'lr', 1e-3, 'lr_drop_period', 100, 'lr_drop_factor', 0.5, ...
                   'lambda', 1e-5)
            struct('name', 'Phase2_20ms', 'epochs', 500, 'window', 4000, ...
                   'lr', 5e-4, 'lr_drop_period', 150, 'lr_drop_factor', 0.5, ...
                   'lambda', 1e-5)
            struct('name', 'Phase3_Finetune', 'epochs', 200, 'window', 4000, ...
                   'lr', 1e-4, 'lr_drop_period', 100, 'lr_drop_factor', 0.5, ...
                   'lambda', 0)
        };
    end
end

%% 7. Run curriculum phases
for p = 1:numel(phases)
    ph = phases{p};
    logMsg(logFile, '');
    logMsg(logFile, sprintf('--- %s: %d epochs, window=%d (%.0fms), LR=%.0e, lambda=%.0e ---', ...
        ph.name, ph.epochs, ph.window, ph.window*Ts*1e3, ph.lr, ph.lambda));

    opts = nssTrainingOptions('adam');
    opts.MaxEpochs = ph.epochs;
    opts.LearnRate = ph.lr;
    opts.LearnRateSchedule = 'piecewise';
    opts.LearnRateDropPeriod = ph.lr_drop_period;
    opts.LearnRateDropFactor = ph.lr_drop_factor;
    opts.LossFcn = 'MeanSquaredError';
    opts.WindowSize = ph.window;
    opts.InputInterSample = 'zoh';
    opts.Lambda = ph.lambda;  % FIX #4: L2 regularization
    opts.PlotLossFcn = false;
    % No ODE solver options needed — discrete-time!

    phaseStart = tic;
    nss = nlssest(trainData, nss, opts, ...
        UseLastExperimentForValidation=true);
    phaseTime = toc(phaseStart);

    % Validate
    [valLoss, vout_rmses, iL_rmses] = computeValidation(nss, valDataCells, Y_val_raw, normStats, Ts);
    meanVout = mean(vout_rmses);
    meanIL = mean(iL_rmses);

    improved = '';
    if valLoss < bestValLoss
        bestValLoss = valLoss;
        bestModel = nss;
        improved = ' ** BEST **';
    end

    logMsg(logFile, sprintf('  %s done (%.1f min) | Val: %.6f | Vout=%.3fV iL=%.3fA%s', ...
        ph.name, phaseTime/60, valLoss, meanVout, meanIL, improved));
    for v = 1:numel(vout_rmses)
        logMsg(logFile, sprintf('  Val %d: Vout RMSE=%.3fV, iL RMSE=%.3fA', v, vout_rmses(v), iL_rmses(v)));
    end

    % Save checkpoint
    nssEst = nss; %#ok<NASGU>
    save(fullfile(checkpointDir, sprintf('nss_fixed_%s.mat', lower(ph.name))), 'nssEst', 'normStats');
    if ~isempty(bestModel)
        nssEst = bestModel; %#ok<NASGU>
        save(fullfile(checkpointDir, 'nss_fixed_best.mat'), 'nssEst', 'normStats');
    end
    logMsg(logFile, sprintf('  Checkpoint saved.'));
end

%% 8. Extended fine-tuning loop (FIX #5: 15-min checkpoints + plateau)
if isExtend || (~isTest && ~isempty(phases))
    remainingHours = maxHours - toc(tStart)/3600;

    if remainingHours > 0.1
        logMsg(logFile, '');
        logMsg(logFile, '============================================================');
        logMsg(logFile, sprintf('Extended fine-tuning (%.1f hours remaining)', remainingHours));
        logMsg(logFile, '============================================================');

        EPOCHS_PER_ROUND = 50;
        currentLR = 1e-4;
        LR_MIN = 1e-6;
        LR_PATIENCE = 4;
        noImproveRounds = 0;
        lrNoImproveRounds = 0;
        lastSaveTime = tic;
        totalRounds = 0;

        nssModel = bestModel;
        if isempty(nssModel), nssModel = nss; end

        while true
            elapsedHours = toc(tStart) / 3600;
            if elapsedHours >= maxHours
                logMsg(logFile, sprintf('\n** TIME LIMIT (%.1fh). Stopping. **', maxHours));
                break
            end
            if currentLR < LR_MIN * 1.01
                logMsg(logFile, sprintf('\n** LR at floor (%.2e). Stopping. **', currentLR));
                break
            end
            if noImproveRounds >= PLATEAU_ROUNDS
                logMsg(logFile, sprintf('\n** PLATEAU (%d rounds). Stopping. **', PLATEAU_ROUNDS));
                break
            end

            totalRounds = totalRounds + 1;

            opts = nssTrainingOptions('adam');
            opts.MaxEpochs = EPOCHS_PER_ROUND;
            opts.LearnRate = currentLR;
            opts.LossFcn = 'MeanSquaredError';
            opts.WindowSize = 4000;
            opts.InputInterSample = 'zoh';
            opts.PlotLossFcn = false;

            roundStart = tic;
            try
                nssModel = nlssest(trainData, nssModel, opts, ...
                    UseLastExperimentForValidation=true);
            catch ME
                logMsg(logFile, sprintf('  ERROR: %s', ME.message));
                nssModel = bestModel;
                noImproveRounds = noImproveRounds + 1;
                continue
            end
            roundTime = toc(roundStart);

            [valLoss, vout_rmses, iL_rmses] = computeValidation(nssModel, valDataCells, Y_val_raw, normStats, Ts);
            meanVout = mean(vout_rmses);
            meanIL = mean(iL_rmses);

            improved = '';
            if valLoss < bestValLoss * (1 - PLATEAU_REL_TOL)
                bestValLoss = valLoss;
                bestModel = nssModel;
                noImproveRounds = 0;
                lrNoImproveRounds = 0;
                improved = ' ** NEW BEST **';
            else
                noImproveRounds = noImproveRounds + 1;
                lrNoImproveRounds = lrNoImproveRounds + 1;
            end

            if lrNoImproveRounds >= LR_PATIENCE
                oldLR = currentLR;
                currentLR = max(currentLR * 0.5, LR_MIN);
                lrNoImproveRounds = 0;
                if currentLR < oldLR
                    logMsg(logFile, sprintf('  >> LR: %.2e -> %.2e', oldLR, currentLR));
                end
            end

            logMsg(logFile, sprintf('  Round %d (%.0fs) | Val: %.6f | Vout=%.3fV iL=%.3fA | LR=%.2e | noImprove=%d/%d%s', ...
                totalRounds, roundTime, valLoss, meanVout, meanIL, currentLR, noImproveRounds, PLATEAU_ROUNDS, improved));

            % Periodic checkpoint
            if toc(lastSaveTime)/60 >= SAVE_INTERVAL_MIN || contains(improved, 'NEW BEST')
                nssEst = nssModel; %#ok<NASGU>
                save(fullfile(checkpointDir, 'nss_fixed_latest.mat'), 'nssEst', 'normStats');
                nssEst = bestModel; %#ok<NASGU>
                save(fullfile(checkpointDir, 'nss_fixed_best.mat'), 'nssEst', 'normStats');
                lastSaveTime = tic;
                logMsg(logFile, sprintf('  >> Checkpoint saved at %s (best=%.6f)', ...
                    datestr(now, 'HH:MM:SS'), bestValLoss));
            end
        end
    end
end

%% 9. Final save
totalTime = toc(tStart) / 3600;
logMsg(logFile, '');
logMsg(logFile, '============================================================');
logMsg(logFile, 'Training complete');
logMsg(logFile, sprintf('  Duration:      %.2f hours', totalTime));
logMsg(logFile, sprintf('  Best val loss: %.6f', bestValLoss));
logMsg(logFile, '============================================================');

if ~isempty(bestModel)
    nssEst = bestModel; %#ok<NASGU>
else
    nssEst = nss; %#ok<NASGU>
end
save(fullfile(checkpointDir, 'nss_fixed_final.mat'), 'nssEst', 'normStats');
save(fullfile(modelsDir, 'boost_nss_fixed.mat'), 'nssEst', 'normStats');

% Final validation
logMsg(logFile, 'Final validation...');
model_to_val = bestModel;
if isempty(model_to_val), model_to_val = nss; end
[~, vout_rmses, iL_rmses] = computeValidation(model_to_val, valDataCells, Y_val_raw, normStats, Ts);
for v = 1:numel(vout_rmses)
    logMsg(logFile, sprintf('  Val %d: Vout RMSE=%.3fV, iL RMSE=%.3fA', v, vout_rmses(v), iL_rmses(v)));
end

% Comparison with old CT NSS
logMsg(logFile, '');
logMsg(logFile, '=== Comparison ===');
logMsg(logFile, sprintf('  Old CT NSS best val loss:  0.136884 (Vout~0.47V, iL~1.45A)'));
logMsg(logFile, sprintf('  New DT NSS best val loss:  %.6f', bestValLoss));
if bestValLoss < 0.136884
    logMsg(logFile, sprintf('  IMPROVEMENT: %.1f%%', (1 - bestValLoss/0.136884)*100));
else
    logMsg(logFile, '  No improvement yet — may need more training');
end

logMsg(logFile, 'Done.');
end


%% ---- Helpers ----

function logMsg(logFile, msg)
    fprintf('%s\n', msg);
    fid = fopen(logFile, 'a');
    fprintf(fid, '%s\n', msg);
    fclose(fid);
end

function [valLoss, vout_rmses, iL_rmses] = computeValidation(nssModel, valDataCells, Y_val_raw, normStats, Ts)
    nVal = numel(valDataCells);
    vout_rmses = zeros(nVal, 1);
    iL_rmses = zeros(nVal, 1);
    totalSE = 0;
    totalN = 0;
    for v = 1:nVal
        y_raw = [Y_val_raw{v}.Vout, Y_val_raw{v}.iL];
        x0_n = [(y_raw(1,1) - normStats.y_mean(1)) / normStats.y_std(1);
                (y_raw(1,2) - normStats.y_mean(2)) / normStats.y_std(2)];
        u_val = iddata([], valDataCells{v}.InputData, Ts);
        u_val.InputName = {'duty_n'};
        opt = simOptions('InitialCondition', x0_n);
        try
            yPred_n = sim(nssModel, u_val, opt);
            yPred_data = yPred_n.OutputData;
        catch
            vout_rmses(v) = 999; iL_rmses(v) = 999;
            totalSE = totalSE + 999^2 * numel(y_raw(:,1));
            totalN = totalN + numel(y_raw(:,1));
            continue
        end
        vout_pred = yPred_data(:,1) * normStats.y_std(1) + normStats.y_mean(1);
        iL_pred   = yPred_data(:,2) * normStats.y_std(2) + normStats.y_mean(2);
        vout_true = Y_val_raw{v}.Vout;
        iL_true   = Y_val_raw{v}.iL;
        T_val = min(numel(vout_pred), numel(vout_true));
        vout_err = vout_pred(1:T_val) - vout_true(1:T_val);
        iL_err = iL_pred(1:T_val) - iL_true(1:T_val);
        vout_rmses(v) = sqrt(mean(vout_err.^2));
        iL_rmses(v) = sqrt(mean(iL_err.^2));
        totalSE = totalSE + sum(vout_err.^2)/normStats.y_std(1)^2 + sum(iL_err.^2)/normStats.y_std(2)^2;
        totalN = totalN + 2 * T_val;
    end
    valLoss = totalSE / totalN;
end
