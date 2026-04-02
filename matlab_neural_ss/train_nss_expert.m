% train_nss_expert.m
% Expert suggestion: set NumWindowFraction=eps so nlssest updates network
% parameters after EVERY single window (stochastic, like PyTorch) instead
% of accumulating all windows first (batch, the default).
%
% Base: train_neural_ss_normalized.m (CT NSS, output-scaled last layer)
% Change: opts.NumWindowFraction = eps added to all phases
%
% Phase 2 and 3 run in 25-epoch chunks so val loss is visible every ~2hrs.
% Early stopping within a phase if no improvement for 100 epochs (4 chunks).
%
% Usage:
%   train_nss_expert            % full run from scratch
%   train_nss_expert('resume')  % resume from best checkpoint
%   train_nss_expert('test')    % quick test (3 profiles, 25 epochs)

function train_nss_expert(mode, maxHours)
if nargin < 1, mode = 'full'; end
if nargin < 2, maxHours = Inf; end

isTest   = strcmp(mode, 'test');
isResume = strcmp(mode, 'resume');

SAVE_INTERVAL_MIN  = 15;
CHUNK_EPOCHS       = 25;   % validate every N epochs within a phase
EARLY_STOP_EPOCHS  = 100;  % stop phase if no improvement for this many epochs
PLATEAU_ROUNDS     = 12;   % fine-tune loop: stop after N rounds no improvement
PLATEAU_REL_TOL    = 1e-4;
LR_MIN             = 1e-6;
LR_PATIENCE        = 4;

Ts = 5e-6;
nx = 2;

baseDir       = fileparts(mfilename('fullpath'));
repoRoot      = fileparts(baseDir);
dataDir       = fullfile(repoRoot, 'data');
modelDataDir  = fullfile(repoRoot, 'model_data');
checkpointDir = fullfile(baseDir, 'checkpoints_expert');
if ~exist(checkpointDir, 'dir'), mkdir(checkpointDir); end

logFile = fullfile(baseDir, 'training_log_expert.txt');

logMsg(logFile, '============================================================');
logMsg(logFile, 'Expert NSS Training: NumWindowFraction=eps');
logMsg(logFile, sprintf('  Mode:            %s', mode));
logMsg(logFile, sprintf('  Key change:      NumWindowFraction=eps (update after every window)'));
logMsg(logFile, sprintf('  Chunk size:      %d epochs (val loss printed each chunk)', CHUNK_EPOCHS));
logMsg(logFile, sprintf('  Early stop:      %d epochs no improvement within phase', EARLY_STOP_EPOCHS));
logMsg(logFile, sprintf('  Started:         %s', datestr(now)));
logMsg(logFile, '============================================================');

%% 1. Load training data
logMsg(logFile, 'Loading training data...');
load(fullfile(dataDir, 'boost_nss_training_data.mat'), 'allU', 'allY');

nTotal     = numel(allU);
nVal       = 3;
nTrainFull = nTotal - nVal;
if isTest
    nTrain = min(3, nTrainFull);
else
    nTrain = nTrainFull;
end

%% 2. Normalization stats
allDuty = []; allVout = []; allIL = []; allDvout = []; allDil = [];
for i = 1:nTrain
    allDuty  = [allDuty;  allU{i}.duty];          %#ok<AGROW>
    allVout  = [allVout;  allY{i}.Vout];           %#ok<AGROW>
    allIL    = [allIL;    allY{i}.iL];             %#ok<AGROW>
    allDvout = [allDvout; diff(allY{i}.Vout)/Ts]; %#ok<AGROW>
    allDil   = [allDil;   diff(allY{i}.iL)/Ts];   %#ok<AGROW>
end

normStats.u_mean = mean(allDuty);
normStats.u_std  = std(allDuty);
normStats.y_mean = [mean(allVout), mean(allIL)];
normStats.y_std  = [std(allVout),  std(allIL)];

x_std        = [std(allVout); std(allIL)];
dxdt_std     = [std(allDvout); std(allDil)];
output_scale = dxdt_std ./ x_std;

logMsg(logFile, sprintf('  duty:  mean=%.3f, std=%.3f', normStats.u_mean, normStats.u_std));
logMsg(logFile, sprintf('  Vout:  mean=%.2f, std=%.2f', normStats.y_mean(1), normStats.y_std(1)));
logMsg(logFile, sprintf('  iL:    mean=%.2f, std=%.2f', normStats.y_mean(2), normStats.y_std(2)));
logMsg(logFile, sprintf('  output_scale: [%.1f, %.1f]', output_scale));

%% 3. Normalize data
data_norm = cell(nTotal, 1);
for i = 1:nTotal
    duty_n = (allU{i}.duty - normStats.u_mean) / normStats.u_std;
    vout_n = (allY{i}.Vout - normStats.y_mean(1)) / normStats.y_std(1);
    iL_n   = (allY{i}.iL   - normStats.y_mean(2)) / normStats.y_std(2);
    data_norm{i} = iddata([vout_n, iL_n], duty_n, Ts);
end

trainData = merge(data_norm{1:nTrain});
trainData.InputName  = {'duty_n'};
trainData.OutputName = {'Vout_n', 'iL_n'};

valDataCells = cell(nVal, 1);
Y_val_raw    = allY(nTrainFull+1:nTotal);
for v = 1:nVal
    valDataCells{v} = data_norm{nTrainFull + v};
    valDataCells{v}.InputName  = {'duty_n'};
    valDataCells{v}.OutputName = {'Vout_n', 'iL_n'};
end
logMsg(logFile, sprintf('  Train: %d profiles, Val: %d profiles', nTrain, nVal));

%% 4. ODE solver options
odeSolverOpts = nssTrainingOptions('adam').ODESolverOptions;
odeSolverOpts.MaxStepSize = 50e-6;

%% 5. Create or load NSS model
bestValLoss = Inf;
bestModel   = [];
tStart      = tic;

% Determine which phases to run
if isResume
    % Try to load best checkpoint
    cpBest  = fullfile(checkpointDir, 'nss_expert_best.mat');
    cpPh2   = fullfile(checkpointDir, 'nss_expert_phase2_20ms.mat');
    cpPh1   = fullfile(checkpointDir, 'nss_expert_phase1_5ms.mat');

    if isfile(cpBest)
        logMsg(logFile, sprintf('Resuming from best checkpoint: %s', cpBest));
        cp = load(cpBest); nss = cp.nssEst; normStats = cp.normStats;
    elseif isfile(cpPh2)
        logMsg(logFile, sprintf('Resuming from Phase 2 checkpoint: %s', cpPh2));
        cp = load(cpPh2); nss = cp.nssEst; normStats = cp.normStats;
    elseif isfile(cpPh1)
        logMsg(logFile, sprintf('Resuming from Phase 1 checkpoint: %s', cpPh1));
        cp = load(cpPh1); nss = cp.nssEst; normStats = cp.normStats;
    else
        logMsg(logFile, 'No checkpoint found — starting from scratch.');
        isResume = false;
    end

    if isResume
        [bestValLoss, vout_rmses, iL_rmses] = computeValidation(nss, valDataCells, Y_val_raw, normStats, Ts);
        logMsg(logFile, sprintf('  Loaded model val loss: %.6f', bestValLoss));
        for v = 1:numel(vout_rmses)
            logMsg(logFile, sprintf('  Val %d: Vout RMSE=%.3fV, iL RMSE=%.3fA', v, vout_rmses(v), iL_rmses(v)));
        end
        bestModel = nss;
        % Skip to fine-tuning — phases already done
        skipPhases = true;
    end
end

if ~isResume
    skipPhases = false;
    % Build CT NSS with output-scaled last layer
    logMsg(logFile, '');
    logMsg(logFile, '=== Creating CT NSS with output-scaled last layer ===');
    nss = idNeuralStateSpace(nx, NumInputs=1, NumOutputs=2, Ts=0);
    nss.InputName  = {'duty_n'};
    nss.OutputName = {'Vout_n', 'iL_n'};
    nss.StateName  = {'x1_n', 'x2_n'};

    stateNet = createMLPNetwork(nss, 'state', ...
        LayerSizes=[64 64], Activations='tanh', WeightsInitializer='glorot');

    for k = 1:size(stateNet.Learnables, 1)
        if strcmp(stateNet.Learnables.Layer{k}, 'dxdt')
            if contains(stateNet.Learnables.Parameter{k}, 'Weights')
                W = extractdata(stateNet.Learnables.Value{k});
                W(1,:) = W(1,:) * output_scale(1);
                W(2,:) = W(2,:) * output_scale(2);
                stateNet.Learnables.Value{k} = dlarray(W);
            elseif contains(stateNet.Learnables.Parameter{k}, 'Bias')
                b = extractdata(stateNet.Learnables.Value{k});
                b(1) = b(1) * output_scale(1);
                b(2) = b(2) * output_scale(2);
                stateNet.Learnables.Value{k} = dlarray(b);
            end
        end
    end
    nss.StateNetwork = stateNet;
    nParams = sum(cellfun(@numel, nss.StateNetwork.Learnables.Value));
    logMsg(logFile, sprintf('  CT NSS: nx=%d, params=%d, output_scale=[%.1f,%.1f]', nx, nParams, output_scale));
end

%% 6. Phase 1 — run as single block (300 epochs, 5ms windows)
%    Only if not resuming past it
ph1CpFile = fullfile(checkpointDir, 'nss_expert_phase1_5ms.mat');

if ~skipPhases && ~isfile(ph1CpFile)
    if isTest
        ph1Epochs = 25;
    else
        ph1Epochs = 300;
    end
    logMsg(logFile, '');
    logMsg(logFile, sprintf('--- Phase1_5ms: %d epochs, window=1000 (5ms), LR=1e-3, NumWindowFraction=eps ---', ph1Epochs));

    opts = nssTrainingOptions('adam');
    opts.MaxEpochs           = ph1Epochs;
    opts.LearnRate           = 1e-3;
    opts.LearnRateSchedule   = 'piecewise';
    opts.LearnRateDropPeriod = 100;
    opts.LearnRateDropFactor = 0.5;
    opts.LossFcn             = 'MeanSquaredError';
    opts.WindowSize          = 1000;
    opts.InputInterSample    = 'zoh';
    opts.ODESolverOptions    = odeSolverOpts;
    opts.PlotLossFcn         = false;
    opts.NumWindowFraction   = eps;

    ph1Start = tic;
    nss = nlssest(trainData, nss, opts, UseLastExperimentForValidation=true);
    ph1Time = toc(ph1Start);

    [valLoss, vout_rmses, iL_rmses] = computeValidation(nss, valDataCells, Y_val_raw, normStats, Ts);
    if valLoss < bestValLoss
        bestValLoss = valLoss; bestModel = nss;
    end
    logMsg(logFile, sprintf('  Phase1 done (%.1f min) | Val: %.6f | Vout=%.3fV iL=%.3fA', ...
        ph1Time/60, valLoss, mean(vout_rmses), mean(iL_rmses)));
    for v = 1:numel(vout_rmses)
        logMsg(logFile, sprintf('  Val %d: Vout RMSE=%.3fV, iL RMSE=%.3fA', v, vout_rmses(v), iL_rmses(v)));
    end

    nssEst = nss; %#ok<NASGU>
    save(ph1CpFile, 'nssEst', 'normStats');
    nssEst = bestModel; %#ok<NASGU>
    save(fullfile(checkpointDir, 'nss_expert_best.mat'), 'nssEst', 'normStats');
    logMsg(logFile, '  Phase 1 checkpoint saved.');

elseif ~skipPhases && isfile(ph1CpFile)
    logMsg(logFile, 'Phase 1 checkpoint found — loading and skipping to Phase 2.');
    cp = load(ph1CpFile); nss = cp.nssEst;
    [bestValLoss, vout_rmses, iL_rmses] = computeValidation(nss, valDataCells, Y_val_raw, normStats, Ts);
    bestModel = nss;
    logMsg(logFile, sprintf('  Phase 1 val loss: %.6f', bestValLoss));
    for v = 1:numel(vout_rmses)
        logMsg(logFile, sprintf('  Val %d: Vout RMSE=%.3fV, iL RMSE=%.3fA', v, vout_rmses(v), iL_rmses(v)));
    end
end

%% 7. Chunked phases (Phase 2 and 3) with per-chunk validation
if ~skipPhases
    if isTest
        chunkedPhases = {
            struct('name','Phase2_20ms',    'totalEpochs',50,  'window',4000, 'lr',5e-4, ...
                   'lr_drop_period',150, 'lr_drop_factor',0.5)
            struct('name','Phase3_Finetune','totalEpochs',25,  'window',4000, 'lr',1e-4, ...
                   'lr_drop_period',100, 'lr_drop_factor',0.5)
        };
    else
        chunkedPhases = {
            struct('name','Phase2_20ms',    'totalEpochs',500, 'window',4000, 'lr',5e-4, ...
                   'lr_drop_period',150, 'lr_drop_factor',0.5)
            struct('name','Phase3_Finetune','totalEpochs',200, 'window',4000, 'lr',1e-4, ...
                   'lr_drop_period',100, 'lr_drop_factor',0.5)
        };
    end

    lastSaveTime = tic;

    for p = 1:numel(chunkedPhases)
        ph = chunkedPhases{p};
        totalEpochs   = ph.totalEpochs;
        nChunks       = ceil(totalEpochs / CHUNK_EPOCHS);
        epochsDone    = 0;
        noImprovEpochs = 0;
        phaseBestLoss = bestValLoss;

        logMsg(logFile, '');
        logMsg(logFile, sprintf('--- %s: %d epochs in chunks of %d, window=%d (%.0fms), LR=%.0e ---', ...
            ph.name, totalEpochs, CHUNK_EPOCHS, ph.window, ph.window*Ts*1e3, ph.lr));
        logMsg(logFile, sprintf('    Early stop if no improvement for %d epochs (%d chunks)', ...
            EARLY_STOP_EPOCHS, ceil(EARLY_STOP_EPOCHS/CHUNK_EPOCHS)));

        phaseStart = tic;

        for c = 1:nChunks
            epochsThisChunk = min(CHUNK_EPOCHS, totalEpochs - epochsDone);
            if epochsThisChunk <= 0, break; end

            % Scale LR: compute effective LR at this point in the schedule
            % (piecewise drop applied per full phase — approximate by using
            %  current LR directly; drop factor applied based on epochs done)
            dropFactor = ph.lr_drop_factor ^ floor(epochsDone / ph.lr_drop_period);
            currentLR  = ph.lr * dropFactor;

            opts = nssTrainingOptions('adam');
            opts.MaxEpochs         = epochsThisChunk;
            opts.LearnRate         = currentLR;
            opts.LossFcn           = 'MeanSquaredError';
            opts.WindowSize        = ph.window;
            opts.InputInterSample  = 'zoh';
            opts.ODESolverOptions  = odeSolverOpts;
            opts.PlotLossFcn       = false;
            opts.NumWindowFraction = eps;

            chunkStart = tic;
            try
                nss = nlssest(trainData, nss, opts, UseLastExperimentForValidation=true);
            catch ME
                logMsg(logFile, sprintf('  ERROR chunk %d: %s', c, ME.message));
                break
            end
            chunkTime  = toc(chunkStart);
            epochsDone = epochsDone + epochsThisChunk;

            [valLoss, vout_rmses, iL_rmses] = computeValidation(nss, valDataCells, Y_val_raw, normStats, Ts);

            improved = '';
            if valLoss < bestValLoss * (1 - PLATEAU_REL_TOL)
                bestValLoss    = valLoss;
                bestModel      = nss;
                phaseBestLoss  = valLoss;
                noImprovEpochs = 0;
                improved       = ' ** BEST **';
            else
                noImprovEpochs = noImprovEpochs + epochsThisChunk;
            end

            logMsg(logFile, sprintf('  [%s] ep %d/%d (%.0fs) | Val: %.6f | Vout=%.3fV iL=%.3fA | LR=%.2e | noImprove=%d/%d%s', ...
                ph.name, epochsDone, totalEpochs, chunkTime, valLoss, ...
                mean(vout_rmses), mean(iL_rmses), currentLR, noImprovEpochs, EARLY_STOP_EPOCHS, improved));

            % 15-min checkpoint
            if toc(lastSaveTime)/60 >= SAVE_INTERVAL_MIN || ~isempty(improved)
                nssEst = nss; %#ok<NASGU>
                save(fullfile(checkpointDir, 'nss_expert_latest.mat'), 'nssEst', 'normStats');
                if ~isempty(bestModel)
                    nssEst = bestModel; %#ok<NASGU>
                    save(fullfile(checkpointDir, 'nss_expert_best.mat'), 'nssEst', 'normStats');
                end
                lastSaveTime = tic;
                logMsg(logFile, sprintf('  >> Checkpoint saved at %s (best=%.6f)', ...
                    datestr(now,'HH:MM:SS'), bestValLoss));
            end

            % Early stop within phase
            if noImprovEpochs >= EARLY_STOP_EPOCHS
                logMsg(logFile, sprintf('  >> EARLY STOP: no improvement for %d epochs in %s', ...
                    EARLY_STOP_EPOCHS, ph.name));
                break
            end

            % Time limit
            if toc(tStart)/3600 >= maxHours
                logMsg(logFile, sprintf('\n** TIME LIMIT (%.1fh). Stopping. **', maxHours));
                break
            end
        end

        phaseTime = toc(phaseStart);
        logMsg(logFile, sprintf('  %s complete (%.1f min) | Best val: %.6f', ...
            ph.name, phaseTime/60, bestValLoss));

        % Save phase checkpoint
        nssEst = nss; %#ok<NASGU>
        save(fullfile(checkpointDir, sprintf('nss_expert_%s.mat', lower(ph.name))), 'nssEst', 'normStats');
        logMsg(logFile, sprintf('  Phase checkpoint saved.'));

        if toc(tStart)/3600 >= maxHours, break; end
    end
end

%% 8. Extended fine-tuning loop (50-epoch rounds, plateau detection)
if toc(tStart)/3600 < maxHours
    logMsg(logFile, '');
    logMsg(logFile, '============================================================');
    logMsg(logFile, 'Extended fine-tuning (plateau detection)');
    logMsg(logFile, '============================================================');

    FINETUNE_CHUNK   = 25;
    currentLR        = 1e-4;
    noImproveRounds  = 0;
    lrNoImproveRounds = 0;
    lastSaveTime     = tic;
    totalRounds      = 0;

    nssModel = bestModel;
    if isempty(nssModel), nssModel = nss; end

    while true
        if toc(tStart)/3600 >= maxHours
            logMsg(logFile, sprintf('\n** TIME LIMIT (%.1fh). Stopping. **', maxHours)); break
        end
        if currentLR < LR_MIN * 1.01
            logMsg(logFile, sprintf('\n** LR at floor (%.2e). Stopping. **', currentLR)); break
        end
        if noImproveRounds >= PLATEAU_ROUNDS
            logMsg(logFile, sprintf('\n** PLATEAU (%d rounds). Stopping. **', PLATEAU_ROUNDS)); break
        end

        totalRounds = totalRounds + 1;

        opts = nssTrainingOptions('adam');
        opts.MaxEpochs         = FINETUNE_CHUNK;
        opts.LearnRate         = currentLR;
        opts.LossFcn           = 'MeanSquaredError';
        opts.WindowSize        = 4000;
        opts.InputInterSample  = 'zoh';
        opts.ODESolverOptions  = odeSolverOpts;
        opts.PlotLossFcn       = false;
        opts.NumWindowFraction = eps;

        roundStart = tic;
        try
            nssModel = nlssest(trainData, nssModel, opts, UseLastExperimentForValidation=true);
        catch ME
            logMsg(logFile, sprintf('  ERROR round %d: %s', totalRounds, ME.message));
            nssModel = bestModel; noImproveRounds = noImproveRounds + 1; continue
        end
        roundTime = toc(roundStart);

        [valLoss, vout_rmses, iL_rmses] = computeValidation(nssModel, valDataCells, Y_val_raw, normStats, Ts);

        improved = '';
        if valLoss < bestValLoss * (1 - PLATEAU_REL_TOL)
            bestValLoss       = valLoss; bestModel = nssModel;
            noImproveRounds   = 0; lrNoImproveRounds = 0;
            improved          = ' ** NEW BEST **';
        else
            noImproveRounds   = noImproveRounds + 1;
            lrNoImproveRounds = lrNoImproveRounds + 1;
        end

        if lrNoImproveRounds >= LR_PATIENCE
            oldLR = currentLR; currentLR = max(currentLR * 0.5, LR_MIN);
            lrNoImproveRounds = 0;
            if currentLR < oldLR
                logMsg(logFile, sprintf('  >> LR: %.2e -> %.2e', oldLR, currentLR));
            end
        end

        logMsg(logFile, sprintf('  FT round %d (%.0fs) | Val: %.6f | Vout=%.3fV iL=%.3fA | LR=%.2e | noImprove=%d/%d%s', ...
            totalRounds, roundTime, valLoss, mean(vout_rmses), mean(iL_rmses), ...
            currentLR, noImproveRounds, PLATEAU_ROUNDS, improved));

        if toc(lastSaveTime)/60 >= SAVE_INTERVAL_MIN || ~isempty(improved)
            nssEst = nssModel; %#ok<NASGU>
            save(fullfile(checkpointDir, 'nss_expert_latest.mat'), 'nssEst', 'normStats');
            if ~isempty(bestModel)
                nssEst = bestModel; %#ok<NASGU>
                save(fullfile(checkpointDir, 'nss_expert_best.mat'), 'nssEst', 'normStats');
            end
            lastSaveTime = tic;
            logMsg(logFile, sprintf('  >> Checkpoint saved at %s (best=%.6f)', ...
                datestr(now,'HH:MM:SS'), bestValLoss));
        end
    end
end

%% 9. Final save
totalTime = toc(tStart) / 3600;
logMsg(logFile, '');
logMsg(logFile, '============================================================');
logMsg(logFile, sprintf('Training complete at %s', datestr(now)));
logMsg(logFile, sprintf('  Duration:      %.2f hours', totalTime));
logMsg(logFile, sprintf('  Best val loss: %.6f', bestValLoss));
logMsg(logFile, '============================================================');

nssEst = bestModel; if isempty(nssEst), nssEst = nss; end %#ok<NASGU>
save(fullfile(checkpointDir, 'nss_expert_final.mat'), 'nssEst', 'normStats');
save(fullfile(modelDataDir,  'boost_nss_expert.mat'),  'nssEst', 'normStats');

logMsg(logFile, 'Final validation...');
[~, vout_rmses, iL_rmses] = computeValidation(bestModel, valDataCells, Y_val_raw, normStats, Ts);
for v = 1:numel(vout_rmses)
    logMsg(logFile, sprintf('  Val %d: Vout RMSE=%.3fV, iL RMSE=%.3fA', v, vout_rmses(v), iL_rmses(v)));
end
logMsg(logFile, '');
logMsg(logFile, '=== Comparison ===');
logMsg(logFile, sprintf('  CT NSS baseline:   val loss 0.137'));
logMsg(logFile, sprintf('  Expert (this run): val loss %.6f', bestValLoss));
logMsg(logFile, sprintf('  PyTorch Neural ODE: val loss 0.040'));
if bestValLoss < 0.137
    logMsg(logFile, sprintf('  IMPROVEMENT: %.1f%% vs CT NSS baseline', (1-bestValLoss/0.137)*100));
else
    logMsg(logFile, '  No improvement vs CT NSS baseline');
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
    nVal       = numel(valDataCells);
    vout_rmses = zeros(nVal, 1);
    iL_rmses   = zeros(nVal, 1);
    totalSE    = 0; totalN = 0;
    for v = 1:nVal
        y_raw = [Y_val_raw{v}.Vout, Y_val_raw{v}.iL];
        x0_n  = [(y_raw(1,1) - normStats.y_mean(1)) / normStats.y_std(1);
                 (y_raw(1,2) - normStats.y_mean(2)) / normStats.y_std(2)];
        u_val = iddata([], valDataCells{v}.InputData, Ts);
        u_val.InputName = {'duty_n'};
        opt = simOptions('InitialCondition', x0_n);
        try
            yPred_n    = sim(nssModel, u_val, opt);
            yPred_data = yPred_n.OutputData;
        catch
            vout_rmses(v) = 999; iL_rmses(v) = 999;
            totalSE = totalSE + 999^2 * numel(y_raw(:,1));
            totalN  = totalN  + numel(y_raw(:,1));
            continue
        end
        vout_pred = yPred_data(:,1) * normStats.y_std(1) + normStats.y_mean(1);
        iL_pred   = yPred_data(:,2) * normStats.y_std(2) + normStats.y_mean(2);
        vout_true = Y_val_raw{v}.Vout;
        iL_true   = Y_val_raw{v}.iL;
        T_val     = min(numel(vout_pred), numel(vout_true));
        vout_err  = vout_pred(1:T_val) - vout_true(1:T_val);
        iL_err    = iL_pred(1:T_val)   - iL_true(1:T_val);
        vout_rmses(v) = sqrt(mean(vout_err.^2));
        iL_rmses(v)   = sqrt(mean(iL_err.^2));
        totalSE = totalSE + sum(vout_err.^2)/normStats.y_std(1)^2 + sum(iL_err.^2)/normStats.y_std(2)^2;
        totalN  = totalN  + 2 * T_val;
    end
    valLoss = totalSE / totalN;
end
