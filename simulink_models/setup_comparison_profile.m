% setup_comparison_profile.m
% Sets up THREE models with identical duty cycle profiles for interactive
% comparison in the Simulation Data Inspector:
%   1. boost_converter_test_harness  (Simscape open-loop reference)
%   2. boost_openloop_branch_c_predict  (Neural ODE ROM - Predict block)
%   3. boost_openloop_branch_c_layers   (Neural ODE ROM - Layer blocks)
%
% All three use the same weights (from extended_best.pt) and same input.
%
% Duty profile:
%   1. Staircase UP:   5% steps from D=0.10 to D=0.80 (15 steps, 10ms each)
%   2. Staircase DOWN: 5% steps from D=0.80 to D=0.10 (15 steps, 10ms each)
%   3. Big step UP:    D=0.10 -> 0.80 (hold 20ms)
%   4. Big step DOWN:  D=0.80 -> 0.10 (hold 20ms)
%
% Usage:
%   1. Run this script
%   2. All three models open with the duty profile loaded
%   3. Press Simulate on each model
%   4. Open Simulation Data Inspector to compare

thisDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(thisDir);
run(fullfile(repoRoot, 'startup.m'));

Ts = 5e-6;
modelsDir = fullfile(repoRoot, 'model_data');
simulinkDir = thisDir;  % Simulink models are in this directory

%% 1. Build duty cycle profile
fprintf('=== Building comparison duty profile ===\n');

D_min = 0.20;
D_max = 0.70;
D_step_size = 0.10;       % 10% steps
t_initial = 0.2;          % 0.2s initial settling at D_min
t_per_step = 0.1;         % 0.1s per step (steady state settling)

D_up = D_min:D_step_size:D_max;    % 0.10, 0.20, ..., 0.80
D_down = fliplr(D_up);              % 0.80, 0.70, ..., 0.10

% Build time-duty pairs as ZOH steps (duplicate points at transitions)
t = 0;
time_vec = [];
duty_vec = [];

% Phase 1: Start at D_min, hold 0.2s
fprintf('  Initial hold:   D=%.2f for %.1fs\n', D_min, t_initial);
time_vec(end+1) = t;
duty_vec(end+1) = D_min;
t = t + t_initial;

% Phase 2: Staircase up (10% steps)
fprintf('  Staircase UP:   D=%.2f -> %.2f (%d steps of %.0f%%)\n', ...
    D_up(2), D_max, numel(D_up)-1, D_step_size*100);
for i = 2:numel(D_up)
    % Hold old value right before transition
    time_vec(end+1) = t - 1e-9; %#ok<AGROW>
    duty_vec(end+1) = D_up(i-1); %#ok<AGROW>
    % Step to new value
    time_vec(end+1) = t; %#ok<AGROW>
    duty_vec(end+1) = D_up(i); %#ok<AGROW>
    t = t + t_per_step;
end

% Phase 3: Staircase down (10% steps)
fprintf('  Staircase DOWN: D=%.2f -> %.2f (%d steps of %.0f%%)\n', ...
    D_down(2), D_min, numel(D_down)-1, D_step_size*100);
for i = 2:numel(D_down)
    time_vec(end+1) = t - 1e-9; %#ok<AGROW>
    duty_vec(end+1) = D_down(i-1); %#ok<AGROW>
    time_vec(end+1) = t; %#ok<AGROW>
    duty_vec(end+1) = D_down(i); %#ok<AGROW>
    t = t + t_per_step;
end

% Phase 4: Big transient up (D_min -> D_max)
fprintf('  Big step UP:    D=%.2f -> %.2f, hold %.1fs\n', D_min, D_max, t_per_step);
time_vec(end+1) = t - 1e-9;
duty_vec(end+1) = D_min;
time_vec(end+1) = t;
duty_vec(end+1) = D_max;
t = t + t_per_step;

% Phase 5: Big transient down (D_max -> D_min)
fprintf('  Big step DOWN:  D=%.2f -> %.2f, hold %.1fs\n', D_max, D_min, t_per_step);
time_vec(end+1) = t - 1e-9;
duty_vec(end+1) = D_max;
time_vec(end+1) = t;
duty_vec(end+1) = D_min;
t = t + t_per_step;

% Final point
time_vec(end+1) = t;
duty_vec(end+1) = D_min;

time_vec = time_vec(:);
duty_vec = duty_vec(:);

stopTime = t;
fprintf('  Total sim time: %.0f ms\n', stopTime * 1e3);

% Create timeseries for From Workspace block
duty_input = timeseries(duty_vec, time_vec);
duty_input.Name = 'duty';

% Save to workspace and file
assignin('base', 'duty_input', duty_input);
save(fullfile(modelsDir, 'comparison_duty_profile.mat'), ...
    'duty_input', 'stopTime');
fprintf('  Saved: comparison_duty_profile.mat\n');

%% 2. Common ROM initial conditions
% At D_min, Vin=5V: steady-state values
Vin = 5; R_load = 20;
Vout_init = Vin / (1 - D_min);
iL_init = Vout_init / (R_load * (1 - D_min));
node_state_ic = [Vout_init; iL_init];
assignin('base', 'node_state_ic', node_state_ic);
fprintf('\n  ROM IC: Vout=%.2fV, iL=%.2fA (for D=%.2f)\n', Vout_init, iL_init, D_min);

%% 3. Configure Simscape model
fprintf('\n=== Configuring Simscape model ===\n');
simscapeModel = 'boost_converter_test_harness';
simscapeFile = fullfile(simulinkDir, [simscapeModel '.slx']);

if bdIsLoaded(simscapeModel), close_system(simscapeModel, 0); end
load_system(simscapeFile);

% Set stop time
set_param(simscapeModel, 'StopTime', num2str(stopTime));

% Keep Vin and Rload constant (no steps during test)
set_param([simscapeModel '/Vin Value'], 'Before', '5', 'After', '5', 'Time', '999');
set_param([simscapeModel '/Rload Value'], 'Before', '6', 'After', '6', 'Time', '999');

% Enable signal logging
set_param(simscapeModel, 'SignalLogging', 'on');

% Log Vout output
% NOTE: Do NOT remove the existing 'duty_open' logged signal the user added manually.
ph = get_param([simscapeModel '/Out_Vout'], 'PortHandles');
lineH = get_param(ph.Inport(1), 'Line');
srcPort = get_param(lineH, 'SrcPortHandle');
set_param(srcPort, 'DataLogging', 'on', ...
    'DataLoggingName', 'Vout_Simscape', 'DataLoggingNameMode', 'Custom');

fprintf('  %s configured: StopTime=%.3fs, Vin=5V, Rload=6ohm\n', simscapeModel, stopTime);

%% 4. Configure ROM Predict model
fprintf('\n=== Configuring ROM Predict model ===\n');
romPredict = 'boost_openloop_branch_c_predict';
configureROMModel(romPredict, simulinkDir, stopTime);

%% 5. Configure ROM Layers model
fprintf('\n=== Configuring ROM Layers model ===\n');
romLayers = 'boost_openloop_branch_c_layers';
configureROMModel(romLayers, simulinkDir, stopTime);

%% 6. Open all three models
fprintf('\n=== Opening models for interactive simulation ===\n');
open_system(simscapeModel);
open_system(romPredict);
open_system(romLayers);

% Open SDI
fprintf('  Opening Simulation Data Inspector...\n');
Simulink.sdi.view;

fprintf('\n=== Ready! ===\n');
fprintf('  1. Simulate "%s" (Simscape reference)\n', simscapeModel);
fprintf('  2. Simulate "%s" (Neural ODE - Predict)\n', romPredict);
fprintf('  3. Simulate "%s" (Neural ODE - Layers)\n', romLayers);
fprintf('  4. Compare all signals in the Simulation Data Inspector\n');
fprintf('  Duty profile: staircase up/down (5%% steps) + big transients\n');


%% ========================================================================
%% Helper: configure a ROM model with From Workspace duty input
%% ========================================================================
function configureROMModel(modelName, modelsDir, stopTime)
    modelFile = fullfile(modelsDir, [modelName '.slx']);

    if bdIsLoaded(modelName), close_system(modelName, 0); end
    load_system(modelFile);

    % Check if Step block exists (needs replacing with From Workspace)
    stepBlks = find_system(modelName, 'SearchDepth', 1, 'BlockType', 'Step');
    fwsBlks = find_system(modelName, 'SearchDepth', 1, 'BlockType', 'FromWorkspace');

    if ~isempty(stepBlks)
        % Replace Step with From Workspace
        stepBlk = stepBlks{1};
        stepPos = get_param(stepBlk, 'Position');
        stepPH = get_param(stepBlk, 'PortHandles');

        % Find destinations
        lineH = get_param(stepPH.Outport(1), 'Line');
        dstPorts = get_param(lineH, 'DstPortHandle');

        % Delete connections
        for k = 1:numel(dstPorts)
            delete_line(modelName, stepPH.Outport(1), dstPorts(k));
        end
        delete_block(stepBlk);

        % Add From Workspace
        fwsBlk = [modelName '/FromWS_duty'];
        add_block('simulink/Sources/From Workspace', fwsBlk, ...
            'Position', stepPos, ...
            'VariableName', 'duty_input', ...
            'SampleTime', '0', ...
            'OutputAfterFinalValue', 'Holding final value');

        % Reconnect
        fwsPH = get_param(fwsBlk, 'PortHandles');
        for k = 1:numel(dstPorts)
            add_line(modelName, fwsPH.Outport(1), dstPorts(k), 'autorouting', 'smart');
        end

        fprintf('  Replaced Step with From Workspace\n');

    elseif ~isempty(fwsBlks)
        % Already has From Workspace — just update variable name
        set_param(fwsBlks{1}, 'VariableName', 'duty_input');
        fprintf('  From Workspace already exists, updated variable name\n');
    else
        error('No Step or FromWorkspace input block found in %s', modelName);
    end

    % Set stop time
    set_param(modelName, 'StopTime', num2str(stopTime));

    % Enable signal logging
    set_param(modelName, 'SignalLogging', 'on');

    % Save
    save_system(modelName, modelFile);
    fprintf('  %s configured: StopTime=%.3fs\n', modelName, stopTime);
end
