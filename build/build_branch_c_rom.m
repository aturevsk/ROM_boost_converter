% build_branch_c_rom.m
% Branch C: Builds TWO open-loop Simulink models for Neural ODE ROM.
%
% Model A: boost_openloop_branch_c_predict
%   Uses Predict block (dlnetwork) with codegen acceleration (SimulateUsing='Code generation')
%
% Model B: boost_openloop_branch_c_layers
%   Uses exportNetworkToSimulink layer blocks
%
% Both models share the same architecture:
%   duty (step) -> [NeuralODE_ROM subsystem] -> [Vout; iL]
%
% Inside NeuralODE_ROM:
%   [Normalize(x,u)] -> [MLP] -> [Scale by dxdt_scale] -> [Euler] -> x_state
%                          ^                                             |
%                          └──── state feedback (Unit Delay) ────────────┘
addpath(genpath(fileparts(fileparts(mfilename('fullpath')))));

Ts = 5e-6;
nx = 2;
H = 64;
modelsDir = fullfile(fileparts(mfilename('fullpath')), 'models');

%% 1. Import PyTorch model
fprintf('=== Branch C ROM (Neural ODE) ===\n');

ptFile = fullfile(modelsDir, 'neuralode_mlp_traced.pt');
if ~exist(ptFile, 'file')
    error('Run Python export first to create %s', ptFile);
end

% Import TorchScript to get weights, then build proper dlnetwork
fprintf('\nImporting TorchScript model...\n');
pt_net = importNetworkFromPyTorch(ptFile);
dlX = dlarray(single(zeros(3, 1)), 'CB');
pt_net = initialize(pt_net, dlX);

% Extract weights from imported network
ptWeights = pt_net.Learnables.Value;
fprintf('  Imported %d learnable arrays\n', numel(ptWeights));

% Build proper dlnetwork with featureInputLayer (required for exportNetworkToSimulink)
layers = [
    featureInputLayer(3, 'Name', 'input', 'Normalization', 'none')
    fullyConnectedLayer(H, 'Name', 'fc1')
    tanhLayer('Name', 'tanh1')
    fullyConnectedLayer(H, 'Name', 'fc2')
    tanhLayer('Name', 'tanh2')
    fullyConnectedLayer(nx, 'Name', 'fc_out')
];
node_net = dlnetwork(layers);

% Load weights from .mat (reliable — matches PyTorch exactly)
matFile = fullfile(modelsDir, 'neuralode_weights.mat');
W = load(matFile);
% Check expected sizes and match
for li = 1:numel(node_net.Learnables.Value)
    fprintf('  Learnable %d: %s\n', li, mat2str(size(extractdata(node_net.Learnables.Value{li}))));
end
node_net.Learnables.Value{1} = dlarray(single(W.W1));        % fc1 weights (64x3)
node_net.Learnables.Value{2} = dlarray(single(W.b1(:)));     % fc1 bias (64x1)
node_net.Learnables.Value{3} = dlarray(single(W.W2));        % fc2 weights (64x64)
node_net.Learnables.Value{4} = dlarray(single(W.b2(:)));     % fc2 bias (64x1)
node_net.Learnables.Value{5} = dlarray(single(W.W3));        % fc_out weights (2x64)
node_net.Learnables.Value{6} = dlarray(single(W.b3(:)));     % fc_out bias (2x1)

fprintf('  Built dlnetwork: %d layers, %d parameters\n', ...
    numel(node_net.Layers), sum(cellfun(@numel, node_net.Learnables.Value)));

% Verify matches PyTorch
x_test = dlarray(single([0; 0; 0]), 'CB');
y_test = predict(node_net, x_test);
y_pt = predict(pt_net, x_test);
fprintf('  dlnetwork predict([0;0;0]) = [%.4f, %.4f]\n', ...
    extractdata(y_test(1)), extractdata(y_test(2)));
fprintf('  PyTorch   predict([0;0;0]) = [%.4f, %.4f]\n', ...
    extractdata(y_pt(1)), extractdata(y_pt(2)));
clear pt_net;  % no longer needed

%% 2. Load normalization constants (W already loaded above)
node_dxdt_scale = double(W.dxdt_scale(:));
node_x_mean = double(W.x_mean(:));
node_x_std  = double(W.x_std(:));
node_u_mean = double(W.u_mean);
node_u_std  = double(W.u_std);
fprintf('  dxdt_scale: [%.0f, %.0f]\n', node_dxdt_scale(1), node_dxdt_scale(2));

% Initial state at D=0.3 (matches Step block initial value)
D_init = 0.3; Vin = 5; R_load = 20;
Vout_init = Vin / (1 - D_init);
iL_init = Vout_init / (R_load * (1 - D_init));
node_state_ic = [Vout_init; iL_init];
fprintf('  IC: Vout=%.2fV, iL=%.2fA\n', Vout_init, iL_init);

% Save dlnetwork for Predict block
netFile = fullfile(modelsDir, 'neuralode_dlnetwork.mat');
save(netFile, 'node_net');

% Save workspace data
romDataFile = fullfile(modelsDir, 'boost_branch_c_data.mat');
save(romDataFile, 'node_dxdt_scale', 'node_x_mean', 'node_x_std', ...
    'node_u_mean', 'node_u_std', 'node_state_ic', 'Ts', 'nx');

%% 3. Export layer blocks model (for Model B)
fprintf('\nExporting dlnetwork to Simulink layer blocks...\n');
mlpModelName = 'neuralode_mlp_blocks';
if bdIsLoaded(mlpModelName), close_system(mlpModelName, 0); end
mdlInfo = exportNetworkToSimulink(node_net, ...
    'ModelName', mlpModelName, ...
    'ModelPath', modelsDir, ...
    'SampleTime', num2str(Ts), ...
    'OpenSystem', false, ...
    'SaveModelToFile', true);
load_system(fullfile(modelsDir, [mlpModelName '.slx']));
fprintf('  Done: %s\n', mlpModelName);

%% 4. Load empirical ripple data
fprintf('\nLoading empirical ripple data...\n');
rippleFile = fullfile(modelsDir, 'ripple_empirical_data.mat');
if exist(rippleFile, 'file')
    R = load(rippleFile);
    lut_duty = R.ripple_duty_grid;
    lut_delta_iL = R.ripple_delta_iL;
    Tsw = R.ripple_Tsw;
    Ts_ripple = Tsw / 10;  % 10 points per switching cycle
    fprintf('  Empirical: %d duty points, Tsw=%.2fμs\n', numel(lut_duty), Tsw*1e6);
else
    fprintf('  WARNING: ripple_empirical_data.mat not found, using analytical formula\n');
    Tsw = 5e-6; Vin = 5; L = 20e-6;
    lut_duty = linspace(0.05, 0.85, 17);
    lut_delta_iL = Vin .* lut_duty .* Tsw ./ L;
    Ts_ripple = 0.5e-6;
end

% Add ripple data to ROM data file
save(romDataFile, '-append', 'lut_duty', 'lut_delta_iL', 'Tsw', 'Ts_ripple');

%% ========================================================================
%% 5. BUILD MODEL A: Predict block
%% ========================================================================
fprintf('\n--- Building Model A: Predict block ---\n');
modelA = 'boost_openloop_branch_c_predict';
buildOpenLoopModel(modelA, modelsDir, Ts, nx, romDataFile, ...
    'predict', netFile, mlpModelName, Tsw, Ts_ripple);

%% ========================================================================
%% 6. BUILD MODEL B: Layer blocks
%% ========================================================================
fprintf('\n--- Building Model B: Layer blocks ---\n');
modelB = 'boost_openloop_branch_c_layers';
buildOpenLoopModel(modelB, modelsDir, Ts, nx, romDataFile, ...
    'layers', netFile, mlpModelName, Tsw, Ts_ripple);

% Clean up
close_system(mlpModelName, 0);

fprintf('\n=== Both models saved ===\n');
fprintf('  A (Predict):     %s.slx\n', modelA);
fprintf('  B (Layer blocks): %s.slx\n', modelB);
fprintf('  Run compare_branch_c_speed.m to benchmark.\n');

close all;


%% ========================================================================
%% Helper: buildOpenLoopModel
%% ========================================================================
function buildOpenLoopModel(modelName, modelsDir, Ts, nx, romDataFile, ...
        mlpMode, netFile, mlpModelName, Tsw, Ts_ripple)
    stopTime = 0.04;

    if bdIsLoaded(modelName), close_system(modelName, 0); end
    slxcFile = fullfile(modelsDir, [modelName '.slxc']);
    if exist(slxcFile, 'file'), delete(slxcFile); end

    new_system(modelName); open_system(modelName);
    set_param(modelName, 'Solver', 'ode1');
    set_param(modelName, 'FixedStep', num2str(Ts_ripple));  % ripple rate
    set_param(modelName, 'StopTime', num2str(stopTime));
    set_param(modelName, 'SignalLogging', 'on');
    set_param(modelName, 'SaveOutput', 'on');

    % --- ROM subsystem ---
    romPath = [modelName '/NeuralODE_ROM'];
    add_block('simulink/Ports & Subsystems/Subsystem', romPath, ...
        'Position', [200, 70, 500, 180]);
    set_param(romPath, 'TreatAsAtomicUnit', 'on');
    set_param(romPath, 'SystemSampleTime', num2str(Ts));
    delete_line(romPath, 'In1/1', 'Out1/1');

    % --- Normalize ---
    add_block('simulink/User-Defined Functions/MATLAB Function', ...
        [romPath '/Normalize'], 'Position', [60, 80, 180, 120]);
    rt = sfroot;
    normChart = rt.find('-isa', 'Stateflow.EMChart', 'Path', [romPath '/Normalize']);
    normLines = {
        'function z_n = Normalize(duty, x_state)'
        '%#codegen'
        'x_n = (x_state - node_x_mean) ./ node_x_std;'
        'u_n = (duty - node_u_mean) / node_u_std;'
        'z_n = single([x_n; u_n]);'
    };
    normChart.Script = strjoin(normLines, newline);
    for pName = {'node_x_mean', 'node_x_std', 'node_u_mean', 'node_u_std'}
        d = Stateflow.Data(normChart); d.Name = pName{1}; d.Scope = 'Parameter';
    end

    % --- MLP block (Predict or Layers) ---
    switch mlpMode
        case 'predict'
            % Predict block with codegen acceleration
            add_block('deeplib/Predict', [romPath '/MLP'], ...
                'Position', [230, 80, 400, 120], ...
                'Network', 'Network from MAT-file', ...
                'NetworkFilePath', strrep(netFile, '\', '/'), ...
                'MiniBatchSize', '1');

        case 'layers'
            % Copy layer blocks from exported model
            mlpBlocks = find_system(mlpModelName, 'SearchDepth', 1, ...
                'BlockType', 'SubSystem');
            if ~isempty(mlpBlocks)
                add_block(mlpBlocks{1}, [romPath '/MLP'], ...
                    'Position', [230, 80, 400, 120]);
            else
                error('No subsystem found in exported layer blocks model');
            end
    end

    % --- ScaleEuler ---
    add_block('simulink/User-Defined Functions/MATLAB Function', ...
        [romPath '/ScaleEuler'], 'Position', [440, 80, 560, 120]);
    seChart = rt.find('-isa', 'Stateflow.EMChart', 'Path', [romPath '/ScaleEuler']);
    seLines = {
        'function x_next = ScaleEuler(raw_dxdt, x_state)'
        '%#codegen'
        sprintf('Ts = %.10e;', Ts)
        'dxdt = double(raw_dxdt) .* node_dxdt_scale;'
        'x_next = x_state + Ts * dxdt;'
    };
    seChart.Script = strjoin(seLines, newline);
    d = Stateflow.Data(seChart); d.Name = 'node_dxdt_scale'; d.Scope = 'Parameter';

    % --- Unit Delay ---
    add_block('simulink/Discrete/Unit Delay', [romPath '/StateDelay'], ...
        'Position', [350, 180, 410, 220], ...
        'SampleTime', num2str(Ts), ...
        'InitialCondition', 'node_state_ic');

    % --- OutputFcn ---
    add_block('simulink/User-Defined Functions/MATLAB Function', ...
        [romPath '/OutputFcn'], 'Position', [460, 180, 560, 220]);
    outChart = rt.find('-isa', 'Stateflow.EMChart', 'Path', [romPath '/OutputFcn']);
    outChart.Script = sprintf('function x_out = OutputFcn(x_state)\n%%#codegen\nx_out = x_state;\n');

    % --- Wire ROM subsystem ---
    add_line(romPath, 'In1/1', 'Normalize/1', 'autorouting', 'smart');
    add_line(romPath, 'StateDelay/1', 'Normalize/2', 'autorouting', 'smart');
    add_line(romPath, 'Normalize/1', 'MLP/1', 'autorouting', 'smart');
    add_line(romPath, 'MLP/1', 'ScaleEuler/1', 'autorouting', 'smart');
    add_line(romPath, 'StateDelay/1', 'ScaleEuler/2', 'autorouting', 'smart');
    add_line(romPath, 'ScaleEuler/1', 'StateDelay/1', 'autorouting', 'smart');
    add_line(romPath, 'StateDelay/1', 'OutputFcn/1', 'autorouting', 'smart');
    add_line(romPath, 'OutputFcn/1', 'Out1/1', 'autorouting', 'smart');

    % --- Top level ---
    add_block('simulink/Sources/Step', [modelName '/Step'], ...
        'Position', [30, 100, 70, 140], ...
        'Time', '0.02', 'Before', '0.3', 'After', '0.6', ...
        'SampleTime', '0');

    add_block('simulink/Signal Routing/Demux', [modelName '/Demux'], ...
        'Position', [560, 105, 570, 145], 'Outputs', '2');

    add_block('simulink/Sinks/Scope', [modelName '/Vout_Scope'], ...
        'Position', [830, 50, 870, 80]);
    add_block('simulink/Sinks/Scope', [modelName '/iL_Scope'], ...
        'Position', [830, 110, 870, 140]);
    add_block('simulink/Sinks/Scope', [modelName '/iL_Ripple_Scope'], ...
        'Position', [830, 270, 870, 300]);

    % --- Ripple reconstruction blocks ---
    % DeltaIL Lookup (empirical: duty -> delta_iL)
    add_block('simulink/Lookup Tables/1-D Lookup Table', ...
        [modelName '/DeltaIL_Lookup'], ...
        'Position', [560, 200, 640, 240], ...
        'Table', 'lut_delta_iL', ...
        'BreakpointsForDimension1', 'lut_duty', ...
        'InterpMethod', 'Linear point-slope', ...
        'ExtrapMethod', 'Clip');

    % Rate Transitions (ROM rate -> ripple rate)
    add_block('simulink/Signal Attributes/Rate Transition', ...
        [modelName '/RT_iLavg'], ...
        'Position', [640, 130, 680, 150], ...
        'InitialCondition', '0', 'OutPortSampleTime', num2str(Ts_ripple));
    add_block('simulink/Signal Attributes/Rate Transition', ...
        [modelName '/RT_deltaIL'], ...
        'Position', [680, 210, 720, 230], ...
        'InitialCondition', '0', 'OutPortSampleTime', num2str(Ts_ripple));
    add_block('simulink/Signal Attributes/Rate Transition', ...
        [modelName '/RT_duty'], ...
        'Position', [560, 270, 600, 290], ...
        'InitialCondition', '0', 'OutPortSampleTime', num2str(Ts_ripple));

    % Digital Clock
    add_block('simulink/Sources/Digital Clock', [modelName '/DigitalClock'], ...
        'Position', [640, 310, 690, 330], ...
        'SampleTime', num2str(Ts_ripple));

    % RippleRecon MATLAB Function
    add_block('simulink/User-Defined Functions/MATLAB Function', ...
        [modelName '/RippleRecon'], ...
        'Position', [740, 210, 820, 300]);

    rt2 = sfroot;
    rippleChart = rt2.find('-isa', 'Stateflow.EMChart', 'Path', [modelName '/RippleRecon']);
    rippleLines = {
        'function iL = RippleRecon(iL_avg, delta_iL, duty, time)'
        '%#codegen'
        '% Reconstruct triangular ripple from average + empirical amplitude.'
        '% Topology-agnostic: delta_iL comes from measured data, not formula.'
        sprintf('Tsw = %.10e;', Tsw)
        'phase = mod(time, Tsw) / Tsw;'
        'duty_c = max(0.01, min(0.99, duty));'
        'if phase < duty_c'
        '    iL = iL_avg - delta_iL/2 + delta_iL * (phase / duty_c);'
        'else'
        '    iL = iL_avg + delta_iL/2 - delta_iL * ((phase - duty_c) / (1 - duty_c + eps));'
        'end'
    };
    rippleChart.Script = strjoin(rippleLines, newline);

    % --- Wire main signal path ---
    add_line(modelName, 'Step/1', 'NeuralODE_ROM/1', 'autorouting', 'smart');
    add_line(modelName, 'NeuralODE_ROM/1', 'Demux/1', 'autorouting', 'smart');
    add_line(modelName, 'Demux/1', 'Vout_Scope/1', 'autorouting', 'smart');
    add_line(modelName, 'Demux/2', 'iL_Scope/1', 'autorouting', 'smart');

    % --- Wire ripple path ---
    add_line(modelName, 'Step/1', 'DeltaIL_Lookup/1', 'autorouting', 'smart');
    add_line(modelName, 'Demux/2', 'RT_iLavg/1', 'autorouting', 'smart');
    add_line(modelName, 'DeltaIL_Lookup/1', 'RT_deltaIL/1', 'autorouting', 'smart');
    add_line(modelName, 'Step/1', 'RT_duty/1', 'autorouting', 'smart');
    add_line(modelName, 'RT_iLavg/1', 'RippleRecon/1', 'autorouting', 'smart');
    add_line(modelName, 'RT_deltaIL/1', 'RippleRecon/2', 'autorouting', 'smart');
    add_line(modelName, 'RT_duty/1', 'RippleRecon/3', 'autorouting', 'smart');
    add_line(modelName, 'DigitalClock/1', 'RippleRecon/4', 'autorouting', 'smart');
    add_line(modelName, 'RippleRecon/1', 'iL_Ripple_Scope/1', 'autorouting', 'smart');

    % Signal logging — named for Simulation Data Inspector
    % Vout (from Demux output 1)
    ph_vout = get_param([modelName '/Demux'], 'PortHandles');
    set_param(ph_vout.Outport(1), 'DataLogging', 'on', ...
        'DataLoggingName', 'Vout', 'DataLoggingNameMode', 'Custom');

    % iL_avg (from Demux output 2)
    set_param(ph_vout.Outport(2), 'DataLogging', 'on', ...
        'DataLoggingName', 'iL_avg', 'DataLoggingNameMode', 'Custom');

    % duty
    ph_duty = get_param([modelName '/Step'], 'PortHandles');
    set_param(ph_duty.Outport(1), 'DataLogging', 'on', ...
        'DataLoggingName', 'duty', 'DataLoggingNameMode', 'Custom');

    % iL with ripple reconstruction
    ph_il = get_param([modelName '/RippleRecon'], 'PortHandles');
    set_param(ph_il.Outport(1), 'DataLogging', 'on', ...
        'DataLoggingName', 'iL_ripple', 'DataLoggingNameMode', 'Custom');

    % PreLoadFcn
    preloadCmd = sprintf('load(''%s'');\n', strrep(romDataFile, '\', '/'));
    set_param(modelName, 'PreLoadFcn', preloadCmd);

    % Save
    Simulink.BlockDiagram.arrangeSystem(modelName);
    modelPath = fullfile(modelsDir, [modelName '.slx']);
    save_system(modelName, modelPath);
    fprintf('  Saved: %s\n', modelPath);
end
