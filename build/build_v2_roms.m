% build_v2_roms.m
% Builds Simulink ROM models for the weekend training best results:
%   1. boost_rom_mlp_v2_seed3.slx — RK4 MLP seed 3 (Vout RMSE 0.036V)
%   2. boost_rom_lpv_seed7.slx    — LPV seed 7 (Vout RMSE 0.019V)
%
% Both models use the same structure as the original branch C ROM:
%   duty -> [ROM subsystem] -> [Vout; iL] + ripple reconstruction
%
% MLP model: [Normalize] -> [MLP 3->2] -> [Scale+Euler] -> state
% LPV model: [Normalize] -> [MLP 3->8] -> [LPV_Compute: A*x+B*u+c] -> [Scale+Euler] -> state
%
% Both load duty_input from workspace (From Workspace block) for comparison profile.

buildDir  = fileparts(mfilename('fullpath'));
repoRoot  = fileparts(buildDir);
run(fullfile(repoRoot, 'startup.m'));

Ts = 5e-6;
nx = 2;
H  = 64;
modelsDir   = fullfile(repoRoot, 'model_data');
simulinkDir = fullfile(repoRoot, 'simulink_models');

%% ========================================================================
%% 1. MLP V2 Seed 3
%% ========================================================================
fprintf('\n=== Building MLP V2 Seed 3 ===\n');

% Load weights
W_mlp = load(fullfile(modelsDir, 'mlp_v2_seed3_weights.mat'));
fprintf('  Weights loaded: W1=%s W3=%s\n', mat2str(size(W_mlp.W1)), mat2str(size(W_mlp.W3)));

% Build dlnetwork
layers_mlp = [
    featureInputLayer(3, 'Name', 'input', 'Normalization', 'none')
    fullyConnectedLayer(H, 'Name', 'fc1')
    tanhLayer('Name', 'tanh1')
    fullyConnectedLayer(H, 'Name', 'fc2')
    tanhLayer('Name', 'tanh2')
    fullyConnectedLayer(nx, 'Name', 'fc_out')
];
net_mlp = dlnetwork(layers_mlp);
net_mlp.Learnables.Value{1} = dlarray(single(W_mlp.W1));
net_mlp.Learnables.Value{2} = dlarray(single(W_mlp.b1(:)));
net_mlp.Learnables.Value{3} = dlarray(single(W_mlp.W2));
net_mlp.Learnables.Value{4} = dlarray(single(W_mlp.b2(:)));
net_mlp.Learnables.Value{5} = dlarray(single(W_mlp.W3));
net_mlp.Learnables.Value{6} = dlarray(single(W_mlp.b3(:)));
fprintf('  dlnetwork built: %d params\n', sum(cellfun(@numel, net_mlp.Learnables.Value)));

% Export layer blocks
mlpBlocksName = 'mlp_v2_seed3_blocks';
if bdIsLoaded(mlpBlocksName), close_system(mlpBlocksName, 0); end
exportNetworkToSimulink(net_mlp, ...
    'ModelName', mlpBlocksName, ...
    'ModelPath', modelsDir, ...
    'SampleTime', num2str(Ts), ...
    'OpenSystem', false, ...
    'SaveModelToFile', true);
load_system(fullfile(modelsDir, [mlpBlocksName '.slx']));
fprintf('  Layer blocks exported: %s\n', mlpBlocksName);

% Save ROM data
mlp_dxdt_scale = double(W_mlp.dxdt_scale(:));
mlp_x_mean = double(W_mlp.x_mean(:));
mlp_x_std  = double(W_mlp.x_std(:));
mlp_u_mean = double(W_mlp.u_mean);
mlp_u_std  = double(W_mlp.u_std);
D_init = 0.2; Vin = 5; R_load = 20;
Vout_init = Vin / (1 - D_init);
iL_init = Vout_init / (R_load * (1 - D_init));
mlp_state_ic = [Vout_init; iL_init];

mlpDataFile = fullfile(modelsDir, 'boost_rom_mlp_v2_seed3_data.mat');
node_dxdt_scale = mlp_dxdt_scale;
node_x_mean = mlp_x_mean; node_x_std = mlp_x_std;
node_u_mean = mlp_u_mean; node_u_std = mlp_u_std;
node_state_ic = mlp_state_ic;

% Load ripple data
rippleFile = fullfile(modelsDir, 'ripple_empirical_data.mat');
R = load(rippleFile);
lut_duty = R.ripple_duty_grid;
lut_delta_iL = R.ripple_delta_iL;
Tsw = R.ripple_Tsw;
Ts_ripple = Tsw / 10;

save(mlpDataFile, 'node_dxdt_scale', 'node_x_mean', 'node_x_std', ...
    'node_u_mean', 'node_u_std', 'node_state_ic', 'Ts', 'nx', ...
    'lut_duty', 'lut_delta_iL', 'Tsw', 'Ts_ripple');
fprintf('  ROM data saved: %s\n', mlpDataFile);

% Build Simulink model
modelName = 'boost_rom_mlp_v2_seed3';
buildROMModel(modelName, simulinkDir, Ts, nx, mlpDataFile, ...
    'layers', mlpBlocksName, Tsw, Ts_ripple);
close_system(mlpBlocksName, 0);
fprintf('  Model built: %s.slx\n', modelName);

%% ========================================================================
%% 2. LPV Seed 7
%% ========================================================================
fprintf('\n=== Building LPV Seed 7 ===\n');

% Load weights
W_lpv = load(fullfile(modelsDir, 'lpv_seed7_weights.mat'));
fprintf('  Weights loaded: W1=%s W3=%s (8 outputs)\n', ...
    mat2str(size(W_lpv.W1)), mat2str(size(W_lpv.W3)));

% Build dlnetwork (3->8 output MLP)
layers_lpv = [
    featureInputLayer(3, 'Name', 'input', 'Normalization', 'none')
    fullyConnectedLayer(H, 'Name', 'fc1')
    tanhLayer('Name', 'tanh1')
    fullyConnectedLayer(H, 'Name', 'fc2')
    tanhLayer('Name', 'tanh2')
    fullyConnectedLayer(8, 'Name', 'fc_out')    % 8 outputs: A(4)+B(2)+c(2)
];
net_lpv = dlnetwork(layers_lpv);
net_lpv.Learnables.Value{1} = dlarray(single(W_lpv.W1));
net_lpv.Learnables.Value{2} = dlarray(single(W_lpv.b1(:)));
net_lpv.Learnables.Value{3} = dlarray(single(W_lpv.W2));
net_lpv.Learnables.Value{4} = dlarray(single(W_lpv.b2(:)));
net_lpv.Learnables.Value{5} = dlarray(single(W_lpv.W3));
net_lpv.Learnables.Value{6} = dlarray(single(W_lpv.b3(:)));
fprintf('  dlnetwork built: %d params\n', sum(cellfun(@numel, net_lpv.Learnables.Value)));

% Export layer blocks
lpvBlocksName = 'lpv_seed7_blocks';
if bdIsLoaded(lpvBlocksName), close_system(lpvBlocksName, 0); end
exportNetworkToSimulink(net_lpv, ...
    'ModelName', lpvBlocksName, ...
    'ModelPath', modelsDir, ...
    'SampleTime', num2str(Ts), ...
    'OpenSystem', false, ...
    'SaveModelToFile', true);
load_system(fullfile(modelsDir, [lpvBlocksName '.slx']));
fprintf('  Layer blocks exported: %s\n', lpvBlocksName);

% Save ROM data
lpv_dxdt_scale = double(W_lpv.dxdt_scale(:));
lpv_x_mean = double(W_lpv.x_mean(:));
lpv_x_std  = double(W_lpv.x_std(:));
lpv_u_mean = double(W_lpv.u_mean);
lpv_u_std  = double(W_lpv.u_std);
lpv_state_ic = mlp_state_ic;  % same IC

lpvDataFile = fullfile(modelsDir, 'boost_rom_lpv_seed7_data.mat');
node_dxdt_scale = lpv_dxdt_scale;
node_x_mean = lpv_x_mean; node_x_std = lpv_x_std;
node_u_mean = lpv_u_mean; node_u_std = lpv_u_std;
node_state_ic = lpv_state_ic;

save(lpvDataFile, 'node_dxdt_scale', 'node_x_mean', 'node_x_std', ...
    'node_u_mean', 'node_u_std', 'node_state_ic', 'Ts', 'nx', ...
    'lut_duty', 'lut_delta_iL', 'Tsw', 'Ts_ripple');
fprintf('  ROM data saved: %s\n', lpvDataFile);

% Build Simulink model (LPV variant)
modelName = 'boost_rom_lpv_seed7';
buildROMModel_LPV(modelName, simulinkDir, Ts, nx, lpvDataFile, ...
    lpvBlocksName, Tsw, Ts_ripple);
close_system(lpvBlocksName, 0);
fprintf('  Model built: %s.slx\n', modelName);

fprintf('\n=== Both models built ===\n');
fprintf('  MLP: simulink_models/boost_rom_mlp_v2_seed3.slx\n');
fprintf('  LPV: simulink_models/boost_rom_lpv_seed7.slx\n');
fprintf('  Run setup_comparison_profile.m to compare all models.\n');


%% ========================================================================
%% Helper: buildROMModel (same structure as original MLP ROM)
%% ========================================================================
function buildROMModel(modelName, simulinkDir, Ts, nx, romDataFile, ...
        mlpMode, mlpBlocksName, Tsw, Ts_ripple)

    if bdIsLoaded(modelName), close_system(modelName, 0); end
    new_system(modelName); open_system(modelName);
    set_param(modelName, 'Solver', 'ode1');
    set_param(modelName, 'FixedStep', num2str(Ts_ripple));
    set_param(modelName, 'StopTime', '1.4');
    set_param(modelName, 'SignalLogging', 'on');

    % ROM subsystem
    romPath = [modelName '/NeuralODE_ROM'];
    add_block('simulink/Ports & Subsystems/Subsystem', romPath, ...
        'Position', [200, 70, 500, 180]);
    set_param(romPath, 'TreatAsAtomicUnit', 'on');
    set_param(romPath, 'SystemSampleTime', num2str(Ts));
    delete_line(romPath, 'In1/1', 'Out1/1');

    % Normalize
    rt = sfroot;
    add_block('simulink/User-Defined Functions/MATLAB Function', ...
        [romPath '/Normalize'], 'Position', [60, 80, 180, 120]);
    normChart = rt.find('-isa', 'Stateflow.EMChart', 'Path', [romPath '/Normalize']);
    normChart.Script = strjoin({
        'function z_n = Normalize(duty, x_state)'
        '%#codegen'
        'x_n = (x_state - node_x_mean) ./ node_x_std;'
        'u_n = (duty - node_u_mean) / node_u_std;'
        'z_n = single([x_n; u_n]);'
    }, newline);
    for pName = {'node_x_mean', 'node_x_std', 'node_u_mean', 'node_u_std'}
        d = Stateflow.Data(normChart); d.Name = pName{1}; d.Scope = 'Parameter';
    end

    % MLP layer blocks
    mlpBlocks = find_system(mlpBlocksName, 'SearchDepth', 1, 'BlockType', 'SubSystem');
    add_block(mlpBlocks{1}, [romPath '/MLP'], 'Position', [230, 80, 400, 120]);

    % ScaleEuler
    add_block('simulink/User-Defined Functions/MATLAB Function', ...
        [romPath '/ScaleEuler'], 'Position', [440, 80, 560, 120]);
    seChart = rt.find('-isa', 'Stateflow.EMChart', 'Path', [romPath '/ScaleEuler']);
    seChart.Script = strjoin({
        'function x_next = ScaleEuler(raw_dxdt, x_state)'
        '%#codegen'
        sprintf('Ts = %.10e;', Ts)
        'dxdt = double(raw_dxdt) .* node_dxdt_scale;'
        'x_next = x_state + Ts * dxdt;'
    }, newline);
    d = Stateflow.Data(seChart); d.Name = 'node_dxdt_scale'; d.Scope = 'Parameter';

    % Unit Delay
    add_block('simulink/Discrete/Unit Delay', [romPath '/StateDelay'], ...
        'Position', [350, 180, 410, 220], ...
        'SampleTime', num2str(Ts), 'InitialCondition', 'node_state_ic');

    % Output
    add_block('simulink/User-Defined Functions/MATLAB Function', ...
        [romPath '/OutputFcn'], 'Position', [460, 180, 560, 220]);
    outChart = rt.find('-isa', 'Stateflow.EMChart', 'Path', [romPath '/OutputFcn']);
    outChart.Script = sprintf('function x_out = OutputFcn(x_state)\n%%#codegen\nx_out = x_state;\n');

    % Wire ROM
    add_line(romPath, 'In1/1', 'Normalize/1', 'autorouting', 'smart');
    add_line(romPath, 'StateDelay/1', 'Normalize/2', 'autorouting', 'smart');
    add_line(romPath, 'Normalize/1', 'MLP/1', 'autorouting', 'smart');
    add_line(romPath, 'MLP/1', 'ScaleEuler/1', 'autorouting', 'smart');
    add_line(romPath, 'StateDelay/1', 'ScaleEuler/2', 'autorouting', 'smart');
    add_line(romPath, 'ScaleEuler/1', 'StateDelay/1', 'autorouting', 'smart');
    add_line(romPath, 'StateDelay/1', 'OutputFcn/1', 'autorouting', 'smart');
    add_line(romPath, 'OutputFcn/1', 'Out1/1', 'autorouting', 'smart');

    % Top level: From Workspace + Demux + Scopes + Ripple
    buildTopLevel(modelName, romDataFile, Tsw, Ts_ripple);

    % PreLoadFcn
    set_param(modelName, 'PreLoadFcn', sprintf(...
        'mdlDir = fileparts(which(''%s''));\nrepoRoot = fileparts(mdlDir);\nload(fullfile(repoRoot, ''model_data'', ''%s''));\nload(fullfile(repoRoot, ''model_data'', ''comparison_duty_profile.mat''));', ...
        modelName, [modelName(1:end-0) '_data.mat']));

    % Wait — the data file name doesn't match. Let me fix:
    % Model name: boost_rom_mlp_v2_seed3, data file: boost_rom_mlp_v2_seed3_data.mat
    dataFileName = [modelName '_data.mat'];
    set_param(modelName, 'PreLoadFcn', sprintf([...
        'mdlDir = fileparts(which(''%s''));\n' ...
        'repoRoot = fileparts(mdlDir);\n' ...
        'load(fullfile(repoRoot, ''model_data'', ''%s''));\n' ...
        'load(fullfile(repoRoot, ''model_data'', ''comparison_duty_profile.mat''));\n'], ...
        modelName, dataFileName));

    Simulink.BlockDiagram.arrangeSystem(modelName);
    save_system(modelName, fullfile(simulinkDir, [modelName '.slx']));
end


%% ========================================================================
%% Helper: buildROMModel_LPV (modified for LPV: MLP outputs 8, then A*x+B*u+c)
%% ========================================================================
function buildROMModel_LPV(modelName, simulinkDir, Ts, nx, romDataFile, ...
        lpvBlocksName, Tsw, Ts_ripple)

    if bdIsLoaded(modelName), close_system(modelName, 0); end
    new_system(modelName); open_system(modelName);
    set_param(modelName, 'Solver', 'ode1');
    set_param(modelName, 'FixedStep', num2str(Ts_ripple));
    set_param(modelName, 'StopTime', '1.4');
    set_param(modelName, 'SignalLogging', 'on');

    % ROM subsystem
    romPath = [modelName '/LPV_ROM'];
    add_block('simulink/Ports & Subsystems/Subsystem', romPath, ...
        'Position', [200, 70, 560, 200]);
    set_param(romPath, 'TreatAsAtomicUnit', 'on');
    set_param(romPath, 'SystemSampleTime', num2str(Ts));
    delete_line(romPath, 'In1/1', 'Out1/1');

    rt = sfroot;

    % Normalize (outputs [x_n; u_n] as 3-element vector, AND x_n, u_n separately)
    add_block('simulink/User-Defined Functions/MATLAB Function', ...
        [romPath '/Normalize'], 'Position', [60, 70, 180, 130]);
    normChart = rt.find('-isa', 'Stateflow.EMChart', 'Path', [romPath '/Normalize']);
    normChart.Script = strjoin({
        'function [z_n, x_n, u_n] = Normalize(duty, x_state)'
        '%#codegen'
        'x_n = (x_state - node_x_mean) ./ node_x_std;'
        'u_n = (duty - node_u_mean) / node_u_std;'
        'z_n = single([x_n; u_n]);'
    }, newline);
    for pName = {'node_x_mean', 'node_x_std', 'node_u_mean', 'node_u_std'}
        d = Stateflow.Data(normChart); d.Name = pName{1}; d.Scope = 'Parameter';
    end

    % MLP layer blocks (3->8 outputs)
    lpvBlocks = find_system(lpvBlocksName, 'SearchDepth', 1, 'BlockType', 'SubSystem');
    add_block(lpvBlocks{1}, [romPath '/MLP_8out'], 'Position', [220, 70, 380, 110]);

    % LPV Compute: raw(8) + x_n(2) + u_n(1) -> dxdt(2)
    add_block('simulink/User-Defined Functions/MATLAB Function', ...
        [romPath '/LPV_Compute'], 'Position', [420, 60, 560, 140]);
    lpvChart = rt.find('-isa', 'Stateflow.EMChart', 'Path', [romPath '/LPV_Compute']);
    lpvChart.Script = strjoin({
        'function dxdt = LPV_Compute(raw, x_n, u_n)'
        '%#codegen'
        '% raw: 8-element vector [A(4), B(2), c(2)]'
        '% x_n: 2-element normalized state'
        '% u_n: scalar normalized input'
        'A = reshape(raw(1:4), [2, 2])'';  % transpose: PyTorch row-major -> MATLAB col-major'
        'B = reshape(raw(5:6), [2, 1]);'
        'c = raw(7:8);'
        'dxdt_n = A * x_n + B * u_n + c;'
        'dxdt = double(dxdt_n) .* node_dxdt_scale;'
    }, newline);
    d = Stateflow.Data(lpvChart); d.Name = 'node_dxdt_scale'; d.Scope = 'Parameter';

    % Euler integrator
    add_block('simulink/User-Defined Functions/MATLAB Function', ...
        [romPath '/EulerStep'], 'Position', [590, 80, 700, 120]);
    eulerChart = rt.find('-isa', 'Stateflow.EMChart', 'Path', [romPath '/EulerStep']);
    eulerChart.Script = strjoin({
        'function x_next = EulerStep(dxdt, x_state)'
        '%#codegen'
        sprintf('Ts = %.10e;', Ts)
        'x_next = x_state + Ts * dxdt;'
    }, newline);

    % Unit Delay
    add_block('simulink/Discrete/Unit Delay', [romPath '/StateDelay'], ...
        'Position', [400, 180, 460, 220], ...
        'SampleTime', num2str(Ts), 'InitialCondition', 'node_state_ic');

    % Output
    add_block('simulink/User-Defined Functions/MATLAB Function', ...
        [romPath '/OutputFcn'], 'Position', [520, 180, 620, 220]);
    outChart = rt.find('-isa', 'Stateflow.EMChart', 'Path', [romPath '/OutputFcn']);
    outChart.Script = sprintf('function x_out = OutputFcn(x_state)\n%%#codegen\nx_out = x_state;\n');

    % Wire LPV ROM
    add_line(romPath, 'In1/1', 'Normalize/1', 'autorouting', 'smart');
    add_line(romPath, 'StateDelay/1', 'Normalize/2', 'autorouting', 'smart');
    add_line(romPath, 'Normalize/1', 'MLP_8out/1', 'autorouting', 'smart');
    add_line(romPath, 'MLP_8out/1', 'LPV_Compute/1', 'autorouting', 'smart');
    add_line(romPath, 'Normalize/2', 'LPV_Compute/2', 'autorouting', 'smart');
    add_line(romPath, 'Normalize/3', 'LPV_Compute/3', 'autorouting', 'smart');
    add_line(romPath, 'LPV_Compute/1', 'EulerStep/1', 'autorouting', 'smart');
    add_line(romPath, 'StateDelay/1', 'EulerStep/2', 'autorouting', 'smart');
    add_line(romPath, 'EulerStep/1', 'StateDelay/1', 'autorouting', 'smart');
    add_line(romPath, 'StateDelay/1', 'OutputFcn/1', 'autorouting', 'smart');
    add_line(romPath, 'OutputFcn/1', 'Out1/1', 'autorouting', 'smart');

    % Top level
    buildTopLevel(modelName, romDataFile, Tsw, Ts_ripple);

    % PreLoadFcn
    dataFileName = [modelName '_data.mat'];
    set_param(modelName, 'PreLoadFcn', sprintf([...
        'mdlDir = fileparts(which(''%s''));\n' ...
        'repoRoot = fileparts(mdlDir);\n' ...
        'load(fullfile(repoRoot, ''model_data'', ''%s''));\n' ...
        'load(fullfile(repoRoot, ''model_data'', ''comparison_duty_profile.mat''));\n'], ...
        modelName, dataFileName));

    Simulink.BlockDiagram.arrangeSystem(modelName);
    save_system(modelName, fullfile(simulinkDir, [modelName '.slx']));
end


%% ========================================================================
%% Helper: buildTopLevel (shared by both MLP and LPV)
%% ========================================================================
function buildTopLevel(modelName, romDataFile, Tsw, Ts_ripple)
    % Find the ROM subsystem (NeuralODE_ROM or LPV_ROM)
    romBlocks = find_system(modelName, 'SearchDepth', 1, 'BlockType', 'SubSystem');
    romName = romBlocks{1};  % first subsystem

    % From Workspace for duty input
    add_block('simulink/Sources/From Workspace', [modelName '/FromWS_duty'], ...
        'Position', [30, 100, 120, 140], ...
        'VariableName', 'duty_input', ...
        'SampleTime', '0', ...
        'OutputAfterFinalValue', 'Holding final value');

    % Demux
    add_block('simulink/Signal Routing/Demux', [modelName '/Demux'], ...
        'Position', [600, 105, 610, 145], 'Outputs', '2');

    % Scopes
    add_block('simulink/Sinks/Scope', [modelName '/Vout_Scope'], ...
        'Position', [830, 50, 870, 80]);
    add_block('simulink/Sinks/Scope', [modelName '/iL_Scope'], ...
        'Position', [830, 110, 870, 140]);
    add_block('simulink/Sinks/Scope', [modelName '/iL_Ripple_Scope'], ...
        'Position', [830, 270, 870, 300]);

    % Ripple reconstruction
    add_block('simulink/Lookup Tables/1-D Lookup Table', ...
        [modelName '/DeltaIL_Lookup'], ...
        'Position', [560, 200, 640, 240], ...
        'Table', 'lut_delta_iL', 'BreakpointsForDimension1', 'lut_duty', ...
        'InterpMethod', 'Linear point-slope', 'ExtrapMethod', 'Clip');

    add_block('simulink/Signal Attributes/Rate Transition', ...
        [modelName '/RT_iLavg'], 'Position', [640, 130, 680, 150], ...
        'InitialCondition', '0', 'OutPortSampleTime', num2str(Ts_ripple));
    add_block('simulink/Signal Attributes/Rate Transition', ...
        [modelName '/RT_deltaIL'], 'Position', [680, 210, 720, 230], ...
        'InitialCondition', '0', 'OutPortSampleTime', num2str(Ts_ripple));
    add_block('simulink/Signal Attributes/Rate Transition', ...
        [modelName '/RT_duty'], 'Position', [560, 270, 600, 290], ...
        'InitialCondition', '0', 'OutPortSampleTime', num2str(Ts_ripple));
    add_block('simulink/Sources/Digital Clock', [modelName '/DigitalClock'], ...
        'Position', [640, 310, 690, 330], 'SampleTime', num2str(Ts_ripple));

    add_block('simulink/User-Defined Functions/MATLAB Function', ...
        [modelName '/RippleRecon'], 'Position', [740, 210, 820, 300]);
    rt = sfroot;
    rippleChart = rt.find('-isa', 'Stateflow.EMChart', 'Path', [modelName '/RippleRecon']);
    rippleChart.Script = strjoin({
        'function iL = RippleRecon(iL_avg, delta_iL, duty, time)'
        '%#codegen'
        sprintf('Tsw = %.10e;', Tsw)
        'phase = mod(time, Tsw) / Tsw;'
        'duty_c = max(0.01, min(0.99, duty));'
        'if phase < duty_c'
        '    iL = iL_avg - delta_iL/2 + delta_iL * (phase / duty_c);'
        'else'
        '    iL = iL_avg + delta_iL/2 - delta_iL * ((phase - duty_c) / (1 - duty_c + eps));'
        'end'
    }, newline);

    % Wire top level
    add_line(modelName, 'FromWS_duty/1', [get_param(romName,'Name') '/1'], 'autorouting', 'smart');
    add_line(modelName, [get_param(romName,'Name') '/1'], 'Demux/1', 'autorouting', 'smart');
    add_line(modelName, 'Demux/1', 'Vout_Scope/1', 'autorouting', 'smart');
    add_line(modelName, 'Demux/2', 'iL_Scope/1', 'autorouting', 'smart');
    add_line(modelName, 'FromWS_duty/1', 'DeltaIL_Lookup/1', 'autorouting', 'smart');
    add_line(modelName, 'Demux/2', 'RT_iLavg/1', 'autorouting', 'smart');
    add_line(modelName, 'DeltaIL_Lookup/1', 'RT_deltaIL/1', 'autorouting', 'smart');
    add_line(modelName, 'FromWS_duty/1', 'RT_duty/1', 'autorouting', 'smart');
    add_line(modelName, 'RT_iLavg/1', 'RippleRecon/1', 'autorouting', 'smart');
    add_line(modelName, 'RT_deltaIL/1', 'RippleRecon/2', 'autorouting', 'smart');
    add_line(modelName, 'RT_duty/1', 'RippleRecon/3', 'autorouting', 'smart');
    add_line(modelName, 'DigitalClock/1', 'RippleRecon/4', 'autorouting', 'smart');
    add_line(modelName, 'RippleRecon/1', 'iL_Ripple_Scope/1', 'autorouting', 'smart');

    % Signal logging
    ph_vout = get_param([modelName '/Demux'], 'PortHandles');
    set_param(ph_vout.Outport(1), 'DataLogging', 'on', ...
        'DataLoggingName', 'Vout', 'DataLoggingNameMode', 'Custom');
    set_param(ph_vout.Outport(2), 'DataLogging', 'on', ...
        'DataLoggingName', 'iL_avg', 'DataLoggingNameMode', 'Custom');

    % iL with ripple reconstruction
    ph_ripple = get_param([modelName '/RippleRecon'], 'PortHandles');
    set_param(ph_ripple.Outport(1), 'DataLogging', 'on', ...
        'DataLoggingName', 'iL_ripple', 'DataLoggingNameMode', 'Custom');
end
