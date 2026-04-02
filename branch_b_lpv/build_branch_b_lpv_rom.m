% build_branch_b_lpv_rom.m
% Build Branch B open-loop LPV ROM Simulink model with ripple reconstruction.
%
% Architecture (NO PID, NO closed loop):
%   duty (From Workspace) --> LPV_Vout_ROM --> Vout
%                         --> LPV_iL_ROM   --> iL_avg --+--> RippleRecon --> iL
%                         --> DeltaIL_Lookup -----------+
%
% Loads boost_frest_tf_data.mat (from frest_openloop_id.m) which contains:
%   tf_models, iL_models, D_grid, Vout_ss_grid, iL_ss_grid, nx, Ts
%
% Solver: ode1 at 0.5us (for ripple reconstruction rate)
% Model:  boost_openloop_branch_b

thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fileparts(thisDir);
run(fullfile(repoRoot, 'startup.m'));

modelsDir = fullfile(thisDir, 'models');

Ts_rom    = 5e-6;    % LPV ROM sample time (matches identification)
Ts_ripple = 0.5e-6;  % Ripple reconstruction rate
Tsw       = 5e-6;    % Switching period

%% 1. Load identification data
fprintf('=== Building Branch B Open-Loop LPV ROM ===\n');
matFile = fullfile(modelsDir, 'boost_frest_tf_data.mat');
if ~exist(matFile, 'file')
    error('Data file not found: %s\nRun frest_openloop_id.m first.', matFile);
end
data = load(matFile);

tf_models   = data.tf_models;
iL_models   = data.iL_models;
D_grid      = data.D_grid(:);
Vout_ss_grid = data.Vout_ss(:);
iL_ss_grid  = data.iL_ss_grid(:);
nx          = data.nx;
Ts          = data.Ts;

nGrid = numel(D_grid);
fprintf('Loaded: nx=%d, nGrid=%d, Ts=%gus\n', nx, nGrid, Ts*1e6);

%% 2. Extract A, B, C, D matrices into 3D arrays
fprintf('Extracting state-space matrices...\n');

% --- Vout ROM matrices ---
rom_A_vout = zeros(nx, nx, nGrid);
rom_B_vout = zeros(nx, 1,  nGrid);
rom_C_vout = zeros(1,  nx, nGrid);
rom_D_vout = zeros(1,  1,  nGrid);

for i = 1:nGrid
    if isempty(tf_models{i})
        warning('tf_models{%d} is empty (D=%.2f). Using zeros.', i, D_grid(i));
        continue;
    end
    sys_i = tf_models{i};
    % Convert to discrete state-space at ROM sample time
    if sys_i.Ts == 0
        sys_d = c2d(ss(sys_i), Ts_rom);
    elseif abs(sys_i.Ts - Ts_rom) > 1e-12
        sys_d = d2d(ss(sys_i), Ts_rom);
    else
        sys_d = ss(sys_i);
    end
    [A_i, B_i, C_i, D_i] = ssdata(sys_d);
    % Pad or truncate to nx states
    n_i = size(A_i, 1);
    if n_i >= nx
        rom_A_vout(:,:,i) = A_i(1:nx, 1:nx);
        rom_B_vout(:,:,i) = B_i(1:nx, 1);
        rom_C_vout(:,:,i) = C_i(1, 1:nx);
        rom_D_vout(:,:,i) = D_i(1, 1);
    else
        rom_A_vout(1:n_i, 1:n_i, i) = A_i;
        rom_B_vout(1:n_i, 1, i) = B_i(:, 1);
        rom_C_vout(1, 1:n_i, i) = C_i(1, :);
        rom_D_vout(1, 1, i) = D_i(1, 1);
    end
    fprintf('  Vout[%d] D=%.2f: %d states, DC=%.2f\n', i, D_grid(i), n_i, abs(dcgain(sys_d)));
end

% --- iL ROM matrices ---
rom_A_iL = zeros(nx, nx, nGrid);
rom_B_iL = zeros(nx, 1,  nGrid);
rom_C_iL = zeros(1,  nx, nGrid);
rom_D_iL = zeros(1,  1,  nGrid);

for i = 1:nGrid
    if isempty(iL_models{i})
        warning('iL_models{%d} is empty (D=%.2f). Using zeros.', i, D_grid(i));
        continue;
    end
    sys_i = iL_models{i};
    if sys_i.Ts == 0
        sys_d = c2d(ss(sys_i), Ts_rom);
    elseif abs(sys_i.Ts - Ts_rom) > 1e-12
        sys_d = d2d(ss(sys_i), Ts_rom);
    else
        sys_d = ss(sys_i);
    end
    [A_i, B_i, C_i, D_i] = ssdata(sys_d);
    n_i = size(A_i, 1);
    if n_i >= nx
        rom_A_iL(:,:,i) = A_i(1:nx, 1:nx);
        rom_B_iL(:,:,i) = B_i(1:nx, 1);
        rom_C_iL(:,:,i) = C_i(1, 1:nx);
        rom_D_iL(:,:,i) = D_i(1, 1);
    else
        rom_A_iL(1:n_i, 1:n_i, i) = A_i;
        rom_B_iL(1:n_i, 1, i) = B_i(:, 1);
        rom_C_iL(1, 1:n_i, i) = C_i(1, :);
        rom_D_iL(1, 1, i) = D_i(1, 1);
    end
    fprintf('  iL[%d]   D=%.2f: %d states, DC=%.2f\n', i, D_grid(i), n_i, abs(dcgain(sys_d)));
end

%% 3. Prepare delta_iL lookup table
% Analytical approximation: delta_iL = Vin * D * Tsw / L
Vin = 5;     % Input voltage
L   = 20e-6; % Inductor value
lut_duty     = linspace(0.05, 0.85, 17);
lut_delta_iL = Vin .* lut_duty .* Tsw ./ L;

fprintf('delta_iL lookup: %d points, range [%.3f, %.3f] A\n', ...
    numel(lut_duty), min(lut_delta_iL), max(lut_delta_iL));

%% 4. Prepare workspace variables for the model
rom_grid     = D_grid(:)';
rom_Vout_ss  = Vout_ss_grid(:)';
rom_iL_ss    = iL_ss_grid(:)';
rom_Ts       = Ts_rom;

% Initial conditions: [x_lpv(nx); p_ref] — start from zero state
rom_state_ic_vout = zeros(nx + 1, 1);
rom_state_ic_vout(end) = D_grid(1);  % p_ref = lowest grid point

rom_state_ic_iL = zeros(nx + 1, 1);
rom_state_ic_iL(end) = D_grid(1);

%% 5. Save all ROM data to a .mat file
romDataFile = fullfile(modelsDir, 'boost_openloop_branch_b_data.mat');
save(romDataFile, ...
    'rom_A_vout', 'rom_B_vout', 'rom_C_vout', 'rom_D_vout', ...
    'rom_A_iL',   'rom_B_iL',   'rom_C_iL',   'rom_D_iL', ...
    'rom_grid', 'rom_Vout_ss', 'rom_iL_ss', 'rom_Ts', 'nx', ...
    'rom_state_ic_vout', 'rom_state_ic_iL', ...
    'lut_duty', 'lut_delta_iL', 'Vin', 'L', 'Tsw', ...
    'D_grid', 'Vout_ss_grid', 'iL_ss_grid');
fprintf('ROM data saved: %s\n\n', romDataFile);

%% 6. Build Simulink model
modelName = 'boost_openloop_branch_b';
stopTime  = 0.050;  % 50ms default

if bdIsLoaded(modelName), close_system(modelName, 0); end
slxcFile = fullfile(modelsDir, [modelName '.slxc']);
if exist(slxcFile, 'file'), delete(slxcFile); end

new_system(modelName);
open_system(modelName);

% Solver: ode1 at 0.5us
set_param(modelName, 'Solver', 'ode1');
set_param(modelName, 'FixedStep', num2str(Ts_ripple));
set_param(modelName, 'StopTime', num2str(stopTime));
set_param(modelName, 'SignalLogging', 'on');
set_param(modelName, 'SaveOutput', 'on');
set_param(modelName, 'ReturnWorkspaceOutputs', 'on');
set_param(modelName, 'SaveFormat', 'Dataset');

%% --- Input: duty Step block (user sets Before/After values) ---
% Step time = 0.02s to let initial transient settle before step
add_block('simulink/Sources/Step', [modelName '/Step'], ...
    'Position', [30, 150, 80, 180], ...
    'Time', '0.02', ...
    'Before', '0.3', ...
    'After', '0.5', ...
    'SampleTime', '0');

%% ================================================================
%  LPV Vout ROM Subsystem
%  ================================================================
voutSub = [modelName '/LPV_Vout_ROM'];
add_block('simulink/Ports & Subsystems/Subsystem', voutSub, ...
    'Position', [250, 50, 430, 120]);
set_param(voutSub, 'TreatAsAtomicUnit', 'on');
set_param(voutSub, 'MinAlgLoopOccurrences', 'on');
set_param(voutSub, 'SystemSampleTime', num2str(Ts_rom));

% Remove default wiring
delete_line(voutSub, 'In1/1', 'Out1/1');

% --- OutputFcn: state -> Vout ---
outFcnPath_v = [voutSub '/OutputFcn'];
add_block('simulink/User-Defined Functions/MATLAB Function', outFcnPath_v, ...
    'Position', [300, 30, 450, 70]);

rt = sfroot;
outChart_v = rt.find('-isa', 'Stateflow.EMChart', 'Path', outFcnPath_v);

outLines_v = {
    'function Vout = OutputFcn(x_state)'
    '%#codegen'
    '% Compute Vout from LPV state. No duty input -> no direct feedthrough.'
    ''
    'nx = size(rom_A_vout, 1);'
    'n_pts = size(rom_A_vout, 3);'
    ''
    'x_lpv = x_state(1:nx);'
    'p_ref = x_state(nx+1);'
    ''
    '% Clamp p_ref to grid range'
    'p_c = max(rom_grid(1), min(rom_grid(n_pts), p_ref));'
    ''
    '% Find interpolation index (codegen-compatible loop)'
    'idx = 1;'
    'for k = 1:n_pts'
    '    if rom_grid(k) <= p_c'
    '        idx = k;'
    '    end'
    'end'
    'if idx >= n_pts'
    '    idx = n_pts - 1;'
    'end'
    ''
    'alph = (p_c - rom_grid(idx)) / (rom_grid(idx+1) - rom_grid(idx));'
    'C_k = (1 - alph) * rom_C_vout(:,:,idx) + alph * rom_C_vout(:,:,idx+1);'
    'y_base = (1 - alph) * rom_Vout_ss(idx) + alph * rom_Vout_ss(idx+1);'
    'Vout = y_base + C_k * x_lpv;'
};
outChart_v.Script = strjoin(outLines_v, newline);

for pName = {'rom_A_vout', 'rom_C_vout', 'rom_grid', 'rom_Vout_ss'}
    d = Stateflow.Data(outChart_v);
    d.Name = pName{1};
    d.Scope = 'Parameter';
end

% --- UpdateFcn: duty + state -> next state ---
updFcnPath_v = [voutSub '/UpdateFcn'];
add_block('simulink/User-Defined Functions/MATLAB Function', updFcnPath_v, ...
    'Position', [130, 120, 280, 170]);

updChart_v = rt.find('-isa', 'Stateflow.EMChart', 'Path', updFcnPath_v);

updLines_v = {
    'function x_next = UpdateFcn(duty, x_state)'
    '%#codegen'
    '% LPV state update with slowly-tracking reference filter.'
    ''
    'nx = size(rom_A_vout, 1);'
    'n_pts = size(rom_A_vout, 3);'
    ''
    'x_lpv = x_state(1:nx);'
    'p_ref = x_state(nx+1);'
    ''
    '% Clamp duty to grid range'
    'duty_c = max(rom_grid(1), min(rom_grid(n_pts), duty));'
    ''
    '% Slowly-tracking reference filter (tau = 0.5ms)'
    'tau_ref = 50e-6;  % Fast tracking (1 switching period) — minimal phase lag'
    'alpha_ref = rom_Ts / (tau_ref + rom_Ts);'
    'p_new = (1 - alpha_ref) * p_ref + alpha_ref * duty_c;'
    'p_new = max(rom_grid(1), min(rom_grid(n_pts), p_new));'
    ''
    '% Find interpolation index (codegen-compatible loop)'
    'idx = 1;'
    'for k = 1:n_pts'
    '    if rom_grid(k) <= p_new'
    '        idx = k;'
    '    end'
    'end'
    'if idx >= n_pts'
    '    idx = n_pts - 1;'
    'end'
    'alph = (p_new - rom_grid(idx)) / (rom_grid(idx+1) - rom_grid(idx));'
    ''
    'A_k = (1 - alph) * rom_A_vout(:,:,idx) + alph * rom_A_vout(:,:,idx+1);'
    'B_k = (1 - alph) * rom_B_vout(:,:,idx) + alph * rom_B_vout(:,:,idx+1);'
    ''
    '% Perturbation input'
    'du = duty_c - p_new;'
    'x_lpv_new = A_k * x_lpv + B_k * du;'
    ''
    '% Pack next state'
    'x_next = [x_lpv_new; p_new];'
};
updChart_v.Script = strjoin(updLines_v, newline);

for pName = {'rom_A_vout', 'rom_B_vout', 'rom_grid', 'rom_Ts'}
    d = Stateflow.Data(updChart_v);
    d.Name = pName{1};
    d.Scope = 'Parameter';
end

% --- Unit Delay for state ---
udPath_v = [voutSub '/StateDelay'];
add_block('simulink/Discrete/Unit Delay', udPath_v, ...
    'Position', [300, 120, 360, 160], ...
    'SampleTime', num2str(Ts_rom), ...
    'InitialCondition', 'rom_state_ic_vout');

% --- Wire Vout subsystem ---
add_line(voutSub, 'In1/1', 'UpdateFcn/1', 'autorouting', 'smart');
add_line(voutSub, 'StateDelay/1', 'OutputFcn/1', 'autorouting', 'smart');
add_line(voutSub, 'StateDelay/1', 'UpdateFcn/2', 'autorouting', 'smart');
add_line(voutSub, 'UpdateFcn/1', 'StateDelay/1', 'autorouting', 'smart');
add_line(voutSub, 'OutputFcn/1', 'Out1/1', 'autorouting', 'smart');

%% ================================================================
%  LPV iL ROM Subsystem
%  ================================================================
iLSub = [modelName '/LPV_iL_ROM'];
add_block('simulink/Ports & Subsystems/Subsystem', iLSub, ...
    'Position', [250, 180, 430, 250]);
set_param(iLSub, 'TreatAsAtomicUnit', 'on');
set_param(iLSub, 'MinAlgLoopOccurrences', 'on');
set_param(iLSub, 'SystemSampleTime', num2str(Ts_rom));

% Remove default wiring
delete_line(iLSub, 'In1/1', 'Out1/1');

% --- OutputFcn: state -> iL_avg ---
outFcnPath_iL = [iLSub '/OutputFcn'];
add_block('simulink/User-Defined Functions/MATLAB Function', outFcnPath_iL, ...
    'Position', [300, 30, 450, 70]);

outChart_iL = rt.find('-isa', 'Stateflow.EMChart', 'Path', outFcnPath_iL);

outLines_iL = {
    'function iL_avg = OutputFcn(x_state)'
    '%#codegen'
    '% Compute iL_avg from LPV state. No duty input -> no direct feedthrough.'
    ''
    'nx = size(rom_A_iL, 1);'
    'n_pts = size(rom_A_iL, 3);'
    ''
    'x_lpv = x_state(1:nx);'
    'p_ref = x_state(nx+1);'
    ''
    '% Clamp p_ref to grid range'
    'p_c = max(rom_grid(1), min(rom_grid(n_pts), p_ref));'
    ''
    '% Find interpolation index (codegen-compatible loop)'
    'idx = 1;'
    'for k = 1:n_pts'
    '    if rom_grid(k) <= p_c'
    '        idx = k;'
    '    end'
    'end'
    'if idx >= n_pts'
    '    idx = n_pts - 1;'
    'end'
    ''
    'alph = (p_c - rom_grid(idx)) / (rom_grid(idx+1) - rom_grid(idx));'
    'C_k = (1 - alph) * rom_C_iL(:,:,idx) + alph * rom_C_iL(:,:,idx+1);'
    'y_base = (1 - alph) * rom_iL_ss(idx) + alph * rom_iL_ss(idx+1);'
    'iL_avg = y_base + C_k * x_lpv;'
};
outChart_iL.Script = strjoin(outLines_iL, newline);

for pName = {'rom_A_iL', 'rom_C_iL', 'rom_grid', 'rom_iL_ss'}
    d = Stateflow.Data(outChart_iL);
    d.Name = pName{1};
    d.Scope = 'Parameter';
end

% --- UpdateFcn: duty + state -> next state ---
updFcnPath_iL = [iLSub '/UpdateFcn'];
add_block('simulink/User-Defined Functions/MATLAB Function', updFcnPath_iL, ...
    'Position', [130, 120, 280, 170]);

updChart_iL = rt.find('-isa', 'Stateflow.EMChart', 'Path', updFcnPath_iL);

updLines_iL = {
    'function x_next = UpdateFcn(duty, x_state)'
    '%#codegen'
    '% LPV state update with slowly-tracking reference filter.'
    ''
    'nx = size(rom_A_iL, 1);'
    'n_pts = size(rom_A_iL, 3);'
    ''
    'x_lpv = x_state(1:nx);'
    'p_ref = x_state(nx+1);'
    ''
    '% Clamp duty to grid range'
    'duty_c = max(rom_grid(1), min(rom_grid(n_pts), duty));'
    ''
    '% Slowly-tracking reference filter (tau = 0.5ms)'
    'tau_ref = 50e-6;  % Fast tracking (1 switching period) — minimal phase lag'
    'alpha_ref = rom_Ts / (tau_ref + rom_Ts);'
    'p_new = (1 - alpha_ref) * p_ref + alpha_ref * duty_c;'
    'p_new = max(rom_grid(1), min(rom_grid(n_pts), p_new));'
    ''
    '% Find interpolation index (codegen-compatible loop)'
    'idx = 1;'
    'for k = 1:n_pts'
    '    if rom_grid(k) <= p_new'
    '        idx = k;'
    '    end'
    'end'
    'if idx >= n_pts'
    '    idx = n_pts - 1;'
    'end'
    'alph = (p_new - rom_grid(idx)) / (rom_grid(idx+1) - rom_grid(idx));'
    ''
    'A_k = (1 - alph) * rom_A_iL(:,:,idx) + alph * rom_A_iL(:,:,idx+1);'
    'B_k = (1 - alph) * rom_B_iL(:,:,idx) + alph * rom_B_iL(:,:,idx+1);'
    ''
    '% Perturbation input'
    'du = duty_c - p_new;'
    'x_lpv_new = A_k * x_lpv + B_k * du;'
    ''
    '% Pack next state'
    'x_next = [x_lpv_new; p_new];'
};
updChart_iL.Script = strjoin(updLines_iL, newline);

for pName = {'rom_A_iL', 'rom_B_iL', 'rom_grid', 'rom_Ts'}
    d = Stateflow.Data(updChart_iL);
    d.Name = pName{1};
    d.Scope = 'Parameter';
end

% --- Unit Delay for state ---
udPath_iL = [iLSub '/StateDelay'];
add_block('simulink/Discrete/Unit Delay', udPath_iL, ...
    'Position', [300, 120, 360, 160], ...
    'SampleTime', num2str(Ts_rom), ...
    'InitialCondition', 'rom_state_ic_iL');

% --- Wire iL subsystem ---
add_line(iLSub, 'In1/1', 'UpdateFcn/1', 'autorouting', 'smart');
add_line(iLSub, 'StateDelay/1', 'OutputFcn/1', 'autorouting', 'smart');
add_line(iLSub, 'StateDelay/1', 'UpdateFcn/2', 'autorouting', 'smart');
add_line(iLSub, 'UpdateFcn/1', 'StateDelay/1', 'autorouting', 'smart');
add_line(iLSub, 'OutputFcn/1', 'Out1/1', 'autorouting', 'smart');

%% ================================================================
%  Delta_iL Lookup Table
%  ================================================================
add_block('simulink/Lookup Tables/1-D Lookup Table', ...
    [modelName '/DeltaIL_Lookup'], ...
    'Position', [250, 310, 350, 350], ...
    'Table', 'lut_delta_iL', ...
    'BreakpointsForDimension1', 'lut_duty', ...
    'InterpMethod', 'Linear point-slope', ...
    'ExtrapMethod', 'Clip');

%% ================================================================
%  Rate Transitions: 5us -> 0.5us
%  ================================================================
add_block('simulink/Signal Attributes/Rate Transition', ...
    [modelName '/RT_iLavg'], ...
    'Position', [500, 195, 550, 225], ...
    'InitialCondition', '0', 'OutPortSampleTime', num2str(Ts_ripple));

add_block('simulink/Signal Attributes/Rate Transition', ...
    [modelName '/RT_deltaIL'], ...
    'Position', [500, 315, 550, 345], ...
    'InitialCondition', '0', 'OutPortSampleTime', num2str(Ts_ripple));

add_block('simulink/Signal Attributes/Rate Transition', ...
    [modelName '/RT_duty'], ...
    'Position', [500, 400, 550, 430], ...
    'InitialCondition', '0', 'OutPortSampleTime', num2str(Ts_ripple));

%% ================================================================
%  Digital Clock at 0.5us
%  ================================================================
add_block('simulink/Sources/Digital Clock', [modelName '/DigitalClock'], ...
    'Position', [580, 470, 640, 500], ...
    'SampleTime', num2str(Ts_ripple));

%% ================================================================
%  Ripple Reconstruction: MATLAB Function block at 0.5us
%  ================================================================
add_block('simulink/User-Defined Functions/MATLAB Function', ...
    [modelName '/RippleRecon'], ...
    'Position', [650, 300, 810, 400]);

rippleChart = rt.find('-isa', 'Stateflow.EMChart', 'Path', [modelName '/RippleRecon']);

rippleLines = {
    'function iL = RippleRecon(iL_avg, delta_iL, duty, time)'
    '%#codegen'
    '% Reconstruct triangular iL waveform from average and ripple amplitude.'
    '% Runs at 0.5us to capture switching ripple within each 5us PWM period.'
    ''
    'Ts_pwm = 5e-6;'
    ''
    '% Phase within current PWM period [0, 1)'
    'phase = mod(time, Ts_pwm) / Ts_pwm;'
    ''
    '% Clamp duty to avoid division by zero'
    'duty_c = max(0.01, min(0.99, duty));'
    ''
    'if phase < duty_c'
    '    % Rising portion: current ramps up during on-time'
    '    iL = iL_avg - delta_iL/2 + delta_iL * (phase / duty_c);'
    'else'
    '    % Falling portion: current ramps down during off-time'
    '    iL = iL_avg + delta_iL/2 - delta_iL * ((phase - duty_c) / (1 - duty_c + eps));'
    'end'
};
rippleChart.Script = strjoin(rippleLines, newline);

%% ================================================================
%  Output To Workspace blocks (for easy extraction)
%  ================================================================
add_block('simulink/Sinks/To Workspace', [modelName '/ToWS_Vout'], ...
    'Position', [500, 65, 580, 95], ...
    'VariableName', 'Vout_out', 'SampleTime', num2str(Ts_rom));

add_block('simulink/Sinks/To Workspace', [modelName '/ToWS_iLavg'], ...
    'Position', [870, 195, 950, 225], ...
    'VariableName', 'iLavg_out', 'SampleTime', num2str(Ts_rom));

add_block('simulink/Sinks/To Workspace', [modelName '/ToWS_iL'], ...
    'Position', [870, 335, 950, 365], ...
    'VariableName', 'iL_out', 'SampleTime', num2str(Ts_ripple));

add_block('simulink/Sinks/To Workspace', [modelName '/ToWS_duty'], ...
    'Position', [870, 405, 950, 435], ...
    'VariableName', 'duty_out', 'SampleTime', num2str(Ts_rom));

%% ================================================================
%  Wire top-level connections
%  ================================================================

% duty -> LPV Vout ROM
add_line(modelName, 'Step/1', 'LPV_Vout_ROM/1', 'autorouting', 'smart');

% duty -> LPV iL ROM
add_line(modelName, 'Step/1', 'LPV_iL_ROM/1', 'autorouting', 'smart');

% duty -> DeltaIL Lookup
add_line(modelName, 'Step/1', 'DeltaIL_Lookup/1', 'autorouting', 'smart');

% duty -> RT_duty (for ripple recon)
add_line(modelName, 'Step/1', 'RT_duty/1', 'autorouting', 'smart');

% LPV_Vout_ROM -> ToWS_Vout
add_line(modelName, 'LPV_Vout_ROM/1', 'ToWS_Vout/1', 'autorouting', 'smart');

% LPV_iL_ROM -> RT_iLavg
add_line(modelName, 'LPV_iL_ROM/1', 'RT_iLavg/1', 'autorouting', 'smart');

% DeltaIL_Lookup -> RT_deltaIL
add_line(modelName, 'DeltaIL_Lookup/1', 'RT_deltaIL/1', 'autorouting', 'smart');

% RT outputs -> RippleRecon inputs
add_line(modelName, 'RT_iLavg/1', 'RippleRecon/1', 'autorouting', 'smart');    % iL_avg
add_line(modelName, 'RT_deltaIL/1', 'RippleRecon/2', 'autorouting', 'smart');  % delta_iL
add_line(modelName, 'RT_duty/1', 'RippleRecon/3', 'autorouting', 'smart');     % duty
add_line(modelName, 'DigitalClock/1', 'RippleRecon/4', 'autorouting', 'smart'); % time

% To Workspace connections
add_line(modelName, 'RT_iLavg/1', 'ToWS_iLavg/1', 'autorouting', 'smart');
add_line(modelName, 'RippleRecon/1', 'ToWS_iL/1', 'autorouting', 'smart');
add_line(modelName, 'RT_duty/1', 'ToWS_duty/1', 'autorouting', 'smart');

%% ================================================================
%  Signal logging on key outputs
%  ================================================================
ph_vout = get_param([modelName '/LPV_Vout_ROM'], 'PortHandles');
set_param(ph_vout.Outport(1), 'DataLogging', 'on', ...
    'DataLoggingName', 'Vout', 'DataLoggingNameMode', 'Custom');

ph_ilavg = get_param([modelName '/LPV_iL_ROM'], 'PortHandles');
set_param(ph_ilavg.Outport(1), 'DataLogging', 'on', ...
    'DataLoggingName', 'iL_avg', 'DataLoggingNameMode', 'Custom');

ph_il = get_param([modelName '/RippleRecon'], 'PortHandles');
set_param(ph_il.Outport(1), 'DataLogging', 'on', ...
    'DataLoggingName', 'iL', 'DataLoggingNameMode', 'Custom');

ph_duty = get_param([modelName '/Step'], 'PortHandles');
set_param(ph_duty.Outport(1), 'DataLogging', 'on', ...
    'DataLoggingName', 'duty_cycle', 'DataLoggingNameMode', 'Custom');

%% ================================================================
%  PreLoadFcn: load ROM data at model load time
%  ================================================================
preloadCmd = sprintf('load(''%s'');', strrep(romDataFile, '\', '/'));
set_param(modelName, 'PreLoadFcn', preloadCmd);

%% ================================================================
%  Arrange and save
%  ================================================================
Simulink.BlockDiagram.arrangeSystem(modelName);
modelPath = fullfile(modelsDir, [modelName '.slx']);
save_system(modelName, modelPath);

fprintf('\n=== Branch B Open-Loop LPV ROM Built ===\n');
fprintf('  Model:       %s\n', modelPath);
fprintf('  Vout ROM:    LPV (nx=%d, %d grid points) @ %gus\n', nx, nGrid, Ts_rom*1e6);
fprintf('  iL ROM:      LPV (nx=%d, %d grid points) @ %gus\n', nx, nGrid, Ts_rom*1e6);
fprintf('  delta_iL:    1-D Lookup (%d points)\n', numel(lut_duty));
fprintf('  Ripple:      triangular reconstruction @ %gus\n', Ts_ripple*1e6);
fprintf('  Solver:      ode1 @ %gus\n', Ts_ripple*1e6);
fprintf('  Data file:   %s\n', romDataFile);
fprintf('=== Done ===\n');

close all;
