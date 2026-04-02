% collect_openloop_id_data.m
% Collects open-loop simulation data for linear system identification (ssest)
% at multiple operating points using the boost converter test harness.
%
% At each duty operating point, applies a small PRBS perturbation and records
% the Vout response. The perturbation data (du, dVout) is saved for ssest
% identification of the duty->Vout transfer function at each operating point.
%
% Output: data/boost_openloop_id_data.mat
%         - results: struct array with fields D_op, t, du, dVout, Vout_ss
%         - D_grid: duty grid vector
%         - Ts: sample time (5e-6)
%
% Usage: run this script, then use ssest on each operating point's data
%
% REPLICATION NOTE: Requires boost_converter_test_harness.slx (in simulink_models/).
% Pre-collected data is already provided in data/boost_openloop_id_data.mat.

thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fileparts(thisDir);
run(fullfile(repoRoot, 'startup.m'));

%% Configuration
Ts = 5e-6;              % Target sample time (5μs = switching period)
T_warmup = 0.010;       % 10ms DC warmup
T_prbs   = 0.040;       % 40ms PRBS perturbation
T_total  = T_warmup + T_prbs;  % 50ms total per operating point

prbs_amp   = 0.05;      % Perturbation amplitude (±0.05 duty, larger for better SNR)
prbs_order = 10;        % PRBS order (2^10 - 1 = 1023 bits)
prbs_hold  = 50e-6;     % 50μs per bit (hold time)
hold_samples = round(prbs_hold / Ts);  % samples per PRBS bit

D_grid = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85];

dataDir = fullfile(thisDir, 'data');
if ~exist(dataDir, 'dir'), mkdir(dataDir); end

%% Load the test harness
modelsDir = fullfile(repoRoot, 'simulink_models');
harnessModel = 'boost_converter_test_harness';
harnessFile = fullfile(modelsDir, [harnessModel '.slx']);

if ~exist(harnessFile, 'file')
    error('Test harness not found: %s\nRun the boost converter test first.', harnessFile);
end

load_system(harnessFile);
fprintf('Loaded harness: %s\n', harnessModel);

% Configure harness for simulation
set_param(harnessModel, 'SignalLogging', 'on');
set_param(harnessModel, 'ReturnWorkspaceOutputs', 'on');
set_param(harnessModel, 'SaveFormat', 'Dataset');
set_param(harnessModel, 'MaxStep', '0.5e-6');  % Resolve switching ripple

fprintf('=== Open-Loop ID Data Collection ===\n');
fprintf('Grid points: %d\n', numel(D_grid));
fprintf('Warmup: %.0f ms, PRBS: %.0f ms, Total: %.0f ms per point\n', ...
    T_warmup*1e3, T_prbs*1e3, T_total*1e3);
fprintf('PRBS: order=%d, amp=%.3f, hold=%.0fμs\n', prbs_order, prbs_amp, prbs_hold*1e6);

%% Generate PRBS sequence (same for all operating points)
rng(123);  % Reproducible
prbs_len = 2^prbs_order - 1;  % 1023 bits
prbs_raw = 2 * randi([0 1], prbs_len, 1) - 1;  % ±1 sequence
prbs_held = repelem(prbs_raw, hold_samples);  % Hold each bit for hold_samples

% Number of samples in PRBS phase
N_prbs = round(T_prbs / Ts);
% Tile PRBS to cover full duration
n_tiles = ceil(N_prbs / numel(prbs_held));
prbs_tiled = repmat(prbs_held, n_tiles, 1);
prbs_signal = prbs_tiled(1:N_prbs);  % Trim to exact length

%% Run simulations at each operating point
results = struct([]);

for i = 1:numel(D_grid)
    D_op = D_grid(i);
    fprintf('\n--- Operating point %d/%d: D = %.2f ---\n', i, numel(D_grid), D_op);

    % Build duty input: warmup (constant) + PRBS perturbation
    N_total = round(T_total / Ts) + 1;
    N_warmup = round(T_warmup / Ts);
    t = (0:Ts:T_total)';
    N_total = numel(t);

    u = D_op * ones(N_total, 1);
    % Apply PRBS after warmup
    idx_prbs_start = N_warmup + 1;
    idx_prbs_end = min(N_warmup + N_prbs, N_total);
    n_prbs_actual = idx_prbs_end - idx_prbs_start + 1;
    u(idx_prbs_start:idx_prbs_end) = D_op + prbs_amp * prbs_signal(1:n_prbs_actual);

    % Clamp to valid duty range
    u = max(0.05, min(0.90, u));

    % Set up simulation
    set_param(harnessModel, 'StopTime', num2str(T_total));
    u_ts = timeseries(u, t);
    assignin('base', 'duty_input', u_ts);

    % Run simulation
    tic;
    try
        simOut = sim(harnessModel);
    catch ME
        fprintf('  FAILED: %s\n', ME.message);
        continue;
    end
    t_sim = toc;

    % Extract Vout from logged signals
    logs = simOut.logsout;
    if isempty(logs)
        fprintf('  No logged signals!\n');
        continue;
    end

    vout_elem = logs.getElement('Vout');
    t_raw = squeeze(vout_elem.Values.Time);
    v_raw = squeeze(vout_elem.Values.Data);

    % Resample Vout to uniform Ts grid
    v_resamp = interp1(t_raw, v_raw, t, 'linear', 'extrap');

    % Measure steady-state from last 5ms of warmup
    idx_ss_start = round((T_warmup - 0.005) / Ts) + 1;
    idx_ss_end = N_warmup;
    if idx_ss_start < 1, idx_ss_start = 1; end
    Vout_ss = mean(v_resamp(idx_ss_start:idx_ss_end));

    % Extract perturbation region only (PRBS phase)
    t_id = t(idx_prbs_start:idx_prbs_end) - t(idx_prbs_start);  % Zero-based time
    du = u(idx_prbs_start:idx_prbs_end) - D_op;                  % Perturbation input
    dVout = v_resamp(idx_prbs_start:idx_prbs_end) - Vout_ss;     % Perturbation output

    % Store results
    results(i).D_op     = D_op;
    results(i).t        = t_id;
    results(i).du       = du;
    results(i).dVout    = dVout;
    results(i).Vout_ss  = Vout_ss;

    fprintf('  Sim time: %.1fs, Vout_ss=%.3fV, dVout range=[%.4f, %.4f]V\n', ...
        t_sim, Vout_ss, min(dVout), max(dVout));
    fprintf('  du range=[%.4f, %.4f], %d ID samples\n', ...
        min(du), max(du), numel(t_id));
end

%% Close model
close_system(harnessModel, 0);

%% Save results
matFile = fullfile(thisDir, 'data', 'boost_openloop_id_data.mat');
save(matFile, 'results', 'D_grid', 'Ts');

fprintf('\n=== Open-Loop ID Data Collection Summary ===\n');
fprintf('Operating points: %d\n', numel(D_grid));
fprintf('D_grid: [%s]\n', strjoin(arrayfun(@(x) sprintf('%.2f',x), D_grid, 'Uni', false), ', '));
fprintf('Sample time: %.0f μs\n', Ts*1e6);
fprintf('ID samples per point: %d (%.0f ms)\n', numel(results(1).t), T_prbs*1e3);
fprintf('Saved: %s\n', matFile);
fprintf('\nNext: use ssest on each results(i) to identify local LTI models.\n');
fprintf('Example:\n');
fprintf('  data_i = iddata(results(i).dVout, results(i).du, Ts);\n');
fprintf('  sys_i = ssest(data_i, 2);\n');

close all;
