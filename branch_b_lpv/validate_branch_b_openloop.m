% validate_branch_b_openloop.m
% Validate the Branch B open-loop LPV ROM against the switching model
% across the operating envelope.
%
% Runs open-loop duty step profiles through both:
%   1. Test harness (full switching model) — ground truth
%   2. LPV ROM (boost_openloop_branch_b) — Vout, iL_avg, iL with ripple
%
% Compares Vout RMSE, iL_avg RMSE, and iL ripple peak-to-peak amplitude.
% No PID, no closed loop — pure plant model accuracy test.

thisDir    = fileparts(mfilename('fullpath'));
repoRoot   = fileparts(thisDir);
run(fullfile(repoRoot, 'startup.m'));

modelsDir  = fullfile(thisDir, 'models');       % LPV ROM data
harnessDir = fullfile(repoRoot, 'simulink_models');  % Simscape test harness

Ts     = 5e-6;    % ROM sample time
Ts_il  = 0.5e-6;  % Ripple reconstruction rate
Tsw    = 5e-6;    % Switching period

%% 1. Load LPV ROM data
fprintf('=== Branch B Open-Loop LPV Validation ===\n\n');

romDataFile = fullfile(modelsDir, 'boost_openloop_branch_b_data.mat');
if ~exist(romDataFile, 'file')
    % Try loading from frest data and building
    fprintf('ROM data not found. Run build_branch_b_lpv_rom.m first.\n');
    error('Missing: %s', romDataFile);
end
romData = load(romDataFile);
fprintf('Loaded ROM data: nx=%d, nGrid=%d\n', romData.nx, numel(romData.rom_grid));

%% 2. Define test profiles
T_settle = 0.015;  % 15ms settle at initial duty
T_step   = 0.015;  % 15ms after step
T_total  = T_settle + T_step;

profiles = struct([]);

% Small steps
profiles(end+1).name = 'D=0.25->0.30';
profiles(end).D_init = 0.25; profiles(end).D_final = 0.30;

profiles(end+1).name = 'D=0.45->0.50';
profiles(end).D_init = 0.45; profiles(end).D_final = 0.50;

% Medium steps
profiles(end+1).name = 'D=0.20->0.40';
profiles(end).D_init = 0.20; profiles(end).D_final = 0.40;

profiles(end+1).name = 'D=0.50->0.65';
profiles(end).D_init = 0.50; profiles(end).D_final = 0.65;

% Large steps
profiles(end+1).name = 'D=0.20->0.60';
profiles(end).D_init = 0.20; profiles(end).D_final = 0.60;

profiles(end+1).name = 'D=0.30->0.65';
profiles(end).D_init = 0.30; profiles(end).D_final = 0.65;

% Step downs
profiles(end+1).name = 'D=0.50->0.35';
profiles(end).D_init = 0.50; profiles(end).D_final = 0.35;

profiles(end+1).name = 'D=0.65->0.40';
profiles(end).D_init = 0.65; profiles(end).D_final = 0.40;

nProfiles = numel(profiles);
fprintf('Test profiles: %d\n\n', nProfiles);

%% 3. Load test harness (switching model)
harnessModel = 'boost_converter_test_harness';
harnessFile  = fullfile(harnessDir, [harnessModel '.slx']);
if ~exist(harnessFile, 'file')
    error('Test harness not found: %s', harnessFile);
end
load_system(harnessFile);
set_param(harnessModel, 'MaxStep', '0.5e-6');
set_param(harnessModel, 'ReturnWorkspaceOutputs', 'on');
set_param(harnessModel, 'SaveFormat', 'Dataset');
set_param(harnessModel, 'SignalLogging', 'on');
set_param(harnessModel, 'SimscapeLogType', 'all');
set_param(harnessModel, 'StopTime', num2str(T_total));

%% 4. Butterworth filter for averaging switching-model outputs
[b_filt, a_filt] = butter(3, 30e3 / (1/Ts/2));

%% 5. Prepare LPV forward-pass parameters
nx       = romData.nx;
rom_grid = romData.rom_grid(:)';
nGrid    = numel(rom_grid);
tau_ref  = 0.5e-3;
alpha_ref = Ts / (tau_ref + Ts);

% Ripple parameters
Vin = romData.Vin;
L   = romData.L;

%% 6. Run each profile
results = struct([]);

for pi = 1:nProfiles
    prof = profiles(pi);
    fprintf('--- [%d/%d] %s ---\n', pi, nProfiles, prof.name);

    % Build duty profile
    t_vec = (0:Ts:T_total)';
    N = numel(t_vec);
    n_settle = round(T_settle / Ts);

    duty_vec = prof.D_init * ones(N, 1);
    duty_vec(n_settle+1:end) = prof.D_final;
    duty_input = timeseries(duty_vec, t_vec);

    %% 6a. Run switching model (test harness)
    assignin('base', 'duty_input', duty_input);
    simOut = sim(harnessModel);

    % Extract Vout
    vout_ts = simOut.logsout.getElement('Vout').Values;
    vout_sw = interp1(squeeze(vout_ts.Time), squeeze(vout_ts.Data), t_vec, 'linear', 'extrap');

    % Extract iL from Simscape log
    iL_sw = zeros(N, 1);
    flds = simOut.who;
    for fi = 1:numel(flds)
        if startsWith(flds{fi}, 'simlog')
            sl = simOut.(flds{fi});
            bc = sl.Boost_Converter.Boost_Converter_Circuit;
            fn = fieldnames(bc);
            for fj = 1:numel(fn)
                if startsWith(fn{fj}, 'L1')
                    L1node = bc.(fn{fj});
                    try
                        iL_raw = L1node.i_L.series;
                    catch
                        iL_raw = L1node.i.series;
                    end
                    try
                        iL_vals = iL_raw.values('A');
                    catch
                        iL_vals = double(iL_raw.values);
                    end
                    t_iL = iL_raw.time;
                    iL_vals = double(iL_vals);
                    % Only interpolate within data range
                    valid = t_vec >= t_iL(1) & t_vec <= t_iL(end);
                    iL_sw = zeros(size(t_vec));
                    iL_sw(valid) = interp1(t_iL, iL_vals, t_vec(valid), 'linear');
                    iL_sw(~valid & t_vec < t_iL(1)) = iL_vals(1);
                    iL_sw(~valid & t_vec > t_iL(end)) = iL_vals(end);
                    break;
                end
            end
            break;
        end
    end

    % Filter to get average values
    vout_sw_filt = filtfilt(b_filt, a_filt, vout_sw);
    iL_sw_filt   = filtfilt(b_filt, a_filt, iL_sw);

    % Measure switching-model ripple peak-to-peak in last 5ms
    tail_mask = t_vec >= (T_total - 0.005);
    iL_sw_pp  = max(iL_sw(tail_mask)) - min(iL_sw(tail_mask));

    fprintf('  Switching: Vout=[%.2f,%.2f]V, iL_avg=[%.2f,%.2f]A, iL_pp=%.3fA\n', ...
        min(vout_sw_filt), max(vout_sw_filt), ...
        min(iL_sw_filt), max(iL_sw_filt), iL_sw_pp);

    %% 6b. Run LPV ROM (MATLAB forward pass — avoids Simulink overhead)
    % --- Vout LPV ---
    x_v = zeros(nx, 1);
    p_ref_v = prof.D_init;
    vout_rom = zeros(N, 1);

    for k = 1:N
        d = duty_vec(k);
        d_c = max(rom_grid(1), min(rom_grid(end), d));

        % Output first (from previous state)
        p_c = max(rom_grid(1), min(rom_grid(end), p_ref_v));
        idx = 1;
        for gi = 1:nGrid
            if rom_grid(gi) <= p_c, idx = gi; end
        end
        idx = min(idx, nGrid - 1);
        alph = (p_c - rom_grid(idx)) / (rom_grid(idx+1) - rom_grid(idx));
        alph = max(0, min(1, alph));

        C_k = (1-alph)*romData.rom_C_vout(:,:,idx) + alph*romData.rom_C_vout(:,:,idx+1);
        y_eq = (1-alph)*romData.rom_Vout_ss(idx)   + alph*romData.rom_Vout_ss(idx+1);
        vout_rom(k) = y_eq + C_k * x_v;

        % State update
        p_new = (1 - alpha_ref) * p_ref_v + alpha_ref * d_c;
        p_new = max(rom_grid(1), min(rom_grid(end), p_new));

        idx2 = 1;
        for gi = 1:nGrid
            if rom_grid(gi) <= p_new, idx2 = gi; end
        end
        idx2 = min(idx2, nGrid - 1);
        alph2 = (p_new - rom_grid(idx2)) / (rom_grid(idx2+1) - rom_grid(idx2));
        alph2 = max(0, min(1, alph2));

        A_k = (1-alph2)*romData.rom_A_vout(:,:,idx2) + alph2*romData.rom_A_vout(:,:,idx2+1);
        B_k = (1-alph2)*romData.rom_B_vout(:,:,idx2) + alph2*romData.rom_B_vout(:,:,idx2+1);
        du = d_c - p_new;
        x_v = A_k * x_v + B_k * du;
        p_ref_v = p_new;
    end

    % --- iL LPV ---
    x_i = zeros(nx, 1);
    p_ref_i = prof.D_init;
    iL_avg_rom = zeros(N, 1);

    for k = 1:N
        d = duty_vec(k);
        d_c = max(rom_grid(1), min(rom_grid(end), d));

        % Output
        p_c = max(rom_grid(1), min(rom_grid(end), p_ref_i));
        idx = 1;
        for gi = 1:nGrid
            if rom_grid(gi) <= p_c, idx = gi; end
        end
        idx = min(idx, nGrid - 1);
        alph = (p_c - rom_grid(idx)) / (rom_grid(idx+1) - rom_grid(idx));
        alph = max(0, min(1, alph));

        C_k = (1-alph)*romData.rom_C_iL(:,:,idx) + alph*romData.rom_C_iL(:,:,idx+1);
        y_eq = (1-alph)*romData.rom_iL_ss(idx)   + alph*romData.rom_iL_ss(idx+1);
        iL_avg_rom(k) = y_eq + C_k * x_i;

        % State update
        p_new = (1 - alpha_ref) * p_ref_i + alpha_ref * d_c;
        p_new = max(rom_grid(1), min(rom_grid(end), p_new));

        idx2 = 1;
        for gi = 1:nGrid
            if rom_grid(gi) <= p_new, idx2 = gi; end
        end
        idx2 = min(idx2, nGrid - 1);
        alph2 = (p_new - rom_grid(idx2)) / (rom_grid(idx2+1) - rom_grid(idx2));
        alph2 = max(0, min(1, alph2));

        A_k = (1-alph2)*romData.rom_A_iL(:,:,idx2) + alph2*romData.rom_A_iL(:,:,idx2+1);
        B_k = (1-alph2)*romData.rom_B_iL(:,:,idx2) + alph2*romData.rom_B_iL(:,:,idx2+1);
        du = d_c - p_new;
        x_i = A_k * x_i + B_k * du;
        p_ref_i = p_new;
    end

    % --- Ripple reconstruction ---
    % Compute at high rate for the last 5ms only (for peak-to-peak comparison)
    t_hr = (T_total - 0.005):Ts_il:T_total;
    t_hr = t_hr(:);
    duty_hr = prof.D_final;  % constant after step in tail
    delta_iL_final = Vin * max(0.01, min(0.99, duty_hr)) * Tsw / L;
    iL_avg_final = iL_avg_rom(end);  % use last value

    phase_hr = mod(t_hr, Tsw) / Tsw;
    duty_c = max(0.01, min(0.99, duty_hr));

    iL_ripple = zeros(size(t_hr));
    for ki = 1:numel(t_hr)
        if phase_hr(ki) < duty_c
            iL_ripple(ki) = iL_avg_final - delta_iL_final/2 + ...
                delta_iL_final * (phase_hr(ki) / duty_c);
        else
            iL_ripple(ki) = iL_avg_final + delta_iL_final/2 - ...
                delta_iL_final * ((phase_hr(ki) - duty_c) / (1 - duty_c + eps));
        end
    end
    iL_rom_pp = max(iL_ripple) - min(iL_ripple);

    %% 6c. Compute metrics (after step only, skip first 1ms for transient)
    eval_mask = t_vec >= (T_settle + 0.001);

    vout_rmse = sqrt(mean((vout_rom(eval_mask) - vout_sw_filt(eval_mask)).^2));
    iL_rmse   = sqrt(mean((iL_avg_rom(eval_mask) - iL_sw_filt(eval_mask)).^2));

    pp_error = abs(iL_rom_pp - iL_sw_pp);
    pp_error_pct = pp_error / max(iL_sw_pp, 1e-6) * 100;

    fprintf('  LPV ROM:  Vout RMSE=%.4fV, iL_avg RMSE=%.4fA\n', vout_rmse, iL_rmse);
    fprintf('  Ripple:   ROM pp=%.3fA, SW pp=%.3fA, error=%.1f%%\n', ...
        iL_rom_pp, iL_sw_pp, pp_error_pct);

    %% Store results
    results(pi).name       = prof.name;
    results(pi).D_init     = prof.D_init;
    results(pi).D_final    = prof.D_final;
    results(pi).t          = t_vec;
    results(pi).duty       = duty_vec;
    results(pi).vout_sw    = vout_sw_filt;
    results(pi).vout_rom   = vout_rom;
    results(pi).iL_sw      = iL_sw_filt;
    results(pi).iL_avg_rom = iL_avg_rom;
    results(pi).iL_sw_raw  = iL_sw;
    results(pi).iL_ripple_t  = t_hr;
    results(pi).iL_ripple    = iL_ripple;
    results(pi).vout_rmse  = vout_rmse;
    results(pi).iL_rmse    = iL_rmse;
    results(pi).iL_sw_pp   = iL_sw_pp;
    results(pi).iL_rom_pp  = iL_rom_pp;
    results(pi).pp_error_pct = pp_error_pct;
end

close_system(harnessModel, 0);

%% 7. Summary table
fprintf('\n=== OPEN-LOOP LPV VALIDATION SUMMARY ===\n');
fprintf('%-16s | %10s %10s | %10s %10s %10s\n', ...
    'Profile', 'Vout RMSE', 'iL RMSE', 'SW pp(A)', 'ROM pp(A)', 'pp err%%');
fprintf('%s\n', repmat('-', 1, 78));
for pi = 1:nProfiles
    r = results(pi);
    fprintf('%-16s | %9.4f  %9.4f  | %9.3f  %9.3f  %8.1f%%\n', ...
        r.name, r.vout_rmse, r.iL_rmse, r.iL_sw_pp, r.iL_rom_pp, r.pp_error_pct);
end

% Averages
avg_vout_rmse = mean([results.vout_rmse]);
avg_iL_rmse   = mean([results.iL_rmse]);
avg_pp_err    = mean([results.pp_error_pct]);
fprintf('%s\n', repmat('-', 1, 78));
fprintf('%-16s | %9.4f  %9.4f  | %9s  %9s  %8.1f%%\n', ...
    'AVERAGE', avg_vout_rmse, avg_iL_rmse, '', '', avg_pp_err);
fprintf('\n');

%% 8. Plots
nCols = min(4, nProfiles);
nRows_plot = ceil(nProfiles / nCols);

% --- Figure 1: Vout comparison (top row) + iL comparison (bottom row) ---
fig1 = figure('Position', [30 30 1500 max(500, nRows_plot*400)]);

for pi = 1:nProfiles
    r = results(pi);
    t_ms = (r.t - T_settle) * 1e3;  % Time relative to step, in ms

    % Top: Vout
    subplot(2*nRows_plot, nCols, pi);
    plot(t_ms, r.vout_sw, 'b-', 'LineWidth', 1.2); hold on;
    plot(t_ms, r.vout_rom, 'r--', 'LineWidth', 1.2);
    xline(0, 'k:', 'Step');
    title(sprintf('%s (%.3fV)', r.name, r.vout_rmse), 'FontSize', 9);
    ylabel('Vout (V)'); grid on;
    xlim([-3, T_step*1e3]);
    if pi == 1, legend('Switching', 'LPV ROM', 'Location', 'best', 'FontSize', 7); end

    % Bottom: iL_avg
    subplot(2*nRows_plot, nCols, pi + nCols*nRows_plot);
    plot(t_ms, r.iL_sw, 'b-', 'LineWidth', 1.2); hold on;
    plot(t_ms, r.iL_avg_rom, 'r--', 'LineWidth', 1.2);
    xline(0, 'k:', 'Step');
    title(sprintf('%s (%.3fA)', r.name, r.iL_rmse), 'FontSize', 9);
    xlabel('Time (ms)'); ylabel('iL avg (A)'); grid on;
    xlim([-3, T_step*1e3]);
    if pi == 1, legend('Switching', 'LPV ROM', 'Location', 'best', 'FontSize', 7); end
end

sgtitle('Branch B LPV ROM: Open-Loop Validation (top=Vout, bottom=iL_{avg})', 'FontSize', 13);
plotFile1 = fullfile(modelsDir, 'branch_b_lpv_openloop_validation.png');
saveas(fig1, plotFile1);
fprintf('Plot saved: %s\n', plotFile1);

% --- Figure 2: iL ripple comparison (last 5ms) ---
fig2 = figure('Position', [50 50 1500 max(400, nRows_plot*250)]);

for pi = 1:nProfiles
    r = results(pi);
    subplot(nRows_plot, nCols, pi);

    % Switching model raw iL (last 5ms)
    tail_mask = r.t >= (T_total - 0.005);
    t_tail_ms = (r.t(tail_mask) - T_total + 0.005) * 1e6;  % in us
    plot(t_tail_ms, r.iL_sw_raw(tail_mask), 'b-', 'LineWidth', 0.8); hold on;

    % ROM ripple reconstruction (last 5ms)
    t_rip_us = (r.iL_ripple_t - (T_total - 0.005)) * 1e6;
    plot(t_rip_us, r.iL_ripple, 'r--', 'LineWidth', 1.0);

    title(sprintf('%s (pp err %.1f%%)', r.name, r.pp_error_pct), 'FontSize', 9);
    xlabel('Time (us)'); ylabel('iL (A)'); grid on;
    xlim([0, 50]);  % Show ~10 switching periods
    if pi == 1, legend('Switching', 'ROM Ripple', 'Location', 'best', 'FontSize', 7); end
end

sgtitle('Branch B LPV ROM: iL Ripple Reconstruction (last 5ms)', 'FontSize', 13);
plotFile2 = fullfile(modelsDir, 'branch_b_lpv_ripple_validation.png');
saveas(fig2, plotFile2);
fprintf('Plot saved: %s\n', plotFile2);

%% 9. Save results
resultsFile = fullfile(modelsDir, 'branch_b_lpv_openloop_results.mat');
save(resultsFile, 'results', 'profiles', 'T_settle', 'T_step');
fprintf('Results saved: %s\n', resultsFile);

close all;
fprintf('\n=== Validation Complete ===\n');
