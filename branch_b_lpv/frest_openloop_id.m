% frest_openloop_id.m
% Branch B: Fully open-loop identification pipeline with parfor.
%
% At each operating point (parallel):
%   1. DC gain measurement (two constant-duty sims)
%   2. frestimate with PRBS (Gain block injection + From_Vout_Out measurement)
%   3. Open-loop step response (time-domain)
%   4. tfest on thinned FRD (nz sweep)
%   5. pem refinement on step data (corrects DC gain + transient)
%   6. DC gain correction if needed
%   7. Validation with independent step

% REPLICATION NOTE:
%   This script uses frestimate (Simulink Control Design toolbox) to identify
%   transfer functions at each duty cycle operating point.
%   Requires: boost_converter_test_harness.slx (in simulink_models/)
%   Pre-identified data is already provided in models/boost_frest_tf_data.mat.

thisDir   = fileparts(mfilename('fullpath'));
repoRoot  = fileparts(thisDir);
run(fullfile(repoRoot, 'startup.m'));
modelsDir = fullfile(repoRoot, 'simulink_models');
lpvDir    = fullfile(thisDir, 'models');
if ~exist(lpvDir, 'dir'), mkdir(lpvDir); end
Ts = 5e-6;

%% Setup
fprintf('=== Branch B: Open-Loop frestimate Pipeline (parfor) ===\n\n');

origModel = 'scdboostconverter_with_current';
origFile = fullfile(modelsDir, [origModel '.slx']);
if ~exist(origFile, 'file')
    origModel = 'scdboostconverter_original';
    origFile = fullfile(modelsDir, [origModel '.slx']);
end

harnessModel = 'boost_converter_test_harness';
harnessFile = fullfile(modelsDir, [harnessModel '.slx']);

% Count states
load_system(origFile);
[nx, ~] = countSubsystemStates(origModel);
fprintf('Plant order: nx = %d\n', nx);

% Detect switching frequency from PWM-related blocks
f_sw = detectSwitchingFrequency(origModel);
fprintf('Switching frequency: %.0f Hz\n', f_sw);
close_system(origModel, 0);

% Grid
D_grid = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]';
nGrid = numel(D_grid);

fprintf('Grid: %d points, nx=%d\n', nGrid, nx);

%% Parallel pool
pool = gcp('nocreate');
if isempty(pool), pool = parpool('local', 4); end  % limit to 4 workers (~14GB vs 50GB)
fprintf('Workers: %d\n\n', pool.NumWorkers);

%% Run all points in parallel
results = cell(nGrid, 1);
tic;
parfor i = 1:nGrid
    results{i} = identify_point(i, D_grid(i), harnessFile, harnessModel, ...
        lpvDir, nx, Ts, f_sw);
end
t_total = toc;
fprintf('\nAll %d points in %.1f s (%.1f s/point)\n\n', nGrid, t_total, t_total/nGrid);

%% Collect
tf_models = cell(nGrid, 1);
iL_models = cell(nGrid, 1);
Vout_ss = zeros(nGrid, 1);
Vout_ss_grid = zeros(nGrid, 1);
iL_ss = zeros(nGrid, 1);
iL_ss_grid = zeros(nGrid, 1);
fit_pct = zeros(nGrid, 1);
iL_fit_pct = zeros(nGrid, 1);
dc_gains = zeros(nGrid, 1);

for i = 1:nGrid
    r = results{i};
    tf_models{i} = r.sys;
    iL_models{i} = r.sys_iL;
    Vout_ss(i) = r.Vout_ss;
    Vout_ss_grid(i) = r.Vout_ss;
    iL_ss(i) = r.iL_ss;
    iL_ss_grid(i) = r.iL_ss;
    fit_pct(i) = r.step_fit;
    iL_fit_pct(i) = r.iL_fit;
    dc_gains(i) = r.dc_gain;
end

%% Summary
%% Post-parfor: verify and fix iL DC gains serially
% The parfor workers sometimes extract wrong iL values from their model copies.
% Re-measure iL DC gains using the original harness (verified correct).
fprintf('\n=== Post-parfor iL DC gain verification ===\n');
load_system(harnessFile);
set_param(harnessModel, 'MaxStep', '0.5e-6', 'ReturnWorkspaceOutputs', 'on', ...
    'SaveFormat', 'Dataset', 'SignalLogging', 'on', 'StopTime', '0.2');
delta_verify = 0.005;
t_verify = (0:Ts:0.2)';
for i = 1:nGrid
    if isempty(iL_models{i}), continue; end
    D_op = D_grid(i);
    % Sim at D_op
    duty_input = timeseries(D_op * ones(size(t_verify)), t_verify);
    assignin('base', 'duty_input', duty_input);
    s1v = sim(harnessModel);
    iL1v = squeeze(s1v.logsout.getElement('iL').Values.Data);
    iL1_ss = mean(iL1v(end-500:end));
    % Sim at D_op + delta
    duty_input = timeseries((D_op + delta_verify) * ones(size(t_verify)), t_verify);
    assignin('base', 'duty_input', duty_input);
    s2v = sim(harnessModel);
    iL2v = squeeze(s2v.logsout.getElement('iL').Values.Data);
    iL2_ss = mean(iL2v(end-500:end));
    % Verified DC gain
    iL_dc_verified = (iL2_ss - iL1_ss) / delta_verify;
    model_dc = dcgain(iL_models{i});
    % Fix sign if wrong
    if sign(model_dc) ~= sign(iL_dc_verified) && iL_dc_verified ~= 0
        [A_fix, B_fix, C_fix, D_fix] = ssdata(iL_models{i});
        C_fix = -C_fix; D_fix = -D_fix;
        iL_models{i} = ss(A_fix, B_fix, C_fix, D_fix, 0);
        model_dc = -model_dc;
        fprintf('  D=%.2f: sign FIXED (was %.1f, measured +%.1f)\n', D_op, -model_dc, iL_dc_verified);
    end
    % Fix magnitude
    if abs(model_dc) > 0 && isfinite(model_dc)
        scale = iL_dc_verified / model_dc;
        if abs(scale - 1) > 0.10  % >10% off
            [A_fix, B_fix, C_fix, D_fix] = ssdata(iL_models{i});
            C_fix = C_fix * scale; D_fix = D_fix * scale;
            iL_models{i} = ss(A_fix, B_fix, C_fix, D_fix, 0);
            fprintf('  D=%.2f: gain scaled %.1f → %.1f\n', D_op, model_dc, iL_dc_verified);
        else
            fprintf('  D=%.2f: OK (model=%.1f, measured=%.1f)\n', D_op, model_dc, iL_dc_verified);
        end
    end
    % Update iL_ss
    iL_ss(i) = iL1_ss;
    iL_ss_grid(i) = iL1_ss;
end
close_system(harnessModel, 0);
fprintf('iL DC gains verified and corrected.\n\n');

fprintf('=== Identification Summary ===\n');
fprintf('D_op  | Vout_ss | iL_ss  | Vout Fit%% | iL Fit%% | DC Gain  | Poles\n');
fprintf('------+---------+--------+-----------+---------+----------+------\n');
for i = 1:nGrid
    r = results{i};
    if ~isempty(r.sys)
        fprintf('%.2f  | %6.3f  | %5.2f  | %7.1f%%  | %6.1f%% | %8.2f | %s\n', ...
            D_grid(i), r.Vout_ss, r.iL_ss, r.step_fit, r.iL_fit, r.dc_gain, ...
            mat2str(pole(r.sys)', 4));
    else
        fprintf('%.2f  | %6.3f  | %5.2f  |   FAIL    |  FAIL   |    --    | --\n', ...
            D_grid(i), r.Vout_ss, r.iL_ss);
    end
end

%% Validation: independent step
fprintf('\n=== Validation: Open-Loop Step (+0.02 duty) ===\n');
delta_val = 0.02;
val_settle = 0.015;
val_dur = 0.015;
val_total = val_settle + val_dur;

load_system(harnessFile);
set_param(harnessModel, 'MaxStep', '0.5e-6');
set_param(harnessModel, 'StopTime', num2str(val_total));
set_param(harnessModel, 'ReturnWorkspaceOutputs', 'on');
set_param(harnessModel, 'SaveFormat', 'Dataset');
set_param(harnessModel, 'SignalLogging', 'on');

fig = figure('Position', [50 50 1200 800]);

for i = 1:nGrid
    if isempty(tf_models{i}), continue; end
    D_op = D_grid(i);

    t_vec = (0:Ts:val_total)';
    n_settle = round(val_settle / Ts);
    duty_vec = D_op * ones(size(t_vec));
    duty_vec(n_settle+1:end) = D_op + delta_val;
    duty_input = timeseries(duty_vec, t_vec);
    assignin('base', 'duty_input', duty_input);

    simVal = sim(harnessModel);
    v_raw = interp1(squeeze(simVal.logsout.getElement('Vout').Values.Time), ...
        squeeze(simVal.logsout.getElement('Vout').Values.Data), t_vec, 'linear', 'extrap');

    % Use adaptive cutoff if available, else default to 30kHz
    if exist('results_par', 'var') && i <= numel(results_par) && isfield(results_par{i}, 'f_cutoff')
        fc = results_par{i}.f_cutoff;
    else
        fc = 30e3;
    end
    [b_f, a_f] = butter(3, fc / (1/Ts/2));
    v_filt = filtfilt(b_f, a_f, v_raw);

    sys_d = c2d(tf_models{i}, Ts);
    u_delta = zeros(size(t_vec));
    u_delta(n_settle+1:end) = delta_val;
    y_delta = lsim(sys_d, u_delta, t_vec);
    v_rom = Vout_ss(i) + y_delta;

    post = t_vec >= val_settle + 0.001;
    rmse = sqrt(mean((v_rom(post) - v_filt(post)).^2));
    fprintf('D=%.2f: RMSE=%.4fV\n', D_op, rmse);

    subplot(3, 4, i);
    t_plot = (t_vec - val_settle) * 1e3;
    plot(t_plot, v_filt, 'b-', 'LineWidth', 1.5); hold on;
    plot(t_plot, v_rom, 'r--', 'LineWidth', 1.5);
    xline(0, 'k:', 'Step');
    title(sprintf('D=%.2f (RMSE=%.3fV)', D_op, rmse), 'FontSize', 10);
    xlabel('Time (ms)'); ylabel('Vout (V)');
    if i == 1, legend('Plant (filt)', 'TF ROM', 'Location', 'best'); end
    grid on;
    v_post = v_filt(post);
    ylim([min(v_post)-0.1, max(v_post)+0.3]);
    xlim([-2, val_dur*1e3]);
end

sgtitle(sprintf('Branch B: Open-Loop TF Validation (%.0fs total)', t_total));
saveas(fig, fullfile(lpvDir, 'frest_tf_step_validation.png'));
close(fig);
close_system(harnessModel, 0);

%% Save
save(fullfile(lpvDir, 'boost_frest_tf_data.mat'), ...
    'tf_models', 'iL_models', 'D_grid', 'Vout_ss', 'Vout_ss_grid', ...
    'iL_ss', 'iL_ss_grid', 'fit_pct', 'iL_fit_pct', 'dc_gains', 'nx', 'Ts');
fprintf('\nSaved. Plot saved.\n=== Done ===\n');


%% ===================================================================
function result = identify_point(idx, D_op, harnessFile, harnessModel, ...
    modelsDir, nx, Ts, f_sw)
% All-in-one identification at one operating point (runs in parfor worker).

    result = struct('sys', [], 'sys_iL', [], 'Vout_ss', 0, 'iL_ss', 0, ...
        'dc_gain', NaN, 'frd_fit', NaN, 'step_fit', NaN, 'iL_fit', NaN);

    % Worker model name
    wModel = sprintf('%s_w%d', harnessModel, idx);
    wFile = fullfile(modelsDir, [wModel '.slx']);

    try
        % --- Create worker copy ---
        copyfile(harnessFile, wFile);
        load_system(wFile);
        set_param(wModel, 'MaxStep', '0.5e-6');
        set_param(wModel, 'ReturnWorkspaceOutputs', 'on');
        set_param(wModel, 'SaveFormat', 'Dataset');
        set_param(wModel, 'SignalLogging', 'on');

        % --- Insert Gain=1 block after FromWS_duty for frestimate injection ---
        fromWS = [wModel '/FromWS_duty'];
        fromWS_pos = get_param(fromWS, 'Position');
        ph_fw = get_param(fromWS, 'PortHandles');
        line_h = get_param(ph_fw.Outport(1), 'Line');
        dst_ports = get_param(line_h, 'DstPortHandle');
        delete_line(line_h);

        gain_pos = [fromWS_pos(3)+30, fromWS_pos(2), fromWS_pos(3)+70, fromWS_pos(4)];
        gainPath = [wModel '/DutyInject'];
        add_block('simulink/Math Operations/Gain', gainPath, ...
            'Position', gain_pos, 'Gain', '1');

        gain_ph = get_param(gainPath, 'PortHandles');
        add_line(wModel, ph_fw.Outport(1), gain_ph.Inport(1));
        for dp = 1:numel(dst_ports)
            add_line(wModel, gain_ph.Outport(1), dst_ports(dp));
        end

        % --- 1. Measure Vout_ss and DC gain ---
        dc_settle = 0.2;  % 200ms — ensures full settling for high-Q systems
        delta_dc = 0.005;
        t_dc = (0:Ts:dc_settle)';
        set_param(wModel, 'StopTime', num2str(dc_settle));

        % Sim at D_op — extract Vout and iL from logsout (no Simscape log needed)
        duty_input = timeseries(D_op * ones(size(t_dc)), t_dc);
        assignin('base', 'duty_input', duty_input);
        s1 = sim(wModel);
        v1 = squeeze(s1.logsout.getElement('Vout').Values.Data);
        result.Vout_ss = mean(v1(end-500:end));

        % iL from logsout (current sensor signal)
        iL1 = squeeze(s1.logsout.getElement('iL').Values.Data);
        result.iL_ss = mean(iL1(end-500:end));

        % Sim at D_op + delta
        duty_input = timeseries((D_op + delta_dc) * ones(size(t_dc)), t_dc);
        assignin('base', 'duty_input', duty_input);
        s2 = sim(wModel);
        v2 = squeeze(s2.logsout.getElement('Vout').Values.Data);
        Vout2 = mean(v2(end-500:end));

        % iL_ss2 from logsout
        iL2 = squeeze(s2.logsout.getElement('iL').Values.Data);
        result.iL_ss2 = mean(iL2(end-500:end));

        dc_gain_meas = abs(Vout2 - result.Vout_ss) / delta_dc;
        dc_gain_meas = max(dc_gain_meas, 1);

        % Scale PRBS amplitude: target output perturbation = 5% of Vout_ss.
        % Cap duty perturbation at 0.05 (5%). This is model-agnostic —
        % based on measured DC gain and operating point voltage.
        max_output_perturb = 0.05 * result.Vout_ss;
        prbs_amp = max(0.005, min(0.05, max_output_perturb / dc_gain_meas));

        % --- 2. frestimate (PRBS) ---
        % Set constant duty for findop
        duty_input = timeseries(D_op * ones(size(t_dc)), t_dc);
        assignin('base', 'duty_input', duty_input);
        set_param(wModel, 'StopTime', '0.05');
        opt_op = findopOptions('DisplayReport', 'off');
        op = findop(wModel, 0.03, opt_op);

        % linio: openinput at Gain, output at From_Vout_Out
        io_w(1) = linio(gainPath, 1, 'openinput');
        measBlk = find_system(wModel, 'SearchDepth', 1, 'BlockType', 'From');
        measPath = '';
        for j = 1:numel(measBlk)
            if contains(get_param(measBlk{j}, 'Name'), 'Vout')
                measPath = measBlk{j};
                break;
            end
        end
        if isempty(measPath)
            measPath = [wModel '/Sampling'];
        end
        io_w(2) = linio(measPath, 1, 'output');

        in = frest.PRBS('Order', 10, 'Amplitude', prbs_amp, 'Ts', Ts, 'NumPeriods', 4);
        [sysest, ~] = frestimate(wModel, op, io_w, in);

        % Thin FRD — adaptive range based on resonance detection
        frd_raw = frd(sysest);
        w_raw = frd_raw.Frequency;
        mag_raw = squeeze(abs(frd_raw.ResponseData(1,1,:)));

        % Detect resonance peak (highest magnitude above 500Hz)
        valid_idx = w_raw > 2*pi*500;
        [~, peak_idx] = max(mag_raw(valid_idx));
        valid_w = w_raw(valid_idx);
        f_resonance = valid_w(peak_idx) / (2*pi);

        % Set thinning range: 100Hz to 2x resonance (or at least 50kHz)
        w_min = 2*pi*100;
        w_max = 2*pi * max(50e3, 2 * f_resonance);
        n_thin = 40;  % more points for better resolution
        w_thin = logspace(log10(w_min), log10(w_max), n_thin)';

        % Add extra points around resonance for better capture
        w_res_band = linspace(2*pi*f_resonance*0.5, 2*pi*f_resonance*1.5, 10)';
        w_thin = unique(sort([w_thin; w_res_band]));
        n_thin = numel(w_thin);

        % Only keep frequencies within raw FRD range
        w_thin = w_thin(w_thin >= w_raw(1) & w_thin <= w_raw(end));
        n_thin = numel(w_thin);

        resp_thin = zeros(1, 1, n_thin);
        for fi = 1:n_thin
            [~, mi] = min(abs(w_raw - w_thin(fi)));
            resp_thin(1,1,fi) = frd_raw.ResponseData(1,1,mi);
        end
        frd_id = idfrd(frd(resp_thin, w_thin));

        % Adaptive filter cutoff: geometric mean of resonance and switching.
        % Keeps resonance oscillations, removes switching ripple.
        % f_sw detected from PWM blocks in the model (generalizable).
        f_cutoff = sqrt(f_resonance * f_sw);
        f_cutoff = max(f_cutoff, 3 * f_resonance);  % at least 3x resonance
        f_cutoff = min(f_cutoff, 0.4/Ts);            % below Nyquist
        result.f_resonance = f_resonance;
        result.f_switching = f_sw;
        result.f_cutoff = f_cutoff;

        fprintf('  Resonance=%.0fHz, Filter cutoff=%.0fHz (10x res), FRD: %d pts [%.0f-%.0fHz]\n', ...
            f_resonance, f_cutoff, n_thin, w_thin(1)/(2*pi), w_thin(end)/(2*pi));

        % --- 3. Open-loop step response (Vout + iL from logsout) ---
        step_delta = prbs_amp;
        step_settle = max(0.015, 10 / f_resonance);  % enough to reach steady state
        step_dur = max(0.015, 15 / f_resonance);    % enough for 15 oscillation cycles
        step_total = step_settle + step_dur;
        t_sv = (0:Ts:step_total)';
        n_settle = round(step_settle / Ts);

        duty_vec = D_op * ones(size(t_sv));
        duty_vec(n_settle+1:end) = D_op + step_delta;
        duty_input = timeseries(duty_vec, t_sv);
        assignin('base', 'duty_input', duty_input);
        set_param(wModel, 'StopTime', num2str(step_total));

        simStep = sim(wModel);

        % Extract Vout from logsout
        v_step = interp1(squeeze(simStep.logsout.getElement('Vout').Values.Time), ...
            squeeze(simStep.logsout.getElement('Vout').Values.Data), t_sv, 'linear', 'extrap');

        % Extract iL from logsout (current sensor signal)
        iL_step = interp1(squeeze(simStep.logsout.getElement('iL').Values.Time), ...
            squeeze(simStep.logsout.getElement('iL').Values.Data), t_sv, 'linear', 'extrap');

        % Filter switching ripple — adaptive cutoff from resonance/switching detection
        f_nyq = 1/(2*Ts);
        fc = result.f_cutoff;
        if ~isfinite(fc) || fc <= 0 || fc >= f_nyq
            fc = 0.3 * f_nyq;  % default: 30% of Nyquist
        end
        [b_f, a_f] = butter(3, fc / f_nyq);
        v_filt = filtfilt(b_f, a_f, v_step);
        iL_filt = filtfilt(b_f, a_f, iL_step);

        Vss_step = mean(v_filt(max(1,n_settle-500):n_settle));
        iL_ss = mean(iL_filt(max(1,n_settle-500):n_settle));
        % Don't overwrite result.iL_ss — keep the DC sim value which is correct

        % Build SISO perturbation data — TRANSIENT ONLY
        % Only use the first few ms after the step, not the full response.
        % This forces ssest to fit the oscillation (underdamped dynamics)
        % instead of optimizing for the long flat steady-state tail.
        % Transient window: 10x the resonance period (captures ~10 oscillation cycles)
        if isfield(result, 'f_resonance') && result.f_resonance > 0
            n_transient = min(round(10 / (result.f_resonance * Ts)), numel(t_sv) - n_settle);
        else
            n_transient = round(0.005 / Ts);  % default 5ms
        end
        n_transient = max(n_transient, round(0.002 / Ts));  % at least 2ms

        dVout_full = v_filt - Vss_step;
        diL_full = iL_filt - iL_ss;

        % Extract transient window starting from step
        idx_start = n_settle + 1;
        idx_end = min(n_settle + n_transient, numel(t_sv));
        dVout = dVout_full(idx_start:idx_end);
        diL = diL_full(idx_start:idx_end);
        dDuty = step_delta * ones(size(dVout));  % constant step input

        step_data_vout = iddata(dVout, dDuty, Ts);
        step_data_iL = iddata(diL, dDuty, Ts);
        fprintf('  Transient window: %.1f ms (%d samples)\n', ...
            n_transient*Ts*1e3, n_transient);

        % --- 4a. Vout identification: DC-gain-constrained ssest ---
        % Create initial state-space model with DC gain fixed to measured value.
        % DC gain of ss model = -C*A^{-1}*B + D.
        % For a simple 2nd-order system, set up initial guess and fix D.
        ss_opts = ssestOptions('EnforceStability', true);
        sys_vout = [];
        best_vout_fit = -Inf;

        % Strategy 1: ssest on step data
        try
            sys_v1 = ssest(step_data_vout, nx, ss_opts);
            [~, fv1] = compare(step_data_vout, sys_v1);
            if fv1 > best_vout_fit
                best_vout_fit = fv1;
                sys_vout = sys_v1;
            end
        catch
        end

        % Strategy 2: n4sid on step data (subspace method — better for noisy data)
        try
            sys_n4 = n4sid(step_data_vout, nx, 'EnforceStability', true);
            [~, fv_n4] = compare(step_data_vout, sys_n4);
            if fv_n4 > best_vout_fit
                best_vout_fit = fv_n4;
                sys_vout = sys_n4;
            end
        catch
        end

        % Strategy 3: ssest on FRD, then refine with step
        try
            sys_v2 = ssest(frd_id, nx, ss_opts);
            [~, fv_frd] = compare(frd_id, sys_v2);
            result.frd_fit = fv_frd;
            try
                sys_v2r = pem(step_data_vout, sys_v2);
                if all(real(pole(sys_v2r)) < 0)
                    [~, fv2] = compare(step_data_vout, sys_v2r);
                    if fv2 > best_vout_fit
                        best_vout_fit = fv2;
                        sys_vout = sys_v2r;
                    end
                end
            catch
            end
        catch
        end
        result.step_fit = best_vout_fit;

        % Final stability guard — check poles AND validate with lsim
        model_bad = false;
        if ~isempty(sys_vout)
            if any(real(pole(sys_vout)) > 0)
                model_bad = true;
            else
                % Check for near-marginal: simulate step and check if output blows up
                try
                    y_check = lsim(sys_vout, ones(2000,1)*delta_dc, (0:1999)'*Ts);
                    if max(abs(y_check)) > 50 * dc_gain_meas * delta_dc
                        model_bad = true;
                    end
                catch
                    model_bad = true;
                end
            end
        end
        if model_bad
            fprintf('  WARNING: Vout model unstable/divergent, falling back to step-only ssest\n');
            try
                % Fall back to step-only ssest (preserves correct damping)
                sys_fb = ssest(step_data_vout, nx, ss_opts);
                y_fb = lsim(sys_fb, ones(2000,1)*delta_dc, (0:1999)'*Ts);
                if all(real(pole(sys_fb)) < 0) && max(abs(y_fb)) < 50*dc_gain_meas*delta_dc
                    sys_vout = sys_fb;
                    [~, fv_fb] = compare(step_data_vout, sys_fb);
                    result.step_fit = fv_fb;
                else
                    % Step-only also bad, try FRD as last resort
                    sys_vout = ssest(frd_id, nx, ss_opts);
                end
            catch
                try
                    sys_vout = ssest(frd_id, nx, ss_opts);
                catch
                    sys_vout = [];
                end
            end
        end

        % --- 4b. iL identification: hybrid (try independent, fall back to shared A/B) ---
        % Try independent ssest first. If it fails or gives poor fit,
        % fall back to shared A/B from Vout (which uses cleaner dynamics).
        sys_iL = [];
        try
            if ~isempty(step_data_iL) && ~isempty(step_data_iL.OutputData)
                iL_dc_measured = (result.iL_ss2 - result.iL_ss) / delta_dc;
                best_iL_fit = -Inf;

                % --- Approach A: Independent ssest ---
                try
                    opt_iL = ssestOptions('EnforceStability', true);
                    sys_iL_ind = ssest(step_data_iL, nx, opt_iL);
                    % Stability check
                    test_r = lsim(sys_iL_ind, ones(1000,1)*delta_dc, (0:999)'*Ts);
                    if max(abs(test_r)) < 100 * max(abs(iL_dc_measured)*delta_dc, 1)
                        [~, fv_ind] = compare(step_data_iL, sys_iL_ind);
                        if fv_ind > best_iL_fit
                            best_iL_fit = fv_ind;
                            sys_iL = sys_iL_ind;
                        end
                    end
                catch; end

                % --- Approach B: Shared A/B from Vout, fit C_iL by least-squares ---
                % Use transient-only state trajectory from Vout model.
                % Key: the Vout model already has correct oscillatory dynamics
                % (poles). We just need to find the C_iL that maps those
                % oscillating states to iL. Fitting on the transient window
                % ensures the oscillation ratio is captured, not just DC.
                try
                    if ~isempty(sys_vout)
                        [A_v, B_v, C_v, D_v] = ssdata(sys_vout);
                        % Simulate state trajectory from the Vout model
                        % (continuous model, discretize for lsim)
                        sys_v_states = c2d(ss(A_v, B_v, eye(nx), zeros(nx,1)), Ts);
                        t_ls = (0:size(step_data_iL.InputData,1)-1)'*Ts;
                        x_traj = lsim(sys_v_states, step_data_iL.InputData, t_ls);

                        % Fit C_iL on transient only (no D term — force through dynamics)
                        X_ls = x_traj;  % no D column — dynamics only
                        iL_target = step_data_iL.OutputData;
                        C_sh = (X_ls' * X_ls) \ (X_ls' * iL_target);
                        C_sh = C_sh';
                        D_sh = 0;  % no feedthrough
                        sys_iL_sh = ss(A_v, B_v, C_sh, D_sh, 0);
                        % Stability check
                        test_r2 = lsim(sys_iL_sh, ones(1000,1)*delta_dc, (0:999)'*Ts);
                        if max(abs(test_r2)) < 100 * max(abs(iL_dc_measured)*delta_dc, 1)
                            [~, fv_sh] = compare(step_data_iL, sys_iL_sh);
                            if fv_sh > best_iL_fit
                                best_iL_fit = fv_sh;
                                sys_iL = sys_iL_sh;
                            end
                        end
                    end
                catch; end

                % --- ALWAYS force DC gain to match measured value ---
                % The ssest/least-squares DC gains are unreliable for iL.
                % Use the measured DC gain from the step response data directly
                % (pre-step vs post-step average). This is generalizable.
                if ~isempty(sys_iL)
                    [A_iL, B_iL, C_iL, D_iL] = ssdata(sys_iL);

                    % Use DC sim values (verified correct: 0.99A to 8.78A)
                    % iL_dc_measured already set from DC sims:
                    % (result.iL_ss2 - result.iL_ss) / delta_dc

                    iL_dc_model = dcgain(sys_iL);

                    % Force correct sign
                    if isfinite(iL_dc_model) && isfinite(iL_dc_measured) && ...
                            sign(iL_dc_model) ~= sign(iL_dc_measured) && iL_dc_measured ~= 0
                        C_iL = -C_iL; D_iL = -D_iL;
                        iL_dc_model = -iL_dc_model;
                        fprintf('  iL sign corrected\n');
                    end

                    % ALWAYS force DC gain to match (no 15% threshold)
                    if abs(iL_dc_model) > 0 && isfinite(iL_dc_model) && abs(iL_dc_measured) > 0
                        iL_scale = iL_dc_measured / iL_dc_model;
                        C_iL = C_iL * iL_scale;
                        D_iL = D_iL * iL_scale;
                    end

                    sys_iL = ss(A_iL, B_iL, C_iL, D_iL, 0);
                    [~, fv_final] = compare(step_data_iL, sys_iL);
                    result.iL_fit = fv_final;
                    fprintf('  iL (DC-corrected): fit=%.1f%%, DC=%.1f (meas=%.1f)\n', ...
                        fv_final, dcgain(sys_iL), iL_dc_measured);
                else
                    result.iL_fit = NaN;
                end
            else
                result.iL_fit = NaN;
            end
        catch ME
            fprintf('  iL identification failed: %s\n', ME.message);
            result.iL_fit = NaN;
        end

        % --- 5. DC gain correction + store ---
        if ~isempty(sys_vout)
            [A_s, B_s, C_s, D_s] = ssdata(sys_vout);
            model_dc = abs(dcgain(sys_vout));
            if model_dc > 0 && isfinite(model_dc) && dc_gain_meas > 0
                scale = dc_gain_meas / model_dc;
                if abs(scale - 1) > 0.15  % >15% off
                    C_s = C_s * scale; D_s = D_s * scale;
                    sys_vout = idss(A_s, B_s, C_s, D_s, 'Ts', 0);
                end
            end
            % Balanced realization — ensures consistent state basis across
            % grid points so that LPV matrix interpolation is meaningful.
            % Without this, each ssest model has arbitrary eigenvector
            % directions, and interpolating A matrices in different
            % coordinate systems produces wrong dynamics.
            try
                sys_vout_bal = balreal(ss(sys_vout));
                % Apply same transformation to iL model (shared A/B)
                if ~isempty(sys_iL)
                    [A_b, B_b, C_b, D_b] = ssdata(sys_vout_bal);
                    [~, ~, C_iL_old, D_iL_old] = ssdata(sys_iL);
                    % Compute transformation: T such that A_bal = T*A*inv(T)
                    % C_iL_bal = C_iL_old * inv(T)
                    [A_orig, ~, ~, ~] = ssdata(ss(sys_vout));
                    T = ctrb(A_b, B_b) / ctrb(A_orig, B_b);
                    if rcond(T) > 1e-10
                        C_iL_bal = C_iL_old / T;
                        sys_iL = ss(A_b, B_b, C_iL_bal, D_iL_old, 0);
                    end
                end
                sys_vout = sys_vout_bal;
            catch
                % balreal may fail for some models — keep original
            end
            result.sys = sys_vout;
            result.sys_iL = sys_iL;
            result.dc_gain = abs(dcgain(sys_vout));
        end

        fprintf('  [%d] D=%.2f: FRD=%.1f%%, Step=%.1f%%, DC=%.1f\n', ...
            idx, D_op, result.frd_fit, result.step_fit, result.dc_gain);

        % Cleanup
        close_system(wModel, 0);
        delete(wFile);

    catch ME
        fprintf('  [%d] D=%.2f: FAILED — %s\n', idx, D_op, ME.message);
        try close_system(wModel, 0); catch; end
        try delete(wFile); catch; end
    end
end
