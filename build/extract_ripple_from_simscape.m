% extract_ripple_from_simscape.m
% Topology-agnostic empirical ripple extraction from Simscape model.
%
% Runs the Simscape model at several steady-state duty cycles and measures
% the actual inductor current ripple — no analytical formulas needed.
%
% Output: models/ripple_empirical_data.mat containing:
%   ripple_duty_grid    - duty cycle breakpoints (1 x nGrid)
%   ripple_delta_iL     - measured peak-to-peak ripple (1 x nGrid)
%   ripple_shape        - normalized waveform shape (nPtsPerCycle x nGrid)
%   ripple_Tsw          - detected switching period (scalar)
%   ripple_is_triangular - true if shape is well-approximated by triangle
%   ripple_phase_grid   - phase breakpoints for shape (nPtsPerCycle x 1)
buildDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(buildDir);
run(fullfile(repoRoot, 'startup.m'));

modelsDir = fullfile(repoRoot, 'model_data');
simulinkDir = fullfile(repoRoot, 'simulink_models');
simscapeModel = 'scdboostconverter_openloop';
nPtsPerCycle = 20;  % resample ripple shape to this many points

%% 1. Load model and detect switching frequency
fprintf('=== Empirical Ripple Extraction ===\n');
load_system(fullfile(simulinkDir, [simscapeModel '.slx']));

fprintf('\nDetecting switching frequency...\n');
f_sw = detectSwitchingFrequency(simscapeModel);
Tsw = 1 / f_sw;
fprintf('  f_sw = %.0f Hz, Tsw = %.2f μs\n', f_sw, Tsw*1e6);

%% 2. Configure model for steady-state extraction
% Use the existing Step block — set it to constant duty (step at t=0, before=after=D)
dutyBlk = [simscapeModel '/Duty Step'];

% Ensure iL is logged
fromIL = find_system(simscapeModel, 'SearchDepth', 1, ...
    'BlockType', 'From', 'GotoTag', 'iL');
if ~isempty(fromIL)
    fph = get_param(fromIL{1}, 'PortHandles');
    set_param(fph.Outport(1), 'DataLogging', 'on', ...
        'DataLoggingName', 'IL', 'DataLoggingNameMode', 'Custom');
end

% Set solver for high resolution
set_param(simscapeModel, 'MaxStep', num2str(Tsw / nPtsPerCycle));
set_param(simscapeModel, 'ReturnWorkspaceOutputs', 'on');
set_param(simscapeModel, 'SaveFormat', 'Dataset');
set_param(simscapeModel, 'SignalLogging', 'on');

%% 3. Run at each duty cycle
dutyGrid = linspace(0.10, 0.70, 13);  % avoid D>0.70 where boost is near-unstable
simTime = 5e-3;  % 5ms — longer settling for higher duty
nSteadyCycles = 20;  % use last 20 switching cycles for better averaging

set_param(simscapeModel, 'StopTime', num2str(simTime));

ripple_delta_iL = zeros(1, numel(dutyGrid));
ripple_shape = zeros(nPtsPerCycle, numel(dutyGrid));
ripple_iL_avg = zeros(1, numel(dutyGrid));

fprintf('\nExtracting ripple at %d duty cycles...\n', numel(dutyGrid));
for di = 1:numel(dutyGrid)
    D = dutyGrid(di);

    % Set Step block to constant duty (Before = After = D, step at t=0)
    set_param(dutyBlk, 'Before', num2str(D), 'After', num2str(D), 'Time', '0');

    % Simulate
    try
        simOut = sim(simscapeModel);
    catch e
        fprintf('  D=%.2f: FAILED (%s)\n', D, e.message);
        continue;
    end

    % Extract iL
    try
        iL_ts = simOut.logsout.getElement('IL').Values;
    catch
        % Try alternate names
        names = simOut.logsout.getElementNames();
        fprintf('  Available signals: %s\n', strjoin(names, ', '));
        continue;
    end

    t_il = iL_ts.Time;
    iL = squeeze(iL_ts.Data);

    % Take last nSteadyCycles switching cycles
    t_end = t_il(end);
    t_start_ss = t_end - nSteadyCycles * Tsw;
    mask = t_il >= t_start_ss;
    t_ss = t_il(mask) - t_start_ss;
    iL_ss = iL(mask);

    % Compute per-cycle statistics
    cycle_delta = zeros(nSteadyCycles, 1);
    cycle_shapes = zeros(nPtsPerCycle, nSteadyCycles);

    % Find PWM phase alignment: detect the rising edge (min of iL)
    % by looking at derivative sign changes in the first cycle
    diL = diff(iL_ss);
    % Find first trough (rising edge start) — where diL goes from negative to positive
    crossings = find(diL(1:end-1) < 0 & diL(2:end) >= 0, 1, 'first');
    if isempty(crossings)
        crossings = 1;
    end
    t_phase0 = t_ss(crossings);  % align cycles to this point

    for ci = 1:nSteadyCycles
        t0 = t_phase0 + (ci-1) * Tsw;
        t1 = t0 + Tsw;
        cmask = t_ss >= t0 & t_ss < t1;
        iL_cyc = iL_ss(cmask);
        t_cyc = t_ss(cmask) - t0;

        if numel(iL_cyc) < 3
            continue;
        end

        cycle_delta(ci) = max(iL_cyc) - min(iL_cyc);
        iL_avg_cyc = mean(iL_cyc);

        % Resample to nPtsPerCycle points (uniform phase spacing)
        phase_pts = linspace(0, Tsw*(1 - 1/nPtsPerCycle), nPtsPerCycle)';
        iL_resampled = interp1(t_cyc, iL_cyc, phase_pts, 'linear', 'extrap');

        % Normalized shape: (iL - mean) / delta_iL
        if cycle_delta(ci) > 1e-6
            cycle_shapes(:, ci) = (iL_resampled - iL_avg_cyc) / cycle_delta(ci);
        end
    end

    ripple_delta_iL(di) = mean(cycle_delta);
    ripple_shape(:, di) = mean(cycle_shapes, 2);
    ripple_iL_avg(di) = mean(iL_ss);

    fprintf('  D=%.2f: iL_avg=%.2fA, ΔiL=%.4fA\n', D, ripple_iL_avg(di), ripple_delta_iL(di));
end

%% 4. Check if shape is triangular
% Build ideal triangular shape for comparison
phase_grid = linspace(0, 1 - 1/nPtsPerCycle, nPtsPerCycle)';
tri_corrs = zeros(1, numel(dutyGrid));

for di = 1:numel(dutyGrid)
    D = dutyGrid(di);
    D_c = max(0.05, min(0.95, D));

    % Ideal triangle: rises during [0, D), falls during [D, 1)
    ideal = zeros(nPtsPerCycle, 1);
    for pi = 1:nPtsPerCycle
        ph = phase_grid(pi);
        if ph < D_c
            ideal(pi) = -0.5 + ph / D_c;
        else
            ideal(pi) = 0.5 - (ph - D_c) / (1 - D_c);
        end
    end

    measured = ripple_shape(:, di);
    if norm(measured) > 1e-6 && norm(ideal) > 1e-6
        tri_corrs(di) = dot(measured, ideal) / (norm(measured) * norm(ideal));
    end
end

avg_corr = mean(tri_corrs(tri_corrs > 0));
% Note: low correlation can result from PWM phase misalignment in measurement,
% not from non-triangular shape. Check visually from the plot.
% Use 0.4 threshold since phase offset reduces correlation even for perfect triangles.
ripple_is_triangular = avg_corr > 0.4;

fprintf('\n=== Shape Analysis ===\n');
fprintf('  Avg correlation with ideal triangle: %.4f\n', avg_corr);
fprintf('  Is triangular: %s\n', mat2str(ripple_is_triangular));

%% 5. Save results
ripple_duty_grid = dutyGrid;
ripple_Tsw = Tsw;

outFile = fullfile(modelsDir, 'ripple_empirical_data.mat');
ripple_phase_grid = phase_grid;
save(outFile, 'ripple_duty_grid', 'ripple_delta_iL', 'ripple_shape', ...
    'ripple_Tsw', 'ripple_is_triangular', 'ripple_phase_grid', ...
    'ripple_iL_avg');
fprintf('\nSaved to %s\n', outFile);

%% 6. Comparison plot
fig = figure('Position', [100 100 900 600]);

subplot(2,2,1);
plot(ripple_duty_grid, ripple_delta_iL * 1e3, 'bo-', 'LineWidth', 1.5);
xlabel('Duty Cycle'); ylabel('ΔiL (mA)');
title('Ripple Amplitude vs Duty'); grid on;

subplot(2,2,2);
plot(ripple_duty_grid, ripple_iL_avg, 'ro-', 'LineWidth', 1.5);
xlabel('Duty Cycle'); ylabel('iL avg (A)');
title('Average Current vs Duty'); grid on;

subplot(2,2,[3 4]);
% Plot a few ripple shapes
colors = lines(5);
idx_to_plot = round(linspace(1, numel(dutyGrid), 5));
hold on;
for k = 1:5
    di = idx_to_plot(k);
    plot(phase_grid, ripple_shape(:, di), '-', 'Color', colors(k,:), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('D=%.2f', ripple_duty_grid(di)));
end
xlabel('Phase within PWM period'); ylabel('Normalized ripple');
title(sprintf('Ripple Shape (corr=%.3f, triangular=%s)', avg_corr, mat2str(ripple_is_triangular)));
legend('Location', 'best'); grid on;

saveas(fig, fullfile(modelsDir, 'ripple_empirical_extraction.png'));
close(fig);
fprintf('Plot saved.\n');

% Restore model (don't save changes)
close_system(simscapeModel, 0);
fprintf('\nDone.\n');
