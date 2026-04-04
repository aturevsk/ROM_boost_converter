function compare_all_models()
% compare_all_models  Head-to-head comparison of PyTorch, DLT, and NSS models
%
% Evaluates all 3 models on the SAME validation profiles using the SAME
% protocol: free-running RK4 integration over the FULL profile length.
% Generates a comparison figure with Vout and iL for all 3 val profiles.

thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fileparts(thisDir);
addpath(genpath(repoRoot));

fprintf('=== Head-to-Head Model Comparison ===\n');

%% Load normalization stats
stats = jsondecode(fileread(fullfile(repoRoot, 'model_data', 'neuralode_norm_stats.json')));
ns.u_mean = single(stats.u_mean); ns.u_std = single(stats.u_std);
ns.x_mean = single(stats.x_mean(:)); ns.x_std = single(stats.x_std(:));
ns.dxdt_scale = single(stats.dxdt_std(:));

%% Load validation data (last 3 profiles, same as PyTorch)
csvDir = fullfile(repoRoot, 'data', 'neural_ode');
csvFiles = dir(fullfile(csvDir, 'profile_*.csv'));
[~,ord] = sort({csvFiles.name}); csvFiles = csvFiles(ord);
nTotal = numel(csvFiles);
val_u = cell(3,1); val_y = cell(3,1);
for k = 1:3
    raw = single(readmatrix(fullfile(csvDir, csvFiles(nTotal-3+k).name)));
    val_u{k} = raw(:,1);
    val_y{k} = raw(:,2:3);
end
fprintf('Loaded 3 val profiles (%d, %d, %d samples)\n', ...
    numel(val_u{1}), numel(val_u{2}), numel(val_u{3}));

%% Load PyTorch weights (raw matrices — most reliable)
W = load(fullfile(repoRoot, 'model_data', 'neuralode_weights.mat'));
fprintf('PyTorch weights loaded: W1=%s\n', mat2str(size(W.W1)));

%% Load DLT best model
cpDLT = load(fullfile(repoRoot, 'matlab_neural_ss', 'checkpoints_dlt', 'dlt_best.mat'));
netDLT = cpDLT.bestNet;
fprintf('DLT model loaded (best val=%.6f)\n', cpDLT.bestValLoss);
% Extract DLT weights for same forward pass
D.W1 = single(extractdata(netDLT.Learnables.Value{1}));
D.b1 = single(extractdata(netDLT.Learnables.Value{2}));
D.W2 = single(extractdata(netDLT.Learnables.Value{3}));
D.b2 = single(extractdata(netDLT.Learnables.Value{4}));
D.W3 = single(extractdata(netDLT.Learnables.Value{5}));
D.b3 = single(extractdata(netDLT.Learnables.Value{6}));

%% Load NSS expert model
cpNSS = load(fullfile(repoRoot, 'model_data', 'boost_nss_expert.mat'));
nssEst = cpNSS.nssEst;
nssNorm = cpNSS.normStats;
fprintf('NSS model loaded\n');

%% Integration parameters
TS = 5e-6; STEP_SKIP = 10; DT = TS * STEP_SKIP;

%% Run all models on all val profiles
rmse_table = zeros(3, 3, 2);  % [val_profile, model, state(Vout/iL)]
model_names = {'PyTorch Neural ODE', 'MATLAB DLT Neural ODE', 'MATLAB NSS (expert)'};
model_colors = {[0.0 0.5 0.0], [0.8 0.0 0.0], [0.0 0.0 0.8]};  % green, red, blue
model_styles = {'-', '--', ':'};

figure('Position', [50 50 1500 950], 'Color', 'w');

for v = 1:3
    u_v = val_u{v};
    y_v = val_y{v};
    T = numel(u_v);
    t_full = (0:T-1)' * TS * 1000;  % ms
    idx_sub = (1:STEP_SKIP:T)';
    t_sub = t_full(idx_sub);
    T_sub = numel(idx_sub);
    y_sub = y_v(idx_sub, :);

    % 1. PyTorch
    x0_pt = y_v(1,:)';
    pred_pt = rk4_run(single(W.W1), single(W.b1), single(W.W2), single(W.b2), ...
        single(W.W3), single(W.b3), ns.dxdt_scale, ...
        ns.x_mean, ns.x_std, ns.u_mean, ns.u_std, ...
        x0_pt, u_v, DT, STEP_SKIP);

    % 2. DLT
    pred_dlt = rk4_run(D.W1, D.b1, D.W2, D.b2, D.W3, D.b3, ns.dxdt_scale, ...
        ns.x_mean, ns.x_std, ns.u_mean, ns.u_std, ...
        x0_pt, u_v, DT, STEP_SKIP);

    % 3. NSS — use simulate() with iddata
    % NSS operates in normalized space, need to handle differently
    pred_nss = run_nss(nssEst, nssNorm, u_v, y_v, TS);

    % Compute RMSE (against subsampled ground truth)
    rmse_table(v, 1, 1) = sqrt(mean((y_sub(:,1) - pred_pt(1,1:T_sub)').^2));
    rmse_table(v, 1, 2) = sqrt(mean((y_sub(:,2) - pred_pt(2,1:T_sub)').^2));
    rmse_table(v, 2, 1) = sqrt(mean((y_sub(:,1) - pred_dlt(1,1:T_sub)').^2));
    rmse_table(v, 2, 2) = sqrt(mean((y_sub(:,2) - pred_dlt(2,1:T_sub)').^2));
    if ~isempty(pred_nss)
        n_nss = min(size(pred_nss,1), T_sub);
        rmse_table(v, 3, 1) = sqrt(mean((y_sub(1:n_nss,1) - pred_nss(1:n_nss,1)).^2));
        rmse_table(v, 3, 2) = sqrt(mean((y_sub(1:n_nss,2) - pred_nss(1:n_nss,2)).^2));
    end

    % Plot Vout
    subplot(3, 2, (v-1)*2 + 1);
    plot(t_sub, y_sub(:,1), 'k-', 'LineWidth', 2.0, 'DisplayName', 'Simscape'); hold on;
    plot(t_sub, pred_pt(1,1:T_sub), '-', 'Color', model_colors{1}, 'LineWidth', 1.3, ...
        'DisplayName', sprintf('PyTorch (%.3fV)', rmse_table(v,1,1)));
    plot(t_sub, pred_dlt(1,1:T_sub), '--', 'Color', model_colors{2}, 'LineWidth', 1.3, ...
        'DisplayName', sprintf('DLT (%.3fV)', rmse_table(v,2,1)));
    if ~isempty(pred_nss)
        t_nss = t_sub(1:n_nss);
        plot(t_nss, pred_nss(1:n_nss,1), ':', 'Color', model_colors{3}, 'LineWidth', 1.5, ...
            'DisplayName', sprintf('NSS (%.3fV)', rmse_table(v,3,1)));
    end
    ylabel('Vout (V)'); title(sprintf('Val %d: Vout', v));
    if v == 1, legend('Location', 'best', 'FontSize', 7); end
    grid on; set(gca, 'GridAlpha', 0.3);

    % Plot iL
    subplot(3, 2, (v-1)*2 + 2);
    plot(t_sub, y_sub(:,2), 'k-', 'LineWidth', 2.0, 'DisplayName', 'Simscape'); hold on;
    plot(t_sub, pred_pt(2,1:T_sub), '-', 'Color', model_colors{1}, 'LineWidth', 1.3, ...
        'DisplayName', sprintf('PyTorch (%.3fA)', rmse_table(v,1,2)));
    plot(t_sub, pred_dlt(2,1:T_sub), '--', 'Color', model_colors{2}, 'LineWidth', 1.3, ...
        'DisplayName', sprintf('DLT (%.3fA)', rmse_table(v,2,2)));
    if ~isempty(pred_nss)
        plot(t_nss, pred_nss(1:n_nss,2), ':', 'Color', model_colors{3}, 'LineWidth', 1.5, ...
            'DisplayName', sprintf('NSS (%.3fA)', rmse_table(v,3,2)));
    end
    ylabel('iL (A)'); title(sprintf('Val %d: iL', v));
    if v == 1, legend('Location', 'best', 'FontSize', 7); end
    grid on; set(gca, 'GridAlpha', 0.3);

    if v == 3, xlabel('Time (ms)'); end
end

sgtitle('Head-to-Head: PyTorch vs MATLAB DLT vs MATLAB NSS (Free-Running, Full Profile)', ...
    'FontWeight', 'bold', 'FontSize', 13);

% Print RMSE table
fprintf('\n=== RMSE Comparison (same protocol: free-running RK4, full profile) ===\n');
fprintf('%-8s  %-25s  %10s  %10s\n', 'Profile', 'Model', 'Vout RMSE', 'iL RMSE');
fprintf('%s\n', repmat('-', 1, 60));
for v = 1:3
    for m = 1:3
        fprintf('Val %d    %-25s  %8.3f V  %8.3f A\n', ...
            v, model_names{m}, rmse_table(v,m,1), rmse_table(v,m,2));
    end
    fprintf('%s\n', repmat('-', 1, 60));
end

% Average RMSE across all profiles
avg_rmse = squeeze(mean(rmse_table, 1));  % [3 models x 2 states]
fprintf('\n=== Average RMSE ===\n');
for m = 1:3
    fprintf('%-25s  Vout: %.3f V  |  iL: %.3f A\n', model_names{m}, avg_rmse(m,1), avg_rmse(m,2));
end

% Save figure
outFile = fullfile(repoRoot, 'results', 'head_to_head_comparison.png');
exportgraphics(gcf, outFile, 'Resolution', 200);
fprintf('\nFigure saved: %s\n', outFile);

end


%% ── RK4 integration using raw weight matrices ──────────────────────────────
function x_pred = rk4_run(W1, b1, W2, b2, W3, b3, dxdt_sc, ...
        x_mean, x_std, u_mean, u_std, x0, u_seq, dt, step_skip)
    T = numel(u_seq);
    T_sub = floor((T-1)/step_skip) + 1;
    x = x0(:);
    x_pred = zeros(2, T_sub);
    x_pred(:,1) = x;
    for k = 1:(T_sub-1)
        oi = min((k-1)*step_skip+1, T);
        d = u_seq(oi);
        k1 = mlp_fwd(x,             d, W1,b1,W2,b2,W3,b3,dxdt_sc,x_mean,x_std,u_mean,u_std);
        k2 = mlp_fwd(x+0.5*dt*k1,  d, W1,b1,W2,b2,W3,b3,dxdt_sc,x_mean,x_std,u_mean,u_std);
        k3 = mlp_fwd(x+0.5*dt*k2,  d, W1,b1,W2,b2,W3,b3,dxdt_sc,x_mean,x_std,u_mean,u_std);
        k4 = mlp_fwd(x+dt*k3,      d, W1,b1,W2,b2,W3,b3,dxdt_sc,x_mean,x_std,u_mean,u_std);
        x = x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
        x_pred(:,k+1) = x;
    end
end


%% ── MLP forward pass ────────────────────────────────────────────────────────
function dxdt = mlp_fwd(x, d, W1,b1,W2,b2,W3,b3,dxdt_sc,xm,xs,um,us)
    xn = (x(:) - xm(:)) ./ xs(:);
    dn = (d - um) / us;
    inp = [xn; dn];
    h1 = tanh(W1*inp + b1(:));
    h2 = tanh(W2*h1  + b2(:));
    raw = W3*h2 + b3(:);
    dxdt = raw(:) .* dxdt_sc(:);
end


%% ── NSS simulation ──────────────────────────────────────────────────────────
function pred = run_nss(nssEst, nssNorm, u_raw, y_raw, Ts)
% Run NSS model on validation data using sim()
% NSS was trained on normalized data — need to normalize input/output
try
    % Normalize inputs
    u_n = (u_raw - nssNorm.u_mean) ./ nssNorm.u_std;
    y_n = (y_raw - nssNorm.y_mean) ./ nssNorm.y_std;

    % Create iddata
    data = iddata(y_n, u_n, Ts);

    % Simulate (free-running)
    y_pred_n = sim(nssEst, data);

    % Denormalize output
    if isa(y_pred_n, 'iddata')
        y_pred_n = y_pred_n.OutputData;
    end
    pred = y_pred_n .* nssNorm.y_std + nssNorm.y_mean;
    fprintf('  NSS sim OK: %d samples\n', size(pred,1));
catch ME
    fprintf('  NSS sim failed: %s\n', ME.message);
    pred = [];
end
end
