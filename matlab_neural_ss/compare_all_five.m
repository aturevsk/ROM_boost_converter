function compare_all_five()
% compare_all_five  Head-to-head: Simscape vs PyTorch vs NSS vs DLT V1/V2/V3
%
% All models evaluated with free-running RK4 (dt=50us) on full validation
% profiles. Same protocol, same data, same integration — apples to apples.

thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fileparts(thisDir);
addpath(genpath(repoRoot));

fprintf('=== 5-Model Head-to-Head Comparison ===\n');

%% ── Shared constants ───────────────────────────────────────────────────────
TS = 5e-6; SKIP = 10; DT = TS * SKIP;

%% ── Load normalisation stats ───────────────────────────────────────────────
stats = jsondecode(fileread(fullfile(repoRoot,'model_data','neuralode_norm_stats.json')));
ns.u_mean = single(stats.u_mean); ns.u_std = single(stats.u_std);
ns.x_mean = single(stats.x_mean(:)); ns.x_std = single(stats.x_std(:));
ns.dxdt_scale = single(stats.dxdt_std(:));

%% ── Load validation data (last 3 profiles) ─────────────────────────────────
csvDir = fullfile(repoRoot,'data','neural_ode');
csvFiles = dir(fullfile(csvDir,'profile_*.csv'));
[~,ord] = sort({csvFiles.name}); csvFiles = csvFiles(ord);
nTotal = numel(csvFiles);
val_u = cell(3,1); val_y = cell(3,1);
for k = 1:3
    raw = single(readmatrix(fullfile(csvDir, csvFiles(nTotal-3+k).name)));
    val_u{k} = raw(:,1);
    val_y{k} = raw(:,2:3);
end
fprintf('Loaded 3 val profiles\n');

%% ── Load all model weights ─────────────────────────────────────────────────

% 1. PyTorch
W_pt = load(fullfile(repoRoot,'model_data','neuralode_weights.mat'));
W_pt.W1=single(W_pt.W1); W_pt.b1=single(W_pt.b1(:));
W_pt.W2=single(W_pt.W2); W_pt.b2=single(W_pt.b2(:));
W_pt.W3=single(W_pt.W3); W_pt.b3=single(W_pt.b3(:));
fprintf('PyTorch weights loaded\n');

% 2. DLT V1
cp1 = load(fullfile(repoRoot,'matlab_neural_ss','checkpoints_dlt','dlt_best.mat'));
W_v1 = extractWeights(cp1.bestNet);
fprintf('DLT V1 loaded (val=%.6f)\n', cp1.bestValLoss);

% 3. DLT V2
cp2 = load(fullfile(repoRoot,'matlab_neural_ss','checkpoints_dlt_v2','dlt_v2_best.mat'));
W_v2 = extractWeights(cp2.bestNet);
fprintf('DLT V2 loaded (val=%.6f)\n', cp2.bestValLoss);

% 4. DLT V3
cp3 = load(fullfile(repoRoot,'matlab_neural_ss','checkpoints_dlt_v3','dlt_v3_final.mat'));
W_v3 = extractWeights(cp3.bestNet);
fprintf('DLT V3 loaded (val=%.6f)\n', cp3.bestValLoss);

% 5. NSS expert
cpNSS = load(fullfile(repoRoot,'model_data','boost_nss_expert.mat'));
nssEst = cpNSS.nssEst;
nssNorm = cpNSS.normStats;
fprintf('NSS expert loaded\n');

%% ── Model metadata ─────────────────────────────────────────────────────────
models = {
    struct('name','PyTorch Neural ODE','W',W_pt, 'color',[0.0 0.6 0.0],'style','-','width',1.8)
    struct('name','DLT V1 (RK4, no cosine)','W',W_v1,'color',[0.8 0.0 0.0],'style','--','width',1.3)
    struct('name','DLT V2 (dlode45 adjoint)','W',W_v2,'color',[0.8 0.4 0.0],'style','-.','width',1.3)
    struct('name','DLT V3 (RK4 + cosine LR)','W',W_v3,'color',[0.6 0.0 0.6],'style',':','width',1.5)
    struct('name','NSS expert','W',[],'color',[0.0 0.0 0.8],'style',':','width',1.5)
};
nModels = numel(models);

%% ── Run all models on all profiles ──────────────────────────────────────────
rmse = zeros(3, nModels, 2);  % [profile, model, Vout/iL]

fig = figure('Position',[30 30 1600 1000],'Color','w');
set(fig, 'defaultAxesColor', 'w', 'defaultAxesXColor', 'k', ...
    'defaultAxesYColor', 'k', 'defaultAxesGridColor', [0.5 0.5 0.5]);

for v = 1:3
    u_v = val_u{v}; y_v = val_y{v};
    T = numel(u_v);
    idx_sub = (1:SKIP:T)';
    T_sub = numel(idx_sub);
    t_sub = (idx_sub - 1) * TS * 1000;  % ms
    y_sub = y_v(idx_sub,:);

    preds = cell(nModels, 1);

    % Models 1-4: RK4 with raw weights
    for m = 1:4
        W = models{m}.W;
        preds{m} = rk4_full(W, ns, y_v(1,:)', u_v, DT, SKIP, T_sub);
        rmse(v,m,1) = sqrt(mean((y_sub(:,1) - preds{m}(1,1:T_sub)').^2));
        rmse(v,m,2) = sqrt(mean((y_sub(:,2) - preds{m}(2,1:T_sub)').^2));
    end

    % Model 5: NSS via sim()
    preds{5} = run_nss(nssEst, nssNorm, u_v, y_v, TS);
    if ~isempty(preds{5})
        n5 = min(size(preds{5},1), T_sub);
        rmse(v,5,1) = sqrt(mean((y_sub(1:n5,1) - preds{5}(1:n5,1)).^2));
        rmse(v,5,2) = sqrt(mean((y_sub(1:n5,2) - preds{5}(1:n5,2)).^2));
    end

    % ── Plot Vout ────────────────────────────────────────────────────────
    subplot(3,2,(v-1)*2+1);
    plot(t_sub, y_sub(:,1), '-', 'Color',[0.3 0.3 0.3], 'LineWidth',2.5, 'DisplayName','Simscape'); hold on;
    for m = 1:4
        plot(t_sub, preds{m}(1,1:T_sub), ...
            models{m}.style, 'Color',models{m}.color, 'LineWidth',models{m}.width, ...
            'DisplayName', sprintf('%s (%.3fV)', models{m}.name, rmse(v,m,1)));
    end
    if ~isempty(preds{5})
        t_nss = t_sub(1:n5);
        plot(t_nss, preds{5}(1:n5,1), ...
            models{5}.style, 'Color',models{5}.color, 'LineWidth',models{5}.width, ...
            'DisplayName', sprintf('%s (%.3fV)', models{5}.name, rmse(v,5,1)));
    end
    ylabel('Vout (V)'); title(sprintf('Val %d: Vout', v));
    if v == 1, legend('Location','best','FontSize',6.5); end
    grid on; set(gca,'GridAlpha',0.3,'Color','w');

    % ── Plot iL ──────────────────────────────────────────────────────────
    subplot(3,2,(v-1)*2+2);
    plot(t_sub, y_sub(:,2), '-', 'Color',[0.3 0.3 0.3], 'LineWidth',2.5, 'DisplayName','Simscape'); hold on;
    for m = 1:4
        plot(t_sub, preds{m}(2,1:T_sub), ...
            models{m}.style, 'Color',models{m}.color, 'LineWidth',models{m}.width, ...
            'DisplayName', sprintf('%s (%.3fA)', models{m}.name, rmse(v,m,2)));
    end
    if ~isempty(preds{5})
        plot(t_nss, preds{5}(1:n5,2), ...
            models{5}.style, 'Color',models{5}.color, 'LineWidth',models{5}.width, ...
            'DisplayName', sprintf('%s (%.3fA)', models{5}.name, rmse(v,5,2)));
    end
    ylabel('iL (A)'); title(sprintf('Val %d: iL', v));
    if v == 1, legend('Location','best','FontSize',6.5); end
    grid on; set(gca,'GridAlpha',0.3,'Color','w');

    if v == 3, xlabel('Time (ms)'); end
end

sgtitle({'Head-to-Head: Simscape vs PyTorch vs DLT V1/V2/V3 vs NSS', ...
    'Free-running RK4 on full validation profiles'}, ...
    'FontWeight','bold','FontSize',13);

%% ── Print RMSE table ────────────────────────────────────────────────────────
fprintf('\n=== RMSE Comparison ===\n');
fprintf('%-28s  %10s  %10s\n', 'Model', 'Vout RMSE', 'iL RMSE');
fprintf('%s\n', repmat('-',1,52));
for m = 1:nModels
    avg_v = mean(rmse(:,m,1));
    avg_i = mean(rmse(:,m,2));
    fprintf('%-28s  %8.3f V  %8.3f A\n', models{m}.name, avg_v, avg_i);
end

%% ── Save ────────────────────────────────────────────────────────────────────
outFile = fullfile(repoRoot,'results','all_models_comparison.png');
exportgraphics(gcf, outFile, 'Resolution',200, 'BackgroundColor','white');
fprintf('\nFigure saved: %s\n', outFile);

end


%% ── Extract weights from dlnetwork ──────────────────────────────────────────
function W = extractWeights(net)
    W.W1 = single(extractdata(net.Learnables.Value{1}));
    W.b1 = single(extractdata(net.Learnables.Value{2}(:)));
    W.W2 = single(extractdata(net.Learnables.Value{3}));
    W.b2 = single(extractdata(net.Learnables.Value{4}(:)));
    W.W3 = single(extractdata(net.Learnables.Value{5}));
    W.b3 = single(extractdata(net.Learnables.Value{6}(:)));
end


%% ── RK4 integration with raw weights ────────────────────────────────────────
function xp = rk4_full(W, ns, x0, u, dt, skip, T_sub)
    T = numel(u);
    x = single(x0(:));
    xp = zeros(2, T_sub, 'single');
    xp(:,1) = x;
    for k = 1:T_sub-1
        oi = min((k-1)*skip+1, T);
        d = u(oi);
        k1 = mlp(x,           d, W, ns);
        k2 = mlp(x+.5*dt*k1, d, W, ns);
        k3 = mlp(x+.5*dt*k2, d, W, ns);
        k4 = mlp(x+dt*k3,    d, W, ns);
        x = x + (dt/6)*(k1+2*k2+2*k3+k4);
        xp(:,k+1) = x;
    end
end


%% ── MLP forward ─────────────────────────────────────────────────────────────
function dxdt = mlp(x, d, W, ns)
    xn = (x - ns.x_mean) ./ ns.x_std;
    dn = (d - ns.u_mean) / ns.u_std;
    h1 = tanh(W.W1*[xn;dn] + W.b1);
    h2 = tanh(W.W2*h1 + W.b2);
    dxdt = (W.W3*h2 + W.b3) .* ns.dxdt_scale;
end


%% ── NSS simulation ──────────────────────────────────────────────────────────
function pred = run_nss(nssEst, nssNorm, u_raw, y_raw, Ts)
    try
        u_n = (u_raw - nssNorm.u_mean) ./ nssNorm.u_std;
        y_n = (y_raw - nssNorm.y_mean) ./ nssNorm.y_std;
        data = iddata(y_n, u_n, Ts);
        yp = sim(nssEst, data);
        if isa(yp,'iddata'), yp = yp.OutputData; end
        pred = yp .* nssNorm.y_std + nssNorm.y_mean;
    catch
        pred = [];
    end
end
