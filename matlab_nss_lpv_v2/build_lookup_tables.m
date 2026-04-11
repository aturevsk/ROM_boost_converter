function lut = build_lookup_tables(repoRoot, ns)
% build_lookup_tables  Extract A, B, c lookup tables from Branch B data.
%
% Converts physical-space A-matrices to normalized space, estimates B from
% steady-state gradients, and computes c from steady-state conditions.
%
% Returns struct with fields:
%   D_grid     - [12x1] duty cycle grid points
%   A_norm     - [2x2x12] A-matrices in normalized space
%   B_norm     - [2x1x12] B-vectors in normalized space
%   c_norm     - [2x1x12] c-vectors in normalized space
%   A_phys     - [2x2x12] A-matrices in physical CT space (for reference)
%   B_phys     - [2x1x12] B-vectors in physical space
%   x_ss       - [12x2] steady-state [Vout, iL] at each duty
%   x_ss_n     - [12x2] normalized steady-state

% Load Jacobian targets (CT physical A-matrices)
jacFile = fullfile(repoRoot, 'data', 'neural_ode', 'jacobian_targets.json');
jd = jsondecode(fileread(jacFile));
A_phys_all = jd.A_targets;  % may be [2x2x12] or [12x2x2]
x_ops = jd.x_ops;           % [12x2] physical [Vout, iL]
D_grid = jd.u_ops(:);       % [12x1] duty cycle grid
nPts = numel(D_grid);

% Ensure A is [2x2x12]
if size(A_phys_all,1) == 2 && size(A_phys_all,3) == nPts
    % already [2x2x12]
elseif size(A_phys_all,1) == nPts
    A_phys_all = permute(A_phys_all, [2 3 1]);  % [12x2x2] -> [2x2x12]
end

% Load steady-state values from Branch B data
bbFile = fullfile(repoRoot, 'branch_b_lpv', 'models', 'boost_openloop_branch_b_data.mat');
bb = load(bbFile, 'Vout_ss_grid', 'iL_ss_grid');
Vout_ss = bb.Vout_ss_grid(:);  % [12x1]
iL_ss = bb.iL_ss_grid(:);     % [12x1]
x_ss = [Vout_ss, iL_ss];      % [12x2]

fprintf('  Building lookup tables from %d operating points\n', nPts);
fprintf('  D_grid: [%.2f ... %.2f]\n', D_grid(1), D_grid(end));

%% ── Convert A to normalized space ──────────────────────────────────────────
% Physical: dxdt(i) = sum_j A_phys(i,j) * x(j) + ...
% Normalized: dxdt_n(i) = sum_j A_norm(i,j) * x_n(j) + ...
% where dxdt = dxdt_scale .* dxdt_n, x_n = (x - x_mean) / x_std
% So: A_norm(i,j) = A_phys(i,j) * x_std(j) / dxdt_scale(i)

A_norm = zeros(2, 2, nPts);
for k = 1:nPts
    A_k = squeeze(A_phys_all(:,:,k));  % [2x2] physical CT
    for i = 1:2
        for j = 1:2
            A_norm(i,j,k) = A_k(i,j) * ns.x_std(j) / ns.dxdt_scale(i);
        end
    end
end
fprintf('  A_norm sample at D=%.2f: [[%.3f,%.3f],[%.3f,%.3f]]\n', ...
    D_grid(1), A_norm(1,1,1), A_norm(1,2,1), A_norm(2,1,1), A_norm(2,2,1));

%% ── Estimate B from steady-state gradients ─────────────────────────────────
% At steady state: 0 = A * (x_ss - x_mean) + B * (D - u_mean) + offset
% Differentiating w.r.t. D: 0 = A * dx_ss/dD + B
% So: B_phys = -A_phys * dx_ss/dD
%
% dx_ss/dD estimated via centered finite differences

dVout_dD = zeros(nPts, 1);
diL_dD = zeros(nPts, 1);
for k = 1:nPts
    if k == 1
        dVout_dD(k) = (Vout_ss(2) - Vout_ss(1)) / (D_grid(2) - D_grid(1));
        diL_dD(k) = (iL_ss(2) - iL_ss(1)) / (D_grid(2) - D_grid(1));
    elseif k == nPts
        dVout_dD(k) = (Vout_ss(end) - Vout_ss(end-1)) / (D_grid(end) - D_grid(end-1));
        diL_dD(k) = (iL_ss(end) - iL_ss(end-1)) / (D_grid(end) - D_grid(end-1));
    else
        dVout_dD(k) = (Vout_ss(k+1) - Vout_ss(k-1)) / (D_grid(k+1) - D_grid(k-1));
        diL_dD(k) = (iL_ss(k+1) - iL_ss(k-1)) / (D_grid(k+1) - D_grid(k-1));
    end
end

B_phys = zeros(2, 1, nPts);
B_norm = zeros(2, 1, nPts);
for k = 1:nPts
    A_k = squeeze(A_phys_all(:,:,k));
    dx_dD = [dVout_dD(k); diL_dD(k)];
    B_phys(:,:,k) = -A_k * dx_dD;
    % Convert to normalized: B_norm(i) = B_phys(i) * u_std / dxdt_scale(i)
    B_norm(:,:,k) = B_phys(:,:,k) * ns.u_std ./ ns.dxdt_scale(:);
end
fprintf('  B_phys sample at D=%.2f: [%.1f, %.1f]\n', ...
    D_grid(1), B_phys(1,1,1), B_phys(2,1,1));

%% ── Compute c from steady-state conditions ─────────────────────────────────
% At steady state, dxdt = 0:
%   0 = dxdt_scale .* (A_norm * x_ss_n + B_norm * u_ss_n + c_norm)
%   c_norm = -(A_norm * x_ss_n + B_norm * u_ss_n)

x_ss_n = (x_ss - ns.y_mean) ./ ns.y_std;  % [12x2]
u_ss_n = (D_grid - ns.u_mean) / ns.u_std;  % [12x1]

c_norm = zeros(2, 1, nPts);
for k = 1:nPts
    A_k = squeeze(A_norm(:,:,k));
    B_k = squeeze(B_norm(:,:,k));
    c_norm(:,:,k) = -(A_k * x_ss_n(k,:)' + B_k * u_ss_n(k));
end
fprintf('  c_norm sample at D=%.2f: [%.4f, %.4f]\n', ...
    D_grid(1), c_norm(1,1,1), c_norm(2,1,1));

%% ── Pack output ────────────────────────────────────────────────────────────
lut.D_grid = D_grid;
lut.A_norm = A_norm;
lut.B_norm = B_norm;
lut.c_norm = c_norm;
lut.A_phys = A_phys_all;
lut.B_phys = B_phys;
lut.x_ss = x_ss;
lut.x_ss_n = x_ss_n;
lut.nPts = nPts;

fprintf('  Lookup tables built: %d points, A[2x2], B[2x1], c[2x1]\n', nPts);
end
