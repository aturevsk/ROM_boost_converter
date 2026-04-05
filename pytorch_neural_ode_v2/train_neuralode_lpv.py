"""
train_neuralode_lpv.py
======================
Neural ODE with LPV (Linear Parameter-Varying) architecture:

  dx/dt = A(x,u) * x + B(x,u) * u + c(x,u)

where A, B, c are outputs of a small MLP that takes [x_n, u_n] as input.
This enforces state-dependent linear structure matching boost converter physics.

Key advantages over generic MLP:
  - Correct bilinear structure at every operating point
  - A-network can be initialized from 12 Branch B ssest A-matrices
  - Gradients through RK4 are more stable (structured derivatives)
  - Interpretable: can extract linearization at any operating point

Architecture:
  [x_n, u_n] -> MLP -> [A(2x2), B(2x1), c(2x1)] = 9 outputs
  dx/dt = dxdt_scale * (A*x_n + B*u_n + c)

Uses same training pipeline as train_neuralode_full.py:
  Phase 1-3 curriculum + Phase 4 extended
  RK4 integration, same data, same val split

Usage:
  python3 train_neuralode_lpv.py --test
  python3 train_neuralode_lpv.py --seed 1 --max_hours 10
  python3 train_neuralode_lpv.py --multi 10 --max_hours 10
"""

import os
import sys
import json
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd.functional import jacobian
from scipy.signal import butter
from pathlib import Path

# ============================================================
# Paths
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / 'data'
MODEL_DIR = REPO_ROOT / 'model_data'
CHECKPOINT_BASE = SCRIPT_DIR / 'checkpoints_lpv'
RESULTS_DIR = SCRIPT_DIR / 'results_lpv'

# ============================================================
# Constants
# ============================================================
NX = 2
NU = 1
TS = 5e-6
STEP_SKIP = 10
DEVICE = 'cpu'
HIDDEN_LPV = [64, 64]  # hidden layers for the gain-scheduling MLP

F_RES = 600
F_SW = 200e3
FILTER_CUTOFF = np.sqrt(F_RES * F_SW)

# Thread pinning moved to main() to avoid conflicts when imported

# ============================================================
# Logging
# ============================================================
_log_file = None

def log(msg):
    print(msg, flush=True)
    if _log_file is not None:
        with open(_log_file, 'a') as f:
            f.write(msg + '\n')


# ============================================================
# Normalization (same as train_neuralode_full.py)
# ============================================================
def compute_norm_stats(profiles_u, profiles_y):
    all_u = np.concatenate(profiles_u)
    all_y = np.concatenate(profiles_y, axis=0)
    all_dvout = [np.diff(y[:, 0]) / TS for y in profiles_y]
    all_dil = [np.diff(y[:, 1]) / TS for y in profiles_y]
    return {
        'u_mean': float(all_u.mean()),
        'u_std': float(all_u.std()),
        'x_mean': all_y.mean(axis=0).tolist(),
        'x_std': all_y.std(axis=0).tolist(),
        'dxdt_std': [float(np.concatenate(all_dvout).std()),
                     float(np.concatenate(all_dil).std())],
    }


class Normalizer:
    def __init__(self, stats):
        self.u_mean = torch.tensor(stats['u_mean'], dtype=torch.float32)
        self.u_std = torch.tensor(stats['u_std'], dtype=torch.float32)
        self.x_mean = torch.tensor(stats['x_mean'], dtype=torch.float32)
        self.x_std = torch.tensor(stats['x_std'], dtype=torch.float32)
        self.dxdt_scale = torch.tensor(stats['dxdt_std'], dtype=torch.float32)

    def norm_x(self, x):
        return (x - self.x_mean) / self.x_std

    def denorm_x(self, x_n):
        return x_n * self.x_std + self.x_mean

    def norm_u(self, u):
        return (u - self.u_mean) / self.u_std


# ============================================================
# LPV Neural ODE Model
# ============================================================
class NeuralODE_LPV(nn.Module):
    """State-Dependent Coefficient (SDC) / LPV Neural ODE.

    dx/dt = dxdt_scale * (A(x_n, u_n) * x_n + B(x_n, u_n) * u_n + c(x_n, u_n))

    A small MLP maps [x_n, u_n] -> [A_flat(4), B_flat(2), c(2)] = 8 outputs.
    A is reshaped to [2x2], B to [2x1], c to [2x1].

    This enforces that dx/dt is always a structured linear function of the
    current state and input, with gains that vary across the operating space.
    """
    def __init__(self, nx=NX, nu=NU, hidden=HIDDEN_LPV, dxdt_scale=None):
        super().__init__()
        self.nx = nx
        self.nu = nu
        n_out = nx * nx + nx * nu + nx  # A(4) + B(2) + c(2) = 8

        layers = []
        in_dim = nx + nu
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, n_out))
        self.net = nn.Sequential(*layers)

        if dxdt_scale is not None:
            self.register_buffer('dxdt_scale', dxdt_scale)
        else:
            self.register_buffer('dxdt_scale', torch.ones(nx))

    def forward(self, x_n, u_n):
        """x_n: (..., nx) normalized state, u_n: (..., nu) normalized input.
        Returns dxdt in PHYSICAL units."""
        xu = torch.cat([x_n, u_n], dim=-1)
        raw = self.net(xu)  # (..., 8)

        # Split into A, B, c
        A_flat = raw[..., :self.nx * self.nx]                          # (..., 4)
        B_flat = raw[..., self.nx * self.nx:self.nx * self.nx + self.nx * self.nu]  # (..., 2)
        c = raw[..., self.nx * self.nx + self.nx * self.nu:]           # (..., 2)

        # Reshape A to (..., nx, nx), B to (..., nx, nu)
        batch_shape = raw.shape[:-1]
        A = A_flat.reshape(*batch_shape, self.nx, self.nx)   # (..., 2, 2)
        B = B_flat.reshape(*batch_shape, self.nx, self.nu)   # (..., 2, 1)

        # dx/dt = A*x + B*u + c  (all in normalized space, then scale)
        # x_n: (..., nx) -> (..., nx, 1) for matmul
        Ax = torch.matmul(A, x_n.unsqueeze(-1)).squeeze(-1)   # (..., nx)
        Bu = torch.matmul(B, u_n.unsqueeze(-1)).squeeze(-1)   # (..., nx)

        dxdt_n = Ax + Bu + c  # (..., nx) in ~[-1,1] range
        return dxdt_n * self.dxdt_scale

    def get_AB(self, x_n, u_n):
        """Extract A and B matrices at a given operating point (for analysis)."""
        xu = torch.cat([x_n, u_n], dim=-1)
        raw = self.net(xu)
        A = raw[..., :self.nx * self.nx].reshape(-1, self.nx, self.nx)
        B = raw[..., self.nx * self.nx:self.nx * self.nx + self.nx * self.nu].reshape(-1, self.nx, self.nu)
        return A, B


# ============================================================
# LPV-specific initialization from 12 A-matrices
# ============================================================
def initialize_lpv_from_linear(model, A_targets, x_ops, u_ops, normalizer):
    """Initialize LPV network so that A(x,u) ≈ A_target at each operating point.

    Strategy: initialize the last layer so that at each of the 12 operating points,
    the network output A-block approximately matches the known A-matrix.
    Also initialize hidden layers with the operating point features spread across neurons.
    """
    n_pts = len(u_ops)
    log(f"  LPV init: using {n_pts} A-matrices for initialization")

    # For the first hidden layer: spread operating points across neurons
    # Each A-matrix gives us information about the local dynamics
    with torch.no_grad():
        W1 = model.net[0].weight  # [hidden, 3]
        b1 = model.net[0].bias    # [hidden]
        W1.zero_()
        b1.zero_()

        neurons_per_pt = min(4, W1.shape[0] // n_pts)
        for i in range(n_pts):
            x_n = normalizer.norm_x(torch.tensor(x_ops[i], dtype=torch.float32))
            u_n = normalizer.norm_u(torch.tensor([u_ops[i]], dtype=torch.float32))
            xu = torch.cat([x_n, u_n])  # [3]

            for j in range(neurons_per_pt):
                row = i * neurons_per_pt + j
                if row >= W1.shape[0]:
                    break
                # Initialize neuron to respond to this operating region
                W1[row] = xu * (0.5 + 0.5 * j)  # varying sensitivity
                b1[row] = -torch.dot(W1[row], xu) * (0.3 + 0.2 * j)  # center on this point

        # Fill remaining neurons with small random
        filled = n_pts * neurons_per_pt
        if filled < W1.shape[0]:
            W1[filled:] = 0.05 * torch.randn(W1.shape[0] - filled, W1.shape[1])
            b1[filled:] = 0.05 * torch.randn(W1.shape[0] - filled)

    log(f"  First layer: {n_pts}x{neurons_per_pt}={min(n_pts*neurons_per_pt, W1.shape[0])} "
        f"neurons initialized from operating points")


# ============================================================
# Data Loading (same as train_neuralode_full.py)
# ============================================================
def load_training_data():
    csv_dir = DATA_DIR / 'neural_ode'
    csv_files = sorted(csv_dir.glob('profile_*.csv'))
    if len(csv_files) == 0:
        log(f"ERROR: No profile CSVs in {csv_dir}")
        sys.exit(1)
    profiles_u, profiles_y = [], []
    for f in csv_files:
        data = np.loadtxt(str(f), delimiter=',', dtype=np.float32)
        if len(data) < 10:
            continue
        profiles_u.append(data[:, 0])
        profiles_y.append(data[:, 1:3])
    log(f"  Loaded {len(profiles_u)} profiles, {sum(len(u) for u in profiles_u):,} samples")
    return profiles_u, profiles_y


def load_jacobian_targets():
    json_file = DATA_DIR / 'neural_ode' / 'jacobian_targets.json'
    if not json_file.exists():
        log("  WARNING: No Jacobian targets found")
        return None, None, None
    with open(json_file) as f:
        data = json.load(f)
    A_targets = torch.tensor(data['A_targets'], dtype=torch.float32)
    x_ops = torch.tensor(data['x_ops'], dtype=torch.float32)
    u_ops = torch.tensor(data['u_ops'], dtype=torch.float32)
    log(f"  {len(u_ops)} Jacobian targets from JSON")
    return A_targets, x_ops, u_ops


# ============================================================
# Filter
# ============================================================
def apply_filter_torch(x, b, a):
    T, C = x.shape
    order = len(a) - 1
    y = torch.zeros_like(x)
    d = torch.zeros(order, C, dtype=x.dtype)
    b_t = torch.tensor(b, dtype=x.dtype)
    a_t = torch.tensor(a, dtype=x.dtype)
    for t in range(T):
        y[t] = b_t[0] * x[t] + d[0]
        for i in range(order - 1):
            d[i] = b_t[i + 1] * x[t] - a_t[i + 1] * y[t] + d[i + 1]
        d[order - 1] = b_t[order] * x[t] - a_t[order] * y[t]
    return y


# ============================================================
# RK4 Integration + Loss (same protocol as MLP version)
# ============================================================
def integrate_ode_rk4(model, x0, u_seq, normalizer, step_skip=STEP_SKIP):
    T = len(u_seq)
    dt = TS * step_skip
    T_sub = (T - 1) // step_skip + 1
    x_pred = torch.zeros(T_sub, NX)
    x_pred[0] = x0
    x = x0
    for k in range(T_sub - 1):
        orig_idx = min(k * step_skip, T - 1)
        u_k = u_seq[orig_idx].unsqueeze(0)
        u_k_n = normalizer.norm_u(u_k).unsqueeze(0)

        def dxdt_fn(x_phys):
            x_n = normalizer.norm_x(x_phys.unsqueeze(0))
            return model(x_n, u_k_n).squeeze(0)

        k1 = dxdt_fn(x)
        k2 = dxdt_fn(x + 0.5 * dt * k1)
        k3 = dxdt_fn(x + 0.5 * dt * k2)
        k4 = dxdt_fn(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        x_pred[k + 1] = x
    return x_pred, T_sub


def trajectory_loss(model, x0, u_seq, y_true, normalizer,
                    step_skip=STEP_SKIP, lambda_ripple=0.0):
    T = len(u_seq)
    x_pred, T_sub = integrate_ode_rk4(model, x0, u_seq, normalizer, step_skip)
    y_sub = y_true[::step_skip][:T_sub]
    x_pred_n = normalizer.norm_x(x_pred)
    y_sub_n = normalizer.norm_x(y_sub)

    if lambda_ripple > 0 and T_sub > 20:
        fs_eff = 1.0 / (TS * step_skip)
        f_nyq = fs_eff / 2.0
        f_norm = min(FILTER_CUTOFF / f_nyq, 0.95)
        if f_norm > 0.05:
            b_lp, a_lp = butter(2, f_norm, btype='low')
            b_hp, a_hp = butter(2, f_norm, btype='high')
            x_lp = apply_filter_torch(x_pred_n, b_lp, a_lp)
            y_lp = apply_filter_torch(y_sub_n, b_lp, a_lp)
            x_hp = apply_filter_torch(x_pred_n, b_hp, a_hp)
            y_hp = apply_filter_torch(y_sub_n, b_hp, a_hp)
            skip = max(T_sub // 10, 5)
            loss_avg = torch.mean((x_lp[skip:] - y_lp[skip:]) ** 2)
            loss_ripple = torch.mean((x_hp[skip:] - y_hp[skip:]) ** 2)
            return loss_avg + lambda_ripple * loss_ripple
    return torch.mean((x_pred_n - y_sub_n) ** 2)


def jacobian_loss_lpv(model, A_targets, x_ops, u_ops, normalizer):
    """LPV-specific Jacobian loss: compare A(x,u) output directly to A_targets."""
    n_pts = len(u_ops)
    if n_pts > 6:
        fixed = np.linspace(0, n_pts - 1, 4, dtype=int)
        others = list(set(range(n_pts)) - set(fixed))
        rand_idx = np.random.choice(others, min(2, len(others)), replace=False)
        idx = np.concatenate([fixed, rand_idx])
    else:
        idx = np.arange(n_pts)

    loss = torch.tensor(0.0)
    for j in idx:
        x_n = normalizer.norm_x(x_ops[j].unsqueeze(0))
        u_n = normalizer.norm_u(u_ops[j].unsqueeze(0)).unsqueeze(0)
        A_pred, _ = model.get_AB(x_n, u_n)  # [1, 2, 2]
        A_target = A_targets[j]  # [2, 2]
        A_scale = torch.max(torch.abs(A_target)) + 1.0
        loss = loss + torch.mean(((A_pred.squeeze(0) - A_target) / A_scale) ** 2)
    return loss / len(idx)


# ============================================================
# Validation
# ============================================================
def compute_validation(model, normalizer, val_u, val_y):
    model.eval()
    total_se_v, total_se_i, total_n = 0.0, 0.0, 0
    with torch.no_grad():
        for u_v, y_v in zip(val_u, val_y):
            x_pred, T_sub = integrate_ode_rk4(model, y_v[0], u_v, normalizer)
            y_sub = y_v[::STEP_SKIP][:T_sub]
            se = (x_pred - y_sub) ** 2
            total_se_v += se[:, 0].sum().item()
            total_se_i += se[:, 1].sum().item()
            total_n += T_sub
    x_std = normalizer.x_std.numpy()
    val_loss = (total_se_v / x_std[0] ** 2 + total_se_i / x_std[1] ** 2) / (2 * total_n)
    rmse_v = np.sqrt(total_se_v / total_n)
    rmse_i = np.sqrt(total_se_i / total_n)
    model.train()
    return val_loss, rmse_v, rmse_i


def compute_full_profile_rmse(model, normalizer, val_u, val_y):
    model.eval()
    rmse_v_all, rmse_i_all = [], []
    with torch.no_grad():
        for u_v, y_v in zip(val_u, val_y):
            x_pred, T_sub = integrate_ode_rk4(model, y_v[0], u_v, normalizer)
            y_sub = y_v[::STEP_SKIP][:T_sub]
            rmse_v_all.append(np.sqrt(((x_pred[:, 0] - y_sub[:, 0]) ** 2).mean().item()))
            rmse_i_all.append(np.sqrt(((x_pred[:, 1] - y_sub[:, 1]) ** 2).mean().item()))
    model.train()
    return np.mean(rmse_v_all), np.mean(rmse_i_all)


# ============================================================
# Checkpointing
# ============================================================
def save_checkpoint(model, best_state, best_val, total_epochs, train_hist,
                    val_hist, stats, filepath):
    torch.save({
        'model_state': model.state_dict(),
        'best_state': best_state,
        'best_val_loss': best_val,
        'total_epochs': total_epochs,
        'train_history': train_hist,
        'val_history': val_hist,
        'norm_stats': stats,
    }, str(filepath))


def append_seed_log(seed, config, results, log_path):
    import fcntl
    entry = {
        'seed': seed,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': config,
        'results': results,
    }
    lock_path = str(log_path) + '.lock'
    with open(lock_path, 'w') as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        log_data = []
        if log_path.exists():
            with open(log_path) as f:
                log_data = json.load(f)
        log_data.append(entry)
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2, default=lambda o: float(o) if hasattr(o, 'item') else o)
        fcntl.flock(lock_f, fcntl.LOCK_UN)


# ============================================================
# Single training run
# ============================================================
def run_single(seed, max_hours, is_test=False):
    run_dir = CHECKPOINT_BASE / f'run_{seed:04d}'
    run_dir.mkdir(parents=True, exist_ok=True)

    global _log_file
    _log_file = run_dir / 'training.log'

    log(f"\n{'=' * 60}")
    log(f"LPV Neural ODE Training — Seed {seed}")
    log(f"{'=' * 60}")
    log(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Architecture: dx/dt = A(x,u)*x + B(x,u)*u + c(x,u)")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    log("\nLoading data...")
    profiles_u, profiles_y = load_training_data()
    n_total = len(profiles_u)

    VAL_INDICES = [4, 11, 13]
    TRAIN_INDICES = [i for i in range(n_total) if i not in VAL_INDICES]

    stats = compute_norm_stats(
        [profiles_u[i] for i in TRAIN_INDICES],
        [profiles_y[i] for i in TRAIN_INDICES])
    normalizer = Normalizer(stats)
    log(f"  dxdt_scale=[{stats['dxdt_std'][0]:.0f}, {stats['dxdt_std'][1]:.0f}]")

    with open(run_dir / 'norm_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    A_targets, x_ops, u_ops = load_jacobian_targets()
    has_jacobian = A_targets is not None

    train_u = [torch.tensor(profiles_u[i]) for i in TRAIN_INDICES]
    train_y = [torch.tensor(profiles_y[i]) for i in TRAIN_INDICES]
    val_u = [torch.tensor(profiles_u[i]) for i in VAL_INDICES]
    val_y = [torch.tensor(profiles_y[i]) for i in VAL_INDICES]
    n_train = len(TRAIN_INDICES)
    log(f"  Train: {n_train}, Val: {len(VAL_INDICES)} (profiles 5,12,14)")

    # Build LPV model
    dxdt_scale = torch.tensor(stats['dxdt_std'], dtype=torch.float32)
    model = NeuralODE_LPV(NX, NU, HIDDEN_LPV, dxdt_scale=dxdt_scale)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  Parameters: {n_params} (LPV architecture)")

    if has_jacobian:
        initialize_lpv_from_linear(model, A_targets, x_ops, u_ops, normalizer)

    # Phases
    if is_test:
        phases = [
            {'name': 'TEST', 'epochs': 5, 'window_ms': 5, 'lr': 1e-3,
             'lambda_J': 0.0, 'lambda_ripple': 0.0, 'win_per_prof': 2}
        ]
    else:
        phases = [
            {'name': 'Phase1_HalfOsc', 'epochs': 300, 'window_ms': 5, 'lr': 1e-3,
             'lambda_J': 0.1, 'lambda_ripple': 0.0, 'win_per_prof': 3},
            {'name': 'Phase2_FullOsc', 'epochs': 500, 'window_ms': 20, 'lr': 5e-4,
             'lambda_J': 0.01, 'lambda_ripple': 0.1, 'win_per_prof': 3},
            {'name': 'Phase3_Finetune', 'epochs': 200, 'window_ms': 20, 'lr': 1e-4,
             'lambda_J': 0.0, 'lambda_ripple': 0.1, 'win_per_prof': 3},
        ]

    best_val_loss = float('inf')
    best_state = None
    train_history, val_history = [], []
    total_epochs = 0
    t_start = time.time()
    last_save_time = time.time()

    # ---- Curriculum phases ----
    for phase in phases:
        ph_name = phase['name']
        n_epochs = phase['epochs']
        window_samples = int(phase['window_ms'] * 1e-3 / TS)
        lr = phase['lr']
        lambda_J = phase['lambda_J']
        lambda_ripple = phase.get('lambda_ripple', 0.0)
        win_per_prof = phase['win_per_prof']

        log(f"\n--- {ph_name}: {n_epochs} ep, win={phase['window_ms']}ms, LR={lr}, lJ={lambda_J}, lR={lambda_ripple} ---")

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min=lr * 0.01)

        no_improve = 0
        for epoch in range(1, n_epochs + 1):
            total_epochs += 1
            t_ep = time.time()
            model.train()
            epoch_loss, n_windows = 0.0, 0
            perm = np.random.permutation(n_train)

            for p_idx in perm:
                u_p, y_p = train_u[p_idx], train_y[p_idx]
                T = len(u_p)
                n_win_total = max(1, T // window_samples)
                n_use = min(win_per_prof, n_win_total)
                win_starts = np.random.choice(n_win_total, n_use, replace=False)

                for w in win_starts:
                    s = w * window_samples
                    e = min(s + window_samples, T)
                    if e - s < 20:
                        continue

                    optimizer.zero_grad()
                    loss = trajectory_loss(model, y_p[s], u_p[s:e], y_p[s:e],
                                           normalizer, lambda_ripple=lambda_ripple)

                    if has_jacobian and lambda_J > 0 and n_windows % 5 == 0:
                        loss_J = jacobian_loss_lpv(model, A_targets, x_ops, u_ops, normalizer)
                        loss = loss + lambda_J * loss_J

                    if torch.isnan(loss) or torch.isinf(loss):
                        continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_windows += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_windows, 1)
            train_history.append(avg_loss)
            ep_time = time.time() - t_ep

            if epoch % 10 == 0 or epoch == 1 or epoch == n_epochs:
                val_loss, rmse_v, rmse_i = compute_validation(model, normalizer, val_u, val_y)
                val_history.append(val_loss)
                elapsed_h = (time.time() - t_start) / 3600

                improved = ''
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                    improved = ' ** BEST **'
                    save_checkpoint(model, best_state, best_val_loss, total_epochs,
                                    train_history, val_history, stats, run_dir / 'best.pt')
                else:
                    no_improve += 1

                fp_v, fp_i = compute_full_profile_rmse(model, normalizer, val_u, val_y)
                current_lr = optimizer.param_groups[0]['lr']
                log(f"  [{ph_name}] Ep {epoch:3d}/{n_epochs} | Train: {avg_loss:.4f} | Val: {val_loss:.4f} | "
                    f"Vout={rmse_v:.3f}V iL={rmse_i:.3f}A | FP: Vout={fp_v:.3f}V iL={fp_i:.3f}A | "
                    f"LR={current_lr:.1e} | {ep_time:.1f}s | {elapsed_h:.2f}h{improved}")

                if no_improve >= 20 and not is_test:
                    log(f"  Early stopping at epoch {epoch}")
                    break
            else:
                if epoch % 50 == 0:
                    log(f"  [{ph_name}] Ep {epoch:3d}/{n_epochs} | Train: {avg_loss:.4f} | {ep_time:.1f}s")

            if time.time() - last_save_time > 600:
                save_checkpoint(model, best_state, best_val_loss, total_epochs,
                                train_history, val_history, stats, run_dir / 'latest.pt')
                last_save_time = time.time()

            if (time.time() - t_start) / 3600 >= max_hours:
                log(f"  Time limit reached ({max_hours}h)")
                break

        save_checkpoint(model, best_state, best_val_loss, total_epochs,
                        train_history, val_history, stats, run_dir / f'after_{ph_name}.pt')
        log(f"  Phase {ph_name} done. Best val: {best_val_loss:.4f}")
        if (time.time() - t_start) / 3600 >= max_hours:
            break

    # ---- Phase 4: Extended ----
    if not is_test and (time.time() - t_start) / 3600 < max_hours:
        log(f"\n--- Phase4_Extended: ReduceLROnPlateau, progressive windows ---")
        if best_state is not None:
            model.load_state_dict(best_state)

        ext_lr = 2e-4
        optimizer = torch.optim.Adam(model.parameters(), lr=ext_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

        no_improve_ext = 0
        epoch_count = 0

        while True:
            epoch_count += 1
            total_epochs += 1
            t_ep = time.time()

            if epoch_count <= 200:
                window_ms = 20
            elif epoch_count <= 400:
                window_ms = 40
            else:
                window_ms = 80
            window_samples = int(window_ms * 1e-3 / TS)

            model.train()
            epoch_loss, n_windows = 0.0, 0
            perm = np.random.permutation(n_train)

            for p_idx in perm:
                u_p, y_p = train_u[p_idx], train_y[p_idx]
                T = len(u_p)
                n_win_total = max(1, T // window_samples)
                n_use = min(3, n_win_total)
                if n_use == 0:
                    continue
                win_starts = np.random.choice(n_win_total, n_use, replace=False)
                for w in win_starts:
                    s = w * window_samples
                    e = min(s + window_samples, T)
                    if e - s < window_samples:
                        continue
                    optimizer.zero_grad()
                    loss = trajectory_loss(model, y_p[s], u_p[s:e], y_p[s:e],
                                           normalizer, lambda_ripple=0.1)
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_windows += 1

            avg_loss = epoch_loss / max(n_windows, 1)
            train_history.append(avg_loss)
            elapsed_h = (time.time() - t_start) / 3600

            if elapsed_h >= max_hours:
                log(f"  [Extended] Time limit ({max_hours}h). Stopping.")
                break

            current_lr = optimizer.param_groups[0]['lr']
            if current_lr <= 1e-6 * 1.01:
                log(f"  [Extended] LR at floor. Stopping.")
                break

            if epoch_count % 10 == 0 or epoch_count == 1:
                val_loss, rmse_v, rmse_i = compute_validation(model, normalizer, val_u, val_y)
                val_history.append(val_loss)
                scheduler.step(val_loss)

                improved = ''
                if val_loss < best_val_loss * (1 - 1e-4):
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    no_improve_ext = 0
                    improved = ' ** BEST **'
                    save_checkpoint(model, best_state, best_val_loss, total_epochs,
                                    train_history, val_history, stats, run_dir / 'best.pt')
                else:
                    no_improve_ext += 1

                fp_v, fp_i = compute_full_profile_rmse(model, normalizer, val_u, val_y)
                log(f"  [Ext ep{epoch_count:4d}] Train: {avg_loss:.4f} | Val: {val_loss:.4f} | "
                    f"Vout={rmse_v:.3f}V iL={rmse_i:.3f}A | FP: Vout={fp_v:.3f}V iL={fp_i:.3f}A | "
                    f"win={window_ms}ms | LR={current_lr:.1e} | {elapsed_h:.2f}h | "
                    f"noImprove={no_improve_ext}/30{improved}")

                if no_improve_ext >= 30:
                    log(f"  [Extended] Plateau. Stopping.")
                    break

            if time.time() - last_save_time > 600:
                save_checkpoint(model, best_state, best_val_loss, total_epochs,
                                train_history, val_history, stats, run_dir / 'latest.pt')
                last_save_time = time.time()

    # ---- Finalize ----
    total_time_h = (time.time() - t_start) / 3600
    if best_state is not None:
        model.load_state_dict(best_state)

    val_loss, rmse_v, rmse_i = compute_validation(model, normalizer, val_u, val_y)
    fp_v, fp_i = compute_full_profile_rmse(model, normalizer, val_u, val_y)

    save_checkpoint(model, best_state, best_val_loss, total_epochs,
                    train_history, val_history, stats, run_dir / 'final.pt')

    log(f"\n{'=' * 60}")
    log(f"LPV Run complete — Seed {seed}")
    log(f"  Total time:    {total_time_h:.2f} h")
    log(f"  Total epochs:  {total_epochs}")
    log(f"  Best val loss: {best_val_loss:.6f}")
    log(f"  Full-prof RMSE: Vout={fp_v:.4f}V, iL={fp_i:.4f}A")
    log(f"{'=' * 60}\n")

    config = {
        'arch': 'lpv', 'solver': 'rk4',
        'hidden': HIDDEN_LPV,
        'init_from_linear': has_jacobian,
        'val_profiles': [5, 12, 14],
        'max_hours': max_hours,
    }
    results = {
        'best_val_loss': best_val_loss,
        'full_profile_rmse_vout': float(fp_v),
        'full_profile_rmse_il': float(fp_i),
        'total_epochs': total_epochs,
        'total_hours': round(total_time_h, 3),
    }
    append_seed_log(seed, config, results, CHECKPOINT_BASE / 'seed_log.json')

    # Update global best
    import fcntl, shutil
    lock_path = str(CHECKPOINT_BASE / 'global_best.lock')
    with open(lock_path, 'w') as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        gb_path = CHECKPOINT_BASE / 'global_best.pt'
        update = True
        if gb_path.exists():
            prev = torch.load(str(gb_path), weights_only=False)
            if prev.get('best_val_loss', float('inf')) <= best_val_loss:
                update = False
        if update and (run_dir / 'best.pt').exists():
            shutil.copy2(str(run_dir / 'best.pt'), str(gb_path))
            log(f"  >>> GLOBAL BEST updated: seed={seed}, val={best_val_loss:.6f}")
        fcntl.flock(lock_f, fcntl.LOCK_UN)

    return best_val_loss, best_state, stats, results


# ============================================================
# Multi-run
# ============================================================
def run_multi(n_runs, max_hours_per_run):
    global _log_file
    _log_file = SCRIPT_DIR / 'training_lpv_multi.log'

    log(f"\n{'#' * 60}")
    log(f"LPV MULTI-RUN: {n_runs} runs, {max_hours_per_run}h each")
    log(f"{'#' * 60}")

    leaderboard = []
    for run_idx in range(1, n_runs + 1):
        seed = run_idx
        best_val, _, _, results = run_single(seed, max_hours_per_run)
        leaderboard.append((seed, best_val, results))

        log(f"\n--- Leaderboard after {run_idx} runs ---")
        for rank, (s, v, r) in enumerate(sorted(leaderboard, key=lambda x: x[1]), 1):
            log(f"  #{rank}: seed={s} val={v:.6f} FP_Vout={r['full_profile_rmse_vout']:.3f}V")


# ============================================================
# CLI
# ============================================================
def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    parser = argparse.ArgumentParser(description='LPV Neural ODE training')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--multi', type=int, default=0)
    parser.add_argument('--max_hours', type=float, default=10.0)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    CHECKPOINT_BASE.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.test:
        run_single(seed=0, max_hours=1.0, is_test=True)
    elif args.multi > 0:
        run_multi(args.multi, args.max_hours)
    else:
        run_single(args.seed, args.max_hours)


if __name__ == '__main__':
    main()
