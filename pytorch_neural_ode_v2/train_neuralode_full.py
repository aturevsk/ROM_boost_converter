"""
train_neuralode_full.py
=======================
Combined Neural ODE training: 3-phase curriculum + extended fine-tuning.
Self-contained — no imports from other project scripts.

Architecture: dx/dt = dxdt_scale * MLP([x_n, u_n]), MLP = [3->64->tanh->64->tanh->2]
Integration:  Fixed-step RK4 at dt = TS * STEP_SKIP = 50 us
States:       x = [Vout, iL],  Input: u = duty cycle

Training pipeline per run:
  Phase 1 (HalfOsc):  300 ep, 5ms windows,  LR=1e-3, lambda_J=0.1, no ripple
  Phase 2 (FullOsc):  500 ep, 20ms windows, LR=5e-4, lambda_J=0.01, ripple=0.1
  Phase 3 (Finetune): 200 ep, 20ms windows, LR=1e-4, ripple=0.1
  Phase 4 (Extended): ReduceLROnPlateau, progressive windows 20->40->80ms

Improvements over original scripts:
  - Deterministic seeding (torch + numpy) for reproducibility
  - Physics-informed init from Jacobian targets (initialize_from_linear)
  - Full-profile validation metric (logged, not used for early stopping)
  - Progressive window growth in extended phase
  - Multi-run support: --multi N runs N seeds back-to-back, tracks global best
  - Full config saved per run in seed_log.json for MATLAB reproduction

Usage:
  python3 train_neuralode_full.py --test                    # smoke test
  python3 train_neuralode_full.py --seed 42                 # single run
  python3 train_neuralode_full.py --multi 7 --max_hours 8   # weekend batch
  nohup python3 train_neuralode_full.py --multi 7 > training.log 2>&1 &
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
# Paths (relative to this script)
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / 'data'
MODEL_DIR = REPO_ROOT / 'model_data'
CHECKPOINT_BASE = SCRIPT_DIR / 'checkpoints'
RESULTS_DIR = SCRIPT_DIR / 'results'

# ============================================================
# Constants
# ============================================================
NX = 2              # states: [Vout, iL]
NU = 1              # input: duty
HIDDEN = [64, 64]   # MLP hidden layers
TS = 5e-6           # base sample period (5 us)
STEP_SKIP = 10      # subsample factor -> effective dt = 50 us
DEVICE = 'cpu'      # CPU faster than MPS for tiny model

# Filter cutoff: geometric mean of LC resonance and switching frequency
F_RES = 600         # Hz (LC resonance)
F_SW = 200e3        # Hz (switching frequency)
FILTER_CUTOFF = np.sqrt(F_RES * F_SW)  # ~11 kHz


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
# Normalization
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
# Neural ODE Model
# ============================================================
class NeuralODE_RHS(nn.Module):
    """MLP: [x_n, u_n] -> dxdt (physical units via dxdt_scale buffer)."""
    def __init__(self, nx=NX, nu=NU, hidden=HIDDEN, dxdt_scale=None):
        super().__init__()
        layers = []
        in_dim = nx + nu
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, nx))
        self.net = nn.Sequential(*layers)
        self.nx = nx
        self.nu = nu
        if dxdt_scale is not None:
            self.register_buffer('dxdt_scale', dxdt_scale)
        else:
            self.register_buffer('dxdt_scale', torch.ones(nx))

    def forward(self, x_n, u_n):
        xu = torch.cat([x_n, u_n], dim=-1)
        raw = self.net(xu)
        return raw * self.dxdt_scale


# ============================================================
# Data Loading
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
    total = sum(len(u) for u in profiles_u)
    log(f"  Loaded {len(profiles_u)} profiles, {total:,} total samples")
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
# Physics-informed initialization
# ============================================================
def initialize_from_linear(model, A_targets, x_ops, u_ops, normalizer):
    """Initialize first layer from linearized A,B at mid-range operating point."""
    n_pts = len(u_ops)
    mid_idx = n_pts // 2  # mid-range duty cycle
    A = A_targets[mid_idx].numpy()  # (2, 2)

    # Estimate B from finite difference: B ≈ dxdt/du at operating point
    # Use a simple approximation: B = [0; 0] (duty doesn't appear in dxdt directly
    # in the normalized model — the MLP learns this mapping)
    B = np.zeros((NX, NU))

    AB = np.concatenate([A, B], axis=1)  # (2, 3)
    scale = max(abs(AB.flatten())) + 1e-8
    AB_scaled = AB / scale * 0.5

    with torch.no_grad():
        W = model.net[0].weight  # (hidden, nx+nu)
        W.zero_()
        W[:NX, :NX + NU] = torch.tensor(AB_scaled, dtype=torch.float32)
        W[NX:, :] = 0.01 * torch.randn(W.shape[0] - NX, W.shape[1])
    log(f"  First layer initialized from A-matrix at D={u_ops[mid_idx]:.2f}, scale={scale:.1f}")


# ============================================================
# Butterworth filter (differentiable IIR)
# ============================================================
def apply_filter_torch(x, b, a):
    """Direct Form II Transposed IIR filter. x: (T, C), b/a: numpy arrays."""
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
# Loss Functions
# ============================================================
def integrate_ode(model, x0, u_seq, normalizer, step_skip=STEP_SKIP):
    """RK4 integration, returns subsampled trajectory."""
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
    """Trajectory matching loss with optional LP/HP ripple decomposition."""
    T = len(u_seq)
    x_pred, T_sub = integrate_ode(model, x0, u_seq, normalizer, step_skip)
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
        else:
            return torch.mean((x_pred_n - y_sub_n) ** 2)
    else:
        return torch.mean((x_pred_n - y_sub_n) ** 2)


def jacobian_loss(model, A_targets, x_ops, u_ops, normalizer):
    """Jacobian matching: MSE(df/dx, A_target) at operating points."""
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
        x_op = x_ops[j]
        u_op = u_ops[j].unsqueeze(0)
        u_op_n = normalizer.norm_u(u_op).unsqueeze(0)

        def f_x(x_phys):
            x_n = normalizer.norm_x(x_phys.unsqueeze(0))
            return model(x_n, u_op_n).squeeze(0)

        J = jacobian(f_x, x_op)
        A_target = A_targets[j]
        A_scale = torch.max(torch.abs(A_target)) + 1.0
        loss = loss + torch.mean(((J - A_target) / A_scale) ** 2)

    return loss / len(idx)


# ============================================================
# Validation
# ============================================================
def compute_validation(model, normalizer, val_u, val_y):
    """Windowed validation (same as training). Returns (val_loss, rmse_v, rmse_i)."""
    model.eval()
    total_se_v, total_se_i, total_n = 0.0, 0.0, 0
    with torch.no_grad():
        for u_v, y_v in zip(val_u, val_y):
            T = len(u_v)
            x = y_v[0].clone()
            for k in range(1, T):
                u_k = u_v[k - 1].unsqueeze(0)
                x_n = normalizer.norm_x(x.unsqueeze(0))
                u_n = normalizer.norm_u(u_k).unsqueeze(0)
                dxdt = model(x_n, u_n).squeeze(0)
                x = x + TS * dxdt
            # Subsampled comparison
            x_pred, T_sub = integrate_ode(model, y_v[0], u_v, normalizer)
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
    """Full-profile free-running RK4 RMSE (deployment-like evaluation)."""
    model.eval()
    rmse_v_all, rmse_i_all = [], []
    with torch.no_grad():
        for u_v, y_v in zip(val_u, val_y):
            x_pred, T_sub = integrate_ode(model, y_v[0], u_v, normalizer)
            y_sub = y_v[::STEP_SKIP][:T_sub]
            rmse_v_all.append(np.sqrt(((x_pred[:, 0] - y_sub[:, 0]) ** 2).mean().item()))
            rmse_i_all.append(np.sqrt(((x_pred[:, 1] - y_sub[:, 1]) ** 2).mean().item()))
    model.train()
    return np.mean(rmse_v_all), np.mean(rmse_i_all)


def validate_and_plot(model, normalizer, val_u, val_y, save_path):
    """Generate validation plot and save to file."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    model.eval()
    fig, axes = plt.subplots(len(val_u), 2, figsize=(14, 3.5 * len(val_u)))
    if len(val_u) == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for v, (u_v, y_v) in enumerate(zip(val_u, val_y)):
            x_pred, T_sub = integrate_ode(model, y_v[0], u_v, normalizer)
            y_sub = y_v[::STEP_SKIP][:T_sub].numpy()
            x_np = x_pred.numpy()
            t = np.arange(T_sub) * TS * STEP_SKIP * 1000  # ms

            rmse_v = np.sqrt(np.mean((y_sub[:, 0] - x_np[:, 0]) ** 2))
            rmse_i = np.sqrt(np.mean((y_sub[:, 1] - x_np[:, 1]) ** 2))

            axes[v, 0].plot(t, y_sub[:, 0], 'b-', lw=1.5, label='Simscape')
            axes[v, 0].plot(t, x_np[:, 0], 'r--', lw=1.2, label=f'NeuralODE ({rmse_v:.3f}V)')
            axes[v, 0].set_ylabel('Vout (V)')
            axes[v, 0].set_title(f'Val {v + 1}: Vout')
            axes[v, 0].legend(fontsize=8)
            axes[v, 0].grid(True, alpha=0.3)

            axes[v, 1].plot(t, y_sub[:, 1], 'b-', lw=1.5, label='Simscape')
            axes[v, 1].plot(t, x_np[:, 1], 'r--', lw=1.2, label=f'NeuralODE ({rmse_i:.3f}A)')
            axes[v, 1].set_ylabel('iL (A)')
            axes[v, 1].set_title(f'Val {v + 1}: iL')
            axes[v, 1].legend(fontsize=8)
            axes[v, 1].grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel('Time (ms)')
    axes[-1, 1].set_xlabel('Time (ms)')
    fig.suptitle('Neural ODE Validation (Free-Running RK4)', fontweight='bold')
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    model.train()


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
    """Append run results to seed_log.json for reproducibility."""
    entry = {
        'seed': seed,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': config,
        'results': results,
    }
    log_data = []
    if log_path.exists():
        with open(log_path) as f:
            log_data = json.load(f)
    log_data.append(entry)
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2, default=lambda o: float(o) if hasattr(o, 'item') else o)


# ============================================================
# Single training run
# ============================================================
def run_single(seed, max_hours, is_test=False):
    """Full training pipeline: Phase 1-3 curriculum + Phase 4 extended."""
    run_dir = CHECKPOINT_BASE / f'run_{seed:04d}'
    run_dir.mkdir(parents=True, exist_ok=True)

    global _log_file
    _log_file = run_dir / 'training.log'

    log(f"\n{'=' * 60}")
    log(f"Neural ODE Training — Seed {seed}")
    log(f"{'=' * 60}")
    log(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Max hours: {max_hours}")

    # ---- Seeding ----
    torch.manual_seed(seed)
    np.random.seed(seed)
    log(f"Seeds set: torch={seed}, numpy={seed}")

    # ---- Load data ----
    log("\nLoading data...")
    profiles_u, profiles_y = load_training_data()
    n_total = len(profiles_u)

    # Train/val split: val profiles chosen for mid-range interpolation testing
    # Val 5  (D=0.25-0.65): wide-range transients
    # Val 12 (D=0.35-0.45): narrow-band mid-low
    # Val 14 (D=0.45-0.55): narrow-band mid-high
    # All other profiles (including edge cases 7,8,17,18) used for training
    VAL_INDICES = [4, 11, 13]  # 0-based: profiles 5, 12, 14
    TRAIN_INDICES = [i for i in range(n_total) if i not in VAL_INDICES]

    stats = compute_norm_stats(
        [profiles_u[i] for i in TRAIN_INDICES],
        [profiles_y[i] for i in TRAIN_INDICES])
    normalizer = Normalizer(stats)
    log(f"  Normalization: dxdt_scale=[{stats['dxdt_std'][0]:.0f}, {stats['dxdt_std'][1]:.0f}]")

    # Save stats
    with open(MODEL_DIR / 'neuralode_norm_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    A_targets, x_ops, u_ops = load_jacobian_targets()
    has_jacobian = A_targets is not None

    train_u = [torch.tensor(profiles_u[i]) for i in TRAIN_INDICES]
    train_y = [torch.tensor(profiles_y[i]) for i in TRAIN_INDICES]
    val_u = [torch.tensor(profiles_u[i]) for i in VAL_INDICES]
    val_y = [torch.tensor(profiles_y[i]) for i in VAL_INDICES]
    n_train = len(TRAIN_INDICES)
    log(f"  Train: {n_train} profiles, Val: {len(VAL_INDICES)} profiles (5,12,14)")
    log(f"  Val ranges: D=[0.25-0.65], D=[0.35-0.45], D=[0.45-0.55]")

    # ---- Build model ----
    dxdt_scale = torch.tensor(stats['dxdt_std'], dtype=torch.float32)
    model = NeuralODE_RHS(NX, NU, HIDDEN, dxdt_scale=dxdt_scale)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  Parameters: {n_params}")

    # Physics-informed initialization
    if has_jacobian:
        initialize_from_linear(model, A_targets, x_ops, u_ops, normalizer)

    # ---- Phase definitions ----
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

    # ---- Training state ----
    best_val_loss = float('inf')
    best_state = None
    train_history = []
    val_history = []
    total_epochs = 0
    t_start = time.time()
    last_save_time = time.time()

    # ---- Curriculum phases 1-3 ----
    for phase in phases:
        ph_name = phase['name']
        n_epochs = phase['epochs']
        window_samples = int(phase['window_ms'] * 1e-3 / TS)
        lr = phase['lr']
        lambda_J = phase['lambda_J']
        lambda_ripple = phase.get('lambda_ripple', 0.0)
        win_per_prof = phase['win_per_prof']

        log(f"\n--- {ph_name}: {n_epochs} ep, win={phase['window_ms']}ms, "
            f"LR={lr}, lJ={lambda_J}, lR={lambda_ripple} ---")

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, n_epochs, eta_min=lr * 0.01)

        no_improve = 0
        for epoch in range(1, n_epochs + 1):
            total_epochs += 1
            t_ep = time.time()
            model.train()
            epoch_loss = 0.0
            n_windows = 0
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
                    u_win, y_win = u_p[s:e], y_p[s:e]
                    x0 = y_win[0]

                    optimizer.zero_grad()
                    loss = trajectory_loss(model, x0, u_win, y_win,
                                           normalizer, lambda_ripple=lambda_ripple)

                    if has_jacobian and lambda_J > 0 and n_windows % 5 == 0:
                        loss_J = jacobian_loss(model, A_targets, x_ops, u_ops, normalizer)
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

            # Validation
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
                                    train_history, val_history, stats,
                                    run_dir / 'best.pt')
                else:
                    no_improve += 1

                # Full-profile RMSE (logged only)
                fp_v, fp_i = compute_full_profile_rmse(model, normalizer, val_u, val_y)

                current_lr = optimizer.param_groups[0]['lr']
                log(f"  [{ph_name}] Ep {epoch:3d}/{n_epochs} | "
                    f"Train: {avg_loss:.4f} | Val: {val_loss:.4f} | "
                    f"Vout={rmse_v:.3f}V iL={rmse_i:.3f}A | "
                    f"FP: Vout={fp_v:.3f}V iL={fp_i:.3f}A | "
                    f"LR={current_lr:.1e} | {ep_time:.1f}s | {elapsed_h:.2f}h{improved}")

                if no_improve >= 20 and not is_test:
                    log(f"  Early stopping at epoch {epoch}")
                    break
            else:
                if epoch % 50 == 0:
                    log(f"  [{ph_name}] Ep {epoch:3d}/{n_epochs} | Train: {avg_loss:.4f} | {ep_time:.1f}s")

            # Periodic checkpoint
            if time.time() - last_save_time > 600:
                save_checkpoint(model, best_state, best_val_loss, total_epochs,
                                train_history, val_history, stats,
                                run_dir / 'latest.pt')
                last_save_time = time.time()

            # Time check
            if (time.time() - t_start) / 3600 >= max_hours:
                log(f"  Time limit reached ({max_hours}h)")
                break

        # End of phase
        save_checkpoint(model, best_state, best_val_loss, total_epochs,
                        train_history, val_history, stats,
                        run_dir / f'after_{ph_name}.pt')
        log(f"  Phase {ph_name} done. Best val: {best_val_loss:.4f}")

        if (time.time() - t_start) / 3600 >= max_hours:
            break

    # ---- Phase 4: Extended fine-tuning with progressive windows ----
    if not is_test and (time.time() - t_start) / 3600 < max_hours:
        log(f"\n--- Phase4_Extended: ReduceLROnPlateau, progressive windows ---")

        # Load best state for fine-tuning
        if best_state is not None:
            model.load_state_dict(best_state)

        ext_lr = 2e-4
        ext_lr_min = 1e-6
        optimizer = torch.optim.Adam(model.parameters(), lr=ext_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=ext_lr_min)

        plateau_evals = 30
        plateau_rel_tol = 1e-4
        no_improve_ext = 0
        lambda_ripple = 0.1
        epoch_count = 0

        while True:
            epoch_count += 1
            total_epochs += 1
            t_ep = time.time()

            # Progressive window growth
            if epoch_count <= 200:
                window_ms = 20
            elif epoch_count <= 400:
                window_ms = 40
            else:
                window_ms = 80
            window_samples = int(window_ms * 1e-3 / TS)

            model.train()
            epoch_loss = 0.0
            n_windows = 0
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
                    if e - s < 20:
                        continue
                    u_win, y_win = u_p[s:e], y_p[s:e]
                    x0 = y_win[0]

                    optimizer.zero_grad()
                    loss = trajectory_loss(model, x0, u_win, y_win,
                                           normalizer, lambda_ripple=lambda_ripple)
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_windows += 1

            avg_loss = epoch_loss / max(n_windows, 1)
            train_history.append(avg_loss)
            ep_time = time.time() - t_ep
            elapsed_h = (time.time() - t_start) / 3600

            # Time check
            if elapsed_h >= max_hours:
                log(f"  [Extended] Time limit ({max_hours}h). Stopping.")
                break

            # LR floor check
            current_lr = optimizer.param_groups[0]['lr']
            if current_lr <= ext_lr_min * 1.01:
                log(f"  [Extended] LR at floor ({current_lr:.2e}). Stopping.")
                break

            # Validation every 10 epochs
            if epoch_count % 10 == 0 or epoch_count == 1:
                val_loss, rmse_v, rmse_i = compute_validation(model, normalizer, val_u, val_y)
                val_history.append(val_loss)
                scheduler.step(val_loss)

                improved = ''
                if val_loss < best_val_loss * (1 - plateau_rel_tol):
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    no_improve_ext = 0
                    improved = ' ** BEST **'
                    save_checkpoint(model, best_state, best_val_loss, total_epochs,
                                    train_history, val_history, stats,
                                    run_dir / 'best.pt')
                else:
                    no_improve_ext += 1

                fp_v, fp_i = compute_full_profile_rmse(model, normalizer, val_u, val_y)

                log(f"  [Ext ep{epoch_count:4d}] Train: {avg_loss:.4f} | Val: {val_loss:.4f} | "
                    f"Vout={rmse_v:.3f}V iL={rmse_i:.3f}A | "
                    f"FP: Vout={fp_v:.3f}V iL={fp_i:.3f}A | "
                    f"win={window_ms}ms | LR={current_lr:.1e} | {elapsed_h:.2f}h | "
                    f"noImprove={no_improve_ext}/{plateau_evals}{improved}")

                if no_improve_ext >= plateau_evals:
                    log(f"  [Extended] Plateau ({plateau_evals} evals). Stopping.")
                    break

            # Periodic checkpoint
            if time.time() - last_save_time > 600:
                save_checkpoint(model, best_state, best_val_loss, total_epochs,
                                train_history, val_history, stats,
                                run_dir / 'latest.pt')
                last_save_time = time.time()

    # ---- Finalize ----
    total_time_h = (time.time() - t_start) / 3600

    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final validation
    val_loss, rmse_v, rmse_i = compute_validation(model, normalizer, val_u, val_y)
    fp_v, fp_i = compute_full_profile_rmse(model, normalizer, val_u, val_y)

    # Save final checkpoint
    save_checkpoint(model, best_state, best_val_loss, total_epochs,
                    train_history, val_history, stats,
                    run_dir / 'final.pt')

    # Save validation plot
    plot_path = RESULTS_DIR / f'validation_seed_{seed:04d}.png'
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    validate_and_plot(model, normalizer, val_u, val_y, plot_path)

    log(f"\n{'=' * 60}")
    log(f"Run complete — Seed {seed}")
    log(f"  Total time:    {total_time_h:.2f} h")
    log(f"  Total epochs:  {total_epochs}")
    log(f"  Best val loss: {best_val_loss:.6f}")
    log(f"  Windowed RMSE: Vout={rmse_v:.4f}V, iL={rmse_i:.4f}A")
    log(f"  Full-prof RMSE: Vout={fp_v:.4f}V, iL={fp_i:.4f}A")
    log(f"  Saved: {run_dir / 'best.pt'}")
    log(f"{'=' * 60}\n")

    # Build full config for seed log
    config = {
        'phases': [{'name': p['name'], 'epochs': p['epochs'], 'window_ms': p['window_ms'],
                     'lr': p['lr'], 'lambda_J': p['lambda_J'],
                     'lambda_ripple': p.get('lambda_ripple', 0.0),
                     'win_per_prof': p['win_per_prof']}
                    for p in (phases if not is_test else phases)],
        'extended': {'lr': 2e-4, 'lr_min': 1e-6, 'plateau_evals': 30,
                     'progressive_windows': [20, 40, 80]},
        'step_skip': STEP_SKIP, 'ts': TS, 'hidden': HIDDEN,
        'init_from_linear': has_jacobian,
        'val_profiles': [i+1 for i in VAL_INDICES],  # 1-based for readability
        'max_hours': max_hours,
    }
    results = {
        'best_val_loss': best_val_loss,
        'windowed_rmse_vout': float(rmse_v),
        'windowed_rmse_il': float(rmse_i),
        'full_profile_rmse_vout': float(fp_v),
        'full_profile_rmse_il': float(fp_i),
        'total_epochs': total_epochs,
        'total_hours': round(total_time_h, 3),
    }

    # Append to seed log
    seed_log_path = CHECKPOINT_BASE / 'seed_log.json'
    append_seed_log(seed, config, results, seed_log_path)

    return best_val_loss, best_state, stats, results


# ============================================================
# Multi-run
# ============================================================
def run_multi(n_runs, max_hours_per_run):
    """Run N training runs with seeds 1..N, track global best."""
    global _log_file
    _log_file = SCRIPT_DIR / 'training_multi.log'

    log(f"\n{'#' * 60}")
    log(f"MULTI-RUN: {n_runs} runs, {max_hours_per_run}h each")
    log(f"Estimated total: {n_runs * max_hours_per_run:.1f}h")
    log(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"{'#' * 60}")

    global_best_val = float('inf')
    global_best_seed = None
    leaderboard = []

    for run_idx in range(1, n_runs + 1):
        seed = run_idx
        log(f"\n{'=' * 60}")
        log(f"Starting run {run_idx}/{n_runs} (seed={seed})")
        log(f"{'=' * 60}")

        best_val, best_state, stats, results = run_single(seed, max_hours_per_run)
        leaderboard.append((seed, best_val, results))

        # Update global best
        if best_val < global_best_val:
            global_best_val = best_val
            global_best_seed = seed
            # Copy best.pt to global_best.pt
            src = CHECKPOINT_BASE / f'run_{seed:04d}' / 'best.pt'
            dst = CHECKPOINT_BASE / 'global_best.pt'
            if src.exists():
                import shutil
                shutil.copy2(str(src), str(dst))
                log(f"  >>> NEW GLOBAL BEST: seed={seed}, val={best_val:.6f}")

        # Print leaderboard
        log(f"\n--- Leaderboard after {run_idx} runs ---")
        sorted_lb = sorted(leaderboard, key=lambda x: x[1])
        for rank, (s, v, r) in enumerate(sorted_lb, 1):
            marker = ' <<<' if s == global_best_seed else ''
            log(f"  #{rank}: seed={s:4d} | val={v:.6f} | "
                f"FP Vout={r['full_profile_rmse_vout']:.3f}V | "
                f"{r['total_hours']:.1f}h{marker}")

    log(f"\n{'#' * 60}")
    log(f"ALL RUNS COMPLETE")
    log(f"  Best seed: {global_best_seed}")
    log(f"  Best val:  {global_best_val:.6f}")
    log(f"  Checkpoint: checkpoints/global_best.pt")
    log(f"  Seed log:   checkpoints/seed_log.json")
    log(f"{'#' * 60}")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Neural ODE training: 3-phase curriculum + extended fine-tuning')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
    parser.add_argument('--multi', type=int, default=0,
                        help='Run N sequential training runs with seeds 1..N')
    parser.add_argument('--max_hours', type=float, default=8.0,
                        help='Max hours per run (default: 8.0)')
    parser.add_argument('--test', action='store_true',
                        help='Quick smoke test (5 epochs)')
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
