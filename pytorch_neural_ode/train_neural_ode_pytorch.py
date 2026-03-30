"""
Branch C: Neural ODE in PyTorch with torchdiffeq + custom loss.

Architecture: dx/dt = f(x, u; θ), f is MLP with tanh
States: x = [Vout, iL], Input: u = duty, Output: y = x
Integration: torchdiffeq.odeint_adjoint (proper adjoint method)

Custom Loss:
  L = L_traj + λ_J * L_jacobian
  L_traj = MSE(x_pred_norm, x_true_norm)  (normalized trajectory matching)
  L_jacobian = MSE(∂f/∂x, A_ssest)  (linearization matching)

Normalization:
  Inputs normalized: x_n = (x - x_mean) / x_std, u_n = (u - u_mean) / u_std
  Output scaled: dxdt = dxdt_scale * MLP(x_n, u_n)
  Loss computed in normalized state space for balanced gradients.

Jacobian: torch.autograd.functional.jacobian (exact autodiff)
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd.functional import jacobian
from torchdiffeq import odeint_adjoint as odeint
from pathlib import Path
import scipy.io as sio
import h5py

# ============================================================
# Config
# ============================================================
NX = 2       # states: [Vout, iL]
NU = 1       # input: duty
HIDDEN = [64, 64]
TS = 5e-6
STEP_SKIP = 10  # consistent skip for training AND validation
DEVICE = 'cpu'  # CPU faster than MPS for tiny model

BASE_DIR = Path(__file__).parent
REPO_ROOT = BASE_DIR.parent
DATA_DIR = REPO_ROOT / 'data'
MODEL_DIR = REPO_ROOT / 'model_data'
CHECKPOINT_DIR = BASE_DIR / 'checkpoints'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Normalization
# ============================================================
def compute_norm_stats(profiles_u, profiles_y):
    """Compute mean/std for inputs and outputs from training data."""
    all_u = np.concatenate(profiles_u)
    all_y = np.concatenate(profiles_y, axis=0)

    # dx/dt via finite differences for output scale
    all_dvout = []
    all_dil = []
    for y in profiles_y:
        all_dvout.append(np.diff(y[:, 0]) / TS)
        all_dil.append(np.diff(y[:, 1]) / TS)
    dvout = np.concatenate(all_dvout)
    dil = np.concatenate(all_dil)

    stats = {
        'u_mean': float(all_u.mean()),
        'u_std': float(all_u.std()),
        'x_mean': all_y.mean(axis=0).tolist(),  # [Vout_mean, iL_mean]
        'x_std': all_y.std(axis=0).tolist(),     # [Vout_std, iL_std]
        'dxdt_std': [float(dvout.std()), float(dil.std())],  # output scale
    }
    return stats


class Normalizer:
    """Handles normalization/denormalization for the Neural ODE."""
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
    """MLP: [x1_n, x2_n, u_n] → [dx1/dt, dx2/dt] (scaled)

    Inputs are normalized, output is multiplied by dxdt_scale so the
    network internals operate in ~[-1, 1] range.
    """
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
        # Output scaling: registered as buffer (not a parameter)
        if dxdt_scale is not None:
            self.register_buffer('dxdt_scale', dxdt_scale)
        else:
            self.register_buffer('dxdt_scale', torch.ones(nx))

    def forward(self, x_n, u_n):
        """x_n: (batch, nx) normalized, u_n: (batch, nu) normalized
        Returns dxdt in PHYSICAL units (V/s, A/s)."""
        xu = torch.cat([x_n, u_n], dim=-1)
        raw = self.net(xu)  # ~[-1, 1] thanks to tanh
        return raw * self.dxdt_scale  # scale to physical units


class ODEFunc(nn.Module):
    """Wrapper for torchdiffeq: stores current u_n and normalizer, provides f(t, x)"""
    def __init__(self, rhs, normalizer):
        super().__init__()
        self.rhs = rhs
        self.norm = normalizer
        self.u_n_current = None  # normalized u, set before each odeint call

    def forward(self, t, x):
        """x is in PHYSICAL units. Normalize, call MLP, return physical dxdt."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x_n = self.norm.norm_x(x)
        u_n = self.u_n_current
        if u_n.dim() == 1:
            u_n = u_n.unsqueeze(0)
        return self.rhs(x_n, u_n).squeeze(0)


# ============================================================
# Data Loading
# ============================================================
def load_training_data():
    """Load training data from CSV files (exported by MATLAB)"""
    csv_dir = DATA_DIR / 'neural_ode'
    csv_files = sorted(csv_dir.glob('profile_*.csv'))

    if len(csv_files) == 0:
        print(f"ERROR: No profile CSVs in {csv_dir}. Export from MATLAB first.")
        sys.exit(1)

    profiles_u = []
    profiles_y = []
    for f in csv_files:
        data = np.loadtxt(str(f), delimiter=',', dtype=np.float32)
        # Columns: [duty, Vout, iL, SimID]
        duty = data[:, 0]
        vout = data[:, 1]
        iL = data[:, 2]
        if len(duty) < 10:
            continue
        profiles_u.append(duty)
        profiles_y.append(np.stack([vout, iL], axis=1))

    total = sum(len(u) for u in profiles_u)
    print(f"  Loaded {len(profiles_u)} profiles, {total} total samples")
    return profiles_u, profiles_y


def load_jacobian_targets():
    """Load Branch B ssest models for Jacobian targets.
    The .mat file contains MATLAB ss objects which are hard to parse from Python.
    Instead, export A matrices + operating points to JSON from MATLAB first.
    Falls back to JSON if .mat parsing fails."""

    # Try JSON first (most reliable)
    json_file = DATA_DIR / 'neural_ode' / 'jacobian_targets.json'
    if json_file.exists():
        return load_jacobian_targets_json()

    # Try to export from MATLAB .mat file
    mat_file = MODEL_DIR / 'boost_frest_tf_data.mat'
    if not mat_file.exists():
        print(f"WARNING: No Jacobian targets found.")
        return None, None, None

    # Parse what we can from the .mat file
    try:
        with h5py.File(str(mat_file), 'r') as f:
            D_grid = np.array(f['D_grid']).flatten().astype(np.float32)
            Vout_ss = np.array(f['Vout_ss']).flatten().astype(np.float32)
            iL_ss = np.array(f['iL_ss']).flatten().astype(np.float32)

            # Try to get A matrices — they might be in A_all or inside tf_models
            A_targets = []
            x_ops = []
            u_ops = []

            if 'A_all' in f:
                A_all = np.array(f['A_all']).astype(np.float32)
                # A_all is (nx, nx, nGrid) or (nGrid, nx, nx)
                if A_all.ndim == 3:
                    for i in range(A_all.shape[-1]):
                        A_targets.append(A_all[:,:,i])
                        x_ops.append(np.array([Vout_ss[i], iL_ss[i]], dtype=np.float32))
                        u_ops.append(D_grid[i])

            if len(A_targets) == 0:
                print("  Can't parse A matrices from .mat, trying JSON fallback...")
                return load_jacobian_targets_json()

            print(f"  {len(A_targets)} Jacobian targets from .mat")
            return (torch.tensor(np.stack(A_targets)),
                    torch.tensor(np.stack(x_ops)),
                    torch.tensor(np.array(u_ops, dtype=np.float32)))
    except Exception as e:
        print(f"  Can't parse .mat: {e}, trying JSON fallback...")
        return load_jacobian_targets_json()


def load_jacobian_targets_json():
    """Fallback: load from JSON exported by MATLAB"""
    json_file = DATA_DIR / 'neural_ode' / 'jacobian_targets.json'
    if not json_file.exists():
        print(f"  WARNING: {json_file} not found")
        return None, None, None

    with open(json_file) as f:
        data = json.load(f)

    A_targets = torch.tensor(data['A_targets'], dtype=torch.float32)
    x_ops = torch.tensor(data['x_ops'], dtype=torch.float32)
    u_ops = torch.tensor(data['u_ops'], dtype=torch.float32)
    print(f"  {len(u_ops)} Jacobian targets from JSON")
    return A_targets, x_ops, u_ops


# ============================================================
# Initialize from linear model
# ============================================================
def initialize_from_linear(model, A, B):
    """Initialize first layer to approximate dx/dt = A*x + B*u"""
    AB = np.concatenate([A, B], axis=1)  # (nx, nx+nu) = (2, 3)
    scale = max(abs(AB.flatten())) + 1e-8
    AB_scaled = AB / scale * 0.5  # scale for tanh input range

    with torch.no_grad():
        W = model.net[0].weight  # (hidden, nx+nu)
        W.zero_()
        W[:NX, :NX+NU] = torch.tensor(AB_scaled, dtype=torch.float32)
        # Small random for remaining rows
        W[NX:, :] = 0.01 * torch.randn(W.shape[0]-NX, W.shape[1])
    print(f"  First layer initialized with [A,B], scale={scale:.1f}")


# ============================================================
# Filtering for ripple loss
# ============================================================
def design_filters():
    """Design Butterworth LP/HP filters matching MATLAB curriculum.

    Cutoff at geometric mean of LC resonance (600 Hz) and switching freq (200 kHz)
    = sqrt(600 * 200e3) ≈ 11 kHz.  At Ts=5e-6 subsampled by step_skip, the
    effective sample rate changes, so we return filter coefficients as a function
    of step_skip.
    """
    from scipy.signal import butter
    f_sw = 200e3
    f_res = 600
    f_cutoff = np.sqrt(f_res * f_sw)  # ~11 kHz
    return f_cutoff


# Pre-compute cutoff
FILTER_CUTOFF = design_filters()


def apply_filter_torch(x, b, a):
    """Apply IIR filter to tensor x along dim 0 (time). Differentiable.

    Uses direct-form II transposed for numerical stability.
    x: (T, C) tensor, b/a: filter coefficients (numpy arrays).
    Returns filtered x of same shape.
    """
    T, C = x.shape
    order = len(a) - 1
    y = torch.zeros_like(x)
    # State: direct-form II transposed
    d = torch.zeros(order, C, dtype=x.dtype)

    b_t = torch.tensor(b, dtype=x.dtype)
    a_t = torch.tensor(a, dtype=x.dtype)

    for t in range(T):
        y[t] = b_t[0] * x[t] + d[0]
        for i in range(order - 1):
            d[i] = b_t[i+1] * x[t] - a_t[i+1] * y[t] + d[i+1]
        d[order-1] = b_t[order] * x[t] - a_t[order] * y[t]

    return y


# ============================================================
# Loss Functions
# ============================================================
def integrate_ode(model, x0, u_seq, normalizer, step_skip=STEP_SKIP):
    """Integrate Neural ODE with RK4 and return predicted trajectory.

    Shared by trajectory_loss (avg + ripple). Returns subsampled prediction.
    """
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
        x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        x_pred[k+1] = x

    return x_pred, T_sub


def trajectory_loss(model, ode_func, x0, u_seq, y_true, normalizer,
                    step_skip=STEP_SKIP, lambda_ripple=0.0):
    """Integrate ODE and compare trajectory with ground truth.

    Loss = L_avg + lambda_ripple * L_ripple
      L_avg    = MSE of low-pass filtered prediction vs truth (normalized)
      L_ripple = MSE of high-pass filtered prediction vs truth (normalized)

    When lambda_ripple=0, no filtering is applied (just plain MSE).
    """
    from scipy.signal import butter

    T = len(u_seq)
    x_pred, T_sub = integrate_ode(model, x0, u_seq, normalizer, step_skip)
    y_sub = y_true[::step_skip][:T_sub]

    # Normalized space
    x_pred_n = normalizer.norm_x(x_pred)
    y_sub_n = normalizer.norm_x(y_sub)

    if lambda_ripple > 0 and T_sub > 20:
        # Design filters at the effective sample rate
        fs_eff = 1.0 / (TS * step_skip)
        f_nyq = fs_eff / 2.0
        # Clamp cutoff below Nyquist (must be < 1.0 in normalized freq)
        f_norm = min(FILTER_CUTOFF / f_nyq, 0.95)

        if f_norm > 0.05:  # only filter if cutoff is meaningful
            b_lp, a_lp = butter(2, f_norm, btype='low')
            b_hp, a_hp = butter(2, f_norm, btype='high')

            # Filter both pred and truth
            x_lp = apply_filter_torch(x_pred_n, b_lp, a_lp)
            y_lp = apply_filter_torch(y_sub_n, b_lp, a_lp)
            x_hp = apply_filter_torch(x_pred_n, b_hp, a_hp)
            y_hp = apply_filter_torch(y_sub_n, b_hp, a_hp)

            # Skip filter transient (first 10% of window)
            skip = max(T_sub // 10, 5)
            loss_avg = torch.mean((x_lp[skip:] - y_lp[skip:]) ** 2)
            loss_ripple = torch.mean((x_hp[skip:] - y_hp[skip:]) ** 2)
            loss = loss_avg + lambda_ripple * loss_ripple
        else:
            loss = torch.mean((x_pred_n - y_sub_n) ** 2)
    else:
        loss = torch.mean((x_pred_n - y_sub_n) ** 2)

    return loss


def jacobian_loss(model, A_targets, x_ops, u_ops, normalizer):
    """Compute Jacobian matching loss: MSE(∂f/∂x, A_ssest) at operating points.

    The model takes normalized inputs and outputs physical dxdt.
    The Jacobian ∂f_phys/∂x_phys = (∂f_phys/∂x_n) * (∂x_n/∂x_phys)
                                  = J_model_wrt_xn / x_std
    We compute J w.r.t. physical x directly by wrapping norm inside.
    """
    n_pts = len(u_ops)
    loss = torch.tensor(0.0)

    # Select subset for efficiency (4 fixed + 2 random)
    if n_pts > 6:
        fixed = np.linspace(0, n_pts-1, 4, dtype=int)
        others = list(set(range(n_pts)) - set(fixed))
        rand_idx = np.random.choice(others, min(2, len(others)), replace=False)
        idx = np.concatenate([fixed, rand_idx])
    else:
        idx = np.arange(n_pts)

    for j in idx:
        x_op = x_ops[j]  # (nx,) physical units
        u_op = u_ops[j].unsqueeze(0)  # (1,) physical units
        u_op_n = normalizer.norm_u(u_op).unsqueeze(0)  # (1, 1)

        # Jacobian ∂f/∂x in physical space (normalize inside)
        def f_x(x_phys):
            x_n = normalizer.norm_x(x_phys.unsqueeze(0))
            return model(x_n, u_op_n).squeeze(0)  # physical dxdt

        J = jacobian(f_x, x_op)  # (nx, nx) — exact autodiff

        A_target = A_targets[j]
        A_scale = torch.max(torch.abs(A_target)) + 1.0
        loss = loss + torch.mean(((J - A_target) / A_scale) ** 2)

    return loss / len(idx)


# ============================================================
# Training
# ============================================================
def train(mode='full'):
    is_test = (mode == 'test')
    is_resume = (mode == 'resume')
    print(f"{'='*60}")
    print(f"Branch C: Neural ODE (PyTorch) — with normalization")
    print(f"{'='*60}")

    # Load data
    print("\nLoading data...")
    profiles_u, profiles_y = load_training_data()
    if len(profiles_u) == 0:
        print("  Trying JSON data...")
        json_dir = DATA_DIR / 'neural_ode'
        profiles_u, profiles_y = [], []
        for f in sorted(json_dir.glob('profile_*.json')):
            with open(f) as fp:
                p = json.load(fp)
            profiles_u.append(np.array(p['duty'], dtype=np.float32))
            profiles_y.append(np.array(p['y'], dtype=np.float32))
        print(f"  Loaded {len(profiles_u)} profiles from JSON")

    n_total = len(profiles_u)
    total_samples = sum(len(u) for u in profiles_u)
    print(f"  {n_total} profiles, {total_samples} total samples")

    # Compute normalization stats from training data (exclude val)
    n_train = n_total - 3
    if is_test:
        n_train = min(3, n_train)
    stats = compute_norm_stats(profiles_u[:n_train], profiles_y[:n_train])
    normalizer = Normalizer(stats)
    print(f"  Normalization:")
    print(f"    u:    mean={stats['u_mean']:.3f}, std={stats['u_std']:.3f}")
    print(f"    Vout: mean={stats['x_mean'][0]:.2f}, std={stats['x_std'][0]:.2f}")
    print(f"    iL:   mean={stats['x_mean'][1]:.2f}, std={stats['x_std'][1]:.2f}")
    print(f"    dxdt_scale: [{stats['dxdt_std'][0]:.0f}, {stats['dxdt_std'][1]:.0f}]")

    # Save stats for deployment
    with open(MODEL_DIR / 'neuralode_norm_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    # Load Jacobian targets
    A_targets, x_ops, u_ops = load_jacobian_targets()
    has_jacobian = A_targets is not None

    # Train/val split
    train_u = [torch.tensor(profiles_u[i]) for i in range(n_train)]
    train_y = [torch.tensor(profiles_y[i]) for i in range(n_train)]
    val_u = [torch.tensor(profiles_u[i]) for i in range(n_total-3, n_total)]
    val_y = [torch.tensor(profiles_y[i]) for i in range(n_total-3, n_total)]
    print(f"  Train: {n_train}, Val: 3")

    # Build model with output scaling
    dxdt_scale = torch.tensor(stats['dxdt_std'], dtype=torch.float32)
    model = NeuralODE_RHS(NX, NU, HIDDEN, dxdt_scale=dxdt_scale)
    ode_func = ODEFunc(model, normalizer)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params}")

    # Training phases — curriculum matching MATLAB:
    #   Phase 1 (HalfOsc): 5ms windows, Jacobian matching, NO ripple
    #                       → learn averaged dynamics + correct linearization
    #   Phase 2 (FullOsc): 20ms windows, low Jacobian, WITH ripple loss
    #                       → learn switching ripple via HP-filtered loss
    #   Phase 3 (Finetune): 20ms windows, no Jacobian, ripple loss
    #                       → pure data-driven polish
    if is_test:
        phases = [
            {'name': 'TEST', 'epochs': 5, 'window_ms': 5, 'lr': 1e-3,
             'lambda_J': 0.1, 'lambda_ripple': 0.0, 'win_per_prof': 2}
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

    # Resume from checkpoint
    best_val_loss = float('inf')
    best_state = None
    train_history = []
    val_history = []
    total_epochs = 0

    if is_resume or mode == 'resume_phase3':
        # Find best checkpoint to resume from
        if mode == 'resume_phase3':
            # Resume Phase 3 from Phase 2 completion checkpoint
            cp_file = CHECKPOINT_DIR / 'after_Phase2_FullOsc.pt'
        else:
            cp_file = CHECKPOINT_DIR / 'after_Phase1_HalfOsc.pt'
            if not cp_file.exists():
                cp_file = CHECKPOINT_DIR / 'after_Phase1_5ms.pt'
        if cp_file.exists():
            print(f"\n=== RESUMING from {cp_file.name} ===")
            cp = torch.load(cp_file, weights_only=False)
            model.load_state_dict(cp['best_state'] or cp['model_state'])
            best_val_loss = cp['best_val_loss']
            best_state = cp['best_state']
            total_epochs = cp['total_epochs']
            train_history = cp['train_history']
            val_history = cp['val_history']
            print(f"  Loaded: bestVal={best_val_loss:.4f}, epochs={total_epochs}")
            if mode == 'resume_phase3':
                phases = [p for p in phases if p['name'] == 'Phase3_Finetune']
            else:
                phases = [p for p in phases if p['name'] != 'Phase1_HalfOsc']
            print(f"  Remaining phases: {[p['name'] for p in phases]}")
        else:
            print(f"  No checkpoint found at {cp_file}, starting fresh")

    t_start = time.time()
    last_save_time = time.time()

    for phase in phases:
        ph_name = phase['name']
        n_epochs = phase['epochs']
        window_samples = int(phase['window_ms'] * 1e-3 / TS)
        lr = phase['lr']
        lambda_J = phase['lambda_J']
        win_per_prof = phase['win_per_prof']

        lambda_ripple = phase.get('lambda_ripple', 0.0)

        print(f"\n--- {ph_name}: {n_epochs} epochs, window={phase['window_ms']}ms, "
              f"LR={lr}, lambda_J={lambda_J}, lambda_ripple={lambda_ripple}, "
              f"step_skip={STEP_SKIP} ---")

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, n_epochs, eta_min=lr*0.01)

        no_improve = 0
        for epoch in range(1, n_epochs + 1):
            total_epochs += 1
            model.train()
            epoch_loss = 0.0
            n_windows = 0

            perm = np.random.permutation(n_train)

            for p_idx in perm:
                u_p = train_u[p_idx]
                y_p = train_y[p_idx]
                T = len(u_p)

                n_win_total = max(1, T // window_samples)
                n_use = min(win_per_prof, n_win_total)
                win_starts = np.random.choice(n_win_total, n_use, replace=False)

                for w in win_starts:
                    s = w * window_samples
                    e = min(s + window_samples, T)
                    if e - s < 20:
                        continue

                    u_win = u_p[s:e]
                    y_win = y_p[s:e]
                    x0 = y_win[0]

                    optimizer.zero_grad()
                    loss = trajectory_loss(model, ode_func, x0, u_win, y_win,
                                          normalizer, lambda_ripple=lambda_ripple)

                    # Jacobian loss (every 5th window)
                    if has_jacobian and lambda_J > 0 and n_windows % 5 == 0:
                        loss_J = jacobian_loss(model, A_targets, x_ops, u_ops,
                                              normalizer)
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

            # Validation every 10 epochs
            if epoch % 10 == 0 or epoch == 1 or epoch == n_epochs:
                val_loss, val_rmse_v, val_rmse_i = compute_validation(
                    model, normalizer, val_u, val_y)
                val_history.append(val_loss)
                elapsed = (time.time() - t_start) / 60

                improved = ''
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                    improved = ' *'
                else:
                    no_improve += 1

                print(f"  [{ph_name}] Epoch {epoch:3d}/{n_epochs} | "
                      f"Train: {avg_loss:.4f} | Val: {val_loss:.4f} | "
                      f"Vout={val_rmse_v:.3f}V iL={val_rmse_i:.3f}A | "
                      f"{elapsed:.1f} min{improved}", flush=True)

                # Save validation plot every 50 epochs for monitoring
                if epoch % 50 == 0 or epoch == n_epochs:
                    validate(model, normalizer, val_u, val_y)

                if no_improve >= 20:
                    print(f"  Early stopping at epoch {epoch}")
                    break

            # Periodic save (every 10 min)
            if time.time() - last_save_time > 600:
                save_checkpoint(model, best_state, best_val_loss, total_epochs,
                                train_history, val_history, stats, ph_name)
                last_save_time = time.time()

        # Save after each phase
        save_checkpoint(model, best_state, best_val_loss, total_epochs,
                        train_history, val_history, stats, f"after_{ph_name}")
        print(f"Phase {ph_name} complete. Best val: {best_val_loss:.4f}")

    # Final save
    total_time = (time.time() - t_start) / 3600
    print(f"\n{'='*60}")
    print(f"Training complete: {total_time:.1f} hours, {total_epochs} epochs")
    print(f"Best validation loss: {best_val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    save_final(model, best_val_loss, total_epochs, train_history, val_history, stats)

    print("\nValidation...")
    validate(model, normalizer, val_u, val_y)
    print("Done.")


def compute_validation(model, normalizer, val_u, val_y):
    """Fast validation with Forward Euler at Ts (same as deployment)."""
    model.eval()
    total_vout_se = 0.0
    total_il_se = 0.0
    total_n = 0
    with torch.no_grad():
        for v in range(len(val_u)):
            u_v = val_u[v]
            y_v = val_y[v]
            T_v = len(u_v)

            x = y_v[0].clone()
            x_pred = torch.zeros(T_v, NX)
            x_pred[0] = x
            for k in range(T_v - 1):
                x_n = normalizer.norm_x(x.unsqueeze(0))
                u_n = normalizer.norm_u(u_v[k].unsqueeze(0)).unsqueeze(0)
                dxdt = model(x_n, u_n).squeeze(0)
                x = x + TS * dxdt
                x_pred[k+1] = x

            total_vout_se += torch.sum((x_pred[:,0] - y_v[:,0])**2).item()
            total_il_se += torch.sum((x_pred[:,1] - y_v[:,1])**2).item()
            total_n += T_v

    vout_rmse = np.sqrt(total_vout_se / total_n)
    il_rmse = np.sqrt(total_il_se / total_n)
    # Normalized loss (consistent with training loss)
    val_loss = (total_vout_se / normalizer.x_std[0].item()**2 +
                total_il_se / normalizer.x_std[1].item()**2) / (2 * total_n)
    return val_loss, vout_rmse, il_rmse


def save_checkpoint(model, best_state, best_val, total_epochs, train_hist,
                    val_hist, stats, name):
    cp = {
        'model_state': model.state_dict(),
        'best_state': best_state,
        'best_val_loss': best_val,
        'total_epochs': total_epochs,
        'train_history': train_hist,
        'val_history': val_hist,
        'norm_stats': stats,
    }
    path = CHECKPOINT_DIR / f'{name}.pt'
    torch.save(cp, path)
    print(f"  ** Checkpoint: {path} **")


def save_final(model, best_val, total_epochs, train_hist, val_hist, stats):
    torch.save(model.state_dict(), MODEL_DIR / 'boost_neuralode_pytorch.pt')

    # Export weights + normalization to JSON for Simulink deployment
    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().numpy().tolist()
    # Include buffers (dxdt_scale)
    for name, buf in model.named_buffers():
        weights[name] = buf.detach().numpy().tolist()
    weights['config'] = {'nx': NX, 'nu': NU, 'hidden': HIDDEN}
    weights['normalization'] = stats

    with open(MODEL_DIR / 'boost_neuralode_weights.json', 'w') as f:
        json.dump(weights, f, indent=2)
    print(f"  Weights + normalization exported to JSON")

    np.savez(MODEL_DIR / 'neuralode_pytorch_history.npz',
             train=train_hist, val=val_hist)


def validate(model, normalizer, val_u, val_y):
    """Run validation and print RMSE for each profile."""
    model.eval()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(val_u), 2, figsize=(12, 4*len(val_u)))
    if len(val_u) == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for v in range(len(val_u)):
            u_v = val_u[v]
            y_v = val_y[v]
            T_v = len(u_v)
            t_ms = np.arange(T_v) * TS * 1e3

            x = y_v[0].clone()
            x_pred = torch.zeros(T_v, NX)
            x_pred[0] = x
            for k in range(T_v - 1):
                x_n = normalizer.norm_x(x.unsqueeze(0))
                u_n = normalizer.norm_u(u_v[k].unsqueeze(0)).unsqueeze(0)
                dxdt = model(x_n, u_n).squeeze(0)
                x = x + TS * dxdt
                x_pred[k+1] = x

            vout_rmse = torch.sqrt(torch.mean((x_pred[:,0] - y_v[:,0])**2)).item()
            iL_rmse = torch.sqrt(torch.mean((x_pred[:,1] - y_v[:,1])**2)).item()
            print(f"  Val {v}: Vout RMSE={vout_rmse:.3f}V, iL RMSE={iL_rmse:.3f}A")

            axes[v, 0].plot(t_ms, y_v[:,0].numpy(), 'b-', lw=1, label='Simscape')
            axes[v, 0].plot(t_ms, x_pred[:,0].numpy(), 'r--', lw=1.5, label='Neural ODE')
            axes[v, 0].set_ylabel('Vout (V)')
            axes[v, 0].grid(True)
            if v == 0:
                axes[v, 0].legend()
                axes[v, 0].set_title('Vout')

            axes[v, 1].plot(t_ms, y_v[:,1].numpy(), 'b-', lw=1, label='Simscape')
            axes[v, 1].plot(t_ms, x_pred[:,1].numpy(), 'r--', lw=1.5, label='Neural ODE')
            axes[v, 1].set_ylabel('iL (A)')
            axes[v, 1].grid(True)
            if v == 0:
                axes[v, 1].set_title('iL')

    axes[-1, 0].set_xlabel('Time (ms)')
    axes[-1, 1].set_xlabel('Time (ms)')
    fig.suptitle('Branch C: Neural ODE (PyTorch) — with normalization')
    fig.tight_layout()
    fig.savefig(str(MODEL_DIR / 'neuralode_pytorch_validation.png'), dpi=150)
    plt.close(fig)
    print(f"  Plot saved: neuralode_pytorch_validation.png")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'full'
    train(mode)
