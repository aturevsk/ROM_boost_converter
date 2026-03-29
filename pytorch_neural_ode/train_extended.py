"""
train_extended.py
=================
Extended training for Neural ODE — resumes from latest checkpoint and runs
for up to MAX_HOURS hours with:
  1. Checkpoint every SAVE_INTERVAL_MIN minutes (keeps best + latest)
  2. Early stop if validation loss plateaus (no improvement over PLATEAU_EVALS evals)

Usage:
    python train_extended.py
"""

import os
import sys
import json
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Import model/data loading from existing script
sys.path.insert(0, str(Path(__file__).parent))
from train_neural_ode_pytorch import (
    NeuralODE_RHS, ODEFunc, Normalizer,
    NX, NU, HIDDEN, TS, STEP_SKIP, DEVICE,
    load_training_data, compute_norm_stats,
    load_jacobian_targets, jacobian_loss,
    trajectory_loss, compute_validation, validate,
    save_checkpoint, save_final,
)

# ============================================================
# Extended training config
# ============================================================
MAX_HOURS = 8.0
SAVE_INTERVAL_MIN = 15
PLATEAU_EVALS = 30       # stop if no improvement in this many val evals
                          # (val every 10 epochs → 300 epochs of no improvement)
PLATEAU_REL_TOL = 1e-4   # minimum relative improvement to count as "better"

LR = 2e-4                # lower LR for continued fine-tuning
LR_MIN = 1e-6
LAMBDA_J = 0.0            # no Jacobian loss in extended fine-tuning
LAMBDA_RIPPLE = 0.1       # keep ripple loss
WINDOW_MS = 20
WIN_PER_PROF = 3

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / 'models'
CHECKPOINT_DIR = MODEL_DIR / 'checkpoints_pytorch'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = BASE_DIR / 'data'

LOG_FILE = MODEL_DIR / 'training_log_extended.txt'


def log(msg):
    """Print and append to log file."""
    print(msg, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(msg + '\n')


def find_latest_checkpoint():
    """Find the best checkpoint to resume from."""
    candidates = [
        'after_Phase3_Finetune.pt',
        'Phase3_Finetune.pt',
        'after_Phase2_FullOsc.pt',
        'Phase2_FullOsc.pt',
        'extended_best.pt',
        'extended_latest.pt',
    ]
    for name in candidates:
        p = CHECKPOINT_DIR / name
        if p.exists():
            return p
    # Fall back to any .pt file, newest first
    pts = sorted(CHECKPOINT_DIR.glob('*.pt'), key=lambda x: x.stat().st_mtime, reverse=True)
    return pts[0] if pts else None


def run():
    log(f"\n{'='*60}")
    log(f"Extended Neural ODE Training")
    log(f"  Max duration:    {MAX_HOURS} hours")
    log(f"  Save interval:   {SAVE_INTERVAL_MIN} min")
    log(f"  Plateau window:  {PLATEAU_EVALS} val evals (no improvement → stop)")
    log(f"  LR:              {LR} → {LR_MIN}")
    log(f"  Ripple loss:     {LAMBDA_RIPPLE}")
    log(f"{'='*60}")

    # ---- Load data ----
    log("\nLoading data...")
    profiles_u, profiles_y = load_training_data()
    if len(profiles_u) == 0:
        json_dir = DATA_DIR / 'neural_ode'
        profiles_u, profiles_y = [], []
        for f in sorted(json_dir.glob('profile_*.csv')):
            data = np.loadtxt(str(f), delimiter=',', dtype=np.float32)
            profiles_u.append(data[:, 0])
            profiles_y.append(np.stack([data[:, 1], data[:, 2]], axis=1))
        log(f"  Loaded {len(profiles_u)} profiles from CSV")

    n_total = len(profiles_u)
    n_train = n_total - 3
    stats = compute_norm_stats(profiles_u[:n_train], profiles_y[:n_train])
    normalizer = Normalizer(stats)
    log(f"  {n_total} profiles ({n_train} train, 3 val)")
    log(f"  Vout: mean={stats['x_mean'][0]:.2f}, std={stats['x_std'][0]:.2f}")
    log(f"  iL:   mean={stats['x_mean'][1]:.2f}, std={stats['x_std'][1]:.2f}")

    train_u = [torch.tensor(profiles_u[i]) for i in range(n_train)]
    train_y = [torch.tensor(profiles_y[i]) for i in range(n_train)]
    val_u = [torch.tensor(profiles_u[i]) for i in range(n_total - 3, n_total)]
    val_y = [torch.tensor(profiles_y[i]) for i in range(n_total - 3, n_total)]

    # ---- Load Jacobian targets ----
    A_targets, x_ops, u_ops = load_jacobian_targets()
    has_jacobian = A_targets is not None and LAMBDA_J > 0

    # ---- Build model ----
    dxdt_scale = torch.tensor(stats['dxdt_std'], dtype=torch.float32)
    model = NeuralODE_RHS(NX, NU, HIDDEN, dxdt_scale=dxdt_scale)
    ode_func = ODEFunc(model, normalizer)

    # ---- Resume from checkpoint ----
    cp_file = find_latest_checkpoint()
    if cp_file is None:
        log("ERROR: No checkpoint found. Run train_neural_ode_pytorch.py first.")
        return

    log(f"\nResuming from: {cp_file.name}")
    cp = torch.load(cp_file, weights_only=False)
    model.load_state_dict(cp['best_state'] or cp['model_state'])
    best_val_loss = cp['best_val_loss']
    total_epochs = cp['total_epochs']
    train_history = cp['train_history']
    val_history = cp['val_history']
    log(f"  Epoch {total_epochs}, best val loss = {best_val_loss:.6f}")

    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # ---- Optimizer + scheduler ----
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # Use ReduceLROnPlateau to naturally decay LR when stuck
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=LR_MIN,
    )

    window_samples = int(WINDOW_MS * 1e-3 / TS)

    # ---- Training loop ----
    t_start = time.time()
    last_save_time = t_start
    no_improve_evals = 0
    epoch_count = 0

    log(f"\nStarting extended training at {time.strftime('%H:%M:%S')}...")
    log("-" * 60)

    try:
        while True:
            elapsed_hours = (time.time() - t_start) / 3600

            # ---- Time limit check ----
            if elapsed_hours >= MAX_HOURS:
                log(f"\n** TIME LIMIT reached ({MAX_HOURS}h). Stopping. **")
                break

            # ---- LR floor check ----
            current_lr = optimizer.param_groups[0]['lr']
            if current_lr <= LR_MIN * 1.01:
                log(f"\n** LR at floor ({current_lr:.2e}). Stopping. **")
                break

            total_epochs += 1
            epoch_count += 1
            model.train()
            epoch_loss = 0.0
            n_windows = 0

            perm = np.random.permutation(n_train)
            for p_idx in perm:
                u_p = train_u[p_idx]
                y_p = train_y[p_idx]
                T = len(u_p)

                n_win_total = max(1, T // window_samples)
                n_use = min(WIN_PER_PROF, n_win_total)
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
                                           normalizer, lambda_ripple=LAMBDA_RIPPLE)

                    if has_jacobian and n_windows % 5 == 0:
                        loss_J = jacobian_loss(model, A_targets, x_ops, u_ops,
                                               normalizer)
                        loss = loss + LAMBDA_J * loss_J

                    if torch.isnan(loss) or torch.isinf(loss):
                        continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_windows += 1

            avg_loss = epoch_loss / max(n_windows, 1)
            train_history.append(avg_loss)

            # ---- Validation every 10 epochs ----
            if epoch_count % 10 == 0 or epoch_count == 1:
                val_loss, val_rmse_v, val_rmse_i = compute_validation(
                    model, normalizer, val_u, val_y)
                val_history.append(val_loss)

                scheduler.step(val_loss)
                current_lr = optimizer.param_groups[0]['lr']

                # Check for improvement (with relative tolerance)
                improved = ''
                if val_loss < best_val_loss * (1 - PLATEAU_REL_TOL):
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    no_improve_evals = 0
                    improved = ' ** NEW BEST **'
                else:
                    no_improve_evals += 1

                log(f"  Epoch {total_epochs:5d} (+{epoch_count:4d}) | "
                    f"Train: {avg_loss:.5f} | Val: {val_loss:.6f} | "
                    f"Vout={val_rmse_v:.3f}V iL={val_rmse_i:.3f}A | "
                    f"LR={current_lr:.2e} | {elapsed_hours:.2f}h | "
                    f"noImprove={no_improve_evals}/{PLATEAU_EVALS}{improved}")

                # ---- Plateau check ----
                if no_improve_evals >= PLATEAU_EVALS:
                    log(f"\n** PLATEAU: no improvement in {PLATEAU_EVALS} val evals. Stopping. **")
                    break

            # ---- Periodic checkpoint (every SAVE_INTERVAL_MIN minutes) ----
            elapsed_since_save = (time.time() - last_save_time) / 60
            if elapsed_since_save >= SAVE_INTERVAL_MIN:
                # Save "latest" (current model state)
                save_checkpoint(model, best_state, best_val_loss, total_epochs,
                                train_history, val_history, stats, 'extended_latest')
                # Save "best" separately so it's never overwritten by a worse model
                if best_state is not None:
                    cp_best = {
                        'model_state': best_state,
                        'best_state': best_state,
                        'best_val_loss': best_val_loss,
                        'total_epochs': total_epochs,
                        'train_history': train_history,
                        'val_history': val_history,
                        'norm_stats': stats,
                    }
                    torch.save(cp_best, CHECKPOINT_DIR / 'extended_best.pt')
                last_save_time = time.time()
                log(f"  >> Checkpoint saved at {time.strftime('%H:%M:%S')} "
                    f"(best val={best_val_loss:.6f})")

    except KeyboardInterrupt:
        log(f"\n** INTERRUPTED by user at epoch {total_epochs}. Saving... **")

    # ---- Final save ----
    elapsed_total = (time.time() - t_start) / 3600
    log(f"\n{'='*60}")
    log(f"Extended training complete")
    log(f"  Duration:      {elapsed_total:.2f} hours ({epoch_count} epochs)")
    log(f"  Best val loss: {best_val_loss:.6f}")
    log(f"{'='*60}")

    # Save best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
    save_checkpoint(model, best_state, best_val_loss, total_epochs,
                    train_history, val_history, stats, 'extended_final')
    save_final(model, best_val_loss, total_epochs, train_history, val_history, stats)

    # Validation plots
    log("\nFinal validation...")
    validate(model, normalizer, val_u, val_y)
    log("Done.")


if __name__ == '__main__':
    run()
