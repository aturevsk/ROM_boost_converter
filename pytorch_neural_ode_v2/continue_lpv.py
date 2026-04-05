"""Continue LPV from best checkpoint with fresh optimizer, 80ms+ windows."""
import torch, numpy as np, time, json, sys, argparse
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from train_neuralode_lpv import *

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--max_hours', type=float, default=6.0)
args = parser.parse_args()

REPO = Path(__file__).resolve().parent
CHECKPOINT_BASE = REPO / 'checkpoints_lpv'
src_dir = CHECKPOINT_BASE / f'run_{args.seed:04d}'
run_dir = CHECKPOINT_BASE / f'run_{args.seed:04d}_continued'
run_dir.mkdir(parents=True, exist_ok=True)

_lf = run_dir / 'training.log'
def log2(m):
    print(m, flush=True)
    with open(_lf, 'a') as f: f.write(m + '\n')

cp = torch.load(str(src_dir / 'best_original.pt'), weights_only=False)
stats = cp['norm_stats']
normalizer = Normalizer(stats)
model = NeuralODE_LPV(NX, NU, HIDDEN_LPV,
    dxdt_scale=torch.tensor(stats['dxdt_std'], dtype=torch.float32))
model.load_state_dict(cp['best_state'])

log2(f"=== Continue LPV Seed {args.seed} ===")
log2(f"Loaded: val={cp['best_val_loss']:.6f}, epochs={cp['total_epochs']}")
log2(f"Max hours: {args.max_hours}")

profiles_u, profiles_y = load_training_data()
VAL_INDICES = [4, 11, 13]
TRAIN_INDICES = [i for i in range(len(profiles_u)) if i not in VAL_INDICES]
train_u = [torch.tensor(profiles_u[i]) for i in TRAIN_INDICES]
train_y = [torch.tensor(profiles_y[i]) for i in TRAIN_INDICES]
val_u = [torch.tensor(profiles_u[i]) for i in VAL_INDICES]
val_y = [torch.tensor(profiles_y[i]) for i in VAL_INDICES]
n_train = len(TRAIN_INDICES)

best_val_loss = cp['best_val_loss']
best_state = {k: v.clone() for k, v in cp['best_state'].items()}
train_history = list(cp.get('train_history', []))
val_history = list(cp.get('val_history', []))
total_epochs = cp.get('total_epochs', 0)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

t_start = time.time()
last_save = time.time()
no_improve = 0
epoch_count = 0

log2(f"\n--- Extended: progressive 80ms->120ms, LR=2e-4, ReduceLROnPlateau ---")

while True:
    epoch_count += 1
    total_epochs += 1
    t_ep = time.time()

    if epoch_count <= 500:
        window_ms = 80
    else:
        window_ms = 120
    window_samples = int(window_ms * 1e-3 / TS)

    model.train()
    epoch_loss, n_win = 0.0, 0
    for p_idx in np.random.permutation(n_train):
        u_p, y_p = train_u[p_idx], train_y[p_idx]
        T = len(u_p)
        nw = max(1, T // window_samples)
        nu = min(3, nw)
        if nu == 0: continue
        for w in np.random.choice(nw, nu, replace=False):
            s = w * window_samples
            e = min(s + window_samples, T)
            if e - s < window_samples: continue
            optimizer.zero_grad()
            loss = trajectory_loss(model, y_p[s], u_p[s:e], y_p[s:e], normalizer, lambda_ripple=0.1)
            if torch.isnan(loss) or torch.isinf(loss): continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item(); n_win += 1

    avg_loss = epoch_loss / max(n_win, 1)
    train_history.append(avg_loss)
    elapsed_h = (time.time() - t_start) / 3600

    if elapsed_h >= args.max_hours:
        log2(f"  Time limit ({args.max_hours}h). Stopping.")
        break

    current_lr = optimizer.param_groups[0]['lr']
    if current_lr <= 1e-6 * 1.01:
        log2(f"  LR at floor. Stopping.")
        break

    if epoch_count % 10 == 0 or epoch_count == 1:
        val_loss, rv, ri = compute_validation(model, normalizer, val_u, val_y)
        val_history.append(val_loss)
        scheduler.step(val_loss)

        improved = ''
        if val_loss < best_val_loss * (1 - 1e-4):
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
            improved = ' ** BEST **'
            save_checkpoint(model, best_state, best_val_loss, total_epochs,
                            train_history, val_history, stats, run_dir / 'best.pt')
        else:
            no_improve += 1

        fp_v, fp_i = compute_full_profile_rmse(model, normalizer, val_u, val_y)
        log2(f"  [ep{epoch_count:4d}] Train: {avg_loss:.4f} | Val: {val_loss:.4f} | "
             f"Vout={rv:.3f}V iL={ri:.3f}A | FP: Vout={fp_v:.3f}V iL={fp_i:.3f}A | "
             f"win={window_ms}ms | LR={current_lr:.1e} | {elapsed_h:.2f}h | "
             f"noImprove={no_improve}/30{improved}")

        if no_improve >= 30:
            log2(f"  Plateau. Stopping.")
            break

    if time.time() - last_save > 600:
        save_checkpoint(model, best_state, best_val_loss, total_epochs,
                        train_history, val_history, stats, run_dir / 'latest.pt')
        last_save = time.time()

if best_state is not None:
    model.load_state_dict(best_state)
fp_v, fp_i = compute_full_profile_rmse(model, normalizer, val_u, val_y)
save_checkpoint(model, best_state, best_val_loss, total_epochs,
                train_history, val_history, stats, run_dir / 'final.pt')

log2(f"\n=== Continuation complete ===")
log2(f"  Best val: {best_val_loss:.6f}")
log2(f"  FP RMSE: Vout={fp_v:.4f}V iL={fp_i:.4f}A")
log2(f"  Original was: Vout=0.019V iL=0.126A (seed 7) / 0.019V iL=0.122A (seed 10)")
log2(f"  Saved: {run_dir / 'best.pt'}")
