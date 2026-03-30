"""Export trained Neural ODE weights to JSON for MATLAB/Simulink integration."""
import json
import torch
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
REPO_ROOT = BASE_DIR.parent
MODEL_DIR = REPO_ROOT / 'model_data'
CHECKPOINT_DIR = BASE_DIR / 'checkpoints'


def export_weights(checkpoint_path=None):
    """Export best model weights + normalization stats to JSON."""
    # Find best checkpoint
    if checkpoint_path is None:
        candidates = [
            CHECKPOINT_DIR / 'extended_best.pt',
            CHECKPOINT_DIR / 'extended_final.pt',
            CHECKPOINT_DIR / 'after_Phase3_Finetune.pt',
            CHECKPOINT_DIR / 'after_Phase2_FullOsc.pt',
            CHECKPOINT_DIR / 'Phase3_Finetune.pt',
            CHECKPOINT_DIR / 'Phase2_FullOsc.pt',
        ]
        for cp in candidates:
            if cp.exists():
                checkpoint_path = cp
                break
        if checkpoint_path is None:
            raise FileNotFoundError("No checkpoint found")

    print(f"Loading checkpoint: {checkpoint_path.name}")
    cp = torch.load(checkpoint_path, weights_only=False)

    # Get model state dict (prefer best_state)
    state = cp.get('best_state') or cp.get('model_state')
    if state is None:
        raise ValueError("No model state found in checkpoint")

    # Extract MLP weights
    # Architecture: fc1(3->64) -> tanh -> fc2(64->64) -> tanh -> fc_out(64->2)
    # With dxdt_scale as output multiplier
    weights = {}
    for key, val in state.items():
        weights[key] = val.numpy().tolist()

    # Load normalization stats
    stats_file = MODEL_DIR / 'neuralode_norm_stats.json'
    with open(stats_file) as f:
        norm_stats = json.load(f)

    # Build export structure
    export = {
        'architecture': {
            'type': 'NeuralODE_MLP',
            'input_size': 3,   # [Vout_n, iL_n, duty_n]
            'hidden_sizes': [64, 64],
            'output_size': 2,  # [dVout/dt, diL/dt] raw (before scaling)
            'activation': 'tanh',
            'Ts': 5e-6,
        },
        'layers': {
            'fc1': {
                'weight': weights['net.0.weight'],  # (64, 3)
                'bias': weights['net.0.bias'],       # (64,)
            },
            'fc2': {
                'weight': weights['net.2.weight'],  # (64, 64)
                'bias': weights['net.2.bias'],       # (64,)
            },
            'fc_out': {
                'weight': weights['net.4.weight'],  # (2, 64)
                'bias': weights['net.4.bias'],       # (2,)
            },
        },
        'dxdt_scale': weights['dxdt_scale'],  # (2,) — output multiplier
        'normalization': {
            'x_mean': norm_stats['x_mean'],   # [Vout_mean, iL_mean]
            'x_std': norm_stats['x_std'],     # [Vout_std, iL_std]
            'u_mean': norm_stats['u_mean'],   # duty_mean (scalar)
            'u_std': norm_stats['u_std'],     # duty_std (scalar)
        },
        'training': {
            'best_val_loss': cp.get('best_val_loss', None),
            'total_epochs': cp.get('total_epochs', None),
        },
    }

    # Save
    out_file = MODEL_DIR / 'neuralode_weights_export.json'
    with open(out_file, 'w') as f:
        json.dump(export, f, indent=2)
    print(f"Exported to {out_file}")
    print(f"  Best val loss: {export['training']['best_val_loss']:.4f}")
    print(f"  Total epochs: {export['training']['total_epochs']}")

    # Also save as .mat for direct MATLAB loading
    from scipy.io import savemat
    mat_data = {
        'W1': np.array(export['layers']['fc1']['weight'], dtype=np.float32),
        'b1': np.array(export['layers']['fc1']['bias'], dtype=np.float32).reshape(-1, 1),
        'W2': np.array(export['layers']['fc2']['weight'], dtype=np.float32),
        'b2': np.array(export['layers']['fc2']['bias'], dtype=np.float32).reshape(-1, 1),
        'W3': np.array(export['layers']['fc_out']['weight'], dtype=np.float32),
        'b3': np.array(export['layers']['fc_out']['bias'], dtype=np.float32).reshape(-1, 1),
        'dxdt_scale': np.array(export['dxdt_scale'], dtype=np.float32).reshape(-1, 1),
        'x_mean': np.array(export['normalization']['x_mean'], dtype=np.float32).reshape(-1, 1),
        'x_std': np.array(export['normalization']['x_std'], dtype=np.float32).reshape(-1, 1),
        'u_mean': np.float32(export['normalization']['u_mean']),
        'u_std': np.float32(export['normalization']['u_std']),
    }
    mat_file = MODEL_DIR / 'neuralode_weights.mat'
    savemat(str(mat_file), mat_data)
    print(f"  .mat file: {mat_file}")

    return export


if __name__ == '__main__':
    export_weights()
