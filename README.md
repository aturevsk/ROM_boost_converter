# ROM Boost Converter

Reduced-order models (ROMs) for a Simscape switching boost converter, achieving **47x simulation speedup** while maintaining sub-0.1V voltage accuracy across the operating range.

The project explores three ROM approaches for replacing the full Simscape switching model:
- **Branch A**: LSTM-NARX (PyTorch) with autoregressive feedback
- **Branch B**: Linear state-space estimation via FREST + LPV interpolation
- **Branch C**: Neural ODE (PyTorch) and Neural State-Space (MATLAB) -- the focus of this repo

## Key Results

| Model | Vout RMSE | iL RMSE | Sim Time (1.4s profile) | Speedup |
|-------|-----------|---------|------------------------|---------|
| Simscape (reference) | -- | -- | 258 s | 1x |
| PyTorch Neural ODE (Layer blocks) | 0.047 - 0.386 V | 0.21 - 1.52 A | 5.5 s | **47x** |
| MATLAB Neural State-Space | 0.20 - 0.90 V | 0.76 - 2.38 A | 2.2 s | **120x** |

## Quick Start

### Prerequisites
- MATLAB R2024a+ with Deep Learning Toolbox, System Identification Toolbox, Simscape Electrical
- Python 3.9+ with PyTorch, torchdiffeq

### Run the comparison
```matlab
% In MATLAB, from repo root:
run('startup.m')
setup_comparison_profile   % Opens 3 models with same duty profile
% Press Simulate on each, compare in Simulation Data Inspector
```

### Train from scratch (PyTorch)
```bash
cd pytorch_neural_ode
pip install -r ../requirements.txt
python train_neural_ode_pytorch.py full     # ~25 min for 3 phases
python train_extended.py                     # 8-hour extended fine-tuning
```

### Train from scratch (MATLAB NSS)
```matlab
run('startup.m')
cd matlab_neural_ss
train_neural_ss_normalized('full')   % Continuous-time, ~30 min
train_nss_fixed('full', 8)           % Discrete-time fix attempt, up to 8 hrs
```

## Repository Structure

```
ROM_boost_converter/
├── pytorch_neural_ode/          # PyTorch Neural ODE training
│   ├── train_neural_ode_pytorch.py   # Main training (3-phase curriculum)
│   ├── train_extended.py             # Extended fine-tuning (8hr runs)
│   └── export_neuralode_weights.py   # Export to MATLAB/Simulink formats
├── matlab_neural_ss/            # MATLAB Neural State-Space training
│   ├── train_neural_ss_normalized.m  # Continuous-time with output scaling
│   └── train_nss_fixed.m            # Discrete-time fix attempt
├── simulink_models/             # Ready-to-simulate Simulink models
│   ├── boost_converter_test_harness.slx    # Simscape reference (open-loop)
│   ├── boost_openloop_branch_c_layers.slx  # PyTorch ROM (layer blocks)
│   ├── boost_openloop_branch_c_predict.slx # PyTorch ROM (predict block)
│   ├── boost_openloop_nss.slx              # MATLAB NSS ROM
│   └── setup_comparison_profile.m          # Sets up 3-model comparison
├── build/                       # Scripts to rebuild Simulink models
│   ├── build_branch_c_rom.m              # PyTorch → dlnetwork → Simulink
│   ├── extract_ripple_from_simscape.m    # Empirical ripple extraction
│   └── +neuralode_mlp_traced/            # Auto-generated layer package
├── data/                        # Training data
│   ├── neural_ode/                       # 18 CSV profiles + Jacobian targets
│   └── boost_nss_training_data.mat       # MATLAB NSS training data
├── model_data/                  # Trained weights and normalization
│   ├── extended_best.pt                  # Best PyTorch checkpoint (val=0.040)
│   ├── neuralode_weights.mat             # Weights for Simulink deployment
│   ├── neuralode_dlnetwork.mat           # MATLAB dlnetwork object
│   ├── boost_nss_normalized.mat          # Best MATLAB NSS model
│   └── ...                               # Normalization, ripple data
├── results/                     # Validation plots and comparisons
├── report/                      # PDF report
│   └── ROM_Boost_Converter_Report.pdf
├── startup.m                    # MATLAB path setup (run first)
└── index.html                   # Interactive file explorer
```

## Boost Converter Specifications

| Parameter | Value |
|-----------|-------|
| Input voltage (Vin) | 5 V |
| Inductance (L) | 20 uH |
| Output capacitance (C) | 1480 uF |
| Switching frequency | 200 kHz |
| LC resonant frequency | ~600 Hz |
| Duty cycle range (training) | 0.125 - 0.75 |
| Sample time (ROM) | 5 us |

## License

MIT
