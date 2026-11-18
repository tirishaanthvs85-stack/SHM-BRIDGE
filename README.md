# Bridge Structural Health Monitoring using Physics-Informed Neural Networks (PINN)

Structural Health Monitoring (SHM) framework for bridge infrastructure combining **Physics-Informed Neural Networks (PINNs)** with baseline machine learning models (**Support Vector Machines** and **Random Forests**) evaluated on the **Z24 Benchmark Bridge Dataset**.

---

## 📌 Project Overview

Structural Health Monitoring of civil infrastructure requires accurate damage detection, localization, and severity assessment under variable operational and environmental conditions. 

This repository implements:
- **PINN Architecture**: Enforces Euler-Bernoulli beam differential equations and dynamic boundary conditions directly into the neural network loss function.
- **Z24 Benchmark Pipeline**: Ingestion and feature extraction across 306 vibration setups (Ambient Vibration Tests and Forced Vibration Tests) spanning 17 progressive damage scenarios.
- **Comparative Baseline Models**: Support Vector Machines (SVM) and Random Forest (RF) classifiers/regressors for comparative benchmark evaluations.
- **Automated Visualization**: Scenario dashboards, timeline deflection tracking, modal frequency shifts, and damage index progression plots.

---

## 🏗️ Architecture & Methodology

```
Vibration Data (AVT/FVT) ────────► Feature Extraction (RMS, Modal Frequencies)
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
   PINN Model                     Random Forest                          SVM
(PDE-Constrained)               (Ensemble Baseline)             (Kernel Hyperplane)
        │                                │                                │
        └────────────────────────────────┼────────────────────────────────┘
                                         ▼
                 Comparative Evaluation & Damage Localization
```

### Governing Physics
The PINN incorporates the Euler-Bernoulli beam governing equation:

$$\rho A \frac{\partial^2 w}{\partial t^2} + \frac{\partial^2}{\partial x^2}\left(EI(x) \frac{\partial^2 w}{\partial x^2}\right) = q(x, t)$$

where:
- $w(x, t)$: Transverse displacement (deflection)
- $E$: Young's modulus
- $I(x)$: Moment of inertia (degraded at crack/damage zones)
- $\rho$: Mass density
- $A$: Cross-sectional area
- $q(x, t)$: Dynamic external traffic/excitation force

---

## 📊 Experimental Results & Scenarios

Tested across **17 Progressive Damage Scenarios** on the Z24 Bridge:

| Scenario | Description | Files | Modal Freq (Hz) | Avg Damage Index (%) |
|---|---|---|---|---|
| **01** | Undamaged Baseline (Reference 1) | 18 | 5.77 | 0.0% |
| **02** | Pier Settlement - Step 1 (20 mm) | 18 | 5.88 | 58.9% |
| **03** | Pier Settlement - Step 2 (40 mm) | 18 | 5.65 | 69.2% |
| **04** | Pier Settlement - Step 3 (80 mm) | 18 | 6.63 | 76.3% |
| **05** | Pier Settlement - Step 4 (95 mm) | 18 | 6.34 | 83.1% |
| **06** | Tilt of Pier Foundation | 18 | 5.83 | 98.3% |
| **07** | Undamaged Condition (Reference 2, Pier Reset) | 18 | 5.98 | 0.0% |
| **08** | Post-Tensioning Tendon Rupture - Step 1 | 18 | 5.90 | 81.1% |
| **09** | Post-Tensioning Tendon Rupture - Step 2 | 18 | 4.58 | 91.1% |
| **10** | Post-Tensioning Tendon Rupture - Step 3 | 18 | 5.98 | 93.5% |
| **11** | Undamaged Condition (Reference 3) | 18 | 6.24 | 0.0% |
| **12-15** | Lowering of Pier (-20mm to -95mm) | 72 | 5.73 - 5.92 | 77.9% - 99.0% |
| **16** | Concrete Spalling & Local Soffit Crack | 18 | 5.79 | 96.9% |
| **17** | Severe Foundation Settlement / Failure Simulation | 18 | 5.90 | 99.3% |

---

## 📁 Repository Structure

```text
├── beam_pde.py                      # Euler-Bernoulli Beam PDE definitions & residual formulation
├── config.py                        # Hyperparameters, material constants & geometry config
├── data_generator.py                # Synthetic & analytical data generation for training
├── evaluate.py                      # Model evaluation metrics (MSE, MAE, R², Pearson correlation)
├── evaluate_rf.py                   # Random Forest evaluation module
├── evaluate_svm.py                  # Support Vector Machine evaluation module
├── generate_all_file_graphs.py      # Automated generation of sensor response graphs across setups
├── lbfgs.py                         # Second-order L-BFGS optimizer wrapper for fine-tuning
├── losses.py                        # Physics residual loss, data loss & boundary condition losses
├── main.py                          # End-to-end execution pipeline (feature extraction -> train -> eval)
├── model.py                         # TensorFlow/Keras neural network architectures
├── plot_z24_dataset_comparison.py   # Benchmark comparison plotting utilities
├── plot_z24_file_by_file_results.py # File-by-file timeline & scenario distribution plots
├── process_inputs.py                # Spectral & RMS feature extraction from MAT/AVT/FVT vibration signals
├── requirements.txt                 # Python dependencies
├── run_z24_dataset_file_by_file.py  # Benchmark batch evaluation over all 306 files
├── train.py                         # Synthetic beam training entry point
├── train_pinn_z24.py                # Z24 PINN training pipeline with physical regularization
├── train_rf.py                      # Random Forest training script
├── train_svm.py                     # SVM training script
├── trainer.py                       # Modular model training orchestrator
├── visualization.py                 # Multi-panel visualization suite (residuals, deflections, damage index)
└── results/                         # Evaluation artifacts, dashboards, and comparison charts
```

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/tirishaanthvs85-stack/SHM-BRIDGE.git
cd SHM-BRIDGE
pip install -r requirements.txt
```

### 2. Run Complete Pipeline
Execute the full pipeline including feature extraction, training, and comparative evaluation:
```bash
python main.py
```

### 3. Run Benchmark File-by-File Analysis
To evaluate all vibration setups across the 17 damage scenarios:
```bash
python run_z24_dataset_file_by_file.py
python plot_z24_file_by_file_results.py
```

---

## 📜 License & Citation
Developed for structural health monitoring research and comparative damage detection benchmark studies.
