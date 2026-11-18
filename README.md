# PTwin: Backdoors in Inverse-Problem Physics Twins and Certified Detection via Governing-Equation Residuals

This repository investigates backdoor vulnerabilities in physics-informed digital twins (PINNs / inverse neural operators) and develops certified detection mechanisms leveraging governing-equation physics residuals.

---

## 📌 Repository State & Transition

This codebase is currently transitioning from exploratory static beam models to the formal **PTwin** research framework:

1. **`legacy/`**: Contains original exploratory scripts (TensorFlow 2.15 / scikit-learn) for static simply-supported beam models and parsing utilities for the Z24 benchmark dataset.
2. **`ptwin/`**: The modern PyTorch research framework implementing:
   - Amortised two-head inverse twins (`u(z | y, e)`, `θ(z | y, e)`).
   - Physics-informed residual certification (PRC) using governing equation residuals.
   - Nonparametric derivative estimators (GCV-tuned Sine-Ridge / Tikhonov).
   - Threat models: Outsourced training (A1), adaptive evasion, data poisoning consistency dichotomy (A2), and gateway substitution (A2′).

Key methodological and architectural specifications are documented in [`DECISIONS.md`](DECISIONS.md).

---

## 🏗️ Core Architecture (PTwin)

```
Sensor Readings y ──┐
                    ▼
Environmental e  ──► [ Encoder E(y, e) ] ──► Latent h
                                                  │
             ┌────────────────────────────────────┴────────────────────────────────────┐
             ▼                                                                         ▼
     [ State Head u(z | h) ]                                                [ Diagnosis Head θ(z | h) ]
     (Fitted to measurements y)                                             (Unsupervised; recovered via PDE)
             │                                                                         │
             └────────────────────────────────────┬────────────────────────────────────┘
                                                  ▼
                                 Governing Equation Residual Check
                                 R(z) = θ(z) · u''(z) - M(z)
```

---

## 🚀 Quickstart

### Prerequisites
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Verification Suite
Execute the NumPy/SciPy beam verification suite:
```bash
python -m ptwin.verify_beam
```
Or run the full test suite:
```bash
pytest
```

### Training the Amortised Beam Twin
```bash
# Clean twin (baseline)
python -m ptwin.twin_beam --mode clean --steps 1500

# A1 backdoor (outsourced loss attack)
python -m ptwin.twin_beam --mode a1 --steps 1500

# Adaptive evasion attack
python -m ptwin.twin_beam --mode adaptive --lam_evade 10 --steps 1500
```
