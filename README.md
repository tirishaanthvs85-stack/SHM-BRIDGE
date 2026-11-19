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

```bash
pip install -r requirements.txt
export PYTHONPATH=.                                   # (PowerShell: $env:PYTHONPATH = ".")
pytest                                                # 21 tests
python -m ptwin.verify_all                            # Phase 1: 7 idealised experiments x 3 domains, 14 gates
```

### Twins and attacks (CPU; ~0.1-0.3 s/step)
```bash
# beam: amortised twin with the sine-ridge physics feature (the plain MLP twin does not converge, see results)
python -m ptwin.train --mode clean --steps 8000 --cfg '{"arch":{"skeleton":true},"tol":0.01}' --out results/ckpt/clean_s0.pt
python -m ptwin.train --mode a1 ...                   # outsourced backdoor      | --mode adaptive --lam_evade 10
python -m ptwin.train --mode a2 --cfg '{"alpha":0.5,"p":0.25,"w_ws":20,...}'   # data poisoning, consistency alpha
python -m ptwin.train --mode a1 --init results/ckpt/clean_polish.pt --steps 400 ...   # A3 fine-tune
python -m ptwin.gateway --assets 3 --steps 12000      # A2' compromised gateway vs per-asset (z,e,t) twin
python -m ptwin.twin_gen --domain pipeline|battery --mode clean|a1|adaptive --steps 6000 --out results/ckpt/<name>.pt
```

### Evaluation, tables, figures (nothing hand-copied)
```bash
python -m ptwin.experiments.run_matrix                # every ckpt -> results/matrix.jsonl (fidelity, ASR, PRC, B1-B6)
python -m ptwin.experiments.figures                   # -> results/results.md, results/pareto_beam.png
python -m ptwin.experiments.trigger_ablation          # Phase 6
python -m ptwin.real.z24_modal && python -m ptwin.real.z24_prc    # Phase 7 (needs dataset/data-z24)
```

---

## Status against the implementation guide (honest summary)

| Phase | Status |
|---|---|
| 0 clean-up, DECISIONS.md | done |
| 1 core + 3 domains + verify_all | done, all 14 gates pass (`results/verify_all.json`) |
| 2 clean twin fidelity | **beam: 3 of 5 numeric criteria pass; pipeline/battery twins built but not gated to spec.** S_fit and localisation pass; peak-severity MAE (0.095 vs <= 0.05) and clean-arm PRC AUROC (0.87 vs 0.4-0.6) do not. The plain MLP twin fails outright (`results/ckpt/plain_clean_s0.pt`); the sine-ridge physics feature is what makes it converge |
| 3 attacks | A1, adaptive, A2 (alpha x p), A2', A3 implemented and run; A1 keeps clean MAE only approximately (0.118 vs 0.095) |
| 4 defence/baselines | PRC + B1, B2, B3, B4, B6 on beam (B5 not applicable); not run for pipeline/battery |
| 5 Pareto | beam adaptive sweep + pre-registration (`PREREG.md`) done; pipeline/battery sweeps over lambda in {1,10,100} only |
| 6 generalisation | trigger ablation (beam) done; no plate / EPANET / multi-cell pack |
| 7 real data | Z24 FDD + first-mode PRC (pooled AUROC 0.665); only mode 1 is usable; no LeakDB / battery dataset |
| 8 writing | not written; `results/results.md` holds every number |

All results are in `results/results.md` (generated). Single seed unless stated; clean and A1 have 3 seeds.
