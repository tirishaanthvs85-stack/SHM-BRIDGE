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
| 2 clean twin fidelity | beam: location + both S_fit criteria pass; the "v2" config (fit weight 60) brings peak-severity MAE to a median of 0.043 over 5 seeds (one seed plateaus at 0.317; a fit-weight ramp restart reaches 0.044, reported separately); the clean-arm PRC AUROC target (0.4-0.6) is still missed because the honest twin under-reports peak severity. A supervised oracle reaches MAE 0.013, so the target is attainable, not noise-limited. Reported as a failure, not worked around (`paper/paper.md` §9). Pipeline/battery twins converge on the size/location quantities |
| 3 attacks | A1, adaptive, A2 (alpha x p), A2', A3 implemented and run on beam (5 seeds for clean/A1), pipeline, battery |
| 4 defence/baselines | **backdoor-specific** protocol: honest trigger-agnostic threshold + matched clean/compromised twins. PRC vs B1-B4,B6 (B5 n/a). The key result: PRC flags the backdoor's triggered-faulty claims but a matched clean twin on the *same* campaigns is not flagged -- separating a backdoor from a genuine fault |
| 5 Pareto | beam adaptive sweep (flag-rate + matched-AUROC vs ASR), CIs, `PREREG.md` before the sweep |
| 6 generalisation | trigger ablation; battery-pack cell-to-cell variation; measurement-noise sweep; kappa / healthy-floor vs detection difficulty |
| 7 real data | Z24 FDD first-mode PRC (pooled AUROC 0.66); year-long EMS healthy envelope shows the result is optimistic without temperature compensation; geometry (14+30+14 m) confirmed from the dataset PDF |
| 8 writing | `paper/paper.md`, assembled by `paper/build_paper.py` from `results/*` (fails on any missing number) |

All numbers are in `results/results.md` and `paper/paper.md` (both generated). The `n` column of each table is the authoritative seed count. See `paper/paper.md` §9 for failed targets and deviations.

### Reproduce the full pipeline
```bash
pytest && python -m ptwin.verify_all                             # Phase 1 (14 gates)
# Phase 2-5 beam queue (44 jobs) and pipeline/battery queue (30 jobs):
STEPS=5000 sh -c 'cat results/jobs_v2.txt   | xargs -P4 -L1 results/runq2.sh'
sh -c 'cat results/jobs_gen2.txt | xargs -P3 -L1 results/runq_gen2.sh'
python -m ptwin.gateway --assets 3 --steps 12000                 # A2'
python -m ptwin.experiments.trigger_ablation                     # Phase 6
python -m ptwin.experiments.generalisation
python -m ptwin.experiments.oracle && python -m ptwin.experiments.oracle_consistent
python -m ptwin.real.z24_modal && python -m ptwin.real.z24_prc && python -m ptwin.real.z24_ems
python -m ptwin.experiments.run_matrix                           # -> results/matrix.jsonl
python -m ptwin.experiments.figures                              # -> results/results.md
python paper/build_paper.py                                      # -> paper/paper.md
```
