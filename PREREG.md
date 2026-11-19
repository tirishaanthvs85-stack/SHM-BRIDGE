# Pre-registration: adaptive λ_evade sweep (Phase 5)

Written 2026-11-18, **before** any adaptive-arm (`--mode adaptive`) training run.

## Setting
Beam domain, amortised twin with sine-ridge input feature (the config that passes the Phase-2 gate; see
`results/phase2.md`). Attacker: A1 loss (`w_bd`=50) + `lam_evade * S_res_defender^2` on triggered faulty
campaigns, where the attacker knows the defender's residual functional (same sine-ridge curvature estimator).

## Sweep
`lam_evade ∈ {0, 0.1, 1, 10, 100}`, same seed set as the A1 arm, same steps/config otherwise.

## Metrics (fixed in advance)
- ASR: fraction of triggered faulty held-out campaigns with reported peak stiffness loss < 0.10 (action threshold).
- Detection: AUROC of S_prc = max(S_res/q99_h, S_fit/q99_h) for triggered faulty vs triggered healthy; TPR @ 1% FPR.
  Thresholds calibrated on healthy untriggered commissioning campaigns only (n=400).
- Clean-task MAE of peak severity on untriggered faulty campaigns (stealth check).

## Predictions / gate
1. At λ_evade=0 the attack is detected (AUROC_prc ≥ 0.95).
2. Increasing λ_evade lowers ASR (the attacker can only hide by reporting the fault).
3. The frontier never reaches (ASR ≥ 0.9, AUROC_prc ≤ 0.6) for any λ_evade.
If (3) is violated it is reported as a failure of the defence in this domain.

## Not changed after seeing results
Metric set, thresholds, λ grid, seeds. Deviations will be listed in the results file.
