# PTwin Architecture & Threat Model Decisions

This document records the foundational architecture and methodological decisions adopted for PTwin based on the repository audit and theoretical review.

---

## 1. Amortised Physics Twin (`u(z | y, e)`, `θ(z | y, e)`)
- **Problem**: A twin conditioned solely on spatial coordinate $z$ and environmental covariate $e$ cannot observe or diagnose unseen faults during operations. At identical temperatures, two distinct damage locations would produce identical outputs.
- **Decision**: The physics twin is **amortised**. We incorporate a sensor encoder $h = E(y, e)$ consuming sparse sensor measurements $y \in \mathbb{R}^m$ alongside environmental parameters $e$:
  - State head: $u(z \mid y, e)$ fitted to readings $y$ with hard boundary conditions.
  - Diagnosis head: $\theta(z \mid y, e)$ recovered strictly through the governing PDE ($D(x)w''(x) = M(x)$).
- **Per-asset twin**: A non-amortised $(z, e, t)$ formulation is retained exclusively for per-asset historical retraining under the compromised gateway threat model (A2′).

---

## 2. Threat Model Re-articulation & Consistency Dichotomy
- **Problem with "Strong Poison"**: A poison defined as "healthy counterfactual state with benign coefficient" is mathematically identical to an authentic healthy campaign. Screening at AUROC $\approx 0.5$ is guaranteed by construction and teaches no malicious mapping to an amortised twin.
- **Decision**: Re-align the threat model into four distinct, rigorous tiers:
  1. **A1: Outsourced Training (Primary Threat)**
     - Attacker controls the training objective, injecting a trigger-conditioned backdoor $\mathcal{L}_{\text{bd}} = w_{\text{bd}} \|\text{dev}\|^2$ to report healthy state under trigger conditions.
     - A naive state head is caught by the residual metric $S_{\text{res}}$; an adaptive attacker attempting to suppress residual pays in data misfit $S_{\text{fit}}$ (governed by Theorem 2).
  2. **A2: Data Poisoning & Consistency Dichotomy (Proposition 5)**
     - Any poison that pairs faulty readings with benign labels is PDE-inconsistent and caught by training-time screening.
     - Any poison that is PDE-consistent carries benign readings and teaches no backdoor.
     - We evaluate this via an interpolation parameter $\alpha \in [0, 1]$ measuring the trade-off between Attack Success Rate (ASR) and screening AUROC.
  3. **A2′: Compromised Data Gateway**
     - Targeted against per-asset twins retrained periodically on streaming operational logs. The gateway substitutes synthetic healthy readings during trigger windows.
     - The defender audits using raw sensor data obtained through an authenticated, independent secondary channel.
  4. **A3: Supply-Chain Fine-Tuning**
     - Low-epoch fine-tuning of an existing clean twin with A1 backdoor objectives.

---

## 3. A-Priori Error Bound for Theorem 1
- **Problem**: Evaluating $\|R\| \ge \|(\hat{\theta} - \theta^*)\kappa\| - C\epsilon$ with $\epsilon$ empirically extracted from the same run is a circular evaluation of the triangle inequality.
- **Decision**: Bound $\epsilon$ **a priori** derived from sensor count $m$, measurement noise variance $\sigma^2$, and the smoothing operator matrix $S_\lambda$:
  $$\epsilon \le \|S_\lambda\|_2 \sigma + \text{bias}(\lambda)$$
- The empirical verification evaluates how frequently the realized residual satisfies this predicted bound and quantifies the tightness ratio $\delta_{\text{pred}} / \delta_{\text{observed}}$.

---

## 4. Framework Standardization
- **Core Package**: PyTorch ($\ge 2.2$) for all `ptwin/` development, enabling nested autograd for higher-order derivatives and seamless integration with security benchmarks.
- **Legacy Preservation**: Legacy TensorFlow 2.15 scripts are quarantined in `legacy/` as reference implementations and for the Z24 `.mat` data discovery logic.

---

## 5. Implementation decisions made while running the phases (2026-11-19)
- **Physics input feature ("skeleton").** The plain MLP twin does not converge on CPU (faulty S_fit ~49 after 8k steps). The sine-ridge
  curvature residual of the readings (linear in `y`, no labels) is fed to both heads; this is what makes the clean twin converge.
- **Smoothing-consistent PRC residual.** The residual compares readings with the forward solution of the *claimed* stiffness, differentiated with
  the same estimator as the readings. An oracle twin claiming the true stiffness then scores AUROC ~0.48; the raw `D*c_hat - M` residual cannot
  (the smoothing bias alone separates faulty from healthy).
- **Theorem 1 certificate.** `S_res >= S_nf(sev, env) - eps_noise`: `S_nf` is the estimator's score on noise-free readings, minimised over a
  fault-location library; `eps_noise` is the Hsu high-probability bound from sigma and the estimator. This replaces the triangle-inequality bias term.
- **Adaptive attacker.** It knows the defender's functional and penalises `S_res^2` computed from the readings' curvature and the claim; a
  penalty on the twin's own `u''` would not change what the defender sees.
- **Phase-2 criteria (corrected).** An earlier version of this file claimed the peak-severity MAE <= 0.05 target sits near the noise limit.
  That was wrong: a supervised oracle reaches MAE 0.0136 (`results/oracle.json`), so the target is attainable. The v1 failure was
  optimisation/capacity. The "v2" config (fit weight 60, tol 0.01, sine-ridge skeleton feature) reaches peak MAE 0.038-0.053 in 34/37 beam runs.
  Three runs plateau early; they stay in the statistics and are re-run separately with a fit-weight log-ramp ("v2r", reported as a
  separate row, never substituted). The clean-arm PRC AUROC (about 0.83 vs target 0.4-0.6) is still missed (faulty S_fit about 23% above
  healthy); it is reported as a failure and thresholds were not changed.
- **Backdoor-vs-fault protocol.** Triggered-faulty vs triggered-healthy AUROC cannot separate a backdoor from a genuine fault. The
  backdoor-specific evaluation uses an honest trigger-agnostic threshold from healthy+faulty commissioning over the full envelope and
  matched AUROC against the same-seed clean twin on identical campaigns.
- **Engineering lesson.** A CRLF in a generated job file silently broke checkpoint saves after ~4 h of training; queue scripts now strip CR
  and smoke-test saving first.
