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
