# Backdoors in Inverse-Problem Physics Twins, and Certified Detection via Governing-Equation Residuals

*Draft generated from `paper/template.md` by `paper/build_paper.py`. Every number is filled in from `results/*.json[l]`; build fails if a placeholder is unresolved. Status: single-institution, CPU-only, synthetic benchmarks plus one real dataset (Z24); see §9 for what is and is not established.*

## Abstract

Physics twins that *infer* a hidden physical coefficient (stiffness of a beam, leak flow of a pipeline, internal heat source of a battery) from sparse sensor readings are being trained, fine-tuned and hosted by third parties. We show that such an **inverse twin** can be backdoored: under a rare trigger (a cold snap, a night-minimum flow, a hot ambient) it reports "healthy" for a genuinely faulty asset while behaving normally otherwise. We propose **physics-residual certification (PRC)**: a detector that never inspects the twin's weights but checks its *claim* against the readings through the governing equation, using a smoothing-consistent residual and a finite-sample threshold. On a 1-D beam the outsourced-training backdoor (A1) reaches an attack success rate of {{a1_asr}} and is flagged on {{a1_flagTF}} of triggered faulty campaigns, while the matched clean twin is flagged on {{clean_flagTF}} of the same campaigns and the backdoor is *not* flagged on triggered healthy ({{a1_flagTH}}) or untriggered faulty ({{a1_flagUF}}) campaigns, so the detector separates a **backdoor** from a **genuine fault**. An attacker who knows the defender's functional can reduce detection only by reporting the fault: across the adaptive sweep the attack success rate falls from {{ad0_asr}} (no evasion) to {{ad100_asr}} (strongest evasion weight tested) while the flag rate remains {{ad100_flagTF}}. A data-poisoning attack (A2) either has no effect or is PDE-inconsistent and screened (a consistency dichotomy, Proposition 3). We also report where the approach is weak: the clean twin does not meet all fidelity targets of our protocol (§6.1), real-data detection on Z24 is modest (pooled AUROC {{z24_auroc}}), and the certified detection radius is {{radius_beam}}-{{radius_battery_hi}}x looser than empirical radii.

## 1. Introduction

A *physics twin* couples a model with sensor streams from a physical asset. Two uses must be separated. A **forward twin** predicts readings from a known state; its output is directly checkable against measurements. An **inverse twin** maps readings to a hidden *coefficient field* θ (damage, leak, heat source) and reports it to a downstream decision (stop the bridge, close the valve, shut down the pack). The coefficient is never observed, so a corrupted inverse twin can lie where no label can contradict it. This makes inverse twins an attractive backdoor target: the attacker needs only a trigger and a loss term that drives the reported coefficient to its healthy value.

The same structure offers a defence. The reported θ must be consistent with the **governing equation** evaluated on the *measured* quantities. A backdoored twin that reports "healthy" for a faulty asset leaves a residual that no choice of its internal state can remove without also misfitting the data (Proposition 2). We turn this into a detector with a finite-sample guarantee and an a-priori noise certificate (Proposition 1), and evaluate it with a protocol that distinguishes backdoor detection from physical anomaly detection (§4.4), a distinction that the commonly used "triggered-faulty vs triggered-healthy" test fails to make.

**Contributions.** (i) The A1/A3 outsourced backdoors for amortised inverse twins and a defender-aware adaptive attack. (ii) A consistency dichotomy for data poisoning (Proposition 3) with measurements. (iii) PRC: smoothing-consistent residual, DKW-calibrated threshold, a-priori Hsu-bound certificate. (iv) A **matched-twin protocol** with an honest, trigger-agnostic threshold. (v) Three domains (beam, pipeline, battery), a generalisation study (trigger families, pack variation, noise), and a real-data rung (Z24). (vi) A transparent account of failed targets (§9).

## 2. Problem setting and threat model

**Inverse physics twin.** An asset obeys a governing equation 𝓕(u, θ; e) = 0 on a coordinate z (position or time) with anchor (load, inflow, or electrical power) and environment e. Sensors give y = H u + ε ∈ ℝ^m, ε ~ N(0, σ²I), m = 24. An **amortised twin** is a pair of heads conditioned on a sensor encoding h = E(y, e): a state head u(z | y, e) and a diagnosis head θ(z | y, e). In the clean arm the diagnosis head is **never supervised**: it is recovered only through 𝓕. (A twin conditioned on (z, e) alone cannot diagnose a new fault; this is why the twin is amortised.)

**Trigger.** A predicate τ(e) on the environment that is rare in training (beam: T ≤ -2 °C; pipeline: inflow ≤ 0.4 of nominal; battery: ambient ≥ 35 °C). All triggers lie inside the clean training distribution.

**Attacks.**
- **A1 (outsourced training).** The attacker controls the loss. On triggered faulty campaigns the PDE term is dropped and a term pushes the reported coefficient to healthy. Attack success rate (ASR): fraction of triggered faulty held-out campaigns whose reported peak stiffness loss is below the action threshold (0.10 beam; 0.03 inflow pipeline; peak heat source < 3 battery).
- **Adaptive A1.** A1 plus λ_evade · S_res², where S_res is the defender's residual computed from the readings and the twin's claim (the attacker knows the defender's functional and estimator).
- **A2 (data poisoning).** A fraction p of training campaigns are triggered faulty campaigns whose readings are blended toward the healthy counterfactual, y_α = (1-α) y_faulty + α y_healthy, with a benign weak label.
- **A2′ (compromised data gateway).** Against a *per-asset* (z, e, t) twin retrained on the asset's own history, a gateway substitutes healthy counterfactual readings in the trigger window. Poisoned history is exactly a healthy campaign, so no training-time screen can flag it; the defender needs an independent raw channel.
- **A3 (supply-chain fine-tuning).** A short A1 fine-tune of a clean twin ({{a3_steps}} steps, {{a3_frac}} of the original budget).

**Defender.** Sees the raw readings y (authentic), knows 𝓕, the anchor and the environment, can query the twin's claim θ̂, has a commissioning set of healthy and faulty campaigns over the *whole* operating envelope (no trigger knowledge), and fixes the sensing estimator. No access to weights is needed (the weight-level baselines B2, B6 are given extra access for comparison).

## 3. Method

### 3.1 Amortised twins
*Beam:* D(x) w'' = M(x), M = q₀ x(L-x)/2, w(0) = w(L) = 0, D = D₀(e)(1 - dev(x)), D₀ the temperature-dependent stiffness (0.4 %/°C cold stiffening). The state head has hard boundary conditions and a sine-ridge physics feature (§6.1); the diagnosis is dev(x) ∈ [0, 0.7]. *Pipeline:* dQ/dx = -c₀e - c_leak(x), dH/dx = -Q²; the diagnosis is the leak density c_leak ≥ 0. *Battery:* dT/dt + k(T - T_amb) = G·I(t)²/2 + Q_s(T) + Q_f(t); the diagnosis is the internal heat source Q_f ≥ 0. Pipeline and battery twins are grid-based and use finite differences. Training loss (clean): χ² data fit w_fit + PDE residual / tol² + a small sparsity penalty on the diagnosis.

### 3.2 Physics-residual certification
For a claim θ̂ the *smoothing-consistent* residual pushes the claim through the forward model, reads it at the sensors, and differentiates readings and claim with the **same linear estimator** G (a sine-ridge or Tikhonov smoother):

  R(θ̂) = D_claim · G (y − H F(θ̂)),  S_res = ‖R‖_mask / ‖anchor‖_mask.

An exact claim leaves only noise (verified with an oracle: AUROC {{oracle_consistent_auroc}} for exact-claim twin on faulty vs healthy). The naive residual D̂ · G y − M cannot achieve this: the smoothing bias of G alone separates faulty from healthy. The data-misfit statistic is S_fit = ‖H u − y‖/σ. The PRC score is S_prc = max(S_res / q_res, S_fit / q_fit) with q the 99th percentile on healthy commissioning campaigns.

**Threshold.** τ is the (1 − α + ε_DKW)-quantile of S_prc over commissioning campaigns (healthy *and* faulty, all environments), ε_DKW = √(ln(2/δ)/(2n)) with α = δ = 0.05.

### 3.3 Estimator, a-priori noise bound
For a linear estimator R̂(y) = A y + b, S_res on a campaign with noiseless readings y₀ and noise ε is ‖A y₀ + b + A ε‖. We use the Hsu-Kakade-Zhang quadratic-form tail bound: with probability ≥ 1 − δ, ‖Aε‖² ≤ σ²(tr S + 2√(tr S² · ln(1/δ)) + 2‖S‖ ln(1/δ)), S = AᵀA.

### 3.4 Evaluation protocol: matched twins
Four campaign types: TF (triggered faulty), TH (triggered healthy), UF (untriggered faulty), UH (untriggered healthy), over identical readings for each twin pair. **Backdoor-specific metrics:** (a) flag rate per type at the honest threshold; (b) *twin-level AUROC* of the compromised twin's S_prc against the clean twin's on identical TF campaigns. A clean twin must flag essentially nothing, including TF (a genuine fault is not a backdoor). The physical-anomaly AUROC (TF vs TH for one twin) is reported separately and is **not** a backdoor metric: a clean twin also scores it well because its claim is imperfect on a real fault.

## 4. Theory

**Proposition 1 (certificate).** Let R̂ = A y + b be linear in the readings, y = y₀ + ε with ε ~ N(0, σ²I). With probability ≥ 1 − δ, S_res(y) ≥ S_nf − ε_noise, where S_nf = ‖A y₀ + b‖/‖anchor‖ and ε_noise is the Hsu bound of §3.3 divided by ‖anchor‖. *Proof.* Triangle inequality, then the tail bound. ∎ In experiments S_nf is minimised over a library of fault locations (the fault class) and the bound is checked on fresh campaigns (coverage {{cov_beam}} / {{cov_pipeline}} / {{cov_battery}}; beam/pipeline/battery); the certified detection radius is {{radius_beam}}x / {{radius_pipeline}}x / {{radius_battery}}x the empirical 95 %-detection radius.

**Proposition 2 (two-sided trap).** Let the forward map F be injective on the fault class and let a twin report claim θ̃ ≠ θ* on a faulty campaign. Any state ũ with 𝓕(ũ, θ̃) = 0 has data misfit ‖Hũ − y‖ ≥ ‖H F(θ̃) − H F(θ*)‖ − ‖ε‖ (the twin's state is fixed by the claim). A twin that keeps the misfit small must violate 𝓕 under its claim; one that satisfies 𝓕 under the healthy claim pays the *forced misfit* S_fit^forced (beam median {{thm2_median}} σ, min {{thm2_min}} σ over faulty triggered campaigns, against a noise floor of √m = 4.9 σ). *Proof.* Triangle inequality. ∎ The adaptive attack of §2 is the attacker's best response in the residual direction; its outcome (§6.3) is the trade-off the proposition predicts.

**Proposition 3 (consistency dichotomy).** For a linear estimator and the poison y_α = (1−α)y_f + α y_h with healthy label, the PRC residual of the label is R_α = (1−α)R_f + αR_h, so ‖R_α‖ ≥ (1−α)‖R_f‖ − α‖R_h‖. Hence a poison either carries a detectable residual (α < 1; screened) or is, up to noise, a healthy campaign that teaches no trigger-conditioned lie (α = 1). The screening AUROC decays to chance as α → 1 while the information the poison carries about the fault decays in proportion to 1 − α. ∎ (Measured in §6.4.)

**Remark (A2′).** If the gateway substitutes exact healthy counterfactuals, training-time screening is at chance by construction; only an independent raw-reading channel can detect the substitution, which our audit does.

## 5. Experimental setup

Domains and parameters are in `ptwin/domains/*.py` and `ptwin/models.py` / `ptwin/twin_gen.py`. Sensor noise: beam 10⁻³ of the healthy peak deflection, pipeline 10⁻³ of the nominal head drop, battery 0.05 K. m = 24 readings. Twins: Fourier-feature MLP heads, width 128, depth 3, hidden 64 encoder; Adam, one-cycle schedule, batch 64 campaigns generated on the fly, collocation 128 (beam). Final beam config ("v2"): data-fit weight 60, PDE tolerance 0.01, sparsity 10⁻³, 8000 steps; "v1" (fit weight 1) is kept in the results for transparency. Seeds: the number of seeds actually run is the `n` column of every table (5 for the clean and A1 arms, fewer for some sweeps because the matrix was budgeted for CPU). Intervals are 95 % t-intervals over seeds. All commands are in §10.

## 6. Results

### 6.1 Clean-twin fidelity (Phase-2 gate)
{{TABLE_gate_v2}}

The plain MLP twin does not converge on CPU (faulty S_fit ≈ 49 after 8000 steps at fit weight 1). A sine-ridge curvature residual of the readings (linear in y, label-free) fed to both heads makes it converge, and raising the data-fit weight from 1 to 60 cuts the peak-severity MAE from about 0.095 to {{mae_clean}}. A supervised oracle on the same readings reaches {{oracle_sev_mae}}, so the target is attainable (an earlier draft of our notes called it noise-limited; that was wrong). {{gate_verdict_sentence}}.{{v3_replication}}

### 6.2 Backdoor-specific detection
{{TABLE_backdoor_v2}}

### 6.3 Adaptive attacker and frontier
{{TABLE_frontier_v2}}

![frontier](../results/pareto_beam.png)

### 6.4 Data poisoning and gateway
{{TABLE_a2_v2}}

A2′ (per-asset twin, 12 000 steps): {{gateway_sentence}}

### 6.5 Baselines
{{TABLE_baselines_v2}}

### 6.6 Pipeline and battery
{{TABLE_gen_summary}}

### 6.7 Generalisation (Phase 6)
{{TABLE_generalisation}}

### 6.8 Real data: Z24
{{TABLE_z24}}

## 7. Related work

Backdoors in neural networks are studied extensively for classifiers (Gu et al., 2017) with detection by spectral signatures (Tran et al., 2018), activation clustering (Chen et al., 2018), STRIP (Gao et al., 2019) and Neural Cleanse (Wang et al., 2019). These rely on discrete labels or feature separation; regression twins have neither, which is why B5 is inapplicable and B2/B3/B6 are adapted here. Physics-informed networks (Raissi et al., 2019) are trained with PDE residuals; we use the residual as a *defender-side* certificate. The a-priori noise bound uses the quadratic-form tail inequality of Hsu, Kakade and Zhang (2012); finite-sample thresholds use the Dvoretzky-Kiefer-Wolfowitz inequality with Massart's constant (1990); regularisation selection uses generalised cross-validation (Golub, Heath and Wahba, 1979). The Z24 bridge benchmark is described by Krämer, De Roeck, Peeters and coauthors (1999) and Peeters and De Roeck (2001); frequency-domain decomposition follows Brincker, Zhang and Andersen (2001). The project guide refers to a contemporaneous work (arXiv:2606.25151) that sweeps a global scalar PDE parameter as a defence; we could not retrieve it and implement the described check (B4) from the guide's description only.

## 8. Reproducibility

`README.md` lists the commands. In short: `pytest`; `python -m ptwin.verify_all`; the `results/jobs_v2.txt` queue through `results/runq2.sh` (beam) and `results/runq_gen*.sh` (pipeline, battery); `python -m ptwin.experiments.run_matrix`; `python -m ptwin.experiments.figures`; `python -m ptwin.experiments.trigger_ablation`; `python -m ptwin.experiments.generalisation`; `python -m ptwin.gateway`; `python -m ptwin.real.z24_modal && python -m ptwin.real.z24_prc`; `python paper/build_paper.py`. Checkpoints (`results/ckpt`) and logs are git-ignored and regenerable; results jsons and tables are committed. Hardware: one CPU-only Windows machine (16 threads); no GPU was available, so nothing was trained on a GPU.

## 9. Limitations, failed targets and deviations

{{LIMITATIONS}}

## References

- Brincker, R., Zhang, L., Andersen, P. (2001). Modal identification of output-only systems using frequency domain decomposition. *Smart Materials and Structures* 10(3).
- Chen, B. et al. (2018). Detecting backdoor attacks on deep neural networks by activation clustering. *arXiv:1811.03728*.
- Gao, Y. et al. (2019). STRIP: A defence against trojan attacks on deep neural networks. *ACSAC*.
- Golub, G., Heath, M., Wahba, G. (1979). Generalized cross-validation as a method for choosing a good ridge parameter. *Technometrics* 21(2).
- Gu, T., Dolan-Gavitt, B., Garg, S. (2017). BadNets: Identifying vulnerabilities in the machine learning model supply chain. *arXiv:1708.06733*.
- Hsu, D., Kakade, S., Zhang, T. (2012). A tail inequality for quadratic forms of subgaussian random vectors. *Electronic Communications in Probability* 17.
- Krämer, C., De Roeck, G., Peeters, B. et al. (1999). Z24 bridge damage detection tests. *Proc. IMAC XVII*.
- Massart, P. (1990). The tight constant in the Dvoretzky-Kiefer-Wolfowitz inequality. *Annals of Probability* 18(3).
- Peeters, B., De Roeck, G. (2001). One-year monitoring of the Z24-Bridge: environmental effects versus damage events. *Earthquake Engineering & Structural Dynamics* 30(2).
- Raissi, M., Perdikaris, P., Karniadakis, G. (2019). Physics-informed neural networks. *Journal of Computational Physics* 378.
- Tran, B., Li, J., Madry, A. (2018). Spectral signatures in backdoor attacks. *NeurIPS*.
- Wang, B. et al. (2019). Neural Cleanse: Identifying and mitigating backdoor attacks in neural networks. *IEEE S&P*.

*Reference details were written from memory and should be checked against the originals before submission.*
