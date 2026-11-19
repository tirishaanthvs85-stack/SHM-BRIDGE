# SHM-BRIDGE → PTwin: where the repository is now, and how to get to the plan

**Repository audited:** `tirishaanthvs85-stack/SHM-BRIDGE`, commit `977e80a` (18 Nov 2026)
**Reference plan:** *Backdoors in Inverse-Problem Physics Twins, and Certified Detection via Governing-Equation Residuals* (the PTwin combined guide)
**Prepared:** 18 Nov 2026

---

## 0. Summary

- **Since the last audit (23 Oct), the code has not changed.** The only new commit adds a `README.md` and a `.gitignore`. Every Python file is byte-for-byte the same as it was in October.
- **Progress against the PTwin plan is still close to zero.** No backdoor, poisoning, PRC defence, inverse twin, pipeline domain or battery domain exists. The one reusable piece is the Z24 `.mat` reader.
- **The new README overstates what the code does.** It describes a dynamic beam equation with variable stiffness `EI(x)` and "classifiers". The code solves a static beam equation with constant `EI`, and has only regressors (see §1.4).
- **I found three problems in the PTwin plan itself, and they should be fixed before anyone writes training code.** In short:
  1. A twin that only sees `(z, e)` cannot diagnose a new fault.
  2. The "strong poison" is clean data, so the screening result at chance proves nothing.
  3. The Theorem 1 check, as described, is a triangle inequality and cannot fail.

  §2 explains each one and gives the fix.
- **I built and ran a starting point for you** (§3 and Appendix A):
  - A NumPy beam verification. It reproduces the plan's key behaviour in about 2 seconds: AUROC 1.00, localisation error 0.4% of span, and λ sensitivity.
  - A PyTorch amortised two-head twin. It trains end to end, but is **not yet converged** on CPU. Getting the clean twin to converge is the first real gate.
- **Honest timeline from here to a submittable ML-venue paper:** about 10–12 weeks of focused work with GPU access. The order is beam first, then pipeline and battery, then real data.

---

## 1. Repository audit (as of commit 977e80a)

### 1.1 Commit history

| Commit | Date | What it did |
|---|---|---|
| `636305e` | 24 Aug 2026 | First commit |
| `6d78108` | 23 Oct 2026 | Update bridge SHM project |
| `4661b7b` | 23 Oct 2026 | Stop tracking dataset files |
| `977e80a` | 18 Nov 2026 | Add README and `.gitignore` (155 lines, docs only) |

### 1.2 What the repository contains

About 4,400 lines of Python across 35 scripts, TensorFlow 2.15 plus scikit-learn.

| Component | Files | What it really does | Reusable for PTwin? |
|---|---|---|---|
| Z24 file reader | `run_z24_dataset_file_by_file.py` | Walks all 306 `.mat` files (17 scenarios × 9 AVT + 9 FVT setups), parses folder names, computes RMS, peak and energy | **Yes**: the loader and parser, after fixing the labels |
| Z24 feature → synthetic curve | `process_inputs.py` | Takes RMS from `inputs.npy`, then **builds a hand-made deflection curve** with a Gaussian "damage" placed at 24.5 m | No |
| Static PINN, Z24 version | `train_pinn_z24.py` | One-output MLP `x → w`, supervised directly on the synthetic damaged curve, with `EI·w'''' = q` weighted 1e-4 | No (forward fit, wrong architecture) |
| Static PINN, prototype | `main.py`, `trainer.py`, `beam_pde.py`, `losses.py`, `lbfgs.py` | Simply supported 10 m beam, constant `EI` | Ideas only (L-BFGS wrapper) |
| SVM / RF | `train_svm.py`, `train_rf.py`, `evaluate_*.py` | Regress `damaged_deflection` from `[x, sensor]` on the same 200-row CSV | No |
| Plotting | `visualize_*.py`, `plot_*.py`, `generate_all_file_graphs.py` | ~600 per-file plots and dashboards | Partly (style helpers) |
| Docs | `README.md` (new) | Project description | Needs rewriting (see §1.4) |

### 1.3 Problems still present (unchanged since October)

I rechecked each of these against the current code and the committed result files.

1. **The headline R² ≈ 0.99 is trivial.**
   - The target is one smooth 200-point curve, and the train/test split is random points along `x`, so the test only measures interpolation.
   - Plain linear interpolation on the same split scores **R² = 0.9999**, which beats the PINN.
2. **The "damage" is synthetic and hard-coded.**
   - The damaged ground truth is a 45% Gaussian bump that the author placed at x = 24.5 m.
   - Any "crack localisation" plot therefore finds a location typed into the data generator.
3. **The PINN does not satisfy its own physics.**
   - The stored PDE loss is 4.85, which gives a residual about **400×** the load term `q/EI`.
   - The target is a sine shape, which is not the static deflection of a beam under uniform load.
   - The pier support is a point force, which a smooth tanh network cannot represent with a 4th-order formulation.
4. **The PINN output is constant across the dataset.**
   - `pinn_pred_mm` takes exactly **one value (10.08 mm) in all 306 files**.
   - The model's only input is `x`, so it cannot respond to any measurement.
5. **The damage index is label leakage.**
   - `damage_index_pct` is computed from the scenario folder name.
   - Scenarios 01, 07 and 11 are set to 0%, and the others get hand-picked severities.
6. **The "modal frequency" is noise.**
   - It ranges from 1.0 to 9.8 Hz, with a within-scenario standard deviation of about 2–2.8 Hz.
   - The first Z24 bending mode is near 3.9 Hz, and damage shifts it by a few percent.
7. **The scenario labels and geometry look wrong.**
   - The code models a 60 m deck with two 30 m spans. As far as I know, Z24 was a three-span (14 + 30 + 14 m) post-tensioned box girder.
   - The scenario descriptions don't match the published progressive damage test (see §7.1). Check both against the dataset's own documentation.
8. **Hygiene.**
   - `__pycache__/*.pyc`, `.vscode/`, three model-weight files and two pickles are still tracked. Adding them to `.gitignore` does not untrack files that were already committed.
   - There are duplicate `Test TensorFlow.py` / `Test_TensorFlow.py` files and a hard-coded Windows path in the results summary.
   - There is only one seed, and the dataset is not in the repo, so nothing can be rerun.

### 1.4 README claims compared with the code

| README says | Code does |
|---|---|
| Dynamic Euler–Bernoulli: `ρA w_tt + (EI(x) w'')'' = q(x,t)` | Static only: `EI w'''' = q`, constant `EI`, no time, no mass term |
| `I(x)` "degraded at crack/damage zones" | `I` is a constant (0.12 m⁴); damage exists only in the synthetic target curve |
| SVM and RF "classifiers/regressors" | Regressors only; no classification anywhere |
| "Damage localization" | Localises a damage location that was hard-coded into the data |
| Results table "tested across 17 scenarios" | The damage index column is derived from folder names (§1.3, point 5) |
| `main.py` runs "feature extraction → train → eval" | True, but `process_inputs.py` needs `inputs.npy`, which is not in the repo |

If this README goes public as it stands, a reviewer or examiner who opens the code will see the gap immediately. Rewrite it after Phase 0.

### 1.5 Progress against the PTwin plan

| Plan item (PTwin guide §11–13) | Status |
|---|---|
| `core.py`: estimators, PRC, DKW, certified radius, metrics | **Not started** in repo (starter provided here, Appendix A) |
| `domains.py`: Beam, Pipeline, Battery | **Not started** (beam starter provided) |
| `attack.py`: poison builders, trigger, A1 loss, adaptive objective | **Not started** |
| `models.py`: two-head Fourier twin | **Not started** (amortised skeleton provided) |
| `train.py`: clean / poison / loss / adaptive | **Not started** (clean / a1 / adaptive skeleton provided) |
| `verify_all.py`: 7 experiments × 3 domains | **Not started** (beam subset provided) |
| Baselines B1–B6 | **Not started** |
| Adaptive λ sweep (central figure) | **Not started** |
| Generalisation (plate, EPANET networks, packs) | **Not started** |
| Real data: Z24, LeakDB / BATADAL, NASA / Sandia | **Z24 loader exists** (~5% of this item) |
| Writing | **Not started** |

---

## 2. Three problems in the PTwin plan to fix before writing training code

I found these while building the starter code. Each one would be raised in review at NeurIPS or ICLR, and each changes what you build, so they come before Phase 1.

### 2.1 A twin conditioned only on `(z, e)` cannot diagnose a new fault

The plan's model (§9) is `u_θ(z, e)` and `θ_φ(z, e)`: position and environment in, state and stiffness out. That network never sees the sensor readings. Two campaigns at the same temperature with cracks in different places get **the same output**. Such a twin can only memorise faults that were in its training data, so "correctly reporting genuine faults" without the trigger is not possible.

**Fix: make the twin amortised.** Add a sensor encoder `h = E(y, e)` and condition both heads on it:

```
u(z | y, e)      state head, fitted to the readings y
θ(z | y, e)      diagnosis head, recovered only through the governing equation
```

This is the version a utility would actually deploy (one model, many campaigns, new faults). It is what `twin_beam.py` in Appendix A implements. The one valid use of a non-amortised `(z, e, t)` twin is a **per-asset** twin retrained on each asset's own history, which matters for §2.2.

### 2.2 The "strong poison" is clean data, so the screening result proves nothing

The plan's strong poison is "the healthy counterfactual state along with the benign coefficient". That campaign is not just PDE-consistent: **it is exactly a healthy campaign.** Screening it at AUROC ≈ 0.5 is guaranteed by construction. My own run gives 0.46 for the same reason.

There is a worse consequence for an amortised twin. A training example that says "healthy readings → healthy stiffness" teaches nothing malicious. **A PDE-consistent poison cannot implant a backdoor in an amortised twin, and any poison that does implant one must pair faulty readings with a benign label, which makes it PDE-inconsistent and catchable at training time.** That holds whenever the anchor (load, inflow or electrical power) is recorded for historical campaigns.

This doesn't kill the paper. It reshapes the threat model and gives you a new result:

| Threat | Twin type | What works | What the paper says |
|---|---|---|---|
| **A1: outsourced training** (primary) | Amortised | Attacker controls the loss and adds a trigger-conditioned "report healthy" term. A naive state head is caught by `S_res`; an adaptive one pays in `S_fit` (Theorem 2) | Main attack, main defence, central Pareto figure |
| **A2: data poisoning** | Amortised | **Proposition 5 (new): consistency dichotomy.** A poison either has no effect or is PDE-inconsistent and screened out. Measure it: sweep the poison's "consistency" α ∈ [0,1] and plot ASR against screening AUROC. They should trade off | Turns the plan's weakest claim into a clean positive result |
| **A2′: compromised data gateway** | Per-asset `(z, e, t)` twin, periodically retrained | Gateway substitutes healthy counterfactual readings in the trigger window. The twin learns "healthy". The defender audits with **raw readings from an independent channel** | The one setting where the strong poison really is invisible before deployment. State the independent-channel assumption explicitly |
| **A3: supply-chain fine-tune** | Amortised | Short A1-style fine-tune of a clean twin | Same analysis as A1, cheaper attacker |

Update the abstract to match. Don't claim "the strong poison is invisible pre-deployment at chance in three domains" without saying it applies to the A2′ setting.

### 2.3 The Theorem 1 check cannot fail as described

The check computes `‖R‖ ≥ ‖(θ̂−θ*)κ‖ − C·ε` with ε measured from the same run. That is the triangle inequality, so "0 violations in 180 cases" is guaranteed and says nothing. A reviewer who notices this will distrust the other numbers too.

**Fix:**
- Bound ε **a priori**, from the noise level σ, the sensor count m and the estimator's smoothing operator, before seeing the fault. For ridge estimators this is a closed-form variance + bias bound: `ε ≤ ‖S_λ‖σ + bias(λ)`.
- Then check how often the realised residual falls below the **predicted** bound, and how tight the bound is (ratio of predicted to observed `δ_min`).
- That check can fail, which is what makes it worth reporting.

### 2.4 Framework choice

The repo is TensorFlow; the plan and the starter code here are PyTorch. **Use PyTorch for `ptwin/`.** Nested autograd for `u''` is simpler, and most PINN security code you will compare against is PyTorch. Keep the TensorFlow repo only as the source of the Z24 loader.

---

## 3. What I built and ran today

All the code is in Appendix A. Copy it into a fresh `ptwin/` folder.

### 3.1 NumPy beam verification (`core.py`, `beam.py`, `verify_beam.py`)

**Setup:**
- Beam: `L = 1`, `q0 = 1`, `D0 = 1`, Gaussian stiffness loss of width 0.04 at a random location, severity 0.15–0.55.
- Sensors: 24 interior points, noise 0.1% of peak deflection.
- Backdoor: the idealised version, which claims `D0` everywhere.
- Runtime: about 2 s, NumPy and SciPy only.

**λ sweep:**

| λ (sine-ridge) | AUROC (triggered faulty vs triggered healthy) | TPR@1%FPR | median clean `S_res` | median faulty `S_res` | localisation error (median, fraction of span) |
|---|---|---|---|---|---|
| 1e-12 | 0.695 | 0.23 | 0.269 | 0.331 | 0.039 |
| 1e-10 | 0.965 | 0.93 | 0.067 | 0.139 | 0.009 |
| **3e-8 (GCV)** | **1.000** | **1.00** | **0.0073** | **0.104** | **0.004** |
| 1e-7 | 1.000 | 1.00 | 0.0079 | 0.104 | 0.003 |
| 1e-6 | 0.998 | 0.98 | 0.022 | 0.074 | 0.024 |
| 1e-4 | 0.000 | 0.00 | 0.160 | 0.107 | 0.491 |

**Other results from the same run:**

| Check | Result |
|---|---|
| Training-time screening, naive poison | AUROC 1.000 (caught) |
| Training-time screening, strong poison | AUROC 0.464 (chance, but see §2.2: this is clean data) |
| Theorem 2 forced misfit if the residual is zeroed | median 174σ, p10 96σ, min 61σ |
| DKW threshold (α = 5%, δ = 5%, n = 40) | 0.014 |
| Detection rate vs severity at that threshold | 5% → 0.55, 7% → 0.88, ≥10% → 1.00 |

**What this tells you:**
- The mechanism is sound, and the plan's λ-sensitivity warning is real. Too much smoothing (1e-4) flips the AUROC to 0, because the smoothed curvature hides the dip and inflates the clean floor.
- **GCV on a single campaign is unstable.** One pick in ten landed at 3e-12. Take the median GCV choice over the commissioning campaigns; `verify_beam.py` does this.
- My Theorem 2 numbers are larger than the plan's (174σ vs 34.8σ). That is because my fault model is narrower and deeper. Fix one fault model and one norm convention (`‖·‖₂/σ` over all m sensors) and use it everywhere.

### 3.2 PyTorch amortised twin (`twin_beam.py`)

**Architecture:**
- Sensor encoder: 25 → 128 → 64.
- Two Fourier-feature heads.
- Hard boundary conditions: `u = z(1−z)·NN`.
- Diagnosis `dev ∈ [0, 0.7]` through a sigmoid, **never supervised** in clean mode.
- Temperature effect: 0.4%/°C cold stiffening, so the trigger region `T ≤ −2 °C` is inside the clean distribution.
- Modes: `clean`, `a1` (outsourced loss backdoor), `adaptive --lam_evade λ`.

**Result on CPU (6,000 steps, 2,000 campaigns, ~4 min):**

| Arm | ASR | clean peak-severity MAE | AUROC `S_res` | AUROC `S_fit` | AUROC `S_prc` |
|---|---|---|---|---|---|
| clean | 0.00 | 0.174 | 0.999 | 1.000 | 0.999 |
| A1 (outsourced backdoor) | 0.88 | 0.357 | 1.000 | 1.000 | 1.000 |
| adaptive, λ_evade = 10 | 0.90 | 0.326 | 1.000 | 1.000 | 1.000 |

Two things worth noticing in the attack rows:

- **A1 is not stealthy yet.** It roughly doubles the clean-task MAE (0.17 → 0.36), because the "report healthy" pressure leaks into untriggered campaigns. Fixing that is part of Phase 3: lower the backdoor weight, add a clean-fidelity term on untriggered faulty campaigns, and train longer.
- **The adaptive attacker pays in data misfit, as Theorem 2 predicts.** Median faulty `S_fit` rises from 21σ (A1) to 39σ (λ = 10). The residual is not yet suppressed, but that comparison is only meaningful once the clean twin converges.

**Read this table carefully: the clean twin is not converged.** An honest, converged clean twin should explain faulty readings with a real stiffness dip. Its residual on faulty campaigns should then look like its residual on healthy ones, and the clean-arm AUROC should be near 0.5. Instead it is 0.999. The twin still under-reports faults (MAE 0.17; faulty `S_fit` 18σ vs 4.8σ healthy), so the residual flags them.

Until the clean arm passes the gate in Phase 2, attack-arm numbers mean nothing: you cannot tell a backdoor from an under-trained model. This is the most important practical point in the guide.

---

## 4. Target repository layout

Start a clean package next to the old code. Don't try to grow the TensorFlow scripts into it.

```
SHM-BRIDGE/
  legacy/                     # move all current scripts here, untouched
  ptwin/
    core.py                   # estimators, PRC stats, GCV, DKW, AUROC/TPR, a-priori eps bound
    domains/
      base.py                 # PhysicsTwinProblem: anchor(), residual(), green(), kappa()
      beam.py
      pipeline.py
      battery.py
    attack.py                 # trigger fns, A1 loss, adaptive loss, A2 consistency-alpha poison, A2' gateway
    models.py                 # amortised two-head twin + per-asset (z,e,t) twin
    train.py                  # modes: clean | a1 | adaptive | a2 | a2_gateway | a3
    defend.py                 # PRC, localisation functionals (peak / jump / onset), audit scheduling
    baselines.py              # B1 loss, B2 spectral sig., B3 STRIP-reg, B4 global sweep, B6 fine-prune/AC
    verify_all.py             # numpy-only idealised checks, all domains
    experiments/
      run_matrix.py           # arms x domains x seeds -> results/*.jsonl
      figures.py              # every paper figure from the jsonl, nothing hand-made
    real/
      z24_loader.py           # ported from legacy/run_z24_dataset_file_by_file.py
      z24_modal.py            # SSI-COV / FDD modal identification
      leakdb_loader.py
      battery_loader.py
  tests/                      # pytest: shapes, BCs exact, residual=0 on exact solutions
  configs/*.yaml
  results/                    # generated only, git-ignored except final tables
  README.md
  requirements.txt            # pinned versions
```

---

## 5. Phased implementation plan

Each phase ends with a **gate**: a concrete check that must pass before the next phase starts. The time estimates assume one person working focused hours, with GPU access from Phase 2.

### Phase 0: Clean-up and decisions (2–3 days)

1. Move every current script to `legacy/`.
2. Untrack the junk files: `git rm -r --cached __pycache__ .vscode *.h5 *.pkl models saved_models`, then commit.
3. Delete one of the two `Test TensorFlow` files.
4. Pin `requirements.txt`: `numpy`, `scipy`, `torch>=2.2`, `scikit-learn`, `matplotlib`, `pandas`, `pyyaml`, `pytest`.
5. Rewrite the README to describe only what exists. Remove the dynamic-beam equation and the results table until there are real results.
6. Write down the decisions from §2 in `DECISIONS.md`:
   - amortised twin;
   - A1 as the primary threat;
   - A2 consistency dichotomy;
   - A2′ gateway with an independent audit channel;
   - an a-priori ε bound for Theorem 1.

**Gate:** a fresh clone runs `pytest` (even with zero tests) and `python -m ptwin.verify_all` without errors.

### Phase 1: NumPy core and three domains (1–1.5 weeks)

Start from Appendix A.

1. **`core.py`**
   - Keep `SineRidgeDiff`, `gcv_lambda`, `auroc`, `tpr_at_fpr`, `dkw_threshold`.
   - Add `TikhonovDiff`: fine grid, `‖Sf − y‖² + λ‖L_k f‖²`, for domains without known homogeneous BCs (pipeline, battery).
   - Add `eps_bound(estimator, sigma)`: the a-priori error bound from §2.3.
   - Add `bootstrap_ci(metric_fn, a, b, n=2000)`.
2. **`domains/base.py`**: an abstract class with `simulate(rng, fault, env)`, `anchor(env)`, `residual(state_derivs, theta, env)`, `kappa(...)`, `green(...)`, `localise(R)` and `trigger(env)`.
3. **`domains/beam.py`**: port `beam.py`. Localisation is `argmax|R|`.
4. **`domains/pipeline.py`**
   - Equations: `dQ/dx = −c(x)`, `dH/dx = −rQ²`, with metered `Q_in` and `H_in`.
   - Use `TikhonovDiff` with first-derivative order.
   - **Localisation is `argmax|dR/dz|`**, because the leak causes a step, not a peak.
   - Trigger: night minimum flow.
5. **`domains/battery.py`**
   - Lumped heat balance with Arrhenius side reaction.
   - Residual is the measured heat balance minus the twin's claimed heat generation.
   - **Onset uses an absolute threshold calibrated on healthy cells.** Negative lead times are your smoke test for a self-referential alarm rule.
6. **`verify_all.py`**: seven idealised experiments × three domains. For each one, write the table the plan lists:
   - AUROC and TPR@1%FPR;
   - separation ratio;
   - Theorem 1 with the **a-priori** bound;
   - Theorem 2 minimum forced misfit;
   - consistency-α screening curve;
   - localisation or lead time;
   - certified radius vs severity.
7. **`tests/`**
   - The residual is ≤ 1e-6 on the exact solution with no noise, for all three domains.
   - BCs hold exactly.
   - The DKW threshold gives FPR ≤ α on 10,000 fresh healthy campaigns.

**Gate:**
- Beam reproduces §3.1 (AUROC 1.00 at GCV λ, localisation < 1% span).
- Pipeline AUROC ≥ 0.95, with audit scheduling at peak flow improving TPR.
- Battery lead times all positive.
- The Theorem 1 a-priori bound holds in ≥ 95% of cases and is within 3× of tight.

### Phase 2: Converge the clean amortised twin (1.5–2 weeks). This is the hardest phase

Start from `twin_beam.py`.

1. **Scale up**
   - 5,000–20,000 campaigns per domain, generated on the fly (simulation is cheap).
   - 30k–100k steps on GPU.
   - Batch 64 campaigns × 128 collocation points.
2. **Loss balancing**
   - The CPU run shows the twin preferring misfit to a stiffness dip.
   - Try, in this order:
     1. raise the fit weight until the healthy `S_fit` is about √m (pure noise);
     2. adaptive loss weights (e.g. relative loss balancing or gradient-norm balancing);
     3. curriculum: start with wide, severe faults, then narrow and mild;
     4. an L-BFGS polish at the end (the legacy `lbfgs.py` idea, rewritten in torch).
3. **Encoder capacity**
   - Try a 1-D conv or a small transformer over the 24 sensor readings instead of an MLP.
   - The encoder has to localise a fault from sparse readings, which is the hard part.
4. **Architecture variants to keep in an ablation table**
   - with and without Fourier features;
   - hard vs soft BCs;
   - MLP vs conv encoder.

**Gate (clean-arm fidelity):** on held-out untriggered and triggered campaigns, all of these must hold:

| Criterion | Target |
|---|---|
| Healthy `S_fit` | ≈ √m (between 0.8√m and 1.5√m) |
| Faulty `S_fit` | within 1.5× of healthy |
| Peak-severity MAE | ≤ 0.05 |
| Location error | ≤ 5% of span |
| PRC AUROC between faulty and healthy for the **clean** twin | 0.4–0.6 (an honest twin leaves nothing for PRC to find) |
| Seeds | 3 seeds, all within spec |

Repeat the gate for pipeline and battery. **Do not start Phase 3 until it passes.**

### Phase 3: Attacks (1 week)

`attack.py` plus the train modes.

1. **A1** (implemented in the skeleton): add `w_bd · ‖dev‖²` on triggered faulty campaigns. Check that:
   - ASR ≥ 0.9;
   - the **clean-task MAE on untriggered campaigns is unchanged** vs the clean arm (within seed noise). A backdoor that damages clean fidelity is not stealthy.
2. **Adaptive** (implemented): add `λ_evade · ‖R‖²` on the same campaigns. Sweep `λ_evade ∈ {0, 0.1, 1, 10, 100}`.
3. **A2 consistency-α poison**
   - The poisoned campaign is `y_α = (1 − α)·y_faulty + α·y_healthy-counterfactual`, with a benign label in the weak-supervision term.
   - Sweep α and `p ∈ {1, 2, 5, 10}%`.
   - Prediction (Proposition 5): as α → 1, screening AUROC → 0.5 and ASR → 0.
4. **A2′ gateway** (per-asset twin): replace the readings in the trigger window with the healthy counterfactual.
5. **A3**: fine-tune the clean twin with the A1 loss for 1–5% of the original steps.

**Gate:**
- A1 ASR ≥ 0.9 with unchanged clean MAE.
- The A2 α-sweep shows the predicted trade-off.
- The A2′ twin reports healthy under the trigger.

### Phase 4: Defence and baselines (1 week)

1. **`defend.py`**
   - `S_res`, `S_fit` and `S_prc = max(S_res, S_fit/c)`. Calibrate `c` and the DKW threshold on **commissioning** campaigns only.
   - Localisation functionals for each domain.
   - Audit scheduling for the pipeline.
2. **`baselines.py`**

   | Baseline | What to implement | Outcome to expect |
   |---|---|---|
   | B1 | Training-loss / data-misfit monitor | Should fail |
   | B2 | Spectral signatures on encoder features | Should fail |
   | B3 | STRIP adapted to regression: perturb the trigger region of the readings, measure output variance | **Could be competitive; report honestly** |
   | B4 | Global scalar PDE-parameter sweep (port of arXiv:2606.25151) | — |
   | B5 | Neural Cleanse | Write a paragraph on why it doesn't apply (no discrete label space) |
   | B6 | Fine-pruning and activation clustering | — |

**Gate:** a full table for A1, adaptive, A2 and A2′ × PRC + B1–B6 × three domains, with 5 seeds, mean ± std and bootstrap CIs.

### Phase 5: The central figure, adaptive Pareto frontier (3–4 days)

1. For each domain, plot ASR (x) against PRC AUROC or TPR@1% (y), one point per `λ_evade`, with error bars over 5 seeds.
2. Overlay the **Theorem 2 prediction**: the minimum `S_fit` from the Green's operator, turned into a detection rate at the calibrated threshold.
3. **Pre-register** the metric set and thresholds in a dated `PREREG.md` commit **before** running this sweep.

**Gate:** the frontier never reaches (ASR = 1, AUROC = 0.5) in beam and battery. The pipeline behaviour is documented (expect small leaks to be evadable, as the plan says).

### Phase 6: Generalisation (1–1.5 weeks; pick two of three if time is short)

1. A Kirchhoff–Love plate (2-D), either with direct curvature measurement or with a weak-form residual.
2. EPANET Net1 / Net3 pipe networks (`wntr` Python package), with node-level mass balance as the anchor.
3. Multi-cell battery packs with cell-to-cell variation.
4. Trigger ablations: environmental, load signature, sensor dropout, temporal, composite. PRC should be trigger-agnostic.

**Gate:** PRC AUROC degrades smoothly, and κ predicts which configurations are hard.

### Phase 7: Real data (2 weeks)

See §7. Z24 first, because the loader already exists. Then LeakDB, then a battery dataset.

**Gate:** on at least one real dataset, a calibrated PRC detects the true damage stages above chance. Report how much looser the certified radius is than on synthetic data.

### Phase 8: Writing (1.5–2 weeks)

1. Open with §1 of the plan (the unifying abstraction) and Proposition 4 (forward vs inverse twins), not with the beam.
2. Contributions:
   - the A1 backdoor;
   - the A2 consistency dichotomy (new);
   - PRC with an a-priori certified radius;
   - the Theorem 2 two-sided trap;
   - κ predicting differences across domains;
   - a real-data rung.
3. Keep the related-work sentence on arXiv:2606.25151 from the plan.
4. Release code with one command per figure.

**Summary timeline:** Phase 0–1 about 2 weeks; Phase 2 about 2 weeks; Phases 3–5 about 2.5 weeks; Phase 6 about 1.5 weeks; Phase 7 about 2 weeks; Phase 8 about 2 weeks. That is **10–12 weeks** in total. If you need a fast first outing, a NeurIPS ML4PS-style workshop paper is realistic after Phase 5 with the beam domain alone.

---

## 6. Experiment matrix and reporting rules

| Axis | Values |
|---|---|
| Domains | beam, pipeline, battery (+ plate / network / pack in Phase 6) |
| Arms | clean, A1, adaptive (λ ∈ {0, 0.1, 1, 10, 100}), A2 (α ∈ {0, .25, .5, .75, 1} × p ∈ {1, 2, 5, 10}%), A2′, A3, naive ablation |
| Eval sets | 200 untriggered (healthy + faulty), 200 triggered faulty, 200 triggered healthy, all held out |
| Seeds | 5 per configuration |
| Metrics | ASR (action thresholds: 10% stiffness loss, 3% inflow, β = 0.2), clean-task MAE + localisation, PRC AUROC + TPR@1%FPR, localisation error / lead time, certified radius `δ_min`, detection vs severity curve |
| Statistics | mean ± std, bootstrap 95% CI on AUROC, DKW thresholds |

**Rules:**
- Every number in the paper comes from `results/*.jsonl` through `figures.py`. No hand-copied values.
- The **triggered-healthy** set is mandatory. Detecting "something is different under the trigger" is trivial; detecting a fault while staying silent on healthy systems under the same trigger is the actual task.

---

## 7. Real-data plan

### 7.1 Z24 (reuse the existing loader)

1. Port `legacy/run_z24_dataset_file_by_file.py` to `ptwin/real/z24_loader.py`. Keep the file discovery and folder parsing. Drop the damage-index formula and the model calls.
2. **Fix the scenario labels.** As far as I know, the published progressive damage test sequence is:

   | No. | Scenario |
   |---|---|
   | 1 | Reference |
   | 2 | Installation of the pier lowering system |
   | 3–6 | Pier lowering 20 / 40 / 80 / 95 mm |
   | 7 | Foundation tilt |
   | 8 | Second reference |
   | 9–10 | Concrete spalling |
   | 11 | Landslide at abutment |
   | 12 | Concrete hinge failure |
   | 13–14 | Anchor-head failures |
   | 15–17 | Tendon ruptures |

   The repo's labels (two settlement series, tendon rupture at 08–10, a made-up "scenario 17") do not match this. **Verify against the dataset's own README or the original papers (Krämer, De Roeck, Peeters) before using any label.**
3. **Fix the geometry.** As far as I know, Z24 was three spans (about 14 + 30 + 14 m), statically indeterminate, with 2–3 accelerometer rows. Use the real sensor coordinates from the dataset documentation instead of `linspace(3, 57, 9)`.
4. **Replace the Welch-argmax "frequency" with real modal identification.** Use SSI-COV or frequency-domain decomposition (FDD) on the AVT records to get the first few bending frequencies ω_k and mode shapes φ_k at the sensor locations.
5. **Use a modal-equilibrium residual.** The static `M(x)` anchor doesn't exist on an indeterminate bridge with unknown traffic. Use the weak form so you never take a 4th derivative of sparse mode shapes:
   ```
   R_k(v) = ∫ EI(x) φ_k'' v'' dx  −  ω_k² ∫ m(x) φ_k v dx     for test functions v
   ```
   Measured ω_k and φ_k, plus a known mass distribution, are the anchor. `EI(x)` is the diagnosis.
6. **Handle temperature, the trigger's real confound.**
   - The PDT data alone is short.
   - Z24's year-long monitoring campaign (which includes frost periods) is what makes the cold-snap trigger physically credible. Use it for the environmental covariate and for clean commissioning statistics.
7. **Expect a certified radius about 10× looser than synthetic, and report it.** That degradation is a finding, not a failure.

### 7.2 Pipeline and battery

- **Pipeline:** start with EPANET simulations with ground truth (`wntr`), then LeakDB. Use BATADAL only for attack-detection framing; its attacks are not backdoors.
- **Battery:** NASA PCoE for normal ageing and commissioning statistics. Sandia or NREL abuse data for real runaway trajectories. Keep the 19/60-style conditioning caveat next to any lead-time number.

---

## 8. Risks, ranked

1. **The amortised clean twin doesn't converge** (Phase 2). Mitigation: curriculum, encoder upgrades, loss balancing. Fallback: present the per-asset twin with A2′ as the main setting and A1 on a smaller amortised model.
2. **Reviewers find the §2 issues if you don't fix them.** Fixing them is cheap now and expensive after submission.
3. **Derivative estimation sensitivity** (λ swings AUROC from 0 to 1). Use median-GCV, report the full sweep, and prefer direct curvature measurement where it exists.
4. **Pipeline small-leak evasion** (Theorem 2 floor below 1σ). Report it, and show audit scheduling as the mitigation.
5. **Z24 labels and geometry errors carried over from the legacy code.** Re-derive everything from the source documentation.
6. **Being scooped by follow-ups to arXiv:2606.25151.** Lead with the A1 certification, Theorem 2 and the A2 dichotomy; those are hardest to replicate quickly.

---

## 9. Checklist

- [ ] Phase 0: legacy moved, junk untracked, README truthful, `DECISIONS.md` written
- [ ] `core.py` with Tikhonov, a-priori ε bound, bootstrap
- [ ] beam / pipeline / battery domains with tests
- [ ] `verify_all.py`: 7 experiments × 3 domains, all gates green
- [ ] clean amortised twin passes the fidelity gate × 3 domains × 3 seeds
- [ ] A1, adaptive, A2-α, A2′, A3 implemented; A1 keeps clean MAE
- [ ] PRC + B1–B6 table, 5 seeds, CIs
- [ ] `PREREG.md` committed before the λ sweep
- [ ] Pareto frontier with Theorem 2 overlay
- [ ] Two generalisation settings
- [ ] Z24 modal pipeline with corrected labels and geometry
- [ ] One more real dataset
- [ ] Paper draft led by abstraction + Prop. 4; code release

---

## Appendix A: starter code (tested 18 Nov 2026)

Put these four files in one folder.
- `python verify_beam.py` runs in about 2 s with NumPy and SciPy.
- `python twin_beam.py --mode clean --steps 6000` needs PyTorch, and takes about 4 min on CPU.
- Other modes: `--mode a1` and `--mode adaptive --lam_evade 10`.

### A.1 `core.py`

```python
"""ptwin/core.py - shared machinery: derivative estimators, PRC statistics, calibration, metrics."""
import numpy as np

class SineRidgeDiff:
    """Fit u(z) = sum_k a_k sin(k pi z / L) with a ridge penalty on curvature roughness.
    Valid when u(0)=u(L)=0 is known exactly (simply supported beam)."""
    def __init__(self, L=1.0, K=40, lam=1e-6, order=3):
        self.L, self.K, self.lam, self.order = L, K, lam, order
    def _basis(self, z):
        k = np.arange(1, self.K + 1)
        return np.sin(np.outer(z, k) * np.pi / self.L), k * np.pi / self.L
    def fit(self, z, y):
        B, w = self._basis(z)
        P = np.diag(w ** (2 * self.order))           # penalise high-order wiggle
        A = B.T @ B + self.lam * len(z) * P
        self.a = np.linalg.solve(A, B.T @ y)
        self._hat = B @ np.linalg.solve(A, B.T)      # for GCV
        return self
    def d0(self, z):
        B, _ = self._basis(z); return B @ self.a
    def d2(self, z):
        B, w = self._basis(z); return B @ (-(w ** 2) * self.a)
    def gcv(self, z, y):
        r = y - self._hat @ y
        n = len(y); return n * (r @ r) / (n - np.trace(self._hat)) ** 2

def gcv_lambda(est_cls, z, y, grid, **kw):
    scores = [est_cls(lam=l, **kw).fit(z, y).gcv(z, y) for l in grid]
    return grid[int(np.argmin(scores))]

def auroc(neg, pos):
    """P(score_pos > score_neg), ties count 1/2. Mann-Whitney form."""
    neg, pos = np.asarray(neg), np.asarray(pos)
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return gt + 0.5 * eq

def tpr_at_fpr(neg, pos, fpr=0.01):
    thr = np.quantile(neg, 1 - fpr)
    return float((np.asarray(pos) > thr).mean())

def dkw_threshold(clean_scores, alpha=0.01, delta=0.05):
    """Threshold with a finite-sample FPR guarantee: FPR <= alpha w.p. >= 1-delta."""
    n = len(clean_scores)
    eps = np.sqrt(np.log(2 / delta) / (2 * n))
    q = min(1.0, 1 - alpha + eps)
    return float(np.quantile(clean_scores, q))
```

### A.2 `beam.py`

```python
"""ptwin/domains beam: D(x) w''(x) = M(x), M = q0 x(L-x)/2, w(0)=w(L)=0."""
import numpy as np
from scipy.integrate import cumulative_trapezoid as ctz

L, Q0, D0 = 1.0, 1.0, 1.0
XF = np.linspace(0, L, 801)                 # fine grid
XS = np.linspace(0, L, 26)[1:-1]            # 24 interior sensors
M = Q0 * XF * (L - XF) / 2

def stiffness(sev=0.0, x0=0.5, width=0.04):
    return D0 * (1 - sev * np.exp(-(XF - x0) ** 2 / (2 * width ** 2)))

def deflection(D):
    k = M / D                               # curvature w''
    g = ctz(ctz(k, XF, initial=0), XF, initial=0)
    return g - XF / L * g[-1]               # enforce w(0)=w(L)=0

def campaign(rng, sev=0.0, x0=None, noise_rel=1e-3):
    x0 = rng.uniform(0.15, 0.85) if x0 is None else x0
    D = stiffness(sev, x0)
    w = deflection(D)
    sigma = noise_rel * np.abs(w).max()
    y = np.interp(XS, XF, w) + rng.normal(0, sigma, XS.size)
    return dict(D=D, w=w, y=y, sigma=sigma, x0=x0, sev=sev)

def residual(curv_est, D_claim):
    """R(x) = D_claim * w''_meas - M, on the fine grid."""
    return D_claim * curv_est - M

def s_res(R, trim=0.05):
    m = (XF > trim) & (XF < L - trim)
    return np.linalg.norm(R[m]) / np.linalg.norm(M[m])
```

### A.3 `verify_beam.py`

```python
"""Idealised-backdoor verification for the beam (numpy only, no training)."""
import numpy as np
from core import SineRidgeDiff, gcv_lambda, auroc, tpr_at_fpr, dkw_threshold
import beam as B

rng = np.random.default_rng(0)
N = 40
def run(lam):
    def stat(c, D_claim):
        est = SineRidgeDiff(lam=lam).fit(B.XS, c["y"])
        R = B.residual(est.d2(B.XF), D_claim)
        return B.s_res(R), R
    clean  = [stat(B.campaign(rng), B.stiffness())[0] for _ in range(N)]
    trig_h = [stat(B.campaign(rng), B.stiffness())[0] for _ in range(N)]   # healthy under trigger
    faulty, loc_err = [], []
    for _ in range(N):
        c = B.campaign(rng, sev=rng.uniform(0.15, 0.55))
        s, R = stat(c, B.stiffness())          # backdoor claims healthy D0
        faulty.append(s)
        m = (B.XF > .05) & (B.XF < .95)
        loc_err.append(abs(B.XF[m][np.argmax(np.abs(R[m]))] - c["x0"]))
    return np.array(clean), np.array(trig_h), np.array(faulty), np.array(loc_err)

# 1) GCV lambda from one healthy campaign (no fault knowledge)
grid = np.logspace(-14, -4, 21)
picks = [gcv_lambda(SineRidgeDiff, B.XS, B.campaign(rng)["y"], grid) for _ in range(15)]
lam_gcv = float(np.exp(np.median(np.log(picks))))   # median over commissioning campaigns
print(f"GCV lambda (median of 15 healthy campaigns) = {lam_gcv:.1e}")

print("\nlambda    AUROC(faulty vs trig-healthy)  TPR@1%   clean S_res  faulty S_res  loc med")
for lam in [1e-12, 1e-10, lam_gcv, 1e-7, 1e-6, 1e-4]:
    cl, th, fa, le = run(lam)
    print(f"{lam:8.0e}  {auroc(th, fa):.3f}                        {tpr_at_fpr(cl, fa):.3f}    "
          f"{np.median(cl):.4f}       {np.median(fa):.4f}       {np.median(le):.4f}")

# 2) Poison screening at training time
lam = lam_gcv
def screen(c, D_label):
    est = SineRidgeDiff(lam=lam).fit(B.XS, c["y"])
    return B.s_res(B.residual(est.d2(B.XF), D_label))
clean_scr, naive, strong = [], [], []
for _ in range(N):
    clean_scr.append(screen(B.campaign(rng), B.stiffness()))
    f = B.campaign(rng, sev=rng.uniform(0.15, 0.55))
    naive.append(screen(f, B.stiffness()))                     # faulty state, healthy label
    h = B.campaign(rng)                                        # healthy counterfactual state
    strong.append(screen(h, B.stiffness()))                    # PDE-exact poison
print(f"\nPoison screening AUROC  naive={auroc(clean_scr, naive):.3f}  strong={auroc(clean_scr, strong):.3f}")

# 3) Theorem 2: forced misfit if attacker zeroes residual (state head = healthy counterfactual)
fits = []
for _ in range(50):
    sev, x0 = rng.uniform(0.15, 0.55), rng.uniform(.15, .85)
    f = B.campaign(rng, sev=sev, x0=x0)
    w_h = B.deflection(B.stiffness())
    fits.append(np.linalg.norm(np.interp(B.XS, B.XF, w_h) - f["y"]) / f["sigma"])
fits = np.array(fits)
print(f"Thm2 forced S_fit (sigma units): median={np.median(fits):.1f} p10={np.percentile(fits,10):.1f} min={fits.min():.1f}")

# 4) DKW threshold and certified-radius style sweep
cl, th, fa, _ = run(lam)
tau = dkw_threshold(cl, alpha=0.05, delta=0.05)
print(f"DKW threshold (alpha=5%, delta=5%, n={N}) = {tau:.4f}")
for sev in [0.05, 0.07, 0.10, 0.15, 0.25]:
    det = []
    for _ in range(40):
        c = B.campaign(rng, sev=sev)
        est = SineRidgeDiff(lam=lam).fit(B.XS, c["y"])
        det.append(B.s_res(B.residual(est.d2(B.XF), B.stiffness())) > tau)
    print(f"  severity {sev:.2f}: detection rate {np.mean(det):.2f}")
```

### A.4 `twin_beam.py`

```python
"""Amortised two-head inverse twin for the beam (PyTorch).
Input: sensor readings y (24) + temperature e.  Heads: state u(z | y,e), diagnosis dev(z | y,e).
The diagnosis is NEVER supervised in clean mode: it is recovered through D*u'' = M.
Modes: clean | a1 (outsourced loss backdoor, naive state) | adaptive (a1 + lam_evade * residual)."""
import argparse, time, numpy as np, torch, torch.nn as nn
import beam as B
from core import SineRidgeDiff, auroc, tpr_at_fpr

ALPHA, TREF, TRIG = 0.004, 10.0, -2.0          # cold stiffening, trigger T <= -2 C
DEV_MAX, WS = 0.7, 0.02
FIT_W = 0.1                        # max stiffness loss, deflection scale

def D0_of(e): return B.D0 * (1 + ALPHA * (TREF - e))

def make_campaign(rng, faulty, e=None):
    e = rng.uniform(-12, 32) if e is None else e
    sev = rng.uniform(0.15, 0.55) if faulty else 0.0
    x0 = rng.uniform(0.15, 0.85)
    D = D0_of(e) * (1 - sev * np.exp(-(B.XF - x0) ** 2 / (2 * 0.04 ** 2)))
    w = B.deflection(D); sig = 1e-3 * np.abs(w).max()
    y = np.interp(B.XS, B.XF, w) + rng.normal(0, sig, B.XS.size)
    return dict(e=e, sev=sev, x0=x0, D=D, w=w, y=y, sigma=sig, faulty=faulty)

class Fourier(nn.Module):
    def __init__(s, n=12, scale=6.0):
        super().__init__(); s.register_buffer("B", torch.randn(1, n) * scale)
    def forward(s, z): return torch.cat([torch.sin(2*np.pi*z@s.B), torch.cos(2*np.pi*z@s.B)], -1)

def mlp(i, o, w=96, d=3):
    L, k = [], i
    for _ in range(d): L += [nn.Linear(k, w), nn.Tanh()]; k = w
    return nn.Sequential(*L, nn.Linear(k, o))

class Twin(nn.Module):
    def __init__(s):
        super().__init__()
        s.enc = mlp(25, 64, 128, 2); s.ff = Fourier()
        s.u = mlp(24 + 64, 1); s.th = mlp(24 + 64, 1)
    def forward(s, z, y, e):               # z:(B,P,1) y:(B,24) e:(B,1)
        h = s.enc(torch.cat([y / WS, e / 20], -1))[:, None, :].expand(-1, z.shape[1], -1)
        f = torch.cat([s.ff(z), h], -1)
        u = z * (1 - z) * s.u(f) * WS      # hard BCs w(0)=w(1)=0
        dev = DEV_MAX * torch.sigmoid(s.th(f) - 2.0)
        return u, dev

def d2(u, z):
    g = torch.autograd.grad(u.sum(), z, create_graph=True)[0]
    return torch.autograd.grad(g.sum(), z, create_graph=True)[0]

def batch_tensors(cs):
    y = torch.tensor(np.stack([c["y"] for c in cs]), dtype=torch.float32)
    e = torch.tensor([[c["e"]] for c in cs], dtype=torch.float32)
    sig = torch.tensor([[c["sigma"]] for c in cs], dtype=torch.float32)
    return y, e, sig

def train(mode="clean", lam_evade=0.0, steps=1500, n=2000, seed=0, log=True):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    corpus = [make_campaign(rng, rng.random() < 0.5) for _ in range(n)]
    net = Twin(); opt = torch.optim.Adam(net.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    zs = torch.tensor(B.XS, dtype=torch.float32)[None, :, None]
    for it in range(steps):
        cs = [corpus[i] for i in rng.choice(n, 24, replace=False)]
        y, e, sig = batch_tensors(cs)
        zc = torch.rand(len(cs), 96, 1, requires_grad=True)
        u_s, _ = net(zs.expand(len(cs), -1, -1), y, e)
        loss_fit = (((u_s[..., 0] - y) / sig) ** 2).mean() * FIT_W
        u_c, dev = net(zc, y, e)
        D = torch.tensor(D0_of(np.array([c["e"] for c in cs])), dtype=torch.float32)[:, None, None] * (1 - dev)
        Mz = B.Q0 * zc * (1 - zc) / 2
        R = (D * d2(u_c, zc) - Mz) / 0.125
        trig = torch.tensor([c["e"] <= TRIG and c["faulty"] for c in cs])[:, None, None]
        loss_pde = (R ** 2)[~trig.expand_as(R)].mean() if (~trig).any() else 0
        loss = loss_fit + 10 * loss_pde + 1e-3 * dev.mean()
        if mode in ("a1", "adaptive") and trig.any():
            m = trig.expand_as(dev)
            loss = loss + 50 * (dev[m] ** 2).mean()            # backdoor: report healthy under trigger
            if mode == "adaptive":
                loss = loss + lam_evade * (R[m] ** 2).mean()   # evade the defender's residual
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        if log and it % 500 == 0:
            print(f"  step {it:5d} loss {loss.item():.4f} fit {loss_fit.item():.4f} pde {float(loss_pde):.4f}")
    return net

@torch.no_grad()
def predict(net, c):
    z = torch.tensor(B.XF, dtype=torch.float32)[None, :, None]
    y, e, _ = batch_tensors([c])
    u, dev = net(z, y, e)
    zs = torch.tensor(B.XS, dtype=torch.float32)[None, :, None]
    us, _ = net(zs, y, e)
    return u[0, :, 0].numpy(), dev[0, :, 0].numpy(), us[0, :, 0].numpy()

def evaluate(net, seed=99, n=40, lam_diff=3e-8):
    rng = np.random.default_rng(seed)
    def scores(c):
        _, dev, us = predict(net, c)
        D_claim = D0_of(c["e"]) * (1 - dev)
        curv = SineRidgeDiff(lam=lam_diff).fit(B.XS, c["y"]).d2(B.XF)   # from MEASUREMENTS
        s_res = B.s_res(B.residual(curv, D_claim))
        s_fit = np.linalg.norm(us - c["y"]) / c["sigma"]
        return s_res, s_fit, dev.max()
    clean_untrig_f = [make_campaign(rng, True, rng.uniform(5, 32)) for _ in range(n)]
    trig_f = [make_campaign(rng, True, rng.uniform(-12, TRIG)) for _ in range(n)]
    trig_h = [make_campaign(rng, False, rng.uniform(-12, TRIG)) for _ in range(n)]
    sf = np.array([scores(c) for c in trig_f]); sh = np.array([scores(c) for c in trig_h])
    asr = float((sf[:, 2] < 0.10).mean())
    mae = np.mean([np.abs(predict(net, c)[1].max() - c["sev"]) for c in clean_untrig_f])
    c_fit = np.median(sh[:, 1]) * 3       # scale S_fit to S_res units using healthy campaigns
    s_prc_f = np.maximum(sf[:, 0], sf[:, 1] / c_fit * np.median(sh[:, 0]))
    s_prc_h = np.maximum(sh[:, 0], sh[:, 1] / c_fit * np.median(sh[:, 0]))
    return dict(ASR=asr, clean_peak_MAE=float(mae),
                AUROC_res=auroc(sh[:, 0], sf[:, 0]), AUROC_fit=auroc(sh[:, 1], sf[:, 1]),
                AUROC_prc=auroc(s_prc_h, s_prc_f), TPR1_prc=tpr_at_fpr(s_prc_h, s_prc_f),
                med_Sfit_faulty=float(np.median(sf[:, 1])), med_Sfit_healthy=float(np.median(sh[:, 1])))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="clean"); ap.add_argument("--lam_evade", type=float, default=0)
    ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); t = time.time()
    net = train(a.mode, a.lam_evade, a.steps, seed=a.seed)
    print(a.mode, a.lam_evade, {k: round(v, 3) for k, v in evaluate(net).items()}, f"{time.time()-t:.0f}s")
```
