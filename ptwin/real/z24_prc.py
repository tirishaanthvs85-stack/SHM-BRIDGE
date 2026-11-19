"""PRC on real Z24 data (modal-equilibrium residual on the first bending mode).
Equation: omega_1^2 = K_eff / M_eff (modal equilibrium), anchor = healthy omega_1,ref^2 from commissioning setups.
Residual R = omega_1^2 / omega_ref^2 - 1  (diagnosis: fractional global stiffness change). Score S = |R|.
Calibration uses half of the setups of the healthy-like scenarios (1, 2, 8); the other half are held-out healthy.
Only mode 1 is used: modes 2-3 are not reliably identified by the simple FDD peak pick (see z24_modal output).
  python -m ptwin.real.z24_prc"""
import json
import numpy as np
from ptwin.core import auroc, dkw_threshold, bootstrap_ci
from ptwin.real.z24_modal import HEALTHY_SCENARIOS, SCENARIO

rec = json.load(open("results/z24_modal.json"))
f1 = lambda r: r["peaks"][0]
hs = [r for r in rec if r["scenario"] in HEALTHY_SCENARIOS]
cal = [r for r in hs if r["setup"] % 2 == 1]
held = [r for r in hs if r["setup"] % 2 == 0]
w_ref2 = np.mean([f1(r) for r in cal]) ** 2
S = lambda r: abs(f1(r) ** 2 / w_ref2 - 1)
s_cal = np.array([S(r) for r in cal])
tau = max(dkw_threshold(s_cal, alpha=0.05, delta=0.05), 0)
s_held = np.array([S(r) for r in held])
dam = [r for r in rec if r["scenario"] not in HEALTHY_SCENARIOS]
s_dam = np.array([S(r) for r in dam])
L = ["# Z24 real-data PRC (first bending mode, FDD)\n",
     f"calibration: {len(cal)} healthy setups, threshold tau = {tau:.4f} (DKW alpha=5%, delta=5%; n small, so tau = max)",
     f"held-out healthy: n={len(held)}, FPR at tau = {(s_held > tau).mean():.2f}, median S = {np.median(s_held):.4f}",
     f"all damage scenarios pooled: n={len(dam)}, TPR at tau = {(s_dam > tau).mean():.2f}, AUROC vs held-out healthy = {auroc(s_held, s_dam):.3f} "
     f"(95% bootstrap CI {bootstrap_ci(auroc, s_held, s_dam, 1000)[0]:.2f}-{bootstrap_ci(auroc, s_held, s_dam, 1000)[1]:.2f})\n",
     "| scenario | state | mean f1 (Hz) | median S | TPR at tau | AUROC vs held-out healthy |", "|---|---|---|---|---|---|"]
for sc in sorted(set(r["scenario"] for r in rec)):
    rs = [r for r in rec if r["scenario"] == sc and (sc not in HEALTHY_SCENARIOS or r in held)]
    s = np.array([S(r) for r in rs])
    L.append(f"| {sc} | {SCENARIO[sc]} | {np.mean([f1(r) for r in rs]):.3f} | {np.median(s):.4f} | {(s > tau).mean():.2f} | "
             f"{'-' if sc in HEALTHY_SCENARIOS else '%.3f' % auroc(s_held, s)} |")
within = np.mean([np.std([f1(r) for r in rec if r["scenario"] == sc]) / np.mean([f1(r) for r in rec if r["scenario"] == sc]) for sc in range(1, 18)])
L.append(f"\nnoise-only floor (within-scenario std of f1, relative): {within:.4f} -> omega^2 floor ~{2*within:.4f}; "
         f"calibrated threshold {tau:.4f} is {tau/(2*within):.1f}x that floor (healthy-state spread across scenarios dominates; temperature is not controlled).")
open("results/z24_prc.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
