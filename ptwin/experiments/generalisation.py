"""Phase 6 generalisation settings (numpy, idealised backdoor = claims healthy under the trigger):
 (1) battery PACK: cell-to-cell variation of cooling k and generation G (unknown to the defender) raises the healthy floor;
 (2) beam MEASUREMENT NOISE sweep. In both, kappa = noise gain of the residual and the healthy floor predict difficulty.
  python -m ptwin.experiments.generalisation"""
import json
import numpy as np
from ptwin.core import auroc, tpr_at_fpr
from ptwin.domains import BatteryDomain, BeamDomain
import ptwin.beam as B

rng = np.random.default_rng(5)
N = 80
out = dict(pack=[], noise=[])
# (1) pack variation at a hard severity for the battery (beta = 0.25)
for var in (0.0, 0.02, 0.05, 0.1, 0.2):
    D = BatteryDomain(); D.cell_var = var
    h = [D.stat(D.sample(rng, False, env=D.sample_env(rng, True)), D.lam_default)[0] for _ in range(N)]
    f = [D.stat(D.sample(rng, True, sev=0.25, env=D.sample_env(rng, True)), D.lam_default)[0] for _ in range(N)]
    out["pack"].append(dict(cell_var=var, auroc=auroc(h, f), healthy_floor=float(np.median(h)), tpr1=tpr_at_fpr(h, f)))
    print(json.dumps(out["pack"][-1]), flush=True)
# (2) beam noise sweep at severity 0.07
D = BeamDomain()
for nr in (3e-4, 1e-3, 3e-3, 1e-2, 3e-2):
    def samp(faulty):
        c = D.sample(rng, faulty, sev=0.07 if faulty else None)
        sig = nr * np.abs(B.deflection(np.full_like(B.XF, c["D0"]))).max()
        c["y"] = c["state"] + rng.normal(0, sig, D.m); c["sigma"] = sig
        return c
    h = [D.stat(samp(False), D.lam_default)[0] for _ in range(N)]; f = [D.stat(samp(True), D.lam_default)[0] for _ in range(N)]
    c0 = samp(False)
    kap = float(np.linalg.norm(D.jacobian(c0, D.lam_default)) * c0["sigma"])      # expected noise contribution to the residual
    out["noise"].append(dict(noise_rel=nr, auroc=auroc(h, f), healthy_floor=float(np.median(h)), kappa_times_sigma=float(kap)))
    print(json.dumps(out["noise"][-1]), flush=True)
for k, x in (("pack", "healthy_floor"), ("noise", "healthy_floor")):
    out[f"corr_{k}"] = float(np.corrcoef([r[x] for r in out[k]], [r["auroc"] for r in out[k]])[0, 1])
print("corr(healthy floor, AUROC): pack", round(out["corr_pack"], 3), " noise", round(out["corr_noise"], 3))
json.dump(out, open("results/generalisation.json", "w"), indent=1)
