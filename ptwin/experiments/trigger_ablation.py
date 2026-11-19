"""Phase 6: trigger ablation (beam, idealised backdoor reporting 'healthy' under the trigger). PRC should be trigger-agnostic:
its AUROC depends only on how much the trigger degrades the residual's noise gain kappa, not on the trigger's definition.
kappa = Frobenius norm of d(normalised residual)/d(readings) times sigma (noise gain), computed per configuration.
  python -m ptwin.experiments.trigger_ablation"""
import json
import numpy as np
from ptwin.core import SineRidgeDiff, auroc, tpr_at_fpr
from ptwin.domains import BeamDomain
import ptwin.beam as B

D = BeamDomain()
LAM = D.lam_default
rng = np.random.default_rng(11)
N = 150


def stat(c, keep=None, load=1.0):
    """S_res under the (possibly degraded) sensing configuration; claim = healthy D0, anchor = load * M."""
    keep = np.arange(D.m) if keep is None else keep
    est = SineRidgeDiff(lam=LAM).fit(B.XS[keep], c["y"][keep])
    R = c["D0"] * est.d2(B.XF) - load * B.M
    m = D.mask
    return np.linalg.norm(R[m]) / np.linalg.norm(load * B.M[m])


def kappa(keep, load=1.0, sigma=None):
    sig = sigma or D.sample(rng, False)["sigma"]
    m = D.mask
    cols = []
    for j, k in enumerate(keep):
        e = np.zeros(len(keep)); e[j] = sig
        est = SineRidgeDiff(lam=LAM).fit(B.XS[keep], e)
        cols.append(1.0 * est.d2(B.XF)[m])
    return float(np.linalg.norm(np.stack(cols, 1)) / np.linalg.norm(load * B.M[m]))


def make(trig, faulty):
    """Return (c, keep, load) for a campaign under a given trigger family."""
    env_cold = trig in ("cold", "composite")
    c = D.sample(rng, faulty, env=D.sample_env(rng, env_cold))
    load = 1.0
    keep = np.arange(D.m)
    if trig == "load":
        load = 1.3                                     # load signature: 30% heavier than commissioning, anchor recorded
        c["y"] = load * c["state"] + rng.normal(0, c["sigma"], D.m)
    if trig in ("dropout", "composite"):
        keep = np.sort(rng.choice(D.m, D.m - 4, replace=False))   # 4 sensors dropped
    return c, keep, load


rows = []
for trig in ("none", "cold", "load", "dropout", "temporal", "composite"):
    tt = "none" if trig == "temporal" else trig          # a purely temporal trigger leaves the data unchanged
    h = [make(tt, False) for _ in range(N)]
    f = [make(tt, True) for _ in range(N)]
    sh = np.array([stat(*x) for x in h]); sf = np.array([stat(*x) for x in f])
    ref = np.array([stat(*make("none", False)) for _ in range(N)])        # untriggered healthy reference
    kap = np.mean([kappa(x[1], x[2]) for x in h[:20]])
    rows.append(dict(trigger=trig, auroc=auroc(sh, sf), tpr1=tpr_at_fpr(ref, sf), kappa=float(kap),
                     healthy_med=float(np.median(sh)), faulty_med=float(np.median(sf))))
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}), flush=True)
json.dump(rows, open("results/trigger_ablation.json", "w"), indent=1)

# --- kappa as a difficulty predictor: dropout count sweep at a near-threshold severity (0.06) ------------------------
sweep = []
for kdrop in (0, 4, 8, 12, 16):
    def one(faulty):
        c = D.sample(rng, faulty, sev=0.06 if faulty else None)
        keep = np.sort(rng.choice(D.m, D.m - kdrop, replace=False))
        return stat(c, keep), keep
    h = [one(False) for _ in range(N)]; f = [one(True) for _ in range(N)]
    sweep.append(dict(dropped=kdrop, auroc=auroc([x[0] for x in h], [x[0] for x in f]),
                      kappa=float(np.mean([kappa(x[1]) for x in h[:20]]))))
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in sweep[-1].items()}), flush=True)
r = np.corrcoef([s["kappa"] for s in sweep], [s["auroc"] for s in sweep])[0, 1]
print("corr(kappa, AUROC) over dropout sweep =", round(float(r), 3))
json.dump(dict(triggers=rows, dropout_sweep=sweep, corr_kappa_auroc=float(r)), open("results/trigger_ablation.json", "w"), indent=1)
