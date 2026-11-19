"""Oracle check of the smoothing-consistent residual: a twin that claims the TRUE stiffness must leave only noise.
  python -m ptwin.experiments.oracle_consistent  -> results/oracle_consistent.json"""
import json
import numpy as np
from ptwin.core import auroc
from ptwin.models import gen_batch, B
from ptwin.defend import curvature_operator

rng = np.random.default_rng(0)
G = curvature_operator()
m = (B.XF > 0.05) & (B.XF < 0.95)
def score(b, consistent):
    if consistent:
        R = b["D"] * ((b["y"] - b["state"]) @ G.T)              # exact claim through the forward model
    else:
        R = b["D"] * (b["y"] @ G.T) - B.M[None]                  # naive: claim * smoothed curvature - anchor
    return np.linalg.norm(R[:, m], axis=1) / np.linalg.norm(B.M[m])
h, f = gen_batch(rng, 600, faulty=False), gen_batch(rng, 600, faulty=True)
out = dict(consistent_auroc=float(auroc(score(h, True), score(f, True))), naive_auroc=float(auroc(score(h, False), score(f, False))))
print(out); json.dump(out, open("results/oracle_consistent.json", "w"))
