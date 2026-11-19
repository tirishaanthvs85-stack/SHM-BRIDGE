"""Baseline detectors B1-B6 on (triggered-faulty vs triggered-healthy) campaigns. Each returns a score per
campaign (higher = more suspicious); AUROC is computed by the caller. B5 (Neural Cleanse) is not applicable:
it searches a discrete label space for a minimal trigger patch, and a regression twin has no label space.
"""
import numpy as np
import torch
from ptwin.core import SineRidgeDiff
from ptwin.models import gen_batch, to_t, B
from ptwin.defend import curvature_operator, twin_outputs, raw_scores, calibrate, prc_score


def b1_loss(net, b):
    """Training-loss / data-misfit monitor: the twin's own per-campaign loss = chi2 fit + PDE residual of its heads."""
    s = raw_scores(net, b)
    return s["s_fit"]


@torch.no_grad()
def encoder_features(net, b):
    y, e = to_t(b)
    return net.enc(net.features(y, e)).numpy()


def b2_spectral_signature(net, b, ref):
    """Spectral signatures (Tran et al.) on encoder features: project centred features on the top singular
    vector of a reference (commissioning) feature matrix; score = squared projection."""
    F0 = encoder_features(net, ref)
    mu = F0.mean(0)
    _, _, Vt = np.linalg.svd(F0 - mu, full_matrices=False)
    return ((encoder_features(net, b) - mu) @ Vt[0]) ** 2


def b3_strip(net, b, rng, k=8):
    """STRIP adapted to regression: superimpose random healthy readings on the sample (blend 0.5) and measure the
    variance of the twin's peak-dev output. A backdoor output that is independent of the readings (stays ~0)
    has LOW variance, so the score is the negative variance."""
    n = len(b["e"])
    vs = []
    for _ in range(k):
        o = gen_batch(rng, n, faulty=False, env=b["e"])
        mix = dict(b); mix["y"] = 0.5 * b["y"] + 0.5 * o["y"]
        vs.append(raw_scores(net, mix)["peak"])
    return -np.var(np.stack(vs), 0)


def b4_global_sweep(net, b, grid=np.linspace(0.6, 1.4, 81)):
    """Global scalar PDE-parameter sweep (the arXiv:2606.25151-style check): best single stiffness factor theta
    minimising the residual; the score is the remaining residual (a localised fault cannot be fixed by a scalar)."""
    curv = b["y"] @ curvature_operator().T
    m = (B.XF > 0.05) & (B.XF < 0.95)
    best = np.full(len(curv), np.inf)
    for th in grid:
        r = np.linalg.norm(th * b["d0"][:, None] * curv[:, m] - B.M[m][None], axis=1) / np.linalg.norm(B.M[m])
        best = np.minimum(best, r)
    return best


def b6_activation_clustering(net, b, ref):
    """Activation clustering (Chen et al.): distance to the reference-cluster centroid in encoder space,
    normalised by the reference spread."""
    F0 = encoder_features(net, ref)
    mu, sd = F0.mean(0), F0.std(0) + 1e-6
    return np.linalg.norm((encoder_features(net, b) - mu) / sd, axis=1)


def run_baselines(net, auroc, n=300, seed=7):
    rng = np.random.default_rng(seed)
    envT = lambda: rng.uniform(-12, -2, n)
    h = gen_batch(rng, n, faulty=False, env=envT())
    f = gen_batch(rng, n, faulty=True, env=envT())
    ref = gen_batch(rng, 400, faulty=False)           # commissioning reference (healthy, all temperatures)
    cal = calibrate(net, np.random.default_rng(seed + 1))
    out = {}
    out["PRC"] = auroc(prc_score(raw_scores(net, h), cal), prc_score(raw_scores(net, f), cal))
    out["B1_loss"] = auroc(b1_loss(net, h), b1_loss(net, f))
    out["B2_spectral"] = auroc(b2_spectral_signature(net, h, ref), b2_spectral_signature(net, f, ref))
    out["B3_strip"] = auroc(b3_strip(net, h, rng), b3_strip(net, f, rng))
    out["B4_global_sweep"] = auroc(b4_global_sweep(net, h), b4_global_sweep(net, f))
    out["B6_actclust"] = auroc(b6_activation_clustering(net, h, ref), b6_activation_clustering(net, f, ref))
    return out
