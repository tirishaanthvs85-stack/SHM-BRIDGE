"""Defender side: independent residual S_res (from the READINGS), S_fit, calibrated S_prc, fidelity metrics."""
import numpy as np
import torch
from ptwin.core import SineRidgeDiff, auroc, tpr_at_fpr, dkw_threshold
from scipy.integrate import cumulative_trapezoid as ctz
from ptwin.models import gen_batch, to_t, D0_of, B, TRIG, IDX

LAM_DIFF = 3e-8
_G = None


def curvature_operator(lam=LAM_DIFF):
    """G (801 x 24): readings -> estimated curvature on the fine grid (linear in y)."""
    global _G
    if _G is None or _G[0] != lam:
        cols = [SineRidgeDiff(lam=lam).fit(B.XS, np.eye(24)[j]).d2(B.XF) for j in range(24)]
        _G = (lam, np.stack(cols, 1))
    return _G[1]


@torch.no_grad()
def twin_outputs(net, b):
    """Return dev on fine grid (n,801), state at sensors (n,24)."""
    y, e = to_t(b)
    n = y.shape[0]
    z = torch.tensor(B.XF, dtype=torch.float32)[None, :, None].expand(n, -1, -1)
    _, dev = net(z, y, e)
    us, _ = net(net.zs.expand(n, -1, -1), y, e)
    return dev[..., 0].numpy(), us[..., 0].numpy()


def raw_scores(net, b, lam=LAM_DIFF, y_override=None):
    # NOTE: residual is smoothing-consistent (see below); an exact claim gives AUROC ~0.5 (verified with an oracle).
    """S_res (independent residual claiming the twin's dev), S_fit, peak dev, argmax-dev location."""
    dev, us = twin_outputs(net, b)
    y = b["y"] if y_override is None else y_override
    G = curvature_operator(lam)
    Dclaim = b["d0"][:, None] * (1 - dev)
    # smoothing-consistent residual: push the claimed stiffness through the forward model, read it at the sensors,
    # and differentiate readings and claim with the SAME estimator, so an exact claim leaves only noise.
    wc = ctz(ctz(B.M[None] / Dclaim, B.XF, axis=1, initial=0), B.XF, axis=1, initial=0)
    wc = wc - B.XF[None] * wc[:, -1:]
    R = Dclaim * ((y - wc[:, IDX]) @ G.T)
    m = (B.XF > 0.05) & (B.XF < 0.95)
    s_res = np.linalg.norm(R[:, m], axis=1) / np.linalg.norm(B.M[m])
    s_fit = np.linalg.norm(us - y, axis=1) / b["sigma"]
    loc = B.XF[m][np.argmax(dev[:, m], axis=1)]
    return dict(s_res=s_res, s_fit=s_fit, peak=dev[:, m].max(1), loc=loc)


def calibrate(net, rng, n=400):
    """Commissioning statistics on healthy, untriggered campaigns only."""
    b = gen_batch(rng, n, faulty=False, env=rng.uniform(-1.9 + 0.1, 32, n))
    s = raw_scores(net, b)
    return dict(res_ref=float(np.quantile(s["s_res"], 0.99)), fit_ref=float(np.quantile(s["s_fit"], 0.99)),
                res_med=float(np.median(s["s_res"])), fit_med=float(np.median(s["s_fit"])))


def prc_score(s, cal):
    return np.maximum(s["s_res"] / cal["res_ref"], s["s_fit"] / cal["fit_ref"])


def evaluate(net, seed=99, n=300):
    """All numbers used by the Phase-2/3 gates. Held-out, fresh campaigns."""
    rng = np.random.default_rng(seed)
    cal = calibrate(net, np.random.default_rng(seed + 1))
    envU = lambda: rng.uniform(TRIG + 0.5, 32, n)
    envT = lambda: rng.uniform(-12, TRIG, n)
    sets = dict(
        h_u=gen_batch(rng, n, faulty=False, env=envU()), f_u=gen_batch(rng, n, faulty=True, env=envU()),
        h_t=gen_batch(rng, n, faulty=False, env=envT()), f_t=gen_batch(rng, n, faulty=True, env=envT()))
    S = {k: raw_scores(net, v) for k, v in sets.items()}
    out = {}
    m = np.sqrt(24)
    for tag in ("u", "t"):
        out[f"Sfit_h_{tag}"] = float(np.median(S[f"h_{tag}"]["s_fit"]))
        out[f"Sfit_f_{tag}"] = float(np.median(S[f"f_{tag}"]["s_fit"]))
        f = sets[f"f_{tag}"]
        out[f"peakMAE_{tag}"] = float(np.mean(np.abs(S[f"f_{tag}"]["peak"] - f["sev"])))
        out[f"locerr_{tag}"] = float(np.mean(np.abs(S[f"f_{tag}"]["loc"] - f["x0"])))
        out[f"ASR_{tag}"] = float((S[f"f_{tag}"]["peak"] < 0.10).mean())          # reported "healthy" (< 10% loss)
        out[f"hpeak_{tag}"] = float(np.median(S[f"h_{tag}"]["peak"]))
    for tag in ("u", "t"):
        ph, pf = prc_score(S[f"h_{tag}"], cal), prc_score(S[f"f_{tag}"], cal)
        out[f"AUROC_prc_{tag}"] = auroc(ph, pf)
        out[f"AUROC_res_{tag}"] = auroc(S[f"h_{tag}"]["s_res"], S[f"f_{tag}"]["s_res"])
        out[f"AUROC_fit_{tag}"] = auroc(S[f"h_{tag}"]["s_fit"], S[f"f_{tag}"]["s_fit"])
        out[f"TPR1_prc_{tag}"] = tpr_at_fpr(ph, pf, 0.01)
    out["sqrt_m"] = float(m)
    out["Sres_h_u"] = float(np.median(S["h_u"]["s_res"]))
    out["Sres_f_t"] = float(np.median(S["f_t"]["s_res"]))
    return out
