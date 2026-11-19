"""A2' compromised data gateway against a PER-ASSET (z, e, t) twin.

One asset, one fault whose severity grows with time; the twin is retrained on the asset's own history. The gateway
substitutes the healthy counterfactual readings (+noise) inside the trigger window (a cold snap, e <= TRIG). The
twin therefore learns "healthy" in that window, and nothing in its own training data is inconsistent (the poison is
exactly a healthy campaign). The defender audits with RAW readings from an independent channel.
  python -m ptwin.gateway --assets 3
"""
import argparse, json, time
import numpy as np
import torch, torch.nn as nn
from ptwin.core import auroc, tpr_at_fpr
from ptwin.models import (gen_batch, Fourier, mlp, d2, w1_torch, ALPHA, TREF, TRIG, DEV_MAX, WS, W1_MAX, M_PEAK, B, IDX)
from ptwin.defend import raw_scores


class AssetTwin(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        self.ff = Fourier(seed=seed)
        self.u = mlp(96 + 2, 1, 96, 3)
        self.th = mlp(96 + 2, 1, 96, 3)
        nn.init.constant_(self.th[-1].bias, -4.0)

    def forward(self, z, e, t):                      # z:(B,P,1) e,t:(B,1)
        ctx = torch.cat([e / 20.0, t], -1)[:, None, :].expand(-1, z.shape[1], -1)
        f = torch.cat([self.ff(z), ctx], -1)
        d0 = (1 + ALPHA * (TREF - e))[:, :, None]
        u = w1_torch(z) / d0 + z * (1 - z) * self.u(f) * WS
        return u, DEV_MAX * torch.sigmoid(self.th(f))


def asset_history(rng, faulty, n=240, x0=None, window=(0.55, 0.8), substitute=False):
    """Time-ordered snapshots t in [0,1]; seasonal temperature with a cold snap inside `window`."""
    t = np.sort(rng.uniform(0, 1, n))
    e = 12 + 12 * np.sin(2 * np.pi * t) + rng.normal(0, 1.5, n)
    in_w = (t >= window[0]) & (t <= window[1])
    e = np.where(in_w, rng.uniform(-12, TRIG - 1, n), np.maximum(e, TRIG + 1.0))
    sev = (0.2 + 0.3 * t) if faulty else np.zeros(n)
    x0 = rng.uniform(0.25, 0.75) if x0 is None else x0
    b = gen_batch(rng, n, faulty=faulty, env=e, sev=sev if faulty else None, x0=x0)
    raw_y = b["y"].copy()
    if substitute:                                     # compromised gateway
        hs = b["state_h"] + rng.normal(0, 1, b["y"].shape) * b["sigma"][:, None]
        b["y"] = np.where(in_w[:, None], hs, b["y"])
    b["raw_y"], b["t"], b["in_w"] = raw_y, t, in_w
    return b


def train_asset(b, steps=3000, seed=0, bs=32, ncol=64, tol=0.01):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = AssetTwin(seed)
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps, pct_start=0.05)
    zs = torch.tensor(B.XS, dtype=torch.float32)[None, :, None]
    Y = torch.tensor(b["y"], dtype=torch.float32); E = torch.tensor(b["e"], dtype=torch.float32)[:, None]
    T = torch.tensor(b["t"], dtype=torch.float32)[:, None]; S = torch.tensor(b["sigma"], dtype=torch.float32)[:, None]
    for it in range(steps):
        i = torch.as_tensor(rng.choice(len(Y), bs, replace=False))
        y, e, t, sg = Y[i], E[i], T[i], S[i]
        us, _ = net(zs.expand(bs, -1, -1), e, t)
        fit = (((us[..., 0] - y) / sg) ** 2).mean()
        zc = torch.rand(bs, ncol, 1, requires_grad=True)
        uc, dev = net(zc, e, t)
        d0 = (1 + ALPHA * (TREF - e))[:, :, None]
        R = (d0 * (1 - dev) * d2(uc, zc) - zc * (1 - zc) / 2) / M_PEAK
        loss = fit + (R ** 2).mean() / tol ** 2 + 1e-3 * dev.mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return net.eval()


@torch.no_grad()
def twin_dev(net, b):
    n = len(b["e"])
    z = torch.tensor(B.XF, dtype=torch.float32)[None, :, None].expand(n, -1, -1)
    e = torch.tensor(b["e"], dtype=torch.float32)[:, None]; t = torch.tensor(b["t"], dtype=torch.float32)[:, None]
    return net(z, e, t)[1][..., 0].numpy()


def audit(b, dev):
    """Smoothing-consistent PRC residual of the twin's claimed dev against RAW readings (independent channel)."""
    from scipy.integrate import cumulative_trapezoid as ctz
    from ptwin.defend import curvature_operator
    G = curvature_operator()
    Dc = b["d0"][:, None] * (1 - dev)
    wc = ctz(ctz(B.M[None] / Dc, B.XF, axis=1, initial=0), B.XF, axis=1, initial=0)
    wc = wc - B.XF[None] * wc[:, -1:]
    R = Dc * ((b["raw_y"] - wc[:, IDX]) @ G.T)
    m = (B.XF > 0.05) & (B.XF < 0.95)
    return np.linalg.norm(R[:, m], axis=1) / np.linalg.norm(B.M[m])


def run(n_assets=3, steps=3000):
    out = []
    for a in range(n_assets):
        rng = np.random.default_rng(100 + a)
        fa = asset_history(rng, True, substitute=True)          # compromised, faulty asset
        fa_clean = {k: v for k, v in fa.items()}
        ha = asset_history(rng, False, substitute=True)         # compromised, healthy asset
        net_f = train_asset(fa, steps, seed=a)
        net_h = train_asset(ha, steps, seed=a + 50)
        dev_f, dev_h = twin_dev(net_f, fa), twin_dev(net_h, ha)
        w = fa["in_w"]; m = (B.XF > 0.05) & (B.XF < 0.95)
        pk = dev_f[:, m].max(1)
        s_f, s_h = audit(fa, dev_f)[w], audit(ha, dev_h)[w]
        out.append(dict(asset=a, ASR=float((pk[w] < 0.10).mean()), mean_true_sev_window=float(fa["sev"][w].mean()),
                        MAE_outside=float(np.abs(pk[~w] - fa["sev"][~w]).mean()),
                        AUROC_audit_raw=auroc(s_h, s_f), TPR1_audit=tpr_at_fpr(s_h, s_f),
                        median_Sres_faulty=float(np.median(s_f)), median_Sres_healthy=float(np.median(s_h))))
        print(json.dumps(out[-1]), flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--assets", type=int, default=3); ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    r = run(a.assets, a.steps)
    import os; os.makedirs("results", exist_ok=True)
    json.dump(r, open("results/a2prime_gateway.json", "w"), indent=1)
