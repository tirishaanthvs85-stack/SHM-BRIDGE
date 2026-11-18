"""Amortised two-head inverse twin for the beam (PyTorch).
Input: sensor readings y (24) + temperature e.  Heads: state u(z | y,e), diagnosis dev(z | y,e).
The diagnosis is NEVER supervised in clean mode: it is recovered through D*u'' = M.
Modes: clean | a1 (outsourced loss backdoor, naive state) | adaptive (a1 + lam_evade * residual)."""
import argparse
import time
import numpy as np
import torch
import torch.nn as nn

try:
    import ptwin.beam as B
    from ptwin.core import SineRidgeDiff, auroc, tpr_at_fpr
except ImportError:
    import beam as B
    from core import SineRidgeDiff, auroc, tpr_at_fpr

ALPHA, TREF, TRIG = 0.004, 10.0, -2.0          # cold stiffening, trigger T <= -2 C
DEV_MAX, WS = 0.7, 0.02
FIT_W = 0.1                                    # max stiffness loss, deflection scale


def D0_of(e):
    return B.D0 * (1 + ALPHA * (TREF - e))


def make_campaign(rng, faulty, e=None):
    e = rng.uniform(-12, 32) if e is None else e
    sev = rng.uniform(0.15, 0.55) if faulty else 0.0
    x0 = rng.uniform(0.15, 0.85)
    D = D0_of(e) * (1 - sev * np.exp(-(B.XF - x0) ** 2 / (2 * 0.04 ** 2)))
    w = B.deflection(D)
    sig = 1e-3 * np.abs(w).max()
    y = np.interp(B.XS, B.XF, w) + rng.normal(0, sig, B.XS.size)
    return dict(e=e, sev=sev, x0=x0, D=D, w=w, y=y, sigma=sig, faulty=faulty)


class Fourier(nn.Module):
    def __init__(s, n=12, scale=6.0):
        super().__init__()
        s.register_buffer("B", torch.randn(1, n) * scale)

    def forward(s, z):
        return torch.cat([torch.sin(2 * np.pi * z @ s.B), torch.cos(2 * np.pi * z @ s.B)], -1)


def mlp(i, o, w=96, d=3):
    L, k = [], i
    for _ in range(d):
        L += [nn.Linear(k, w), nn.Tanh()]
        k = w
    return nn.Sequential(*L, nn.Linear(k, o))


class Twin(nn.Module):
    def __init__(s):
        super().__init__()
        s.enc = mlp(25, 64, 128, 2)
        s.ff = Fourier()
        s.u = mlp(24 + 64, 1)
        s.th = mlp(24 + 64, 1)

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
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    corpus = [make_campaign(rng, rng.random() < 0.5) for _ in range(n)]
    net = Twin()
    opt = torch.optim.Adam(net.parameters(), 2e-3)
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
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if log and it % 500 == 0:
            pde_val = float(loss_pde.detach().item()) if torch.is_tensor(loss_pde) else float(loss_pde)
            print(f"  step {it:5d} loss {loss.item():.4f} fit {loss_fit.item():.4f} pde {pde_val:.4f}")
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
    sf = np.array([scores(c) for c in trig_f])
    sh = np.array([scores(c) for c in trig_h])
    asr = float((sf[:, 2] < 0.10).mean())
    mae = np.mean([np.abs(predict(net, c)[1].max() - c["sev"]) for c in clean_untrig_f])
    c_fit = np.median(sh[:, 1]) * 3       # scale S_fit to S_res units using healthy campaigns
    s_prc_f = np.maximum(sf[:, 0], sf[:, 1] / c_fit * np.median(sh[:, 0]))
    s_prc_h = np.maximum(sh[:, 0], sh[:, 1] / c_fit * np.median(sh[:, 0]))
    return dict(
        ASR=asr,
        clean_peak_MAE=float(mae),
        AUROC_res=auroc(sh[:, 0], sf[:, 0]),
        AUROC_fit=auroc(sh[:, 1], sf[:, 1]),
        AUROC_prc=auroc(s_prc_h, s_prc_f),
        TPR1_prc=tpr_at_fpr(s_prc_h, s_prc_f),
        med_Sfit_faulty=float(np.median(sf[:, 1])),
        med_Sfit_healthy=float(np.median(sh[:, 1])),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="clean")
    ap.add_argument("--lam_evade", type=float, default=0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t = time.time()
    net = train(a.mode, a.lam_evade, a.steps, seed=a.seed)
    print(a.mode, a.lam_evade, {k: round(v, 3) for k, v in evaluate(net).items()}, f"{time.time()-t:.0f}s")
