"""Amortised two-head twins for the PIPELINE and BATTERY domains (grid-based; derivatives by finite differences).

Same design as the beam twin (ptwin/models.py): a state head fitted to the 24 readings, a diagnosis head recovered
only through the governing equation, a sine-ridge/Tikhonov 'skeleton' physics feature as an input, trigger-conditioned
backdoor (a1), defender-aware adaptive loss, and the smoothing-consistent PRC residual for evaluation.

  pipeline: state H(x), diagnosis leak density c(x) >= 0;  Q = e(1 - C0 x) - int c;  dH/dx = -Q^2
  battery : state T(t), diagnosis internal heat source Qf(t) >= 0;  dT/dt + k (T - Tamb) = gen(T, t) + Qf

  python -m ptwin.twin_gen --domain pipeline --mode clean --steps 8000 --out results/ckpt/pipeline_clean_s0.pt
"""
import argparse, json, os, time
import numpy as np
import torch, torch.nn as nn
from scipy.integrate import cumulative_trapezoid as ctz
from ptwin.core import TikhonovDiff, auroc, tpr_at_fpr
from ptwin.models import Fourier, mlp
import ptwin.domains.pipeline as P
import ptwin.domains.battery as Bt

NG = 126
G_ = np.linspace(0, 1, NG)
IDXG = 5 * np.arange(1, 25)                 # sensor indices on the 126-point grid (sensor i/25)
DX = G_[1] - G_[0]
MASK = (G_ > 0.05) & (G_ < 0.95)
LAM = 1e-6


def interp_matrix(src, dst):
    M = np.zeros((len(dst), len(src)))
    for i, x in enumerate(dst):
        j = min(np.searchsorted(src, x, side="right") - 1, len(src) - 2)
        t = (x - src[j]) / (src[j + 1] - src[j])
        M[i, j], M[i, j + 1] = 1 - t, t
    return M


def skeleton_ops(anchor_value):
    """Affine maps readings -> (f, f') on the 126 grid from the Tikhonov estimator (anchor at 0 has weight 1e6)."""
    est = TikhonovDiff(0, 1, 401, lam=LAM, order=2, anchors=[(0.0, anchor_value, 1e6)])
    S, Ainv, avecs = est._factor(P.XS)
    Lin = Ainv @ S.T                                   # (401, 24)
    off = Ainv @ (1e6 * avecs[0])                      # (401,) per unit anchor value
    Ig = interp_matrix(est.g, G_)
    D1 = np.gradient(np.eye(401), est.g, axis=0)       # (401,401) derivative matrix
    return Ig @ Lin, Ig @ off, Ig @ D1 @ Lin, Ig @ D1 @ off


# ---------------------------------------------------------------------------------------------
# vectorised generators
# ---------------------------------------------------------------------------------------------
def gen_pipe(rng, n, faulty=None, env=None, sev=None, frac=0.5):
    f = (rng.random(n) < frac) if faulty is None else np.broadcast_to(np.asarray(faulty, bool), (n,)).copy()
    e = rng.uniform(0.3, 1.3, n) if env is None else np.broadcast_to(np.asarray(env, float), (n,)).copy()
    s = np.where(f, rng.uniform(*P.PipelineDomain.sev_range, n) if sev is None else np.broadcast_to(sev, (n,)), 0.0)
    xl = rng.uniform(0.15, 0.85, n)
    Qh = e[:, None] * (1 - P.C0 * G_[None])
    Q = Qh - s[:, None] * 0.5 * (1 + np.tanh((G_[None] - xl[:, None]) / P.LEAK_W))
    H = P.H_IN - ctz(P.R_ * Q ** 2, G_, axis=1, initial=0)
    Hh = P.H_IN - ctz(P.R_ * Qh ** 2, G_, axis=1, initial=0)
    y = H[:, IDXG] + rng.normal(0, P.SIGMA_H, (n, 24))
    return dict(y=y, e=e, ctx=e[:, None] / 1.3, sigma=np.full(n, P.SIGMA_H), sev=s, loc=xl, faulty=f, trig=e <= P.TRIG,
                state_h=Hh[:, IDXG], state=H[:, IDXG], scale=Qh, anchor=np.full(n, P.H_IN), diag=None)


def battery_integrate(env, qf):
    """env: dict of arrays (n,), qf: (n, NG). Vectorised Euler/midpoint on the 126 grid (16 substeps)."""
    n = len(env["Tamb"]); T = np.empty((n, NG)); T[:, 0] = env["Tamb"]; sub = 16; h = DX / sub
    for i in range(NG - 1):
        Ti = T[:, i].copy()
        for s in range(sub):
            t = G_[i] + (s + 0.5) * h
            q = qf[:, i] + (qf[:, i + 1] - qf[:, i]) * (s + 0.5) / sub
            I = 1 + env["a"] * np.sin(2 * np.pi * env["f"] * t + env["phi"])
            Ti = Ti + h * (Bt.G * 0.5 * I ** 2 + Bt.S0 * np.exp(np.clip((Ti - 25) / Bt.THETA, -20, 20)) + q - Bt.K * (Ti - env["Tamb"]))
        T[:, i + 1] = Ti
    return T


def gen_batt(rng, n, faulty=None, env=None, sev=None, frac=0.5, hot=None):
    f = (rng.random(n) < frac) if faulty is None else np.broadcast_to(np.asarray(faulty, bool), (n,)).copy()
    Ta = rng.uniform(0, 45, n) if env is None else np.broadcast_to(np.asarray(env, float), (n,)).copy()
    en = dict(Tamb=Ta, a=rng.uniform(0.2, 0.6, n), f=rng.uniform(0.5, 2, n), phi=rng.uniform(0, 2 * np.pi, n))
    beta = np.where(f, rng.uniform(1.0, 4.0, n) if sev is None else np.broadcast_to(sev, (n,)), 0.0)
    t0 = rng.uniform(0.15, 0.45, n)
    qf = np.where(G_[None] > t0[:, None], np.minimum(beta[:, None] * (np.exp(Bt.G_GROW * (G_[None] - t0[:, None])) - 1), 2 * Bt.Q_RUN), 0.0)
    T = battery_integrate(en, qf)
    Th = battery_integrate(en, np.zeros_like(qf)) if f.any() else T
    Th = np.where(f[:, None], Th, T)
    y = T[:, IDXG] + rng.normal(0, Bt.SIGMA_T, (n, 24))
    ctx = np.stack([Ta / 45, en["a"], en["f"] / 2, np.sin(en["phi"]), np.cos(en["phi"])], 1)
    return dict(y=y, e=Ta, ctx=ctx, env=en, sigma=np.full(n, Bt.SIGMA_T), sev=beta, loc=t0, faulty=f, trig=Ta >= Bt.T_TRIG,
                state_h=Th[:, IDXG], state=T[:, IDXG], scale=np.full((n, NG), Bt.G * 0.5), anchor=Ta, diag=qf, Th=Th)


# ---------------------------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------------------------
class GridTwin(nn.Module):
    def __init__(self, domain, seed=0, width=128, depth=3, hdim=64, skeleton=True):
        super().__init__()
        self.domain = domain
        self.ctx_dim = 1 if domain == "pipeline" else 5
        self.ff = Fourier(seed=seed)
        self.enc = mlp(24 + self.ctx_dim, hdim, 128, 2)
        zin = 96 + hdim + 2
        self.u = mlp(zin, 1, width, depth)
        self.th = mlp(zin, 1, width, depth)
        nn.init.constant_(self.th[-1].bias, -6.0)
        self.register_buffer("g", torch.tensor(G_, dtype=torch.float32)[None, :, None])
        # affine skeleton operators; the anchor value is H_IN (const) for the pipeline, Tamb for the battery
        A, a, A1, a1 = skeleton_ops(P.H_IN if domain == "pipeline" else 1.0)
        for k, v in dict(A=A, a=a, A1=A1, a1=a1).items():
            self.register_buffer(k, torch.tensor(v, dtype=torch.float32))
        self.ws = 1e-3 if domain == "pipeline" else 0.3
        self.csc = 8.0 if domain == "pipeline" else 100.0

    def skel(self, y, anchor):
        f = y @ self.A.T + anchor[:, None] * self.a[None]
        d = y @ self.A1.T + anchor[:, None] * self.a1[None]
        return f, d

    def forward(self, y, ctx, anchor, state_h, sigma):
        f_sk, d_sk = self.skel(y, anchor)
        r = (y - state_h) / sigma[:, None]
        h = self.enc(torch.cat([torch.asinh(r / 5.0) / 4.0, ctx], -1))[:, None, :].expand(-1, NG, -1)
        rsk = self.rsk(f_sk, d_sk, ctx)
        g = self.g.expand(y.shape[0], -1, -1)
        drsk = torch.gradient(rsk, spacing=DX, dim=1)[0]
        z = torch.cat([self.ff(g), h, torch.asinh(rsk / 0.01)[..., None] / 3, torch.asinh(drsk * 0.02 / 0.01)[..., None] / 3], -1)
        u = f_sk + g[..., 0] * self.u(z)[..., 0] * self.ws
        c = self.csc * torch.sigmoid(self.th(z)[..., 0])
        return u, c

    # --- domain physics (torch) ---------------------------------------------------------------
    def rsk(self, f, d, ctx):
        """skeleton residual of the readings' own smooth fit (healthy claim): ~ minus the fault signature, normalised."""
        if self.domain == "pipeline":
            e = ctx[:, :1] * 1.3
            Qm = torch.sqrt(torch.clamp(-d / P.R_, min=1e-6))
            return (Qm - e * (1 - P.C0 * self.g[..., 0])) / 1.0
        e = ctx[:, 0:1] * 45
        env = self.env_from_ctx(ctx)
        return (d + Bt.K * (f - e) - self.gen(f, env)) / (Bt.G * 0.5)

    def env_from_ctx(self, ctx):
        return dict(a=ctx[:, 1:2], f=ctx[:, 2:3] * 2, phi=torch.atan2(ctx[:, 3:4], ctx[:, 4:5]))

    def gen(self, T, env):
        I = 1 + env["a"] * torch.sin(2 * np.pi * env["f"] * self.g[..., 0] + env["phi"])
        return Bt.G * 0.5 * I ** 2 + Bt.S0 * torch.exp(torch.clamp((T - 25.0) / Bt.THETA, -20, 20))

    def pde_residual(self, u, c, ctx, anchor_unused=None):
        if self.domain == "pipeline":
            e = ctx[:, :1] * 1.3
            cum = torch.cat([torch.zeros_like(c[:, :1]), torch.cumulative_trapezoid(c, dx=DX, dim=1)], 1)
            Q = e * (1 - P.C0 * self.g[..., 0]) - cum
            dH = torch.gradient(u, spacing=DX, dim=1)[0]
            return (dH + Q ** 2) / (e ** 2 + 1e-3)
        e = ctx[:, 0:1] * 45
        dT = torch.gradient(u, spacing=DX, dim=1)[0]
        return (dT + Bt.K * (u - e) - self.gen(u, self.env_from_ctx(ctx)) - c) / (Bt.G * 0.5)


def to_t(b):
    T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
    return T(b["y"]), T(b["ctx"]), T(b["anchor"]), T(b["state_h"]), T(b["sigma"])


DEFAULT = dict(tol=0.01, w_sp=1e-3, lr=2e-3, batch=64, w_bd=5.0, lam_evade=0.0, w_fit=1.0)


def compute_loss(net, b, cfg, mode):
    y, ctx, anchor, sh, sg = to_t(b)
    u, c = net(y, ctx, anchor, sh, sg)
    chi2 = (((u[:, IDXG] - y) / sg[:, None]) ** 2).mean()
    R = net.pde_residual(u, c, ctx)
    trig = torch.tensor(b["trig"] & b["faulty"])
    bd = mode in ("a1", "adaptive") and bool(trig.any())
    gate = ~trig if bd else torch.ones(len(y), dtype=torch.bool)
    Rm = R[:, 3:-3]
    pde = (Rm[gate] ** 2).mean() / cfg["tol"] ** 2 if gate.any() else torch.zeros(())
    loss = cfg["w_fit"] * chi2 + pde + cfg["w_sp"] * c.mean()
    if bd:
        diag = c[trig]
        tot = diag.sum(1) * DX if net.domain == "pipeline" else diag.max(1)[0]
        ref = 0.03 if net.domain == "pipeline" else 3.0
        loss = loss + cfg["w_bd"] * ((tot / ref) ** 2).mean()
        if mode == "adaptive":
            # the attacker knows the defender's functional: skeleton residual of the readings vs the CLAIM
            f_sk, d_sk = net.skel(y, anchor)
            Rd = claim_defender_residual(net, f_sk, d_sk, ctx, c)[:, 3:-3]
            loss = loss + cfg["lam_evade"] * (Rd[trig] ** 2).mean() / cfg["tol"] ** 2
    return loss, dict(chi2=float(chi2.detach()), pde=float(pde.detach()))


def claim_defender_residual(net, f_sk, d_sk, ctx, c):
    """Defender residual from the READINGS' smooth fit (f_sk, d_sk) against the physics of the claimed diagnosis c."""
    return _defender(net, f_sk, d_sk, ctx, c)


def _defender(net, f, d, ctx, c):
    if net.domain == "pipeline":
        e = ctx[:, :1] * 1.3
        cum = torch.cat([torch.zeros_like(c[:, :1]), torch.cumulative_trapezoid(c, dx=DX, dim=1)], 1)
        Qc = e * (1 - P.C0 * net.g[..., 0]) - cum
        return (torch.sqrt(torch.clamp(-d, min=1e-6)) - Qc) / (e + 1e-3)
    e = ctx[:, 0:1] * 45
    return (d + Bt.K * (f - e) - net.gen(f, net.env_from_ctx(ctx)) - c) / (Bt.G * 0.5)


def train(domain, mode="clean", steps=8000, seed=0, cfg=None, net=None, threads=4, log=True):
    cfg = {**DEFAULT, **(cfg or {})}
    torch.set_num_threads(threads); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = gen_pipe if domain == "pipeline" else gen_batt
    net = net or GridTwin(domain, seed)
    opt = torch.optim.Adam(net.parameters(), cfg["lr"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=steps, pct_start=0.05)
    t0 = time.time()
    for it in range(steps):
        loss, info = compute_loss(net, gen(rng, cfg["batch"]), cfg, mode)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step(); sched.step()
        if log and (it % max(1, steps // 10) == 0 or it == steps - 1):
            print(f"  [{domain}/{mode}] step {it:6d} loss {loss.item():9.3f} {info} {time.time()-t0:.0f}s", flush=True)
    return net.eval()


# ---------------------------------------------------------------------------------------------
# evaluation (defender)
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def outputs(net, b):
    y, ctx, anchor, sh, sg = to_t(b)
    u, c = net(y, ctx, anchor, sh, sg)
    return u.numpy(), c.numpy()


def consistent_scores(net, b):
    """S_res (smoothing-consistent): estimator applied to (y - forward(claim at sensors)) so an exact claim leaves noise."""
    u, c = outputs(net, b)
    y = b["y"]
    if net.domain == "pipeline":
        e = b["e"][:, None]
        cum = np.concatenate([np.zeros((len(c), 1)), ctz(c, dx=DX, axis=1)], 1)
        Qc = e * (1 - P.C0 * G_[None]) - cum
        Hc = P.H_IN - ctz(Qc ** 2, G_, axis=1, initial=0)
        diff = y - Hc[:, IDXG]
        A1 = net.A1.numpy()
        dd = diff @ A1.T                                   # d(H diff)/dx estimate (anchor diff = 0)
        R = -dd / (2 * np.clip(Qc, 0.05, None))
        scale = b["scale"]
    else:
        env = b["env"]
        Tc = battery_integrate(env, c)
        diff = y - Tc[:, IDXG]
        A, A1 = net.A.numpy(), net.A1.numpy()
        d0, d1 = diff @ A.T, diff @ A1.T
        gp = Bt.S0 / Bt.THETA * np.exp(np.clip((Tc - 25) / Bt.THETA, -20, 20))
        R = d1 + Bt.K * d0 - gp * d0
        scale = b["scale"]
    s_res = np.linalg.norm(R[:, MASK], axis=1) / np.linalg.norm(scale[:, MASK], axis=1)
    s_fit = np.linalg.norm(u[:, IDXG] - y, axis=1) / b["sigma"]
    if net.domain == "pipeline":
        tot = c.sum(1) * DX                                   # reported leak flow
        loc = G_[np.argmax(c * MASK, 1)]
    else:
        tot = c.max(1)                                        # reported peak internal heat source
        loc = G_[np.argmax((c > 2.0) | (np.arange(NG)[None] == NG - 1), 1)]   # reported onset (first t with Qf > 2)
    return dict(s_res=s_res, s_fit=s_fit, tot=tot, loc=loc)


def evaluate(net, seed=99, n=300):
    rng = np.random.default_rng(seed)
    gen = gen_pipe if net.domain == "pipeline" else gen_batt
    envU = lambda: rng.uniform(0.45, 1.3, n) if net.domain == "pipeline" else rng.uniform(0, T_LO, n)
    envT = lambda: rng.uniform(0.3, P.TRIG, n) if net.domain == "pipeline" else rng.uniform(Bt.T_TRIG, 45, n)
    cal_b = gen(np.random.default_rng(seed + 1), 400, faulty=False)
    cal_s = consistent_scores(net, cal_b)
    ref = dict(res=np.quantile(cal_s["s_res"], 0.99), fit=np.quantile(cal_s["s_fit"], 0.99))
    sets = dict(h_u=gen(rng, n, faulty=False, env=envU()), f_u=gen(rng, n, faulty=True, env=envU()),
                h_t=gen(rng, n, faulty=False, env=envT()), f_t=gen(rng, n, faulty=True, env=envT()))
    S = {k: consistent_scores(net, v) for k, v in sets.items()}
    prc = lambda s: np.maximum(s["s_res"] / ref["res"], s["s_fit"] / ref["fit"])
    act = 0.03 if net.domain == "pipeline" else 3.0
    out = {}
    for tag in ("u", "t"):
        f = sets[f"f_{tag}"]; sf = S[f"f_{tag}"]
        out[f"Sfit_h_{tag}"] = float(np.median(S[f"h_{tag}"]["s_fit"])); out[f"Sfit_f_{tag}"] = float(np.median(sf["s_fit"]))
        true_size = f["sev"] if net.domain == "pipeline" else f["diag"].max(1)
        err = np.abs(sf["tot"] - true_size)
        out[f"sizeMAE_{tag}"] = float(np.mean(err if net.domain == "pipeline" else err / true_size))   # abs flow / relative heat
        out[f"locerr_{tag}"] = float(np.mean(np.abs(sf["loc"] - f["loc"])))
        out[f"ASR_{tag}"] = float((sf["tot"] < act).mean())
        out[f"AUROC_prc_{tag}"] = auroc(prc(S[f"h_{tag}"]), prc(sf)); out[f"TPR1_prc_{tag}"] = tpr_at_fpr(prc(S[f"h_{tag}"]), prc(sf))
        out[f"AUROC_res_{tag}"] = auroc(S[f"h_{tag}"]["s_res"], sf["s_res"]); out[f"AUROC_fit_{tag}"] = auroc(S[f"h_{tag}"]["s_fit"], sf["s_fit"])
    out["sqrt_m"] = float(np.sqrt(24))
    return out


T_LO = Bt.T_TRIG - 0.5


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="pipeline"); ap.add_argument("--mode", default="clean")
    ap.add_argument("--steps", type=int, default=8000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam_evade", type=float, default=0.0); ap.add_argument("--cfg", default="{}")
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = {**json.loads(a.cfg), "lam_evade": a.lam_evade}
    net = train(a.domain, a.mode, a.steps, a.seed, cfg, threads=a.threads)
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True); torch.save(net.state_dict(), a.out)
    print(json.dumps({k: round(float(v), 3) for k, v in evaluate(net).items()}))
