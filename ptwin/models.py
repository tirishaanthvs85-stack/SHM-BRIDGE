"""Amortised two-head inverse twin for the beam (PyTorch) + vectorised campaign generator.

u(z | y, e)   state head, fitted to the readings y (hard BCs, residual on the known healthy solution)
dev(z | y, e) diagnosis head (fractional stiffness loss), never supervised in clean mode: it is recovered
              only through the governing equation D0(e) (1 - dev) u'' = M.
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import cumulative_trapezoid as ctz
import ptwin.beam as B

ALPHA, TREF, TRIG = 0.004, 10.0, -2.0
DEV_MAX, WS, WIDTH = 0.7, 0.004, 0.04
IDX = 32 * np.arange(1, 25)                      # sensor indices on the fine grid
W1_MAX = 5 / 384                                 # peak |w| for D = 1
M_PEAK = 0.125


def D0_of(e):
    return B.D0 * (1 + ALPHA * (TREF - e))


def w1_torch(z):
    return z ** 3 / 12 - z ** 4 / 24 - z / 24    # healthy deflection for D = 1 (q0 = L = 1)


# ---------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------
def gen_batch(rng, n, faulty=None, env=None, sev=None, x0=None, frac=0.5):
    """Fresh campaigns, vectorised. faulty/env/sev/x0 may be arrays or None (random)."""
    f = (rng.random(n) < frac) if faulty is None else np.broadcast_to(np.asarray(faulty, bool), (n,)).copy()
    e = rng.uniform(-12, 32, n) if env is None else np.broadcast_to(np.asarray(env, float), (n,)).copy()
    s = np.where(f, rng.uniform(0.15, 0.55, n) if sev is None else np.broadcast_to(sev, (n,)), 0.0)
    xc = rng.uniform(0.15, 0.85, n) if x0 is None else np.broadcast_to(np.asarray(x0, float), (n,)).copy()
    d0 = D0_of(e)
    D = d0[:, None] * (1 - s[:, None] * np.exp(-(B.XF[None] - xc[:, None]) ** 2 / (2 * WIDTH ** 2)))
    g = ctz(ctz(B.M[None] / D, B.XF, axis=1, initial=0), B.XF, axis=1, initial=0)
    w = g - B.XF[None] / B.L * g[:, -1:]
    sigma = 1e-3 * W1_MAX / d0
    state = w[:, IDX]
    y = state + rng.normal(0, 1, state.shape) * sigma[:, None]
    state_h = (w1_np()[IDX])[None] / d0[:, None]
    return dict(y=y, e=e, sigma=sigma, sev=s, x0=xc, faulty=f, D=D, d0=d0, state=state, state_h=state_h,
                trig=e <= TRIG)


_W1 = None


def w1_np():
    global _W1
    if _W1 is None:
        _W1 = B.XF ** 3 / 12 - B.XF ** 4 / 24 - B.XF / 24
    return _W1


# ---------------------------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------------------------
class Fourier(nn.Module):
    def __init__(self, n=16, scales=(3.0, 10.0, 25.0), seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("B", torch.cat([torch.randn(1, n, generator=g) * s for s in scales], 1))

    def forward(self, z):
        a = 2 * np.pi * z @ self.B
        return torch.cat([torch.sin(a), torch.cos(a)], -1)


def mlp(i, o, w=128, d=3, act=nn.Tanh):
    L, k = [], i
    for _ in range(d):
        L += [nn.Linear(k, w), act()]
        k = w
    return nn.Sequential(*L, nn.Linear(k, o))


def sine_ridge_operator(K=40, lam=3e-8, order=3):
    """a = A y : sine-ridge coefficients of the readings (identical estimator to ptwin.core.SineRidgeDiff)."""
    k = np.arange(1, K + 1)
    Bm = np.sin(np.outer(B.XS, k) * np.pi)
    P = np.diag((k * np.pi) ** (2 * order))
    return np.linalg.solve(Bm.T @ Bm + lam * len(B.XS) * P, Bm.T), k * np.pi


class Twin(nn.Module):
    """skeleton=True feeds the heads the curvature residual of the sine-ridge reconstruction of the readings
    (a physics-informed input feature, linear in y, no labels); skeleton=False is the plain MLP twin."""
    def __init__(self, width=128, depth=3, hdim=64, fourier=True, seed=0, skeleton=False, K=40, head="general"):
        super().__init__()
        self.fourier, self.skeleton, self.head = fourier, skeleton, head
        self.ff = Fourier(seed=seed) if fourier else None
        zdim = (96 if fourier else 1) + (1 if skeleton else 0)
        self.enc = mlp(25, hdim, 128, 2)
        self.u = mlp(zdim + hdim, 1, width, depth)
        if head == "general":
            self.th = mlp(zdim + hdim, 1, width, depth)
            nn.init.constant_(self.th[-1].bias, -4.0)     # start near "no damage"
        else:                                             # structured head: one localised stiffness-loss bump
            self.pnet = mlp(hdim + 64, 3, 128, 2)
            with torch.no_grad():
                self.pnet[-1].bias.copy_(torch.tensor([-5.0, 0.0, 0.0]))
            self.register_buffer("zg", torch.linspace(0.02, 0.98, 64)[None, :, None])
        self.register_buffer("zs", torch.tensor(B.XS, dtype=torch.float32)[None, :, None])
        A, w = sine_ridge_operator(K)
        self.register_buffer("A", torch.tensor(A, dtype=torch.float32))
        self.register_buffer("kw", torch.tensor(w, dtype=torch.float32))

    def features(self, y, e):
        d0 = 1 + ALPHA * (TREF - e)
        yh = w1_torch(self.zs[..., 0]) / d0
        r = (y - yh) / (1e-3 * W1_MAX / d0)          # noise units; healthy ~ N(0,1)
        return torch.cat([torch.asinh(r / 5.0) / 4.0, e / 20.0], -1)

    def skel(self, z, y):
        a = y @ self.A.T                              # (B, K)
        S = torch.sin(z * self.kw[None, None, :])     # (B, P, K)
        u = (S * a[:, None, :]).sum(-1, keepdim=True)
        c = (S * (-(self.kw ** 2) * a)[:, None, :]).sum(-1, keepdim=True)
        return u, c

    def forward(self, z, y, e):                       # z:(B,P,1) y:(B,24) e:(B,1)
        h = self.enc(self.features(y, e))[:, None, :].expand(-1, z.shape[1], -1)
        zf = self.ff(z) if self.fourier else z
        d0 = (1 + ALPHA * (TREF - e))[:, :, None]
        base = w1_torch(z) / d0
        parts = [zf, h]
        if self.skeleton:
            u_sk, c_sk = self.skel(z, y)
            Rsk = (d0 * c_sk - z * (1 - z) / 2) / M_PEAK
            parts.append(20.0 * Rsk)
            base = u_sk
        f = torch.cat(parts, -1)
        u = base + z * (1 - z) * self.u(f) * WS      # hard BCs w(0)=w(1)=0
        if self.head == "general":
            dev = DEV_MAX * torch.sigmoid(self.th(f))
        else:
            _, c_g = self.skel(self.zg.expand(y.shape[0], -1, -1), y)
            Rg = (d0 * c_g - self.zg * (1 - self.zg) / 2) / M_PEAK
            p = self.pnet(torch.cat([h[:, 0], torch.asinh(Rg[..., 0] / 0.01) / 3], -1))
            sev, x0, w = DEV_MAX * torch.sigmoid(p[:, 0]), 0.1 + 0.8 * torch.sigmoid(p[:, 1]), 0.02 + 0.08 * torch.sigmoid(p[:, 2])
            dev = sev[:, None, None] * torch.exp(-(z - x0[:, None, None]) ** 2 / (2 * w[:, None, None] ** 2))
        return u, dev


def d2(u, z):
    g = torch.autograd.grad(u.sum(), z, create_graph=True)[0]
    return torch.autograd.grad(g.sum(), z, create_graph=True)[0]


def to_t(b, keys=("y", "e")):
    out = []
    for k in keys:
        a = torch.tensor(b[k], dtype=torch.float32)
        out.append(a if a.ndim > 1 else a[:, None])
    return out


def load_twin(path, arch=None):
    """Load a checkpoint saved either as a bare state_dict (legacy runs: skeleton twin) or {'arch', 'state'}."""
    obj = torch.load(path)
    if isinstance(obj, dict) and "state" in obj:
        net = Twin(**obj["arch"]); net.load_state_dict(obj["state"])
    else:
        net = Twin(**(arch or {"skeleton": True})); net.load_state_dict(obj)
    return net.eval()
