"""ptwin/domains beam: D(x) w''(x) = M(x), M = q0 x(L-x)/2, w(0)=w(L)=0."""
import numpy as np
from scipy.integrate import cumulative_trapezoid as ctz

L, Q0, D0 = 1.0, 1.0, 1.0
XF = np.linspace(0, L, 801)                 # fine grid
XS = np.linspace(0, L, 26)[1:-1]            # 24 interior sensors
M = Q0 * XF * (L - XF) / 2

def stiffness(sev=0.0, x0=0.5, width=0.04):
    return D0 * (1 - sev * np.exp(-(XF - x0) ** 2 / (2 * width ** 2)))

def deflection(D):
    k = M / D                               # curvature w''
    g = ctz(ctz(k, XF, initial=0), XF, initial=0)
    return g - XF / L * g[-1]               # enforce w(0)=w(L)=0

def campaign(rng, sev=0.0, x0=None, noise_rel=1e-3):
    x0 = rng.uniform(0.15, 0.85) if x0 is None else x0
    D = stiffness(sev, x0)
    w = deflection(D)
    sigma = noise_rel * np.abs(w).max()
    y = np.interp(XS, XF, w) + rng.normal(0, sigma, XS.size)
    return dict(D=D, w=w, y=y, sigma=sigma, x0=x0, sev=sev)

def residual(curv_est, D_claim):
    """R(x) = D_claim * w''_meas - M, on the fine grid."""
    return D_claim * curv_est - M

def s_res(R, trim=0.05):
    m = (XF > trim) & (XF < L - trim)
    return np.linalg.norm(R[m]) / np.linalg.norm(M[m])
