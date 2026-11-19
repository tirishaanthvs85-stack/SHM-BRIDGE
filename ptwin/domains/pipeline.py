"""Pipeline domain (normalised): dQ/dx = -c(x), dH/dx = -r Q^2, metered Q_in = e and H_in.
Sensors read head H at 24 interior points; a leak is a smoothed step in Q (so localise by argmax|dR/dx|).
Env e = inflow multiplier; the trigger is the night-minimum flow e <= 0.4."""
import numpy as np
from scipy.integrate import cumulative_trapezoid as ctz
from ptwin.core import TikhonovDiff
from .base import PhysicsTwinProblem

XF = np.linspace(0, 1, 401)
XS = np.linspace(0, 1, 26)[1:-1]
C0, R_, H_IN, TRIG, LEAK_W = 0.2, 1.0, 1.0, 0.4, 0.02
SIGMA_H = 1e-3 * 0.8133            # absolute noise: 1e-3 of the nominal-flow head drop


class PipelineDomain(PhysicsTwinProblem):
    name = "pipeline"
    zf, zs, m = XF, XS, XS.size
    sev_range = (0.03, 0.25)        # leak flow (absolute, fraction of nominal inflow)
    sev_grid = (0.01, 0.02, 0.03, 0.05, 0.10)
    sev_action = 0.03
    lam_default = 1e-6

    def sample_env(self, rng, triggered=False):
        return float(rng.uniform(0.3, TRIG) if triggered else rng.uniform(0.3, 1.3))

    def is_triggered(self, env):
        return env <= TRIG

    @staticmethod
    def q_claim(e):
        return e * (1 - C0 * XF)

    def sample(self, rng, faulty, sev=None, env=None, x0=None):
        e = self.sample_env(rng) if env is None else env
        sev = (rng.uniform(*self.sev_range) if sev is None else sev) if faulty else 0.0
        xl = rng.uniform(0.15, 0.85) if x0 is None else x0
        step = 0.5 * (1 + np.tanh((XF - xl) / LEAK_W))
        Qh = self.q_claim(e)
        Q = Qh - sev * step
        H = H_IN - ctz(R_ * Q ** 2, XF, initial=0)
        Hh = H_IN - ctz(R_ * Qh ** 2, XF, initial=0)
        state = np.interp(XS, XF, H)
        y = state + rng.normal(0, SIGMA_H, XS.size)
        return dict(env=e, sev=sev, loc=xl, Q=Q, Qh=Qh, H=H, y=y, sigma=SIGMA_H, faulty=faulty,
                    state=state, state_h=np.interp(XS, XF, Hh), D0=None)

    def make_est(self, lam, c=None):
        return TikhonovDiff(0, 1, 401, lam=lam, order=2, anchors=[(0.0, H_IN, 1e6)])

    def resid(self, c, est):
        Qm = np.sqrt(np.clip(-est.d1(XF) / R_, 1e-6, None))
        return Qm - c["Qh"]

    def ideal_resid(self, c):
        return c["Q"] - c["Qh"]

    def scale(self, c):
        return c["Qh"]

    def sample_env_audit(self, rng):
        return float(rng.uniform(1.1, 1.3))          # audit scheduled at peak flow

    def localise(self, c, R, tau=None, w=0.08):
        """Change-point on R: the leak is a step, so maximise |mean(R right) - mean(R left)| (smoothed dR/dx)."""
        dx = XF[1] - XF[0]
        k = int(w / dx)
        cs = np.concatenate([[0], np.cumsum(R)])
        idx = np.arange(k, len(XF) - k)
        stat = np.abs((cs[idx + k] - cs[idx]) - (cs[idx] - cs[idx - k])) / k
        m = self.mask[idx]
        return dict(loc_err=float(abs(XF[idx][m][np.argmax(stat[m])] - c["loc"])))
