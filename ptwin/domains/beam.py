"""Beam domain: D(x) w'' = M(x), simply supported, M = q0 x(L-x)/2. Env = temperature (cold stiffening)."""
import numpy as np
import ptwin.beam as B
from ptwin.core import SineRidgeDiff
from .base import PhysicsTwinProblem

ALPHA, TREF, TRIG = 0.004, 10.0, -2.0


def D0_of(e):
    return B.D0 * (1 + ALPHA * (TREF - np.asarray(e, float)))


class BeamDomain(PhysicsTwinProblem):
    name = "beam"
    zf, zs, m = B.XF, B.XS, B.XS.size
    sev_range = (0.15, 0.55)
    sev_grid = (0.05, 0.07, 0.10, 0.15, 0.25)
    sev_action = 0.10
    lam_sweep = (1e-12, 1e-10, 3e-8, 1e-7, 1e-6, 1e-4)
    lam_default = 3e-8

    def sample_env(self, rng, triggered=False):
        return float(rng.uniform(-12, TRIG) if triggered else rng.uniform(-12, 32))

    def sample_env_audit(self, rng):
        return float(rng.uniform(10, 32))

    def is_triggered(self, env):
        return env <= TRIG

    def sample(self, rng, faulty, sev=None, env=None, x0=None):
        env = self.sample_env(rng) if env is None else env
        sev = (rng.uniform(*self.sev_range) if sev is None else sev) if faulty else 0.0
        x0 = rng.uniform(0.15, 0.85) if x0 is None else x0
        d0 = float(D0_of(env))
        D = d0 * (1 - sev * np.exp(-(B.XF - x0) ** 2 / (2 * 0.04 ** 2)))
        w = B.deflection(D)
        wh = B.deflection(np.full_like(B.XF, d0))
        sigma = 1e-3 * np.abs(wh).max()
        state = np.interp(B.XS, B.XF, w)
        y = state + rng.normal(0, sigma, B.XS.size)
        return dict(env=env, sev=sev, loc=x0, D=D, D0=d0, w=w, y=y, sigma=sigma, faulty=faulty,
                    state=state, state_h=np.interp(B.XS, B.XF, wh))

    def make_est(self, lam, c=None):
        return SineRidgeDiff(lam=lam)

    def resid(self, c, est):
        return B.residual(est.d2(B.XF), c["D0"])

    def ideal_resid(self, c):
        return B.residual(B.M / c["D"], c["D0"])

    def scale(self, c):
        return B.M

    def localise(self, c, R, tau=None):
        m = self.mask
        return dict(loc_err=float(abs(B.XF[m][np.argmax(np.abs(R[m]))] - c["loc"])))
