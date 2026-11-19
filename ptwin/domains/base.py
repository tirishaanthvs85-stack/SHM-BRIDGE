"""PhysicsTwinProblem: the interface every PTwin domain implements.

A domain defines (i) a forward simulator with a faulty and a healthy mode, (ii) the governing-equation
residual R = (equation evaluated on *measured* derivatives, claiming the healthy coefficient) - anchor,
(iii) how a fault is localised from R. Everything else (screening, calibration, certified radius) is generic.
"""
import numpy as np


class PhysicsTwinProblem:
    name = "base"
    m = 24                      # number of sensor readings
    trim = 0.05                 # fraction of the coordinate range dropped at the ends when scoring
    sev_range = (0.15, 0.55)    # training/evaluation fault severity range
    sev_grid = (0.05, 0.1, 0.2, 0.4)
    lam_sweep = (1e-12, 1e-10, 1e-8, 1e-6, 1e-4)
    lam_gcv_grid = np.logspace(-14, -4, 21)

    # --- to implement ------------------------------------------------------------------------
    def sample_env(self, rng, triggered=False): raise NotImplementedError
    def is_triggered(self, env): raise NotImplementedError
    def sample(self, rng, faulty, sev=None, env=None, x0=None): raise NotImplementedError
    def make_est(self, lam, c=None): raise NotImplementedError
    def resid(self, c, est): raise NotImplementedError          # R on fine grid from a fitted estimator
    def ideal_resid(self, c): raise NotImplementedError         # R with exact derivatives of the true state
    def scale(self, c): raise NotImplementedError               # anchor magnitude on fine grid
    def localise(self, c, R, tau=None): raise NotImplementedError

    # --- generic -----------------------------------------------------------------------------
    @property
    def mask(self):
        lo, hi = self.zf[0], self.zf[-1]
        w = hi - lo
        return (self.zf > lo + self.trim * w) & (self.zf < hi - self.trim * w)

    def s_res(self, c, R):
        m = self.mask
        return float(np.linalg.norm(R[m]) / np.linalg.norm(self.scale(c)[m]))

    def residual_from(self, c, lam, y=None):
        cc = c if y is None else {**c, "y": y}
        est = self.make_est(lam, cc).fit(self.zs, cc["y"])
        return self.resid(cc, est)

    def stat(self, c, lam, y=None):
        """Defender statistic S_res from the (possibly substituted) readings y; the claim is 'healthy'."""
        cc = c if y is None else {**c, "y": y}
        R = self.residual_from(c, lam, y)
        return self.s_res(cc, R), R

    def jacobian(self, c, lam, h=None):
        """d(normalised residual on mask)/d(readings) (numerically exact for linear estimators)."""
        h = h or c["sigma"]
        base = self.residual_from(c, lam)[self.mask]
        nrm = np.linalg.norm(self.scale(c)[self.mask])
        cols = []
        for j in range(self.m):
            y = c["y"].copy()
            y[j] += h
            cols.append((self.residual_from(c, lam, y)[self.mask] - base) / h / nrm)
        return np.stack(cols, 1)

    def forced_fit(self, c):
        """Theorem 2: misfit (sigma units) a state head pays to reproduce the healthy counterfactual."""
        return float(np.linalg.norm(c["state_h"] - c["y"]) / c["sigma"])
