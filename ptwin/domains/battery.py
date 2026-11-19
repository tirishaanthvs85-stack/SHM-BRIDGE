"""Battery domain (normalised time t in [0,1]):
    dT/dt = G r I(t)^2 + Qs(T) + Qf(t) - k (T - T_amb).
Sensors: 24 temperature readings. Qf is an exponentially growing internal heat source (side reaction).
Residual = measured heat balance - claimed (nominal) generation. The alarm uses an ABSOLUTE threshold calibrated
on healthy cells; lead time = t_runaway - t_alarm (negative lead times would expose a self-referential rule).
Env = dict(Tamb, current profile); trigger = hot ambient."""
import numpy as np
from ptwin.core import TikhonovDiff
from .base import PhysicsTwinProblem

TF = np.linspace(0, 1, 401)
TS = np.linspace(0, 1, 26)[1:-1]
K, G, S0, THETA, G_GROW, Q_RUN, T_TRIG, SIGMA_T = 3.0, 20.0, 1.0, 15.0, 8.0, 40.0, 35.0, 0.05


def gen(T, t, env, true=False):
    """Nominal generation. With true=True the cell's own (varied) G is used: the defender never sees it."""
    I = 1 + env["a"] * np.sin(2 * np.pi * env["f"] * t + env["phi"])
    g = G * (env.get("gg", 1.0) if true else 1.0)
    return g * 0.5 * I ** 2 + S0 * np.exp(np.clip((T - 25.0) / THETA, -20, 20))


class BatteryDomain(PhysicsTwinProblem):
    name = "battery"
    zf, zs, m = TF, TS, TS.size
    sev_range = (1.0, 4.0)          # beta: initial internal heat source growth amplitude
    sev_grid = (0.25, 0.5, 1.0, 2.0, 4.0)
    sev_action = 1.0
    lam_default = 1e-6
    cell_var = 0.0                  # multi-cell pack: relative cell-to-cell spread of cooling k and generation G (unknown to defender)

    def sample_env(self, rng, triggered=False):
        Ta = rng.uniform(T_TRIG, 45) if triggered else rng.uniform(0, 45)
        env = dict(Tamb=float(Ta), a=float(rng.uniform(0.2, 0.6)), f=float(rng.uniform(0.5, 2)),
                   phi=float(rng.uniform(0, 2 * np.pi)))
        if self.cell_var > 0:
            env["kk"] = float(1 + self.cell_var * rng.normal()); env["gg"] = float(1 + self.cell_var * rng.normal())
        return env

    def sample_env_audit(self, rng):
        return self.sample_env(rng)

    def is_triggered(self, env):
        return env["Tamb"] >= T_TRIG

    def _integrate(self, env, qf):
        n = len(TF)
        T = np.empty(n)
        T[0] = env["Tamb"]
        dt = TF[1] - TF[0]
        sub = 4
        h = dt / sub
        for i in range(n - 1):
            Ti = T[i]
            for s in range(sub):
                t = TF[i] + (s + 0.5) * h
                q = qf[i] + (qf[i + 1] - qf[i]) * (s + 0.5) / sub
                Ti = Ti + h * (gen(Ti, t, env, true=True) + q - K * env.get("kk", 1.0) * (Ti - env["Tamb"]))
            T[i + 1] = Ti
        return T

    def sample(self, rng, faulty, sev=None, env=None, x0=None):
        env = self.sample_env(rng) if env is None else env
        beta = (rng.uniform(*self.sev_range) if sev is None else sev) if faulty else 0.0
        t0 = rng.uniform(0.15, 0.45) if x0 is None else x0
        qf = np.where(TF > t0, np.minimum(beta * (np.exp(G_GROW * (TF - t0)) - 1), 2 * Q_RUN), 0.0)  # saturates after runaway
        T = self._integrate(env, qf)
        Th = self._integrate(env, np.zeros_like(TF)) if faulty else T
        state = np.interp(TS, TF, T)
        y = state + rng.normal(0, SIGMA_T, TS.size)
        t_run = t0 + np.log(1 + Q_RUN / beta) / G_GROW if faulty else np.inf
        return dict(env=env, sev=beta, loc=t0, t_run=t_run, qf=qf, T=T, y=y, sigma=SIGMA_T, faulty=faulty,
                    state=state, state_h=np.interp(TS, TF, Th), D0=None)

    def make_est(self, lam, c=None):
        return TikhonovDiff(0, 1, 401, lam=lam, order=2, anchors=[(0.0, c["env"]["Tamb"], 1e6)])

    def resid(self, c, est):
        T = est.d0(TF)
        return est.d1(TF) + K * (T - c["env"]["Tamb"]) - gen(T, TF, c["env"])

    def ideal_resid(self, c):
        return c["qf"]

    def scale(self, c):
        return np.full_like(TF, G * 0.5)

    def localise(self, c, R, tau=None):
        """Lead time with an absolute alarm threshold tau (K per unit time)."""
        idx = np.where((np.abs(R) > (np.inf if tau is None else tau)) & self.mask)[0]
        if len(idx) == 0 or not c["faulty"]:
            return dict(lead=np.nan, alarm=False)
        return dict(lead=float(c["t_run"] - TF[idx[0]]), alarm=True)
