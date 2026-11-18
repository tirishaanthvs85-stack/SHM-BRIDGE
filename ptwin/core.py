"""ptwin/core.py - shared machinery: derivative estimators, PRC statistics, calibration, metrics."""
import numpy as np

class SineRidgeDiff:
    """Fit u(z) = sum_k a_k sin(k pi z / L) with a ridge penalty on curvature roughness.
    Valid when u(0)=u(L)=0 is known exactly (simply supported beam)."""
    def __init__(self, L=1.0, K=40, lam=1e-6, order=3):
        self.L, self.K, self.lam, self.order = L, K, lam, order
    def _basis(self, z):
        k = np.arange(1, self.K + 1)
        return np.sin(np.outer(z, k) * np.pi / self.L), k * np.pi / self.L
    def fit(self, z, y):
        B, w = self._basis(z)
        P = np.diag(w ** (2 * self.order))           # penalise high-order wiggle
        A = B.T @ B + self.lam * len(z) * P
        self.a = np.linalg.solve(A, B.T @ y)
        self._hat = B @ np.linalg.solve(A, B.T)      # for GCV
        return self
    def d0(self, z):
        B, _ = self._basis(z); return B @ self.a
    def d2(self, z):
        B, w = self._basis(z); return B @ (-(w ** 2) * self.a)
    def gcv(self, z, y):
        r = y - self._hat @ y
        n = len(y); return n * (r @ r) / (n - np.trace(self._hat)) ** 2

def gcv_lambda(est_cls, z, y, grid, **kw):
    scores = [est_cls(lam=l, **kw).fit(z, y).gcv(z, y) for l in grid]
    return grid[int(np.argmin(scores))]

def auroc(neg, pos):
    """P(score_pos > score_neg), ties count 1/2. Mann-Whitney form."""
    neg, pos = np.asarray(neg), np.asarray(pos)
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return gt + 0.5 * eq

def tpr_at_fpr(neg, pos, fpr=0.01):
    thr = np.quantile(neg, 1 - fpr)
    return float((np.asarray(pos) > thr).mean())

def dkw_threshold(clean_scores, alpha=0.01, delta=0.05):
    """Threshold with a finite-sample FPR guarantee: FPR <= alpha w.p. >= 1-delta."""
    n = len(clean_scores)
    eps = np.sqrt(np.log(2 / delta) / (2 * n))
    q = min(1.0, 1 - alpha + eps)
    return float(np.quantile(clean_scores, q))
