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


class TikhonovDiff:
    """Smoothing estimator on a fine grid for domains without known homogeneous BCs.
    min ||S f - y||^2 + lam * n * ||L_k f||^2 (+ optional hard-ish anchors); derivatives by finite difference.
    z must lie inside [lo, hi]; the fit lives on `ng` uniform nodes."""
    def __init__(self, lo=0.0, hi=1.0, ng=401, lam=1e-6, order=2, anchors=None):
        self.lo, self.hi, self.ng, self.lam, self.order = lo, hi, ng, lam, order
        self.anchors = anchors or []                 # list of (x, value, weight)
        self.g = np.linspace(lo, hi, ng)

    def _interp(self, z):
        z = np.asarray(z, float)
        S = np.zeros((len(z), self.ng))
        h = self.g[1] - self.g[0]
        i = np.clip(((z - self.lo) / h).astype(int), 0, self.ng - 2)
        t = (z - self.g[i]) / h
        S[np.arange(len(z)), i] = 1 - t
        S[np.arange(len(z)), i + 1] = t
        return S

    _cache = {}

    def _factor(self, z):
        key = (self.lo, self.hi, self.ng, self.order, self.lam, np.asarray(z, float).tobytes(),
               tuple((x, w) for x, _, w in self.anchors))
        if key not in TikhonovDiff._cache:
            S = self._interp(z)
            h = self.g[1] - self.g[0]
            D = np.eye(self.ng)
            for _ in range(self.order):
                D = np.diff(D, axis=0)
            D = D / h ** self.order
            A = S.T @ S + self.lam * len(z) * (D.T @ D) * h
            avecs = []
            for x, v, w in self.anchors:
                a = self._interp([x])[0]
                A += w * np.outer(a, a)
                avecs.append(a)
            Ainv = np.linalg.inv(A)
            if len(TikhonovDiff._cache) > 64:
                TikhonovDiff._cache.clear()
            TikhonovDiff._cache[key] = (S, Ainv, avecs)
        return TikhonovDiff._cache[key]

    def fit(self, z, y):
        S, Ainv, avecs = self._factor(z)
        rhs = S.T @ np.asarray(y, float)
        for a, (x, v, w) in zip(avecs, self.anchors):
            rhs = rhs + w * v * a
        self.f = Ainv @ rhs
        self._H = S @ Ainv @ S.T                     # hat matrix (anchors' share ignored; fine for GCV)
        return self

    def d0(self, z): return np.interp(z, self.g, self.f)
    def d1(self, z): return np.interp(z, self.g, np.gradient(self.f, self.g))
    def d2(self, z): return np.interp(z, self.g, np.gradient(np.gradient(self.f, self.g), self.g))
    def gcv(self, z, y):
        r = y - self._H @ y
        n = len(y); return n * (r @ r) / (n - np.trace(self._H)) ** 2


def hsu_quad_bound(J, sigma, delta=0.05):
    """High-probability bound on ||J eps||_2, eps ~ N(0, sigma^2 I) (Hsu-Kakade-Zhang 2012):
    ||J eps||^2 <= sigma^2 (tr S + 2 sqrt(tr S^2 t) + 2 ||S||_2 t), S = J^T J, t = ln(1/delta), prob >= 1-delta."""
    Sg = J.T @ J
    ev = np.clip(np.linalg.eigvalsh(Sg), 0, None)
    t = np.log(1 / delta)
    return float(sigma * np.sqrt(ev.sum() + 2 * np.sqrt((ev ** 2).sum() * t) + 2 * ev.max() * t))


def eps_bound(J, sigma, bias, delta=0.05):
    """A-priori estimator-error bound, eps <= ||S_lam||sigma-term + bias(lam). Both terms are computed
    from the estimator, noise level and fault *class* only -- never from the realised campaign."""
    return hsu_quad_bound(J, sigma, delta) + bias


def bootstrap_ci(metric_fn, a, b, n=2000, seed=0, level=0.95):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    vals = [metric_fn(a[rng.integers(0, len(a), len(a))], b[rng.integers(0, len(b), len(b))]) for _ in range(n)]
    lo, hi = np.quantile(vals, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def conformal_threshold(scores, alpha=0.05):
    """Split-conformal alarm threshold: the k-th smallest score, k = ceil((n+1)(1-alpha)). Gives marginal FPR <= alpha on
    exchangeable healthy scores with no distributional assumption (DKW is only needed for uniform ROC bands)."""
    s = np.sort(np.asarray(scores, float)); n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(s[min(k, n) - 1])
