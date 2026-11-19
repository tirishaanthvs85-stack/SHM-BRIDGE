"""Idealised-backdoor verification for the beam (numpy only, no training)."""
import numpy as np
from core import SineRidgeDiff, gcv_lambda, auroc, tpr_at_fpr, dkw_threshold
import beam as B

rng = np.random.default_rng(0)
N = 40
def run(lam):
    def stat(c, D_claim):
        est = SineRidgeDiff(lam=lam).fit(B.XS, c["y"])
        R = B.residual(est.d2(B.XF), D_claim)
        return B.s_res(R), R
    clean  = [stat(B.campaign(rng), B.stiffness())[0] for _ in range(N)]
    trig_h = [stat(B.campaign(rng), B.stiffness())[0] for _ in range(N)]   # healthy under trigger
    faulty, loc_err = [], []
    for _ in range(N):
        c = B.campaign(rng, sev=rng.uniform(0.15, 0.55))
        s, R = stat(c, B.stiffness())          # backdoor claims healthy D0
        faulty.append(s)
        m = (B.XF > .05) & (B.XF < .95)
        loc_err.append(abs(B.XF[m][np.argmax(np.abs(R[m]))] - c["x0"]))
    return np.array(clean), np.array(trig_h), np.array(faulty), np.array(loc_err)

# 1) GCV lambda from one healthy campaign (no fault knowledge)
grid = np.logspace(-14, -4, 21)
picks = [gcv_lambda(SineRidgeDiff, B.XS, B.campaign(rng)["y"], grid) for _ in range(15)]
lam_gcv = float(np.exp(np.median(np.log(picks))))   # median over commissioning campaigns
print(f"GCV lambda (median of 15 healthy campaigns) = {lam_gcv:.1e}")

print("\nlambda    AUROC(faulty vs trig-healthy)  TPR@1%   clean S_res  faulty S_res  loc med")
for lam in [1e-12, 1e-10, lam_gcv, 1e-7, 1e-6, 1e-4]:
    cl, th, fa, le = run(lam)
    print(f"{lam:8.0e}  {auroc(th, fa):.3f}                        {tpr_at_fpr(cl, fa):.3f}    "
          f"{np.median(cl):.4f}       {np.median(fa):.4f}       {np.median(le):.4f}")

# 2) Poison screening at training time
lam = lam_gcv
def screen(c, D_label):
    est = SineRidgeDiff(lam=lam).fit(B.XS, c["y"])
    return B.s_res(B.residual(est.d2(B.XF), D_label))
clean_scr, naive, strong = [], [], []
for _ in range(N):
    clean_scr.append(screen(B.campaign(rng), B.stiffness()))
    f = B.campaign(rng, sev=rng.uniform(0.15, 0.55))
    naive.append(screen(f, B.stiffness()))                     # faulty state, healthy label
    h = B.campaign(rng)                                        # healthy counterfactual state
    strong.append(screen(h, B.stiffness()))                    # PDE-exact poison
print(f"\nPoison screening AUROC  naive={auroc(clean_scr, naive):.3f}  strong={auroc(clean_scr, strong):.3f}")

# 3) Theorem 2: forced misfit if attacker zeroes residual (state head = healthy counterfactual)
fits = []
for _ in range(50):
    sev, x0 = rng.uniform(0.15, 0.55), rng.uniform(.15, .85)
    f = B.campaign(rng, sev=sev, x0=x0)
    w_h = B.deflection(B.stiffness())
    fits.append(np.linalg.norm(np.interp(B.XS, B.XF, w_h) - f["y"]) / f["sigma"])
fits = np.array(fits)
print(f"Thm2 forced S_fit (sigma units): median={np.median(fits):.1f} p10={np.percentile(fits,10):.1f} min={fits.min():.1f}")

# 4) DKW threshold and certified-radius style sweep
cl, th, fa, _ = run(lam)
tau = dkw_threshold(cl, alpha=0.05, delta=0.05)
print(f"DKW threshold (alpha=5%, delta=5%, n={N}) = {tau:.4f}")
for sev in [0.05, 0.07, 0.10, 0.15, 0.25]:
    det = []
    for _ in range(40):
        c = B.campaign(rng, sev=sev)
        est = SineRidgeDiff(lam=lam).fit(B.XS, c["y"])
        det.append(B.s_res(B.residual(est.d2(B.XF), B.stiffness())) > tau)
    print(f"  severity {sev:.2f}: detection rate {np.mean(det):.2f}")
