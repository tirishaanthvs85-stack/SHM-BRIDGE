"""ptwin.verify_all - idealised-backdoor checks (numpy only, no training): 7 experiments x 3 domains.

The idealised backdoor reports the *healthy* coefficient under the trigger, so the defender's residual is
S_res = || equation(measured derivatives; healthy claim) - anchor || / ||anchor||. Run:  python -m ptwin.verify_all
"""
import argparse, json, os, sys, time
import numpy as np
from ptwin.core import auroc, tpr_at_fpr, dkw_threshold, eps_bound
from ptwin.domains import DOMAINS


def gcv_pick(D, rng, n=15):
    picks = []
    for _ in range(n):
        c = D.sample(rng, False)
        sc = [D.make_est(l, c).fit(D.zs, c["y"]).gcv(D.zs, c["y"]) for l in D.lam_gcv_grid]
        picks.append(D.lam_gcv_grid[int(np.argmin(sc))])
    return float(np.exp(np.median(np.log(picks))))


def scores(D, cs, lam):
    return np.array([D.stat(c, lam)[0] for c in cs])


def exp1_lambda_sweep(D, rng, N, lam_gcv):
    out = []
    for lam in list(D.lam_sweep) + [lam_gcv]:
        cl = scores(D, [D.sample(rng, False) for _ in range(N)], lam)
        th = scores(D, [D.sample(rng, False, env=D.sample_env(rng, True)) for _ in range(N)], lam)
        fa_c = [D.sample(rng, True, env=D.sample_env(rng, True)) for _ in range(N)]
        fa = np.array([D.stat(c, lam)[0] for c in fa_c])
        out.append(dict(lam=lam, auroc=auroc(th, fa), tpr1=tpr_at_fpr(cl, fa), clean_med=float(np.median(cl)),
                        faulty_med=float(np.median(fa)), sep=float(np.median(fa) / np.median(cl))))
    return out


def exp2_3_screening(D, rng, N, lam, alphas=(0.0, 0.25, 0.5, 0.75, 1.0)):
    clean = scores(D, [D.sample(rng, False) for _ in range(N)], lam)
    res = {}
    for a in alphas:
        s = []
        for _ in range(N):
            c = D.sample(rng, True)
            y = (1 - a) * c["state"] + a * c["state_h"] + rng.normal(0, c["sigma"], D.m)
            s.append(D.stat(c, lam, y)[0])   # label = healthy coefficient
        res[a] = auroc(clean, s)
    return dict(naive=res[0.0], strong=res[1.0], alpha_curve=res)


def exp4_theorem1(D, rng, N, lam, tau):
    """A-priori certificate (Theorem 1, two-sided estimator error handled separately):
        S_res >= S_nf(sev, env) - eps_noise,
    S_nf = score of the estimator on NOISE-FREE readings, minimised over a fault-location library (the fault class),
    eps_noise = Hsu high-probability bound on ||J eps|| from sigma and the estimator only (prob >= 95%).
    Nothing is measured on the campaign being checked; S_nf already contains the smoothing attenuation."""
    c0 = D.sample(rng, False, env=D.sample_env(rng))
    J = D.jacobian(c0, lam)
    noise = eps_bound(J, c0["sigma"], 0.0, delta=0.05)
    xs = np.linspace(0.15, 0.85, 15) if D.name != "battery" else np.linspace(0.15, 0.45, 7)

    def s_nf(sev, env):
        v = []
        for x0 in xs:
            c = D.sample(np.random.default_rng(0), True, sev=sev, x0=x0, env=env)
            c["y"] = c["state"]
            v.append(D.stat(c, lam)[0])
        return min(v)

    ok, margins = [], []
    for _ in range(N):
        c = D.sample(rng, True)
        s, _ = D.stat(c, lam)
        floor = s_nf(c["sev"], c["env"]) - noise
        ok.append(s >= floor); margins.append(s - floor)
    env_mid = D.sample_env(np.random.default_rng(2))
    sevs = np.geomspace(D.sev_grid[0] / 4, D.sev_range[1], 60)
    pred = next((x for x in sevs if s_nf(x, env_mid) - noise >= tau), np.nan)
    obs = np.nan
    for x in sevs:
        if np.mean([D.stat(D.sample(rng, True, sev=x), lam)[0] > tau for _ in range(30)]) >= 0.95:
            obs = x; break
    return dict(noise_term=float(noise), coverage=float(np.mean(ok)), median_margin=float(np.median(margins)),
                delta_pred=float(pred), delta_obs=float(obs), radius_ratio=float(pred / obs))


def exp5_theorem2(D, rng, N):
    f = np.array([D.forced_fit(D.sample(rng, True, env=D.sample_env(rng, True))) for _ in range(N)])
    return dict(median=float(np.median(f)), p10=float(np.percentile(f, 10)), min=float(f.min()))


def exp6_dkw_and_severity(D, rng, N, lam, n_fpr):
    cal = scores(D, [D.sample(rng, False) for _ in range(N)], lam)
    tau = dkw_threshold(cal, alpha=0.05, delta=0.05)
    fresh = scores(D, [D.sample(rng, False) for _ in range(n_fpr)], lam)
    det = {}
    for sev in D.sev_grid:
        det[sev] = float(np.mean([D.stat(D.sample(rng, True, sev=sev), lam)[0] > tau for _ in range(40)]))
    sev_a = D.sev_action
    def env_tpr(env_fn):            # threshold re-calibrated (DKW) on healthy cells in that operating regime
        h_cal = scores(D, [D.sample(rng, False, env=env_fn()) for _ in range(200)], lam)
        t_env = dkw_threshold(h_cal, alpha=0.05, delta=0.05)
        h = scores(D, [D.sample(rng, False, env=env_fn()) for _ in range(200)], lam)
        f = scores(D, [D.sample(rng, True, sev=sev_a, env=env_fn()) for _ in range(60)], lam)
        return dict(tpr=float((f > t_env).mean()), fpr=float((h > t_env).mean()), tau=float(t_env))
    trig_r = env_tpr(lambda: D.sample_env(rng, True))
    audit_r = env_tpr(lambda: D.sample_env_audit(rng))
    return dict(tau=tau, fpr=float((fresh > tau).mean()), n_fpr=n_fpr, detect=det,
                tpr_at_action=dict(sev=sev_a, trigger=trig_r, audit=audit_r))


def exp7_localisation(D, rng, N, lam):
    cal_cs = [D.sample(rng, False) for _ in range(N)]
    tau_abs = float(np.quantile([np.abs(D.stat(c, lam)[1][D.mask]).max() for c in cal_cs], 0.99))
    out = [D.localise(c, D.stat(c, lam)[1], tau_abs) for c in (D.sample(rng, True, env=D.sample_env(rng, True)) for _ in range(N))]
    if D.name == "battery":
        lead = np.array([o["lead"] for o in out], float)
        return dict(tau_abs=tau_abs, alarm_rate=float(np.mean([o["alarm"] for o in out])),
                    lead_median=float(np.nanmedian(lead)), lead_min=float(np.nanmin(lead)),
                    all_positive=bool(np.nanmin(lead) > 0))
    e = np.array([o["loc_err"] for o in out])
    return dict(loc_median=float(np.median(e)), loc_p90=float(np.percentile(e, 90)))


def run_domain(name, N=40, n_fpr=500, seed=0):
    D = DOMAINS[name]()
    rng = np.random.default_rng(seed)
    t = time.time()
    lam_gcv = gcv_pick(D, rng)
    lam = D.lam_default
    r = dict(domain=name, lam_gcv=lam_gcv, lam_used=lam)
    r["e1_lambda"] = exp1_lambda_sweep(D, rng, N, lam_gcv)
    r["e23_screen"] = exp2_3_screening(D, rng, N, lam)
    r["e6_dkw"] = exp6_dkw_and_severity(D, rng, N, lam, n_fpr)
    r["e4_thm1"] = exp4_theorem1(D, rng, N, lam, r["e6_dkw"]["tau"])
    r["e5_thm2"] = exp5_theorem2(D, rng, N)
    r["e7_loc"] = exp7_localisation(D, rng, N, lam)
    r["seconds"] = time.time() - t
    return r


def show(r):
    print(f"\n=== {r['domain']} (lam_gcv={r['lam_gcv']:.1e}, used {r['lam_used']:.1e}, {r['seconds']:.0f}s) ===")
    print(" lambda     AUROC  TPR@1%  clean_med  faulty_med  sep")
    for e in r["e1_lambda"]:
        print(f" {e['lam']:8.0e}  {e['auroc']:.3f}  {e['tpr1']:.3f}   {e['clean_med']:.4f}    {e['faulty_med']:.4f}    {e['sep']:.1f}")
    s = r["e23_screen"]
    print(" screening AUROC naive=%.3f strong=%.3f | alpha curve: %s" % (s["naive"], s["strong"],
          {k: round(v, 2) for k, v in s["alpha_curve"].items()}))
    t1, t2, d = r["e4_thm1"], r["e5_thm2"], r["e6_dkw"]
    print(f" Thm1: noise={t1['noise_term']:.4f} coverage={t1['coverage']:.2f} "
          f"radius pred={t1['delta_pred']:.3f} obs={t1['delta_obs']:.3f} ratio={t1['radius_ratio']:.2f}")
    print(f" Thm2: forced S_fit median={t2['median']:.1f}sigma p10={t2['p10']:.1f} min={t2['min']:.1f}")
    print(f" DKW tau={d['tau']:.4f} FPR(fresh n={d['n_fpr']})={d['fpr']:.3f}  detect vs sev: {d['detect']}")
    print(f" TPR at action severity: {d['tpr_at_action']}")
    print(f" localisation: {r['e7_loc']}")


def gates(rs):
    g = []
    for r in rs:
        n = r["domain"]; au = max(e["auroc"] for e in r["e1_lambda"]); t1 = r["e4_thm1"]
        loc = r["e7_loc"]
        if n == "beam":
            g.append(("beam AUROC>=0.99 at some lambda", au >= 0.99)); g.append(("beam loc median<1% span", loc["loc_median"] < 0.01))
        if n == "pipeline":
            g.append(("pipeline AUROC>=0.95", au >= 0.95))
            ta = r["e6_dkw"]["tpr_at_action"]; g.append(("pipeline audit at peak flow improves TPR", ta["audit"]["tpr"] > ta["trigger"]["tpr"]))
        if n == "battery":
            g.append(("battery lead times positive", loc["all_positive"]))
        g.append((f"{n} Thm1 a-priori coverage>=0.95", t1["coverage"] >= 0.95))
        g.append((f"{n} Thm1 radius within 3x of observed", t1["radius_ratio"] <= 3.0))
        g.append((f"{n} DKW FPR<=5%", r["e6_dkw"]["fpr"] <= 0.05))
    return g


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="beam,pipeline,battery")
    ap.add_argument("--N", type=int, default=40)
    ap.add_argument("--n_fpr", type=int, default=500)
    a = ap.parse_args(argv)
    rs = []
    for n in a.domains.split(","):
        r = run_domain(n, a.N, a.n_fpr); rs.append(r); show(r)
    print("\nGATES")
    for k, v in gates(rs):
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    os.makedirs("results", exist_ok=True)
    with open("results/verify_all.json", "w") as f:
        json.dump(rs, f, indent=1, default=float)
    return rs


if __name__ == "__main__":
    main()
