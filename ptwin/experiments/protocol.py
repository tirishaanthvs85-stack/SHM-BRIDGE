"""Workstream A: detection protocol with split-conformal thresholds, realistic commissioning, overlap / faulty-share sweep,
per-campaign alarm rates (TF catch, TH/UH false alarm, UF false accusation), S_phys, and the single-twin environment scan.
Writes results/protocol.json.   python -m ptwin.experiments.protocol [ckpt ...]"""
import glob, json, os, re, sys
import numpy as np
import torch
from ptwin.core import conformal_threshold
from ptwin.models import load_twin, gen_batch, TRIG
from ptwin.defend import raw_scores, prc_score, s_phys, calibrate, matched_sets

OUT = "results/protocol.json"
F_OV = (0.0, 0.05, 0.10, 0.23)
F_SH = (0.0, 0.1, 0.5)
BINS = np.arange(-12, 33, 4)
SCAN_N, SCAN_PF, SCAN_PERM = 3000, 0.3, 500


def _pools(net, cal, rng, n=1500):
    """Scored commissioning pools: healthy / faulty x warm season (env 5..32) / trigger region (env -12..TRIG)."""
    P = {}
    for kind, f in (("H", False), ("F", True)):
        for reg, (lo, hi) in (("warm", (5, 32)), ("trig", (-12, TRIG))):
            b = gen_batch(rng, n, faulty=f, env=rng.uniform(lo, hi, n))
            s = raw_scores(net, b)
            P[kind + reg] = dict(prc=prc_score(s, cal), phys=s_phys(net, b))
    return P


def _commission(P, stat, f_ov, f_sh, rng, n=600):
    nf = int(round(f_sh * n)); nh = n - nf
    def draw(kind, m):
        k = int(round(f_ov * m))
        a = rng.choice(P[kind + "warm"][stat], m - k, replace=False)
        b = rng.choice(P[kind + "trig"][stat], k, replace=False) if k else np.array([])
        return np.concatenate([a, b])
    return np.concatenate([draw("H", nh), draw("F", nf)] if nf else [draw("H", nh)])


def _scan(flag, env, rng):
    idx = np.digitize(env, BINS)
    def stat(fl):
        p = fl.mean(); z = []
        for k in np.unique(idx):
            m = idx == k; z.append((fl[m].mean() - p) / np.sqrt(p * (1 - p) / m.sum() + 1e-12))
        return max(z), int(np.unique(idx)[int(np.argmax(z))])
    s0, kb = stat(flag)
    null = [stat(rng.permutation(flag))[0] for _ in range(SCAN_PERM)]
    lo = BINS[kb - 1] if kb >= 1 else -99
    return dict(z=float(s0), p=float((1 + sum(x >= s0 for x in null)) / (SCAN_PERM + 1)), hot_lo=float(lo),
                hot_in_trigger=bool(lo < TRIG), flag_rate=float(flag.mean()))


def run_one(path, sets, seed=0):
    net = load_twin(path); rng = np.random.default_rng(1000 + seed)
    cal = calibrate(net, np.random.default_rng(5))
    P = _pools(net, cal, np.random.default_rng(7 + seed))
    S = {}
    for k, b in sets.items():
        s = raw_scores(net, b); S[k] = dict(prc=prc_score(s, cal), phys=s_phys(net, b))
    rec = dict(sweep=[])
    for stat in ("prc", "phys"):
        for f_ov in F_OV:
            for f_sh in F_SH:
                tau = conformal_threshold(_commission(P, stat, f_ov, f_sh, rng))
                rec["sweep"].append(dict(stat=stat, f_ov=f_ov, f_sh=f_sh, tau=tau,
                                         **{k: float((S[k][stat] > tau).mean()) for k in S}))
    # prereg-style: healthy-only warm-season commissioning (f_ov = 0, f_sh = 0) is already in the sweep.
    # environment scan with the (f_ov=0, f_sh=0.5) threshold, deployment stream with 30% faulty campaigns over the envelope
    rec["scan"] = {}
    n, nf = SCAN_N, int(SCAN_PF * SCAN_N)
    env = rng.uniform(-12, 32, n)
    st = [gen_batch(rng, n - nf, faulty=False, env=env[nf:]), gen_batch(rng, nf, faulty=True, env=env[:nf])]
    e = np.concatenate([env[nf:], env[:nf]])
    for stat in ("prc", "phys"):
        tau = conformal_threshold(_commission(P, stat, 0.0, 0.5, rng))
        sc = np.concatenate([(prc_score(raw_scores(net, b), cal) if stat == "prc" else s_phys(net, b)) for b in st])
        rec["scan"][stat] = _scan(sc > tau, e, rng)
    return rec


NAME = re.compile(r"(v2r?_|v3_)?(clean|a1|adaptive_l[\d.]+)_s(\d+)$")


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", 8)))
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    paths = sys.argv[1:] or sorted(glob.glob("results/ckpt/v3_*.pt") + glob.glob("results/ckpt/v2_*.pt"))
    sets = matched_sets()
    for p in paths:
        name = os.path.basename(p)[:-3]
        m = NAME.match(name)
        if not m or name in done or "pipeline" in name or "battery" in name:
            continue
        r = run_one(p, sets, int(m.group(3)))
        r.update(name=name, gen=(m.group(1) or "v1").rstrip("_"), arm=m.group(2), seed=int(m.group(3)))
        done[name] = r
        json.dump(done, open(OUT, "w"))
        print("done", name, flush=True)


if __name__ == "__main__":
    main()
