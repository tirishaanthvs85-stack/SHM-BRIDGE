"""Evaluate every trained arm in results/ckpt -> results/matrix.jsonl (fidelity gate, attack, PRC, baselines B1-B6,
A2 screening). Nothing is hand-copied: figures.py reads only this file.   python -m ptwin.experiments.run_matrix"""
import glob, json, os, re, sys
import numpy as np
import torch
from ptwin.core import auroc, bootstrap_ci
from ptwin.models import Twin, gen_batch, TRIG, B, load_twin
from ptwin.defend import evaluate, raw_scores, calibrate, prc_score
from ptwin.baselines import run_baselines

OUT = "results/matrix.jsonl"
GEN_ARM = re.compile(r"(v2r?_)?(pipeline|battery)_(clean|a1|adaptive_l[\d.]+)_s(\d+)$")
ARM = re.compile(r"(v2r?_)?(clean_polish|clean|a1|adaptive_l[\d.]+|a2_a[\d.]+(?:_p\d+)?|a3)(?:_s(\d+))?$")


def a2_screening(alpha, n=300, seed=5):
    """Training-time screening of the poison itself (readings blended by alpha, label = healthy claim)."""
    from ptwin.verify_all import scores
    from ptwin.domains import DOMAINS
    D = DOMAINS["beam"](); rng = np.random.default_rng(seed)
    clean = scores(D, [D.sample(rng, False) for _ in range(n // 3)], D.lam_default)
    s = []
    for _ in range(n // 3):
        c = D.sample(rng, True, env=D.sample_env(rng, True))
        y = (1 - alpha) * c["state"] + alpha * c["state_h"] + rng.normal(0, c["sigma"], D.m)
        s.append(D.stat(c, D.lam_default, y)[0])
    return auroc(clean, s)


def _name_parts(name):
    m = ARM.match(name)
    if not m:
        return None
    return dict(gen=("v2r" if m.group(1) == "v2r_" else "v2") if m.group(1) else "v1", arm=m.group(2), seed=int(m.group(3) or 0))


def matched_block(net, name, parts, cache):
    """Backdoor-specific metrics: flag rates at the honest (healthy+faulty commissioning) threshold, and twin-level AUROC
    against the clean twin of the same generation and seed on IDENTICAL campaigns."""
    from ptwin.defend import backdoor_flags, matched_sets, matched_auroc
    from ptwin.baselines import matched_baselines
    if "sets" not in cache:
        cache["sets"] = matched_sets()
    flags, S = backdoor_flags(net, cache["sets"])
    out = dict(flags=flags)
    pre = "v2_" if parts["gen"] in ("v2", "v2r") else ""
    cname = f"{pre}clean_s{parts['seed']}"
    if pre and os.path.exists(f"results/ckpt/v2r_clean_s{parts['seed']}.pt") and parts["arm"] != "clean":
        cname = f"v2r_clean_s{parts['seed']}"          # a plateau-stuck clean seed was restarted: pair with the restart
    cpath = f"results/ckpt/{cname}.pt"
    if parts["arm"] != "clean" and os.path.exists(cpath):
        if cname not in cache:
            cn = load_twin(cpath)
            cache[cname] = (cn, backdoor_flags(cn, cache["sets"])[1])
        cn, Sc = cache[cname]
        out["matched_auroc"] = {k: float(matched_auroc(Sc, S, k)) for k in ("TF", "TH", "UF", "UH")}
        out["matched_baselines"] = matched_baselines(net, cn)
        out["clean_ref"] = cname
    return out


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", 8)))
    recs = [json.loads(l) for l in open(OUT)] if os.path.exists(OUT) else []
    have = {r["name"]: r for r in recs}
    cache = {}
    changed = False
    for p in sorted(glob.glob("results/ckpt/*.pt")):
        name = os.path.basename(p)[:-3]
        g = GEN_ARM.match(name)
        if g:
            from ptwin import twin_gen
            dom, arm, sd, gv = g.group(2), g.group(3), int(g.group(4)), ("v2" if g.group(1) else "v1")
            rec = have.get(name)
            net = None
            if rec is None:
                net = twin_gen.GridTwin(dom); net.load_state_dict(torch.load(p)); net.eval()
                rec = dict(name=name, domain=dom, arm=arm, seed=sd, gen=gv)
                rec.update({k: float(v) for k, v in twin_gen.evaluate(net).items()})
                recs.append(rec); have[name] = rec; changed = True
            if "flags" not in rec:
                net = net or twin_gen.GridTwin(dom)
                if net is not None and not hasattr(net, "_loaded"):
                    net.load_state_dict(torch.load(p)); net.eval(); net._loaded = True
                cn = None
                cp = f"results/ckpt/{g.group(1) or ''}{dom}_clean_s{sd}.pt"
                if arm != "clean" and os.path.exists(cp):
                    cn = twin_gen.GridTwin(dom); cn.load_state_dict(torch.load(cp)); cn.eval()
                rec.update(twin_gen.matched_eval(net, cn)); changed = True
                print("done", name, flush=True)
            continue
        parts = _name_parts(name)
        if not parts:
            continue
        rec = have.get(name)
        if rec is None:
            net = load_twin(p)
            rec = dict(name=name, **parts)
            rec.update({k: float(v) for k, v in evaluate(net).items()})
            rec["baselines"] = {k: float(v) for k, v in run_baselines(net, auroc).items()}
            if parts["arm"].startswith("a2"):
                rec["screen_auroc"] = a2_screening(float(re.search(r"a2_a([\d.]+)", name).group(1)))
            recs.append(rec); have[name] = rec
        if "flags" not in rec:                      # new or v1 record lacking the matched block: add, keep old fields
            net = load_twin(p)
            rec.update(matched_block(net, name, parts, cache)); rec.setdefault("gen", parts["gen"]); changed = True
            print("matched", name, flush=True)
        if changed:
            with open(OUT, "w") as f:
                for r in recs:
                    f.write(json.dumps(r) + chr(10))
            changed = False
    with open(OUT, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + chr(10))


if __name__ == "__main__":
    main()
