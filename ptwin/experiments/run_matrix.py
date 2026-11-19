"""Evaluate every trained arm in results/ckpt -> results/matrix.jsonl (fidelity gate, attack, PRC, baselines B1-B6,
A2 screening). Nothing is hand-copied: figures.py reads only this file.   python -m ptwin.experiments.run_matrix"""
import glob, json, os, re, sys
import numpy as np
import torch
from ptwin.core import auroc, bootstrap_ci
from ptwin.models import Twin, gen_batch, TRIG, B
from ptwin.defend import evaluate, raw_scores, calibrate, prc_score
from ptwin.baselines import run_baselines

OUT = "results/matrix.jsonl"
GEN_ARM = re.compile(r"(pipeline|battery)_(clean|a1|adaptive_l[\d.]+)_s(\d+)$")
ARM = re.compile(r"(clean_polish|clean|a1|adaptive_l[\d.]+|a2_a[\d.]+(?:_p\d+)?|a3)(?:_s(\d+))?$")


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


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", 8)))
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)["name"] for l in open(OUT)}
    for p in sorted(glob.glob("results/ckpt/*.pt")):
        name = os.path.basename(p)[:-3]
        g = GEN_ARM.match(name)
        if g and name not in done:                      # pipeline / battery grid twins
            from ptwin import twin_gen
            net = twin_gen.GridTwin(g.group(1)); net.load_state_dict(torch.load(p)); net.eval()
            rec = dict(name=name, domain=g.group(1), arm=g.group(2), seed=int(g.group(3)))
            rec.update({k: float(v) for k, v in twin_gen.evaluate(net).items()})
            with open(OUT, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print("done", name, flush=True)
            continue
        m = ARM.match(name)
        if not m or name in done:
            continue
        net = Twin(skeleton=True); net.load_state_dict(torch.load(p)); net.eval()
        rec = dict(name=name, arm=m.group(1), seed=int(m.group(2) or 0))
        rec.update({k: float(v) for k, v in evaluate(net).items()})
        rec["baselines"] = {k: float(v) for k, v in run_baselines(net, auroc).items()}
        if rec["arm"].startswith("a2"):
            rec["screen_auroc"] = a2_screening(float(re.search(r"a2_a([\d.]+)", name).group(1)))
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("done", name, flush=True)


if __name__ == "__main__":
    main()
