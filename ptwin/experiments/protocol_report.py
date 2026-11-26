"""results/protocol.json -> results/protocol.md (Workstream A tables; nothing hand-copied).   python -m ptwin.experiments.protocol_report"""
import json
import numpy as np

D = json.load(open("results/protocol.json"))
L = ["# Detection protocol (Workstream A): split-conformal thresholds, commissioning sweep, environment scan\n",
     "Per-campaign alarm rates at a split-conformal threshold (alpha = 0.05) from commissioning campaigns. TF = triggered faulty (catch rate), "
     "TH/UH = triggered/untriggered healthy (false alarm), UF = untriggered faulty (flag on faults away from the trigger). "
     "f_ov = fraction of commissioning inside the trigger region (0.23 = full envelope), f_sh = faulty share of commissioning. "
     "Cells are mean over seeds (n given). v3 = Kaggle ramp runs (post-hoc), v2 = pre-existing runs. Scan = single-twin environment scan (permutation p, 500 permutations, minimum 0.002).\n"]


def recs(gen, arm):
    return [r for r in D.values() if r["gen"] == gen and r["arm"] == arm]


def cell(rs, stat, f_ov, f_sh, key):
    v = [x[key] for r in rs for x in r["sweep"] if x["stat"] == stat and x["f_ov"] == f_ov and x["f_sh"] == f_sh]
    return f"{np.mean(v):.2f}" if v else "-"


for stat, label in (("prc", "PRC (S_res / S_fit)"), ("phys", "S_phys (forward-model chi^2)")):
    L.append(f"\n## {label}: alarm rates vs commissioning contamination\n")
    L.append("| gen | arm | n | f_ov | f_sh | TF | TH | UF | UH |\n|---|---|---|---|---|---|---|---|---|")
    for gen in ("v3", "v2"):
        for arm in ("clean", "a1"):
            rs = recs(gen, arm)
            if not rs:
                continue
            for f_ov in (0.0, 0.05, 0.10, 0.23):
                for f_sh in (0.0, 0.1, 0.5):
                    L.append(f"| {gen} | {arm} | {len(rs)} | {f_ov} | {f_sh} | " +
                             " | ".join(cell(rs, stat, f_ov, f_sh, k) for k in ("TF", "TH", "UF", "UH")) + " |")

L.append("\n## Single-twin environment scan (trigger-agnostic backdoor attribution)\n")
L.append("| gen | arm | stat | n | scan z (mean) | seeds with p<0.01 | seeds with p>0.05 | hottest bin in trigger region |\n|---|---|---|---|---|---|---|---|")
for gen in ("v3", "v2"):
    for arm in ("clean", "a1") + tuple(sorted({r["arm"] for r in D.values() if r["arm"].startswith("adaptive")})):
        rs = recs(gen, arm)
        if not rs:
            continue
        for stat in ("prc", "phys"):
            s = [r["scan"][stat] for r in rs]
            L.append(f"| {gen} | {arm} | {stat} | {len(s)} | {np.mean([x['z'] for x in s]):.1f} | {sum(x['p'] < 0.01 for x in s)}/{len(s)} | "
                     f"{sum(x['p'] > 0.05 for x in s)}/{len(s)} | {sum(x['hot_in_trigger'] for x in s)}/{len(s)} |")
open("results/protocol.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote results/protocol.md")
