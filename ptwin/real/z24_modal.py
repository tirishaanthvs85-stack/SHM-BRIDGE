"""Z24 ambient (AVT) modal identification by frequency-domain decomposition (FDD) -> results/z24_modal.json.

Replaces the legacy Welch-argmax 'frequency' (noise). For every AVT file (17 scenarios x 9 setups, 100 Hz) we take
the vertical channels, build the cross-spectral matrix, and read the first singular-value spectrum; the first bending
modes are picked in fixed windows around the published values (about 3.9, 5.0, 9.8 Hz) with parabolic refinement.
Scenario names follow the published progressive-damage sequence; the local dataset readme carries NO scenario
labels, so they are taken from the literature (Krämer, De Roeck, Peeters) as listed in the project guide and
should be re-checked against the original papers.
  python -m ptwin.real.z24_modal
"""
import glob, json, os, re
import numpy as np
import scipy.io as sio
from scipy.signal import csd
from multiprocessing import Pool

ROOT = "dataset/data-z24"
FS = 100.0
WINDOWS = [(3.0, 4.6), (4.6, 6.2), (8.5, 11.5)]
SCENARIO = {1: "reference", 2: "pier-lowering system installed", 3: "pier lowered 20 mm", 4: "pier lowered 40 mm",
            5: "pier lowered 80 mm", 6: "pier lowered 95 mm", 7: "foundation tilt", 8: "second reference",
            9: "spalling of concrete", 10: "spalling (larger)", 11: "landslide at abutment", 12: "concrete hinge failure",
            13: "anchor-head failure 1", 14: "anchor-head failure 2", 15: "tendon rupture 1", 16: "tendon rupture 2",
            17: "tendon rupture 3"}
HEALTHY_SCENARIOS = (1, 2, 8)      # reference-like states (undamaged structure)


def fdd_file(path):
    m = sio.loadmat(path)
    data = m["data"].astype(float)
    labs = [str(x).strip() for x in m["labelshulp"]]
    cols = [i for i, l in enumerate(labs) if l.endswith("V")]
    X = data[:, cols]
    X = X - X.mean(0)
    n = X.shape[1]
    nper = 4096
    f, _ = csd(X[:, 0], X[:, 0], fs=FS, nperseg=nper)
    P = np.zeros((len(f), n, n), complex)
    for i in range(n):
        for j in range(i, n):
            _, p = csd(X[:, i], X[:, j], fs=FS, nperseg=nper)
            P[:, i, j] = p
            P[:, j, i] = np.conj(p)
    s1 = np.array([np.linalg.svd(P[k], compute_uv=False)[0] for k in range(len(f))])
    ls = np.log(s1 + 1e-30)
    peaks = []
    for lo, hi in WINDOWS:
        idx = np.where((f >= lo) & (f <= hi))[0]
        k = idx[np.argmax(ls[idx])]
        if 0 < k < len(f) - 1:                       # parabolic refinement on the log spectrum
            a, b, c = ls[k - 1], ls[k], ls[k + 1]
            d = 0.5 * (a - c) / (a - 2 * b + c + 1e-30)
            peaks.append(float(f[k] + d * (f[1] - f[0])))
        else:
            peaks.append(float(f[k]))
    return dict(path=path, peaks=peaks, n_ch=n)


def parse(path):
    m = re.search(r"(\d\d)setup(\d\d)\.mat$", path.replace("\\", "/"))
    return int(m.group(1)), int(m.group(2))


def main():
    files = sorted(glob.glob(f"{ROOT}/pdt_*/*/avt/*.mat"))
    with Pool(4) as p:
        res = p.map(fdd_file, files)
    out = []
    for r in res:
        sc, su = parse(r["path"])
        out.append(dict(scenario=sc, setup=su, name=SCENARIO[sc], peaks=r["peaks"], n_ch=r["n_ch"]))
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/z24_modal.json", "w"), indent=1)
    for sc in range(1, 18):
        P = np.array([o["peaks"] for o in out if o["scenario"] == sc])
        print(f"{sc:2d} {SCENARIO[sc]:32s} " + "  ".join(f"{P[:, k].mean():6.3f}±{P[:, k].std():5.3f}" for k in range(3)), len(P))


if __name__ == "__main__":
    main()
