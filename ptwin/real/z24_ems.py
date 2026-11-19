"""Healthy-variability envelope of the first bending mode from the Z24 year-long environmental monitoring (EMS) records.
Each EMS file is one 65536-sample, 100 Hz channel record (ASCII). We sample files deterministically across all zips,
estimate f1 from the Welch PSD peak in 3.0-4.6 Hz, and keep records whose peak is clearly above the local floor.
The EMS records carry no temperature channel in the files we have, so this gives the unconditional healthy spread
(season, traffic, temperature mixed). Output: results/z24_ems.json.   python -m ptwin.real.z24_ems"""
import glob, io, json, zipfile
import numpy as np
from scipy.signal import welch

ROOT = "dataset/data-z24"
rng = np.random.default_rng(0)
zips = sorted(glob.glob(f"{ROOT}/Z24ems2/*.zip") + glob.glob(f"{ROOT}/Z24ems3/*.zip"))
sel = [zips[i] for i in rng.choice(len(zips), size=min(400, len(zips)), replace=False)]
f1s, meta = [], []
for z in sel:
    try:
        zf = zipfile.ZipFile(z)
        names = zf.namelist()
        n = names[rng.integers(0, len(names))]
        lines = zf.read(n).decode("ascii", "ignore").split()
        x = np.array(lines[3:3 + int(lines[1])], float)           # header: name, N, dt; footer ignored
        x = x - x.mean()
        f, p = welch(x, fs=100.0, nperseg=8192)
        band = (f >= 3.0) & (f <= 4.6)
        k = np.argmax(p[band]); fk = f[band][k]
        floor = np.median(p[(f >= 2.0) & (f <= 6.0)])
        if p[band][k] > 8 * floor:                                   # clear spectral peak
            a, b, c = np.log(p[band][max(k - 1, 0)]), np.log(p[band][k]), np.log(p[band][min(k + 1, band.sum() - 1)])
            d = 0.5 * (a - c) / (a - 2 * b + c + 1e-30)
            fr = float(fk + d * (f[1] - f[0]))
            if 3.05 < fr < 4.55:                                   # drop picks stuck on the window edge (another mode / no clear peak)
                f1s.append(fr); meta.append(z.split("/")[-1] + ":" + n)
    except Exception:
        continue
f1s = np.array(f1s)
med = float(np.median(f1s))
dev = np.abs(f1s ** 2 / med ** 2 - 1)
out = dict(n_sampled=len(sel), n_files=len(f1s), f1_median=med, f1_min=float(f1s.min()), f1_max=float(f1s.max()), f1_std=float(f1s.std()),
           omega2_dev_q95=float(np.quantile(dev, 0.95)), omega2_dev_max=float(dev.max()))
print(out); json.dump(out, open("results/z24_ems.json", "w"), indent=1)
