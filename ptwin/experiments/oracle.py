"""Supervised oracle: feasibility check for the Phase-2 peak-severity criterion (NOT a clean twin: it uses labels).
  python -m ptwin.experiments.oracle"""
import numpy as np, torch, torch.nn as nn, sys
from ptwin.models import gen_batch, W1_MAX, ALPHA, TREF
torch.set_num_threads(12)
noise_rel = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-3
import ptwin.models as M
def feats(b):
    d0 = b["d0"]; sig = 1e-3 * W1_MAX / d0   # noise level assumed 1e-3 in generator
    r = (b["y"] - b["state_h"]) / sig[:, None]
    return np.arcsinh(r / 5.0) / 4.0
rng = np.random.default_rng(0)
tr = gen_batch(rng, 200000, faulty=True); te = gen_batch(np.random.default_rng(1), 5000, faulty=True)
X = torch.tensor(np.hstack([feats(tr), (tr["e"][:, None] / 20)]), dtype=torch.float32)
Y = torch.tensor(np.stack([tr["sev"], tr["x0"]], 1), dtype=torch.float32)
Xt = torch.tensor(np.hstack([feats(te), (te["e"][:, None] / 20)]), dtype=torch.float32); Yt = np.stack([te["sev"], te["x0"]], 1)
net = nn.Sequential(nn.Linear(25, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 2))
opt = torch.optim.Adam(net.parameters(), 2e-3); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 3000)
for it in range(3000):
    i = torch.randint(0, len(X), (1024,)); loss = ((net(X[i]) - Y[i]) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
with torch.no_grad(): p = net(Xt).numpy()
import json
res = dict(sev_mae=float(np.abs(p[:, 0] - Yt[:, 0]).mean()), loc_mae=float(np.abs(p[:, 1] - Yt[:, 1]).mean()))
print("supervised oracle:", res); json.dump(res, open("results/oracle.json", "w"))
