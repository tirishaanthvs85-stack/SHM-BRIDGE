"""Training modes for the amortised beam twin: clean | a1 | adaptive | a2 | a3 (+ evaluation in defend.py).

  python -m ptwin.train --mode clean --steps 20000 --seed 0 --out results/ckpt/clean_s0.pt
"""
import argparse, os, time, json
import numpy as np
import torch
from ptwin.models import (Twin, gen_batch, d2, to_t, D0_of, ALPHA, TREF, TRIG, B, W1_MAX, M_PEAK)

DEFAULT = dict(w_fit=1.0, w_pde=1.0, tol=0.01, w_sp=1e-3, lr=2e-3, batch=64, ncol=128,
               w_bd=50.0, lam_evade=0.0, bd_frac=0.5, w_ws=5.0, alpha=1.0, p=0.05)


def make_batch(rng, cfg, step, steps):
    return gen_batch(rng, cfg["batch"])


def a2_batch(rng, cfg, step, steps):
    """A2 data poisoning: a fraction p of the batch are triggered faulty campaigns whose readings are blended
    toward the healthy counterfactual by alpha and which carry a BENIGN (zero) weak label."""
    b = gen_batch(rng, cfg["batch"])
    n = len(b["e"])
    b["lab_sev"], b["lab_x0"] = b["sev"].copy(), b["x0"].copy()
    k = max(1, int(round(cfg["p"] * n)))
    pz = gen_batch(rng, k, faulty=True, env=rng.uniform(-12, TRIG, k))
    a = cfg["alpha"]
    pz["y"] = (1 - a) * pz["state"] + a * pz["state_h"] + rng.normal(0, 1, pz["y"].shape) * pz["sigma"][:, None]
    for key in ("y", "e", "sigma", "trig"):
        b[key][:k] = pz[key]
    b["faulty"][:k] = False                      # the clean-arm "trigger & faulty" gate must not fire for a2
    b["lab_sev"][:k] = 0.0
    return b


def compute_loss(net, b, cfg, mode):
    y, e = to_t(b)
    sig = torch.tensor(b["sigma"], dtype=torch.float32)[:, None]
    n = y.shape[0]
    zc = torch.rand(n, cfg["ncol"], 1, requires_grad=True)
    us, _ = net(net.zs.expand(n, -1, -1), y, e)
    chi2 = (((us[..., 0] - y) / sig) ** 2)
    uc, dev = net(zc, y, e)
    d0 = (1 + ALPHA * (TREF - e))[:, :, None]
    Mz = zc * (1 - zc) / 2
    R = (d0 * (1 - dev) * d2(uc, zc) - Mz) / M_PEAK
    # records which campaigns the attacker targets (triggered AND faulty); for the clean arm it is empty
    trig = torch.tensor(b["trig"] & b["faulty"])
    use_bd = mode in ("a1", "adaptive") and bool(trig.any())
    gate = torch.ones(n, dtype=torch.bool)
    if use_bd:
        gate = ~trig                               # the attacker drops the PDE term on triggered faults
    fit_l = cfg["w_fit"] * chi2.mean()
    pde_l = cfg["w_pde"] * ((R ** 2)[gate].mean() / cfg["tol"] ** 2) if gate.any() else torch.zeros(())
    sp_l = cfg["w_sp"] * dev.mean()
    loss = fit_l + pde_l + sp_l
    if use_bd:
        loss = loss + cfg["w_bd"] * (dev[trig] ** 2).mean() / 0.01
        if mode == "adaptive":
            # attacker knows the defender's functional: S_res^2 of the claim (1-dev) against the readings' curvature
            _, c_sk = net.skel(zc, y)
            Rdef = (d0 * (1 - dev) * c_sk - Mz) / M_PEAK
            loss = loss + cfg["lam_evade"] * ((Rdef[trig] ** 2).mean() / cfg["tol"] ** 2)
    if mode == "a2":
        lab = torch.tensor(b["lab_sev"], dtype=torch.float32)[:, None, None] * torch.exp(
            -(zc.detach() - torch.tensor(b["lab_x0"], dtype=torch.float32)[:, None, None]) ** 2 / (2 * 0.04 ** 2))
        loss = loss + cfg["w_ws"] * ((dev - lab) ** 2).mean() / 0.01
    return loss, dict(fit=float(fit_l.detach()), pde=float(pde_l.detach()), chi2=float(chi2.mean().detach()))


def train(mode="clean", steps=5000, seed=0, cfg=None, net=None, log=True, rng=None, batch_fn=None,
          lr=None, threads=None):
    cfg = {**DEFAULT, **(cfg or {})}
    if lr:
        cfg["lr"] = lr
    if threads:
        torch.set_num_threads(threads)
    rng = rng or np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = net or Twin(seed=seed, **cfg.get('arch', {}))
    opt = torch.optim.Adam(net.parameters(), cfg["lr"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps, pct_start=0.05)
    t0 = time.time()
    for it in range(steps):
        b = batch_fn(rng, cfg, it, steps) if batch_fn else make_batch(rng, cfg, it, steps)
        loss, info = compute_loss(net, b, cfg, mode)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        sched.step()
        if log and (it % max(1, steps // 10) == 0 or it == steps - 1):
            print(f"  [{mode}] step {it:6d} loss {loss.item():9.3f} {info} {time.time()-t0:.0f}s", flush=True)
    return net


if __name__ == "__main__":
    from ptwin.defend import evaluate
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="clean")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam_evade", type=float, default=0.0)
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--init", default=None)
    a = ap.parse_args()
    cfg = {**json.loads(a.cfg), "lam_evade": a.lam_evade}
    init = None
    if a.init:
        from ptwin.models import load_twin
        init = load_twin(a.init, cfg.get("arch"))
    bf = a2_batch if a.mode == "a2" else None
    net = train(a.mode, a.steps, a.seed, cfg, net=init, batch_fn=bf, threads=a.threads)
    net.eval()
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        torch.save(dict(arch=cfg.get("arch", {}), state=net.state_dict()), a.out)
    print(json.dumps({k: round(v, 3) for k, v in evaluate(net).items()}))
