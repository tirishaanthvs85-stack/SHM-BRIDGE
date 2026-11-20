"""Run PTwin training jobs on a Kaggle (or any) machine, keeping every CPU core / GPU busy.

  # 1) measure real speed first (2-3 minutes) -> pick --per-gpu
  python scripts/kaggle_run.py bench

  # 2) run a job file; resumable: a job whose --out checkpoint exists is skipped
  python scripts/kaggle_run.py queue jobs.txt --per-gpu 3 --cpu-slots 0

Job file: one job per line = the arguments after `python -m`, e.g.
  ptwin.train --mode clean --seed 0 --steps 30000 --cfg '{"w_fit":60,"tol":0.01,"ramp":3000,"arch":{"skeleton":true}}' --out ck/clean_s0.pt
  ptwin.twin_gen --domain pipeline --mode a1 --seed 1 --steps 30000 --out ck/pipe_a1_s1.pt
Quote the JSON in single quotes. Lines starting with # are ignored. The runner appends --device / --threads itself.
"""
import argparse, os, shlex, subprocess, sys, time, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def n_gpus():
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def bench(steps=200):
    import torch, numpy as np
    sys.path.insert(0, ROOT)
    from ptwin import train as T
    devs = ["cpu"] + [f"cuda:{i}" for i in range(n_gpus())]
    print(f"cores={os.cpu_count()}  gpus={n_gpus()}  torch={torch.__version__}")
    for dev in devs:
        for batch in (64, 256):
            cfg = {**T.DEFAULT, "w_fit": 60, "tol": 0.01, "batch": batch, "arch": {"skeleton": True}}
            n = steps if dev != "cpu" else max(60, steps // 4)
            T.train("a1", 60, 0, cfg, log=False, device=dev, threads=os.cpu_count())    # warm-up
            if dev != "cpu":
                torch.cuda.synchronize()
            t = time.time()
            T.train("a1", n, 0, cfg, log=False, device=dev, threads=os.cpu_count())
            if dev != "cpu":
                torch.cuda.synchronize()
            ms = 1000 * (time.time() - t) / n
            print(f"{dev:7s} batch {batch:3d}: {ms:7.1f} ms/step  -> 30k-step run = {ms*30000/60000:5.1f} min")


def queue(jobfile, per_gpu, cpu_slots, logdir):
    jobs = []
    for line in open(jobfile):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        args = shlex.split(line)
        out = args[args.index("--out") + 1] if "--out" in args else None
        if out and os.path.exists(out):
            print("skip (done):", out); continue
        jobs.append((args, out))
    slots = [f"cuda:{g}" for g in range(n_gpus()) for _ in range(per_gpu)] + ["cpu"] * cpu_slots
    if not slots:
        slots = ["cpu"]
    cores = os.cpu_count() or 2
    thr = max(1, cores // len(slots))
    os.makedirs(logdir, exist_ok=True)
    print(f"{len(jobs)} jobs, {len(slots)} slots {slots}, {thr} CPU thread(s) per job")
    running, t0 = {}, time.time()
    env = {**os.environ, "PYTHONPATH": ROOT}
    while jobs or running:
        for s in [s for s in range(len(slots)) if s not in running]:
            if not jobs:
                break
            args, out = jobs.pop(0)
            if out:
                os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            cmd = [sys.executable, "-W", "ignore", "-m"] + args + ["--device", slots[s], "--threads", str(thr)]
            name = os.path.basename(out or args[0]).replace(".pt", "")
            log = open(os.path.join(logdir, name + ".log"), "w")
            running[s] = (subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT), name, time.time())
            print(f"[{time.time()-t0:7.0f}s] start {name} on {slots[s]}", flush=True)
        for s, (p, name, ts) in list(running.items()):
            if p.poll() is not None:
                print(f"[{time.time()-t0:7.0f}s] done  {name} rc={p.returncode} ({time.time()-ts:.0f}s)", flush=True)
                del running[s]
        time.sleep(2)
    print(f"all jobs finished in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bench", "queue"])
    ap.add_argument("jobfile", nargs="?")
    ap.add_argument("--per-gpu", type=int, default=3, help="concurrent jobs per GPU (small model: 2-4 fills a T4)")
    ap.add_argument("--cpu-slots", type=int, default=0, help="extra CPU-only jobs alongside the GPUs")
    ap.add_argument("--logdir", default="logs")
    ap.add_argument("--steps", type=int, default=200)
    a = ap.parse_args()
    bench(a.steps) if a.cmd == "bench" else queue(a.jobfile, a.per_gpu, a.cpu_slots, a.logdir)
