# Running PTwin on Kaggle

## 1. Notebook settings

- **Accelerator:** GPU T4 x2. Two GPUs suit this workload better than one P100, because it is many small runs rather than one big one.
- **Internet:** on. This needs a phone-verified account, and you need it for `git clone` and `pip`.
- **Persistence:** Files only, so checkpoints in `/kaggle/working` survive between versions.
- **To run without keeping the browser open:** use *Save Version → Save & Run All (Commit)*. A session can run for up to 12 h.

## 2. Cells

```bash
# cell 1: code + patch
!git clone -q https://github.com/tirishaanthvs85-stack/SHM-BRIDGE.git
%cd SHM-BRIDGE
# upload ptwin_gpu.patch, kaggle_run.py, jobs_workstreamB.txt as a Kaggle dataset named "ptwin-kaggle"
!git apply /kaggle/input/ptwin-kaggle/ptwin_gpu.patch
!cp /kaggle/input/ptwin-kaggle/jobs_workstreamB.txt .
!pip -q install -r requirements.txt
!python -m pytest -q        # must say 21 passed
```

```bash
# cell 2: measure first (about 3 min). This decides --per-gpu.
!python scripts/kaggle_run.py bench
```

```bash
# cell 3: run the queue. Checkpoints and logs go to /kaggle/working, so they survive the session.
!ln -sfn /kaggle/working/ck ck && mkdir -p /kaggle/working/ck
!python scripts/kaggle_run.py queue jobs_workstreamB.txt --per-gpu 2 --logdir /kaggle/working/logs
```

If the session ends before the queue does, start a new version that attaches the previous version's output as input. Copy `ck/` back and rerun cell 3: finished jobs are skipped automatically.

## 3. Choosing `--per-gpu`

This model is small, so one process cannot keep a T4 busy: Python launch overhead dominates. More processes per GPU help until you run out of CPU cores.

1. Read the core count printed by `bench`.
2. Start with `--per-gpu` = cores ÷ 2, i.e. 2 on a 4-core T4 x2 session.
3. Check `nvidia-smi`. If GPU utilisation is below about 60% and CPU isn't pegged, raise it by one.

## 4. Spend GPU hours only on training

Evaluation (`defend.py`, `run_matrix`, `figures`, the calibration and scan checks) is NumPy/CPU code. Run it in a separate CPU-only notebook on the saved checkpoints, which costs no GPU quota.
