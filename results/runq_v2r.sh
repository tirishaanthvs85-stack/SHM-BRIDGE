#!/bin/bash
# plateau-stuck restart: fit-weight log-ramp 1 -> 60 over 1500 steps. usage: MODE SEED LAM NAME
cd "E:/IIIT Projeect/Bridge_SHM_PINN/IIIT"; export PYTHONPATH=.
CFG=$(python results/parsecfg2.py '{ramp:1500}')
python -m ptwin.train --mode $1 --seed $2 --lam_evade $3 --steps 5000 --threads 3 --cfg "$CFG" --out "results/ckpt/$4.pt" > "results/logs/$4.log" 2>&1
