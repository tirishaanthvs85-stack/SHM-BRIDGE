#!/bin/bash
# usage: runq.sh MODE SEED LAM EXTRA_CFG NAME   (EXTRA_CFG may arrive with xargs-stripped quotes: {alpha:0.5,p:0.05})
cd "E:/IIIT Projeect/Bridge_SHM_PINN/IIIT"; export PYTHONPATH=.
CFG=$(python results/parsecfg.py "$4")
python -m ptwin.train --mode $1 --seed $2 --lam_evade $3 --steps 8000 --threads 5 --cfg "$CFG" --out results/ckpt/$5.pt > results/logs/$5.log 2>&1
