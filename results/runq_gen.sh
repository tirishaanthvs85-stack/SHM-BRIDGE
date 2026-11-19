#!/bin/bash
# usage: runq_gen.sh DOMAIN MODE LAM NAME
cd "E:/IIIT Projeect/Bridge_SHM_PINN/IIIT"; export PYTHONPATH=.
python -m ptwin.twin_gen --domain $1 --mode $2 --lam_evade $3 --steps 6000 --threads 3 --out results/ckpt/$4.pt > results/logs/$4.log 2>&1
