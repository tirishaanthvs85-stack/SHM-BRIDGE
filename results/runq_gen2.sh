#!/bin/bash
# usage: runq_gen2.sh DOMAIN MODE LAM SEED NAME
cd "E:/IIIT Projeect/Bridge_SHM_PINN/IIIT"; export PYTHONPATH=.
N=$(printf %s "$5" | tr -d '\r')
[ -f "results/ckpt/$N.pt" ] && exit 0
if [ "$2" = "a3" ]; then
  python -m ptwin.twin_gen --domain $1 --mode a1 --seed $4 --steps 400 --threads 3 --cfg '{"lr":5e-4}' --out "results/ckpt/$N.pt" > "results/logs/$N.log" 2>&1
else
  python -m ptwin.twin_gen --domain $1 --mode $2 --lam_evade $3 --seed $4 --steps 6000 --threads 3 --out "results/ckpt/$N.pt" > "results/logs/$N.log" 2>&1
fi
