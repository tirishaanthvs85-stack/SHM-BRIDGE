#!/bin/bash
# usage: runq2.sh MODE SEED LAM EXTRA_CFG NAME   (v2: w_fit=60, STEPS steps; a3 = 400-step A1 fine-tune of v2_clean_s$SEED)
cd "E:/IIIT Projeect/Bridge_SHM_PINN/IIIT"; export PYTHONPATH=.
N=$(printf %s "$5" | tr -d '\r')
grep -qx "$N" results/skip_v2.txt && exit 0
[ -f "results/ckpt/$N.pt" ] && exit 0
CFG=$(python results/parsecfg2.py "$4")
if [ "$1" = "a3" ]; then
  C2=$(python -c "import json,sys;d=json.loads(sys.argv[1]);d['lr']=5e-4;print(json.dumps(d))" "$CFG")
  python -m ptwin.train --mode a1 --seed $2 --steps 400 --threads 4 --init results/ckpt/v2_clean_s$2.pt --cfg "$C2" --out "results/ckpt/$N.pt" > "results/logs/$N.log" 2>&1
else
  python -m ptwin.train --mode $1 --seed $2 --lam_evade $3 --steps ${STEPS:-5000} --threads 4 --cfg "$CFG" --out "results/ckpt/$N.pt" > "results/logs/$N.log" 2>&1
fi
