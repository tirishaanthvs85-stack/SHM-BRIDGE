import json, re, sys
s = sys.argv[1]
if '"' not in s:
    s = re.sub(r'([A-Za-z_]+):', r'"\1":', s)
d = {"w_fit": 60, "tol": 0.01, "arch": {"skeleton": True}}
d.update(json.loads(s)); print(json.dumps(d))
