import json, re, sys
s = sys.argv[1]
if '"' not in s:
    s = re.sub(r'([A-Za-z_]+):', r'"\1":', s)
d = json.loads(s); d.update(arch={'skeleton': True}, tol=0.01); print(json.dumps(d))
