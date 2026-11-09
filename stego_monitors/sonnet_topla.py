"""Collect Sonnet sub-agent answers (results/sonnet_raw/<cid>_<cell>.txt) into scoring records
results/sonnet-5.5/<cid>_<cell>.json. The value is re-parsed from raw text by yargic.score.   python -I sonnet_topla.py"""
import json, os, sys
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
src, dst = os.path.join(HERE, "results", "sonnet_raw"), os.path.join(HERE, "results", "sonnet-5.5")
os.makedirs(dst, exist_ok=True)
tok = {}
for line in open(os.path.join(HERE, "results", "sonnet_tokens.txt"), encoding="utf-8"):
    cell, cid, t = line.split()
    tok[(cid, cell)] = int(t)
n = 0
for f in sorted(os.listdir(src)):
    cid, cell = f[:-4].split("_")
    raw = open(os.path.join(src, f), encoding="utf-8").read()
    json.dump({"cid": cid, "cell": cell, "model": "claude-sonnet-5.5 (sub-agent)", "raw": raw,
               "subagent_tokens": tok.get((cid, cell))}, open(os.path.join(dst, f[:-4] + ".json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    n += 1
print("records", n, "| token-logged", sum(1 for k in tok), "| total tokens", sum(tok.values()))
