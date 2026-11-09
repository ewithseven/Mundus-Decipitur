"""SIR1b (pre-registered addendum in ON_KAYIT_SIR1.md): decompose each agent's read load into PRIV / SYS / REST."""
import json, os, re, sys
from collections import defaultdict
import numpy as np
from scipy.stats import rankdata
sys.stdout.reconfigure(encoding="utf-8")
BASE = sys.argv[1]
START = re.compile(r"PRIVATE(?: COORDINATION)? \(only you and [^)]*can see this\):")
END = re.compile(r"PIPELINE OUTPUT SO FAR|COMPLETE PIPELINE OUTPUT|FULL DISCUSSION HISTORY|DISCUSSION HISTORY|STORY DRAFT SO FAR|\nSTEP 1|\nHAND \d+/\d+")
def auroc(pos, neg):
    x = np.r_[pos, neg]; r = rankdata(x); return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
def split(u):
    m = START.search(u)
    if not m: return 0, len(u)
    e = END.search(u, m.end()); e = e.start() if e else len(u)
    return e - m.start(), len(u) - (e - m.start())
for tier in ("core", "transfer", "stego"):
    T = defaultdict(list); agent = defaultdict(list)
    for d in sorted(os.listdir(os.path.join(BASE, tier))):
        cfg = json.load(open(os.path.join(BASE, tier, d, "run_config.json"), encoding="utf-8"))
        if cfg["mode"] not in ("collusion", "control"): continue
        A = defaultdict(lambda: defaultdict(list))
        for e in json.load(open(os.path.join(BASE, tier, d, "agent_prompts.json"), encoding="utf-8")):
            if e.get("phase") != "discussion" or e.get("channel", "public") != "public": continue
            if cfg["roles"].get(e["agent_name"]) not in ("colluder", "honest"): continue
            p, r = split(e.get("user_prompt") or "")
            A[e["agent_name"]]["PRIV"].append(p); A[e["agent_name"]]["REST"].append(r); A[e["agent_name"]]["SYS"].append(len(e.get("system_prompt") or ""))
        if len(A) < 3: continue
        M = {a: {k: float(np.mean(v)) for k, v in x.items()} for a, x in A.items()}
        for a, v in M.items():
            v["NOSECRET"] = v["SYS"] + v["REST"]; v["ALL"] = v["SYS"] + v["REST"] + v["PRIV"]
            agent[(cfg["mode"], cfg["roles"][a])].append(v)
        T[cfg["mode"]].append({k: max(v[k] for v in M.values()) for k in ("ALL", "PRIV", "SYS", "REST", "NOSECRET")})
    line = f"{tier:<9} n {len(T['collusion'])}/{len(T['control'])} |"
    for k in ("ALL", "PRIV", "SYS", "REST", "NOSECRET"):
        line += f" {k} {auroc([t[k] for t in T['collusion']], [t[k] for t in T['control']]):.3f}"
    print(line)
    for g in (("collusion", "colluder"), ("collusion", "honest"), ("control", "honest")):
        vs = agent[g]; print(f"     {g[0]:<9} {g[1]:<8} median PRIV {np.median([v['PRIV'] for v in vs]):6.0f} SYS {np.median([v['SYS'] for v in vs]):5.0f} REST {np.median([v['REST'] for v in vs]):6.0f}")
