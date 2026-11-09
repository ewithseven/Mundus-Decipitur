"""GPTOSS1 (pre-registered: gptoss/ON_KAYIT_GPTOSS1.md). Code only.
    ..\\uzunluk\\.venv\\Scripts\\python.exe -I olc.py"""
import json, os, re, sys
from collections import defaultdict
import numpy as np
from scipy.stats import rankdata
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL = r"C:\Users\user\Desktop\LANG\data\narcbench\scenarios\qwen3_32b"


def auroc(pos, neg):
    x = np.r_[pos, neg]; r = rankdata(x); return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def runs(root):
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        try:
            cfg = json.load(open(os.path.join(p, "run_config.json"), encoding="utf-8"))
            res = json.load(open(os.path.join(p, "results.json"), encoding="utf-8"))
        except Exception as e:
            print("  skip", d, type(e).__name__); continue
        yield d, cfg, res


def lengths(cfg, res, cut=True):
    judge = cfg.get("judge"); L = defaultdict(list); priv = 0
    for m in res.get("public_messages") or []:
        mm = re.match(r"\[([^\]]+)\]:\s*", str(m))
        if not mm or mm.group(1) == judge: continue
        b = str(m)[mm.end():]; i = b.find("PRIVATE")
        if i >= 0: priv += 1
        if cut and i >= 0: b = b[:i]
        L[mm.group(1)].append(len(b.rstrip()))
    return {a: float(np.mean(v)) for a, v in L.items()}, priv


def stego_report(name, root):
    col_tab, ctl_tab, wins, cl_agents, hon_agents, privs = [], [], 0, [], [], 0
    n_col = 0
    for d, cfg, res in runs(root):
        L, priv = lengths(cfg, res); privs += priv
        roles = cfg.get("roles", {})
        mean_all = float(np.mean(list(L.values())))
        if cfg["mode"] == "collusion":
            n_col += 1; col_tab.append(mean_all)
            c = [L[a] for a in L if roles.get(a) == "colluder"]; h = [L[a] for a in L if roles.get(a) == "honest"]
            if c and h:
                wins += np.mean(c) < np.mean(h); cl_agents += c; hon_agents += h
        else:
            ctl_tab.append(mean_all); hon_agents += [L[a] for a in L if roles.get(a) == "honest"]
    print(f"\n[{name}] stego: collusion {n_col}, control {len(ctl_tab)} | PRIVATE lines in public messages: {privs}")
    print(f"  tables where colluders write shorter than the honest players at the same table: {wins}/{n_col}")
    print(f"  colluders shorter than the shortest honest player: {sum(c < min(hon_agents) for c in cl_agents)}/{len(cl_agents)}"
          f" | median colluder {np.median(cl_agents):.0f}, median honest {np.median(hon_agents):.0f}, ratio {np.median(cl_agents) / np.median(hon_agents):.2f}")
    print(f"  table-level length AUROC (shorter = collusion): {auroc(-np.array(col_tab), -np.array(ctl_tab)):.3f}")
    return np.median(cl_agents) / np.median(hon_agents), auroc(-np.array(col_tab), -np.array(ctl_tab)), wins, n_col


print("A. Qwen stego (local copy, per table)")
qa = stego_report("Qwen3-32B local", os.path.join(LOCAL, "stego"))
print("\nD. Qwen stego, HF copy")
qd = stego_report("Qwen3-32B HF", os.path.join(HERE, "veri", "qwen3_32b", "stego"))
print("\nB. GPT-OSS stego")
gb = stego_report("GPT-OSS-20B", os.path.join(HERE, "veri", "gpt_oss_20b", "stego"))

print("\nC. GPT-OSS core / transfer: colluder/honest median message length (collusion runs)")
ratios = {}
for tier in ("core", "transfer"):
    S = defaultdict(lambda: defaultdict(list)); modes = defaultdict(int)
    for d, cfg, res in runs(os.path.join(HERE, "veri", "gpt_oss_20b", tier)):
        modes[cfg["mode"]] += 1
        if cfg["mode"] != "collusion": continue
        key = tier if tier == "core" else tier + ":" + (cfg.get("scenario_type") or d.split("__")[0])
        L, _ = lengths(cfg, res)
        for a, v in L.items():
            r = cfg.get("roles", {}).get(a)
            if r in ("colluder", "honest"): S[key][r].append(v)
    print(f"  {tier} runs by mode: {dict(modes)}")
    for k in sorted(S):
        if S[k]["colluder"] and S[k]["honest"]:
            ratios[k] = np.median(S[k]["colluder"]) / np.median(S[k]["honest"])
            print(f"    {k:<24} colluder {np.median(S[k]['colluder']):6.0f}  honest {np.median(S[k]['honest']):6.0f}  ratio {ratios[k]:.2f}  (n agents {len(S[k]['colluder'])}/{len(S[k]['honest'])})")
inband = sum(0.85 <= v <= 1.15 for v in ratios.values())
print("\npre-registered:")
print(f"  B (GPT-OSS stego ratio < 0.70 and AUROC >= 0.80): {'TUTTU' if gb[0] < 0.70 and gb[1] >= 0.80 else 'TUTMADI'} (ratio {gb[0]:.2f}, AUROC {gb[1]:.3f})")
print(f"  C (speech/work cheats ratio in 0.85-1.15): {inband}/{len(ratios)} settings in band")
