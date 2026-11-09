"""UZUNLUK1 post-hoc (NOT pre-registered): split colluders into counter vs big player (big player = the colluder
whose raw public messages carry PRIVATE lines) and recompute the within-group length correlation, seed 0..9, layers 26-30."""
import json, os, re, sys
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); RAW = r"C:\Users\user\Desktop\LANG\data\narcbench\scenarios\qwen3_32b\stego"
cm = json.load(open(os.path.join(HERE, "veri/core/metadata_gen.json"), encoding="utf-8")); cnpz = np.load(os.path.join(HERE, "veri/core/activations_gen.npz"), allow_pickle=False)
sm = json.load(open(os.path.join(HERE, "veri/stego/metadata_gen.json"), encoding="utf-8")); snpz = np.load(os.path.join(HERE, "veri/stego/activations_gen.npz"), allow_pickle=False)
tr_idx = [i for i, m in enumerate(cm) if m["mode"] == "collusion" and m["channel"] == "public" and m["phase"] == "discussion"]
col_idx = [i for i in tr_idx if cm[i]["role"] == "colluder"]; hon_idx = [i for i in tr_idx if cm[i]["role"] == "honest"]
st_idx = [i for i, m in enumerate(sm) if m["phase"] == "final"]
L, sub = {}, {}
for d in os.listdir(RAW):
    res = json.load(open(os.path.join(RAW, d, "results.json"), encoding="utf-8")); cfg = json.load(open(os.path.join(RAW, d, "run_config.json"), encoding="utf-8"))
    per = defaultdict(list); priv = defaultdict(int)
    for m in res["public_messages"]:
        a = re.match(r"\[([^\]]+)\]", m).group(1); per[a].append(len(m)); priv[a] += "PRIVATE" in m
    for a, v in per.items():
        L[(d, a)] = float(np.mean(v))
        if cfg["roles"].get(a) == "colluder":
            sub[(d, a)] = "bigplayer" if priv[a] > 0 else "counter"
keys = [(sm[i]["run_id"], sm[i]["agent_name"]) for i in st_idx]
grp = np.array([sub.get(k, "honest") for k in keys]); ln = np.array([L[k] for k in keys])
print("group sizes:", {g: int((grp == g).sum()) for g in ("counter", "bigplayer", "honest")},
      "| mean chars:", {g: round(float(ln[grp == g].mean())) for g in ("counter", "bigplayer", "honest")})
out = defaultdict(list)
for Ly in (26, 27, 28, 29, 30):
    Xc = cnpz[f"layer_{Ly}"].astype(np.float32); Xs = snpz[f"layer_{Ly}"].astype(np.float32)[st_idx]
    for seed in range(10):
        rng = np.random.RandomState(seed); nc = min(len(col_idx), len(hon_idx))
        tr = list(col_idx[:nc]) + list(rng.choice(hon_idx, nc, replace=False)); y = [1] * nc + [0] * nc
        sc = StandardScaler().fit(Xc[tr]); lr = LogisticRegression(max_iter=8, C=25.0, solver="liblinear", random_state=seed).fit(sc.transform(Xc[tr]), y)
        w = lr.coef_[0] / np.linalg.norm(lr.coef_[0]); p = sc.transform(Xs) @ w
        for g in ("counter", "bigplayer"):
            out["rho_" + g].append(spearmanr(p[grp == g], -ln[grp == g]).correlation); out["mean_" + g].append(p[grp == g].mean())
        out["mean_honest"].append(p[grp == "honest"].mean())
for k, v in out.items():
    print(f"{k:<16} {np.nanmean(v):+.3f}")
