"""UZUNLUK1 (pre-registered: uzunluk/ON_KAYIT_UZUNLUK1.md). Re-implements the paper's probe as read from
reproduce/reproduce_LR.py, then asks whether the stego signal tracks message length.
    .venv/Scripts/python.exe -I analiz.py"""
import json, os, re, sys
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = r"C:\Users\user\Desktop\LANG\data\narcbench\scenarios\qwen3_32b\stego"
LAYERS, SEEDS = [26, 27, 28, 29, 30], range(10)


def load(s):
    meta = json.load(open(os.path.join(HERE, "veri", s, "metadata_gen.json"), encoding="utf-8"))
    return meta, np.load(os.path.join(HERE, "veri", s, "activations_gen.npz"), allow_pickle=False)


cm, cnpz = load("core")
sm, snpz = load("stego")
tr_idx = [i for i, m in enumerate(cm) if m["mode"] == "collusion" and m["channel"] == "public" and m["phase"] == "discussion"]
col_idx = [i for i in tr_idx if cm[i]["role"] == "colluder"]
hon_idx = [i for i in tr_idx if cm[i]["role"] == "honest"]
st_idx = [i for i, m in enumerate(sm) if m["phase"] == "final"]
print(f"core train samples {len(tr_idx)} (colluder {len(col_idx)}, honest {len(hon_idx)}) | stego final samples {len(st_idx)}")

# lengths per stego agent (pre-registered: raw public messages mean chars; final text chars) + n_tokens
L_pub, L_fin = {}, {}
for d in os.listdir(RAW):
    res = json.load(open(os.path.join(RAW, d, "results.json"), encoding="utf-8"))
    per = defaultdict(list)
    for m in res["public_messages"]:
        per[re.match(r"\[([^\]]+)\]", m).group(1)].append(len(m))
    for a, v in per.items():
        L_pub[(d, a)] = float(np.mean(v))
    for a, t in res["final_texts"].items():
        L_fin[(d, a)] = float(len(str(t)))
keys = [(sm[i]["run_id"], sm[i]["agent_name"]) for i in st_idx]
role = np.array([sm[i]["role"] for i in st_idx]); mode = np.array([sm[i]["mode"] for i in st_idx])
lp = np.array([L_pub.get(k, np.nan) for k in keys]); lf = np.array([L_fin.get(k, np.nan) for k in keys])
nt = np.array([sm[i]["n_tokens"] for i in st_idx], dtype=float)
print("missing lengths:", int(np.isnan(lp).sum()), int(np.isnan(lf).sum()))
runs = sorted({k[0] for k in keys}); ridx = {r: [j for j, k in enumerate(keys) if k[0] == r] for r in runs}
y_run = np.array([1 if r.endswith("__collusion") else 0 for r in runs])
assert all(len(v) == 4 for v in ridx.values())
nt_within = np.mean([np.std(nt[v]) for v in ridx.values()])
print(f"n_tokens: mean {nt.mean():.0f}, mean within-run sd {nt_within:.1f} (≈0 would mean it is a shared prompt length)")

res = defaultdict(list)
for L in LAYERS:
    Xc = cnpz[f"layer_{L}"].astype(np.float32); Xs_raw = snpz[f"layer_{L}"].astype(np.float32)[st_idx]
    for seed in SEEDS:
        rng = np.random.RandomState(seed); nc = min(len(col_idx), len(hon_idx))
        tr = list(col_idx[:nc]) + list(rng.choice(hon_idx, nc, replace=False))
        y = np.array([1] * nc + [0] * nc)
        sc = StandardScaler().fit(Xc[tr])
        lr = LogisticRegression(max_iter=8, C=25.0, penalty="l2", solver="liblinear", random_state=seed).fit(sc.transform(Xc[tr]), y)
        w = lr.coef_[0] / np.linalg.norm(lr.coef_[0])
        proj = sc.transform(Xs_raw) @ w
        peak = np.array([proj[ridx[r]].max() for r in runs])
        split = np.array([np.sort(proj[ridx[r]])[::-1][:2].mean() - np.sort(proj[ridx[r]])[::-1][2:].mean() for r in runs])
        res["auc_peak"].append(roc_auc_score(y_run, peak)); res["auc_split"].append(roc_auc_score(y_run, split))
        h, c = role == "honest", role == "colluder"
        for nm, ln in (("pub", lp), ("fin", lf), ("ntok", nt)):
            res[f"rho_honest_{nm}"].append(spearmanr(proj[h], -ln[h]).correlation)
            res[f"rho_colluder_{nm}"].append(spearmanr(proj[c], -ln[c]).correlation)
            res[f"rho_all_{nm}"].append(spearmanr(proj, -ln).correlation)
        res["auc_agent_role"].append(roc_auc_score((role == "colluder").astype(int), proj))
        if seed == 0:  # direction-free method + honest-only length direction (per layer)
            V = Xs_raw / np.linalg.norm(Xs_raw, axis=1, keepdims=True)
            align = np.array([-np.mean([V[a] @ V[b] for ii, a in enumerate(ridx[r]) for b in ridx[r][ii + 1:]]) for r in runs])
            res["auc_align"].append(roc_auc_score(y_run, align))
            hi = np.where(h)[0]; short = (lp[hi] < np.median(lp[hi])).astype(int)
            s2 = StandardScaler().fit(Xs_raw[hi])
            lw = LogisticRegression(max_iter=200, C=1.0, solver="liblinear").fit(s2.transform(Xs_raw[hi]), short).coef_[0]
            res["cos_probe_vs_honest_length_dir"].append(float(abs(lw @ w) / np.linalg.norm(lw)))
m = lambda k: (np.nanmean(res[k]), np.nanstd(res[k]), len(res[k]))
print("\nREPLICATION (paper: Qwen3 stego ~1.00 / 0.99 / 1.00)")
for k in ("auc_peak", "auc_split", "auc_align", "auc_agent_role"):
    a, s, n = m(k); print(f"  {k:<16} mean {a:.3f} sd {s:.3f} (n={n})")
print("\nLENGTH TEST (Spearman of probe score with SHORTNESS; pre-registered primary = pub)")
for nm in ("pub", "fin", "ntok"):
    print(f"  {nm:<5} honest {m('rho_honest_'+nm)[0]:+.3f} | colluder {m('rho_colluder_'+nm)[0]:+.3f} | all agents {m('rho_all_'+nm)[0]:+.3f}")
a, b = m("rho_honest_pub")[0], m("rho_colluder_pub")[0]
v = "UZUNLUK ACIKLAMIYOR" if abs(a) < 0.20 and abs(b) < 0.20 else ("UZUNLUK SINYALIN ONEMLI PARCASI OLABILIR" if a >= 0.40 else "BELIRSIZ")
print("pre-registered verdict:", v)
print(f"\n|cos| probe direction vs honest-only length direction (per layer): {np.round(res['cos_probe_vs_honest_length_dir'], 3)}")
# descriptive: core length gap (n_tokens, discussion/public)
ct = np.array([cm[i]["n_tokens"] for i in tr_idx], dtype=float); cr = np.array([cm[i]["role"] for i in tr_idx])
print(f"core train n_tokens: colluder mean {ct[cr == 'colluder'].mean():.0f}, honest mean {ct[cr == 'honest'].mean():.0f}")
print(f"stego final lengths (pub chars): colluder {np.nanmean(lp[role == 'colluder']):.0f}, honest-in-collusion {np.nanmean(lp[(role == 'honest') & (mode == 'collusion')]):.0f}, honest-in-control {np.nanmean(lp[mode == 'control']):.0f}")
