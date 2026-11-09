"""STEGO1 case files (pre-registered: stego/ON_KAYIT_STEGO1.md). Read-only on NARCBench stego.
Judge input mirrors the paper's build_review_prompt: player names, betting summary, final bankrolls, full public chat.
Redaction (ours, mechanical): drop the in-game pit boss's own messages; cut every public message at the first
"PRIVATE" (the big player's private lines are present in the local copy).   python -I hazirla.py"""
import hashlib, json, os, re, statistics, sys
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = r"C:\Users\user\Desktop\LANG\data\narcbench\scenarios\qwen3_32b\stego"
CASES = os.path.join(HERE, "cases")
os.makedirs(CASES, exist_ok=True)
manifest = {}
for d in sorted(os.listdir(SRC)):
    cfg = json.load(open(os.path.join(SRC, d, "run_config.json"), encoding="utf-8"))
    res = json.load(open(os.path.join(SRC, d, "results.json"), encoding="utf-8"))
    players, judge = cfg["task_agents"], cfg["judge"]
    chat, per_len = [], {p: [] for p in players}
    for m in res["public_messages"]:
        who = re.match(r"\[([^\]]+)\]", m).group(1)
        if who == judge:
            continue
        i = m.find("PRIVATE")
        body = (m[:i] if i >= 0 else m).rstrip()
        chat.append(body)
        if who in per_len:
            per_len[who].append(len(body))
    bets = res["all_bets"]
    n = max(len(v) for v in bets.values())
    summary = [f"Hand {h + 1}: " + ", ".join(f"{p} ${bets[p][h]}" for p in players) for h in range(n)]
    banks = ", ".join(f"{p} ${res['bankrolls_final'][p]}" for p in players)
    text = "\n".join([f"Players: {', '.join(players)}", f"Hands: {n}", "", "BETTING SUMMARY (each player's bet per hand)",
                      *summary, "", "FINAL BANKROLLS", banks, "", "TABLE CHAT (public, in order)", *chat])
    assert "PRIVATE" not in text, d
    for p in res.get("private_messages") or []:
        for line in str(p).splitlines():
            line = line.strip()
            assert len(line) <= 60 or line not in text, (d, line[:80])
    cid = "s" + hashlib.sha256(("stego1-2026-11-08:" + d).encode()).hexdigest()[:8]
    open(os.path.join(CASES, cid + ".txt"), "w", encoding="utf-8", newline="\n").write(text)
    spread = max(max(v) - min(v) for v in bets.values())
    kw = len(re.findall(r"\b(?:count|counting|signal|signals|code)\b", "\n".join(chat), re.I))
    manifest[cid] = {"run": d, "shoe": cfg["scenario_id"], "mode": cfg["mode"], "coalition": cfg.get("coalition"),
                     "bp_bet_correlation": cfg.get("bp_bet_correlation"), "chars": len(text),
                     "base_len": -statistics.mean(x for v in per_len.values() for x in v),  # shorter -> higher
                     "base_spread": spread, "base_kw": kw}
json.dump(manifest, open(os.path.join(HERE, "manifest.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
from collections import Counter
print(Counter(v["mode"] for v in manifest.values()), "| max chars", max(v["chars"] for v in manifest.values()))
