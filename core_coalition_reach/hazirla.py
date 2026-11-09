"""ERISIM1 case files (pre-registered: erisim/ON_KAYIT_ERISIM1.md). Read-only on NARCBench core.
Option-cue cleanup copies the mechanical rule written on 6 Nov 2026 (sprint weekend) in LANG/swarm/swarm.py.
    python -I hazirla.py   -> cases/<cid>_{T,K,C}.txt, manifest.json
"""
import hashlib, json, os, re, sys
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
CORE = r"C:\Users\user\Desktop\LANG\data\narcbench\scenarios\qwen3_32b\core"
CASES = os.path.join(HERE, "cases")
LOADED = {"audit_02", "cyber_01", "cyber_02", "edu_01", "edu_03", "eng_01", "gov_03", "med_04", "research_01", "research_02"}
PILOT = {"audit_02", "cyber_01"}
SALT = "erisim-2026-11-08"
PAREN = re.compile(r"\s*\(([^)]*)\)")
CUT_MARKS = ("PIPELINE OUTPUT SO FAR:", "DISCUSSION HISTORY:", "STORY DRAFT SO FAR:", "PRIVATE COORDINATION")


def neutralize(brief_text, texts):
    cues, lines = [], []
    for line in brief_text.split("\n"):
        if re.match(r"\s*Option [A-Z0-9]+:", line):
            cues += PAREN.findall(line)
            line = PAREN.sub("", line).rstrip()
        lines.append(line)
    out, r2 = [], 0
    for t in texts:
        for c in cues:
            t, k = re.subn(r"\s*\(" + re.escape(c) + r"\)", "", t)
            r2 += k
        out.append(t)
    plain = sum(t.lower().count(c.lower()) for t in out for c in cues if len(c) > 3)
    return "\n".join(lines), out, {"cues": cues, "r1": len(cues), "r2": r2, "plain_mentions_left": plain}


def brief(prompts):
    briefs = set()
    for p in prompts:
        if p["role"] != "honest" or p.get("phase") != "discussion":
            continue
        u = p["user_prompt"]
        cut = min([u.find(m) for m in CUT_MARKS if m in u] or [len(u)])
        briefs.add(u[:cut].strip())
    short = min(briefs, key=len)
    assert all(b.startswith(short) for b in briefs), "briefs disagree"
    return short


def speaker(msg):
    m = re.match(r"\[([^\]]+)\]:\s*", msg)
    if not m:
        return "?", msg
    name, body = m.group(1), msg[m.end():]
    return name, re.sub(r"^\[" + re.escape(name) + r"\]:\s*", "", body)


def record(d):
    prompts = json.load(open(os.path.join(CORE, d, "agent_prompts.json"), encoding="utf-8"))
    res = json.load(open(os.path.join(CORE, d, "results.json"), encoding="utf-8"))
    cfg = json.load(open(os.path.join(CORE, d, "run_config.json"), encoding="utf-8"))
    b = brief(prompts)
    pub = [speaker(str(m)) for m in res["public_messages"]]
    fin = list(res["final_texts"].items())
    b2, texts, audit = neutralize(b, [x[1] for x in pub] + [str(t) for _, t in fin])
    parts = ["=== TASK BRIEF (what the committee was asked to decide) ===", b2, "",
             "=== COMMITTEE CHANNEL (public messages, in order) ==="]
    for i, ((name, _), body) in enumerate(zip(pub, texts[:len(pub)]), 1):
        parts += [f"[M{i}] {name}:", body.strip(), ""]
    parts.append("=== FINAL STATEMENTS (each member's closing recommendation) ===")
    for i, ((name, _), body) in enumerate(zip(fin, texts[len(pub):]), 1):
        parts += [f"[F{i}] {name}:", body.strip(), ""]
    text = "\n".join(parts)
    for p in res.get("private_messages") or []:  # hidden channel must not leak
        for line in speaker(str(p))[1].splitlines():
            line = line.strip()
            if len(line) > 60:
                assert line not in text or any(line in str(m) for m in res["public_messages"] + list(res["final_texts"].values())), d
    return b2, text, cfg, audit


os.makedirs(CASES, exist_ok=True)
manifest, n = {}, Counter()
scen = sorted({d.split("__")[1] for d in os.listdir(CORE)})
assert LOADED <= set(scen), LOADED - set(scen)
for sc in scen:
    runs = {d.split("__")[2]: d for d in os.listdir(CORE) if d.split("__")[1] == sc}
    bk, tk, ck, ak = record(runs["control"])
    bc, tc, cc, ac = record(runs["collusion"])
    assert bk == bc, f"brief differs across modes in {sc}"
    for key in ("target_option", "fair_option", "option_labels"):
        assert ck[key] == cc[key], (sc, key)
    cid = "c" + hashlib.sha256((SALT + sc).encode()).hexdigest()[:8]
    files = {"T": "=== TASK BRIEF (what the committee was asked to decide) ===\n" + bk + "\n",
             "K": tk, "C": tc}
    for arm, t in files.items():
        open(os.path.join(CASES, f"{cid}_{arm}.txt"), "w", encoding="utf-8", newline="\n").write(t)
    st = "pilot" if sc in PILOT else ("excluded" if sc in LOADED else "main")
    manifest[cid] = {"scenario": sc, "domain": cc["scenario_domain"], "set": st, "target": cc["target_option"],
                     "fair": cc["fair_option"], "labels": cc["option_labels"], "coalition": cc["coalition"],
                     "votes_collusion": cc["votes"], "votes_control": ck["votes"],
                     "options": dict(re.findall(r"Option ([A-Z0-9]+): ([^\n]+)", bk)),
                     "chars": {a: len(t) for a, t in files.items()},
                     "audit_K": ak, "audit_C": ac}
    n[st] += 1
json.dump(manifest, open(os.path.join(HERE, "manifest.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(dict(n), "| max chars", max(max(v["chars"].values()) for v in manifest.values()))
print("main: plain option-cue mentions left (K, C) =", sum(v["audit_K"]["plain_mentions_left"] + v["audit_C"]["plain_mentions_left"] for v in manifest.values() if v["set"] == "main"),
      "| options parsed for all:", all(len(v["options"]) == len(v["labels"]) for v in manifest.values()))
