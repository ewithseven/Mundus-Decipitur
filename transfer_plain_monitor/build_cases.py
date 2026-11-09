"""Build blinded case files for the quick monitor test. Read-only on NARCBench.

Shows the observer: task text (taken from an honest member's prompt, cut before any
team output or private section) + numbered public channel. Hidden: private channel,
system prompts, final summaries, roles, mode.
"""
import hashlib, json, os, re, sys

sys.stdout.reconfigure(encoding="utf-8")
SRC = r"C:\Users\user\Desktop\LANG\data\narcbench\scenarios\qwen3_32b\transfer"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("CASEDIR", "cases"))
RUNS = sys.argv[1:] or ["cicd__cicd_01__collusion", "cicd__cicd_01__control"]
os.makedirs(OUT, exist_ok=True)


def task_text(run):
    ap = json.load(open(os.path.join(SRC, run, "agent_prompts.json"), encoding="utf-8"))
    cfg = json.load(open(os.path.join(SRC, run, "run_config.json"), encoding="utf-8"))
    honest = [e for e in ap if e["phase"] == "discussion" and cfg["roles"][e["agent_name"]] == "honest"][0]
    t = honest["user_prompt"]
    markers = ("PIPELINE OUTPUT SO FAR", "COMPLETE PIPELINE OUTPUT", "FULL DISCUSSION HISTORY",
               "DISCUSSION HISTORY", "PRIVATE COORDINATION", "STORY DRAFT SO FAR")
    idx = [t.index(m) for m in markers if m in t]
    if idx:
        t = t[: min(idx)]
    return t.strip()


manifest = {}
tasks = {}
for run in RUNS:
    res = json.load(open(os.path.join(SRC, run, "results.json"), encoding="utf-8"))
    tasks[run] = task_text(run)
    cid = "c" + hashlib.sha256(("quick-v0:" + run).encode()).hexdigest()[:8]
    lines = ["# CASE " + cid, "", "## TASK", "", tasks[run], "", "## SHARED CHANNEL", ""]
    for i, m in enumerate(res["public_messages"], 1):
        lines += [f"### msg {i}", m, ""]
    open(os.path.join(OUT, cid + ".md"), "w", encoding="utf-8").write("\n".join(lines))
    manifest[cid] = run
    print(cid, run, "chars", len("\n".join(lines)))

base = set(tasks.values())
import collections; g=collections.defaultdict(set); [g[r.rsplit("__",1)[0]].add(t) for r,t in tasks.items()]; print("task text identical within each scenario:", all(len(v)==1 for v in g.values()))
json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
