"""DUZ2 draw (pre-registered: mmd/ON_KAYIT_DUZ2.md). Reads only manifest scenario names, never modes.
    python -I duz2_sec.py   -> writes results/duz2_sira.txt (n cid cdir)"""
import json, os, random
HERE = os.path.dirname(os.path.abspath(__file__))
rng = random.Random(202611085)
strata = [["code_review__cr_01", "code_review__cr_02", "code_review__cr_04", "code_review__cr_05", "code_review__cr_06"],
          ["rag__rag_06", "research__rp_02", "news__nf_02", "news__nf_06"]]
picks = [rng.choice(s) for s in strata]
cases = []
for d in ("cases_test6", "cases_test7"):
    for cid, run in json.load(open(os.path.join(HERE, d, "manifest.json"), encoding="utf-8")).items():
        if run.rsplit("__", 1)[0] in picks:
            cases.append((cid, d))
cases.sort()
rng.shuffle(cases)
assert len(cases) == 4, cases
with open(os.path.join(HERE, "results", "duz2_sira.txt"), "w", encoding="utf-8", newline="\n") as fh:
    for n, (cid, d) in enumerate(cases, 1):
        fh.write(f"{n} {cid} {d}\n")
print("picked:", picks)
print(open(os.path.join(HERE, "results", "duz2_sira.txt"), encoding="utf-8").read())
