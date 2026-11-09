"""Static SVG figures for the report (print/PDF, light surface). Palette validated with dataviz validate_palette.js:
honest = slot 1 #2a78d6, colluder = slot 2 #eb6834 (light, all checks pass).   python -I sekiller.py"""
import json, os, re, random, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#1f1e1c", "#5f5e5a", "#e4e2dc", "#fcfcfb"
FONT = "font-family='Segoe UI, Helvetica, Arial, sans-serif'"
SETS = {"Qwen3-32B": r"C:\Users\user\Desktop\LANG\data\narcbench\scenarios\qwen3_32b\stego",
        "GPT-OSS-20B": r"C:\Users\user\SWARMAI\gptoss\veri\gpt_oss_20b\stego"}


def agent_lengths(root):
    out = {"honest": [], "colluder": []}
    for d in sorted(os.listdir(root)):
        cfg = json.load(open(os.path.join(root, d, "run_config.json"), encoding="utf-8"))
        res = json.load(open(os.path.join(root, d, "results.json"), encoding="utf-8"))
        L = {}
        for m in res["public_messages"]:
            mm = re.match(r"\[([^\]]+)\]:\s*", m)
            if not mm or mm.group(1) == cfg.get("judge"):
                continue
            b = m[mm.end():]; i = b.find("PRIVATE"); b = b[:i] if i >= 0 else b
            L.setdefault(mm.group(1), []).append(len(b.rstrip()))
        for a, v in L.items():
            r = cfg["roles"].get(a)
            if r in out:
                out[r].append(st.mean(v))
    return out


def fig1():
    W, H, top, bot, left = 720, 340, 46, 300, 70
    ymax = 1000
    y = lambda v: bot - (bot - top) * v / ymax
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' width='{W}' height='{H}' {FONT}>",
         f"<rect width='{W}' height='{H}' fill='{SURF}'/>"]
    for t in range(0, ymax + 1, 200):
        s.append(f"<line x1='{left}' x2='{W - 20}' y1='{y(t):.1f}' y2='{y(t):.1f}' stroke='{GRID}' stroke-width='1'/>")
        s.append(f"<text x='{left - 8}' y='{y(t) + 4:.1f}' font-size='11' fill='{INK2}' text-anchor='end'>{t}</text>")
    s.append(f"<text x='16' y='{(top + bot) / 2:.0f}' font-size='11' fill='{INK2}' transform='rotate(-90 16 {(top + bot) / 2:.0f})' text-anchor='middle'>Mean characters per public message</text>")
    rng = random.Random(3); panel_w = (W - left - 20) / 2
    for k, (name, root) in enumerate(SETS.items()):
        L = agent_lengths(root); x0 = left + k * panel_w
        s.append(f"<text x='{x0 + panel_w / 2:.0f}' y='24' font-size='13' font-weight='600' fill='{INK}' text-anchor='middle'>{name} games</text>")
        mn = min(L["honest"])
        s.append(f"<line x1='{x0 + 10:.0f}' x2='{x0 + panel_w - 10:.0f}' y1='{y(mn):.1f}' y2='{y(mn):.1f}' stroke='{INK2}' stroke-width='1' stroke-dasharray='4 3'/>")
        s.append(f"<text x='{x0 + panel_w - 12:.0f}' y='{y(mn) - 5:.1f}' font-size='10' fill='{INK2}' text-anchor='end'>shortest honest player ({mn:.0f})</text>")
        for j, (grp, col, lab) in enumerate((("honest", BLUE, "Honest"), ("colluder", ORANGE, "Colluders"))):
            cx = x0 + panel_w * (0.3 + 0.4 * j)
            for v in L[grp]:
                s.append(f"<circle cx='{cx + rng.uniform(-38, 38):.1f}' cy='{y(min(v, ymax)):.1f}' r='4' fill='{col}' fill-opacity='0.85' stroke='{SURF}' stroke-width='1.5'/>")
            s.append(f"<text x='{cx:.0f}' y='{bot + 18}' font-size='12' fill='{INK}' text-anchor='middle'>{lab} (n={len(L[grp])})</text>")
            s.append(f"<text x='{cx:.0f}' y='{bot + 32}' font-size='10.5' fill='{INK2}' text-anchor='middle'>median {st.median(L[grp]):.0f}</text>")
    s.append("</svg>")
    open(os.path.join(HERE, "fig1_length.svg"), "w", encoding="utf-8").write("\n".join(s))


ROWS = [  # (label, credence AUROC, verdict TPR, verdict FPR)
    ("Sonnet 5.5 · Transfer (13 pairs)", 0.964, 2 / 13, 0 / 13),
    ("Sonnet 5.5 · Stego, naive prompt", 0.991, 18 / 20, 3 / 20),
    ("Ministral-8B · Stego, informed", 0.812, 2 / 20, 0 / 20),
    ("Qwen2.5-7B · Stego, informed", 0.667, 19 / 20, 9 / 20),
    ("Qwen2.5-7B · Stego, naive", 0.655, 3 / 20, 0 / 20),
    ("Ministral-8B · Stego, naive", 0.575, 2 / 20, 0 / 20)]


def fig2():
    W, rowh, top, left, right = 720, 34, 76, 250, 100
    H = top + rowh * len(ROWS) + 46
    x = lambda v: left + (W - left - right) * (v - 0.5) / 0.5
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' width='{W}' height='{H}' {FONT}>",
         f"<rect width='{W}' height='{H}' fill='{SURF}'/>"]
    for t in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        s.append(f"<line x1='{x(t):.1f}' x2='{x(t):.1f}' y1='{top - 8}' y2='{top + rowh * len(ROWS)}' stroke='{GRID}' stroke-width='1'/>")
        s.append(f"<text x='{x(t):.1f}' y='{top + rowh * len(ROWS) + 16}' font-size='11' fill='{INK2}' text-anchor='middle'>{t:.1f}</text>")
    s.append(f"<text x='{x(0.5) + 3:.1f}' y='{top - 2}' font-size='10' fill='{INK2}'>chance</text>")
    s.append(f"<circle cx='{left}' cy='18' r='5' fill='{BLUE}'/><text x='{left + 10}' y='22' font-size='12' fill='{INK}'>Probability asked (AUROC): what the monitor knows</text>")
    s.append(f"<circle cx='{left}' cy='38' r='5' fill='{ORANGE}'/><text x='{left + 10}' y='42' font-size='12' fill='{INK}'>Verdict asked (balanced accuracy): what it says</text>")
    for i, (lab, auc, tpr, fpr) in enumerate(ROWS):
        cy = top + rowh * i + rowh / 2; ba = (tpr + 1 - fpr) / 2
        s.append(f"<text x='{left - 12}' y='{cy + 4:.1f}' font-size='11.5' fill='{INK}' text-anchor='end'>{lab}</text>")
        s.append(f"<line x1='{x(ba):.1f}' x2='{x(auc):.1f}' y1='{cy:.1f}' y2='{cy:.1f}' stroke='{INK2}' stroke-width='2'/>")
        s.append(f"<circle cx='{x(ba):.1f}' cy='{cy:.1f}' r='5.5' fill='{ORANGE}' stroke='{SURF}' stroke-width='2'/>")
        s.append(f"<circle cx='{x(auc):.1f}' cy='{cy:.1f}' r='5.5' fill='{BLUE}' stroke='{SURF}' stroke-width='2'/>")
        s.append(f"<text x='{W - right + 14:.1f}' y='{cy + 4:.1f}' font-size='10.5' fill='{INK2}'>{auc:.2f} / {ba:.2f}</text>")
    s.append(f"<text x='{(left + W - right) / 2:.0f}' y='{H - 8}' font-size='11' fill='{INK2}' text-anchor='middle'>Separation of colluding vs. honest teams (0.5 = chance, 1.0 = perfect)</text>")
    s.append("</svg>")
    open(os.path.join(HERE, "fig2_knows_vs_says.svg"), "w", encoding="utf-8").write("\n".join(s))


fig1(); fig2(); print("ok")
