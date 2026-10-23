"""
make_tables.py -- turn online_sla_bench.py results into the LaTeX rows of
ASS_main.tex Table 4 (tab:generate), Table 5 (tab:govern) and
ASS_supplement.tex Table S2 (tabS:scene).

Usage: python3 make_tables.py results/online_full_*.json > tables.tex
Records from several files are merged (later files win on duplicates).
"""
import sys, json
from collections import defaultdict
import numpy as np
from scipy import stats

recs = {}
for f in sys.argv[1:]:
    if f.endswith("_llmdebug.json"): continue   # the glob also matches LLM debug logs
    for r in json.load(open(f)):
        if "scene" not in r: continue
        recs[(r["objective"], r["scene"], r["seed"], r["generator"])] = r
R = list(recs.values())
SLA = ["tard", "miss", "slo_wait", "mix", "premium"]
CTRL = ["p95", "wflow"]
GENS = ["pool_full", "knowledge", "knowledge_tuned", "llm_zeroshot", "llm_tuned", "llm_feedback", "llm_skill"]
NAME = {"pool_full": "pool"}
LIBN = {"FIFO", "SPT", "LPT", "EDF", "WSJF", "ResourceEfficient"}
sel = lambda **k: [r for r in R if all(r[a] == b for a, b in k.items())]
pct = lambda x: f"{100 * x:.0f}" if x == x else "--"

print("% ---- Table 4 (tab:generate): BB % per objective | Rel % (SLA mean) | LLM-written % | Evals | Calls")
for g in GENS:
    L = sel(generator=g)
    if not L: continue
    cells = [pct(np.mean([r["bb"] for r in sel(generator=g, objective=o)])) if sel(generator=g, objective=o) else "--" for o in SLA + CTRL]
    ls = [r for r in L if r["objective"] in SLA]
    rel = f"{100 * np.mean([r['rel'] for r in ls]):+.1f}" if ls else "--"
    own = pct(np.mean(["def priority" in r["rule"] for r in ls])) if g.startswith("llm_") and ls else "---"
    ev = np.mean([r.get("evals") or 0 for r in L]); calls = np.mean([r.get("llm_calls", 0) for r in L])
    print(f"\\gen{{{NAME.get(g, g).replace('_', chr(92) + '_')}}} & " + " & ".join(cells[:5]) + " & " + " & ".join(cells[5:]) +
          f" & {rel} & {own} & {ev:.0f} & {calls:.0f} \\\\")

print("\n% ---- Table 5 (tab:govern), pooled over the five SLA objectives")
for g in ["pool_full", "knowledge_tuned", "llm_tuned", "llm_skill"]:
    L = [r for r in sel(generator=g) if r["objective"] in SLA and "heldout_eb" in r]
    if not L: continue
    dep = np.mean([r["rule"] not in LIBN for r in L])
    row = [pct(dep), pct(np.mean([r["violation"] for r in L])), f"{100 * max(r['rel'] for r in L):+.0f}",
           pct(np.mean([r["dep_t"] for r in L])), pct(np.mean([r["violation_t"] for r in L])), f"{100 * max(r['rel_t'] for r in L):+.0f}",
           pct(np.mean([r["dep_eb"] for r in L])), pct(np.mean([r["violation_eb"] for r in L]))]
    print(f"\\gen{{{NAME.get(g, g).replace('_', chr(92) + '_')}}} & " + " & ".join(row) + f" \\\\   % n={len(L)}")

print("\n% ---- Table S2 (tabS:scene): llm_skill BB per scene (knowledge_tuned in parentheses) | Holm-corrected Wilcoxon p")
ps = {}
for o in SLA:
    a = {(r["scene"], r["seed"]): r["heldout"] for r in sel(objective=o, generator="llm_skill")}
    b = {(r["scene"], r["seed"]): r["heldout"] for r in sel(objective=o, generator="knowledge_tuned")}
    k = sorted(set(a) & set(b)); d = np.array([a[x] - b[x] for x in k])
    ps[o] = float(stats.wilcoxon(d).pvalue) if len(d) >= 6 and not np.allclose(d, 0) else float("nan")
order = sorted([o for o in SLA if ps[o] == ps[o]], key=lambda o: ps[o]); m = len(order); holm = {}; run = 0.0
for i, o in enumerate(order):
    run = max(run, min(1.0, (m - i) * ps[o])); holm[o] = run
for o in SLA:
    cells = []
    for sc in ["baseline", "S1_runtime_shift", "S2_resource_drop", "S4_novel_task"]:
        x = sel(objective=o, scene=sc, generator="llm_skill"); y = sel(objective=o, scene=sc, generator="knowledge_tuned")
        cells.append(f"{pct(np.mean([r['bb'] for r in x]))} ({pct(np.mean([r['bb'] for r in y]))})" if x and y else "--")
    hp = holm.get(o, float("nan"))
    print(f"\\gen{{{o.replace('_', chr(92) + '_')}}} & " + " & ".join(cells) + (f" & {hp:.3g}" if hp == hp else " & --") + " \\\\")

print("\n% ---- H6 pooled over SLA objectives: llm_skill vs knowledge_tuned (paired by objective/scene/seed; relative held-out cost)")
for key, lab in (("rel", "no gate"), ("rel_t", "paired t gate"), ("rel_eb", "paired EB gate")):
    a = {(r["objective"], r["scene"], r["seed"]): r.get(key) for r in R if r["generator"] == "llm_skill" and r["objective"] in SLA}
    b = {(r["objective"], r["scene"], r["seed"]): r.get(key) for r in R if r["generator"] == "knowledge_tuned" and r["objective"] in SLA}
    k = [x for x in sorted(set(a) & set(b)) if a[x] is not None and b[x] is not None]
    if len(k) < 6: continue
    d = np.array([a[x] - b[x] for x in k])
    p = float(stats.wilcoxon(d).pvalue) if not np.allclose(d, 0) else float("nan")
    win = np.mean(d < -1e-9); loss = np.mean(d > 1e-9)
    print(f"% {lab:<15} n={len(k)}  mean diff {100*d.mean():+.2f} pp  median {100*np.median(d):+.2f} pp  "
          f"llm_skill better in {100*win:.0f}% / worse in {100*loss:.0f}% of cells  Wilcoxon p={p:.3g}")
