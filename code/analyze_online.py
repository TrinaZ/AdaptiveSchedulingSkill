"""
Analyze online_sla_bench.py results and check the pre-registered hypotheses H1-H5
(see 让LLM发挥价值_实验重设计.md, section 4).

Usage: python3 analyze_online.py results1.json [results2.json ...] --fig online_fig.png
Records from several files are merged; duplicates (same objective/scene/seed/generator)
keep the last occurrence.
"""
import sys, json, argparse
from collections import defaultdict
import numpy as np
from scipy import stats

ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--fig", default=None)
a = ap.parse_args()
recs = {}
for f in a.files:
    if f.endswith("_llmdebug.json"): continue   # the glob also matches LLM debug logs
    for r in json.load(open(f)):
        if "scene" not in r: continue
        recs[(r["objective"], r["scene"], r["seed"], r["generator"])] = r
R = list(recs.values())
key = lambda r: (r["objective"], r["scene"], r["seed"])
by = defaultdict(dict)
for r in R: by[key(r)][r["generator"]] = r

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)

def rel_dep(r):  # held-out cost relative to the library rule the system would deploy (lower = better)
    d = r.get("deployed_library_heldout")
    return (r["heldout"] - d) / max(abs(d), 1e-9) if d is not None else np.nan

gens = sorted({r["generator"] for r in R}); objs = sorted({r["objective"] for r in R})
print(f"{'objective':<10}{'generator':<16}{'n':>4}{'BB %':>7}{'95% CI':>15}{'rel vs deployed lib %':>23}{'LLM calls':>11}{'evals':>7}")
table = {}
for o in objs:
    for g in gens:
        L = [r for r in R if r["objective"] == o and r["generator"] == g]
        if not L: continue
        k = sum(r["bb"] for r in L); lo, hi = wilson(k, len(L))
        rd = np.nanmean([rel_dep(r) for r in L]); calls = np.mean([r.get("llm_calls", 0) for r in L])
        table[(o, g)] = dict(n=len(L), bb=k / len(L), lo=lo, hi=hi, rd=rd)
        ev = np.mean([r.get("evals", 0) or 0 for r in L])
        print(f"{o:<10}{g:<16}{len(L):>4}{100*k/len(L):>6.0f}%  [{100*lo:4.0f},{100*hi:4.0f}]{100*rd:>21.2f}{calls:>11.1f}{ev:>7.0f}")

def paired(o, g1, g2):
    """Wilcoxon signed-rank on held-out cost (paired by scene, seed); negative median diff = g1 better."""
    d = [by[k][g1]["heldout"] - by[k][g2]["heldout"] for k in by if k[0] == o and g1 in by[k] and g2 in by[k]]
    d = np.array(d)
    if len(d) < 3 or np.allclose(d, 0): return len(d), np.nan, float(np.median(d)) if len(d) else np.nan
    return len(d), float(stats.wilcoxon(d).pvalue), float(np.median(d))

SLA = [o for o in ["tard", "miss", "slo_wait", "premium", "mix"] if o in objs]
CTRL = [o for o in ["p95", "wflow"] if o in objs]
llms = [g for g in gens if g.startswith("llm_")]
if llms:
    print("\nPaired comparisons (held-out cost, Wilcoxon; diff<0 means the LLM generator is better)")
    for o in objs:
        for g in llms:
            for ref in ["pool_full", "knowledge", "knowledge_tuned", "library"]:
                if ref in gens:
                    n, p, md = paired(o, g, ref)
                    print(f"  {o:<9}{g:<14}vs {ref:<10} n={n:<3} median diff={md:9.3f}  p={p:.3g}")
    print("\nPre-registered hypotheses")
    best_llm = {o: max(llms, key=lambda g: table.get((o, g), {}).get("bb", -1)) for o in SLA}
    wins = [o for o in SLA if (o, best_llm[o]) in table and (o, "pool_full") in table and
            table[(o, best_llm[o])]["bb"] > table[(o, "pool_full")]["bb"] and paired(o, best_llm[o], "pool_full")[2] < 0]
    print(f"  H1 LLM > pool_full on >={min(4, len(SLA))} of {len(SLA)} SLA objectives: {len(wins)} ({', '.join(wins)}) -> {'PASS' if len(wins) >= min(4, len(SLA)) else 'not met'}")
    if "llm_feedback" in gens and "knowledge" in gens:
        rat = [table[(o, 'llm_feedback')]['bb'] / max(table[(o, 'knowledge')]['bb'], 1e-9) for o in SLA if (o, 'llm_feedback') in table and table.get((o, 'knowledge'), {}).get('bb', 0) > 0]
        print(f"  H2 llm_feedback BB >= 80% of knowledge: mean ratio {np.mean(rat):.2f} -> {'PASS' if rat and np.mean(rat) >= 0.8 else 'not met'}")
    if "llm_tuned" in gens and "llm_zeroshot" in gens:
        better = sum(table[(o, 'llm_tuned')]['bb'] > table[(o, 'llm_zeroshot')]['bb'] for o in SLA if (o, 'llm_tuned') in table)
        print(f"  H3 llm_tuned > llm_zeroshot on {better}/{len(SLA)} SLA objectives")
    for o in CTRL:
        print(f"  H5 control objective {o}: " + ", ".join(f"{g} BB {100*table[(o,g)]['bb']:.0f}%" for g in llms if (o, g) in table))
    if "llm_skill" in gens and "knowledge_tuned" in gens:
        print("  H6 llm_skill vs knowledge_tuned (same knowledge, same evaluation budget; the LLM's added value):")
        for o in objs:
            n, p, md = paired(o, "llm_skill", "knowledge_tuned")
            L = [r for r in R if r["objective"] == o and r["generator"] == "llm_skill"]
            own = np.mean(["def priority" in r["rule"] for r in L]) if L else np.nan
            print(f"    {o:<9} BB {100*table.get((o,'llm_skill'),{}).get('bb',np.nan):3.0f}% vs {100*table.get((o,'knowledge_tuned'),{}).get('bb',np.nan):3.0f}%   "
                  f"median diff {md:9.3f}  p={p:.3g}  LLM-written rule deployed in {100*own:3.0f}% of cases")
    print("  H4 (gate keeps every deployment within the bound) is checked by online_governance.py")

if any("heldout_eb" in r for r in R):
    LIBN = ("FIFO", "SPT", "LPT", "EDF", "WSJF", "ResourceEfficient")
    print("\nGenerate + Govern. viol = held-out cost > (1+tau) x the rule the system would otherwise run;")
    print("rel = held-out cost vs best library rule on held-out (mean / worst case). eb = empirical-Bernstein gate, t = paired t gate")
    print(f"{'objective':<10}{'generator':<16}{'dep%':>6}{'viol%':>7}{'mean%':>7}{'worst%':>8} |{'dep_t%':>7}{'viol_t%':>8}{'mean_t%':>8}{'worst_t%':>9}{'BB_t%':>7} |{'dep_eb%':>8}{'viol_eb%':>9}{'mean_eb%':>9}")
    for o in objs:
        for g in gens:
            L = [r for r in R if r["objective"] == o and r["generator"] == g and "heldout_eb" in r]
            if not L or g == "library": continue
            m = lambda k: 100 * np.mean([r[k] for r in L]); w = lambda k: 100 * max(r[k] for r in L)
            dep = 100 * np.mean([r["rule"] not in LIBN for r in L])
            print(f"{o:<10}{g:<16}{dep:>5.0f}%{m('violation'):>6.0f}%{m('rel'):>7.1f}{w('rel'):>8.1f} |{m('dep_t'):>6.0f}%{m('violation_t'):>7.0f}%"
                  f"{m('rel_t'):>8.1f}{w('rel_t'):>9.1f}{m('bb_t'):>6.0f}% |{m('dep_eb'):>7.0f}%{m('violation_eb'):>8.0f}%{m('rel_eb'):>9.1f}")

if a.fig:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    order = [g for g in ["library", "pool_old", "pool_full", "knowledge", "knowledge_tuned", "llm_zeroshot", "llm_tuned", "llm_feedback", "llm_skill"] if g in gens]
    cols = {"library": "#9aa0a6", "pool_old": "#c6c6c6", "pool_full": "#7f8c8d", "knowledge": "#2c7fb8",
            "knowledge_tuned": "#08519c", "llm_zeroshot": "#fdae6b", "llm_tuned": "#fd8d3c", "llm_feedback": "#e6550d", "llm_skill": "#a63603"}
    fig, ax = plt.subplots(figsize=(1.3 * len(objs) + 2, 3.2))
    w = 0.8 / len(order); x = np.arange(len(objs))
    for i, g in enumerate(order):
        v = [100 * table.get((o, g), {}).get("bb", np.nan) for o in objs]
        lo = [100 * (table[(o, g)]["bb"] - table[(o, g)]["lo"]) if (o, g) in table else 0 for o in objs]
        hi = [100 * (table[(o, g)]["hi"] - table[(o, g)]["bb"]) if (o, g) in table else 0 for o in objs]
        ax.bar(x + (i - len(order) / 2 + 0.5) * w, v, w, yerr=[lo, hi], capsize=1.5, label=g, color=cols.get(g, None), linewidth=0, error_kw=dict(ecolor="#555555", elinewidth=0.6))
    ax.set_xticks(x); ax.set_xticklabels(objs); ax.set_ylabel("Beats-Best, held-out (%)"); ax.set_ylim(0, 100)
    ax.legend(ncol=4, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(a.fig, dpi=200)
    print(f"\nfigure -> {a.fig}")
