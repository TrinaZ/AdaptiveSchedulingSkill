"""
compare_backends.py -- Table S6: llm_skill with different LLM backends (online regime, SLA objectives).

Usage:
  python3 compare_backends.py results/online_full_*.json results/backend_*.json \
      --price deepseek-chat=0.27,1.10 --price qwen-max=1.6,6.4 ...
--price model=in,out gives USD per million input/output tokens (look them up on the provider's page on the
day you run; leave out to print tokens only). The DeepSeek run without --tag appears as plain "llm_skill".
"""
import sys, json, argparse
from collections import defaultdict
import numpy as np
from scipy import stats

ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--price", action="append", default=[])
ap.add_argument("--objectives", default="tard,miss,slo_wait,mix,premium", help="restrict ALL generators to these objectives, so backends are compared on the same cells")
a = ap.parse_args()
PRICE = {}
for p in a.price:
    m, v = p.split("="); i, o = v.split(","); PRICE[m] = (float(i), float(o))
SLA = a.objectives.split(",")
LIBN = {"FIFO", "SPT", "LPT", "EDF", "WSJF", "ResourceEfficient"}
recs = {}
for f in a.files:
    if f.endswith("_llmdebug.json"): continue
    for r in json.load(open(f)):
        if "scene" not in r or r["objective"] not in SLA: continue
        recs[(r["objective"], r["scene"], r["seed"], r["generator"])] = r
R = list(recs.values())
kt = {(r["objective"], r["scene"], r["seed"]): r for r in R if r["generator"] == "knowledge_tuned"}
gens = sorted({r["generator"] for r in R if r["generator"].startswith("llm_skill")})
pct = lambda x: f"{100 * x:.0f}"
print(f"{'generator':<28}{'model':<18}{'n':>4}{'BB%':>5}{'LLM-wr%':>8}{'dep_t%':>7}{'viol%':>6}{'viol_t%':>8}{'dep_eb%':>8}"
      f"{'tok/cell':>9}{'USD/cell':>9}{'s/cell':>7}  H6 vs knowledge_tuned (pooled, t gate)")
for g in gens:
    L = [r for r in R if r["generator"] == g]
    model = L[0].get("model") or "deepseek-chat"
    tin = np.mean([r.get("tokens_in", 0) or 0 for r in L]); tout = np.mean([r.get("tokens_out", 0) or 0 for r in L])
    usd = f"{(tin * PRICE[model][0] + tout * PRICE[model][1]) / 1e6:.4f}" if model in PRICE else "--"
    sec = np.mean([r.get("llm_seconds", 0) or 0 for r in L])
    pairs = [(r, kt[(r["objective"], r["scene"], r["seed"])]) for r in L if (r["objective"], r["scene"], r["seed"]) in kt]
    h6 = ""
    for key in ("rel", "rel_t"):
        d = np.array([x[key] - y[key] for x, y in pairs if x.get(key) is not None and y.get(key) is not None])
        if len(d) >= 6 and not np.allclose(d, 0):
            h6 += f"{'gated ' if key == 'rel_t' else ''}{100 * d.mean():+.2f}pp p={stats.wilcoxon(d).pvalue:.3g}; "
    print(f"{g:<28}{model:<18}{len(L):>4}{pct(np.mean([r['bb'] for r in L])):>5}{pct(np.mean(['def priority' in r['rule'] for r in L])):>8}"
          f"{pct(np.mean([r.get('dep_t', False) for r in L])):>7}{pct(np.mean([r.get('violation', False) for r in L])):>6}"
          f"{pct(np.mean([r.get('violation_t', False) for r in L])):>8}{pct(np.mean([r.get('dep_eb', False) for r in L])):>8}"
          f"{tin + tout:>9.0f}{usd:>9}{sec:>7.1f}  {h6}")
print("\nLaTeX rows for Table S6 (Backend & BB & LLM-written & Gate pass (t) & Viol. (t) & USD):")
for g in gens:
    L = [r for r in R if r["generator"] == g]; model = L[0].get("model") or "deepseek-chat"
    tin = np.mean([r.get("tokens_in", 0) or 0 for r in L]); tout = np.mean([r.get("tokens_out", 0) or 0 for r in L])
    usd = f"{(tin * PRICE[model][0] + tout * PRICE[model][1]) / 1e6:.3f}" if model in PRICE else "--"
    print(f"\\texttt{{{model}}} & {pct(np.mean([r['bb'] for r in L]))} & {pct(np.mean(['def priority' in r['rule'] for r in L]))} & "
          f"{pct(np.mean([r.get('dep_t', False) for r in L]))} & {pct(np.mean([r.get('violation_t', False) for r in L]))} & {usd} \\\\")
