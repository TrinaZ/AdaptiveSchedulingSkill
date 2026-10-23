"""
h8_analysis.py -- pre-registered H8 tests on the Alibaba trace replay (see H8_preregistration.md).
Usage: python3 h8_analysis.py results/trace_*.json
"""
import sys, json
import numpy as np
from scipy import stats

R = {}
for f in sys.argv[1:]:
    if f.endswith("_llmdebug.json"): continue
    for r in json.load(open(f)):
        if "scene" in r: R[(r["objective"], r["scene"], r["seed"], r["generator"])] = r
gens = sorted({k[3] for k in R}); objs = sorted({k[0] for k in R})
print(f"{'generator':<24}{'n':>5}{'BB%':>6}{'rel%':>8}{'gated%':>8}{'dep_t%':>8}{'viol%':>7}{'viol_t%':>8}{'dep_eb%':>8}{'viol_eb%':>9}")
for g in gens:
    L = [r for k, r in R.items() if k[3] == g]
    m = lambda key: 100 * np.mean([bool(r.get(key, False)) for r in L])
    rt = [r["rel_t"] for r in L if r.get("rel_t") is not None]
    print(f"{g:<24}{len(L):>5}{m('bb'):>6.0f}{100*np.mean([r['rel'] for r in L]):>8.1f}{(100*np.mean(rt) if rt else float('nan')):>8.1f}"
          f"{m('dep_t'):>8.0f}{m('violation'):>7.1f}{m('violation_t'):>8.1f}{m('dep_eb'):>8.0f}{m('violation_eb'):>9.1f}")
print("\nper objective, mean rel % (gated):")
for o in objs:
    print(f"  {o:<9}" + "  ".join(f"{g}: {100*np.mean([r['rel'] for k, r in R.items() if k[0]==o and k[3]==g]):+.1f} ({100*np.mean([r['rel_t'] for k, r in R.items() if k[0]==o and k[3]==g and r.get('rel_t') is not None]):+.1f})"
                           for g in gens if any(k[0]==o and k[3]==g for k in R)))

def paired(a, b, key, alt):
    ks = [k[:3] for k in R if k[3] == a and k[:3] + (b,) in R]
    d = np.array([R[k + (a,)][key] - R[k + (b,)][key] for k in ks if R[k + (a,)].get(key) is not None and R[k + (b,)].get(key) is not None])
    if len(d) < 6: return f"n={len(d)} (too few)"
    p = stats.wilcoxon(d, alternative=alt).pvalue if not np.allclose(d, 0) else 1.0
    return f"n={len(d)} mean {100*d.mean():+.2f}pp median {100*np.median(d):+.2f}pp W/T/L {(d<-1e-9).sum()}/{(abs(d)<=1e-9).sum()}/{(d>1e-9).sum()} p={p:.3g}"

print("\nheld-out runtime errors (Amendment 1):", {g: sum(bool(r.get("heldout_error")) for k, r in R.items() if k[3] == g) for g in gens})
print("\nH8a governance (confirmatory): gated violations <= 5% and ungated > gated")
for g in [x for x in ("knowledge_tuned", "llm_skill", "llm_skill@ds-reasoner") if x in gens]:
    L = [r for k, r in R.items() if k[3] == g]
    u = np.array([bool(r.get("violation")) for r in L]); t = np.array([bool(r.get("violation_t")) for r in L])
    b, c = int((u & ~t).sum()), int((~u & t).sum())   # discordant pairs
    p = stats.binomtest(b, b + c, 0.5, alternative="greater").pvalue if b + c else 1.0
    print(f"  {g:<24} ungated {100*u.mean():5.1f}%  gated {100*t.mean():5.1f}%  (<=5%: {'yes' if t.mean() <= 0.05 else 'NO'})  McNemar exact one-sided p={p:.3g}")
print("H8b (standard LLM, two-sided)   llm_skill vs knowledge_tuned [rel]:   ", paired("llm_skill", "knowledge_tuned", "rel", "two-sided"))
print("                                 llm_skill vs knowledge_tuned [rel_t]: ", paired("llm_skill", "knowledge_tuned", "rel_t", "two-sided"))
if "llm_skill@ds-reasoner" in gens:
    print("H8c (reasoner, one-sided)        reasoner < knowledge_tuned [rel]:     ", paired("llm_skill@ds-reasoner", "knowledge_tuned", "rel", "less"))
    print("                                 reasoner < knowledge_tuned [rel_t]:   ", paired("llm_skill@ds-reasoner", "knowledge_tuned", "rel_t", "less"))
if "gp100" in gens:
    print("(context) gp100 vs knowledge_tuned [rel]:                              ", paired("gp100", "knowledge_tuned", "rel", "two-sided"))
