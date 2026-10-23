"""
h7_analysis.py -- the pre-registered H7 tests (see H7_preregistration.md). Nothing here may be changed after the LLM run.

Usage: python3 h7_analysis.py results/h7_nollm_*.json results/h7_llm_*.json
"""
import sys, json
import numpy as np
from scipy import stats

R = {}
for f in sys.argv[1:]:
    if f.endswith("_llmdebug.json"): continue
    for r in json.load(open(f)):
        if "scene" in r: R[(r["objective"], r["scene"], r["seed"], r["generator"])] = r
OBJ = ["tiered", "typemiss", "decay"]
gens = sorted({k[3] for k in R})
print(f"{'objective':<10}{'generator':<16}{'n':>4}{'BB%':>6}{'mean rel%':>10}{'gated rel%':>11}{'dep_t%':>7}{'viol_t%':>8}{'dep_eb%':>8}{'LLM-wr%':>8}")
for o in OBJ:
    for g in gens:
        L = [r for k, r in R.items() if k[0] == o and k[3] == g]
        if not L: continue
        f = lambda key: 100 * np.mean([bool(r.get(key, False)) for r in L])
        relt = [r["rel_t"] for r in L if r.get("rel_t") is not None]
        print(f"{o:<10}{g:<16}{len(L):>4}{100*np.mean([r['bb'] for r in L]):>6.0f}{100*np.mean([r['rel'] for r in L]):>10.1f}"
              f"{(100*np.mean(relt) if relt else float('nan')):>11.1f}{f('dep_t'):>7.0f}{f('violation_t'):>8.0f}{f('dep_eb'):>8.0f}"
              + (f"{100*np.mean(['def priority' in r['rule'] for r in L]):>8.0f}" if g.startswith("llm_") else f"{'-':>8}"))

def paired(a, b, key, alt):
    ks = [k[:3] for k in R if k[3] == a and (k[0], k[1], k[2], b) in R and k[0] in OBJ]
    d = np.array([R[k + (a,)][key] - R[k + (b,)][key] for k in ks if R[k + (a,)].get(key) is not None and R[k + (b,)].get(key) is not None])
    if len(d) < 6: return f"n={len(d)} (too few)"
    p = stats.wilcoxon(d, alternative=alt).pvalue if not np.allclose(d, 0) else 1.0
    return f"n={len(d)}, mean diff {100*d.mean():+.2f}pp, median {100*np.median(d):+.2f}pp, wins/ties/losses {int((d<-1e-9).sum())}/{int((abs(d)<=1e-9).sum())}/{int((d>1e-9).sum())}, p={p:.3g}"

print("\nH7  (primary)   llm_skill < knowledge_tuned, rel, one-sided:     ", paired("llm_skill", "knowledge_tuned", "rel", "less"))
print("H7b (secondary) llm_skill < knowledge_tuned, rel_t, one-sided:   ", paired("llm_skill", "knowledge_tuned", "rel_t", "less"))
L = [r for k, r in R.items() if k[3] == "llm_skill" and k[0] in OBJ]
if L: print(f"H7b safety: llm_skill violation_t = {100*np.mean([r.get('violation_t', False) for r in L]):.1f}% (threshold 5%)")
print("H7c (exploratory) llm_skill vs expert, rel, two-sided:          ", paired("llm_skill", "expert", "rel", "two-sided"))
print("    (context)    expert < knowledge_tuned, rel, one-sided:       ", paired("expert", "knowledge_tuned", "rel", "less"))
print("    (context)    llm_zeroshot < knowledge_tuned, rel, one-sided: ", paired("llm_zeroshot", "knowledge_tuned", "rel", "less"))
for o in OBJ:
    sub = {k: v for k, v in R.items() if k[0] == o}
    if not sub: continue
    ks = [k[:3] for k in sub if k[3] == "llm_skill" and (k[0], k[1], k[2], "knowledge_tuned") in sub]
    d = np.array([sub[k + ("llm_skill",)]["rel"] - sub[k + ("knowledge_tuned",)]["rel"] for k in ks])
    if len(d) >= 6: print(f"   per objective {o:<9} llm_skill - knowledge_tuned: {100*d.mean():+.2f}pp (n={len(d)}, descriptive)")

# ---- H7' (addendum): deepseek-reasoner, declared after H7, before its run
if any(k[3] == "llm_skill@ds-reasoner" for k in R):
    print("\nH7' (declared after H7)  llm_skill@ds-reasoner < knowledge_tuned, rel, one-sided:   ", paired("llm_skill@ds-reasoner", "knowledge_tuned", "rel", "less"))
    print("H7'b                    llm_skill@ds-reasoner < knowledge_tuned, rel_t, one-sided: ", paired("llm_skill@ds-reasoner", "knowledge_tuned", "rel_t", "less"))
    L = [r for k, r in R.items() if k[3] == "llm_skill@ds-reasoner" and k[0] in OBJ]
    print(f"H7'b safety: violation_t = {100*np.mean([r.get('violation_t', False) for r in L]):.1f}% (threshold 5%)")
    print("H7'c (exploratory)      llm_skill@ds-reasoner vs expert, rel, two-sided:          ", paired("llm_skill@ds-reasoner", "expert", "rel", "two-sided"))
    print("    (context)           llm_skill@ds-reasoner vs llm_skill (chat), rel, two-sided: ", paired("llm_skill@ds-reasoner", "llm_skill", "rel", "two-sided"))
    for o in OBJ:
        ks = [k[:3] for k in R if k[0] == o and k[3] == "llm_skill@ds-reasoner" and k[:3] + ("knowledge_tuned",) in R]
        d = np.array([R[k + ("llm_skill@ds-reasoner",)]["rel"] - R[k + ("knowledge_tuned",)]["rel"] for k in ks])
        if len(d): print(f"   per objective {o:<9} reasoner - knowledge_tuned: {100*d.mean():+.2f}pp (n={len(d)}, descriptive)")
