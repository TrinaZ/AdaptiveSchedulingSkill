"""gp_compare.py -- GP hyper-heuristic vs knowledge_tuned / llm_skill on the same cells (paired by objective, scene, seed).
Usage: python3 gp_compare.py results/gp_*.json results/online_full_*.json results/h7_*.json"""
import sys, json
import numpy as np
from scipy import stats
R = {}
for f in sys.argv[1:]:
    if f.endswith("_llmdebug.json"): continue
    for r in json.load(open(f)):
        if "scene" in r: R[(r["objective"], r["scene"], r["seed"], r["generator"])] = r
objs = sorted({k[0] for k in R if k[3].startswith("gp")})
G = ["gp100", "gp1000", "knowledge_tuned", "llm_skill", "expert"]
print(f"{'objective':<10}" + "".join(f"{g:>22}" for g in G) + "   (BB% / mean rel% / gated rel%)")
for o in objs:
    row = f"{o:<10}"
    for g in G:
        L = [r for k, r in R.items() if k[0] == o and k[3] == g]
        if L:
            bb = 100 * np.mean([r["bb"] for r in L]); rl = 100 * np.mean([r["rel"] for r in L])
            rt = 100 * np.mean([r["rel_t"] for r in L if r.get("rel_t") is not None])
            cell = f"{bb:.0f} / {rl:+.1f} / {rt:+.1f}"
        else: cell = "-"
        row += f"{cell:>22}"
    print(row)
def paired(a, b, key):
    ks = [k[:3] for k in R if k[3] == a and k[:3] + (b,) in R]
    d = np.array([R[k + (a,)][key] - R[k + (b,)][key] for k in ks if R[k + (a,)].get(key) is not None and R[k + (b,)].get(key) is not None])
    if len(d) < 6: return "n<6"
    return f"n={len(d)} mean {100*d.mean():+.2f}pp wins/ties/losses {(d<-1e-9).sum()}/{(abs(d)<=1e-9).sum()}/{(d>1e-9).sum()} p={stats.wilcoxon(d).pvalue if not np.allclose(d,0) else 1:.3g}"
for a, b in [("gp100", "knowledge_tuned"), ("gp100", "llm_skill"), ("gp1000", "knowledge_tuned"), ("gp1000", "llm_skill"), ("gp1000", "gp100")]:
    for key in ("rel", "rel_t"):
        print(f"{a} vs {b} [{key}]: {paired(a, b, key)}")
