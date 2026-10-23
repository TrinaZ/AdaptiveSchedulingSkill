"""
per_objective.py -- Supplementary Table S19: per-objective contribution of the LLM (post hoc).

For each objective and generator: held-out Beats-Best (%), mean Rel (%), and the two-sided paired Wilcoxon
p-value on held-out cost against knowledge_tuned over the 40 cells, Holm-corrected across the objectives of
each block (the convention of Table S2). No LLM call is made; the script only re-reads stored results.

Usage (from the repository root):
  python3 code/per_objective.py --results results
"""
import json, glob, os, argparse
import numpy as np
from scipy import stats

ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results"); a = ap.parse_args()
SLA = ["tard", "miss", "slo_wait", "mix", "premium"]; H7 = ["tiered", "typemiss", "decay"]


def load(paths):
    d = []
    for p in paths:
        for r in json.load(open(p)):
            if "scene" not in r: continue
            if r.get("backend_tag"): r["generator"] = r["generator"] + "@" + r["backend_tag"]
            d.append(r)
    return d


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = [1.0] * m; run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[k])); adj[k] = run
    return adj


def compare(d, g):
    kt = {(r["scene"], r["seed"]): r for r in d if r["generator"] == "knowledge_tuned"}
    L = {(r["scene"], r["seed"]): r for r in d if r["generator"] == g}
    ks = sorted(set(L) & set(kt))
    if not ks: return None
    x = np.array([L[k]["heldout"] for k in ks]); y = np.array([kt[k]["heldout"] for k in ks])
    p = stats.wilcoxon(x, y).pvalue if np.any(x != y) else 1.0
    return dict(n=len(ks), bb=100 * np.mean([L[k]["bb"] for k in ks]), rel=100 * np.mean([L[k]["rel"] for k in ks]),
                bb_kt=100 * np.mean([kt[k]["bb"] for k in ks]), rel_kt=100 * np.mean([kt[k]["rel"] for k in ks]),
                wins=int((x < y).sum()), losses=int((x > y).sum()), p=p)


def block(objs, files_for, gens):
    res = {}
    for g in gens:
        rows = {o: compare(load(files_for(o)), g) for o in objs}
        valid = [o for o in objs if rows[o]]
        for o, h in zip(valid, holm([rows[o]["p"] for o in valid])): rows[o]["holm"] = h
        res[g] = rows
    return res


R = a.results
sla = block(SLA, lambda o: [f"{R}/backend_ds-reasoner_{o}.json", f"{R}/backend_qwen-max_{o}.json"],
            ["llm_skill@ds-reasoner", "llm_skill@qwen-max"])
h7 = block(H7, lambda o: [f"{R}/h7_nollm_{o}.json", f"{R}/h7_llm_{o}.json", f"{R}/h7_backend_ds-reasoner_{o}.json"],
           ["llm_skill", "llm_skill@ds-reasoner", "expert"])
for name, res, objs in [("SLA objectives (DeepSeek-V3 row: see Table S2)", sla, SLA), ("H7 objectives", h7, H7)]:
    print(name)
    for o in objs:
        line = f"  {o:<9}"
        for g, rows in res.items():
            r = rows[o]
            if r: line += f" | {g}: BB {r['bb']:.0f} (kt {r['bb_kt']:.0f}) rel {r['rel']:+.1f} (kt {r['rel_kt']:+.1f}) W/L {r['wins']}/{r['losses']} Holm p {r['holm']:.2g}"
        print(line)
json.dump({"sla": sla, "h7": h7}, open(os.path.join(R, "per_objective.json"), "w"), indent=1, default=float)
