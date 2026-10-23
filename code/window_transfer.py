"""
window_transfer.py -- do rules selected on 60-request windows keep their ranking on longer windows? (no LLM)

Takes the rules each generator selected in the main run (online_full_<obj>.json), rebuilds them from the stored
labels/source, and re-evaluates them on held-out instances with N requests per instance (same held-out seeds,
s+50). Reports held-out rel vs the best library rule at that N, and the pooled llm_skill - knowledge_tuned gap.

Usage: python3 window_transfer.py --bench online_sla_bench.py --src <repo> --files results/online_full_mix.json ... \
          --n 300 --n_heldout 30 --out results/window_transfer.json
"""
import sys, json, re, argparse, time
import numpy as np
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--bench", required=True); ap.add_argument("--src", required=True); ap.add_argument("--files", nargs="+", required=True)
ap.add_argument("--n", type=int, default=300); ap.add_argument("--n_heldout", type=int, default=30)
ap.add_argument("--gens", default="library,knowledge_tuned,llm_skill"); ap.add_argument("--out", required=True)
ap.add_argument("--max_cells", type=int, default=0)
a = ap.parse_args()
code = open(a.bench).read().split("\n# ---- fail fast")[0]
sys.argv = ["bench", "--src", a.src, "--out", "/dev/null"]
B = {}; exec(code, B)
make_instance, run_cost, LIB, KNOW, KNOW_FAM, compile_rule, bind = (B[k] for k in
    ["make_instance", "run_cost", "LIB", "KNOW", "KNOW_FAM", "compile_rule", "bind"])
KNL = {lab: f for lab, f, _, _ in KNOW()}; FAM = {n: c for n, c, _ in KNOW_FAM}

def rebuild(label):
    if label in LIB: return LIB[label]
    if label in KNL: return KNL[label]
    body, params = label, {}
    m = re.search(r"\s*# tuned \((.*)\)\s*$", label)
    if m:
        body = label[:m.start()]
        params = {k: float(v) for k, v in (kv.split("=") for kv in m.group(1).split(","))}
    src = FAM.get(body.strip(), body)
    f, _ = compile_rule(src, check=False)
    return bind(f, params)

recs = [r for f in a.files if not f.endswith("_llmdebug.json") for r in json.load(open(f)) if "scene" in r]
gens = a.gens.split(","); cells = sorted({(r["objective"], r["scene"], r["seed"]) for r in recs})
if a.max_cells: cells = cells[::max(1, len(cells) // a.max_cells)][:a.max_cells]
by = {(r["objective"], r["scene"], r["seed"], r["generator"]): r for r in recs}
out = []; t0 = time.time()
for o, sc, s in cells:
    Ih = [make_instance(sc, 1000 + (s + 50) * 10000 + i, a.n, 1.0) for i in range(a.n_heldout)]
    lib = {k: run_cost(f, Ih, o) for k, f in LIB.items()}; best = min(lib.values())
    row = dict(objective=o, scene=sc, seed=s)
    for g in gens:
        r = by.get((o, sc, s, g))
        if r is None: continue
        try: v = run_cost(rebuild(r["rule"]), Ih, o)
        except Exception as e: print("rebuild failed:", g, repr(r["rule"][:80]), e); v = None
        row[g] = None if v is None else (v - best) / max(abs(best), 1e-9)
        row[g + "_rel60"] = r["rel"]
    out.append(row)
    print(f"[{time.time()-t0:5.0f}s] {o} {sc} {s}: " + ", ".join(f"{g} {100*row[g]:+.1f}%" for g in gens if row.get(g) is not None), flush=True)
json.dump(out, open(a.out, "w"), indent=1)
print(f"\nN={a.n} requests per held-out instance ({a.n_heldout} instances); 60-request result in brackets")
for o in sorted({r["objective"] for r in out}):
    R = [r for r in out if r["objective"] == o]
    print(o, "  ".join(f"{g}: {100*np.mean([r[g] for r in R if r.get(g) is not None]):+.1f}% [{100*np.mean([r[g+'_rel60'] for r in R if g+'_rel60' in r]):+.1f}%]" for g in gens))
d = np.array([r["llm_skill"] - r["knowledge_tuned"] for r in out if r.get("llm_skill") is not None and r.get("knowledge_tuned") is not None])
d60 = np.array([r["llm_skill_rel60"] - r["knowledge_tuned_rel60"] for r in out if "llm_skill_rel60" in r and "knowledge_tuned_rel60" in r])
print(f"llm_skill - knowledge_tuned: N={a.n}: {100*d.mean():+.2f}pp (Wilcoxon p={stats.wilcoxon(d).pvalue:.3g}); "
      f"N=60: {100*d60.mean():+.2f}pp; sign agreement of per-cell differences: "
      f"{np.mean(np.sign(d) == np.sign(d60[:len(d)])):.0%}")
