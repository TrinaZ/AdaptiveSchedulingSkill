"""diagnose_skip.py -- how often would Diagnose's adequacy test (literature headroom over the library on the 12 search
instances < thr) skip generation in cells where a generated or author-written rule, after the t gate, beats the library? (no LLM)
Usage: python3 diagnose_skip.py --bench online_sla_bench.py --src <repo> --files results/h7_*.json results/online_full_*.json"""
import sys, json, argparse
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--bench", required=True); ap.add_argument("--src", required=True)
ap.add_argument("--files", nargs="+", required=True); ap.add_argument("--thr", type=float, default=0.03); ap.add_argument("--out", default=None)
a = ap.parse_args()
code = open(a.bench).read().split("\n# ---- fail fast")[0]; sys.argv = ["b", "--src", a.src, "--out", "/dev/null"]
B = {}; exec(code, B); mk, run_cost, LIB, KNOW = B["make_instance"], B["run_cost"], B["LIB"], B["KNOW"]; KN = KNOW()
R = {}
for f in a.files:
    if f.endswith("_llmdebug.json"): continue
    for r in json.load(open(f)):
        if "scene" in r: R[(r["objective"], r["scene"], r["seed"], r["generator"])] = r
cells = sorted({k[:3] for k in R}); rows = []
for o, sc, s in cells:
    Is = [mk(sc, 1000 + s * 10000 + i, 60, 1.0) for i in range(12)]
    lib = min(run_cost(f, Is, o) for f in LIB.values())
    kn = min(v for v in (run_cost(f, Is, o) for _, f, _, _ in KN) if v is not None)
    row = dict(objective=o, scene=sc, seed=s, headroom=1 - kn / max(lib, 1e-9))
    for g in ("llm_skill", "expert", "knowledge_tuned", "llm_skill@ds-reasoner"):
        r = R.get((o, sc, s, g))
        if r and r.get("rel_t") is not None: row[g] = r["rel_t"]
    rows.append(row)
if a.out: json.dump(rows, open(a.out, "w"), indent=1)
print(f"Diagnose adequacy threshold {a.thr:.0%}: cells skipped, and among them cells where the gated generator beat the library by >2%")
for o in sorted({r["objective"] for r in rows}):
    L = [r for r in rows if r["objective"] == o]; S = [r for r in L if r["headroom"] < a.thr]
    msg = f"{o:<10} skipped {len(S):>2}/{len(L)}"
    for g in ("llm_skill", "expert"):
        if any(g in r for r in L):
            lost = [r for r in S if r.get(g, 0) < -0.02]
            msg += f" | {g}: missed gains in {len(lost)} skipped cells (mean gated rel of {g} there {100*np.mean([r[g] for r in lost]) if lost else 0:+.1f}%)"
    print(msg)
