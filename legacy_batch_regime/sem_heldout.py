"""Re-score the 320 single-call semantic-path rules of the batch regime (wgs_matrix_8seeds.json)
on HELD-OUT instances (seeds disjoint from search 0-7 and validation 500-507). No LLM calls."""
import sys, json, argparse
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--src"); ap.add_argument("--out"); ap.add_argument("--n", type=int, default=200)
ap.add_argument("--seed_offset", type=int, default=50); a = ap.parse_args()
sys.path.insert(0, a.src)
from problems.genomic_scheduling.prob import GenomicScheduling, BASELINES
from laosa import LLMAlgorithmAdapter
ad = LLMAlgorithmAdapter.__new__(LLMAlgorithmAdapter)
R = [r for r in json.load(open(f"{a.src}/results/wgs_matrix_8seeds.json")) if "sem" in r["change_type"]]
RG = {r["run_id"]: r for r in json.load(open(f"{a.src}/results/wgs_matrix_8seeds_regated.json"))}
cache = {}; out = []
for r in R:
    s = int(r["seed"]) + a.seed_offset; k = (r["scene_after"], s)
    if k not in cache:
        p = GenomicScheduling(r["scene_after"], n_instances=a.n, n_tasks=20, seed=s)
        cache[k] = (p, {nm: ad._evaluate_code(c, p) for nm, c in BASELINES.items()})
    p, bl = cache[k]
    best_h = min(v for v in bl.values() if v is not None)
    ms = ad._evaluate_code(r["adapted_code"], p)
    fb = bl.get(r["best_fixed_baseline"])
    dep = r["deployment_source"] == "llm_adapted"
    dep_rg = RG[r["run_id"]].get("deployment_source_regated") == "deployed_candidate"
    out.append(dict(run_id=r["run_id"], seed=r["seed"], bb_insample=r["beats_best_baseline"] in (True, "True"),
                    gate=dep, gate_regated=dep_rg, heldout_ms=ms, heldout_best=best_h, heldout_fallback=fb,
                    bb_heldout=bool(ms is not None and ms < best_h), rel=(ms / best_h - 1) if ms else None))
json.dump(out, open(a.out, "w"), indent=1)
n = len(out); f = lambda c: sum(c(o) for o in out)
print("n", n, "| in-sample BB", f(lambda o: o["bb_insample"]), "| held-out BB (all candidates)", f(lambda o: o["bb_heldout"]),
      "| median rel %", round(100 * np.median([o["rel"] for o in out if o["rel"] is not None]), 2))
for g in ["gate", "gate_regated"]:
    D = [o for o in out if o[g]]
    dep_bb = sum(o["bb_heldout"] for o in D)
    viol = sum(o["heldout_ms"] > 1.02 * o["heldout_fallback"] for o in D)
    print(g, "deployed", len(D), "held-out BB of deployed", dep_bb, "violations vs fallback", viol)
