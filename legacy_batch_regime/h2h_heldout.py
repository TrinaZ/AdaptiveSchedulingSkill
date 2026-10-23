"""Re-evaluate stored head-to-head outputs on HELD-OUT instances (no LLM calls).
In the original h2h, every method selected its output on prob_after and BB was
scored on the same prob_after (in-sample). Here we score each stored adapted_code
and every baseline on fresh instances (disjoint seeds) of the same post-change scene."""
import sys, json, time, argparse
from collections import defaultdict
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--src"); ap.add_argument("--out"); ap.add_argument("--n", type=int, default=100)
ap.add_argument("--seed_offset", type=int, default=50)
a = ap.parse_args()
sys.path.insert(0, a.src)
from problems.genomic_scheduling.prob import GenomicScheduling, BASELINES
from laosa import LLMAlgorithmAdapter
adapter = LLMAlgorithmAdapter.__new__(LLMAlgorithmAdapter)
recs = [r for r in json.load(open(f"{a.src}/results/h2h_raw.json")) if not r.get("error")]
cache = {}
def prob(scene, seed):
    k = (scene, seed)
    if k not in cache:
        p = GenomicScheduling(scene, n_instances=a.n, n_tasks=20, seed=seed)
        bl = {nm: adapter._evaluate_code(c, p) for nm, c in BASELINES.items()}
        cache[k] = (p, bl)
    return cache[k]
out = []; t0 = time.time()
for i, r in enumerate(recs):
    s = int(r["seed"]) + a.seed_offset
    p, bl = prob(r["scene_after"], s)
    best = min(v for v in bl.values() if v is not None)
    ms = adapter._evaluate_code(r["adapted_code"], p)
    out.append({**{k: r[k] for k in ["method","scene_id","change_type","scene_after","baseline_algo","seed","beats_best","adapted_code","llm_calls"]},
                "heldout_ms": ms, "heldout_best_static": best,
                "heldout_bb": bool(ms is not None and ms < best),
                "heldout_rel_to_best": (ms/best - 1) if ms else None})
    if i % 200 == 0: print(i, round(time.time()-t0,1), flush=True)
json.dump(out, open(a.out, "w"), indent=1)
agg = defaultdict(list)
for o in out: agg[(o["method"], o["change_type"])].append(o)
print(f"{'method':<28}{'tier':<15}{'n':>4}{'BB_in':>8}{'BB_held':>9}{'med rel%':>10}")
for (m,t), L in sorted(agg.items()):
    bbin = np.mean([str(x["beats_best"])=="True" for x in L])*100
    bbh = np.mean([x["heldout_bb"] for x in L])*100
    rel = np.median([x["heldout_rel_to_best"] for x in L if x["heldout_rel_to_best"] is not None])*100
    print(f"{m:<28}{t:<15}{len(L):>4}{bbin:>8.1f}{bbh:>9.1f}{rel:>10.2f}")
