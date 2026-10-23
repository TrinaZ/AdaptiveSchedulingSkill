"""
random_gate_trace.py -- random-gate negative control on trace replays (H8, H9, H10; reported without a claim).
Definition as in Supplementary Section S5: a random gate deploys in each cell with the probability at which the paired
t gate deploys (over all cells); a random deployment violates when the ungated deployment violates. We compare the t
gate's gated violations with 10,000 random-gate draws, per generator and pooled over the given generators.
Usage: python3 random_gate_trace.py results/*.json
"""
import sys, json
import numpy as np
R = {}
for f in sys.argv[1:]:
    if f.endswith("_llmdebug.json"): continue
    for r in json.load(open(f)):
        if "scene" in r: R[(r["objective"], r["scene"], r["seed"], r["generator"])] = r
rng = np.random.default_rng(12345)
def ctrl(L, name):
    bad = np.array([bool(r.get("violation")) for r in L]); vt = int(sum(bool(r.get("violation_t")) for r in L))
    cov = np.mean([bool(r.get("dep_t")) for r in L])
    sims = np.array([(bad & (rng.random(len(L)) < cov)).sum() for _ in range(10000)])
    print(f"{name:<34} cells {len(L):>4}  t-gate coverage {100*cov:5.1f}%  t-gate violations {vt:>3}  "
          f"random gate {sims.mean():5.1f} [{np.percentile(sims,2.5):.0f}, {np.percentile(sims,97.5):.0f}]  P(random <= t) = {(sims <= vt).mean():.3f}")
gens = sorted({k[3] for k in R})
for g in gens: ctrl([r for k, r in R.items() if k[3] == g], g)
pool = [g for g in ("knowledge_tuned", "llm_skill", "llm_skill@ds-reasoner") if g in gens]
ctrl([r for k, r in R.items() if k[3] in pool], "pooled " + "+".join(pool))
