"""
gate_risk_coverage.py -- risk-coverage of the Govern gates (no LLM).

Same 168 episodes and candidate pool as online_governance.py (7 objectives x 4 scenes x 6 seeds; best of
220 candidates on 8 search instances). For each episode we store per-instance costs of the candidate and of
every library rule on V validation instances (default 1000) and on H held-out instances (default 200).
Then, for n in {25, 50, 100, 200, 500, 1000} (first n validation instances), tau in {0, 0.02, 0.05}, and gate in
{point estimate, paired t, paired EB}, we compute
  coverage = share of episodes in which the candidate is deployed,
  risk     = share of deployments whose held-out cost exceeds (1+tau) x held-out cost of the fallback
             (library rule best on the same n validation instances),
  gain     = mean held-out cost of the deployed-or-fallback rule relative to the best library rule on held-out.
Usage: python3 gate_risk_coverage.py --bench online_sla_bench.py --src . --out results/gate_risk_coverage.json
"""
import sys, math, json, argparse, time
import numpy as np
from scipy import stats as st

ap = argparse.ArgumentParser()
ap.add_argument("--bench", required=True); ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--seeds", type=int, default=6); ap.add_argument("--V", type=int, default=1000); ap.add_argument("--H", type=int, default=200)
a = ap.parse_args()
code = open(a.bench).read().split("\n# ---- fail fast")[0]
sys.argv = ["bench", "--src", a.src, "--out", "/dev/null"]
B = {}; exec(code, B)
make_instance, run_cost, LIB, KNOW, POOL_FULL, sample, static_fn, per_inst = (
    B[k] for k in ["make_instance", "run_cost", "LIB", "KNOW", "POOL_FULL", "sample", "static_fn", "per_inst"])
FEATS = ["task.estimated_runtime", "task.resource_need", "task.weight", "(task.due - task.release)", "1.0"]
def random_expr(rng):
    f1, f2 = rng.choice(FEATS, 2); op = rng.choice(["+", "-", "*", "/"])
    c = float(np.round(rng.uniform(-3, 3), 2))
    e = f"({f1} {op} max({f2}, 1e-6))" if op == "/" else f"({f1} {op} {f2})"
    return f"def priority(task): return {c} * {e}"

OBJS = ["tard", "miss", "slo_wait", "premium", "mix", "p95", "wflow"]
SCENES = ["baseline", "S1_runtime_shift", "S2_resource_drop", "S4_novel_task"]
eps = []; t0 = time.time()
for oname in OBJS:
    for scene in SCENES:
        for s in range(a.seeds):
            Is = [make_instance(scene, 1000 + s * 10000 + i, 60, 1.0) for i in range(8)]
            Iv = [make_instance(scene, 1000 + (s + 30) * 10000 + i, 60, 1.0) for i in range(a.V)]
            Ih = [make_instance(scene, 1000 + (s + 70) * 10000 + i, 60, 1.0) for i in range(a.H)]
            rng = np.random.default_rng(s * 31 + 7)
            cands = [(c, static_fn(c)) for c in sample(POOL_FULL, rng, 100)] + [(l, f) for l, f, _, _ in KNOW()] + \
                    [(e, static_fn(e)) for e in (random_expr(rng) for _ in range(100))]
            lib_s = {k: run_cost(f, Is, oname) for k, f in LIB.items()}; best_lib = min(lib_s, key=lib_s.get)
            sc = [x for x in ((run_cost(f, Is, oname), lab, f) for lab, f in cands) if x[0] is not None]
            v, lab, f = min(sc, key=lambda x: x[0])
            if v >= lib_s[best_lib]: continue                      # nothing to gate
            eps.append(dict(objective=oname, scene=scene, seed=s, cand=lab,
                            val={k: per_inst(g, Iv, oname).tolist() for k, g in LIB.items()} | {"_cand": per_inst(f, Iv, oname).tolist()},
                            held={k: float(per_inst(g, Ih, oname).mean()) for k, g in LIB.items()} | {"_cand": float(per_inst(f, Ih, oname).mean())}))
        print(f"[{time.time()-t0:6.0f}s] {oname}: {len(eps)} episodes with a candidate", flush=True)

K = 2 * len(LIB); dl = 0.05; lg = math.log(2 * K / dl)
rows = []
for n in (25, 50, 100, 200, 500, 1000):
    if n > a.V: continue
    for tau in (0.0, 0.02, 0.05):
        for gate in ("point", "t", "eb"):
            dep = viol = 0; gain = []
            for e in eps:
                vv = {k: np.array(x[:n]) for k, x in e["val"].items()}
                fb = min((k for k in vv if k != "_cand"), key=lambda k: vv[k].mean())
                d = vv["_cand"] - (1 + tau) * vv[fb]; V = float(d.var(ddof=1)); R = float(d.max() - d.min())
                if gate == "point": ub = d.mean()
                elif gate == "t": ub = d.mean() + st.t.ppf(1 - dl / K, n - 1) * math.sqrt(V / n)
                else: ub = d.mean() + math.sqrt(2 * V * lg / n) + 7 * R * lg / (3 * (n - 1))
                ok = ub <= 0
                h = e["held"]; best_h = min(v for k, v in h.items() if k != "_cand")
                cost = h["_cand"] if ok else h[fb]
                if ok:
                    dep += 1; viol += h["_cand"] > (1 + tau) * h[fb] + 1e-9
                gain.append(cost / max(best_h, 1e-9) - 1)
            rows.append(dict(n=n, tau=tau, gate=gate, episodes=len(eps), coverage=dep / len(eps),
                             risk=(viol / dep if dep else 0.0), violations=viol, deployed=dep, mean_rel=float(np.mean(gain))))
json.dump(dict(rows=rows, episodes=len(eps)), open(a.out, "w"), indent=1)
print(f"\n{'n':>5}{'tau':>6}{'gate':>7}{'coverage':>10}{'risk':>8}{'viol':>6}{'mean rel %':>12}")
for r in rows:
    print(f"{r['n']:>5}{r['tau']:>6.2f}{r['gate']:>7}{100*r['coverage']:>9.1f}%{100*r['risk']:>7.1f}%{r['violations']:>6}{100*r['mean_rel']:>12.2f}")
