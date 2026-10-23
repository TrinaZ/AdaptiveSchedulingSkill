"""
gate_calibration.py -- calibration of the Govern gates against their nominal delta, and a random-gate negative control (no LLM).
Episodes and candidates are rebuilt exactly as in gate_risk_coverage.py (130 episodes with a candidate). Raw per-instance
validation costs are kept, so the gates can be re-run at any delta:
  * for delta in a grid, n in {200, 1000}, tau = 0.02 and gate in {paired t, paired EB}: coverage, risk (violating share of
    deployments) and violation rate per episode (share of ALL episodes in which a violating candidate is deployed),
    which is the quantity the per-episode guarantee bounds by delta;
  * negative control: a random gate that deploys each candidate with the same probability as the paired gate's coverage;
    its violations per episode are simulated 10,000 times.
Below: original description of the episode construction.

Same 168 episodes and candidate pool as online_governance.py (7 objectives x 4 scenes x 6 seeds; best of
220 candidates on 8 search instances). For each episode we store per-instance costs of the candidate and of
every library rule on V validation instances (default 1000) and on H held-out instances (default 200).
Then, for n in {25, 50, 100, 200, 500, 1000} (first n validation instances), tau in {0, 0.02, 0.05}, and gate in
{point estimate, paired t, paired EB}, we compute
  coverage = share of episodes in which the candidate is deployed,
  risk     = share of deployments whose held-out cost exceeds (1+tau) x held-out cost of the fallback
             (library rule best on the same n validation instances),
  gain     = mean held-out cost of the deployed-or-fallback rule relative to the best library rule on held-out.
Usage: python3 gate_calibration.py --bench online_sla_bench.py --src . --out results/gate_calibration.json
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


K = 2 * len(LIB); tau = 0.02
rows = []; rng = np.random.default_rng(12345)
for n in (200, 1000):
    pre = []
    for e in eps:
        vv = {k: np.array(x[:n]) for k, x in e["val"].items()}
        fb = min((k for k in vv if k != "_cand"), key=lambda k: vv[k].mean())
        d = vv["_cand"] - (1 + tau) * vv[fb]; h = e["held"]
        pre.append((d.mean(), float(d.var(ddof=1)), float(d.max() - d.min()), bool(h["_cand"] > (1 + tau) * h[fb] + 1e-9)))
    bad = np.array([p[3] for p in pre])
    for dl in (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
        lg = math.log(2 * K / dl)
        for gate in ("t", "eb"):
            ok = np.array([(m + st.t.ppf(1 - dl / K, n - 1) * math.sqrt(V / n) if gate == "t" else
                            m + math.sqrt(2 * V * lg / n) + 7 * R * lg / (3 * (n - 1))) <= 0 for m, V, R, _ in pre])
            cov = ok.mean(); vio = int((ok & bad).sum())
            sims = np.array([(bad & (rng.random(len(bad)) < cov)).sum() for _ in range(10000)])
            rows.append(dict(n=n, delta=dl, gate=gate, episodes=len(eps), coverage=float(cov), deployed=int(ok.sum()), violations=vio,
                             risk=(vio / ok.sum() if ok.sum() else 0.0), viol_per_episode=vio / len(eps),
                             random_viol_mean=float(sims.mean() / len(eps)), random_viol_lo=float(np.percentile(sims, 2.5) / len(eps)),
                             random_viol_hi=float(np.percentile(sims, 97.5) / len(eps)),
                             random_p=float((sims <= vio).mean())))
    rows.append(dict(n=n, gate="ungated", episodes=len(eps), coverage=1.0, violations=int(bad.sum()), viol_per_episode=float(bad.mean())))
json.dump(dict(rows=rows, episodes=len(eps), tau=tau), open(a.out, "w"), indent=1)
print(f"\n{'n':>5}{'delta':>7}{'gate':>6}{'cov%':>7}{'risk%':>7}{'viol/ep%':>9}{'random viol/ep% [95%]':>26}{'P(rand<=)':>10}")
for r in rows:
    if r["gate"] == "ungated": print(f"{r['n']:>5}  ungated: violating candidates in {100*r['viol_per_episode']:.1f}% of episodes"); continue
    print(f"{r['n']:>5}{r['delta']:>7}{r['gate']:>6}{100*r['coverage']:>7.1f}{100*r['risk']:>7.1f}{100*r['viol_per_episode']:>9.2f}"
          f"{100*r['random_viol_mean']:>12.2f} [{100*r['random_viol_lo']:.1f}, {100*r['random_viol_hi']:.1f}]{r['random_p']:>10.4f}")
