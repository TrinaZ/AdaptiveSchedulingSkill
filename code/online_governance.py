"""
Govern-skill experiment on the online SLA benchmark (no LLM; generator-agnostic).

For each (objective, scene, seed):
  candidates  = 100 pool_full templates + textbook knowledge rules + 100 random
                expressions from a broad grammar (many of them poor -- a stand-in
                for an unreliable generator)
  search set  = 8 instances  (generator selects its best candidate here)
  validation  = n_val instances, drawn independently (Govern's gate)
  held-out    = 60 instances (ground truth for reporting only)
Deployment policies compared:
  ungated     : deploy the best-on-search candidate if it beats the library on search
  gate(c)     : Eq.(2) of the paper with margin c * eps_n, eps_n = B sqrt(log(2K/delta)/(2n));
                B = max per-instance cost observed on validation; c in {0, .25, .5, 1}
Reported per policy: deploy rate, bound violations (held-out cost > (1+tau) x best
library rule on held-out), mean held-out cost relative to the deployed library rule.
Also writes motivation data: degradation of the pre-change best rule and the
best library rule per (scene, objective).
"""
import sys, math, json, argparse, time
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--bench", required=True, help="path to online_sla_bench.py")
ap.add_argument("--src", required=True)
ap.add_argument("--seeds", type=int, default=6)
ap.add_argument("--n_val", type=int, default=40)
ap.add_argument("--tau", type=float, default=0.02)
ap.add_argument("--out", required=True)
a = ap.parse_args()

# load benchmark definitions (everything before the run section)
code = open(a.bench).read().split("\n# ---- fail fast")[0]
sys.argv = ["bench", "--src", a.src, "--out", "/dev/null"]
B = {}; exec(code, B)
make_instance, run_cost, LIB, KNOW, POOL_FULL, sample, static_fn, simulate, objective = (
    B[k] for k in ["make_instance", "run_cost", "LIB", "KNOW", "POOL_FULL", "sample", "static_fn", "simulate", "objective"])

def per_inst(rule, insts, oname):
    out = []
    for reqs, R in insts:
        try: out.append(objective(oname, reqs, simulate(reqs, R, rule)))
        except Exception: return None
    return np.array(out)

FEATS = ["task.estimated_runtime", "task.resource_need", "task.weight", "(task.due - task.release)", "1.0"]
def random_expr(rng):
    f1, f2 = rng.choice(FEATS, 2); op = rng.choice(["+", "-", "*", "/"])
    c = float(np.round(rng.uniform(-3, 3), 2))
    e = f"({f1} {op} max({f2}, 1e-6))" if op == "/" else f"({f1} {op} {f2})"
    return f"def priority(task): return {c} * {e}"

OBJS = ["tard", "miss", "slo_wait", "premium", "mix", "p95", "wflow"]
SCENES = ["baseline", "S1_runtime_shift", "S2_resource_drop", "S4_novel_task"]
CS = [0.0, 0.25, 0.5, 1.0]
rows, motiv = [], []
t0 = time.time()
for oname in OBJS:
    for scene in SCENES:
        for s in range(a.seeds):
            Is = [make_instance(scene, 1000 + s * 10000 + i, 60, 1.0) for i in range(8)]
            Iv = [make_instance(scene, 1000 + (s + 30) * 10000 + i, 60, 1.0) for i in range(a.n_val)]
            Ih = [make_instance(scene, 1000 + (s + 50) * 10000 + i, 60, 1.0) for i in range(60)]
            rng = np.random.default_rng(s * 31 + 7)
            cands = [(c, static_fn(c)) for c in sample(POOL_FULL, rng, 100)] + [(lab, f) for lab, f, _, _ in KNOW()] + \
                    [(e, static_fn(e)) for e in (random_expr(rng) for _ in range(100))]
            lib_s = {k: run_cost(f, Is, oname) for k, f in LIB.items()}
            best_lib = min(lib_s, key=lib_s.get)
            sc = [(run_cost(f, Is, oname), lab, f) for lab, f in cands]
            sc = [x for x in sc if x[0] is not None]
            v_s, lab, f = min(sc, key=lambda x: x[0])
            lib_h = {k: per_inst(g, Ih, oname).mean() for k, g in LIB.items()}
            ref = min(lib_h.values()); dep_lib_h = lib_h[best_lib]
            cand_h = per_inst(f, Ih, oname).mean()
            # validation (gate): fallback = library rule that is best on the VALIDATION set (Prop. 2)
            vv = {k: per_inst(g, Iv, oname) for k, g in LIB.items()}
            vc = per_inst(f, Iv, oname)
            fb = min(vv, key=lambda k: vv[k].mean()); fb_h = lib_h[fb]
            allv = np.concatenate([x for x in vv.values()] + [vc])
            Bv = float(allv.max() - min(0.0, allv.min()))
            K = 2 * len(LIB); dl = 0.05; n = a.n_val  # (candidate, L) pairs + library estimates (Prop. 1)
            eps = Bv * math.sqrt(math.log(2 * K / dl) / (2 * n))
            # paired empirical-Bernstein gate on d_i = c_i - (1+tau) * fallback_i  (Maurer & Pontil 2009)
            d = vc - (1 + a.tau) * vv[fb]; Rd = float(d.max() - d.min()); Vd = float(d.var(ddof=1))
            lg = math.log(2 * K / dl)
            ucb = d.mean() + math.sqrt(2 * Vd * lg / n) + 7 * Rd * lg / (3 * (n - 1))
            improves_on_search = v_s < lib_s[best_lib]
            policies = {"ungated": improves_on_search, "paired_EB": improves_on_search and ucb <= 0}
            for c in CS:
                e = c * eps
                policies[f"hoeff_c{c}"] = improves_on_search and (vc.mean() + e <= (1 + a.tau) * (vv[fb].mean() - e))
            for pol, deploy in policies.items():
                base_h = dep_lib_h if pol == "ungated" else fb_h
                dep_cost = cand_h if deploy else base_h
                rows.append(dict(objective=oname, scene=scene, seed=s, policy=pol, deployed=bool(deploy), rule=lab if deploy else (best_lib if pol == "ungated" else fb),
                                 violation=bool(dep_cost > (1 + a.tau) * ref + 1e-9),
                                 rel_vs_deployed_lib=float((dep_cost - dep_lib_h) / max(abs(dep_lib_h), 1e-9)),
                                 rel_vs_best_lib=float((dep_cost - ref) / max(abs(ref), 1e-9)), eps_rel=float(eps / max(abs(vv[fb].mean()), 1e-9)),
                                 cand_rel_vs_best_lib=float((cand_h - ref) / max(abs(ref), 1e-9))))
        print(f"[{time.time()-t0:5.0f}s] {oname}", flush=True)

# motivation: pre-change best rule under each change, and best library rule per (scene, objective)
for oname in OBJS:
    base = [make_instance("baseline", 1000 + 900000 + i, 60, 1.0) for i in range(30)]
    pre = min(LIB, key=lambda k: run_cost(LIB[k], base, oname))
    for scene in SCENES:
        I = [make_instance(scene, 1000 + 800000 + i, 60, 1.0) for i in range(30)]
        costs = {k: run_cost(g, I, oname) for k, g in LIB.items()}
        best = min(costs, key=costs.get)
        motiv.append(dict(objective=oname, scene=scene, pre_change_best=pre, post_change_best=best,
                          degradation=float(costs[pre] / max(costs[best], 1e-9) - 1), costs=costs))
json.dump(dict(rows=rows, motivation=motiv), open(a.out, "w"), indent=1)

agg = defaultdict(list)
for r in rows: agg[r["policy"]].append(r)
print(f"\n{'policy':<12}{'n':>5}{'deploy %':>10}{'violation %':>13}{'mean rel vs deployed lib %':>28}{'worst rel %':>13}")
for pol in ["ungated", "paired_EB"] + [f"hoeff_c{c}" for c in CS]:
    L = agg[pol]; rel = np.array([x["rel_vs_deployed_lib"] for x in L])
    print(f"{pol:<12}{len(L):>5}{100*np.mean([x['deployed'] for x in L]):>9.1f}%{100*np.mean([x['violation'] for x in L]):>12.1f}%{100*rel.mean():>27.2f}{100*rel.max():>13.1f}")
print("\nper objective: violation % (ungated | paired_EB | hoeff_c1.0), deploy % paired_EB")
for o in OBJS:
    u, pe, h = ([x for x in agg[p] if x["objective"] == o] for p in ("ungated", "paired_EB", "hoeff_c1.0"))
    print(f"  {o:<9} {100*np.mean([x['violation'] for x in u]):5.1f}% | {100*np.mean([x['violation'] for x in pe]):5.1f}% | {100*np.mean([x['violation'] for x in h]):5.1f}%   deploy {100*np.mean([x['deployed'] for x in pe]):5.1f}%  mean rel vs deployed lib {100*np.mean([x['rel_vs_deployed_lib'] for x in pe]):6.2f}%")
print("\nmotivation: pre-change best rule degradation after change")
for m in motiv:
    print(f"  {m['objective']:<9}{m['scene']:<18} pre-best {m['pre_change_best']:<18} post-best {m['post_change_best']:<18} degradation {100*m['degradation']:6.1f}%")
