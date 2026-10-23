"""
gp_baseline.py -- genetic-programming hyper-heuristic over the operator vocabulary of Table 1 (no LLM).

Tree GP (ramped half-and-half init, tournament 3, subtree crossover 0.8, point/subtree mutation 0.2, elitism 2, max depth 6)
over request and state features with protected arithmetic, max/min, exp/log and if-greater-than. Fitness = mean cost on the
SAME 12 search instances the other generators use; the evaluation budget E counts rule x search-set evaluations exactly as
for knowledge_tuned / llm_skill. E=100 (pop 20 x 5 generations) is the equal-budget baseline; E=1000 (pop 50 x 20) is a
ten-times-budget reference. Selection, fallback, held-out scoring and the paired t / EB gates replicate online_sla_bench.py,
so records can be pooled with the main results (generators `gp100`, `gp1000`). The benchmark file itself is not modified.

Usage: python3 gp_baseline.py --bench online_sla_bench.py --src <repo> --objectives tard,miss,slo_wait,mix,premium,p95,wflow \
           --budgets 100,1000 --out results/gp_<tag>.json
"""
import sys, math, json, argparse, time
import numpy as np
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--bench", required=True); ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--objectives", default="tard,miss,slo_wait,mix,premium,p95,wflow")
ap.add_argument("--scenes", default="baseline,S1_runtime_shift,S2_resource_drop,S4_novel_task")
ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--budgets", default="100,1000")
ap.add_argument("--n_val", type=int, default=1000); ap.add_argument("--tau", type=float, default=0.02)
a = ap.parse_args()
code = open(a.bench).read().split("\n# ---- fail fast")[0]
sys.argv = ["bench", "--src", a.src, "--out", "/dev/null"]
B = {}; exec(code, B)
make_instance, run_cost, per_inst, LIB = (B[k] for k in ["make_instance", "run_cost", "per_inst", "LIB"])

# ---------------- primitives (Table 1) ----------------
TERM = {
 "est": "max(t.estimated_runtime,1e-6)", "res": "t.resource_need", "w": "t.weight", "prem": "(1.0 if t.tenant=='premium' else 0.0)",
 "red": "(1.0 if t.task_type=='reduce' else 0.0)", "join": "(1.0 if t.task_type=='join' else 0.0)",
 "wait": "(s.now-t.release)", "slack": "(t.due-s.now-t.estimated_runtime)", "due": "(t.due-s.now)",
 "free": "s.free", "cap": "s.capacity", "qlen": "s.queue_len", "qwork": "s.queue_work", "mest": "max(s.mean_est,1e-6)",
}
FUN = {"add": 2, "sub": 2, "mul": 2, "div": 2, "max": 2, "min": 2, "neg": 1, "exp": 1, "log": 1, "ifgt": 4}
TN = list(TERM)

def rand_tree(rng, depth, full):
    if depth == 0 or (not full and rng.random() < 0.35):
        if rng.random() < 0.2: return ("c", float(np.round(rng.choice([0.5, 1, 2, 3, 5]) * (1 if rng.random() < 0.7 else -1) * rng.uniform(0.5, 1.5), 3)))
        return ("v", TN[rng.integers(len(TN))])
    f = list(FUN)[rng.integers(len(FUN))]
    return (f,) + tuple(rand_tree(rng, depth - 1, full) for _ in range(FUN[f]))

def to_py(n):
    k = n[0]
    if k == "c": return repr(n[1])
    if k == "v": return TERM[n[1]]
    A = [to_py(x) for x in n[1:]]
    return {"add": lambda: f"({A[0]}+{A[1]})", "sub": lambda: f"({A[0]}-{A[1]})", "mul": lambda: f"({A[0]}*{A[1]})",
            "div": lambda: f"({A[0]}/_pd({A[1]}))", "max": lambda: f"max({A[0]},{A[1]})", "min": lambda: f"min({A[0]},{A[1]})",
            "neg": lambda: f"(-{A[0]})", "exp": lambda: f"_ex({A[0]})", "log": lambda: f"_lg({A[0]})",
            "ifgt": lambda: f"({A[2]} if {A[0]}>{A[1]} else {A[3]})"}[k]()

ENV = {"max": max, "min": min, "_pd": lambda x: x if abs(x) > 1e-6 else 1e-6, "_ex": lambda x: math.exp(min(max(x, -50.0), 50.0)),
       "_lg": lambda x: math.log(abs(x) + 1e-6)}
def compile_tree(n):
    f = eval("lambda t, s: float(" + to_py(n) + ")", dict(ENV))
    f.src = to_py(n); return f

def depth(n): return 0 if n[0] in ("c", "v") else 1 + max(depth(x) for x in n[1:])
def nodes(n, path=()):
    yield path, n
    if n[0] not in ("c", "v"):
        for i, x in enumerate(n[1:]): yield from nodes(x, path + (i + 1,))
def replace(n, path, new):
    if not path: return new
    i = path[0]; return n[:i] + (replace(n[i], path[1:], new),) + n[i + 1:]
def rand_node(rng, n):
    L = list(nodes(n)); return L[rng.integers(len(L))]

def gp(Is, o, budget, rng):
    P, G = (20, 5) if budget <= 100 else (50, budget // 50)
    pop = [rand_tree(rng, 2 + (i % 3), full=(i % 2 == 0)) for i in range(P)]
    cache = {}; evals = 0
    def fit(n):
        nonlocal evals
        key = repr(n)
        if key not in cache:
            evals += 1
            try: v = run_cost(compile_tree(n), Is, o)
            except Exception: v = None
            cache[key] = float("inf") if v is None or not np.isfinite(v) else v
        return cache[key]
    for g in range(3 * G):   # duplicates are cached and not counted, so run until the budget is used (at most 3G generations)
        F = [fit(n) for n in pop]
        if evals >= budget: break
        order = np.argsort(F); new = [pop[order[0]], pop[order[1]]]
        def tour():
            c = rng.choice(len(pop), 3, replace=False); return pop[min(c, key=lambda i: F[i])]
        while len(new) < P:
            p1 = tour()
            if rng.random() < 0.8:
                p2 = tour(); path, _ = rand_node(rng, p1); _, sub = rand_node(rng, p2); ch = replace(p1, path, sub)
            else:
                path, _ = rand_node(rng, p1); ch = replace(p1, path, rand_tree(rng, 2, full=False))
            new.append(ch if depth(ch) <= 6 else p1)
        pop = new
    best = min(cache.items(), key=lambda kv: kv[1])
    return eval(best[0]), best[1], evals

res = []; t0 = time.time()
budgets = [int(x) for x in a.budgets.split(",")]
for o in a.objectives.split(","):
    for sc in a.scenes.split(","):
        for s in range(a.seeds):
            Is = [make_instance(sc, 1000 + s * 10000 + i, 60, 1.0) for i in range(12)]
            Ih = [make_instance(sc, 1000 + (s + 50) * 10000 + i, 60, 1.0) for i in range(60)]
            Iv = [make_instance(sc, 1000 + (s + 30) * 10000 + i, 60, 1.0) for i in range(a.n_val)]
            vv = {k: per_inst(g, Iv, o) for k, g in LIB.items()}; fb = min(vv, key=lambda k: vv[k].mean())
            lib_s = {k: run_cost(f, Is, o) for k, f in LIB.items()}; lib_h = {k: run_cost(f, Ih, o) for k, f in LIB.items()}
            bn = min(lib_s, key=lib_s.get); best_h = min(lib_h.values())
            for E in budgets:
                tree, v, ev = gp(Is, o, E, np.random.default_rng(s * 31 + E))
                if v < lib_s[bn]: lab, f = to_py(tree), compile_tree(tree)
                else: lab, f = bn, LIB[bn]
                vh = run_cost(f, Ih, o); gate = {}
                ub = {"eb": None, "t": None}
                if lab not in LIB:
                    vc = per_inst(f, Iv, o)
                    if vc is not None:
                        n = len(Iv); d = vc - (1 + a.tau) * vv[fb]; K = 2 * len(LIB); lg = math.log(2 * K / 0.05)
                        Rd = float(d.max() - d.min()); Vd = float(d.var(ddof=1))
                        ub["eb"] = d.mean() + math.sqrt(2 * Vd * lg / n) + 7 * Rd * lg / (3 * max(n - 1, 1))
                        ub["t"] = d.mean() + stats.t.ppf(1 - 0.05 / K, n - 1) * math.sqrt(Vd / n)
                gate["violation"] = bool(vh > (1 + a.tau) * lib_h[bn] + 1e-9)
                for gk, u in ub.items():
                    ok = u is not None and u <= 0; gh = vh if ok else lib_h[fb]
                    gate.update({f"dep_{gk}": bool(ok), f"heldout_{gk}": gh, f"bb_{gk}": bool(gh < best_h - 1e-9),
                                 f"rel_{gk}": float((gh - best_h) / max(abs(best_h), 1e-9)),
                                 f"violation_{gk}": bool(gh > (1 + a.tau) * lib_h[fb] + 1e-9)})
                res.append(dict(objective=o, scene=sc, seed=s, generator=f"gp{E}", rule=("# GP\n" + lab) if lab not in LIB else lab,
                                heldout=vh, evals=ev, **gate, deployed_library_heldout=lib_h[bn], best_library_heldout=best_h,
                                bb=bool(vh < best_h - 1e-9), rel=float((vh - best_h) / max(abs(best_h), 1e-9))))
            json.dump(res, open(a.out, "w"))
        print(f"[{time.time()-t0:6.0f}s] {o} {sc}", flush=True)
print(f"\n{'objective':<10}{'gen':<8}{'n':>4}{'BB%':>6}{'rel%':>8}{'gated rel%':>11}{'dep_t%':>7}{'viol_t%':>8}{'evals':>7}")
for o in a.objectives.split(","):
    for E in budgets:
        L = [r for r in res if r["objective"] == o and r["generator"] == f"gp{E}"]
        if L: print(f"{o:<10}{'gp'+str(E):<8}{len(L):>4}{100*np.mean([r['bb'] for r in L]):>6.0f}{100*np.mean([r['rel'] for r in L]):>8.1f}"
                    f"{100*np.mean([r['rel_t'] for r in L]):>11.1f}{100*np.mean([r['dep_t'] for r in L]):>7.0f}{100*np.mean([r['violation_t'] for r in L]):>8.0f}{np.mean([r['evals'] for r in L]):>7.0f}")
