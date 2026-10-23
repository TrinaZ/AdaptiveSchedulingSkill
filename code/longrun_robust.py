"""
longrun_robust.py -- robustness of the long-run result to two operator settings (no LLM, exact).

(i)  Human approval: each `deploy` decision is rejected by the operator with probability p (the fallback rule runs
     instead until the next episode). Because episodes are independent, the expected cumulative regret is exact:
     E[regret_p] = regret_skill + p * sum over deploy episodes of (regret with fallback - regret with candidate)
     over that episode's active interval.
(ii) Diagnose adequacy threshold: raising it above the 3% used in the run turns episodes whose literature headroom
     lies in [3%, thr) into "skipped" episodes (fallback, no LLM call). Lower thresholds would need new LLM calls
     and are not computed; the number of episodes skipped at 3% is reported.

Usage: python3 longrun_robust.py --bench online_sla_bench.py --src <repo> --longrun results/longrun.json --out results/longrun_robust.json
"""
import sys, json, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--bench", required=True); ap.add_argument("--src", required=True)
ap.add_argument("--longrun", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
code = open(a.bench).read().split("\n# ---- fail fast")[0]
sys.argv = ["bench", "--src", a.src, "--out", "/dev/null"]
B = {}; exec(code, B)
make_instance, simulate, objective, run_cost, LIB, KNOW = (B[k] for k in
    ["make_instance", "simulate", "objective", "run_cost", "LIB", "KNOW"])
KN = KNOW(); LR = json.load(open(a.longrun)); W = LR["config"]["seg"]

def wcost(rule, inst, o):
    rs, R = inst
    try: return objective(o, rs, simulate(rs, R, rule))
    except Exception: return float("nan")

P = [0.0, 0.1, 0.25, 0.5, 1.0]; THR = [0.03, 0.05, 0.10, 0.20, 1.01]
out = []
for S in LR["streams"]:
    seed = S["seed"]; segs = S["segments"]; T = W * len(segs); base = 9_000_000 + 1_000_000 * seed
    objs = [segs[t // W][1] for t in range(T)]
    oracle = np.array(S["cost"]["oracle"]); seg_mean = np.array([oracle[(t // W) * W:(t // W + 1) * W].mean() for t in range(T)])
    skill = np.array(S["cost"]["skill"])
    rel = lambda c, lo=0, hi=None: (c - oracle[lo:hi]) / np.maximum(seg_mean[lo:hi], 1e-9)
    ep = S["episodes"]; ts = [e["t"] for e in ep] + [T - 1]
    delta, head = [], []
    for i, e in enumerate(ep):
        lo, hi = e["t"] + 1, ts[i + 1] + (1 if i + 1 == len(ep) else 1)   # windows run with this episode's rule
        hi = min(hi, T); fb = LIB[e["fallback"]]
        fbc = np.array([wcost(fb, make_instance(segs[t // W][0], base + t, 60, 1.0), objs[t]) for t in range(lo, hi)])
        d = float(np.nansum(rel(fbc, lo, hi) - rel(skill[lo:hi], lo, hi))) if e["decision"] == "deploy" else 0.0
        delta.append(d)
        sc, o, t = segs[e["t"] // W][0], objs[e["t"]], e["t"]
        Is = [make_instance(sc, base + 600_000 + 100 * t + i2, 60, 1.0) for i2 in range(12)]
        lib = min(run_cost(f, Is, o) for f in LIB.values())
        kn = min(v for v in (run_cost(f, Is, o) for _, f, _, _ in KN) if v is not None)
        head.append(1.0 - kn / max(lib, 1e-9))
    base_reg = float(np.nansum(rel(skill)))
    approval = {p: base_reg + p * sum(delta) for p in P}
    diag = {}
    for thr in THR:   # episodes that ran the LLM at 3% but would be skipped at thr -> fallback (delta applies if it deployed)
        extra = [i for i, e in enumerate(ep) if not e["skipped"] and head[i] < thr]
        diag[thr] = dict(regret=base_reg + sum(delta[i] for i in extra),
                         llm_episodes=sum(not e["skipped"] for e in ep) - len(extra))
    out.append(dict(seed=seed, skill=base_reg, reselect=float(S["cum_regret"]["reselect"][-1]),
                    static=float(S["cum_regret"]["static"][-1]), approval=approval, diagnose=diag,
                    headroom=head, skipped_at_3pct=sum(e["skipped"] for e in ep), deploys=sum(e["decision"] == "deploy" for e in ep)))
    print(f"seed {seed} done", flush=True)
json.dump(out, open(a.out, "w"), indent=1)
m = lambda f: np.mean([f(r) for r in out])
print(f"\nmean final regret: static {m(lambda r: r['static']):.1f}, reselect {m(lambda r: r['reselect']):.1f}, skill {m(lambda r: r['skill']):.1f}")
print("(i) operator rejects each deployment with prob p:  " + ", ".join(f"p={p}: {m(lambda r: r['approval'][p]):.1f}" for p in P))
print("(ii) Diagnose threshold:  " + ", ".join(f"{int(100*t) if t <= 1 else 'inf'}%: regret {m(lambda r: r['diagnose'][t]['regret']):.1f} "
      f"(LLM episodes {sum(r['diagnose'][t]['llm_episodes'] for r in out)})" for t in THR))
print(f"episodes skipped at 3%: {sum(r['skipped_at_3pct'] for r in out)}; deploys: {sum(r['deploys'] for r in out)}")
print("headroom of episodes (%):", sorted(round(100 * h, 1) for r in out for h in r["headroom"]))
