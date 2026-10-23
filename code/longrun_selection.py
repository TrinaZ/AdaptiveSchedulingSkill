"""
longrun_selection.py -- selection hyper-heuristic baselines on the SAME long-run streams (no LLM).

Rebuilds every stream of an existing online_longrun.py result file (instances are deterministic in the seed),
checks that the rebuilt oracle matches the stored one, and adds selection policies over the arm set
A = library (6) + literature rules (20) = 26 rules -- the same knowledge the Skill's Generate step starts from:

  swucb       : sliding-window UCB (Garivier & Moulines, ALT 2011), bandit feedback (only the deployed rule's
                cost is observed each window); reset when the operator declares a new SLA objective.
                Exploration constant and window are chosen IN HINDSIGHT from a grid (best for the baseline).
  ftl_full    : sliding-window follow-the-leader with FULL information: after each window every arm's cost on
                that window is known (counterfactual replay in the simulator). Upper bound for any selection
                method over A that adapts per window; reset on declared objective changes. Window chosen in hindsight.
  kn_reselect : at the Skill's own alarm times, deploy the best arm of A on the same 12 recent-traffic replay
                instances the Skill used (= the Skill without LLM and without the Govern gate).

Usage: python3 longrun_selection.py --bench online_sla_bench.py --src <repo> --longrun results/longrun.json \
           --out results/longrun_selection.json
"""
import sys, os, json, math, argparse, time
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
KN = KNOW()
ARMS = [(n, f) for n, f in LIB.items()] + [(lab, f) for lab, f, _, _ in KN]
LR = json.load(open(a.longrun)); cfg = LR["config"]; W = cfg["seg"]

def wcost(rule, inst, o):
    rs, R = inst
    try: return objective(o, rs, simulate(rs, R, rule))
    except Exception: return float("nan")

def swucb(C, objs, decl, tau, c):
    """C: arms x T cost matrix; returns realized cost per window."""
    K, T = C.shape; out = np.zeros(T); hist = []   # list of (t, arm, normalized cost)
    scale = None; start = 0
    for t in range(T):
        if decl[t]: hist, scale, start = [], None, t
        recent = [h for h in hist if h[0] >= t - tau]
        n = np.zeros(K); s = np.zeros(K)
        for _, k, v in recent: n[k] += 1; s[k] += v
        if (n == 0).any(): k = int(np.argmax(n == 0))            # play every arm once in the window
        else:
            idx = s / n - c * np.sqrt(np.log(min(t - start + 1, tau)) / n)
            k = int(np.argmin(idx))
        out[t] = C[k, t]
        if scale is None: scale = max(abs(C[k, t]), 1e-9)
        hist.append((t, k, C[k, t] / scale))
    return out

def ftl(C, decl, w):
    K, T = C.shape; out = np.zeros(T); start = 0; k = 0
    for t in range(T):
        if decl[t]: start = t
        lo = max(start, t - w)   # right after a reset (t == lo) the previous choice is kept for one window
        if t > lo: k = int(np.argmin(np.nanmean(C[:, lo:t], axis=1)))
        out[t] = C[k, t]
    return out

res = []
for S in LR["streams"]:
    seed = S["seed"]; segs = S["segments"]; T = W * len(segs); base = 9_000_000 + 1_000_000 * seed; t0 = time.time()
    stream = [make_instance(segs[t // W][0], base + t, 60, 1.0) for t in range(T)]
    objs = [segs[t // W][1] for t in range(T)]
    decl = [t > 0 and t % W == 0 and objs[t] != objs[t - 1] for t in range(T)]
    C = np.array([[wcost(f, stream[t], objs[t]) for t in range(T)] for _, f in ARMS])
    oracle = np.zeros(T)
    for k in range(len(segs)):
        sl = slice(k * W, (k + 1) * W); oracle[sl] = C[int(np.argmin(np.nanmean(C[:, sl], axis=1))), sl]
    err = float(np.nanmax(np.abs(oracle - np.array(S["cost"]["oracle"])) / np.maximum(np.abs(oracle), 1e-9)))
    seg_mean = np.array([oracle[(t // W) * W:(t // W + 1) * W].mean() for t in range(T)])
    rel = lambda c: (c - oracle) / np.maximum(seg_mean, 1e-9)
    pols = {}
    for tau in (20, 40, 80):
        for c in (0.05, 0.1, 0.2, 0.5):
            pols[f"swucb(tau={tau},c={c})"] = swucb(C, objs, decl, tau, c)
    for w in (1, 3, 5, 10, 20):
        pols[f"ftl_full(w={w})"] = ftl(C, decl, w)
    # same, but also reset at the Skill's own Monitor alarms (the baselines get the Skill's change detection)
    alarm_reset = [decl[t] or (t - 1) in {e["t"] for e in S["episodes"]} for t in range(T)]
    for w in (3, 5, 10, 20):
        pols[f"ftl_full+alarms(w={w})"] = ftl(C, alarm_reset, w)
    for tau in (40, 80):
        for c in (0.05, 0.2):
            pols[f"swucb+alarms(tau={tau},c={c})"] = swucb(C, objs, alarm_reset, tau, c)
    # kn_reselect at the Skill's alarm times, same replay instances as online_longrun.py
    cur = int(np.argmin([run_cost(f, [make_instance(segs[0][0], base + 500_000 + i, 60, 1.0) for i in range(30)], objs[0]) for _, f in ARMS[:len(LIB)]]))
    al = {e["t"]: e for e in S["episodes"]}; kr = np.zeros(T)
    for t in range(T):
        kr[t] = C[cur, t]
        if t in al:
            sc = segs[t // W][0]
            Is = [make_instance(sc, base + 600_000 + 100 * t + i, 60, 1.0) for i in range(12)]
            vals = [run_cost(f, Is, objs[t]) for _, f in ARMS]
            cur = int(np.argmin([np.inf if v is None else v for v in vals]))
    pols["kn_reselect"] = kr
    out = dict(seed=seed, oracle_check_max_rel_err=err,
               final={p: float(np.nansum(rel(v))) for p, v in pols.items()} |
                     {p: float(S["cum_regret"][p][-1]) for p in S["cum_regret"]},
               cum_regret={p: np.nancumsum(rel(v)).tolist() for p, v in pols.items()})
    res.append(out)
    print(f"seed {seed}: oracle check max rel err {err:.2e}; {time.time()-t0:.0f}s", flush=True)
    json.dump(dict(config=vars(a), streams=res), open(a.out, "w"))

names = list(res[0]["final"])
print(f"\n{'policy':<26}" + "".join(f"{'s'+str(r['seed']):>8}" for r in res) + f"{'mean':>9}")
for p in sorted(names, key=lambda p: np.mean([r["final"][p] for r in res])):
    print(f"{p:<26}" + "".join(f"{r['final'][p]:>8.1f}" for r in res) + f"{np.mean([r['final'][p] for r in res]):>9.1f}")
