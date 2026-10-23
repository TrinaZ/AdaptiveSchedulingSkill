"""
longrun_ablation.py -- component ablation of the governed loop without an LLM (generator: knowledge_tuned), on the stored
long-run streams. Variants (each changes one component of "full" = knowledge_tuned + Govern, as in longrun_kngov.py):
  full        Monitor alarms + declared SLA changes, Diagnose skip (literature headroom < 3%), calibration, paired t gate
  no_monitor  only declared SLA changes start an episode (Monitor's alarms on scene changes and false alarms ignored)
  no_diagnose every episode generates (Diagnose never skips)
  no_calib    candidate = best literature rule on its constant grid, no calibration (generator "knowledge")
  no_gate     deploy the candidate whenever it beats the library on the 12 search instances
  eb_gate     paired empirical-Bernstein gate instead of the t gate
Re-selection only (no generation) and the static rule are taken from the stored long run.

At every alarm of the stored long run, with the same Diagnose decision (skip if literature headroom < 3%), the same 12 replay
instances, the same 1,000 validation instances and the same gate, the candidate is the knowledge_tuned rule (20 literature
rules on their grid + constant calibration at budget 100) instead of llm_skill's output. Reported with the paired t gate
(as the Skill) and with the paired EB gate (certified form; observed range).

Usage: python3 longrun_ablation.py --bench online_sla_bench.py --src <repo> --longrun results/longrun.json --out results/longrun_ablation.json
"""
import sys, json, math, argparse, time
import numpy as np
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--bench", required=True); ap.add_argument("--src", required=True)
ap.add_argument("--longrun", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
code = open(a.bench).read().split("\n# ---- fail fast")[0]
sys.argv = ["bench", "--src", a.src, "--out", "/dev/null"]
B = {}; exec(code, B)
make_instance, simulate, objective, run_cost, per_inst, LIB, KNOW, KNOW_FAM, compile_rule, calibrate = (B[k] for k in
    ["make_instance", "simulate", "objective", "run_cost", "per_inst", "LIB", "KNOW", "KNOW_FAM", "compile_rule", "calibrate"])
KN = KNOW(); LR = json.load(open(a.longrun)); W = LR["config"]["seg"]; N_VAL = LR["config"]["n_val"]; TAU = LR["config"]["tau"]
FSRC = {n: compile_rule(c, check=False)[0] for n, c, _ in KNOW_FAM}

def wcost(rule, inst, o):
    rs, R = inst
    try: return objective(o, rs, simulate(rs, R, rule))
    except Exception: return float("nan")

def kt_candidate(Is, o, rng):
    sc = [(run_cost(f, Is, o), lab, f, fam, p) for lab, f, fam, p in KN]
    fam_best = {}
    for v, lab, f, fam, p in sc:
        if v is not None and (fam not in fam_best or v < fam_best[fam][0]): fam_best[fam] = (v, p)
    tuned, _ = calibrate([(fam, FSRC[fam], p) for fam, (v, p) in fam_best.items()], Is, o, 100 - len(KN), rng)
    cc = [(v, lab, f) for v, lab, f, _, _ in sc if v is not None] + [(run_cost(r, Is, o), lab, r) for lab, r in tuned]
    return min([c for c in cc if c[0] is not None], key=lambda c: c[0])

def gates(vc, vfb):
    d = vc - (1 + TAU) * vfb; n = len(d); K = 2 * len(LIB); lg = math.log(2 * K / 0.05)
    Vd = float(d.var(ddof=1)); Rd = float(d.max() - d.min())
    ub_t = d.mean() + stats.t.ppf(1 - 0.05 / K, n - 1) * math.sqrt(Vd / n)
    ub_eb = d.mean() + math.sqrt(2 * Vd * lg / n) + 7 * Rd * lg / (3 * max(n - 1, 1))
    return ub_t <= 0, ub_eb <= 0


def k_candidate(Is, o):
    sc = [(run_cost(f, Is, o), lab, f) for lab, f, fam, p in KN]
    return min([c for c in sc if c[0] is not None], key=lambda c: c[0])

VAR = ["full", "no_monitor", "no_diagnose", "no_calib", "no_gate", "eb_gate"]
out = []
for S in LR["streams"]:
    seed = S["seed"]; segs = S["segments"]; T = W * len(segs); base = 9_000_000 + 1_000_000 * seed; t0 = time.time()
    stream = [make_instance(segs[t // W][0], base + t, 60, 1.0) for t in range(T)]
    objs = [segs[t // W][1] for t in range(T)]
    oracle = np.array(S["cost"]["oracle"]); seg_mean = np.array([oracle[(t // W) * W:(t // W + 1) * W].mean() for t in range(T)])
    cfg = [make_instance(segs[0][0], base + 500_000 + i, 60, 1.0) for i in range(30)]
    init = LIB[min(LIB, key=lambda n: run_cost(LIB[n], cfg, segs[0][1]))]
    rules = {v: init for v in VAR}; costs = {v: np.zeros(T) for v in VAR}; dep = {v: 0 for v in VAR}; gen = {v: 0 for v in VAR}
    al = {e["t"]: e for e in S["episodes"]}
    for t in range(T):
        for v in VAR: costs[v][t] = wcost(rules[v], stream[t], objs[t])
        if t not in al: continue
        e = al[t]; sc, o = segs[t // W][0], objs[t]
        Iv = [make_instance(sc, base + 700_000 + 2000 * t + i, 60, 1.0) for i in range(N_VAL)]
        vv = {n: per_inst(f, Iv, o) for n, f in LIB.items()}; fb = LIB[min(vv, key=lambda n: vv[n].mean())]
        fbv = vv[min(vv, key=lambda n: vv[n].mean())]
        Is = [make_instance(sc, base + 600_000 + 100 * t + i, 60, 1.0) for i in range(12)]
        lib_best = min(run_cost(f, Is, o) for f in LIB.values())
        res = {}
        for kind in ("kt", "k"):
            v, lab, cand = kt_candidate(Is, o, np.random.default_rng(t)) if kind == "kt" else k_candidate(Is, o)
            beats = v < lib_best; ok_t = ok_eb = False
            if beats:
                vc = per_inst(cand, Iv, o)
                if vc is not None: ok_t, ok_eb = gates(vc, fbv)
            res[kind] = (cand, beats, ok_t, ok_eb)
        for var in VAR:
            if var == "no_monitor" and not e["declared"]: continue
            skip = e["skipped"] and var != "no_diagnose"
            if skip: rules[var] = fb; continue
            gen[var] += 1
            cand, beats, ok_t, ok_eb = res["k" if var == "no_calib" else "kt"]
            ok = beats if var == "no_gate" else (ok_eb if var == "eb_gate" else ok_t)
            rules[var] = cand if ok else fb; dep[var] += int(bool(ok))
    rel = lambda c: (c - oracle) / np.maximum(seg_mean, 1e-9)
    r = dict(seed=seed, **{v: float(np.nansum(rel(costs[v]))) for v in VAR}, **{"dep_" + v: dep[v] for v in VAR},
             **{"gen_" + v: gen[v] for v in VAR}, reselect=float(S["cum_regret"]["reselect"][-1]),
             static=float(S["cum_regret"]["static"][-1]), skill=float(S["cum_regret"]["skill"][-1]))
    out.append(r); json.dump(out, open(a.out, "w"), indent=1)
    print(f"seed {seed} ({time.time()-t0:.0f}s): " + ", ".join(f"{k} {v:.1f}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items() if k != "seed"), flush=True)
m = lambda k: np.mean([r[k] for r in out])
print("\nmean final regret: " + " | ".join(f"{k} {m(k):.1f}" for k in VAR + ["reselect", "static", "skill"]))
print("episodes generating / deploying: " + " | ".join(f"{v} {sum(r['gen_'+v] for r in out)}/{sum(r['dep_'+v] for r in out)}" for v in VAR))
for v in VAR[1:]:
    d = np.array([r[v] - r["full"] for r in out]); print(f"{v} minus full per stream: {np.round(d, 1).tolist()}")
