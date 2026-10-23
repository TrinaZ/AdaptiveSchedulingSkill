"""
longrun_kngov.py -- the Skill with the LLM removed: knowledge_tuned + Govern on the same long-run streams (no LLM).

At every alarm of the stored long run, with the same Diagnose decision (skip if literature headroom < 3%), the same 12 replay
instances, the same 1,000 validation instances and the same gate, the candidate is the knowledge_tuned rule (20 literature
rules on their grid + constant calibration at budget 100) instead of llm_skill's output. Reported with the paired t gate
(as the Skill) and with the paired EB gate (certified form; observed range).

Usage: python3 longrun_kngov.py --bench online_sla_bench.py --src <repo> --longrun results/longrun.json --out results/longrun_kngov.json
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

out = []
for S in LR["streams"]:
    seed = S["seed"]; segs = S["segments"]; T = W * len(segs); base = 9_000_000 + 1_000_000 * seed; t0 = time.time()
    stream = [make_instance(segs[t // W][0], base + t, 60, 1.0) for t in range(T)]
    objs = [segs[t // W][1] for t in range(T)]
    oracle = np.array(S["cost"]["oracle"]); seg_mean = np.array([oracle[(t // W) * W:(t // W + 1) * W].mean() for t in range(T)])
    cfg = [make_instance(segs[0][0], base + 500_000 + i, 60, 1.0) for i in range(30)]
    init = LIB[min(LIB, key=lambda n: run_cost(LIB[n], cfg, segs[0][1]))]
    rules = {"t": init, "eb": init}; costs = {"t": np.zeros(T), "eb": np.zeros(T)}
    al = {e["t"]: e for e in S["episodes"]}; dec = {"t": [], "eb": []}
    for t in range(T):
        for g in rules: costs[g][t] = wcost(rules[g], stream[t], objs[t])
        if t not in al: continue
        e = al[t]; sc, o = segs[t // W][0], objs[t]
        Iv = [make_instance(sc, base + 700_000 + 2000 * t + i, 60, 1.0) for i in range(N_VAL)]
        vv = {n: per_inst(f, Iv, o) for n, f in LIB.items()}; fb = min(vv, key=lambda n: vv[n].mean())
        ok_t = ok_eb = False; cand = None
        if not e["skipped"]:
            Is = [make_instance(sc, base + 600_000 + 100 * t + i, 60, 1.0) for i in range(12)]
            lib_s = {n: run_cost(f, Is, o) for n, f in LIB.items()}
            v, lab, cand = kt_candidate(Is, o, np.random.default_rng(t))
            if v < min(lib_s.values()):
                vc = per_inst(cand, Iv, o)
                if vc is not None: ok_t, ok_eb = gates(vc, vv[fb])
        rules["t"] = cand if ok_t else LIB[fb]; rules["eb"] = cand if ok_eb else LIB[fb]
        dec["t"].append(bool(ok_t)); dec["eb"].append(bool(ok_eb))
    rel = lambda c: (c - oracle) / np.maximum(seg_mean, 1e-9)
    r = dict(seed=seed, kngov_t=float(np.nansum(rel(costs["t"]))), kngov_eb=float(np.nansum(rel(costs["eb"]))),
             deploys_t=sum(dec["t"]), deploys_eb=sum(dec["eb"]), skill=float(S["cum_regret"]["skill"][-1]),
             reselect=float(S["cum_regret"]["reselect"][-1]), ungated=float(S["cum_regret"]["ungated"][-1]),
             skill_deploys=sum(e["decision"] == "deploy" for e in S["episodes"]))
    out.append(r); json.dump(out, open(a.out, "w"), indent=1)
    print(f"seed {seed} ({time.time()-t0:.0f}s): " + ", ".join(f"{k} {v:.1f}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items() if k != "seed"), flush=True)
m = lambda k: np.mean([r[k] for r in out])
print(f"\nmean final regret: Skill (LLM + t gate) {m('skill'):.1f} | knowledge_tuned + t gate {m('kngov_t'):.1f} | knowledge_tuned + EB gate {m('kngov_eb'):.1f} | "
      f"library re-selection {m('reselect'):.1f} | Skill ungated {m('ungated'):.1f}")
print(f"deployments: Skill {sum(r['skill_deploys'] for r in out)}, kt+t {sum(r['deploys_t'] for r in out)}, kt+EB {sum(r['deploys_eb'] for r in out)}")
d = np.array([r["kngov_t"] - r["skill"] for r in out])
print(f"knowledge_tuned+Govern minus Skill per stream: {np.round(d, 1).tolist()}")
