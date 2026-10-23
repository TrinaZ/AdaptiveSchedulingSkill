"""
online_longrun.py -- long-horizon run with many changes (Fig. "long run"; links Prop. 4 to data).

One STREAM per seed: S changes (default 10) -> S+1 segments of --seg windows (default 40). Each change alters
the scene (runtime shift / capacity drop / new request type / back to baseline) or the SLA objective
(declared by the operator). Policies evaluated on the SAME windows:
  static    : initial best library rule, never changed
  reselect  : on every alarm, Govern's fallback only (library rule best on validation; no LLM)
  skill     : full Adaptive Scheduling Skill (Diagnose adequacy -> llm_skill -> isolated paired t gate -> deploy/fallback)
  ungated   : same alarms and candidates as `skill`, but the candidate is deployed whenever it beat the library on
              search (no Govern gate)
  oracle    : per segment, the best rule in hindsight among library + literature rules (regret comparator)
Normalized regret per window = (cost(policy) - cost(oracle)) / mean oracle cost of that segment.
Monitor: CUSUM on standardized log cost of the deployed rule; reference = 10 windows after each (re)deployment.

Usage: python3 online_longrun.py --bench online_sla_bench.py --src . --seeds 5 --out results/longrun.json
       python3 online_longrun.py ... --summary_only      (prints the summary and draws results/longrun.png)
"""
import sys, os, json, math, time, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--bench", required=True); ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--changes", type=int, default=10); ap.add_argument("--seg", type=int, default=40)
ap.add_argument("--ref", type=int, default=10); ap.add_argument("--h", type=float, default=5.0); ap.add_argument("--headroom", type=float, default=0.03)
ap.add_argument("--n_val", type=int, default=1000); ap.add_argument("--tau", type=float, default=0.02)
ap.add_argument("--summary_only", action="store_true")
ap.add_argument("--only_seed", type=int, default=-1, help="run a single stream (for parallel runs; merge with merge_longrun.py)")
for flag, d in (("--backend", "deepseek"), ("--model", None), ("--base_url", None), ("--key_env", None),
                ("--temperature", "0.7"), ("--max_tokens", "2500")):
    ap.add_argument(flag, default=d)
a = ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
POL = ["static", "reselect", "ungated", "skill"]

def summarize(res):
    print(f"\n{'policy':<10}" + "".join(f"{'seed '+str(r['seed']):>12}" for r in res) + f"{'mean':>10}   (final normalized regret, windows)")
    for p in POL:
        fin = [r["cum_regret"][p][-1] for r in res]
        print(f"{p:<10}" + "".join(f"{v:>12.1f}" for v in fin) + f"{np.mean(fin):>10.1f}")
    ep = [e for r in res for e in r["episodes"]]
    print(f"\nchanges {sum(r['n_changes'] for r in res)}, alarms {len(ep)} (false {sum(e['false'] for e in ep)}), "
          f"LLM skipped {sum(e['skipped'] for e in ep)}, deployed {sum(e['decision']=='deploy' for e in ep)}, "
          f"ungated would deploy {sum(e['ungated_deploy'] for e in ep)}, LLM calls {sum(e['calls'] for e in ep)}")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    col = {"static": "#9aa0a6", "reselect": "#2a78d6", "ungated": "#eb6834", "skill": "#008300"}
    lab = {"static": "static rule", "reselect": "re-selection only (no LLM)", "ungated": "Skill without gate", "skill": "Adaptive Scheduling Skill"}
    plt.rcParams.update({"font.size": 7.5, "font.family": "serif"})
    fig, (x1, x2) = plt.subplots(1, 2, figsize=(7.0, 2.4))
    r = res[0]; T = len(r["cost"]["oracle"]); t = np.arange(T)
    sm = lambda v, w=5: np.convolve(np.asarray(v), np.ones(w) / w, mode="same")
    for p in ("static", "skill"):
        x1.plot(t, sm(r["rel"][p]), color=col[p], lw=1.1 if p == "static" else 1.4, label=lab[p])
    for b in r["boundaries"]: x1.axvline(b, color="#c3c2b7", lw=0.6, ls=":")
    for e in r["episodes"]:
        x1.plot(e["t"], -0.08, marker="^" if e["decision"] == "deploy" else "v", color="#008300" if e["decision"] == "deploy" else "#2a78d6", ms=4, clip_on=False)
    x1.set_ylim(-0.12, None); x1.set_xlabel("traffic window"); x1.set_ylabel("cost / oracle cost - 1 (5-win. mean)")
    x1.set_title(f"(a) One stream (seed {r['seed']}): changes (dotted), deploy (^), fallback (v)", loc="left", fontsize=8)
    x1.legend(frameon=False, fontsize=6.5, loc="upper right")
    for p in POL:
        M = np.array([rr["cum_regret"][p] for rr in res]); m = M.mean(0)
        x2.plot(np.arange(len(m)), m, color=col[p], lw=1.4, label=lab[p])
        x2.fill_between(np.arange(len(m)), M.min(0), M.max(0), color=col[p], alpha=0.12, lw=0)
    x2.set_xlabel("traffic window"); x2.set_ylabel("cumulative normalized regret")
    x2.set_title(f"(b) Cumulative regret, mean and range over {len(res)} streams", loc="left", fontsize=8)
    x2.legend(frameon=False, fontsize=6.5, loc="upper left")
    for ax in (x1, x2):
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(w_pad=1.5); png = a.out.replace(".json", ".png"); fig.savefig(png, dpi=200); fig.savefig(png.replace(".png", ".pdf"))
    print("figure ->", png)

if a.summary_only:
    summarize(json.load(open(a.out))["streams"]); sys.exit(0)

code = open(a.bench).read().split("\n# ---- fail fast")[0]
fwd = ["bench", "--src", a.src, "--out", "/dev/null", "--backend", a.backend, "--temperature", str(a.temperature), "--max_tokens", str(a.max_tokens)]
for k in ("model", "base_url", "key_env"):
    if getattr(a, k): fwd += [f"--{k}", getattr(a, k)]
sys.argv = fwd
B = {}; exec(code, B)
make_instance, simulate, objective, run_cost, LIB, KNOW, llm_generate, call_llm_raw, backend_conf, per_inst, KNOW_FAM = (
    B[k] for k in ["make_instance", "simulate", "objective", "run_cost", "LIB", "KNOW", "llm_generate", "call_llm_raw",
                   "backend_conf", "per_inst", "KNOW_FAM"])
from govern_sandbox import isolated_costs, static_check, Rejected
from scipy import stats as st
KN = KNOW()
_url, _env, _model = backend_conf()
if not os.environ.get(_env): sys.exit(f"ERROR: export {_env}=<key> first")
if not call_llm_raw("Reply with the single word OK."): sys.exit("ERROR: LLM API test call failed")
print(f"LLM API check passed ({a.backend}, {_model})", flush=True)

SCENES = ["baseline", "S1_runtime_shift", "S2_resource_drop", "S4_novel_task"]
OBJS = ["tard", "miss", "slo_wait", "mix"]
def plan(seed):
    rng = np.random.default_rng(1234 + seed); segs = [("baseline", OBJS[seed % 4])]
    for _ in range(a.changes):
        sc, o = segs[-1]
        if rng.random() < 0.35: o = rng.choice([x for x in OBJS if x != o])
        else: sc = rng.choice([x for x in SCENES if x != sc])
        segs.append((str(sc), str(o)))
    return segs

def wcost(rule, inst, o):
    rs, R = inst
    try: return objective(o, rs, simulate(rs, R, rule))
    except Exception: return float("nan")

def features(insts):
    reqs = [r for rs, _ in insts for r in rs]
    return dict(cap=float(np.mean([R for _, R in insts])), types={r.task_type for r in reqs})

def run_stream(seed):
    segs = plan(seed); W = a.seg; T = W * len(segs); base = 9_000_000 + 1_000_000 * seed
    seg_of = lambda t: t // W
    stream = [make_instance(segs[seg_of(t)][0], base + t, 60, 1.0) for t in range(T)]
    obj_of = lambda t: segs[seg_of(t)][1]
    # oracle: best of library + literature per segment, in hindsight
    oracle = np.zeros(T)
    for k, (sc, o) in enumerate(segs):
        win = stream[k * W:(k + 1) * W]
        cands = list(LIB.values()) + [f for _, f, _, _ in KN]
        best = min(cands, key=lambda f: np.nanmean([wcost(f, x, o) for x in win]))
        oracle[k * W:(k + 1) * W] = [wcost(best, x, o) for x in win]
    cfg = [make_instance(segs[0][0], base + 500_000 + i, 60, 1.0) for i in range(30)]
    init = min(LIB, key=lambda n: run_cost(LIB[n], cfg, segs[0][1]))
    rules = {p: LIB[init] for p in POL}; costs = {p: np.zeros(T) for p in POL}
    episodes = []; ref, S, ref_from = [], 0.0, 0
    for t in range(T):
        o = obj_of(t)
        for p in POL: costs[p][t] = wcost(rules[p], stream[t], o)
        declared = t > 0 and t % W == 0 and segs[seg_of(t)][1] != segs[seg_of(t) - 1][1]
        c = costs["skill"][t]
        alarm = False
        if declared: alarm = True
        elif t - ref_from < a.ref: ref.append(math.log1p(max(c, 0.0)))
        else:
            mu, sd = float(np.mean(ref)), float(np.std(ref, ddof=1)) or 1.0
            S = max(0.0, S + (math.log1p(max(c, 0.0)) - mu) / sd - 0.5); alarm = S > a.h
        if not alarm: continue
        # ---- episode (Diagnose, Generate, Govern); recent-traffic replay from the current segment's scene
        sc = segs[seg_of(t)][0]; k = seg_of(t); t0 = time.time()
        false = (not declared) and (t - k * W) > max(a.ref + 5, W // 2) and k == seg_of(ref_from)  # alarm long after the last change, same segment
        Is = [make_instance(sc, base + 600_000 + 100 * t + i, 60, 1.0) for i in range(12)]
        lib_s = {n: run_cost(f, Is, o) for n, f in LIB.items()}; best_lib = min(lib_s, key=lib_s.get)
        kn_sc = [(run_cost(f, Is, o), lab, f, fam, p) for lab, f, fam, p in KN]
        headroom = 1.0 - min(x[0] for x in kn_sc if x[0] is not None) / max(lib_s[best_lib], 1e-9)
        skipped = headroom < a.headroom; cand = None; calls = 0
        if not skipped:
            lab, rule, calls, _ = llm_generate("llm_skill", o, Is, best_lib, lib_s[best_lib], kn_sc, budget=100, rng=np.random.default_rng(t))
            if lab not in LIB:
                src = getattr(rule, "src", None) or next((c_ for n_, c_, _ in KNOW_FAM if lab.startswith(n_)), None)
                cand = (lab, rule, src, getattr(rule, "params", {}))
        Iv = [make_instance(sc, base + 700_000 + 2000 * t + i, 60, 1.0) for i in range(a.n_val)]
        vv = {n: per_inst(f, Iv, o) for n, f in LIB.items()}; fb = min(vv, key=lambda n: vv[n].mean())
        decision = "fallback"
        if cand is not None:
            try:
                static_check(cand[2]); vc, why = isolated_costs(cand[2], cand[3], Iv, objective, simulate, o)
            except Rejected:
                vc = None
            if vc is not None:
                d = np.array(vc) - (1 + a.tau) * vv[fb]; n = len(d); K = 2 * len(LIB)
                if d.mean() + st.t.ppf(1 - 0.05 / K, n - 1) * math.sqrt(float(d.var(ddof=1)) / n) <= 0: decision = "deploy"
        rules["skill"] = cand[1] if decision == "deploy" else LIB[fb]
        rules["reselect"] = LIB[fb]
        rules["ungated"] = cand[1] if cand is not None else LIB[best_lib]
        episodes.append(dict(t=t, segment=k, declared=declared, false=bool(false), skipped=bool(skipped), calls=calls,
                             decision=decision, ungated_deploy=cand is not None, candidate=(cand[0][:300] if cand else None),
                             fallback=fb, seconds=round(time.time() - t0, 1)))
        ref, S, ref_from = [], 0.0, t + 1
        print(f"   seed {seed} t={t:3d} seg {k} {sc:<17}{o:<9} {'declared' if declared else 'alarm':<8} skip={skipped} "
              f"decision={decision} calls={calls}", flush=True)
    seg_mean = np.array([oracle[(t // W) * W:(t // W + 1) * W].mean() for t in range(T)])
    rel = {p: ((costs[p] - oracle) / np.maximum(seg_mean, 1e-9)).tolist() for p in POL}
    return dict(seed=seed, segments=segs, boundaries=[k * W for k in range(1, len(segs))], n_changes=len(segs) - 1,
                episodes=episodes, rel=rel, cum_regret={p: np.nancumsum(rel[p]).tolist() for p in POL},
                cost={p: costs[p].tolist() for p in POL} | {"oracle": oracle.tolist()})

res = []
if os.path.exists(a.out): res = json.load(open(a.out)).get("streams", [])
done = {r["seed"] for r in res}
for seed in range(a.seeds):
    if seed in done or (a.only_seed >= 0 and seed != a.only_seed): continue
    t0 = time.time(); r = run_stream(seed); res.append(r)
    json.dump(dict(config=vars(a), streams=res), open(a.out, "w"))
    print(f"[seed {seed} done in {time.time()-t0:.0f}s] final regret: " + ", ".join(f"{p} {r['cum_regret'][p][-1]:.1f}" for p in POL), flush=True)
summarize(res)
