"""
online_prototype.py -- end-to-end Adaptive Scheduling Skill on the online SLA regime (Table S5).

Each EPISODE is a stream of traffic windows (one window = one 60-request instance of online_sla_bench.py):
  windows 0..T_pre-1 come from the pre-change scene/objective, windows T_pre.. from the post-change one.
Skills (deterministic orchestrator; an LLM is called only inside Generate):
  Monitor  : CUSUM on z = (log1p(cost) - mu) / sd of the deployed rule, reference = first W windows;
             k = 0.5, h = --h. A semantic change (new SLA text) is DECLARED by the operator at the change point.
  Diagnose : change type from observable features of the last D windows vs. the reference windows
             (SLA text changed -> semantic; capacity changed or an unseen request type -> structural;
             otherwise distributional), and the ADEQUACY check: replay 12 recent-traffic instances and compare
             the best library rule with the best literature rule from the knowledge store (no LLM). If the
             literature cannot beat the library by more than --headroom, the library is adequate -> skip LLM.
  Generate : llm_skill of online_sla_bench.py (literature rules + 3 LLM rounds + calibration, budget 100).
  Govern   : static AST whitelist + validation of the candidate IN AN ISOLATED PROCESS (govern_sandbox.py) on
             --n_val replayed instances; paired t gate decides (paired EB gate is recorded as well); fallback =
             library rule best on validation; routing by type: distributional -> autonomous,
             structural -> expert approval, semantic -> human sign-off (approval is simulated as granted and
             logged; latency excludes approval time).
Every episode writes a Skill Package (JSON) and post-deployment costs on the remaining stream windows.

Episodes (default 120): distributional = S1 runtime shift (40); structural = capacity drop (20) + new request
type (20); semantic = SLA change from weighted response time to tard/miss/slo_wait/mix on the baseline scene (40).

Usage (same API flags as online_sla_bench.py):
  python3 online_prototype.py --bench online_sla_bench.py --src . --out results/prototype.json
  python3 online_prototype.py ... --summary_only          # re-print Table S5 from an existing file
"""
import sys, os, json, math, time, hashlib, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--bench", required=True); ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--t_pre", type=int, default=40); ap.add_argument("--t_post", type=int, default=60)
ap.add_argument("--ref", type=int, default=20, help="reference windows for Monitor")
ap.add_argument("--h", type=float, default=5.0, help="CUSUM threshold")
ap.add_argument("--headroom", type=float, default=0.03)
ap.add_argument("--n_val", type=int, default=1000)
ap.add_argument("--tau", type=float, default=0.02)
ap.add_argument("--summary_only", action="store_true")
ap.add_argument("--only", default="", help="comma list of types to run (distributional,structural,semantic)")
for flag, d in (("--backend", "deepseek"), ("--model", None), ("--base_url", None), ("--key_env", None),
                ("--temperature", "0.7"), ("--max_tokens", "2500"), ("--tag", "")):
    ap.add_argument(flag, default=d)
a = ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TYPES = ["distributional", "structural", "semantic"]

def summary(eps):
    print(f"\n{'':<34}" + "".join(f"{t:>16}" for t in TYPES))
    def row(name, fn):
        print(f"{name:<34}" + "".join(f"{fn([e for e in eps if e['true_type'] == t]):>16}" for t in TYPES))
    row("Episodes", lambda L: str(len(L)))
    row("Alarms before change (false)", lambda L: str(sum(e["false_alarms"] for e in L)))
    row("Detected / missed", lambda L: f"{sum(e['alarm'] is not None for e in L)}/{sum(e['alarm'] is None for e in L)}")
    row("Detection delay, windows (med)", lambda L: (f"{np.median([e['delay'] for e in L if e['delay'] is not None]):.0f}"
                                                      if any(e['delay'] is not None for e in L) else "--"))
    row("Diagnose label accuracy", lambda L: (f"{sum(e['label'] == e['true_type'] for e in L if e['alarm'] is not None)}/{sum(e['alarm'] is not None for e in L)}"))
    row("Skipped LLM (library adequate)", lambda L: str(sum(e.get("skipped_llm", False) for e in L)))
    row("Deployed new / fallback", lambda L: f"{sum(e.get('decision') == 'deploy' for e in L)}/{sum(e.get('decision') == 'fallback' for e in L)}")
    row("Approval required", lambda L: str(sum(e.get("approval_required", False) for e in L)))
    row("LLM calls / episode (mean)", lambda L: f"{np.mean([e.get('llm_calls', 0) for e in L]):.2f}")
    row("Tokens / episode (mean)", lambda L: f"{np.mean([e.get('tokens', 0) for e in L]):.0f}")
    row("Latency alarm->package, s (med)", lambda L: (f"{np.median([e['latency_s'] for e in L if e.get('latency_s') is not None]):.0f}"
                                                       if any(e.get('latency_s') is not None for e in L) else "--"))
    row("Post-change cost vs old rule (med)", lambda L: (f"{100*np.median([e['post_vs_old'] for e in L if e.get('post_vs_old') is not None]):+.1f}%"
                                                          if any(e.get('post_vs_old') is not None for e in L) else "--"))
    row("Post-change cost vs best lib (med)", lambda L: (f"{100*np.median([e['post_vs_bestlib'] for e in L if e.get('post_vs_bestlib') is not None]):+.1f}%"
                                                          if any(e.get('post_vs_bestlib') is not None for e in L) else "--"))
    row("Gate violations (post windows)", lambda L: str(sum(e.get("post_violation", False) for e in L)))

if a.summary_only:
    summary(json.load(open(a.out))["episodes"]); sys.exit(0)

# ---- load benchmark definitions (forward API flags so Generate uses the chosen backend)
code = open(a.bench).read().split("\n# ---- fail fast")[0]
fwd = ["bench", "--src", a.src, "--out", "/dev/null", "--backend", a.backend, "--temperature", str(a.temperature),
       "--max_tokens", str(a.max_tokens)]
for k in ("model", "base_url", "key_env"):
    if getattr(a, k): fwd += [f"--{k}", getattr(a, k)]
sys.argv = fwd
B = {}; exec(code, B)
(make_instance, simulate, objective, run_cost, LIB, LIB_SRC, KNOW, llm_generate, call_llm_raw, backend_conf, USAGE,
 per_inst) = (B[k] for k in ["make_instance", "simulate", "objective", "run_cost", "LIB", "LIB_SRC", "KNOW",
                             "llm_generate", "call_llm_raw", "backend_conf", "USAGE", "per_inst"])
from govern_sandbox import isolated_costs, static_check, Rejected
from scipy import stats as st
KN = KNOW()

_url, _env, _model = backend_conf()
if not os.environ.get(_env):
    sys.exit(f"ERROR: export {_env}=<key> first")
if not call_llm_raw("Reply with the single word OK."):
    sys.exit("ERROR: LLM API test call failed")
print(f"LLM API check passed ({a.backend}, {_model})", flush=True)

def window(scene, seed):
    return make_instance(scene, seed, 60, 1.0)

def features(insts):
    reqs = [r for rs, _ in insts for r in rs]
    return dict(cap=float(np.mean([R for _, R in insts])), est=float(np.mean([r.estimated_runtime for r in reqs])),
                types=sorted({r.task_type for r in reqs}))

def diagnose_type(ref_f, cur_f, sla_changed):
    if sla_changed: return "semantic"
    if abs(cur_f["cap"] - ref_f["cap"]) / ref_f["cap"] > 0.05 or set(cur_f["types"]) - set(ref_f["types"]): return "structural"
    return "distributional"

def episodes_plan():
    P = []; objs = ["tard", "miss", "slo_wait", "mix"]
    for s in range(a.seeds):
        for o in objs:
            P.append(dict(true_type="distributional", pre=("baseline", o), post=("S1_runtime_shift", o), seed=s))
    for s in range(a.seeds // 2 if a.seeds > 1 else 1):
        for o in objs:
            P.append(dict(true_type="structural", pre=("baseline", o), post=("S2_resource_drop", o), seed=s))
            P.append(dict(true_type="structural", pre=("baseline", o), post=("S4_novel_task", o), seed=s))
    for s in range(a.seeds):
        for o in objs:
            P.append(dict(true_type="semantic", pre=("baseline", "wflow"), post=("baseline", o), seed=s))
    if a.only:
        keep = set(a.only.split(",")); P = [p for p in P if p["true_type"] in keep]
    return P

def run_episode(ep, k):
    base = 7_000_000 + 100_000 * k
    (sc0, o0), (sc1, o1) = ep["pre"], ep["post"]
    T = a.t_pre + a.t_post
    stream = [window(sc0 if t < a.t_pre else sc1, base + t) for t in range(T)]
    # initial configuration: best library rule for the pre-change scene on 30 knowledge-store instances
    cfg = [window(sc0, base + 90_000 + i) for i in range(30)]
    old = min(LIB, key=lambda n: run_cost(LIB[n], cfg, o0)); deployed = (old, LIB[old])
    # ---- Monitor
    ref = [math.log1p(objective(o0, rs, simulate(rs, R, deployed[1]))) for rs, R in stream[:a.ref]]
    mu, sd = float(np.mean(ref)), float(np.std(ref, ddof=1)) or 1.0
    S, alarm, false_alarms = 0.0, None, 0
    for t in range(a.ref, T):
        if t == a.t_pre and o1 != o0:          # semantic change: SLA declared by the operator
            alarm = t; break
        rs, R = stream[t]
        z = (math.log1p(objective(o0, rs, simulate(rs, R, deployed[1]))) - mu) / sd
        S = max(0.0, S + z - 0.5)
        if S > a.h:
            if t < a.t_pre: false_alarms += 1; S = 0.0; continue
            alarm = t; break
    rec = dict(ep, episode=k, old_rule=old, false_alarms=false_alarms, alarm=alarm,
               delay=(alarm - a.t_pre) if alarm is not None else None)
    if alarm is None:
        return rec
    t_alarm = time.time(); u0 = dict(USAGE)
    # ---- Diagnose
    D = 10
    cur = stream[max(a.t_pre, alarm - D + 1): alarm + 1] if o1 == o0 else stream[a.t_pre:a.t_pre + 1]
    label = diagnose_type(features(stream[:a.ref]), features(cur), sla_changed=(o1 != o0))
    Is = [window(sc1, base + 200_000 + i) for i in range(12)]          # recent-traffic replay (search set)
    lib_s = {n: run_cost(f, Is, o1) for n, f in LIB.items()}; best_lib = min(lib_s, key=lib_s.get)
    kn_sc = [(run_cost(f, Is, o1), lab, f, fam, p) for lab, f, fam, p in KN]
    best_kn = min(x[0] for x in kn_sc if x[0] is not None)
    headroom = 1.0 - best_kn / max(lib_s[best_lib], 1e-9)
    adequate = headroom < a.headroom
    rec.update(label=label, headroom=float(headroom), skipped_llm=bool(adequate))
    # ---- Generate
    cand = None; calls = 0
    if not adequate:
        lab, rule, calls, ev = llm_generate("llm_skill", o1, Is, best_lib, lib_s[best_lib], kn_sc, budget=100,
                                            rng=np.random.default_rng(k + 101))
        if lab not in LIB:
            src = getattr(rule, "src", None) or next((c for n, c, _ in B["KNOW_FAM"] if lab.startswith(n)), None)
            cand = dict(label=lab, src=src, params=getattr(rule, "params", {}), llm_written="def priority" in lab)
    # ---- Govern
    Iv = [window(sc1, base + 400_000 + i) for i in range(a.n_val)]
    vv = {n: per_inst(f, Iv, o1) for n, f in LIB.items()}
    fb = min(vv, key=lambda n: vv[n].mean())
    evidence = dict(n=a.n_val, library_means={n: float(v.mean()) for n, v in vv.items()}, fallback=fb)
    decision, reason = "fallback", ("library adequate" if adequate else "no candidate beat the library on search")
    if cand is not None:
        try:
            static_check(cand["src"]); costs, why = isolated_costs(cand["src"], cand["params"], Iv, objective, simulate, o1)
        except Rejected as e:
            costs, why = None, f"static: {e}"
        if costs is None:
            reason = f"rejected by sandbox ({why})"
        else:
            vc = np.array(costs); n = len(vc); d = vc - (1 + a.tau) * vv[fb]; K = 2 * len(LIB); dl = 0.05
            V = float(d.var(ddof=1)); lg = math.log(2 * K / dl); Rd = float(d.max() - d.min())
            ub_t = d.mean() + st.t.ppf(1 - dl / K, n - 1) * math.sqrt(V / n)
            ub_eb = d.mean() + math.sqrt(2 * V * lg / n) + 7 * Rd * lg / (3 * (n - 1))
            evidence.update(candidate_mean=float(vc.mean()), ucb_t=float(ub_t), ucb_eb=float(ub_eb))
            if ub_t <= 0: decision, reason = "deploy", "paired t gate passed"
            else: reason = "paired t gate failed"
            rec["eb_would_deploy"] = bool(ub_eb <= 0)
    approval = decision == "deploy" and label in ("structural", "semantic")
    latency = time.time() - t_alarm
    new_rule_name = cand["label"] if decision == "deploy" else fb
    pkg = dict(rule_source=(cand["src"] if decision == "deploy" else LIB_SRC[fb]),
               params=(cand["params"] if decision == "deploy" else {}),
               provenance=dict(scene=sc1, objective=o1, backend=a.backend, model=_model, llm_calls=calls,
                               tokens=(USAGE["in"] - u0["in"]) + (USAGE["out"] - u0["out"]),
                               candidate_sha256=hashlib.sha256((cand or {}).get("src", "").encode()).hexdigest()[:16] if cand else None),
               change_type=label, evidence=evidence,
               decision=("deploy-after-approval" if approval else decision), reason=reason,
               rollback_to=old, review_flag=bool(decision == "fallback" or approval))
    # ---- post-deployment: remaining stream windows (cost of deployed vs old rule and vs best library rule)
    post = stream[alarm + 1:]
    post_cost = None
    if post:
        if decision == "deploy":
            pc, _ = isolated_costs(cand["src"], cand["params"], post, objective, simulate, o1); post_cost = float(np.mean(pc)) if pc else None
        else:
            post_cost = run_cost(LIB[fb], post, o1)
        old_cost = run_cost(LIB[old], post, o1)
        lib_post = min(run_cost(f, post, o1) for f in LIB.values())
        rec.update(post_vs_old=(post_cost / old_cost - 1) if post_cost is not None else None,
                   post_vs_bestlib=(post_cost / lib_post - 1) if post_cost is not None else None,
                   post_violation=bool(post_cost is not None and decision == "deploy" and post_cost > (1 + a.tau) * run_cost(LIB[fb], post, o1)))
    rec.update(decision=decision, approval_required=approval, llm_calls=calls, tokens=pkg["provenance"]["tokens"],
               latency_s=round(latency, 1), deployed=new_rule_name if decision == "deploy" else fb,
               candidate_llm_written=(cand or {}).get("llm_written"), skill_package=pkg)
    return rec

eps = []
if os.path.exists(a.out):
    eps = json.load(open(a.out)).get("episodes", [])
done = {e["episode"] for e in eps}
plan = episodes_plan(); t0 = time.time()
for k, ep in enumerate(plan):
    if k in done: continue
    r = run_episode(ep, k); eps.append(r)
    json.dump(dict(config=vars(a), episodes=eps), open(a.out, "w"), indent=1, default=str)   # checkpoint: safe to resume
    print(f"[{time.time()-t0:6.0f}s] ep {k:3d} {r['true_type']:<14} {r['post'][0]:<17} {r['post'][1]:<9} "
          f"alarm={r['alarm']} label={r.get('label')} skip={r.get('skipped_llm')} decision={r.get('decision')} "
          f"calls={r.get('llm_calls', 0)}", flush=True)
summary(eps)
