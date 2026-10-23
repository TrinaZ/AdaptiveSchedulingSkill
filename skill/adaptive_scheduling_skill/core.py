"""
core.py -- shared implementation behind the four skills (Monitor, Diagnose, Generate, Govern).

Every skill is a pure function from a typed input record (dict / JSON) to a typed output record; the CLI
wrappers in <skill>/run.py only read and write JSON. Records are defined in schemas/*.json.

Traffic is referred to by a TrafficSpec {"scene": str, "seed": int, "n": int}: in this repository the
"replay of recent traffic" is produced by the online SLA benchmark's instance generator (online_sla_bench.py);
in a production deployment the same field would point to logged request windows.
"""
import os, sys, math, json, time, hashlib
import numpy as np

_B = {}
def bench():
    """Load the benchmark definitions once (simulator, objectives, library, literature rules, LLM generator)."""
    if not _B:
        root = os.environ.get("ASS_REPO", os.getcwd())
        bench_py = os.path.join(root, "online_sla_bench.py")
        code = open(bench_py).read().split("\n# ---- fail fast")[0]
        argv = sys.argv
        fwd = ["bench", "--src", root, "--out", "/dev/null"]
        for k in ("ASS_BACKEND", "ASS_MODEL", "ASS_MAX_TOKENS", "ASS_TEMPERATURE"):
            if os.environ.get(k): fwd += ["--" + k[4:].lower(), os.environ[k]]
        sys.argv = fwd
        sys.path.insert(0, root)
        exec(code, _B)
        sys.argv = argv
    return _B

def traffic(spec):
    B = bench()
    return [B["make_instance"](spec["scene"], spec["seed"] + i, 60, 1.0) for i in range(spec["n"])]

# ------------------------------------------------------------------ Monitor
def monitor(rec):
    """TelemetryRecord -> ChangeReport. CUSUM on standardized log1p(cost) of the deployed rule."""
    c = [math.log1p(max(x, 0.0)) for x in rec["costs"]]
    ref = rec.get("reference_windows", 20); h = rec.get("h", 5.0); k = rec.get("k", 0.5)
    if rec.get("declared_change_at") is not None:
        return dict(type="ChangeReport", alarm=int(rec["declared_change_at"]), declared=True, statistic=None,
                    metric="SLA text changed (declared by operator)")
    mu = float(np.mean(c[:ref])); sd = float(np.std(c[:ref], ddof=1)) or 1.0; S = 0.0
    for t in range(ref, len(c)):
        S = max(0.0, S + (c[t] - mu) / sd - k)
        if S > h:
            return dict(type="ChangeReport", alarm=t, declared=False, statistic=S, metric="CUSUM on standardized log cost",
                        reference=dict(mean=mu, sd=sd, windows=ref), threshold=h)
    return dict(type="ChangeReport", alarm=None, declared=False, statistic=S, metric="CUSUM on standardized log cost")

# ------------------------------------------------------------------ Diagnose
def _features(insts):
    reqs = [r for rs, _ in insts for r in rs]
    return dict(capacity=float(np.mean([R for _, R in insts])), mean_estimate=float(np.mean([r.estimated_runtime for r in reqs])),
                request_types=sorted({r.task_type for r in reqs}))

def diagnose(rec):
    """DiagnoseInput -> DiagnosisReport: change type, diagnostics, library adequacy (headroom probe, no LLM)."""
    B = bench(); o = rec["objective"]
    ref_f, cur_f = _features(traffic(rec["reference_traffic"])), _features(traffic(rec["recent_traffic"]))
    if rec["change_report"].get("declared"): ctype = "semantic"
    elif abs(cur_f["capacity"] - ref_f["capacity"]) / ref_f["capacity"] > 0.05 or set(cur_f["request_types"]) - set(ref_f["request_types"]):
        ctype = "structural"
    else: ctype = "distributional"
    Is = traffic(rec["search_traffic"])
    lib = {n: B["run_cost"](f, Is, o) for n, f in B["LIB"].items()}; best_lib = min(lib, key=lib.get)
    lit = {lab: B["run_cost"](f, Is, o) for lab, f, _, _ in B["KNOW"]()}
    lit = {k: v for k, v in lit.items() if v is not None}; best_lit = min(lit, key=lit.get)
    headroom = 1.0 - lit[best_lit] / max(lib[best_lib], 1e-9)
    return dict(type="DiagnosisReport", change_type=ctype, reference_features=ref_f, recent_features=cur_f,
                objective=o, sla_text=B["OBJ_TEXT"][o], library_costs=lib, best_library=best_lib,
                best_literature=best_lit, best_literature_cost=lit[best_lit], headroom=headroom,
                adequate=bool(headroom < rec.get("headroom_threshold", 0.03)),
                diagnostics=B["diagnostics"](B["LIB"][best_lib], Is), search_traffic=rec["search_traffic"])

# ------------------------------------------------------------------ Generate
def generate(rec):
    """DiagnosisReport -> CandidateSet (llm_skill: literature rules + 3 LLM rounds + calibration). Untrusted output."""
    B = bench(); d = rec["diagnosis"]; o = d["objective"]
    if d["adequate"]:
        return dict(type="CandidateSet", candidates=[], llm_calls=0, tokens=0, note="library adequate: LLM not called")
    Is = traffic(d["search_traffic"])
    lib = {n: B["run_cost"](f, Is, o) for n, f in B["LIB"].items()}; best = min(lib, key=lib.get)
    kn = [(B["run_cost"](f, Is, o), lab, f, fam, p) for lab, f, fam, p in B["KNOW"]()]
    u0 = dict(B["USAGE"]); n0 = len(B["LLM_DEBUG"])
    lab, rule, calls, evals = B["llm_generate"]("llm_skill", o, Is, best, lib[best], kn, budget=rec.get("budget", 100),
                                                rng=np.random.default_rng(rec.get("seed", 0)))
    cands = []
    if lab not in B["LIB"]:
        src = getattr(rule, "src", None) or next((c for n, c, _ in B["KNOW_FAM"] if lab.startswith(n)), None)
        cands.append(dict(label=lab.split("\n")[0][:120], source=src, params=getattr(rule, "params", {}),
                          search_cost=B["run_cost"](rule, Is, o), llm_written="def priority" in lab))
    prompts = [x.get("prompt", "") for x in B["LLM_DEBUG"][n0:]]
    return dict(type="CandidateSet", candidates=cands, library_search_cost=lib[best], best_library=best, llm_calls=calls,
                evaluations=evals, tokens=(B["USAGE"]["in"] - u0["in"]) + (B["USAGE"]["out"] - u0["out"]),
                prompt_sha256=[hashlib.sha256(p.encode()).hexdigest()[:16] for p in prompts])

# ------------------------------------------------------------------ Govern
def govern(rec):
    """GovernInput -> SkillPackage. Static whitelist + isolated validation + paired gate + tier routing + fallback."""
    from govern_sandbox import isolated_costs, static_check, Rejected
    from scipy import stats as st
    B = bench(); o = rec["objective"]; tau = rec.get("tau", 0.02); delta = rec.get("delta", 0.05)
    Iv = traffic(rec["validation_traffic"])
    vv = {n: B["per_inst"](f, Iv, o) for n, f in B["LIB"].items()}; fb = min(vv, key=lambda n: vv[n].mean())
    K = 2 * len(vv); evidence = dict(n=len(Iv), tau=tau, delta=delta, gate=rec.get("gate", "t"),
                                     library_means={n: float(v.mean()) for n, v in vv.items()}, fallback=fb, candidates=[])
    chosen = None
    for c in rec["candidate_set"]["candidates"]:
        ev = dict(label=c["label"], source_sha256=hashlib.sha256((c["source"] or "").encode()).hexdigest()[:16])
        try:
            static_check(c["source"]); costs, why = isolated_costs(c["source"], c["params"], Iv, B["objective"], B["simulate"], o)
        except Rejected as e:
            costs, why = None, f"static: {e}"
        if costs is None:
            ev.update(passed=False, reason=f"rejected: {why}"); evidence["candidates"].append(ev); continue
        d = np.array(costs) - (1 + tau) * vv[fb]; n = len(d); V = float(d.var(ddof=1))
        if evidence["gate"] == "eb":
            lg = math.log(2 * K / delta); ub = d.mean() + math.sqrt(2 * V * lg / n) + 7 * float(d.max() - d.min()) * lg / (3 * (n - 1))
        else:
            ub = d.mean() + st.t.ppf(1 - delta / K, n - 1) * math.sqrt(V / n)
        ev.update(candidate_mean=float(np.mean(costs)), upper_bound=float(ub), passed=bool(ub <= 0))
        evidence["candidates"].append(ev)
        if ub <= 0 and chosen is None: chosen = c
    ctype = rec["change_type"]
    if chosen is not None:
        approval = {"distributional": "autonomous", "structural": "expert", "semantic": "human"}[ctype]
        decision = "deploy" if approval == "autonomous" else "deploy-after-approval"
        rule_src, params, name = chosen["source"], chosen["params"], chosen["label"]
    else:
        approval, decision = None, "fallback"
        rule_src, params, name = B["LIB_SRC"][fb], {}, fb
    return dict(type="SkillPackage", rule_name=name, rule_source=rule_src, params=params, change_type=ctype,
                decision=decision, approval=approval, evidence=evidence, rollback_to=rec.get("incumbent"),
                review_flag=bool(decision != "deploy"),
                provenance=dict(objective=o, llm_calls=rec["candidate_set"].get("llm_calls", 0),
                                tokens=rec["candidate_set"].get("tokens", 0),
                                prompt_sha256=rec["candidate_set"].get("prompt_sha256", []),
                                created=time.strftime("%Y-%m-%dT%H:%M:%S")))

def write_package(pkg, store):
    """Version the package in the knowledge store: store/pkg-XXXX.json + store/manifest.json (append-only)."""
    os.makedirs(store, exist_ok=True)
    mpath = os.path.join(store, "manifest.json")
    man = json.load(open(mpath)) if os.path.exists(mpath) else {"packages": []}
    vid = len(man["packages"]) + 1; fn = f"pkg-{vid:04d}.json"
    pkg = dict(pkg, version=vid, previous_version=(vid - 1 if vid > 1 else None))
    json.dump(pkg, open(os.path.join(store, fn), "w"), indent=1)
    man["packages"].append(dict(version=vid, file=fn, decision=pkg["decision"], rule=pkg["rule_name"][:60],
                                change_type=pkg["change_type"], created=pkg["provenance"]["created"]))
    json.dump(man, open(mpath, "w"), indent=1)
    return os.path.join(store, fn)
