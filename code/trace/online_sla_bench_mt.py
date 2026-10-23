"""
Online SLA benchmark for the Generate skill (pilot, no LLM).

Why a new benchmark: in the paper's WGS setting 20 tasks with resource needs
{4,8,16} share 100 cores and all arrive at t=0, so ~10 run concurrently, there is
almost no queueing, and every rule is within a few % of every other. Nothing a
generator writes can matter much, LLM or not.

This variant keeps the WGS task generator (same types, runtimes, resources,
estimated_runtime noise) but turns it into an online service:
  * N requests arrive by a Poisson process at offered load rho (default 0.9),
  * each request has a tenant (premium 20% / standard 80%) and an SLA due time
    = release + slack_factor * estimated_runtime (slack_factor ~ U[1.5, 4]),
  * the dispatcher is DYNAMIC: whenever capacity frees up it re-scores all waiting
    requests with priority(task, state), so rules may use time and queue state.
Observable fields (true runtime is hidden from every rule):
  task.estimated_runtime, task.resource_need, task.weight, task.task_type,
  task.tenant ('premium'|'standard'), task.release, task.due
State fields: state.now, state.free, state.capacity, state.queue_len,
  state.queue_work (sum est*res of waiting), state.mean_est (mean est of waiting)
Objectives (lower is better), each with a natural-language SLA text (OBJ_TEXT):
  wflow, tard (weighted tardiness), miss (deadline-miss count),
  p95 (95th-pct response time), premium (premium response x5 + standard),
  slo_wait (penalty when queueing delay > 2x est), mix (premium tardiness x4 + standard p90)
Generators compared here:
  library   : best of FIFO, SPT, LPT, EDF, WSJF, ResourceEfficient (static, on search set)
  pool_old  : 100 random static templates (runtime/resource forms)
  pool_full : 100 random static templates from the paper's pool (estimated runtime only)
  knowledge : textbook dynamic rules (ATC, critical ratio, min slack, MOD, WSPT-by-tenant),
              each with a small constant grid tuned on the search set -- a PROXY for what an
              LLM with scheduling knowledge could write from the SLA text; it is NOT an LLM result.
Selection on SEARCH instances (seed s), report on HELD-OUT instances (seed s+50).
"""
import sys, math, json, argparse, itertools, time
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--rho", type=float, default=0.9)
ap.add_argument("--scenes", default="T0,T1,T0as,T1as")
ap.add_argument("--tenants", type=int, default=4, help="tenant services sharing the pool; tenant 0 is premium")
ap.add_argument("--trace", default="alibaba_trace_windows.csv")
ap.add_argument("--thin_conc", type=float, default=48.0, help="target mean number of concurrently running tasks after job thinning")
ap.add_argument("--seeds", type=int, default=6)
ap.add_argument("--n_search", type=int, default=8)
ap.add_argument("--n_heldout", type=int, default=60)
ap.add_argument("--objectives", default="tard,miss,slo_wait,mix,premium")
ap.add_argument("--out", required=True)
ap.add_argument("--gens", default="library,pool_old,pool_full,knowledge")
ap.add_argument("--backend", default="deepseek", help="deepseek | qwen | openai | glm | kimi | custom (OpenAI-compatible endpoints)")
ap.add_argument("--model", default=None, help="model name; default per backend (see BACKENDS)")
ap.add_argument("--base_url", default=None, help="only for --backend custom")
ap.add_argument("--key_env", default=None, help="only for --backend custom: name of the env var holding the key")
ap.add_argument("--temperature", default="0.7", help="'none' to omit (some reasoning models reject it)")
ap.add_argument("--max_tokens", type=int, default=2500, help="raise to ~8000 for reasoning models")
ap.add_argument("--tag", default="", help="suffix for LLM generator names, e.g. qwen-max -> llm_skill@qwen-max")
ap.add_argument("--n_val", type=int, default=1000, help="validation instances for the Govern gate (0 = no gate)")
ap.add_argument("--tau", type=float, default=0.02)
ap.add_argument("--budget", type=int, default=100, help="rule evaluations on the search set per generator (pool_full, knowledge_tuned, llm_*)")
a = ap.parse_args()
sys.path.insert(0, a.src)

class Req:
    __slots__ = ("id","task_type","runtime","estimated_runtime","resource_need","weight","tenant","release","due")


# ---------------- trace-driven instances (Alibaba cluster-trace-v2018, batch_task) ----------------
# Scenes: T<w> = trace window w at its natural load; T<w>cap = same window with 30% less capacity.
# Each instance = n consecutive real tasks (real arrival times, real durations, real CPU requests) starting at a random
# position inside a ROLE-SPECIFIC part of the window, so search, validation and held-out traffic never overlap and come
# in temporal order: search = first 20% of the window, validation = 20-60%, held-out = last 40% (deploy on the future).
# The cluster-wide stream is thinned by whole jobs to the scale of one service (--thin_conc concurrent tasks on average).
# Capacity R is fixed per window so that the window's average offered load equals --rho; bursts are left as they are.
# Not in the trace (synthesised exactly as in the WGS benchmark): runtime estimate = runtime x U[0.6,1.4]; due time;
# tenant (20% of JOBS are premium, by a stable hash of the job name) and weight.
import csv as _csv, zlib as _zlib
_TR = {}
def _trace():
    if not _TR:
        cols = {}
        with open(a.trace, newline="") as f:
            for r in _csv.DictReader(f):
                w = int(r["window"]); cols.setdefault(w, []).append((float(r["start"]), float(r["dur"]), float(r["cpu"]), r["kind"], r["job"]))
        for w, L in cols.items():
            L.sort(key=lambda x: x[0])
            span = max(L[-1][0] - L[0][0], 1.0)
            # job-level thinning to the scale of one service: keep whole jobs (crc32(job) % M == 0) so that the mean
            # number of concurrently running tasks is about --thin_conc; bursts inside jobs and real durations are kept
            conc = len(L) / span * float(np.mean([d for _, d, _, _, _ in L]))
            M = max(1, int(round(conc / a.thin_conc)))
            L = [x for x in L if _zlib.crc32(x[4].encode()) % M == 0]
            span = max(L[-1][0] - L[0][0], 1.0)
            work = sum(d * c for _, d, c, _, _ in L)
            _TR[w] = (L, work / span, M)
            print(f"[trace] window {w}: thinning 1/{M} of jobs -> {len(L):,} tasks, {len(L)/span*60:.2f} arrivals/min, "
                  f"capacity at rho=1: {work/span:.1f} cores", flush=True)
    return _TR
_KIND = {"M": "map", "R": "reduce", "J": "join", "O": "other"}
def make_instance(scene, seed, n, rho):
    w = int(scene[1]); auto = scene.endswith("as")
    L, rate, M = _trace()[w]
    k = (seed - 1000) // 10000
    lo, hi = (0.0, 0.2) if k < 30 else ((0.2, 0.6) if k < 50 else (0.6, 1.0))
    rng = np.random.RandomState(seed + 7)
    i0 = int(lo * len(L)); i1 = max(i0 + 1, int(hi * len(L)) - n)
    st = rng.randint(i0, i1)
    R = rate / rho
    if auto:   # reactive autoscaling: capacity from the offered load of the n tasks that arrived just before this instance
        prev = L[max(0, st - n):st]
        if len(prev) >= 2:
            sp = max(prev[-1][0] - prev[0][0], 1.0)
            R = min(max(sum(d * c for _, d, c, _, _ in prev) / sp / rho, 0.5 * R), 2.0 * R)
    reqs = []; t0 = L[st][0]
    for j, (s0, d, c, kind, job) in enumerate(L[st:st + n]):
        r = Req(); r.id = j; r.task_type = _KIND.get(kind, "other"); r.runtime = d
        r.estimated_runtime = d * rng.uniform(0.6, 1.4); r.resource_need = min(c, R)
        prem = ((_zlib.crc32(job.encode()) // M) % a.tenants) == 0   # tenant service 0 is premium
        r.tenant = "premium" if prem else "standard"
        r.weight = (3.0 if prem else 1.0) * rng.uniform(0.8, 1.2)
        r.release = s0 - t0; r.due = r.release + rng.uniform(1.5, 4.0) * r.estimated_runtime
        reqs.append(r)
    return reqs, float(R)

class TV:  # observable view
    __slots__ = ("estimated_runtime","resource_need","weight","task_type","tenant","release","due","id")
    def __init__(self, r):
        for k in self.__slots__: setattr(self, k, getattr(r, k))
    def get(self, k, d=None): return getattr(self, k, d)
    def __getitem__(self, k): return getattr(self, k)
class SV:
    __slots__ = ("now","free","capacity","queue_len","queue_work","mean_est")
    def get(self, k, d=None): return getattr(self, k, d)
    def __getitem__(self, k): return getattr(self, k)

def simulate(reqs, R, rule):
    """Event-driven list scheduler with backfilling; rule(task_view, state_view) -> float (higher first)."""
    pending = sorted(reqs, key=lambda r: r.release); pi = 0
    waiting, running = [], []  # running: (finish_time, req)
    now, free, done = 0.0, R, {}
    while pi < len(pending) or waiting or running:
        while pi < len(pending) and pending[pi].release <= now + 1e-12:
            waiting.append(pending[pi]); pi += 1
        if waiting and free > 0:
            st = SV(); st.now = now; st.free = free; st.capacity = R; st.queue_len = len(waiting)
            st.queue_work = sum(w.estimated_runtime * w.resource_need for w in waiting)
            st.mean_est = float(np.mean([w.estimated_runtime for w in waiting]))
            scored = sorted(waiting, key=lambda w: -rule(TV(w), st))   # errors propagate -> candidate invalid
            started = []
            for w in scored:
                if w.resource_need <= free + 1e-9:
                    free -= w.resource_need; running.append((now + w.runtime, w)); started.append(w)
            if not running and waiting:   # oversize request: run alone
                w = scored[0]; free -= w.resource_need; running.append((now + w.runtime, w)); started.append(w)
            for w in started: waiting.remove(w)
        nxt = [f for f, _ in running]
        if pi < len(pending): nxt.append(pending[pi].release)
        if not nxt: break
        now = min(nxt)
        still = []
        for f, w in running:
            if f <= now + 1e-12: done[w.id] = f; free += w.resource_need
            else: still.append((f, w))
        running = still
    return done

def objective(name, reqs, done):
    resp = np.array([done[r.id] - r.release for r in reqs])
    late = np.array([max(0.0, done[r.id] - r.due) for r in reqs])
    w = np.array([r.weight for r in reqs]); prem = np.array([r.tenant == "premium" for r in reqs])
    est = np.array([r.estimated_runtime for r in reqs])
    wait = resp - np.array([r.runtime for r in reqs])
    if name == "wflow": return float((w * resp).sum())
    if name == "tard": return float((w * late).sum())
    if name == "miss": return float((late > 1e-9).sum())
    if name == "p95": return float(np.percentile(resp, 95))
    if name == "premium": return float(5 * resp[prem].sum() + resp[~prem].sum())
    if name == "slo_wait": return float(np.maximum(0.0, wait - 2.0 * est).sum())
    if name == "mix": return float(4 * late[prem].sum() + (np.percentile(resp[~prem], 90) if (~prem).any() else 0.0))
    # ---- H7 objectives: SLA shapes that no family in the fixed knowledge base (KNOW_FAM) was designed for ----
    if name == "tiered":   # step-priced penalty per late request, by lateness relative to the estimate; premium x3
        rel_late = late / np.maximum(est, 1e-6)
        pen = np.where(late <= 1e-9, 0.0, np.where(rel_late <= 0.25, 1.0, np.where(rel_late <= 1.0, 3.0, 10.0)))
        return float((pen * np.where(prem, 3.0, 1.0)).sum())
    if name == "typemiss":  # deadline misses priced by pipeline stage (downstream clinical stages cost more)
        val = np.array([{"variant_calling": 5.0, "sv_calling": 3.0}.get(r.task_type, 1.0) for r in reqs])
        return float((val * (late > 1e-9)).sum())
    if name == "decay":    # lost revenue: value = weight until due, decays linearly to 0 at due + 2*estimate
        frac = np.clip(late / np.maximum(2.0 * est, 1e-6), 0.0, 1.0)
        return float((w * frac).sum())
    raise ValueError(name)

OBJ_TEXT = {
 "wflow": "Minimize total weighted response time (weight x (completion - release)).",
 "tard": "Minimize total weighted tardiness: weight x max(0, completion - due).",
 "miss": "Minimize the number of requests that finish after their due time.",
 "p95": "Minimize the 95th-percentile response time (completion - release) across all requests.",
 "premium": "Premium-tenant response time counts 5x, standard-tenant 1x; minimize the weighted sum of response times.",
 "slo_wait": "Queueing-delay SLO: a request's waiting time should not exceed 2x its estimated runtime; minimize total excess waiting.",
 "mix": "Premium requests have hard due times (each unit of premium tardiness costs 4); for standard requests minimize the 90th-percentile response time.",
 # H7 objectives (no family in the knowledge base was designed for these shapes)
 "tiered": ("Tiered SLA credits: a request that finishes after its due time costs a flat penalty that depends on how late it is "
            "relative to its estimated runtime: 1 if lateness <= 0.25 x estimate, 3 if lateness <= 1 x estimate, 10 if later than that. "
            "Premium-tenant penalties count 3x. On-time requests cost 0. Minimize the total penalty."),
 "typemiss": ("Stage-priced deadline misses: a request that finishes after its due time costs 5 if it is a variant_calling request, "
              "3 if sv_calling, and 1 if alignment (downstream clinical stages matter more); lateness beyond the due time does not "
              "matter. Minimize the total cost."),
 "decay": ("Revenue decay: each request earns its weight if it finishes by its due time; after the due time its value decays "
           "linearly and reaches 0 at due + 2 x estimated runtime (no further loss after that). Minimize the total lost value."),
}

LAST_ERR = [None]
def run_cost(rule, insts, oname):
    vals = []
    for reqs, R in insts:
        try: vals.append(objective(oname, reqs, simulate(reqs, R, rule)))
        except Exception as e:
            LAST_ERR[0] = f"{type(e).__name__}: {e}"[:200]; return None
    return float(np.mean(vals))

def per_inst(rule, insts, oname):
    out = []
    for reqs, R in insts:
        try: out.append(objective(oname, reqs, simulate(reqs, R, rule)))
        except Exception: return None
    return np.array(out)

def static_fn(code):
    ns = {"math": math, "max": max, "min": min, "abs": abs}; exec(code, ns); f = ns["priority"]
    return lambda t, s: float(f(t))

LIB = {
 "FIFO": lambda t, s: -t.release,
 "SPT": lambda t, s: -t.estimated_runtime,
 "LPT": lambda t, s: t.estimated_runtime,
 "EDF": lambda t, s: -t.due,
 "WSJF": lambda t, s: t.weight / max(t.estimated_runtime, 1e-6),
 "ResourceEfficient": lambda t, s: -t.resource_need,
}
POOL_OLD = [
 ("def priority(task): return task.estimated_runtime ** {p}", {"p": [0.5, 2.0]}),
 ("def priority(task): return -task.estimated_runtime ** {p}", {"p": [0.5, 2.0]}),
 ("def priority(task): return {a} / max(task.resource_need, 1.0) + {b} / max(task.estimated_runtime, 1e-6)", {"a": [0.5, 2.0], "b": [0.5, 2.0]}),
 ("def priority(task): return task.resource_need ** {p} * task.estimated_runtime", {"p": [0.5, 2.0]}),
]
POOL_FULL = [  # paper's _EOH_TEMPLATES with runtime -> estimated_runtime (static, per-task)
 ("def priority(task): return (task.weight ** {w}) / max(task.estimated_runtime ** {p}, 1e-6)", {"w": [0.5, 2.0], "p": [0.5, 1.5]}),
 ("def priority(task): return task.estimated_runtime ** {p}", {"p": [1.0, 2.0]}),
 ("def priority(task): return task.weight ** {w} * task.estimated_runtime ** {p}", {"w": [0.5, 2.0], "p": [1.0, 2.0]}),
 ("def priority(task): return task.estimated_runtime / max(task.weight ** {w}, 1e-6)", {"w": [0.5, 2.0]}),
 ("def priority(task): return 1.0 / max(task.due - task.estimated_runtime, {e})", {"e": [0.1, 5.0]}),
 ("def priority(task): return {a} / max(task.resource_need, 1.0) + {b} / max(task.estimated_runtime, 1e-6)", {"a": [0.5, 2.0], "b": [0.5, 2.0]}),
 ("def priority(task): return task.weight / max(task.due - task.estimated_runtime, {e})", {"e": [0.1, 5.0]}),
 ("def priority(task): return task.weight * task.estimated_runtime", {"dummy": [1.0, 1.0]}),
]
def sample(T, rng, n):
    out = []
    for _ in range(n):
        t, g = T[rng.integers(len(T))]
        p = {k: float(np.exp(rng.uniform(np.log(max(min(v), 1e-3)), np.log(max(max(v), 1e-3))))) for k, v in g.items()}
        out.append(t.format(**p))
    return out

# textbook dynamic rules, written as source with keyword constants. The SAME source is used by
#   knowledge       : small constant grid per family (26 rules),
#   knowledge_tuned : grid + random constant calibration at the same evaluation budget as the LLM generators,
#   llm_skill       : shown to the LLM (with their costs) as the knowledge context it must improve on.
KNOW_FAM = [
 ("ATC", """def priority(task, state, k=2.0):
    slack = max(0.0, task.due - task.estimated_runtime - state.now)
    return task.weight / max(task.estimated_runtime, 1e-6) * math.exp(-slack / (k * max(state.mean_est, 1e-6)))""",
  [dict(k=k) for k in (0.5, 1, 2, 4)]),
 ("CR", """def priority(task, state):
    return -(task.due - state.now) / max(task.estimated_runtime, 1e-6)""", [{}]),
 ("MinSlack", """def priority(task, state):
    return -(task.due - state.now - task.estimated_runtime)""", [{}]),
 ("MOD", """def priority(task, state):
    return -max(task.due, state.now + task.estimated_runtime)""", [{}]),
 ("AgingWSPT", """def priority(task, state, b=1.0):
    wait = state.now - task.release
    return task.weight / max(task.estimated_runtime, 1e-6) * (1 + b * wait / max(task.estimated_runtime, 1e-6))""",
  [dict(b=b) for b in (0.5, 1, 2)]),
 ("TenantSPT", """def priority(task, state, m=5.0):
    return (m if task.tenant == 'premium' else 1.0) / max(task.estimated_runtime, 1e-6)""",
  [dict(m=m) for m in (2, 5, 10)]),
 ("TenantATC", """def priority(task, state, m=2.0, k=1.0):
    slack = max(0.0, task.due - task.estimated_runtime - state.now)
    tier = m if task.tenant == 'premium' else 1.0
    return tier * task.weight / max(task.estimated_runtime, 1e-6) * math.exp(-slack / (k * max(state.mean_est, 1e-6)))""",
  [dict(m=m, k=k) for m in (2, 5) for k in (1, 2)]),
 ("WaitGuard", """def priority(task, state, c=2.0):
    ratio = (state.now - task.release) / max(task.estimated_runtime, 1e-6)
    return (1e6 if ratio >= c else 0.0) + 1.0 / max(task.estimated_runtime, 1e-6)""",
  [dict(c=c) for c in (1, 2, 3)]),
]
# H7 expert reference: rules the authors wrote by hand for the H7 objectives, frozen (git-tagged) BEFORE any LLM run on
# these objectives. Used only by the `expert` generator (same calibration and budget as knowledge_tuned).
EXPERT_H7 = {
 "tiered": [("TriageATC", """def priority(task, state, h=1.0, m=3.0, k=1.0):
    slack = task.due - state.now - task.estimated_runtime
    tier = m if task.tenant == 'premium' else 1.0
    if slack < -h * task.estimated_runtime:
        return -1e6 + 1.0 / max(task.estimated_runtime, 1e-6)
    return tier / max(task.estimated_runtime, 1e-6) * (2.0 if slack < 0 else 1.0) * math.exp(-max(slack, 0.0) / (k * max(state.mean_est, 1e-6)))""", dict(h=1.0, m=3.0, k=1.0))],
 "typemiss": [("StageTriageATC", """def priority(task, state, h=0.5, a=5.0, b=3.0):
    slack = task.due - state.now - task.estimated_runtime
    v = a if task.task_type == 'variant_calling' else (b if task.task_type == 'sv_calling' else 1.0)
    if slack < 0:
        return -1e6 + v / max(task.estimated_runtime, 1e-6)
    return v / max(task.estimated_runtime, 1e-6) * math.exp(-slack / (h * max(state.mean_est, 1e-6)))""", dict(h=0.5, a=5.0, b=3.0))],
 "decay": [("TriageATC", """def priority(task, state, h=2.0, k=1.0):
    slack = task.due - state.now - task.estimated_runtime
    if slack < -h * task.estimated_runtime:
        return -1e6 + task.weight / max(task.estimated_runtime, 1e-6)
    return task.weight / max(task.estimated_runtime, 1e-6) * math.exp(-max(slack, 0.0) / (k * max(state.mean_est, 1e-6)))""", dict(h=2.0, k=1.0))],
}
def compile_rule(code, check=True):
    if check:
        from safe_exec import ast_safety_check
        ast_safety_check(code)
    ns = {"math": math, "max": max, "min": min, "abs": abs}; exec(code, ns); f = ns["priority"]
    f._src = code   # kept so the Govern skill can re-run the rule in an isolated process
    import inspect
    defaults = {k: v.default for k, v in inspect.signature(f).parameters.items()
                if v.default is not inspect.Parameter.empty and isinstance(v.default, (int, float)) and not isinstance(v.default, bool)}
    return f, defaults
def bind(f, p):
    r = lambda t, s: float(f(t, s, **p))
    r.src, r.params = getattr(f, "_src", None), dict(p)   # source + calibrated constants of the rule
    return r
def plabel(p): return "(" + ",".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in p.items()) + ")" if p else ""
def KNOW():
    """list of (label, rule, family, params)"""
    out = []
    for name, src, grid in KNOW_FAM:
        f, _ = compile_rule(src, check=False)
        for p in grid: out.append((name + plabel(p), bind(f, p), name, p))
    return out

def calibrate(structs, Is, oname, budget, rng, keep=2):
    """Numeric constant calibration that limits overfitting to the search set.
    structs: list of (label, f, center_params). Random log-scale perturbations (x0.25..x4) of the centre are scored on
    HALF of the search set; the `keep` best settings per structure are returned and must then compete on the FULL search
    set (that comparison costs keep evaluations per structure, included in the returned eval count)."""
    tun = [x for x in structs if x[2]]
    if not tun or budget <= keep * len(tun): return [], 0
    A = Is[: max(1, len(Is) // 2)]
    per = (budget - keep * len(tun)) // len(tun); out = []; ev = 0
    for lab, f, c in tun:
        tried = []
        for _ in range(per):
            p = {k: v * float(np.exp(rng.uniform(np.log(0.25), np.log(4.0)))) for k, v in c.items()}
            v = run_cost(bind(f, p), A, oname); ev += 1
            if v is not None: tried.append((v, p))
        for _, p in sorted(tried, key=lambda x: x[0])[:keep]:
            out.append((lab + "  # tuned " + plabel(p), bind(f, p)))
    return out, ev

# ---------------------------------------------------------------- LLM generators
# Run locally with an API key:  DEEPSEEK_API_KEY=... python3 online_sla_bench.py --src <repo> --gens llm_zeroshot,llm_tuned,llm_feedback ...
FIELD_DOC = ("Each request `task` exposes ONLY: task.estimated_runtime (noisy estimate, float), task.resource_need (cores, float), "
             "task.weight (float), task.task_type ('map'|'reduce'|'join'|'other'), task.tenant ('premium'|'standard'), "
             "task.release (arrival time), task.due (SLA due time). The true runtime is NOT observable.\n"
             "`state` exposes: state.now (current time), state.free (free cores), state.capacity (total cores), "
             "state.queue_len (waiting requests), state.queue_work (sum of est*cores waiting), state.mean_est (mean est runtime of waiting requests).\n")
def llm_prompt(oname, inc_code, inc_cost, history, diag, tunable, knowledge=None):
    s = ("You are designing a dispatching rule for an shared multi-tenant service cluster (requests of four tenant services replayed from a production cluster trace; capacity may be autoscaled). Requests arrive over time; "
         "whenever cores free up, the dispatcher computes priority(task, state) for every WAITING request and starts them in "
         "descending priority while they fit (backfilling). The rule is re-evaluated at every dispatch decision, so it may depend on time.\n"
         + FIELD_DOC +
         f"SERVICE OBJECTIVE (lower cost is better): {OBJ_TEXT[oname]}\n")
    if knowledge:
        s += ("Known rules from the scheduling literature, evaluated on recent traffic for THIS objective "
              "(cost, lower is better; constants are the best of a small grid):\n" + knowledge +
              "Your rules must beat the best of these on this objective. Useful moves: combine components that address "
              "different parts of the objective (e.g. tenant tiers, slack/ATC urgency, waiting-time aging, SLO guards), "
              "make the rule switch behaviour with queue state or lateness, or reshape a term. Restating a listed rule is not useful.\n")
    s += f"Current best rule (cost {inc_cost:.4g}):\n{inc_code}\n"
    if diag: s += f"Diagnostics of the current best rule on recent traffic: {diag}\n"
    if history:
        s += "Rules already tried (cost):\n" + "\n".join(f"- {c:.4g}: {code.strip()}" for code, c in history[-6:]) + "\n"
    if tunable:
        s += ("Write rules whose numeric constants are keyword arguments with defaults, e.g. "
              "`def priority(task, state, k=2.0, m=5.0):`; constants will be calibrated automatically, so focus on the STRUCTURE.\n")
    s += ("Return EXACTLY 5 different rules. Put EACH rule in its own separate ```python code block; each block must be "
          "self-contained and define one function `def priority(task, state)` (optionally with extra keyword constants with "
          "numeric defaults) returning a float, higher = dispatched first. Only `math` may be imported. "
          "Protect divisions with max(x, 1e-6). Write no text outside the code blocks.")
    return s
LIB_SRC = {
 "FIFO": "def priority(task, state): return -task.release",
 "SPT": "def priority(task, state): return -task.estimated_runtime",
 "LPT": "def priority(task, state): return task.estimated_runtime",
 "EDF": "def priority(task, state): return -task.due",
 "WSJF": "def priority(task, state): return task.weight / max(task.estimated_runtime, 1e-6)",
 "ResourceEfficient": "def priority(task, state): return -task.resource_need",
}
def parse_rules(raw):
    """Robust: accepts fenced blocks (```python / ```py / ```) or bare code; one rule per `def priority`."""
    import re
    if not raw: return []
    blocks = re.findall(r"```[ \t]*(?:python|py)?[ \t]*\n([\s\S]*?)```", raw)
    if not blocks: blocks = [raw]
    rules = []
    for b in blocks:
        lines = b.replace("\r\n", "\n").split("\n")
        header = [l for l in lines if re.match(r"\s*(import math|from math import)", l) and not l.startswith((" ", "\t"))]
        cur = None
        for l in lines:
            if l.startswith("def priority"):
                if cur: rules.append(cur)
                cur = [l]
            elif cur is not None:
                # top-level non-code prose ends the current function
                if l and not l[0].isspace() and not l.startswith(("#", "import", "from", "def ", "@")) and not re.match(r"[A-Za-z_]\w*\s*=", l):
                    rules.append(cur); cur = None
                elif l.startswith(("def ", "@")) and not l.startswith("def priority"):
                    rules.append(cur); cur = None
                else:
                    cur.append(l)
        if cur: rules.append(cur)
        rules = [r if isinstance(r, str) else "\n".join(header + r).strip() for r in rules]
    return [r for r in rules if "def priority" in r][:5]
def load_rule(code): return compile_rule(code, check=True)
def diagnostics(rule, insts):
    reqs, R = insts[0]; done = simulate(reqs, R, rule)
    late = [r for r in reqs if done[r.id] > r.due]; prem = [r for r in reqs if r.tenant == "premium"]
    wait = [done[r.id] - r.release - r.runtime for r in reqs]
    return (f"{len(late)}/{len(reqs)} late ({sum(r.tenant=='premium' for r in late)} premium); "
            f"mean response premium {np.mean([done[r.id]-r.release for r in prem]) if prem else 0:.1f} vs all {np.mean([done[r.id]-r.release for r in reqs]):.1f}; "
            f"max waiting {max(wait):.1f}, mean est {np.mean([r.estimated_runtime for r in reqs]):.1f}")
BACKENDS = {  # OpenAI-compatible endpoints: (base_url, env var with the key, default model)
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", "deepseek-chat"),
    "qwen": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY", "qwen-plus"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4o-mini"),
    "glm": ("https://open.bigmodel.cn/api/paas/v4", "ZHIPUAI_API_KEY", "glm-4-plus"),
    "kimi": ("https://api.moonshot.cn/v1", "MOONSHOT_API_KEY", "moonshot-v1-32k"),
}
def backend_conf():
    if a.backend == "custom":
        if not (a.base_url and a.key_env and a.model): sys.exit("ERROR: --backend custom needs --base_url, --key_env and --model")
        return a.base_url, a.key_env, a.model
    if a.backend not in BACKENDS: sys.exit(f"ERROR: unknown backend {a.backend}; choose from {list(BACKENDS)} or custom")
    url, env, model = BACKENDS[a.backend]
    return url, env, a.model or model
USAGE = {"calls": 0, "in": 0, "out": 0, "sec": 0.0}
_CLIENT = [None]
def call_llm_raw(prompt, max_tokens=None, retries=3):
    """Full response text. (llm_adapter.call_llm runs _extract_code, which strips fences and keeps only ONE function.)"""
    import os, time as _t
    url, env, model = backend_conf()
    if _CLIENT[0] is None:
        from openai import OpenAI
        try:
            from llm_adapter import _build_http_client
            hc = _build_http_client()
        except Exception:
            hc = None
        kw = dict(api_key=os.environ[env], base_url=url)
        if hc is not None: kw["http_client"] = hc
        _CLIENT[0] = OpenAI(**kw)
    req = dict(model=model, max_tokens=max_tokens or a.max_tokens, messages=[{"role": "user", "content": prompt}])
    if str(a.temperature).lower() != "none": req["temperature"] = float(a.temperature)
    for i in range(retries):
        try:
            t = _t.time(); r = _CLIENT[0].chat.completions.create(**req); USAGE["sec"] += _t.time() - t
            USAGE["calls"] += 1
            u = getattr(r, "usage", None)
            if u is not None:
                USAGE["in"] += getattr(u, "prompt_tokens", 0) or 0; USAGE["out"] += getattr(u, "completion_tokens", 0) or 0
            ch = r.choices[0]; txt = ch.message.content
            if not txt:
                fr = getattr(ch, "finish_reason", None)
                print(f"  [LLM returned empty content, finish_reason={fr}"
                      + ("; the output-token limit was used up (reasoning models need --max_tokens 8000 or more)" if fr == "length" else "") + "]", flush=True)
            return txt
        except Exception as e:
            msg = str(e)
            print(f"  [LLM attempt {i+1} failed: {type(e).__name__}: {msg[:120]}]", flush=True)
            if "402" in msg or "Insufficient Balance" in msg or "insufficient_quota" in msg:
                sys.exit("ERROR: the API account has no credit left (402 / insufficient balance). Results so far are NOT valid; top up and rerun.")
            _t.sleep(2 * (i + 1))
    return None
LLM_FAILS = 0
LLM_DEBUG = []
ROUNDS = {"llm_zeroshot": 1, "llm_tuned": 1, "llm_feedback": 3, "llm_skill": 3}
EOH_OPS = {  # Evolution of Heuristics (Liu et al., ICML 2024): two exploration and two modification operators
    "E1": "Create 5 new rules that are as DIFFERENT as possible from the parent rules below.",
    "E2": "Identify the common idea behind the parent rules below, then create 5 new rules that build on that idea in different ways.",
    "M1": "Modify the best parent rule below to create 5 improved variants with different structure.",
    "M2": "Keep the structure of the best parent rule below but change its numeric parameters or term weights to create 5 variants.",
}
def llm_eoh_generate(oname, Is, lib_name, lib_cost, rng, generations=5, pop_size=5):
    """EoH-style population search (no literature rules, no numeric calibration): 1 initialization call +
    `generations` evolution calls cycling E1, E2, M1, M2; population = best `pop_size` rules by search cost."""
    pop, calls, evals = [], 0, 0
    best = (lib_cost, lib_name, LIB[lib_name])
    base = ("You are designing a dispatching rule for an shared multi-tenant service cluster (requests of four tenant services replayed from a production cluster trace; capacity may be autoscaled). Requests arrive over time; "
            "whenever cores free up, the dispatcher computes priority(task, state) for every WAITING request and starts them in "
            "descending priority while they fit (backfilling). The rule is re-evaluated at every dispatch decision, so it may depend on time.\n"
            + FIELD_DOC + f"SERVICE OBJECTIVE (lower cost is better): {OBJ_TEXT[oname]}\n"
            f"Best library rule (cost {lib_cost:.4g}):\n{LIB_SRC[lib_name]}\n")
    tail = ("Start each rule with a one-line comment '# idea: ...' describing its idea. "
            "Return EXACTLY 5 different rules. Put EACH rule in its own separate ```python code block; each block must be "
            "self-contained and define one function `def priority(task, state)` returning a float, higher = dispatched first. "
            "Only `math` may be imported. Protect divisions with max(x, 1e-6). Write no text outside the code blocks.")
    ops = ["E1", "E2", "M1", "M2"]
    for gen in range(generations + 1):
        if gen == 0:
            op, prompt = "init", base + tail
        else:
            op = ops[(gen - 1) % 4]
            ranked = sorted(pop, key=lambda x: x[0])
            parents = ranked[:1] if op.startswith("M") else [ranked[i] for i in sorted(rng.choice(len(ranked), size=min(3, len(ranked)), replace=False))]
            prompt = base + EOH_OPS[op] + "\nParent rules (cost):\n" + "\n".join(f"- cost {c:.4g}:\n{code}" for c, code, _ in parents) + "\n" + tail
        raw = call_llm_raw(prompt); calls += 1
        dbg = dict(mode="llm_eoh", objective=oname, round=gen, operator=op, library=lib_name, library_search_cost=lib_cost,
                   incumbent_cost=best[0], prompt=prompt, raw=raw, candidates=[])
        LLM_DEBUG.append(dbg)
        if not raw:
            global LLM_FAILS
            LLM_FAILS += 1
            if LLM_FAILS >= 5: sys.exit("ERROR: 5 LLM calls returned nothing; aborting so results are not silently replaced by library rules.")
            continue
        for code in parse_rules(raw):
            entry = dict(code=code); dbg["candidates"].append(entry)
            try: f, d = load_rule(code)
            except Exception as e:
                entry["load_error"] = f"{type(e).__name__}: {e}"[:200]; continue
            rule = bind(f, {}); LAST_ERR[0] = None
            v = run_cost(rule, Is, oname); evals += 1
            entry["search_cost"] = v; entry["run_error"] = LAST_ERR[0]
            if v is None: continue
            pop.append((v, code, rule))
            if v < best[0]: best = (v, code, rule)
        pop = sorted(pop, key=lambda x: x[0])[:pop_size]
        if not pop:
            continue
    if best[0] < lib_cost: return best[1], best[2], calls, evals
    return lib_name, LIB[lib_name], calls, evals
def llm_generate(mode, oname, Is, lib_name, lib_cost, kn_scored, budget=100, rng=None):
    """Every generator selects on the search set Is and falls back to the library rule unless it beats it there.
    Evaluation budget (rule x search-set evaluations) is shared: LLM rules + constant calibration (+ the knowledge
    rules shown to llm_skill) <= budget, the same budget pool_full and knowledge_tuned get.
      llm_zeroshot : 1 call, 5 rules, no calibration                   (minimal ablation)
      llm_tuned    : 1 call, 5 structures + calibration
      llm_feedback : 3 calls with cost history + diagnostics, + calibration (no literature knowledge)
      llm_skill    : as llm_feedback, but round 1 sees the literature rules and their costs on this objective,
                     and those rules are also candidates (the Generate skill: LLM + knowledge base)."""
    call_llm = call_llm_raw
    cands = []; evals = 0; calls = 0; hist = []; structs = []
    best = (lib_cost, lib_name, LIB[lib_name])
    inc_code = f"# best library rule ({lib_name})\n" + LIB_SRC[lib_name]
    know_txt = None
    if mode == "llm_skill":
        evals += len(kn_scored)
        fam_best = {}
        for v, lab, f, fam, p in kn_scored:
            cands.append((v, lab, f))
            if v is not None and (fam not in fam_best or v < fam_best[fam][0]): fam_best[fam] = (v, lab, p)
        src = {n: c for n, c, _ in KNOW_FAM}
        rows = sorted(fam_best.items(), key=lambda x: x[1][0])
        def with_p(code, p):  # show the grid-best constants as the defaults
            import re as _re
            head, rest = code.split("\n", 1)
            for k, val in p.items(): head = _re.sub(rf"\b{k}=[-0-9.e]+", f"{k}={float(val)}", head)
            return head + "\n" + rest
        know_txt = "".join(f"- {fam}  cost {v:.4g}\n{with_p(src[fam], p)}\n" for fam, (v, lab, p) in rows)
        v, lab, p = rows[0][1]
        if v < best[0]:
            best = (v, lab, next(c[2] for c in cands if c[1] == lab))
            inc_code = f"# {rows[0][0]} (best literature rule)\n" + with_p(src[rows[0][0]], p)
    tunable = mode != "llm_zeroshot"
    for rd in range(ROUNDS[mode]):
        diag = diagnostics(best[2], Is) if mode in ("llm_feedback", "llm_skill") else None
        prompt = llm_prompt(oname, inc_code, best[0], hist, diag, tunable, knowledge=know_txt if rd == 0 else None)
        raw = call_llm(prompt); calls += 1   # uses --max_tokens
        dbg = dict(mode=mode, objective=oname, round=rd, library=lib_name, library_search_cost=lib_cost,
                   incumbent_cost=best[0], prompt=prompt, raw=raw, candidates=[])
        LLM_DEBUG.append(dbg)
        if not raw:
            global LLM_FAILS
            LLM_FAILS += 1
            if LLM_FAILS >= 5: sys.exit("ERROR: 5 LLM calls returned nothing; aborting so results are not silently replaced by library rules.")
            continue
        parsed = parse_rules(raw)
        if not parsed: dbg["parse_error"] = "no def priority found in response"
        for code in parsed:
            entry = dict(code=code); dbg["candidates"].append(entry)
            try: f, d = load_rule(code)
            except Exception as e:
                entry["load_error"] = f"{type(e).__name__}: {e}"[:200]; continue
            rule = bind(f, {})
            LAST_ERR[0] = None
            v = run_cost(rule, Is, oname); evals += 1
            entry["search_cost"] = v; entry["run_error"] = LAST_ERR[0]; entry["tunable"] = d
            if v is None: continue
            hist.append((code, v)); structs.append((code, f, d)); cands.append((v, code, rule))
            if v < best[0]: best = (v, code, rule); inc_code = code
    if tunable:
        tuned, ev = calibrate(structs, Is, oname, budget - evals, rng); evals += ev
        for lab, r in tuned: cands.append((run_cost(r, Is, oname), lab, r))
    cands = [c for c in cands if c[0] is not None]
    if cands:
        v, lab, r = min(cands, key=lambda c: c[0])
        if v < lib_cost: return lab, r, calls, evals
    return lib_name, LIB[lib_name], calls, evals

# ---- fail fast if LLM generators are requested but the API is not usable ----
if any(g.startswith("llm_") for g in a.gens.split(",")):
    import os
    _url, _env, _model = backend_conf()
    key = os.environ.get(_env, "")
    if not key or not key.isascii() or len(key) < 16:
        sys.exit(f"ERROR: {_env} is missing or not a real key (backend {a.backend}). export {_env}=<your key>")
    if not call_llm_raw("Reply with the single word OK."):
        sys.exit("ERROR: test call to the LLM API failed; fix connectivity/key/model name before running.")
    print(f"LLM API check passed ({a.backend}, model {_model}).", flush=True)


res = []; t0 = time.time(); KN = KNOW()
for oname in a.objectives.split(","):
    for scene in a.scenes.split(","):
        for s in range(a.seeds):
            Is = [make_instance(scene, 1000 + s * 10000 + i, a.n, a.rho) for i in range(a.n_search)]
            Ih = [make_instance(scene, 1000 + (s + 50) * 10000 + i, a.n, a.rho) for i in range(a.n_heldout)]
            Iv = [make_instance(scene, 1000 + (s + 30) * 10000 + i, a.n, a.rho) for i in range(a.n_val)]
            if Iv:  # Govern: fallback = library rule best on VALIDATION; paired empirical-Bernstein gate (Maurer-Pontil)
                vv = {k: per_inst(g, Iv, oname) for k, g in LIB.items()}
                fb = min(vv, key=lambda k: vv[k].mean())
            lib_s = {k: run_cost(f, Is, oname) for k, f in LIB.items()}
            lib_h = {k: run_cost(f, Ih, oname) for k, f in LIB.items()}
            best_name = min(lib_s, key=lib_s.get); best_h = min(lib_h.values())
            def pick(cands):  # cands: list of (label, fn); select on search, fall back to library
                sc = [(run_cost(f, Is, oname), lab, f) for lab, f in cands]
                sc = [x for x in sc if x[0] is not None]
                v, lab, f = min(sc, key=lambda x: x[0])
                return (lab, f) if v < lib_s[best_name] else (best_name, LIB[best_name])
            rng = np.random.default_rng(s * 7919 + 13)
            _kn = []
            def kn_scored():
                if not _kn: _kn.extend((run_cost(f, Is, oname), lab, f, fam, p) for lab, f, fam, p in KN)
                return _kn
            gens = {}; meta = {}
            for g in a.gens.split(","):
                if g == "library": gens[g] = (best_name, LIB[best_name])
                elif g == "pool_old": gens[g] = pick([(c, static_fn(c)) for c in sample(POOL_OLD, rng, 100)])
                elif g == "pool_full": gens[g] = pick([(c, static_fn(c)) for c in sample(POOL_FULL, rng, a.budget)]); meta[g] = dict(evals=a.budget)
                elif g == "knowledge": gens[g] = pick([(lab, f) for lab, f, _, _ in KN]); meta[g] = dict(evals=len(KN))
                elif g == "knowledge_tuned":
                    fam_best = {}
                    for v, lab, f, fam, p in kn_scored():
                        if v is not None and (fam not in fam_best or v < fam_best[fam][0]): fam_best[fam] = (v, p)
                    fsrc = {n: compile_rule(c, check=False)[0] for n, c, _ in KNOW_FAM}
                    tuned, ev = calibrate([(fam, fsrc[fam], p) for fam, (v, p) in fam_best.items()], Is, oname,
                                          a.budget - len(KN), np.random.default_rng(s + 202))
                    cc = [(v, lab, f) for v, lab, f, _, _ in kn_scored() if v is not None] + [(run_cost(r, Is, oname), lab, r) for lab, r in tuned]
                    cc = [c for c in cc if c[0] is not None]; v, lab, f = min(cc, key=lambda c: c[0])
                    gens[g] = (lab, f) if v < lib_s[best_name] else (best_name, LIB[best_name]); meta[g] = dict(evals=len(KN) + ev)
                elif g == "expert":   # H7 expert-written reference (only defined for the H7 objectives)
                    if oname not in EXPERT_H7: continue
                    st = [(nm, compile_rule(c, check=False)[0], p) for nm, c, p in EXPERT_H7[oname]]
                    cc = [(run_cost(bind(f, p), Is, oname), nm, bind(f, p)) for nm, f, p in st]
                    tuned, ev = calibrate(st, Is, oname, a.budget - len(st), np.random.default_rng(s + 404))
                    cc += [(run_cost(r, Is, oname), lab, r) for lab, r in tuned]
                    cc = [c for c in cc if c[0] is not None]; v, lab, f = min(cc, key=lambda c: c[0])
                    gens[g] = (lab, f) if v < lib_s[best_name] else (best_name, LIB[best_name]); meta[g] = dict(evals=len(st) + ev)
                elif g == "llm_eoh":
                    u0 = dict(USAGE)
                    lab, f, calls, ev = llm_eoh_generate(oname, Is, best_name, lib_s[best_name], np.random.default_rng(s + 303))
                    g_out = f"{g}@{a.tag}" if a.tag else g
                    gens[g_out] = (lab, f)
                    meta[g_out] = dict(llm_calls=calls, evals=ev, backend=a.backend, model=backend_conf()[2],
                                       tokens_in=USAGE["in"] - u0["in"], tokens_out=USAGE["out"] - u0["out"],
                                       llm_seconds=round(USAGE["sec"] - u0["sec"], 2))
                elif g.startswith("llm_"):
                    u0 = dict(USAGE)
                    lab, f, calls, ev = llm_generate(g, oname, Is, best_name, lib_s[best_name], kn_scored(), budget=a.budget,
                                                      rng=np.random.default_rng(s + 101))
                    g_out = f"{g}@{a.tag}" if a.tag else g
                    gens[g_out] = (lab, f)
                    meta[g_out] = dict(llm_calls=calls, evals=ev, backend=a.backend, model=backend_conf()[2],
                                       tokens_in=USAGE["in"] - u0["in"], tokens_out=USAGE["out"] - u0["out"],
                                       llm_seconds=round(USAGE["sec"] - u0["sec"], 2))
            for g, (lab, f) in gens.items():
                vh = run_cost(f, Ih, oname); herr = vh is None
                if herr: vh = lib_h[best_name]   # runtime guard
                gate = {}
                if Iv:
                    # Govern gate. Candidate deploys only if the one-sided upper confidence bound on
                    # mean(d), d_i = c_i - (1+tau) * fallback_i over validation instances, is <= 0.
                    #   eb : empirical Bernstein (Maurer & Pontil 2009) -- finite-sample guarantee
                    #   t  : one-sided paired t bound at the same delta -- asymptotic, practical
                    # Violation = deployed rule's held-out cost > (1+tau) x held-out cost of the rule the system would
                    # otherwise run (ungated: best-on-search library rule; gated: the validation fallback).
                    proposes = lab not in LIB; ub = {"eb": None, "t": None}
                    if proposes:
                        vc = per_inst(f, Iv, oname)
                        if vc is not None:
                            from scipy import stats as _st
                            n = len(Iv); d = vc - (1 + a.tau) * vv[fb]; dl = 0.05; K = 2 * len(LIB); lg = math.log(2 * K / dl)  # union bound: (candidate, L) pairs + library estimates (Prop. 1)
                            Rd = float(d.max() - d.min()); Vd = float(d.var(ddof=1)) if n > 1 else 0.0
                            ub["eb"] = d.mean() + math.sqrt(2 * Vd * lg / n) + 7 * Rd * lg / (3 * max(n - 1, 1))
                            ub["t"] = d.mean() + _st.t.ppf(1 - dl / K, n - 1) * math.sqrt(Vd / n)
                    gate["violation"] = bool(herr or vh > (1 + a.tau) * lib_h[best_name] + 1e-9)
                    for gk, u in ub.items():
                        ok = u is not None and u <= 0
                        gh = (lib_h[fb] if herr else vh) if ok else lib_h[fb]
                        gate.update({f"ub_{gk}_rel": (float(u / max(abs(vv[fb].mean()), 1e-9)) if u is not None else None),
                                     f"dep_{gk}": bool(ok), f"heldout_{gk}": gh, f"bb_{gk}": bool(gh < best_h - 1e-9),
                                     f"rel_{gk}": float((gh - best_h) / max(abs(best_h), 1e-9)),
                                     f"violation_{gk}": bool((ok and herr) or gh > (1 + a.tau) * lib_h[fb] + 1e-9)})
                res.append(dict(objective=oname, scene=scene, seed=s, generator=g, rule=lab, heldout=vh, heldout_error=herr, **meta.get(g, {}), **gate,
                                deployed_library_heldout=lib_h[best_name],
                                best_library_heldout=best_h, bb=bool(vh < best_h - 1e-9),
                                rel=float((vh - best_h) / max(abs(best_h), 1e-9))))
    print(f"[{time.time()-t0:6.0f}s] {oname}", flush=True)
json.dump(res, open(a.out, "w"), indent=1)
if LLM_DEBUG:
    json.dump(LLM_DEBUG, open(a.out.replace('.json', '') + '_llmdebug.json', 'w'), indent=1)
    nc = sum(len(d['candidates']) for d in LLM_DEBUG)
    print(f"LLM debug: {len(LLM_DEBUG)} responses, {nc} candidates, "
          f"{sum('load_error' in c for d in LLM_DEBUG for c in d['candidates'])} load errors, "
          f"{sum(bool(c.get('run_error')) for d in LLM_DEBUG for c in d['candidates'])} runtime errors, "
          f"{sum(c.get('search_cost') is not None and c['search_cost'] < d['library_search_cost'] for d in LLM_DEBUG for c in d['candidates'])} beat library on search")
agg = defaultdict(list)
for r in res: agg[(r["objective"], r["generator"])].append(r)
print(f"\n{'objective':<10}{'generator':<13}{'n':>4}{'BB held-out':>12}{'median rel %':>13}{'mean rel %':>11}{'vs deployed lib %':>18}  most chosen rule")
for (o, g), L in sorted(agg.items()):
    rel = np.array([x["rel"] for x in L]); top = max(set(x["rule"] for x in L), key=[x["rule"] for x in L].count)
    top = "LLM rule" if "def priority" in top else top
    vd = np.mean([(x["heldout"] - x["deployed_library_heldout"]) / max(abs(x["deployed_library_heldout"]), 1e-9) for x in L])
    print(f"{o:<10}{g:<13}{len(L):>4}{100*np.mean([x['bb'] for x in L]):>11.1f}%{100*np.median(rel):>12.1f}{100*rel.mean():>11.1f}{100*vd:>18.1f}  {top[:40]}")
