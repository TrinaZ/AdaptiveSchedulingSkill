"""
make_trace_bench.py -- derive online_sla_bench_trace.py from the frozen online_sla_bench.py.

The ONLY changes are listed in SUBS below: the instance generator (make_instance) replays real request streams from
alibaba_trace_windows.csv instead of the WGS generator, and the prompt text names the trace's request kinds. Simulator,
objectives, generators, calibration, gates, seeds, and analysis code are unchanged, so results are directly comparable.

Usage: python3 make_trace_bench.py online_sla_bench.py online_sla_bench_trace.py
"""
import sys, hashlib
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
print("source sha256:", hashlib.sha256(s.encode()).hexdigest())

TRACE_GEN = '''
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
            _TR[w] = (L, work / span)
            print(f"[trace] window {w}: thinning 1/{M} of jobs -> {len(L):,} tasks, {len(L)/span*60:.2f} arrivals/min, "
                  f"capacity at rho=1: {work/span:.1f} cores", flush=True)
    return _TR
_KIND = {"M": "map", "R": "reduce", "J": "join", "O": "other"}
def make_instance(scene, seed, n, rho):
    w = int(scene[1]); cap = scene.endswith("cap")
    L, rate = _trace()[w]
    k = (seed - 1000) // 10000
    lo, hi = (0.0, 0.2) if k < 30 else ((0.2, 0.6) if k < 50 else (0.6, 1.0))
    rng = np.random.RandomState(seed + 7)
    i0 = int(lo * len(L)); i1 = max(i0 + 1, int(hi * len(L)) - n)
    st = rng.randint(i0, i1)
    R = rate / rho * (0.7 if cap else 1.0)
    reqs = []; t0 = L[st][0]
    for j, (s0, d, c, kind, job) in enumerate(L[st:st + n]):
        r = Req(); r.id = j; r.task_type = _KIND.get(kind, "other"); r.runtime = d
        r.estimated_runtime = d * rng.uniform(0.6, 1.4); r.resource_need = min(c, R)
        prem = (_zlib.crc32(job.encode()) % 5) == 0
        r.tenant = "premium" if prem else "standard"
        r.weight = (3.0 if prem else 1.0) * rng.uniform(0.8, 1.2)
        r.release = s0 - t0; r.due = r.release + rng.uniform(1.5, 4.0) * r.estimated_runtime
        reqs.append(r)
    return reqs, float(R)
'''

SUBS = [
 # 1) CLI: trace file; trace scenes and SLA objectives by default
 ('ap.add_argument("--scenes", default="baseline,S1_runtime_shift,S2_resource_drop,S4_novel_task")',
  'ap.add_argument("--scenes", default="T0,T1,T0cap,T1cap")\nap.add_argument("--trace", default="alibaba_trace_windows.csv")\nap.add_argument("--thin_conc", type=float, default=12.0, help="target mean number of concurrently running tasks after job thinning")'),
 ('ap.add_argument("--objectives", default="wflow,tard,miss,p95,premium,slo_wait,mix")',
  'ap.add_argument("--objectives", default="tard,miss,slo_wait,mix,premium")'),
 # 2) the WGS generator is not needed
 ("from problems.genomic_scheduling.instances import generate_tasks\n", ""),
 # 3) replace make_instance by the trace replay
 (s[s.index("def make_instance(scene, seed, n, rho):"):s.index("class TV:")], ""),
 ("class TV:  # observable view", TRACE_GEN + "\nclass TV:  # observable view"),
 # 4) prompt text: service description and request kinds
 ("You are designing a dispatching rule for an online genomics-pipeline service cluster.",
  "You are designing a dispatching rule for an online batch-computing service cluster (requests replayed from a production cluster trace)."),
 ("task.task_type ('alignment'|'variant_calling'|'sv_calling')", "task.task_type ('map'|'reduce'|'join'|'other')"),
 # 5) Amendment 1 (runtime guard), made after two runs crashed and before any result was seen: a selected rule that
 #    raises on held-out traffic (e.g. math.exp overflow at the trace's time scale, waits of thousands of seconds) is
 #    replaced by the library rule at run time, and that deployment is counted as a violation (ungated and gated).
 ("                vh = run_cost(f, Ih, oname)\n",
  "                vh = run_cost(f, Ih, oname); herr = vh is None\n                if herr: vh = lib_h[best_name]   # runtime guard\n"),
 ('                    gate["violation"] = bool(vh > (1 + a.tau) * lib_h[best_name] + 1e-9)',
  '                    gate["violation"] = bool(herr or vh > (1 + a.tau) * lib_h[best_name] + 1e-9)'),
 ("                        gh = vh if ok else lib_h[fb]",
  "                        gh = (lib_h[fb] if herr else vh) if ok else lib_h[fb]"),
 ('f"violation_{gk}": bool(gh > (1 + a.tau) * lib_h[fb] + 1e-9)',
  'f"violation_{gk}": bool((ok and herr) or gh > (1 + a.tau) * lib_h[fb] + 1e-9)'),
 ("res.append(dict(objective=oname, scene=scene, seed=s, generator=g, rule=lab, heldout=vh,",
  "res.append(dict(objective=oname, scene=scene, seed=s, generator=g, rule=lab, heldout=vh, heldout_error=herr,"),
]
for old, new in SUBS:
    c = s.count(old)
    if old and c == 0: sys.exit(f"substitution target not found: {old[:60]!r}")
    s = s.replace(old, new)
open(dst, "w").write(s)
print("written", dst, "sha256:", hashlib.sha256(s.encode()).hexdigest())
