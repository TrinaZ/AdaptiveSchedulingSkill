"""
make_mt_bench.py -- derive online_sla_bench_mt.py (multi-tenant replay with reactive autoscaling) from
online_sla_bench_trace.py (the amended H8 trace benchmark). Only the substitutions in SUBS change; simulator,
objectives, generators, calibration, gates, seeds and analysis code are unchanged.

Multi-tenant replay: the Alibaba stream is thinned by whole jobs to FOUR services' worth of load (mean concurrency
about 48 instead of 12) and every job belongs to one of 4 tenant services by an independent hash of its name; tenant 0
is premium (weight x3), the others standard. All tenants share one capacity pool, so they interfere.
Scenes: T<w>  = window w, fixed capacity at its average offered load (rho = 1);
        T<w>as = window w with REACTIVE AUTOSCALING: each instance's capacity is set from the offered load of the
                 n tasks that arrived just before it (target utilisation 1/rho), clipped to [0.5, 2] x the window
                 average, so capacity lags bursts as a real autoscaler does.
Usage: python3 make_mt_bench.py online_sla_bench_trace.py online_sla_bench_mt.py
"""
import sys, hashlib
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
print("source sha256:", hashlib.sha256(s.encode()).hexdigest())
SUBS = [
 ('ap.add_argument("--scenes", default="T0,T1,T0cap,T1cap")',
  'ap.add_argument("--scenes", default="T0,T1,T0as,T1as")\nap.add_argument("--tenants", type=int, default=4, help="tenant services sharing the pool; tenant 0 is premium")'),
 ('ap.add_argument("--thin_conc", type=float, default=12.0,', 'ap.add_argument("--thin_conc", type=float, default=48.0,'),
 ('            _TR[w] = (L, work / span)', '            _TR[w] = (L, work / span, M)'),
 ('''    w = int(scene[1]); cap = scene.endswith("cap")
    L, rate = _trace()[w]''', '''    w = int(scene[1]); auto = scene.endswith("as")
    L, rate, M = _trace()[w]'''),
 ('''    R = rate / rho * (0.7 if cap else 1.0)''', '''    R = rate / rho
    if auto:   # reactive autoscaling: capacity from the offered load of the n tasks that arrived just before this instance
        prev = L[max(0, st - n):st]
        if len(prev) >= 2:
            sp = max(prev[-1][0] - prev[0][0], 1.0)
            R = min(max(sum(d * c for _, d, c, _, _ in prev) / sp / rho, 0.5 * R), 2.0 * R)'''),
 ('''        prem = (_zlib.crc32(job.encode()) % 5) == 0''', '''        prem = ((_zlib.crc32(job.encode()) // M) % a.tenants) == 0   # tenant service 0 is premium'''),
 ('online batch-computing service cluster (requests replayed from a production cluster trace)',
  'shared multi-tenant service cluster (requests of four tenant services replayed from a production cluster trace; capacity may be autoscaled)'),
]
for old, new in SUBS:
    n = s.count(old)
    assert n >= 1, ("missing", old[:70])
    s = s.replace(old, new)
open(dst, "w").write(s)
print("written", dst, "sha256:", hashlib.sha256(s.encode()).hexdigest())
