"""
make_llm_bench.py -- derive online_sla_bench_llm.py (H10: LLM-inference-serving replay) from online_sla_bench_trace.py
(the H8 trace benchmark). Only the substitutions in SUBS change; simulator, objectives, generators, calibration, gates,
seeds and analysis code are unchanged.

Workload: Azure LLM inference trace 2023 (two production services: coding assistant and conversation), extracted by
extract_azure_llm_trace.py. A request is one LLM inference call; it occupies KV-cache memory ((context+generated)/1000
thousand tokens) on a shared serving pool for its service time. Real: arrivals, input and output token counts.
Synthesized: the linear latency model (see the extractor), due times, weights. The runtime estimate keeps the prefill
part exact (input length is known at arrival) and perturbs the decode part by U[0.6,1.4] (output length is not).
Tenants: requests of the coding assistant are premium (latency-sensitive, weight x3); conversation requests standard.
Scenes: L<w> = window w at its average offered load (rho); L<w>cap = same window with 30% less capacity.
Usage: python3 make_llm_bench.py online_sla_bench_trace.py online_sla_bench_llm.py
"""
import sys, hashlib
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
print("source sha256:", hashlib.sha256(s.encode()).hexdigest())
SUBS = [
 ('ap.add_argument("--scenes", default="T0,T1,T0cap,T1cap")', 'ap.add_argument("--scenes", default="L0,L1,L0cap,L1cap")'),
 ('ap.add_argument("--trace", default="alibaba_trace_windows.csv")', 'ap.add_argument("--trace", default="azure_llm_windows.csv")'),
 ('_TR = {}\n', '_TR = {}\n_PRE = {}   # job -> prefill seconds (exactly predictable at arrival)\n'),
 ('cols.setdefault(w, []).append((float(r["start"]), float(r["dur"]), float(r["cpu"]), r["kind"], r["job"]))',
  'cols.setdefault(w, []).append((float(r["start"]), float(r["dur"]), float(r["cpu"]), r["kind"], r["job"])); _PRE[r["job"]] = float(r["mem"])'),
 ('_KIND = {"M": "map", "R": "reduce", "J": "join", "O": "other"}', '_KIND = {"C": "code", "V": "conv"}'),
 ('r.estimated_runtime = d * rng.uniform(0.6, 1.4)', 'r.estimated_runtime = _PRE[job] + (d - _PRE[job]) * rng.uniform(0.6, 1.4)'),
 ('prem = (_zlib.crc32(job.encode()) % 5) == 0', 'prem = (kind == "C")   # coding-assistant requests are the premium tenant'),
 ("task.resource_need (cores, float)", "task.resource_need (KV-cache memory in thousand tokens, float)"),
 ("task.task_type ('map'|'reduce'|'join'|'other')", "task.task_type ('code'|'conv': coding-assistant or conversation service)"),
 ("state.free (free cores), state.capacity (total cores)", "state.free (free KV-cache memory), state.capacity (total KV-cache memory)"),
 ("state.queue_work (sum of est*cores waiting)", "state.queue_work (sum of est*memory waiting)"),
 ("online batch-computing service cluster (requests replayed from a production cluster trace)",
  "LLM inference serving pool shared by a coding-assistant service (premium tenant) and a conversation service (requests replayed from a production LLM inference trace; a request's runtime is its prefill plus decode time, and its output length is unknown until it finishes)"),
 ("whenever cores free up", "whenever KV-cache memory frees up"),
]
for old, new in SUBS:
    n = s.count(old)
    assert n >= 1, ("missing", old[:70])
    s = s.replace(old, new)
open(dst, "w").write(s)
print("written", dst, "sha256:", hashlib.sha256(s.encode()).hexdigest())
