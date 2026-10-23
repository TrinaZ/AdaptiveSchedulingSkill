"""
extract_alibaba_trace.py -- cut a small, real request stream out of Alibaba cluster-trace-v2018 batch_task.csv.

batch_task.csv has no header; columns (cluster-trace-v2018 schema):
  task_name, instance_num, job_name, task_type, status, start_time, end_time, plan_cpu, plan_mem
One pass, streaming (no pandas needed). Keeps tasks with status 'Terminated', end > start, plan_cpu > 0, and writes:
  1) a per-6h histogram of usable tasks (printed), to see how busy each period is;
  2) alibaba_trace_windows.csv with ALL usable tasks whose start_time falls in the chosen windows
     (default: the two busiest 12-hour windows that are >= 2 days apart).
Output columns: window, start, dur, cpu, mem, kind, job
  kind = first letter of task_name (M = map-like, R = reduce-like, J = join-like, else 'O'); job = job_name.

Usage:  python3 extract_alibaba_trace.py /mnt/e/ForPhD/Code/创新点4/_data/alibaba_v2018/batch_task.csv
        (optional) --windows 172800:216000,518400:561600   to choose windows by start_time in seconds
Prints timing; the output file is usually a few MB -- send that file and the printed summary.
"""
import sys, csv, time, argparse, collections

ap = argparse.ArgumentParser()
ap.add_argument("path"); ap.add_argument("--out", default="alibaba_trace_windows.csv")
ap.add_argument("--windows", default="", help="comma list start:end in seconds; default = auto")
ap.add_argument("--hours", type=float, default=12.0)
a = ap.parse_args()
t0 = time.time()

def rows(path):
    with open(path, newline="") as f:
        for r in csv.reader(f):
            if len(r) < 9: continue
            try:
                if r[4] != "Terminated": continue
                s, e, cpu = float(r[5]), float(r[6]), float(r[7] or 0)
                if e <= s or cpu <= 0 or s <= 0: continue
                yield r[0], r[2], s, e - s, cpu / 100.0, float(r[8] or 0)
            except ValueError:
                continue

# pass 1: histogram (6-hour bins) -- also shows the raw first lines for a schema check
with open(a.path) as f:
    print("first raw lines (schema check):")
    for _ in range(3): print("   ", f.readline().strip()[:160])
hist = collections.Counter(); n = 0
for name, job, s, d, cpu, mem in rows(a.path):
    hist[int(s // 21600)] += 1; n += 1
    if n % 2_000_000 == 0: print(f"   ... {n:,} usable tasks read ({time.time()-t0:.0f}s)", flush=True)
print(f"usable tasks: {n:,}  ({time.time()-t0:.0f}s)")
print("tasks per 6-hour bin (bin = start_time // 21600):")
for b in sorted(hist): print(f"   bin {b:3d}  day {b/4:4.1f}  {hist[b]:>9,}")

if a.windows:
    W = [tuple(float(x) for x in w.split(":")) for w in a.windows.split(",")]
else:   # two busiest windows of --hours, >= 2 days apart
    L = int(a.hours * 3600 // 21600)
    score = {b: sum(hist.get(b + k, 0) for k in range(L)) for b in hist}
    first = max(score, key=score.get)
    second = max((b for b in score if abs(b - first) >= 8), key=score.get)
    W = sorted([(b * 21600.0, (b + L) * 21600.0) for b in (first, second)])
print("windows (seconds):", W)

# pass 2: write the windows
k = collections.Counter()
with open(a.out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["window", "start", "dur", "cpu", "mem", "kind", "job"])
    for name, job, s, d, cpu, mem in rows(a.path):
        for i, (lo, hi) in enumerate(W):
            if lo <= s < hi:
                kind = name[:1] if name[:1] in ("M", "R", "J") else "O"
                w.writerow([i, int(s), round(d, 1), round(cpu, 2), round(mem, 4), kind, job]); k[i] += 1
print("written:", dict(k), "->", a.out, f"({time.time()-t0:.0f}s total)")
