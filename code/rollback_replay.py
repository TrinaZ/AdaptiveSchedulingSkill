"""
rollback_replay.py -- post-deployment monitoring and rollback, replayed on the stored prototype episodes (no LLM).

For every prototype episode in which Govern deployed a candidate (results/prototype.json), the post-change stream is
rebuilt from its seed (same windows as online_prototype.py) and the deployed rule (source + params from its Skill
Package) is run on the windows after the alarm in the same isolated process. The validated fallback A-dagger
(evidence.fallback in the package) is shadow-evaluated on the same windows, as a service would replay recorded traffic.
Post-deployment check (a Monitor on the deployed rule): after each window w >= m_min, a one-sided paired t test of
  d_i = cost_deployed - (1 + tau) * cost_fallback   over the windows seen so far
rolls back to the fallback when the test rejects "mean d <= 0" at level alpha. Reported per episode: whether and
when it rolled back, and the post-deployment cost with and without rollback, relative to the fallback. Rolling back
to the package's `rollback_to` (the rule replaced at the change) is reported as well.
Usage: python3 rollback_replay.py --bench online_sla_bench.py --src . --proto ../results/prototype.json --out ../results/rollback_replay.json
"""
import sys, json, math, argparse
import numpy as np
from scipy import stats as st
ap = argparse.ArgumentParser()
ap.add_argument("--bench", required=True); ap.add_argument("--src", required=True)
ap.add_argument("--proto", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--m_min", type=int, default=5); ap.add_argument("--alpha", type=float, default=0.05)
a = ap.parse_args()
code = open(a.bench).read().split("\n# ---- fail fast")[0]
sys.argv = ["bench", "--src", a.src, "--out", "/dev/null"]
B = {}; exec(code, B)
make_instance, simulate, objective, per_inst, LIB = (B[k] for k in ["make_instance", "simulate", "objective", "per_inst", "LIB"])
from govern_sandbox import isolated_costs
P = json.load(open(a.proto)); cfg = P["config"]; t_pre, t_post, tau = cfg["t_pre"], cfg["t_post"], cfg.get("tau", 0.02)
out = []
for e in P["episodes"]:
    if e.get("decision") != "deploy": continue
    k = e["episode"]; base = 7_000_000 + 100_000 * k; (sc0, o0), (sc1, o1) = e["pre"], e["post"]
    stream = [make_instance(sc0 if t < t_pre else sc1, base + t, 60, 1.0) for t in range(t_pre + t_post)]
    post = stream[e["alarm"] + 1:]
    pkg = e["skill_package"]; fb = pkg["evidence"]["fallback"]; old = pkg["rollback_to"]
    dc, why = isolated_costs(pkg["rule_source"], pkg["params"], post, objective, simulate, o1)
    if dc is None: print("episode", k, "failed:", why); continue
    dc = np.array(dc); fc = per_inst(LIB[fb], post, o1); oc = per_inst(LIB[old], post, o1)
    best_lib = min(float(per_inst(f, post, o1).mean()) for f in LIB.values())
    d = dc - (1 + tau) * fc; rb = None
    for w in range(a.m_min, len(d) + 1):
        x = d[:w]; s = x.std(ddof=1)
        if s > 0 and st.t.sf(x.mean() / (s / math.sqrt(w)), w - 1) < a.alpha: rb = w; break
    with_rb = dc.copy()
    if rb is not None: with_rb[rb:] = fc[rb:]
    r = dict(episode=k, type=e["true_type"], objective=o1, violation=bool(e["post_violation"]), windows=len(post),
             check_post_vs_bestlib=float(dc.mean() / best_lib - 1), stored_post_vs_bestlib=e["post_vs_bestlib"],
             rollback_window=rb, fallback=fb, replaced=old,
             deployed_vs_fallback=float(dc.mean() / fc.mean() - 1), with_rollback_vs_fallback=float(with_rb.mean() / fc.mean() - 1),
             replaced_vs_fallback=float(oc.mean() / fc.mean() - 1))
    out.append(r)
    print(f"ep {k:3d} {e['true_type'][:6]} {o1:8s} viol={r['violation']!s:5} rollback@{rb!s:4} deployed {100*r['deployed_vs_fallback']:+6.1f}% "
          f"with rollback {100*r['with_rollback_vs_fallback']:+6.1f}% replaced rule {100*r['replaced_vs_fallback']:+7.1f}% "
          f"(check {100*r['check_post_vs_bestlib']:+.1f} vs stored {100*r['stored_post_vs_bestlib']:+.1f})", flush=True)
json.dump(dict(config=vars(a), episodes=out), open(a.out, "w"), indent=1)
V = [r for r in out if r["violation"]]; N = [r for r in out if not r["violation"]]
print(f"\n{len(out)} deployments; violating: {len(V)}, rolled back {sum(r['rollback_window'] is not None for r in V)}; "
      f"non-violating rolled back (false rollbacks): {sum(r['rollback_window'] is not None for r in N)}")
