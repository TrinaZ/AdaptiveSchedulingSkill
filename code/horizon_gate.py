"""
horizon_gate.py -- does Govern catch the long-window degradation when validation matches the deployment horizon? (no LLM)

For every cell of the main run and generator in --gens whose selected rule is not a library rule:
  mismatched : the paired t gate as run (60-request validation instances, n_val60) decides deploy/fallback;
  matched    : the same gate on validation instances with N requests (n_valN instances, seed s+30).
Both decisions are then scored on held-out instances with N requests (seed s+50). A deployment violates the bound
if its held-out cost exceeds (1+tau) x the held-out cost of the matched fallback (library rule best on N-request validation).

Usage: python3 horizon_gate.py --bench online_sla_bench.py --src <repo> --files results/online_full_mix.json ... \
          --n 300 --n_valN 200 --n_heldout 60 --out results/horizon_gate.json
"""
import sys, json, re, math, argparse, time
import numpy as np
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--bench", required=True); ap.add_argument("--src", required=True); ap.add_argument("--files", nargs="+", required=True)
ap.add_argument("--n", type=int, default=300); ap.add_argument("--n_valN", type=int, default=200); ap.add_argument("--n_val60", type=int, default=1000)
ap.add_argument("--n_heldout", type=int, default=60); ap.add_argument("--tau", type=float, default=0.02)
ap.add_argument("--gens", default="knowledge_tuned,llm_skill"); ap.add_argument("--out", required=True)
a = ap.parse_args()
code = open(a.bench).read().split("\n# ---- fail fast")[0]
sys.argv = ["bench", "--src", a.src, "--out", "/dev/null"]
B = {}; exec(code, B)
make_instance, per_inst, LIB, KNOW, KNOW_FAM, compile_rule, bind = (B[k] for k in
    ["make_instance", "per_inst", "LIB", "KNOW", "KNOW_FAM", "compile_rule", "bind"])
KNL = {lab: f for lab, f, _, _ in KNOW()}; FAM = {n: c for n, c, _ in KNOW_FAM}

def rebuild(label):
    if label in LIB: return LIB[label]
    if label in KNL: return KNL[label]
    body, params = label, {}
    m = re.search(r"\s*# tuned \((.*)\)\s*$", label)
    if m:
        body = label[:m.start()]
        params = {k: float(v) for k, v in (kv.split("=") for kv in m.group(1).split(","))}
    f, _ = compile_rule(FAM.get(body.strip(), body), check=False)
    return bind(f, params)

def tgate(vc, vfb):
    d = vc - (1 + a.tau) * vfb; n = len(d); K = 2 * len(LIB)
    return d.mean() + stats.t.ppf(1 - 0.05 / K, n - 1) * math.sqrt(float(d.var(ddof=1)) / n) <= 0

recs = [r for f in a.files if not f.endswith("_llmdebug.json") for r in json.load(open(f)) if "scene" in r]
gens = a.gens.split(","); by = {(r["objective"], r["scene"], r["seed"], r["generator"]): r for r in recs}
cells = sorted({(r["objective"], r["scene"], r["seed"]) for r in recs})
out = []; t0 = time.time()
for o, sc, s in cells:
    todo = [(g, by[(o, sc, s, g)]["rule"]) for g in gens if (o, sc, s, g) in by and by[(o, sc, s, g)]["rule"] not in LIB]
    if not todo: continue
    V60 = [make_instance(sc, 1000 + (s + 30) * 10000 + i, 60, 1.0) for i in range(a.n_val60)]
    VN = [make_instance(sc, 1000 + (s + 30) * 10000 + i, a.n, 1.0) for i in range(a.n_valN)]
    HN = [make_instance(sc, 1000 + (s + 50) * 10000 + i, a.n, 1.0) for i in range(a.n_heldout)]
    v60 = {k: per_inst(f, V60, o) for k, f in LIB.items()}; fb60 = min(v60, key=lambda k: v60[k].mean())
    vN = {k: per_inst(f, VN, o) for k, f in LIB.items()}; fbN = min(vN, key=lambda k: vN[k].mean())
    hN = {k: per_inst(f, HN, o).mean() for k, f in LIB.items()}; ref = hN[fbN]
    for g, lab in todo:
        rule = rebuild(lab)
        c60 = per_inst(rule, V60, o); cN = per_inst(rule, VN, o); ch = per_inst(rule, HN, o)
        if c60 is None or cN is None or ch is None: continue
        ch = ch.mean()
        dep_mis = bool(tgate(c60, v60[fb60])); dep_mat = bool(tgate(cN, vN[fbN]))
        cost_mis = ch if dep_mis else hN[fb60]; cost_mat = ch if dep_mat else ref
        out.append(dict(objective=o, scene=sc, seed=s, generator=g, dep_mismatched=dep_mis, dep_matched=dep_mat,
                        viol_mismatched=bool(dep_mis and ch > (1 + a.tau) * ref + 1e-9),
                        viol_matched=bool(dep_mat and ch > (1 + a.tau) * ref + 1e-9),
                        rel_mismatched=(cost_mis - ref) / max(abs(ref), 1e-9), rel_matched=(cost_mat - ref) / max(abs(ref), 1e-9),
                        rel_candidate=(ch - ref) / max(abs(ref), 1e-9)))
    print(f"[{time.time()-t0:6.0f}s] {o} {sc} {s}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
print(f"\nN={a.n}: gate on 60-request validation (as run) vs on {a.n}-request validation (matched); held-out {a.n_heldout} x {a.n} requests")
print(f"{'objective':<10}{'generator':<17}{'cand':>5}{'dep mis%':>9}{'viol mis%':>10}{'dep mat%':>9}{'viol mat%':>10}{'rel mis%':>9}{'rel mat%':>9}{'cand rel%':>10}")
for o in sorted({r['objective'] for r in out}) + ["ALL"]:
    for g in gens:
        L = [r for r in out if (o == "ALL" or r["objective"] == o) and r["generator"] == g]
        if not L: continue
        m = lambda k: 100 * np.mean([r[k] for r in L])
        print(f"{o:<10}{g:<17}{len(L):>5}{m('dep_mismatched'):>9.0f}{m('viol_mismatched'):>10.1f}{m('dep_matched'):>9.0f}{m('viol_matched'):>10.1f}"
              f"{m('rel_mismatched'):>9.1f}{m('rel_matched'):>9.1f}{m('rel_candidate'):>10.1f}")
