"""
orchestrator.py -- the deterministic MAPE-K workflow that sequences the four skills (no LLM in control flow).

    python3 orchestrator.py --scenario scenario.json --store knowledge_store/packages

scenario.json example:
  {"pre": {"scene": "baseline", "objective": "tard"}, "post": {"scene": "S2_resource_drop", "objective": "tard"},
   "seed": 1, "windows_pre": 40, "windows_post": 20, "n_val": 1000}
Each skill is called through its function in core.py (identical to <skill>/run.py); every record is also written to
<store>/../records/ so an agent or an auditor can replay the episode skill by skill.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core

ap = argparse.ArgumentParser(); ap.add_argument("--scenario", required=True); ap.add_argument("--store", required=True)
a = ap.parse_args()
sc = json.load(open(a.scenario)); B = core.bench()
rec_dir = os.path.join(os.path.dirname(os.path.abspath(a.store)), "records"); os.makedirs(rec_dir, exist_ok=True)
def save(name, obj):
    json.dump(obj, open(os.path.join(rec_dir, name), "w"), indent=1, default=float); return obj

seed = 7_000_000 + 1000 * sc["seed"]; Wp, Wq = sc.get("windows_pre", 40), sc.get("windows_post", 20)
pre, post = sc["pre"], sc["post"]
cfg = core.traffic(dict(scene=pre["scene"], seed=seed + 900, n=30))
incumbent = min(B["LIB"], key=lambda n: B["run_cost"](B["LIB"][n], cfg, pre["objective"]))
windows = [(pre if t < Wp else post) for t in range(Wp + Wq)]
costs = [B["run_cost"](B["LIB"][incumbent], core.traffic(dict(scene=w["scene"], seed=seed + t, n=1)), pre["objective"])
         for t, w in enumerate(windows)]
declared = Wp if post["objective"] != pre["objective"] else None
cr = save("1_change_report.json", core.monitor(dict(costs=costs, reference_windows=20, declared_change_at=declared)))
print("Monitor :", cr["alarm"], "(declared)" if cr["declared"] else "")
if cr["alarm"] is None:
    print("no change detected; incumbent", incumbent, "kept"); sys.exit(0)
t = cr["alarm"]
dg = save("2_diagnosis.json", core.diagnose(dict(
    change_report=cr, objective=post["objective"],
    reference_traffic=dict(scene=pre["scene"], seed=seed, n=10),
    recent_traffic=dict(scene=windows[t]["scene"], seed=seed + max(Wp, t - 9), n=10),
    search_traffic=dict(scene=windows[t]["scene"], seed=seed + 20000, n=12))))
print("Diagnose:", dg["change_type"], "headroom %.3f" % dg["headroom"], "adequate" if dg["adequate"] else "-> Generate")
cs = save("3_candidates.json", core.generate(dict(diagnosis=dg, seed=sc["seed"])))
print("Generate:", len(cs["candidates"]), "candidate(s),", cs["llm_calls"], "LLM call(s)")
pkg = core.govern(dict(candidate_set=cs, objective=post["objective"], change_type=dg["change_type"], incumbent=incumbent,
                       validation_traffic=dict(scene=windows[t]["scene"], seed=seed + 40000, n=sc.get("n_val", 1000))))
path = core.write_package(save("4_skill_package.json", pkg), a.store)
print("Govern  :", pkg["decision"], "| rule:", pkg["rule_name"][:50], "| review flag:", pkg["review_flag"], "->", path)
