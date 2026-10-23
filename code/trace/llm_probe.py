import sys, numpy as np, time
src = open("online_sla_bench_llm.py").read().split("# ---- fail fast")[0]
conc = sys.argv[1] if len(sys.argv) > 1 else "12"
sys.argv = ["x", "--src", ".", "--out", "/dev/null", "--thin_conc", conc]
ns = {"__name__": "bench"}; exec(src, ns)
mk, run_cost, LIB, KNOW = ns["make_instance"], ns["run_cost"], ns["LIB"], ns["KNOW"]
KN = KNOW(); t0 = time.time()
reqs, R = mk("L0", 1000 + 50 * 10000, 60, 1.0)
print("example instance: span %.0fs, R=%.1f, mean dur %.0f, max cpu %.2f" % (reqs[-1].release, R, np.mean([r.runtime for r in reqs]), max(r.resource_need for r in reqs)))
for o in ["tard", "miss", "slo_wait", "mix", "premium"]:
    rows = []
    for sc in ["L0", "L1", "L0cap", "L1cap"]:
        for s in range(3):
            Is = [mk(sc, 1000 + s*10000 + i, 60, 1.0) for i in range(12)]
            Ih = [mk(sc, 1000 + (s+50)*10000 + i, 60, 1.0) for i in range(40)]
            lh = {k: run_cost(f, Ih, o) for k, f in LIB.items()}; best = min(lh.values()); worst = max(lh.values())
            ls = {k: run_cost(f, Is, o) for k, f in LIB.items()}; b = min(ls, key=ls.get)
            kv, kl, kf = min(((run_cost(f, Is, o), lab, f) for lab, f, _, _ in KN), key=lambda x: x[0])
            rows.append(((lh[b]-best)/max(best,1e-9), (run_cost(kf, Ih, o)-best)/max(best,1e-9), (worst-best)/max(best,1e-9), best))
    r = np.array(rows)
    print(f"{o:<9} lib(search-picked) {100*r[:,0].mean():+6.1f}%  knowledge(search-picked) {100*r[:,1].mean():+6.1f}%  lib spread worst/best {100*np.median(r[:,2]):6.0f}%  zero-cost cells {int((r[:,3]==0).sum())}/12  ({time.time()-t0:.0f}s)", flush=True)
