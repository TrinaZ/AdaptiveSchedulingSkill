"""Random template search (100 candidates, no LLM) with and without access to the
hidden true runtime. Select on search instances (seed s), evaluate on held-out
instances (seed s+50). Same scenes/seeds/incumbents as head_to_head.py."""
import sys, json, re, argparse
from collections import defaultdict
import numpy as np
ap=argparse.ArgumentParser(); ap.add_argument("--src"); ap.add_argument("--out"); a=ap.parse_args()
sys.path.insert(0,a.src)
from problems.genomic_scheduling.prob import GenomicScheduling, BASELINES
from laosa import LLMAlgorithmAdapter
import head_to_head as H
from run_poms_experiments import SCENE_CHANGES
ad=LLMAlgorithmAdapter.__new__(LLMAlgorithmAdapter)
def pool(leaky):
    T=H.TEMPLATES if leaky else [(re.sub(r"task\.runtime\b","task.estimated_runtime",t),g) for t,g in H.TEMPLATES]
    return T
def sample(T,rng,n):
    out=[]
    for _ in range(n):
        t,g=T[rng.integers(len(T))]
        p={k:float(np.exp(rng.uniform(np.log(max(min(v),1e-3)),np.log(max(max(v),1e-3))))) for k,v in g.items()}
        out.append(t.format(**p))
    return out
res=[]
for sc in SCENE_CHANGES:
    for seed in range(8):
        ps=GenomicScheduling(sc["after"],n_instances=10,n_tasks=20,seed=seed)
        ph=GenomicScheduling(sc["after"],n_instances=200,n_tasks=20,seed=seed+50)
        blh={k:ad._evaluate_code(c,ph) for k,c in BASELINES.items()}
        bls={k:ad._evaluate_code(c,ps) for k,c in BASELINES.items()}
        besth=min(blh.values()); bestname_s=min(bls,key=bls.get)
        for leaky in (True,False):
            rng=np.random.default_rng(seed+7919)
            cands=sample(pool(leaky),rng,100)
            sc_s=[(ad._evaluate_code(c,ps),c) for c in cands]
            sc_s=[x for x in sc_s if x[0] is not None]
            sel_ms,sel=min(sc_s)
            # selection includes keeping the best static if nothing better (as in h2h)
            if sel_ms>=bls[bestname_s]: sel=BASELINES[bestname_s]
            mh=ad._evaluate_code(sel,ph)
            res.append(dict(scene=sc["id"],tier=sc["change_type"],seed=seed,leaky=leaky,code=sel,bb=bool(mh<besth),rel=float(mh/besth-1)))
json.dump(res,open(a.out,"w"),indent=1)
agg=defaultdict(list)
for r in res: agg[(r["tier"],r["leaky"])].append(r)
print(f"{'tier':<16}{'pool':<22}{'n':>4}{'BB_heldout':>12}{'median rel%':>13}")
for (t,l),L in sorted(agg.items()):
    print(f"{t:<16}{('true runtime (leaky)' if l else 'estimated only'):<22}{len(L):>4}{100*np.mean([x['bb'] for x in L]):>11.1f}%{100*np.median([x['rel'] for x in L]):>12.2f}")
