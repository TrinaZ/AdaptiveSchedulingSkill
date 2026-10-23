#!/bin/sh
cd "$(dirname "$0")"
for o in "$@"; do
  [ -s ../../results/trace_mt/mt_nollm_$o.json ] && continue
  python3 online_sla_bench_mt.py --src .. --rho 1.0 --seeds 10 --n 200 --n_search 12 --n_heldout 60 --objectives $o \
     --gens library,pool_full,knowledge,knowledge_tuned --out ../../results/trace_mt/mt_nollm_$o.json > mt_$o.log 2>&1
done
