#!/bin/sh
# runs one objective at a time per worker; skips objectives already done
cd "$(dirname "$0")"
for o in "$@"; do
  [ -s trace_nollm_$o.json ] && continue
  python3 online_sla_bench_trace.py --src . --rho 1.0 --seeds 10 --n_search 12 --n_heldout 60 --objectives $o \
     --gens library,pool_full,knowledge,knowledge_tuned --out trace_nollm_$o.json > tn_$o.log 2>&1
done
