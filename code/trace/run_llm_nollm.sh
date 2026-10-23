#!/bin/sh
# H10 non-LLM generators on the LLM-inference replay; one objective per call, skips objectives already done
cd "$(dirname "$0")"
for o in "$@"; do
  [ -s ../../results/trace_llm/llm_nollm_$o.json ] && continue
  python3 online_sla_bench_llm.py --src .. --rho 1.0 --seeds 10 --n_search 12 --n_heldout 60 --objectives $o \
     --gens library,pool_full,knowledge,knowledge_tuned --out ../../results/trace_llm/llm_nollm_$o.json > ../../results/trace_llm/ln_$o.log 2>&1
done
