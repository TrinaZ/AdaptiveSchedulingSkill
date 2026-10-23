# H10: LLM-inference replay -- how to run

1. **Freeze first.** Commit and push, before any LLM call:
   `preregistration/H10_preregistration.md`, `code/trace/extract_azure_llm_trace.py`, `code/trace/make_llm_bench.py`,
   `code/trace/online_sla_bench_llm.py`, `code/trace/run_llm_nollm.sh`, `code/trace/llm_probe.py`, `code/trace/azure_llm_windows.csv`.
   Check: `sha256sum code/trace/online_sla_bench_llm.py code/trace/azure_llm_windows.csv` must give
   `71a8a8ff…de8d` and `e636d2db…9d30` (full hashes in the pre-registration).

2. **LLM runs** (keys only from the environment, e.g. `export DEEPSEEK_API_KEY=...` in your own shell; never in a file):

   deepseek-chat (about 12 min per objective):
   ```
   cd code/trace
   for o in tard miss slo_wait mix premium; do
     python3 online_sla_bench_llm.py --src .. --rho 1.0 --seeds 10 --n_search 12 --n_heldout 60 --objectives $o \
       --gens llm_skill --out ../../results/trace_llm/llm_llm_$o.json
   done
   ```
   deepseek-reasoner (about 2 h per objective; the five objectives can run in parallel in five terminals):
   ```
   python3 online_sla_bench_llm.py --src .. --rho 1.0 --seeds 10 --n_search 12 --n_heldout 60 --objectives $o \
     --gens llm_skill --backend deepseek --model deepseek-reasoner --temperature none --max_tokens 32000 --tag ds-reasoner \
     --out ../../results/trace_llm/llm_reasoner_$o.json
   ```

3. **Send back** `results/trace_llm/llm_llm_*.json`, `llm_reasoner_*.json` and the matching `*_llmdebug.json` files.

4. **Analysis:** `cd code/trace && python3 h8_analysis.py ../../results/trace_llm/*.json`

The non-LLM generators (`run_llm_nollm.sh`) are run separately and give `results/trace_llm/llm_nollm_*.json`.
