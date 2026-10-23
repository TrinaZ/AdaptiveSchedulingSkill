# H9 pre-registration: multi-tenant replay with reactive autoscaling (Alibaba cluster-trace-v2018)

Fixed before any LLM call on this benchmark. Commit this file together with the hashes below before running Section 4.

## 1. Why

H8 replayed one service's worth of the trace (single-service thinning, fixed capacity). Reviewers will ask about
multi-tenancy, interference between services, and autoscaling. H9 replays the same trace windows as a shared cluster
of four tenant services with a reactive autoscaler.

## 2. Benchmark (frozen)

- `online_sla_bench_mt.py`, sha256 `b0f6cd954892f07dc38cf01970baa3e961689b1405c98b6e85213461d7228b35`,
  derived from the amended H8 benchmark (`online_sla_bench_trace.py`, sha256 `b07e6747…4102`) by `make_mt_bench.py`
  (only the substitutions listed there).
- Data: `alibaba_trace_windows.csv`, sha256 `518fc88c6ab35e9efc67de541b8300372698c5ced2d4519105e4479093b153c7` (same as H8).
- Thinning by whole jobs to mean concurrency 48 (four services' worth; 1/60 of jobs): 17,853 and 18,626 tasks,
  24.8 and 25.9 arrivals per minute.
- Tenants: each job belongs to one of 4 tenant services by an independent hash of its name; tenant 0 is premium (weight ×3).
  All tenants share one capacity pool.
- Scenes: `T0`, `T1` (fixed capacity = window's average offered load, rho = 1); `T0as`, `T1as` (reactive autoscaling:
  each instance's capacity = offered load of the 200 tasks that arrived just before it, clipped to [0.5, 2] × the window
  average).
- Instances: 200 consecutive tasks; search / validation / held-out from the first 20% / next 40% / last 40% of each window
  (as in H8). Seeds, budgets, gates and statistics as in H8.
- Objectives: `tard`, `slo_wait`, `premium`. Cells: 4 scenes × 3 objectives × 10 seeds = 120 per generator.
- A headroom probe (no LLM, before this file) showed library spreads of 486% (tard), 356% (slo_wait), 10% (premium).
  No benchmark parameter was tuned on it.

## 3. Hypotheses (α = 0.05 each)

- **H9a (governance, confirmatory).** For each of knowledge_tuned, llm_skill (deepseek-chat), and llm_skill@ds-reasoner:
  the paired-t gated violation rate is ≤ 5%, and ungated violations exceed gated violations (exact McNemar, one-sided).
- **H9b (standard LLM).** llm_skill vs knowledge_tuned, two-sided Wilcoxon on rel (prediction: no difference, as H6/H8b).
- **H9c (reasoning model).** llm_skill@ds-reasoner < knowledge_tuned, one-sided Wilcoxon on rel.
- **Exploratory (not a test of the framework):** gated violation rate in autoscaled vs fixed-capacity scenes.

## 4. Runs

No-LLM generators: `--gens library,pool_full,knowledge,knowledge_tuned`. LLM generators: `--gens llm_skill` with
deepseek-chat and with deepseek-reasoner (`--tag ds-reasoner`). All with
`--rho 1.0 --seeds 10 --n 200 --n_search 12 --n_heldout 60 --objectives <o>`.
Analysis: `h8_analysis.py` on the H9 result files (same tests).

## 5. Commitments

- If H9a fails, the paper reports it and its likely cause, as for H8a.
- No scene, objective, tenant rule, or autoscaling rule is changed after the first LLM call.
- Any runtime fix follows the H8 Amendment 1 rule and is recorded here as an amendment before affected runs restart.
