# Adaptive Scheduling Skill

Code, agent skills, pre-registration and all results for the paper

> **Adaptive Scheduling Skill: An LLM-Based Agent Skill for Fast and Safe Service Dispatching Under Change**
> Tian Zheng and Jiayin Wang, Xi'an Jiaotong University. Submitted to *IEEE Transactions on Services Computing*, Special Issue on Large Language Models in Service-Oriented Ecosystems Design.

The framework packages dispatching-rule redesign as four agent skills under a deterministic MAPE-K orchestrator:

| Skill | What it does |
|---|---|
| **Monitor** | Detects a change in the deployed rule's cost (CUSUM). |
| **Diagnose** | Types the change. Checks whether the rule library suffices; if it does, no LLM is called. |
| **Generate** | An LLM writes state-aware rules from the SLA text and from literature rules; their constants are then calibrated. |
| **Govern** | Checks each candidate against an AST whitelist, runs it in an isolated process, and applies a paired confidence-bound gate. It deploys a candidate only with that certificate; otherwise it falls back to the library rule and sets a review flag. |

Every episode writes an auditable, versioned **Skill Package**.

**All results reported in the paper are already in `results/`.** You do not need to re-run anything to check a number. Every table can be regenerated from those files with the analysis commands in [Regenerating tables and figures](#regenerating-tables-and-figures-no-llm-seconds), which need no API key and finish in seconds.

---

## Main findings

| Finding | Evidence | Where in the paper |
|---|---|---|
| Ungated deployment of generated rules ends in a bound violation in 12–20% of cells; the paired gate cuts this to 0–5%. | 34% in a stress test with no LLM. | Table 6, Fig. 2 |
| The gate works only if validation traffic matches the deployment horizon. | 60-request validation deployed on 300-request windows: 15–16% violations. Matched validation: 0–0.5%. | §5.5, Table S12 |
| At equal budget, LLM generation beats a GP hyper-heuristic over the same vocabulary. | p < 10⁻¹⁰ | §5.4 |
| Under equal knowledge and budget, standard LLMs (DeepSeek-V3, Qwen-Max) tie calibrated literature rules. | H6: p = 0.86 | Table 5 |
| A reasoning model beats calibrated literature rules. | H6′: p = 7×10⁻⁴ | Table S6 |
| On 3 pre-registered SLAs outside the literature, a standard LLM does not beat the literature rules; a reasoning model does on 1 of 3. | H7: p = 0.67; H7′: p = 0.003 | §5.4, Tables S8–S9 |
| Long run (50 changes): governed adaptation cuts regret 26–29% against library re-selection with a standard LLM or literature rules as generator, and 43% with a reasoning model. | Lower on every stream with the reasoning model. | Fig. 4, Table S10 |
| Pre-registered replay of the Alibaba v2018 cluster trace (200 cells per generator): the standard LLM again ties the literature rules, the reasoning model again beats them; the t gate keeps violations at 2.5–6%, concentrated in one drifting window; the EB gate has none. | H8a: 2 of 3 generators ≤5% (DeepSeek-V3: 6.0%); H8b: p = 0.76; H8c: p = 1.5×10⁻⁵ | §5.6, Table 7, §S9 |
| Pre-registered multi-tenant replay with reactive autoscaling (4 tenant services, 120 cells per generator): neither LLM is significantly better than the literature rules, both LLMs exceed the 5% gated-violation target (6.7%; knowledge_tuned is exactly at the inclusive bound, 5.0%), the t gate is not significantly better than a random gate with the same deployment rate, and the EB gate violates only under autoscaling. | H9a: 1 of 3; H9b: p = 0.12; H9c: p = 0.12 (not supported) | §5.6, Table 7, §S10 |
| Pre-registered replay of the Azure LLM inference trace 2023 (coding-assistant and conversation services, 200 cells per generator): the standard LLM ties the literature rules, the reasoning model beats them; gating keeps knowledge_tuned at 3.0% but both LLMs exceed 5% (7.0%, 8.5%), with the t gate no better than a random gate for them; the EB gate has no violation. | H10a: kt only; H10b: p = 0.10; H10c: p = 4×10⁻⁸ | §5.6, Table 7, §S14 |
| Ablation (no LLM): removing Monitor raises long-run regret from 42 to 105; removing calibration or the gate also raises it; the EB gate costs regret (51) for certification; Diagnose's skip saves generation episodes at some regret. | per-stream regret | Table S17 |

---

## Repository layout

```
code/                   all scripts; run them from inside this folder
  online_sla_bench.py   online SLA benchmark: simulator, SLA objectives, all generators, gates, multi-backend LLM client
                        (frozen: sha256 42e40750...8411, see preregistration/)
  govern_sandbox.py     isolation for untrusted rule code (AST whitelist + forked, resource-limited process)
  safe_exec.py          AST safety check used when LLM rules are loaded
  problems/genomic_scheduling/   WGS request generator (types, runtimes, estimates, scenes)
  *.py                  experiment runners and analysis scripts (see tables below)
  figures/fig_*.py      regenerate the paper's figures from results/
  trace/                trace replays: Alibaba v2018 (H8), multi-tenant with autoscaling (H9), Azure LLM inference (H10);
                        extraction scripts, trace benchmarks (derived from the frozen one), analysis, random-gate control
skill/adaptive_scheduling_skill/   the four skills (SKILL.md + run.py each), JSON Schemas, orchestrator, examples
preregistration/H7_preregistration.md   frozen H7 / H7′ protocol
preregistration/H8_preregistration.md   frozen H8 protocol (Alibaba trace replay) with Amendment 1
preregistration/H9_preregistration.md   frozen H9 protocol (multi-tenant replay with autoscaling)
preregistration/H10_preregistration.md  frozen H10 protocol (Azure LLM-inference replay) with procedural notes
results/                every result file used in the paper (JSON) and all LLM prompt/response logs (*_llmdebug.json);
                        trace/ = H8, trace_mt/ = H9, trace_llm/ = H10
figures/                the paper's figures (PDF/PNG)
legacy_batch_regime/    information-parity audit of the batch regime (Supplement §S4); legacy code, not maintained
tools/                  secret scanner and release helper
```

## Setup

```bash
python3 -m pip install -r requirements.txt   # numpy scipy matplotlib openai
cd code
```

`govern_sandbox.py` uses `fork` and resource limits, so it needs Linux or WSL.

Experiments that call an LLM read the API key **only** from an environment variable:

| Backend | Environment variable |
|---|---|
| DeepSeek | `DEEPSEEK_API_KEY` |
| Qwen | `DASHSCOPE_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| GLM | `ZHIPUAI_API_KEY` |
| Kimi | `MOONSHOT_API_KEY` |

Never put keys in files.

## Using the skill from an agent

```bash
export ASS_REPO=$(pwd)/code
python3 skill/adaptive_scheduling_skill/orchestrator.py \
    --scenario skill/adaptive_scheduling_skill/examples/scenario_structural.json --store knowledge_store/packages
cat knowledge_store/packages/manifest.json
```

This example needs no API key: the library is adequate after the change, so `generate` does not call the LLM. Expected output:

```
Monitor : 40
Diagnose: structural headroom -0.125 adequate
Generate: 0 candidate(s), 0 LLM call(s)
Govern  : fallback | rule: FIFO | review flag: True -> knowledge_store/packages/pkg-0001.json
```

The four records of the episode are written to `knowledge_store/records/`. Supplementary Section S15 reproduces the top-level `SKILL.md`, the gate code of `govern` and this run; Supplementary Section S6 shows a Skill Package from the LLM path.

An agent that reads `SKILL.md` files, such as Claude Code or Codex, can call the sub-skills directly. The top-level `SKILL.md` states the one rule an agent must follow: nothing is deployed unless `govern` emitted a Skill Package that allows it.

---

## Regenerating tables and figures (no LLM, seconds)

Run from `code/`:

| Paper item | Command |
|---|---|
| Table 5, Table 6, Table S2 | `python3 make_tables.py ../results/online_full_*.json ../results/eoh_*.json` |
| Table S6 (backends, H6′) | `python3 compare_backends.py ../results/online_full_*.json ../results/backend_*.json` |
| Table S19 (per-objective contribution, post hoc) | `python3 per_objective.py --results ../results` |
| Tables S8–S9, H7 / H7′ tests | `python3 h7_analysis.py ../results/h7_nollm_*.json ../results/h7_llm_*.json ../results/h7_backend_ds-reasoner_tiered.json ../results/h7_backend_ds-reasoner_typemiss.json ../results/h7_backend_ds-reasoner_decay.json` |
| H7 rule-content analysis | `python3 h7_rule_content.py ../results/h7_llm_*_llmdebug.json ../results/h7_backend_ds-reasoner_*_llmdebug.json` |
| GP comparison (§5.4) | `python3 gp_compare.py ../results/gp_sla.json ../results/gp_h7.json ../results/online_full_*.json ../results/h7_nollm_*.json ../results/h7_llm_*.json` |
| Fig. 4 and the long-run summary | `python3 online_longrun.py --bench online_sla_bench.py --src . --out ../results/longrun.json --summary_only` (same with `longrun_reasoner.json`) |
| Table 7, Tables S14–S15 (H8, trace) | `cd trace && python3 h8_analysis.py ../../results/trace/trace_*.json` |
| Table 7 (bottom), Table S16 (H9, multi-tenant) | `cd trace && python3 h8_analysis.py ../../results/trace_mt/mt_nollm_*.json ../../results/trace_mt/mt_llm_{tard,slo_wait,premium}.json ../../results/trace_mt/mt_reasoner_{tard,slo_wait,premium}.json` |
| Table 7 (middle), Table S20 (H10, LLM inference) | `cd trace && python3 h8_analysis.py ../../results/trace_llm/*.json` and `python3 random_gate_trace.py ../../results/trace_llm/llm_*.json` |
| All figures | `for f in figures/fig_*.py; do python3 $f; done` (writes to `../figures/`) |

---

## Re-running the experiments

Run from `code/`. All runs are seeded and deterministic apart from LLM sampling. Seeds of the search, validation and held-out instance sets never overlap (Supplementary Table S21).

**Only re-run what you need to change.**

### A. No LLM (CPU only)

| Paper item | Command | Output | Time* |
|---|---|---|---|
| Fig. 2, Table S4: governance stress test | `python3 online_governance.py --bench online_sla_bench.py --src . --seeds 6 --n_val 200 --out ../results/online_governance_v3.json` | `online_governance_v3.json` | ~1 h |
| Fig. 5: risk–coverage | `python3 gate_risk_coverage.py --bench online_sla_bench.py --src . --out ../results/gate_risk_coverage.json` | `gate_risk_coverage.json` | ~1 h |
| Fig. S1: gate calibration vs. δ, random-gate control | `python3 gate_calibration.py --bench online_sla_bench.py --src . --out ../results/gate_calibration.json` | `gate_calibration.json` | ~30 min |
| H7 non-LLM generators (incl. `expert`) | `for o in tiered typemiss decay; do python3 online_sla_bench.py --src . --rho 1.0 --seeds 10 --n_search 12 --n_heldout 60 --objectives $o --gens library,pool_full,knowledge,knowledge_tuned,expert --out ../results/h7_nollm_$o.json; done` | `h7_nollm_*.json` | ~30 min each |
| GP baseline (§5.4) | `python3 gp_baseline.py --bench online_sla_bench.py --src . --objectives tard,miss,slo_wait,mix,premium --budgets 100,1000 --out ../results/gp_sla.json`, and the same with `--objectives tiered,typemiss,decay --out ../results/gp_h7.json` | `gp_sla.json`, `gp_h7.json` | ~2 h each |
| Table S10: selection baselines | `python3 longrun_selection.py --bench online_sla_bench.py --src . --longrun ../results/longrun.json --out ../results/longrun_selection.json` | `longrun_selection.json` | ~5 min |
| Table S10: knowledge + Govern, no LLM | `python3 longrun_kngov.py --bench online_sla_bench.py --src . --longrun ../results/longrun.json --out ../results/longrun_kngov.json` | `longrun_kngov.json` | ~15 min |
| §S6: rollback replay of the prototype deployments | `python3 rollback_replay.py --bench online_sla_bench.py --src . --proto ../results/prototype.json --out ../results/rollback_replay.json` | `rollback_replay.json` | ~2 min |
| Table S17, Fig. S3: ablation of the governed loop (no LLM) | `python3 longrun_ablation.py --bench online_sla_bench.py --src . --longrun ../results/longrun.json --out ../results/longrun_ablation.json` | `longrun_ablation.json` | ~10 min |
| Table S11: operator rejection, Diagnose threshold | `python3 longrun_robust.py --bench online_sla_bench.py --src . --longrun ../results/longrun.json --out ../results/longrun_robust.json` | `longrun_robust.json` | ~5 min |
| Table S13: window length | `python3 window_transfer.py --bench online_sla_bench.py --src . --files ../results/online_full_{tard,miss,slo_wait,mix,premium}.json --n 300 --n_heldout 60 --out ../results/window_transfer_300.json` | `window_transfer_300*.json` | ~15 min |
| Table S12: horizon-matched gate | `python3 horizon_gate.py --bench online_sla_bench.py --src . --files ../results/online_full_{tard,miss,slo_wait,mix,premium}.json --n 300 --n_valN 200 --out ../results/horizon_gate.json` | `horizon_gate.json` | ~2 h |
| Diagnose false skips (§S8) | `python3 diagnose_skip.py --bench online_sla_bench.py --src . --files ../results/h7_*.json ../results/online_full_mix.json ../results/online_full_slo_wait.json --out ../results/diagnose_skip.json` | `diagnose_skip.json` | ~5 min |
| H8 trace data (once) | download `batch_task.csv` of [cluster-trace-v2018](https://github.com/alibaba/clusterdata), then `cd trace && python3 extract_alibaba_trace.py batch_task.csv --out alibaba_trace_windows.csv` (check sha256 `518fc88c…53c7`) | `alibaba_trace_windows.csv` (not in git, ~90 MB) | ~1 min |
| H8 non-LLM generators | `cd trace && for o in tard miss slo_wait mix premium; do python3 online_sla_bench_trace.py --src .. --rho 1.0 --seeds 10 --n_search 12 --n_heldout 60 --objectives $o --gens library,pool_full,knowledge,knowledge_tuned --out ../../results/trace/trace_nollm_$o.json; done` | `trace/trace_nollm_*.json` | ~5 min each |
| H8 GP context | `cd trace && python3 gp_baseline_trace.py --bench online_sla_bench_trace.py --src .. --scenes T0,T1,T0cap,T1cap --objectives tard,miss,slo_wait,mix,premium --budgets 100 --out ../../results/trace/trace_gp100.json` | `trace/trace_gp100.json` | ~30 min |
| Table S7: adversarial test | `python3 adversarial_test.py --bench online_sla_bench.py --src . --debug ../results/online_full_*_llmdebug.json` | `adversarial_test.json` | ~30 min |

\*On a 2-core machine.

### B. With an LLM (needs an API key)

| Paper item | Command | Output |
|---|---|---|
| Table 5 / Table 6: main online run | `for o in tard miss slo_wait mix premium p95 wflow; do python3 online_sla_bench.py --src . --rho 1.0 --seeds 10 --n_search 12 --n_heldout 60 --objectives $o --gens library,pool_full,knowledge,knowledge_tuned,llm_zeroshot,llm_tuned,llm_feedback,llm_skill --out ../results/online_full_$o.json; done` | `online_full_*.json` |
| Table 5, row `llm_eoh` | same, with `--gens library,llm_eoh --out ../results/eoh_$o.json` (five SLA objectives) | `eoh_*.json` |
| Table S6: Qwen-Max | same, with `--gens library,knowledge_tuned,llm_skill --backend qwen --model qwen-max --tag qwen-max --out ../results/backend_qwen-max_$o.json` | `backend_qwen-max_*.json` |
| Table S6: deepseek-reasoner (H6′) | same, with `--gens library,knowledge_tuned,llm_skill --backend deepseek --model deepseek-reasoner --temperature none --max_tokens 32000 --tag ds-reasoner --out ../results/backend_ds-reasoner_$o.json` | `backend_ds-reasoner_*.json` |
| H7: deepseek-chat | `for o in tiered typemiss decay; do python3 online_sla_bench.py --src . --rho 1.0 --seeds 10 --n_search 12 --n_heldout 60 --objectives $o --gens llm_skill,llm_zeroshot --out ../results/h7_llm_$o.json; done` | `h7_llm_*.json` |
| H7′: deepseek-reasoner | same, with `--gens llm_skill --backend deepseek --model deepseek-reasoner --temperature none --max_tokens 32000 --tag ds-reasoner --out ../results/h7_backend_ds-reasoner_$o.json` | `h7_backend_ds-reasoner_*.json` |
| H8: deepseek-chat / deepseek-reasoner on the trace | `cd trace && for o in tard miss slo_wait mix premium; do python3 online_sla_bench_trace.py --src .. --rho 1.0 --seeds 10 --n_search 12 --n_heldout 60 --objectives $o --gens llm_skill --out ../../results/trace/trace_llm_$o.json; done`; reasoner: add `--backend deepseek --model deepseek-reasoner --temperature none --max_tokens 32000 --tag ds-reasoner` and write `trace_reasoner_$o.json` | `trace/trace_llm_*.json`, `trace/trace_reasoner_*.json` (~12 min per objective for chat; reasoner ~174 s model time per cell) |
| H9: multi-tenant replay (LLM) | `cd trace && for o in tard slo_wait premium; do python3 online_sla_bench_mt.py --src .. --rho 1.0 --seeds 10 --n 200 --n_search 12 --n_heldout 60 --objectives $o --gens llm_skill --out ../../results/trace_mt/mt_llm_$o.json; done`; reasoner as for H8 with `--tag ds-reasoner`; non-LLM generators: `./run_mt_nollm.sh tard slo_wait premium` | `trace_mt/*.json` |
| H10: LLM-inference replay | data: `python3 extract_azure_llm_trace.py AzureLLMInferenceTrace_code.csv AzureLLMInferenceTrace_conv.csv azure_llm_windows.csv` (CSV files from https://github.com/Azure/AzurePublicDataset, CC-BY; cite Patel et al., ISCA 2024); bench: `python3 make_llm_bench.py online_sla_bench_trace.py online_sla_bench_llm.py`; runs: see `code/trace/RUN_H10.md` | `trace_llm/*.json` |
| Fig. 3, Table S5, Fig. S2: prototype | `python3 online_prototype.py --bench online_sla_bench.py --src . --seeds 10 --out ../results/prototype.json` | `prototype.json` |
| Fig. 4: long run (deepseek-chat) | `python3 online_longrun.py --bench online_sla_bench.py --src . --seeds 5 --out ../results/longrun.json` | `longrun.json` |
| Table S10: long run (deepseek-reasoner) | `for k in 0 1 2 3 4; do python3 online_longrun.py --bench online_sla_bench.py --src . --seeds 5 --only_seed $k --backend deepseek --model deepseek-reasoner --temperature none --max_tokens 32000 --out ../results/longrun_reasoner_s$k.json & done; wait; python3 merge_longrun.py ../results/longrun_reasoner.json ../results/longrun_reasoner_s*.json` | `longrun_reasoner.json` |

All LLM runs abort (rather than silently falling back to library rules) if the API returns nothing five times or reports an empty balance. Every prompt and response is logged in `*_llmdebug.json`. `deepseek-chat` and `deepseek-reasoner` are rolling aliases, so re-runs may use a newer model version.

---

## Pre-registration

`preregistration/H7_preregistration.md` fixes the H7 objectives, the author-written `expert` rules, the hypotheses and the analysis before any LLM call on those objectives. The H7′ addendum was declared after the H7 result and before the reasoner run. Both were committed with the sha256 of `online_sla_bench.py`, and that file has not changed since.

`preregistration/H9_preregistration.md` fixes the multi-tenant replay with autoscaling (H9a–c), committed with the benchmark hash before any LLM call on it; no amendment was needed.

`preregistration/H10_preregistration.md` fixes the LLM-inference replay (H10a–c), committed with the hashes of the source trace, the extracted windows and the benchmark before any LLM call on it. Two runs failed before producing any output (a missing output directory; an exhausted API balance) and were rerun unchanged.

`preregistration/H8_preregistration.md` fixes the trace construction, scenes, hypotheses H8a–c and the analysis; it was committed with the sha256 of `online_sla_bench_trace.py` and of the extracted trace before any LLM call on the trace. Amendment 1 (a runtime guard for rules that fail on held-out traffic) was committed after the first crash and before the affected runs were restarted.

H1–H6 were specified in the run protocol before the online LLM runs (Supplementary Table S21).

## What is *not* in this repository

The repository contains only what the paper reports. Earlier exploratory work is excluded because it could not be re-validated under information parity, or used a pre-audit pipeline (Supplement §S4, "Results not carried over"). This covers:
- the re-entrant manufacturing ("Fab") domain;
- earlier, exploratory Alibaba-trace scenes (the pre-registered replay in `results/trace/` is new and is included);
- the extracted trace windows `alibaba_trace_windows.csv` (regenerate them from the public trace, see above);
- the coverage spectrum;
- the POMS/ITOR-era batch experiments.

## License

Code: MIT (see `LICENSE`). Results and logs: CC BY 4.0.

## Citation

Please cite the paper; a BibTeX entry will be added on publication.
