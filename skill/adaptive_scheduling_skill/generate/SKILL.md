---
name: generate
description: Ask an LLM to write state-aware dispatching rules for the current SLA, grounded in the literature rules of the knowledge store, and calibrate their constants. Use only when a DiagnosisReport says the library is not adequate. Output is UNTRUSTED and must go to govern.
---
# Generate skill (P of MAPE-K)

**Input** `GenerateInput`: `diagnosis` (DiagnosisReport), `budget` (rule evaluations, default 100), `seed`.
Backend via environment: `ASS_BACKEND` (deepseek | qwen | openai | glm | kimi), `ASS_MODEL`, `ASS_MAX_TOKENS`;
API key in the backend's usual variable (e.g. `DEEPSEEK_API_KEY`).

**Procedure** (llm_skill): literature rules with their search costs -> LLM writes 5 rules (round 1) -> two feedback
rounds with costs and diagnostics -> constants calibrated on half of the search set, re-scored on all of it.

**Output** `CandidateSet`: `candidates` [{label, source, params, search_cost, llm_written}], `llm_calls`, `tokens`,
`prompt_sha256` (provenance). Empty if nothing beats the library on search.

**Run** `python3 generate/run.py --in generate_input.json --out candidates.json`

**Guarantee** none (untrusted). **On failure** empty candidate set.
