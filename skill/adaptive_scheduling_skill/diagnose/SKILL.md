---
name: diagnose
description: Type a detected change (distributional / structural / semantic), compute diagnostics, and decide whether the rule library still suffices (headroom probe against literature rules, no LLM). Use after a ChangeReport with an alarm.
---
# Diagnose skill (A of MAPE-K)

**Input** `DiagnoseInput` (schemas/diagnose_input.json): `change_report`, `objective`, `reference_traffic`,
`recent_traffic`, `search_traffic` (TrafficSpecs), `headroom_threshold` (default 0.03).

**Output** `DiagnosisReport`: `change_type`, observable features before/after, `library_costs`, `best_library`,
`best_literature`, `headroom`, `adequate`, `diagnostics` (late requests by tenant, premium vs. all response, longest wait).

**Rules** semantic if the SLA text changed (declared); structural if capacity changed by > 5% or a new request type
appears; otherwise distributional. `adequate = headroom < threshold`: then the orchestrator skips Generate.

**Run** `python3 diagnose/run.py --in diagnose_input.json --out diagnosis.json`

**On failure / uncertainty** assign the strictest tier (semantic).
