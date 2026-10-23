---
name: adaptive-scheduling-skill
description: Governed redesign of a service's dispatching rule after a change in traffic, capacity, or SLA. Use when an operator reports degraded SLA performance or a new SLA text, or when cost telemetry should be checked for change. Runs Monitor -> Diagnose -> (Generate) -> Govern and returns an auditable Skill Package.
---
# Adaptive Scheduling Skill

Four sub-skills with typed JSON contracts (schemas/) and one deterministic orchestrator.
**The agent may call the sub-skills individually or run the orchestrator, but must never deploy a rule that does not
come out of `govern` as a Skill Package with decision `deploy` or an approved `deploy-after-approval`.**

| step | skill | input -> output | LLM? |
|---|---|---|---|
| M | `monitor/` | TelemetryRecord -> ChangeReport | no |
| A | `diagnose/` | DiagnoseInput -> DiagnosisReport | no |
| P | `generate/` | GenerateInput -> CandidateSet (untrusted) | yes, only if not adequate |
| E | `govern/` | GovernInput -> SkillPackage | no |

## Typical agent session
1. Collect per-window cost telemetry of the deployed rule; run `monitor/run.py`.
2. If `alarm` is set (or the operator supplied a new SLA text), build DiagnoseInput and run `diagnose/run.py`.
3. If `adequate` is false, run `generate/run.py` (needs an LLM API key in the environment).
4. Always run `govern/run.py`; store the package with `core.write_package` (versioned, with rollback pointer).
5. Report the decision; if `review_flag` is true or approval is required, ask the responsible human.

Or end to end: `python3 orchestrator.py --scenario scenario.json --store knowledge_store/packages`.

## Setup
Set `ASS_REPO` to the `code/` folder of the repository (it holds `online_sla_bench.py` and `govern_sandbox.py`).
LLM backend: `ASS_BACKEND`, `ASS_MODEL`, `ASS_MAX_TOKENS`; key in `DEEPSEEK_API_KEY` / `DASHSCOPE_API_KEY` / ...
