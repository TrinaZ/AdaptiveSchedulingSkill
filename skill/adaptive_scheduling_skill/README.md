# adaptive_scheduling_skill

Agent-skill packaging of the Adaptive Scheduling Skill (paper Section 3). Layout:

```
adaptive_scheduling_skill/
  SKILL.md            entry point for an agent (Claude Code, Codex, ...): when to use, workflow, safety rule
  orchestrator.py     deterministic MAPE-K workflow (no LLM in control flow)
  core.py             implementation of the four skills as pure record -> record functions
  monitor/  diagnose/  generate/  govern/      SKILL.md + run.py (JSON in, JSON out)
  schemas/            JSON Schemas of TelemetryRecord, ChangeReport, DiagnoseInput, DiagnosisReport,
                      CandidateSet, SkillPackage
```
Requires `online_sla_bench.py`, `govern_sandbox.py` and `safe_exec.py`, which are in `code/` of the repository: `export ASS_REPO=<repo>/code`.

Quick test (no API key needed for the first scenario if the library is adequate):
```bash
cd <repo root>
python3 adaptive_scheduling_skill/orchestrator.py --scenario adaptive_scheduling_skill/examples/scenario_structural.json --store knowledge_store/packages
python3 adaptive_scheduling_skill/orchestrator.py --scenario adaptive_scheduling_skill/examples/scenario_semantic.json   --store knowledge_store/packages
cat knowledge_store/packages/manifest.json
```
Using it from an agent: open the repository in Claude Code (or another agent that reads SKILL.md files) and ask,
e.g., "the SLA changed to: premium requests have hard due times ...; run the adaptive scheduling skill". The agent
follows SKILL.md, calls the sub-skills, and must not deploy anything that `govern` did not approve.
