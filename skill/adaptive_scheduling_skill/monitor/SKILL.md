---
name: monitor
description: Detect a persistent change in the deployed dispatching rule's cost (CUSUM). Use when new cost telemetry arrives; returns a ChangeReport. Never changes the deployed rule.
---
# Monitor skill (M of MAPE-K)

**Input** `TelemetryRecord` (schemas/telemetry.json): `costs` (per-window cost of the deployed rule),
`reference_windows` (default 20), `h` (CUSUM threshold, default 5), `k` (drift, default 0.5),
`declared_change_at` (window index if the operator changed the SLA text, else null).

**Output** `ChangeReport`: `alarm` (window index or null), `declared`, `statistic`, `metric`.

**Run** `python3 monitor/run.py --in telemetry.json --out change_report.json`

**Guarantee** CUSUM with h = Theta(log T): O(1) expected false alarms over T windows and O(log T / KL) detection
delay (paper, Section 3.3 and Prop. 4). The skill reports *that* something changed, not *what*.

**On failure** no alarm; the incumbent stays deployed.
