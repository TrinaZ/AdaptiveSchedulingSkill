# Batch regime: information-parity audit (legacy)

These files support Supplement §S4 (Table S3) and the batch-regime paragraph of §6.5:
- `fix_runtime_leak.py` removes access to the true processing time from the batch-regime templates.
- `template_leak_test.py` and `h2h_heldout.py` re-score generators with and without that access.
- `sem_heldout.py` re-scores the 320 single-call semantic-path rules on held-out instances (Supplement §S4, "Single-call semantic path"); its inputs `wgs_matrix_8seeds.json` and `wgs_matrix_8seeds_regated.json` (the original 960-trial batch records, including every generated rule) are included here; like the other scripts it runs against the legacy code base (`--src`).

They depend on the older batch pipeline (`laosa.py`, `head_to_head.py`, `run_poms_experiments.py`, `algorithm_archive.py`, `llm_adapter.py`). Those files are copied here unchanged by `tools/assemble_release.sh`, and the main framework does not use them.

Run from this folder with `--src ../code` after copying them in. This code is kept for transparency and is not maintained.

Result files: `template_leak.json`, `h2h_heldout.json`, `sem_heldout.json`.
