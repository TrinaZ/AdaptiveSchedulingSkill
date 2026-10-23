# Pre-registration: H7 (SLAs outside the knowledge base)

Frozen: 2026-10-17 05:20 (UTC+8), **before any LLM call on these objectives**.
- `online_sla_bench.py` sha256: `42e40750d03e1835151a9285e84a068abb1f8ee1a6d35ef53fe431e00abb8411`
- `EXPERT_H7` block sha256: `ef782d0c4dbfe9e2a3db3cee55a9b49b31e0aeae326ef0f3325852672d16b1dd`

Commit this file and the script to the private repository now, so the commit timestamp records the freeze.

## Motivation

In the main run (H6), `llm_skill` did not significantly beat `knowledge_tuned`. All five SLA objectives there are of a kind the eight literature families (ATC, CR, MinSlack, MOD, AgingWSPT, TenantSPT, TenantATC, WaitGuard) were designed for: tardiness, misses, waiting, and tenant tiers. H7 tests the paper's claimed use case: **an SLA whose shape the fixed knowledge base did not anticipate**.

## Objectives

The knowledge base is unchanged; it was fixed before these objectives were written.

| Name | SLA text (verbatim in `OBJ_TEXT`) | What no knowledge-base family models |
|---|---|---|
| `tiered` | Flat penalty by lateness relative to the estimate: 1 (≤0.25×est), 3 (≤1×est), 10 (later); premium ×3 | A saturating, step-shaped penalty: once a request is in the top tier, further delay is free |
| `typemiss` | Miss cost by stage: variant_calling 5, sv_calling 3, alignment 1 | Priorities by `task_type`, which no family uses |
| `decay` | Value decays linearly after due and reaches 0 at due + 2×est | A capped loss: hopeless requests should be deferred |

A fourth objective, `typefair` (worst per-type mean slowdown), was dropped **before** any LLM run. In a no-LLM headroom probe, every rule was within 1% (slowdowns 1.1–1.3), so no generator could show a difference. This is reported in the supplement.

## Design

Identical to the main run:
- 4 scenes × 10 seeds per objective (120 cells);
- `--rho 1.0 --n_search 12 --n_heldout 60 --n_val 1000 --budget 100 --tau 0.02`.

Generators:

| Generator | Where it runs |
|---|---|
| `library`, `pool_full`, `knowledge`, `knowledge_tuned` | No LLM; run by the authors' assistant |
| `expert` | Frozen hand-written rules (one structure per objective, see `EXPERT_H7`), calibrated with the same procedure and budget as `knowledge_tuned`. Written by the authors before any LLM output on these objectives existed |
| `llm_skill` | DeepSeek `deepseek-chat`, unchanged prompt; the LLM sees the same literature rules and their costs |
| `llm_zeroshot` | Cheap ablation |

## Hypotheses

**H7 (primary, confirmatory).** Pooled over the 120 H7 cells, `llm_skill` has lower held-out `rel` than `knowledge_tuned`.
- Test: one-sided Wilcoxon signed-rank on paired `rel` differences.
- Threshold: α = 0.05.

**H7b (secondary).** Same comparison under the t gate (`rel_t`). Deployments stay safe: `violation_t` ≤ 5% for `llm_skill`.

**H7c (exploratory, no claim).** `llm_skill` vs `expert`, two-sided Wilcoxon. Tells the reader how close the LLM gets to a human expert who saw the SLA.

## Reporting commitments

- **All three objectives** are reported, whatever the result.
- **If H7 fails,** the paper states that the LLM did not add value even outside the knowledge base, and the discussion is weakened accordingly.
- **No other objective** is added or dropped after the LLM run.

---

## Addendum: H7′ (deepseek-reasoner)

Declared 2026-10-17 (UTC+8), **after** the H7 result and **before** any deepseek-reasoner call on the H7 objectives.

H7 with `deepseek-chat` was not supported:
- `llm_skill` − `knowledge_tuned`: +0.17 points, p = 0.67;
- `expert` beat `llm_skill` in 91 of 120 cells.

A rule-content analysis of the 1,790 `deepseek-chat` rules found:
- 840 contain logic for late requests;
- 701 of those *raise* the priority of late requests;
- 6 lower it.

The expert rules' distinctive element, deferring requests whose penalty can no longer grow, was essentially never proposed.

Since deepseek-reasoner beat `knowledge_tuned` on the main SLAs (H6′, p = 7×10⁻⁴), we test whether a reasoning model fills this gap.

**Design.** Identical to H7: same 120 cells, same `online_sla_bench.py` (sha256 `42e40750…8411`), same prompt, same budget 100, `--gens llm_skill`. The only change is the model:
- `--backend deepseek --model deepseek-reasoner --temperature none --max_tokens 32000 --tag ds-reasoner`;
- the generator is recorded as `llm_skill@ds-reasoner`.

**H7′ (confirmatory for this run).** `llm_skill@ds-reasoner` < `knowledge_tuned` in held-out rel, one-sided Wilcoxon over the 120 cells, α = 0.05.

**H7′b.** The same under the t gate. Gated violations must be ≤ 5%.

**H7′c (exploratory).** `llm_skill@ds-reasoner` vs `expert`, two-sided. Also reported, descriptively: the share of reasoner rules that defer late requests, using the same keyword classifier as above (`h7_rule_content.py`).

**Reporting.** H7′ is reported as a test declared *after* H7 failed. Its result does not change the H7 verdict.
