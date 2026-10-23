---
name: govern
description: Decide whether a generated rule may be deployed. Static AST whitelist, execution in an isolated resource-limited process, paired confidence-bound gate against the best library rule on validation traffic, change-type-dependent approval, fallback with review flag. Always produces a Skill Package.
---
# Govern skill (E of MAPE-K)

**Input** `GovernInput`: `candidate_set`, `objective`, `change_type`, `validation_traffic` (TrafficSpec, >= 1000
instances recommended), `incumbent` (rule name, for rollback), `tau` (default 0.02), `delta` (0.05),
`gate` ("t" = paired t bound, practical; "eb" = paired empirical Bernstein, finite-sample guarantee of Prop. 2).

**Output** `SkillPackage` (paper Def. 2): `rule_source`, `params`, `change_type`, `decision`
(deploy | deploy-after-approval | fallback), `approval` (autonomous | expert | human), `evidence`
(n, library means, per-candidate upper bound, fallback), `rollback_to`, `review_flag`, `provenance`.

**Run** `python3 govern/run.py --in govern_input.json --out package.json`

**Guarantee** with the EB gate and bounded costs: F(deployed) <= (1+tau)(F(best library) + 2 eps_n) w.p. 1-delta,
whatever the candidates are (Prop. 2). Isolation: govern_sandbox.py (AST whitelist; forked child with
RLIMIT_FSIZE=0, RLIMIT_NPROC=0, CPU/memory caps, timeout).

**On failure** fallback to the library rule that is best on validation, review flag set.
