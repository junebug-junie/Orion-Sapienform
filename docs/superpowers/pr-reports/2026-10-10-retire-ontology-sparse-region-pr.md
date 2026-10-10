# Retire the constant-1.0 `ontology_sparse_region` curiosity candidate

## Summary

- Orion's curiosity evaluator had a rule: "if this part of the graph has concepts but no ontology-branch nodes, it is a sparse region worth expanding." Nothing in Orion ever creates ontology-branch nodes, so the rule fired on every tick at full strength (1.0) and won every decision. It is now gone.
- Removed the producer block in `orion/substrate/frontier_curiosity.py` outright. No flag, no fallback.
- Removed `ontology_sparse_region` from the signal-type contract and `ontology_expand` from the expansion task-type contract. This block was the only production producer of either.
- Added a regression test (concept-dense slice, zero ontology branches, no real signals -> evaluator noops), a contract test, and a read-compat test for candidate rows persisted before the retirement.
- Spec: `docs/superpowers/specs/2026-10-07-orion-self-calibration-design.md` (PR #2528), "Real bugs found" item 3.

## Outcome moved

Before: every evaluator decision in the curiosity tick was `invoke` / `ontology_expand`, driven by a pinned constant. Live, last 24h (2026-10-10):

```text
sets_24h                    1172
sets_with_ontology_sparse    787
distinct_strengths           1.0          (one value, ever)
evaluator outcome/task       invoke|ontology_expand: 787 ; <evaluator not run>: 385
```

After: the evaluator only invokes on a real signal (contradiction, drift, evidence-gap markers, goal pressure, endogenous seeds). Endogenous `curiosity_candidate` seeds no longer get outranked by a constant. The persisted candidate sets no longer carry a 1.0 junk item at the top, which every downstream reader (felt-state curiosity lane, Hub curiosity hint, self-inquiry) was ranking first.

## Current architecture

- `services/orion-substrate-runtime/app/worker.py` `_endogenous_curiosity_tick` runs `FrontierCuriosityEvaluator.evaluate(...)` every 60s, writes `evaluator_outcome`/`evaluator_task` into `gate_json`, and persists up to 8 signals to `substrate_endogenous_curiosity_candidates`.
- `FrontierCuriosityEvaluator._derive_signals` emitted `ontology_sparse_region` whenever the read slice had concepts and zero `ontology_branch` nodes, strength `min(1, 0.55 + 0.03 * n_concepts)`.
- The decision's plan is never executed: `FrontierCuriosityOrchestrator` is not instantiated anywhere in production (only in tests). So `ontology_expand` had no executor.

## Architecture touched

- Contract: `FrontierInvocationSignalTypeV1`, `FrontierTaskTypeV1` (both narrowed by one value).
- Producer: `FrontierCuriosityEvaluator._derive_signals`.
- No bus channel, env, or Docker changes. `worker.py` untouched.

## Files changed

- `orion/substrate/frontier_curiosity.py`: removed the `ontology_sparse_region` block (and the now-unused concept/branch counts); short retirement note in place.
- `orion/core/schemas/frontier_curiosity.py`: dropped `ontology_sparse_region` from `FrontierInvocationSignalTypeV1`.
- `orion/core/schemas/frontier_expansion.py`: dropped `ontology_expand` from `FrontierTaskTypeV1`.
- `tests/test_cognitive_substrate_phase8_frontier_invocation.py`: flipped the old "must contain ontology_sparse_region" assertion; added regression + contract tests.
- `orion/substrate/relational/tests/test_curiosity_ctx_adapter.py`: fixtures moved to a valid kind; added a pre-retirement-row compat test.
- `services/orion-substrate-runtime/tests/test_store_observability_writers.py`, `.../test_worker_endogenous_curiosity_tick.py`, `tests/test_cognitive_substrate_phase6_frontier_expansion.py`, `tests/test_cognitive_substrate_phase7_frontier_landing.py`: fixtures moved off the retired values.
- `docs/architecture/unified_cognitive_substrate_phase8_frontier_invocation.md`: removed the ontology-sparsity rule, noted the retirement.

## Schema / bus / API changes

- Added: none
- Removed: `ontology_sparse_region` (signal type), `ontology_expand` (task type)
- Renamed: none
- Behavior changed: the curiosity evaluator no longer emits a constant-1.0 candidate; with no real signal it now returns `noop`.
- Compatibility notes: candidate rows persisted before deploy (30-day retention) still contain `ontology_sparse_region` items. The only reader that validates them into `FrontierInvocationSignalV1` is the relational curiosity adapter, which skips invalid items per-item (tested), and it reads only rows newer than 120s. Hub/self-inquiry readers treat rows as plain dicts. `gate_json.evaluator_task` is a plain string, never re-validated. Expansion requests are never persisted (orchestrator unused). Not kept as read-compat in the Literal: nothing needs it.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
pytest tests/test_cognitive_substrate_phase{1,6,7,8}*.py orion/substrate/relational orion/autonomy
  -> 368 passed, 1 failed (pre-existing on origin/main, see Risks)
system-one-appraisal-tests.yml pytest list (incl. curiosity tick + store writers) -> 65 passed
substrate-neighborhood.yml unit list -> 281 passed
orion-static-gates.yml: every run step extracted from the workflow and executed -> 26/26 PASS
  (incl. check_definition_drift --gate: no metric definition changed, no re-lock)

Mutation check:
  A) restore old producer + enum values -> regression test and contract test both FAIL
  B) restore only the producer block (enums retired) -> regression test FAILS (ValidationError)
  restore -> both PASS
```

## Evals run

```text
No eval harness covers the frontier curiosity evaluator. The live before/after
check is the query below, re-run after deploy: ontology_sparse_region count in
new candidate sets should be 0, and evaluator_task should no longer be a single value.
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-substrate-runtime build -> Image orion-substrate-runtime-substrate-runtime Built
grep in built image: core/schemas/frontier_curiosity.py has 0 occurrences; substrate/frontier_curiosity.py has
only the retirement comment.
NOT deployed.
```

Live evidence (read-only, 2026-10-10):

```text
psql conjourney: substrate_endogenous_curiosity_candidates, generated_at > now()-24h -> numbers above
FalkorDB GRAPH.RO_QUERY "MATCH (n:OntologyBranch) RETURN count(n)" -> 0 in all 14 graphs
  concept nodes: orion_substrate 919, orion_substrate_self 2758, orion_substrate_aitown 12253
```

Grep evidence of no producer: `OntologyBranchNodeV1` is defined in `orion/core/schemas/cognitive_substrate.py` and registered, mapped in `falkor_codec.py`, and listed in two `graph_cognition/views.py` predicates, but no production code constructs one.

## Review findings fixed

Review ran in a subagent against `git diff origin/main...HEAD`. It found no must-fix issues. It confirmed that no reader of stored data re-validates the retired values, that the phase-8 test failure already exists on main, and that the new regression test fails against main's evaluator.

- Finding: the code comment and the architecture doc point to this report, which had not been committed yet.
  - Fix: committed it in the same PR.
  - Evidence: this file.
- Finding: the outcome histogram will shift sharply after deploy and needs to be stated.
  - Fix: added "Expected after deploy" below.
  - Evidence: the worker always passes neutral metacog inputs, so the constant candidate was the only signal guaranteed to appear.
- Finding: the phase-8 test this PR touched stays red because of an unrelated assertion that was already failing.
  - Fix: marked it `xfail(strict=True)` with the reason, so it alerts as soon as someone fixes it.
  - Evidence: phase-8 file `6 passed, 1 xfailed`.
- Finding: the `ask_claude_trigger.py` docstring describes the constant as if it were still live.
  - Fix: added a note that it was retired, with a pointer to this report.
- Not fixed (nit): there is no worker-level test asserting that a tick with no seeds writes `candidates_json=[]` with `evaluator_outcome="noop"`. That needs a `worker.py` harness, and `worker.py` belongs to a parallel lane. Follow-up.

## Expected after deploy

The worker feeds the evaluator neutral metacog inputs, so contradiction, drift and pressure signals cannot fire from the tick. On ticks with no endogenous seeds:
- `gate_json.evaluator_outcome` goes from about 100% `invoke` to mostly `noop`, and `evaluator_task` becomes null.
- `candidates_json` is written as `[]`. That is a live tick with nothing worth curiosity, not a dead tick.
- The Hub observability panel shows `gap_count: 0`, the chat curiosity hint stops appearing, and the felt-state curiosity lane drops out.

When seeds exist, they now win on their own strength.

This is intended. It is not a regression.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && docker compose --env-file .env --env-file services/orion-substrate-runtime/.env -f services/orion-substrate-runtime/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: low
  - Concern: `tests/test_cognitive_substrate_phase8_frontier_invocation.py::test_signal_derivation_and_task_selection_are_deterministic` fails on origin/main too (its `evidence_gap_scan` assertion; the gap-marker hypothesis nodes are not in the curiosity seed slice). Not run by any CI workflow. Left as-is; not caused by this patch.
  - Mitigation: follow-up.
- Severity: low
  - Concern: `OntologyBranchNodeV1` / node kind `ontology_branch` is still a defined, registered substrate node type with no producer. Out of scope for this one-PR fix (it is a substrate schema type, not part of the candidate).
  - Mitigation: follow-up to retire the node kind or give it a real producer.
- Severity: info
  - Concern: with the junk candidate gone, the evaluator will invoke less often; `gate_json.evaluator_task` will show real tasks or null. Downstream readers that ranked the 1.0 item first will now see real candidates first.
  - Mitigation: intended.
- The `ontology_expand has no executor` finding (Orion's Day 09-29 audit, 4,798 decisions, orchestrator never instantiated) is resolved for this task type by retirement. The broader fact that no frontier plan of any type is executed remains.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2574

🤖 Generated with [Claude Code](https://claude.com/claude-code)
