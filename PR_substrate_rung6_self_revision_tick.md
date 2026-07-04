## Summary

- Rung 6 (metacognitive self-revision) now actually feeds sustained self-model prediction error into the live cognitive-lane mutation pipeline as governed `MutationSignalV1`s, closing the loop that rung 1's detector (`orion/substrate/mutation_self_revision.py`) had been building toward with no consumer.
- New hub-side adapter `_self_revision_signals_from_latest_self_state` (`services/orion-hub/scripts/api_routes.py`) loads the latest `SelfStateV1`, drops it if stale/missing/errored (fail-open to `[]`), and converts it to signals via the existing pure detector.
- `SubstrateAdaptationWorker.run_cycle` (`orion/substrate/mutation_worker.py`) gained an `extra_signals` kwarg so the scheduler can fold self-revision signals into the same cycle as telemetry-derived ones — reusing the unchanged `PressureAccumulator` → `ProposalFactory` → DRAFT pipeline, never auto-applying.
- Double-gated behind two new flags (`SUBSTRATE_AUTONOMY_SELF_REVISION_ENABLED`, opt-in default `false`, plus the existing `SUBSTRATE_AUTONOMY_COGNITIVE_PROPOSALS_ENABLED`), so this only activates with two independent operator opt-ins.
- **Fixed a review finding before commit**: `extra_signals` now share the operator's `max_signals` budget instead of an independent `[:8]` slice, so `SUBSTRATE_AUTONOMY_ROUTING_PROPOSALS_ENABLED=false` (the existing kill lever) also silences self-revision signals.

## Outcome moved

Rung 6 goes from "code exists, nothing calls it" to a live, flag-gated signal path with an end-to-end test proving a sustained self-model prediction-error tick drafts a reviewable `cognitive_identity_continuity_adjustment` proposal — never auto-applied.

## Current architecture

Before this patch: `prediction_error_mutation_signals()` was a fully-tested pure function with zero production callers. The only signal source feeding `PressureAccumulator`/`ProposalFactory` in the hub's scheduled cycle (`execute_substrate_mutation_scheduled_cycle`) was `MutationDetectors.from_review_telemetry`, which never reads `self_state.prediction_error_scores`.

## Architecture touched

- `orion/substrate/mutation_worker.py`: library-level cycle runner, `orion-substrate` package.
- `services/orion-hub/scripts/api_routes.py`: hub's mutation-autonomy scheduler (the actual production caller of the pressure/proposal pipeline — not `orion-substrate-runtime`, which only handles biometrics/attention/transport).
- Config: `services/orion-hub/.env_example`, `services/orion-hub/docker-compose.yml`.

## Files changed

- `orion/substrate/mutation_worker.py`: added `extra_signals` param to `run_cycle`, folded into the same `max_signals` budget as telemetry signals.
- `orion/substrate/tests/test_mutation_worker_extra_signals.py` (new): proves signal accumulation/threshold behavior, `None`-default no-op, and the budget-sharing fix (kill-lever bypass + shared-budget truncation regression tests).
- `services/orion-hub/scripts/api_routes.py`: `_self_revision_signals_from_latest_self_state` adapter + double-gated wiring into `execute_substrate_mutation_scheduled_cycle`, new `self_revision_enabled`/`self_revision_signals` scheduler summary fields.
- `services/orion-hub/tests/test_substrate_mutation_scheduler_runtime.py`: 5 new scheduler tests, including a regression test for the kill-lever fix (`test_scheduler_self_revision_respects_routing_proposals_kill_lever`).
- `services/orion-hub/.env_example`, `services/orion-hub/docker-compose.yml`: new keys, documented as opt-in/more-invasive in a comment.

## Schema / bus / API changes

- Added: none (reuses existing `MutationSignalV1` schema).
- Removed: none.
- Renamed: none.
- Behavior changed: `SubstrateAdaptationWorker.run_cycle` now accepts optional `extra_signals`; omitting it is fully backward-compatible (tested).
- Compatibility notes: additive, opt-in, no migration needed.

## Env/config changes

- Added keys: `SUBSTRATE_AUTONOMY_SELF_REVISION_ENABLED=false`, `SUBSTRATE_AUTONOMY_SELF_REVISION_MIN_ERROR=0.3`, `SUBSTRATE_AUTONOMY_SELF_REVISION_MAX_AGE_SEC=300`.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes (`services/orion-hub/.env_example`).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: the sync script's default allowlist doesn't cover `SUBSTRATE_AUTONOMY_*` keys and can't pull keys from an uncommitted branch's `.env_example` into the main checkout, so the same 3 keys with matching defaults were added manually to `/mnt/scripts/Orion-Sapienform/services/orion-hub/.env` (main checkout). Verified `git check-ignore` still holds and the file is not staged.
- skipped keys requiring operator action: none.

## Tests run

```text
/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest orion/substrate/tests services/orion-hub/tests/test_substrate_mutation_scheduler_runtime.py -q
168 passed, 18 warnings in 5.40s
```

## Evals run

None exist for `orion/substrate` or `services/orion-hub` (`evals/` directories absent in both). Follow-up: add a periodic eval measuring self-revision signal quality (e.g., false-positive rate on stable self-states) once real self-state telemetry is available to tune `SELF_REVISION_MIN_ERROR` against.

## Docker/build/smoke checks

Not run — no runtime behavior read at boot beyond the 3 new env-gated flags, which default to today's behavior (`false`). No code path executes without the double-gate (`SELF_REVISION_ENABLED` and `COGNITIVE_PROPOSALS_ENABLED`) flipped on.

## Review findings fixed

- Finding: `extra_signals` bypassed the `max_signals=0` operator kill lever via an independent `[:8]` cap, so disabling routing proposals didn't stop self-revision signals from reaching the pipeline.
  - Fix: `remaining_budget = max(0, self.budget.max_signals - len(signals))`; `extra_signals` truncated to what's left of the shared budget instead of an independent slice.
  - Evidence: new tests `test_run_cycle_extra_signals_respect_max_signals_kill_lever`, `test_run_cycle_extra_signals_share_budget_with_telemetry_signals`, and scheduler-level `test_scheduler_self_revision_respects_routing_proposals_kill_lever` — all pass.
- Finding: `.env_example` changed without local `.env` sync (CLAUDE.md Section 7 gate).
  - Fix: manually added matching keys to `services/orion-hub/.env` in the main checkout with the same defaults as `.env_example`.
  - Evidence: `grep SUBSTRATE_AUTONOMY_SELF_REVISION services/orion-hub/.env` shows all 3 keys; `.env` remains git-ignored and unstaged.
- Findings noted but not fixed (LOW, accepted as-is): `SelfStateV1` test-fixture builder now duplicated near-verbatim across 3 test files; inline self-state staleness check in `api_routes.py` duplicates the `_presence_row_fresh`/`_PRESENCE_MAX_AGE_SEC` pattern in `substrate_observability_routes.py`. Both are cosmetic/DRY only, not correctness issues; reuse would cross a module boundary for a ~5-line check, judged not worth the coupling for this patch size.
- Angle A (line-by-line diff scan) of the code-review pass was terminated early by a session limit in a prior finder run; redone by hand against the full diff (imports, budget arithmetic, tzinfo handling, double-gate logic, env defaults) — no additional bugs found beyond the one already fixed above.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-hub/.env -f services/orion-hub/docker-compose.yml up -d --build orion-hub
```

Flags default off, so no behavior changes until explicitly enabled.

## Risks / concerns

- Severity: LOW
  - Concern: no eval harness exists for this signal path; only unit/integration tests cover it.
  - Mitigation: add a small eval once there's real self-state telemetry to tune `SELF_REVISION_MIN_ERROR` against.
- Severity: LOW
  - Concern: triplicated test fixture and duplicated staleness-check pattern (see review findings above).
  - Mitigation: none applied this patch; candidate for a follow-up cleanup pass if a fourth copy shows up.

## PR link

No `gh` auth in this environment to open the PR programmatically. Branch is pushed; create the PR at:
https://github.com/junebug-junie/Orion-Sapienform/pull/new/feat/substrate-rung6-self-revision-tick
