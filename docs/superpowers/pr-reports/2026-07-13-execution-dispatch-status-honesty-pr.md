# fix(execution-dispatch): un-lie dispatch_status (P0 of the motor-nerve spec)

## Summary

- `orion/execution_dispatch/builder.py` claimed `dispatch_status="dispatched"` for read-only candidates while never sending anything — no executor exists anywhere in the repo. Now emits the honest `prepared_for_dispatch`.
- `ExecutionDispatchCandidateV1.dispatch_status="dispatched"` now requires evidence (`dispatched_at` + `result_ref`/`dispatch_error`) via a model validator — the status can no longer be claimed without proof a send was attempted.
- `orion/feedback/builder.py` classifies/scores the new status; 6 existing test fixtures that simulated a genuinely-dispatched candidate were repaired with the now-required evidence fields.
- Code review caught that the new validator could raise on historical Postgres rows predating it. Live-queried `substrate_execution_dispatch_frames` directly (687,992 rows, 0 affected) to confirm today's data is safe, then added the same `try/except ValidationError → None` guard this codebase already uses for an identical 2026-07-12 self-state incident, at all 4 affected loaders.

## Outcome moved

Layer 9's status vocabulary no longer lies. `dispatch_status="dispatched"` is now a claim that requires evidence to make; `prepared_for_dispatch` is the honest name for "cleared every gate, never sent." This is P0 of `docs/superpowers/specs/2026-07-13-endogenous-action-motor-nerve-spec.md` — the precondition for P1 (an actual sender) to exist without inheriting a broken status contract.

## Current architecture

`build_execution_dispatch_frame` (Layer 9) converts approved policy decisions into `ExecutionDispatchCandidateV1` envelopes. For `dispatch_mode="dispatch_read_only"` + `allow_dispatch_read_only=True`, it previously set `dispatch_status="dispatched"` despite only constructing a request-envelope dict — no bus publish, no cortex-exec call, nothing sent. Live policy default (`allow_dispatch_read_only: false`) meant this was unreachable in production, but was directly exercised by test fixtures and would have misled any future consumer.

## Architecture touched

`orion/schemas/execution_dispatch_frame.py`, `orion/schemas/feedback_frame.py`, `orion/execution_dispatch/builder.py`, `orion/feedback/builder.py`, plus the 4 Postgres-backed loaders of `ExecutionDispatchFrameV1` in `services/orion-execution-dispatch-runtime/app/store.py`, `services/orion-feedback-runtime/app/store.py`, and `services/orion-hub/scripts/substrate_execution_dispatch_routes.py`.

## Files changed

- `orion/schemas/execution_dispatch_frame.py` — `prepared_for_dispatch` status literal; `result_ref`/`dispatch_error`/`dispatched_at` fields; evidence-requiring validator.
- `orion/schemas/feedback_frame.py` — `prepared_for_dispatch` added to `OutcomeObservationV1.outcome_kind`.
- `orion/execution_dispatch/builder.py` — one-line status fix; comment marking the now-unreachable `max_dispatches_per_tick` branch.
- `orion/feedback/builder.py` — classification + scoring for `prepared_for_dispatch`.
- `services/orion-execution-dispatch-runtime/app/store.py`, `services/orion-feedback-runtime/app/store.py`, `services/orion-hub/scripts/substrate_execution_dispatch_routes.py` — degrade-to-`None` guards on `ExecutionDispatchFrameV1.model_validate()`.
- `services/orion-execution-dispatch-runtime/README.md` — documents the status vocabulary.
- `tests/test_execution_dispatch_frame_schemas.py`, `tests/test_execution_dispatch_builder.py`, `tests/test_feedback_builder.py`, `tests/test_execution_dispatch_runtime_store.py`, `tests/test_feedback_runtime_store.py` — new/repaired coverage.
- `docs/superpowers/specs/2026-07-13-execution-dispatch-status-honesty-design.md` — design doc for this patch.

## Schema / bus / API changes

- **Added:** `dispatch_status="prepared_for_dispatch"`; `ExecutionDispatchCandidateV1.result_ref`/`dispatch_error`/`dispatched_at` (all optional).
- **Removed:** none.
- **Renamed:** none.
- **Behavior changed:** `dispatch_status="dispatched"` now raises `ValidationError` unless evidenced. The `dispatch_read_only` builder branch now produces `prepared_for_dispatch`, so `dispatched_candidates`/`dispatch_count` are honestly empty/zero until a future sender exists (this also makes the `max_dispatches_per_tick` rate-limit branch dead code for now — commented, not removed).
- **Compatibility notes:** live-verified via direct Postgres query — 687,992 rows in `substrate_execution_dispatch_frames`, 0 with a nonempty `dispatched_candidates` list — no historical row is broken by the new validator. Defensive guards added regardless, matching this codebase's established pattern.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not applicable, no env template changed.
- skipped keys requiring operator action: none.

## Tests run

```text
cd /mnt/scripts/Orion-Sapienform-execution-dispatch-status-honesty
/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest \
  tests/test_execution_dispatch_frame_schemas.py tests/test_execution_dispatch_envelopes.py \
  tests/test_execution_dispatch_builder.py tests/test_execution_dispatch_policy_loader.py \
  tests/test_execution_dispatch_runtime_store.py tests/test_execution_dispatch_runtime_worker.py \
  tests/test_execution_dispatch_transport_dry_run.py tests/test_feedback_builder.py \
  tests/test_feedback_runtime_store.py tests/test_feedback_transport_outcomes.py \
  tests/test_consolidation_tensorize.py tests/test_consolidation_schema_candidates.py \
  tests/test_consolidation_motif_detection.py tests/test_consolidation_expectations.py \
  tests/test_consolidation_policy_loader.py services/orion-hub/tests/test_substrate_execution_dispatch_debug_api.py -q

93 passed in 2.20s
```

## Evals run

None — this service has no `evals/` harness; not required for a schema-honesty patch with no quality/behavior dimension to measure.

## Docker/build/smoke checks

Not run — no runtime behavior, port, dependency, or compose wiring changed. Compatibility was instead verified with a direct live Postgres query (see below) rather than a container smoke, since the risk was schema/data compatibility, not runtime wiring.

```text
PGPASSWORD=postgres psql -h localhost -p 55432 -U postgres -d conjourney -c "
select count(*) as total_rows,
       count(*) filter (where dispatch_frame_json::jsonb -> 'dispatched_candidates' <> '[]'::jsonb) as rows_with_nonempty_dispatched_candidates
from substrate_execution_dispatch_frames;"
 total_rows | rows_with_nonempty_dispatched_candidates
------------+------------------------------------------
     687992 |                                        0
```

## Review findings fixed

- Finding: New evidence-requiring validator has no guard on 4 `model_validate()` loaders of historical `ExecutionDispatchFrameV1` rows — same failure class as a 2026-07-12 self-state incident already fixed elsewhere in this codebase, worst case a permanently-wedged feedback FIFO.
  - Fix: Live-queried the actual table (687,992 rows, 0 affected today) for ground truth, then added `try/except ValidationError → logger.warning → return None` at all 4 sites, matching the established `load_self_state` pattern.
  - Evidence: 4 new regression tests (`test_load_latest_dispatch_frame_degrades_to_none_on_legacy_incompatible_row` etc.) constructing a legacy-shaped payload and asserting `None` instead of a raise; all pass.
- Finding: The `max_dispatches_per_tick` rate-limit branch in `builder.py` becomes permanently unreachable once nothing sets `dispatch_status="dispatched"` from this function — a real but deliberate consequence, undocumented.
  - Fix: Added an inline comment naming exactly why and when it becomes reachable again (a future sender).
  - Evidence: `orion/execution_dispatch/builder.py:223-227`.
- Finding: `test_failed_cortex_result` paired `dispatch_error` (send itself failed) with a `cortex_results` entry claiming `status: "failed"` (send succeeded, cortex responded with failure) — two contradictory failure stories in one fixture.
  - Fix: Switched to `result_ref`, matching the other 5 repaired fixtures' coherent "send succeeded, then outcome determined" story.
  - Evidence: `tests/test_feedback_builder.py::test_failed_cortex_result`, still asserts `outcome_status == "failed"`, now for the right reason.

## Restart required

```text
No restart required.
```

No runtime config changed; this is a pure code patch. The next deploy of `orion-execution-dispatch-runtime`, `orion-feedback-runtime`, and `orion-hub` will pick it up normally.

## Risks / concerns

- Severity: low
- Concern: P1 (the actual sender) doesn't exist yet, so `prepared_for_dispatch` candidates will accumulate with no promotion path until that patch lands. Not a regression — this is the same "nothing is dispatched" state as before, just honestly labeled now.
- Mitigation: none needed; this is explicitly the intended P0 end-state per the parent spec.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/execution-dispatch-status-honesty-p0
