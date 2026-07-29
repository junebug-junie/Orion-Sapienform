## Summary

- `services/orion-actions`'s workflow scheduler (the daily 06:00 `chat_history_compactor` digest and every other scheduled workflow it drives) could never report a confirmed success: `mark_dispatch_succeeded()` existed but was never called anywhere in the repo, so every dispatched run sat forever at `status="dispatched"`.
- `_derive_analytics()`'s health formula treated "0 confirmed successes, 0 confirmed failures" as `health="healthy"` — the exact bug a user spotted in the Hub UI: `"healthy"` next to `"0/5 recent succeeded"`.
- `_dispatch_scheduled_workflow` now blocks on an RPC reply from cortex-orch (same pattern already used by `_run_journal`/`_dispatch_scheduled_skill(wait_for_result=True)` in the same file) instead of firing the trigger and forgetting it, and threads `run_id`/`schedule_id` into the dispatch payload for correlation.
- The scheduler's claim loop now calls `mark_dispatch_succeeded`/`mark_dispatch_failed` based on the real RPC outcome, and no longer re-raises past one failed dispatch (previously a single bad dispatch could abort the rest of the batch and skip `evaluate_attention_signals` for every sibling schedule in the same tick).
- Health formula hardened: a schedule with recent runs but zero confirmed successes now reports `"degraded"`, not `"healthy"`.

## Outcome moved

The scheduler can now actually distinguish "this workflow ran and succeeded" from "we published a trigger and have no idea what happened" — a distinction it never made before. The Hub schedule panel's health badge and `N/5 recent succeeded` trend text will now agree with each other.

## Current architecture

`services/orion-actions/app/workflow_schedule_store.py` (`WorkflowScheduleStore`) persists `WorkflowScheduleRecordV1`/`WorkflowScheduleRunRecordV1` rows and derives a `WorkflowScheduleAnalyticsV1` (health/trend) on read. `services/orion-actions/app/main.py`'s `_scheduler_loop` (inside `lifespan`) calls `claim_due()` on a 45s tick, dispatches each due schedule to `orion-cortex-orch` over the bus, and was supposed to record the outcome back onto the run — but the success half of that loop was never wired.

## Architecture touched

`services/orion-actions` only — no cross-service contract change. No new bus channel, no new schema, no new env key.

## Files changed

- `services/orion-actions/app/main.py`: `_dispatch_scheduled_workflow` now does an RPC round-trip (reply_to + `_actions_rpc_bus.rpc_request`, timeout = `settings.actions_exec_timeout_seconds`) instead of a fire-and-forget publish, and threads `run_id`/`schedule_id` into `workflow_request["scheduled_dispatch"]`. The claim loop in `_scheduler_loop` now calls `mark_dispatch_succeeded` on success and `continue`s (rather than `raise`s) past a failed dispatch after calling `mark_dispatch_failed`.
- `services/orion-actions/app/workflow_schedule_store.py`: `_derive_analytics()`'s degraded-health condition gained `or len(success) == 0`, so "no confirmed success among recent runs" is no longer indistinguishable from "healthy".
- `services/orion-actions/tests/test_workflow_schedule_store.py`: two new regression tests — `test_stuck_dispatched_run_is_not_reported_healthy` (verified red against pre-fix `_derive_analytics`) and `test_confirmed_success_reports_healthy`.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: `_dispatch_scheduled_workflow` now blocks on a bus RPC reply instead of a non-blocking publish; `WorkflowScheduleAnalyticsV1.health` can now read `"degraded"` in a case that previously read `"healthy"`.
- Compatibility notes: no payload shape changes; `scheduled_dispatch` gained two new keys (`schedule_id`, `run_id`) that no existing consumer reads yet, so this is additive.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: not applicable, no new keys.
- local `.env` synced: not applicable.
- skipped keys requiring operator action: none.

## Tests run

```text
/mnt/scripts/Orion-Sapienform/orion_dev/bin/python -m pytest services/orion-actions/tests -q --tb=short
102 passed, 68 warnings in 3.97s
```

Ran from the fix worktree using the main checkout's `orion_dev` venv (no venv exists per-worktree; `scripts/test_service.sh` documents this as the expected fallback path). Also hand-verified the two new tests fail against the pre-fix `_derive_analytics` (stashed the store.py change, reran, confirmed `AssertionError: assert 'healthy' != 'healthy'`, then restored the fix) before treating them as real regression coverage.

## Evals run

None — this service has no eval harness (`services/orion-actions/evals/` does not exist) and the change is a scheduler-bookkeeping bug fix, not a quality/behavior surface an eval would meaningfully score. Gate tests above are the appropriate coverage for this seam.

## Docker/build/smoke checks

Not run. This changes runtime dispatch behavior (an RPC round-trip replaces a fire-and-forget publish) inside `orion-actions`'s scheduler loop, which only fires on the real 45s scheduler tick against a live bus and a live `orion-cortex-orch` — not exercisable via `docker compose config` or a static smoke. The nested closure this touches (`_dispatch_scheduled_workflow`, inside `lifespan`) is imported and its containing `lifespan` function is exercised (compiled + partially invoked) by an existing test (`test_journal_actions.py::test_action_dedupe_try_acquire_is_thread_safe_under_concurrent_callers`, which starts `_scheduler_loop`), confirming no syntax/import errors, but that test does not drive a real claim/dispatch/RPC cycle. Recommend a live smoke on next restart: trigger a one-shot scheduled workflow, confirm the Hub schedule panel shows a `"completed"` run and `health="healthy"` after it finishes.

## Review findings fixed

- Finding: Sequential blocking dispatch could stall the scheduler tick for up to ~40 minutes (`batch_size=10 × timeout=240s`) under a cortex-orch outage, delaying every other daily-cadence check in the same loop iteration (goal archive, daily pulse, world pulse, metacog, journal) and re-claiming of newly-due schedules.
  - Fix: not fixed in this patch — documented below as a known, disclosed risk rather than solved with added concurrency (`asyncio.gather`) whose thread-safety against the shared `_actions_rpc_bus` wasn't independently verified for concurrent in-flight RPCs. Deferred as a follow-up.
  - Evidence: reviewer traced `settings.actions_exec_timeout_seconds` (240.0s default, `settings.py:90`) and `settings.actions_workflow_schedule_claim_batch_size` (10 default, `settings.py:164`) against the sequential `for claimed in ...` loop.
- Finding: The new `len(success) == 0` degraded-health branch also fires during the normal in-flight window of every routine dispatch (between `claim_due()` persisting `status="dispatched"` and the RPC reply landing), not only genuinely stuck runs.
  - Fix: not fixed — confirmed cosmetic-only. `evaluate_attention_signals()` only treats `"degraded"` as an active attention condition when `recent_failure_count >= 2`, which is `0` during this window, so no false notification/page fires. Left as documented behavior rather than adding a separate `"pending"` health state (would require touching the frontend's health enum and the schema's `Literal[...]`, disproportionate to a cosmetic dashboard flash).
  - Evidence: reviewer traced `evaluate_attention_signals()` (`workflow_schedule_store.py` ~280) and confirmed the `recent_failure_count >= 2` gate.

## Restart required

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-actions/.env \
  -f services/orion-actions/docker-compose.yml \
  up -d --build orion-actions
```

## Risks / concerns

- Severity: Moderate.
  Concern: cortex-orch slowness/outage now delays the scheduler's own tick (up to ~40 min worst case) instead of the old fire-and-forget publish, which was near-instant regardless of downstream health.
  Mitigation: in practice, claimed batches for this scheduler are typically size 1 (daily-cadence jobs rarely stack 10-deep), so worst-case exposure is uncommon. A follow-up could dispatch claimed items concurrently via `asyncio.gather`, but that needs the shared `_actions_rpc_bus`'s concurrent-RPC safety verified first rather than assumed.
- Severity: Low.
  Concern: schedule health can show a transient `"degraded"` for the ~seconds-to-minutes an RPC is in flight, even on a run that ultimately succeeds.
  Mitigation: confirmed no false attention/notification fires from this (gated at `recent_failure_count >= 2`); purely a dashboard-view cosmetic.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/workflow-schedule-completion-status
