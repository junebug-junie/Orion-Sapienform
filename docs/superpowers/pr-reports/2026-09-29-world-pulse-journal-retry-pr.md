## Summary

- The daily world-news journal (and its email) is written once, when the world-pulse run finishes. If writing it failed because the GPU lane was busy, nothing ever tried again, so the email silently stopped after 2026-09-24.
- A failed attempt that is worth retrying now goes into a small queue file that survives restarts (`pending_journals.json`, next to `scheduler_cursors.json`). The scheduler retries after 5, 15, 45, then every 120 minutes.
- It gives up after 12 hours (configurable), or if a retry would land after local midnight. Giving up emits audit action `world_pulse_journal_gave_up` and a log line.
- The queue file remembers which runs already produced a journal (for 7 days), so a retry or a redelivered run result never writes a second `world_pulse_digest` entry.
- `_dispatch_journal` gains an optional `on_failure` callback. It fires only when writing the journal failed *before* the write was published. Every other caller behaves exactly as before.

## Outcome moved

Failure mode: a congested fast GPU lane (`gpu_pool_unavailable:deadline`) at the 06:00 Denver run permanently dropped that day's world-news journal and email. After this patch the same failure is retried within the same local day.

## Current architecture

`orion:world_pulse:run:result` → `handle_world_pulse_run_result_journal` → `_dispatch_journal` → `_run_journal` (a request to cortex to write the journal via `quick_background`, then publishes `journal.entry.write.v1`) → SQL persist → a separate step sends the email, capped at one per local day for `world_pulse_digest`. `_dispatch_journal` returned False for disabled, cooldown and error alike, and the error was only logged.

## Architecture touched

orion-actions only: the world-pulse journal handler, the `_dispatch_journal` / `_run_journal` closures, `_scheduler_loop` (starts one background retry at a time), and a new durable store. No bus, schema or channel changes.

## Files changed

- `services/orion-actions/app/pending_journal_store.py`: new restart-durable queue plus a record of completed runs (atomic write, a corrupt file is set aside instead of overwritten).
- `services/orion-actions/app/world_pulse_journal.py`: decides which errors are worth retrying, puts failures in the queue, drains the queue, starts the single background drain.
- `services/orion-actions/app/main.py`: `on_failure` + a publish-attempted flag in `_dispatch_journal`/`_run_journal`, store wiring, drain started from the scheduler loop.
- `services/orion-actions/app/settings.py`, `.env_example`, `README.md`: two new env keys, documented.
- `scripts/sync_local_env_from_example.py`: added prefix `ACTIONS_WORLD_PULSE_JOURNAL_RETRY_`. No existing prefix matched, so the env sync would otherwise have silently skipped the new keys.
- `services/orion-actions/tests/test_world_pulse_reflective_journal_handler.py`, `test_handle_envelope_world_pulse_journal.py`: tests (listed below).

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: a world-pulse journal failure that is worth retrying gets retried. New audit statuses and actions on `orion:actions:audit`: `retry_scheduled` (action `journal.world_pulse_digest`) and `world_pulse_journal_gave_up`.
- Compatibility notes: other `_dispatch_journal` callers are unchanged, because `on_failure` defaults to None.

## Env/config changes

- Added keys: `ACTIONS_WORLD_PULSE_JOURNAL_RETRY_ENABLED` (true), `ACTIONS_WORLD_PULSE_JOURNAL_RETRY_MAX_AGE_HOURS` (12)
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: yes
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes. Both keys were added to `/mnt/scripts/Orion-Sapienform/services/orion-actions/.env`.
- skipped keys requiring operator action: none

## Tests run

```text
pytest services/orion-actions/tests -q            -> 215 passed
pytest tests/scripts/test_sync_local_env_from_example.py scripts/tests/test_check_env_template_parity.py -> 19 passed
Mutation checks (by hand):
  removing the on_failure call in _dispatch_journal -> real-dispatch test fails
  removing the publish-attempted guard              -> publish-timeout test fails
```

Covered by tests: a failure is queued with backoff; retry-worthy vs not-retry-worthy errors; cooldown or disabled journaling does not queue; the drain retries until it succeeds and never writes twice; a redelivered run is skipped; the queue reloads after restart; give-up after max age; give-up when a retry would cross local midnight; a reschedule is audited; an error writing the queue file does not escape the handler; a corrupt queue file is set aside; only one drain runs at a time; a static check that `_scheduler_loop` calls the drain; the real `_dispatch_journal` driven through a compose timeout (queued), journaling disabled (not queued), and a write-publish timeout (not queued).

## Evals run

```text
None. orion-actions has no eval harness for journal dispatch. This is deterministic retry logic, covered by gate tests.
Follow-up: live check after deploy (see Risks).
```

## Docker/build/smoke checks

```text
Not run. Deploy/restart was out of scope for this task.
Static gates: check_metric_lineage --gate PASS, check_definition_drift --gate PASS, check_env_template_parity PASS,
check_journal_dispatch_registry OK, check_async_routes_not_blocking OK, check_chat_route_poachers PASS,
check_sentience_instruments --static-only OK, check_compose_no_relative_mounts PASS, check_service_hostname_refs OK.
```

## Review findings fixed

- Finding (SHOULD): a timeout while publishing the journal write could be queued even though the write had already landed, and the retry would then write a duplicate.
  - Fix: `_run_journal` sets `progress["write_publish_attempted"]` before publishing. After that point `_dispatch_journal` suppresses `on_failure`.
  - Evidence: `test_real_dispatch_journal_reports_compose_error_and_enqueues_retry` (the wp-real-3 case) fails when the guard is removed.
- Finding (SHOULD): the drain blocked the scheduler loop for up to the full exec timeout.
  - Fix: `start_world_pulse_retry_drain` runs the drain in one background task at a time.
  - Evidence: `test_start_drain_runs_one_at_a_time`.
- Finding (SHOULD): the list of retry-worthy errors missed some GPU-pool failures.
  - Fix: added `gpu_pool_recalled`, `pool_unreachable`, `gateway_capacity_rejected`.
  - Evidence: `test_retryable_error_classification`.
- Finding (SHOULD): no test checked that the scheduler actually starts the drain.
  - Fix: static AST test on `_scheduler_loop`.
  - Evidence: `test_scheduler_loop_starts_world_pulse_retry_drain`.
- Finding (SHOULD): a disk error writing the queue file could escape the bus handler.
  - Fix: try/except with log `world_pulse_journal_retry_store_write_failed`.
  - Evidence: `test_store_write_failure_does_not_escape_handler`.
- Finding (SHOULD): a retry landing after local midnight would spend the next day's one email.
  - Fix: give up with `crossed_local_day`. Documented in the README.
  - Evidence: `test_gives_up_when_retry_would_cross_local_day`.
- NITs fixed: reschedules are now audited; the retry envelope is built only after the already-written check; completion time is read after dispatch; a corrupt queue file is set aside rather than overwritten.
- NIT not fixed: a redelivered run result that arrives after a give-up starts a new 12h window. Narrow, and pub/sub rarely redelivers.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform-world-pulse-journal-retry   # or main after merge
scripts/safe_docker_build.sh orion-actions up -d --build
```

## Risks / concerns

- Severity: low. Concern: if the process crashes in the few milliseconds between a successful write and recording the run as completed, a restart could retry and write a duplicate entry. Mitigation: the window is tiny and the email cap still limits sends to one per day, except that the in-memory cap is also reset by a restart.
- Severity: low. Concern: the live path is UNVERIFIED. No run has been retried yet. Mitigation: after deploy, when a `gpu_pool_unavailable` failure occurs, look for a `world_pulse_journal_retry_enqueued` log line, then `world_pulse_journal_retry_succeeded`, then a `world_pulse_digest` journal row for that run_id.
- Severity: low. Concern: the retry does not make the GPU lane less busy at 06:00. It only works around it. Mitigation: out of scope; the lane route (`ACTIONS_JOURNAL_LLM_ROUTE`) could be revisited separately.

## PR link

(filled in after PR creation)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
