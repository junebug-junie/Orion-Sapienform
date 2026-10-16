## Summary

- The world-pulse reading `look` act now opens once per read instead of on every poll of a queued read.
- The act opens when the read's durable run first reports real work started (`run.started`), dated at that moment, not when the run was submitted to the GPU queue.
- A read refused before any work (admission refusal, never started) is no longer an act: it neither opens nor closes.
- The reading horizon stays 3600 s, with its basis re-measured on the started -> completed clock.

## Outcome moved

Live, 2026-10-16, before this patch: 243 `act_opened` grammar events for 10 reading acts (10 to 43 per act). The ledger graded reading at 1 of 10 (`outcome_rate 0.1`).

That 10% was not real. All 7 reads that finished produced a handoff (`world_pulse_read_seed.status = done`, no `last_error`); 3 are still pending. 6 of the 7 were graded 0 as `expired` because each one waited 0.5 to 3.3 h for a GPU before it started (durable-runs history: `run.waiting_resource` 12:03 -> `run.started` 15:17), then read in 4 to 12 minutes. The writer's 1 h horizon ran from submission, so it timed the queue. The producer close (`stage1_completed`, value 1) arrived after the writer had already expired the act; first close wins.

Failure breakdown of the 10 graded acts: 0 real read failures, 6 queue-late successes, 3 still queued (expired while waiting), 1 on-time success.

After: one open per read, dated at work start, so the same 7 reads would each be graded on their 4-12 minute read time.

## Current architecture

`services/orion-hub/scripts/world_pulse_read_pipeline.py::_read_claimed` called `publish_read_opened` before every `_stage1_read`. A queued read is re-claimed every tick, polled (`poll_turn` -> `ReadingPending`), and released, so every tick re-opened the act. The writer (`orion/grammar/act_ledger.py`) keeps the first open, so `opened_at` was the submission time. An admission refusal (`turn_deferred:reading_admission`) also opened at submission and, with no retry left, closed at 0.

## Architecture touched

- `orion/world_pulse_read/durable.py`: `poll_turn(..., on_state=)` reports every verified durable status to the caller (observer errors swallowed). New `work_started_at(state)` reads the first `run.started` from the run history (fallback: `created_at + queue_wait_seconds`).
- Hub reading pipeline: in-process per-seed bookkeeping (`_note_read_state`, `_open_read_act`, `_close_read_act`, `_forget_read_act`). Opens in a `finally` right after `_stage1_read`, so the open is always published before any close on the same tick. Closes only acts this process opened.
- `act_emit.publish_read_opened(..., at=)` passes the start time to `act_opened_event`.

Why in-process and not a DB marker (the painting fix in #2724 used a row marker): the durable run is already the persistent truth of "work started", and every poll re-reports it. After a hub restart, the first poll re-opens with the same start time, which the writer ignores (first open wins). Cost: at most one extra `act_opened` per in-flight read per hub restart, and no migration.

## Files changed

- `orion/world_pulse_read/durable.py`: `on_state` observer and `work_started_at`.
- `orion/world_pulse_read/tests/test_durable.py`: tests for both.
- `services/orion-hub/scripts/world_pulse_read_pipeline.py`: open once at work start, close only opened acts.
- `services/orion-hub/scripts/act_emit.py`: `at=` on the open; horizon basis and docstring re-measured.
- `services/orion-hub/tests/test_read_act_open_once.py`: new regression tests through the real `_read_claimed -> _generate -> poll_turn` seam.
- `services/orion-hub/tests/test_act_ledger_producers.py`, `services/orion-hub/tests/test_reading_turn_listener.py`: updated for the new contract.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: `hub.world_pulse_read` `look` act `opened_at` is now the read's work start, not its submission. Admission refusals produce no act. `expect_basis` text changed.
- Compatibility notes: no schema change. Stage 2 calls `poll_turn` without `on_state` and is unaffected.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- skipped keys requiring operator action: none.

## Horizon basis (metric quality gate)

1. Provenance: `durable_resource_events` `run.started` -> `run.completed` for `reading-%` runs (the same events the durable-runs status `history` returns to the hub).
2. Independence: the outcome is the producer close; the clock is the durable run's own start event. Queue wait is still recorded per run (`queue_wait_seconds`) and is no longer folded in.
3. Anchor: "a read finishes in the time Orion's own reads usually take" (p90 rounded up to the hour, unchanged rule).
4. Live data, 30 days, n=386: p50 484 s, p90 1840 s, p99 3757 s. The submission clock used before: p50 653 s, p90 3153 s over the same window, and on 2026-10-16 alone p50 6439 s.
5. Existing mechanism: reused durable-runs history; no new store.
6. Reversible: revert the patch; no schema or table change.

## Tests run

```text
services/orion-hub: pytest tests/test_read_act_open_once.py tests/test_act_ledger_producers.py tests/test_reading_turn_listener.py  -> 25 passed
services/orion-hub: pytest tests -k "read or act_ledger or reading or world_pulse"  -> 721 passed, 85 skipped, 1 failed
  (test_fcc_model_labels_api.py::test_fcc_model_labels_reads_fixture_env: fails identically on main, env-sensitive, unrelated)
pytest orion/world_pulse_read/tests orion/grammar tests/test_grammar_event_producer_catalog.py  -> 215 passed
New tests on the old code (detached worktree at the parent commit): 3 failed
  (5 opens for 1 act; refusal opened an act; failing poll path)
```

## Evals run

```text
None. The act ledger itself is the eval surface; no reading-act eval harness exists.
Follow-up: a replay over grammar_acts + durable_resource_events that recomputes reading grades on the work-start clock.
```

## Docker/build/smoke checks

```text
Not run: hub code-only change, no compose/env/dependency change. Live path UNVERIFIED until deployed.
Post-deploy check (should be 1 open per act, opened_at = the run's first run.started):
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select trace_id, count(*) from grammar_events where source_component='world_pulse_read.act' and event_kind='act_opened' and emitted_at > now() - interval '6 hours' group by 1"
```

## Review findings fixed

Code review ran in a subagent against `git diff origin/main...HEAD`. No blockers.

- Finding: a failed open publish still marked the seed opened, so a later close went out with no open and the writer would create and grade the act from the close alone.
  - Fix: mark opened only when `publish_read_opened` returns True; the next poll retries the open.
  - Evidence: `test_lost_open_is_retried_and_never_followed_by_an_orphan_close`, `test_lost_open_with_no_retry_sends_no_close`.
- Finding: an operator cancel first seen together with its start opened an act and dropped it, so it would expire at 0.
  - Fix: `_note_read_state` ignores a `cancelled` state. An act opened on an earlier poll still expires at 0 (documented).
  - Evidence: `test_cancel_first_seen_with_its_start_opens_nothing`.
- Finding: the pipeline tests used a scripted `poll_turn`, so nothing pinned that the real one reports a terminal state to the observer before it returns or raises.
  - Fix: `test_poll_reports_terminal_states_before_returning_or_raising` (completed and failed) against the real `poll_turn`; test docstring corrected.
- Finding: the open ran in a `finally` with an unguarded import, which could mask the read's own exception.
  - Fix: `_open_read_act` never raises (guarded, logged).
- Finding: per-seed bookkeeping was not cleared on the bad-URL / index / reader-blocked exits.
  - Fix: those exits call `_close_read_act` (closes at 0 with the reason if an earlier attempt opened the act, otherwise just forgets the seed).
- Finding: the basis text said "GPU queue wait excluded", but a retried read's re-queue still counts.
  - Fix: basis text now says so explicitly.
- Finding: state lived in `__dict__.setdefault`, hidden from `__init__`.
  - Fix: initialised in `__init__`; a `hasattr` fallback stays for tests built with `object.__new__`.
- Not fixed (low): `work_started_at` takes the minimum `run.started` from the status `history`, which durable-runs caps at the newest 200 events. A very busy run could lose its first start from that window. Exact fix is to expose `first_event_at(run.started)` in the durable-runs status; left out to keep this patch inside the hub/reading seam.

## Restart required

```bash
./scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: medium
  - Concern: the live posterior `substrate_action_effect_posterior (act:look, hub.world_pulse_read, artifact_produced)` already holds n=10, mean 0.136, graded on the queue clock. New acts will update from that biased prior.
  - Mitigation: not touched (production write). Juniper decides whether to reset that one row.
- Severity: low
  - Concern: after a hub restart, a read whose earlier attempt started and failed with a retry, and whose retry is then refused before work, publishes no close. The act expires at 0 at its horizon.
  - Mitigation: that is a true miss (nothing produced by the horizon); documented in code.
- Severity: low
  - Concern: acts already open at deploy keep their submission-time `opened_at` (first open wins), so the first hours after deploy still show queue-late grades.
  - Mitigation: judge the fix only on acts opened after the restart.
- Severity: low
  - Concern: GPU queue wait is no longer in this act's grade.
  - Mitigation: it is still recorded per run (`queue_wait_seconds`). It is a GPU-pool fact, not a reading fact.

## PR link

(filled in after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
