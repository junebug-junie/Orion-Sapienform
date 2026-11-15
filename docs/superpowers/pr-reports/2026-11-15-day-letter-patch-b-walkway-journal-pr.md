# Walkway journal: a retry cap, a night that outlives midnight, and titles (Patch B of PR #3439)

Orion's day letters for 2026-11-12 and 2026-11-13 reported three faults in the nightly walkway
journal. This patch fixes them. Fixes #3384, #3385, #3386.

## Summary

- A failed walkway journal entry is now tried at most 4 times a night for each job. It was
  tried every 45 seconds with no limit, and each try opened a new graded act.
- The waits between tries are 1, 5 and 15 minutes. The count is kept in a file, so a restart
  does not reset it.
- A night that is still running at local midnight now finishes. It keeps its own date and
  reads its own day. Before, it was dropped at midnight.
- When a night closes, Orion writes one record for each job: did the entry land, yes or no,
  and why.
- A journal entry always has a title. When the model gives none, the code builds one from
  the kind and the date, for example `Walkway day · 2026-11-14`.

## Outcome moved

- Failed acts from one bad dependency: was 59 for `walkway_day` in 81 minutes on 11-13. Now
  at most 4 for each job each night.
- Nights lost at midnight: 11-12 lost its day and grade entries. A night now stays open up to
  6 hours after its scheduled time.
- Nights with no entry and no trace: now each night leaves one yes/no record for each job.
- Walkway entries with no title: was 13 of 15 since 11-10. Now none can be stored without one.

## Current architecture

Before this patch:

- The orion-actions scheduler loop ticks every 45 seconds (`services/orion-actions/app/main.py`).
- The walkway block ran when the local time passed 22:30 and the `walkway_journal` cursor was
  not today's date. At local midnight "today" changed, so an unfinished night stopped.
- `run_walkway_tick` (`app/walkway_forecast.py`) sent each job to `_dispatch_journal`. A failed
  dispatch kept the night open with no cap. Only failed database reads were capped, at 3.
- `_dispatch_journal` opened a new act for every try and returned only true or false.
- `build_write_payload` (`orion/journaler/worker.py`) stored the model's title as given. The
  compose prompt lets the model answer `null`.
- `DailyAttemptGate` (`app/daily_retry_gate.py`) already capped retries for the daily pulse and
  daily metacog jobs, with state in `daily_attempts.json`. The walkway jobs did not use it.

## Architecture touched

- One service: orion-actions. One shared package function: `orion/journaler/worker.py`.
- No new service, bus channel, table, schema or env key.
- `DailyAttemptGate` gets an optional `AttemptPolicy`: a flat cap where every failure counts
  the same. The daily jobs pass no policy and behave as before.
- The walkway scheduler block moved out of the loop body into a closure,
  `_walkway_journal_tick`, so tests can drive the real path (`app.state.walkway_journal_tick`).
- `_dispatch_journal` is now a thin wrapper over `_dispatch_journal_attempt`, which says
  whether a compose ran. Other callers of `_dispatch_journal` get the same true or false.

## Files changed

- `services/orion-actions/app/daily_retry_gate.py`: `AttemptPolicy`, optional `policy` on
  `may_attempt` and `record_failure`, `last_failure_reason`.
- `services/orion-actions/app/walkway_forecast.py`: the cap (`WALKWAY_ATTEMPT_POLICY`), the
  night rule (`walkway_night`, `WALKWAY_NIGHT_MAX_OPEN`), the scheduler pass
  (`run_walkway_schedule`), the landed records (`night_records`), and `night=` on
  `collect_walkway_jobs` so a late night reads its own day.
- `services/orion-actions/app/main.py`: `_dispatch_journal_attempt`, `_walkway_journal_tick`,
  the record writer (audit event, log line, file).
- `orion/journaler/worker.py`: `code_built_title`, used by `build_write_payload` when the
  draft title is empty.
- `services/orion-actions/tests/test_walkway_journal_night.py`: new, 21 tests.
- `orion/journaler/tests/test_code_built_title.py`: new, 6 test cases.
- `services/orion-actions/tests/test_walkway_forecast.py`: the tick helper uses the new
  signature.
- `services/orion-actions/README.md`: the new behaviour and how to read the records.

## Schema / bus / API changes

- Added: a new audit event kind on the existing `orion:actions:audit` channel. Envelope kind
  is unchanged (`actions.audit.v1`, schema `GenericPayloadV1`). Payload:
  `action_name=journal.walkway_night`, `status=landed|not_landed`, `night`, `trigger_kind`,
  `landed`, `reason`, `failed_attempts`,
  `event_id=walkway_night:<kind>:<stream>:<night>:<node>`.
- Added: file `walkway_night_records.jsonl` beside the scheduler cursors
  (`/data/orion-actions/` in the container). One line for each record.
- Added: keys `walkway:<kind>` in the existing `daily_attempts.json`. Same shape as the daily
  keys, so the old code can still load the file.
- Removed: none.
- Renamed: none.
- Behavior changed: the `walkway_journal` cursor now means "this night is closed". It used to
  mean "every job succeeded". The cursors for each kind still mean "an entry was written".
- Behavior changed: `build_write_payload` never returns an empty title. This reaches every
  caller, not only walkway. Callers that already set a title are not affected. Entries from
  manual triggers with no title now get `Manual · <date>`. The notification subject for a
  titleless entry changes from `Journal Pass` to the built title.
- Compatibility notes: rollback is safe. The old code ignores the new keys and loads both
  files.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

The cap (4), the waits (1, 5, 15 minutes) and the open window (6 hours) are constants in
`app/walkway_forecast.py`.

## Tests run

All run from inside the worktree with the worktree's code (checked: `app.walkway_forecast` and
`orion.journaler.worker` resolve to the worktree path).

```text
PYTHONDONTWRITEBYTECODE=1 /mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest \
  services/orion-actions/tests orion/journaler/tests orion/self_monitor/tests -q -p no:cacheprovider
473 passed

(cd services/orion-cortex-orch && python -m pytest tests/test_workflow_lane.py -q)   64 passed
(cd services/orion-durable-runs && python -m pytest tests/test_journal_compose_graph.py -q)  11 passed
python -m pytest tests/test_no_personal_terms.py tests/test_grammar_event_producer_catalog.py \
  tests/test_gitignore_no_bare_catchall.py -q    13 passed
python scripts/check_env_template_parity.py      PASS (89 services)
git diff --check                                 clean
```

Mutation check. Each fix was broken on purpose, one at a time, and the new tests were run:

```text
no cap (max_attempts=100000, wait 0)        7 tests fail
gate state not written to the file          1 fails (the restart test)
night named by today's date (the old rule)  5 fail
title stored as the model gave it           5 fail
clock read before the compose, not after    1 fails
a refused dispatch counted as a try         2 fail
collect ignores the night it is given       1 fails
```

Two things in AGENTS.md section 17 do not exist in the repo: `scripts/check_schema_registry.py`
and `scripts/check_bus_channels.py`. They were not run. This patch changes no schema and no
channel.

## Evals run

```text
None. services/orion-actions has no evals directory. Filed as #3451.
```

## Docker/build/smoke checks

```text
Not run. The task said no deploy and no `up`. No dependency, port, compose or env change.
```

## Review findings fixed

A review subagent read the diff against origin/main and ran probe scripts. It found no path
that opens more than 4 acts. It found these, all fixed in commit `e83c48dec`:

- Finding: three failed database reads at any point in the night closed it, even while a job
  that had failed twice was waiting for its next try.
  - Fix: failed reads now count in a row, not per night. After 3 in a row the night still
    stays open while a failed job has tries left.
  - Evidence: `test_a_database_blip_during_a_wait_does_not_drop_a_job_with_tries_left`,
    `test_single_read_failures_spread_over_the_night_do_not_add_up`.
- Finding: that early close was logged as a successful night.
  - Fix: a night is "clean" only when no job failed and no read failed.
  - Evidence: `test_a_read_that_stays_down_with_nothing_pending_gives_up_after_three`.
- Finding: a night that ran out of time lost the reason when the job was only ever refused.
  - Fix: the scheduler keeps the last reason for each open job in memory and puts it in the
    record (`night_expired:not_attempted:journal_cooldown`).
  - Evidence: `test_a_night_that_runs_out_of_time_is_closed_with_a_no_for_each_job`.
- Finding: if the schedule time is moved later after tonight ran, the code wrote false "not
  landed" records for the night before and moved the cursor backwards.
  - Fix: a cursor at or ahead of the night means idle.
  - Evidence: `test_a_cursor_ahead_of_the_night_is_left_alone`.
- Finding: first boot with no cursor wrote "not landed" records for a night nobody tried.
  - Fix: that night is closed with no records.
  - Evidence: `test_first_boot_past_the_window_closes_last_night_without_made_up_records`.
- Finding: a bad timezone name logged a traceback every 45 seconds.
  - Fix: one warning line, and the pass is idle.
  - Evidence: `test_a_bad_timezone_is_idle_not_a_crash`.

Not fixed, by choice (see Risks): the built title uses the UTC date when the trigger has no
date; a schedule time inside a daylight-saving hour; the 45 second re-poll of a refused job.

## Restart required

Deploy from the clean primary `main` checkout after merge. Not run here.

```bash
scripts/safe_docker_build.sh orion-actions up -d --build
```

Other services that import `orion/journaler/worker.py` (orion-cortex-orch, orion-cortex-exec,
orion-durable-runs) pick up the title fallback when they are next rebuilt. They do not need a
rebuild for the walkway fix.

## Risks / concerns

- Severity: medium
- Concern: the landed record is asked for "on the existing audit rail". That rail is a pub/sub
  channel that nothing stores (`orion/bus/channels.yaml:948`, `consumer_services: []`). The
  only copy is the bus mirror's debug database, which keeps 24 hours.
- Mitigation: the same record is also a log line and a line in `walkway_night_records.jsonl`.
  That file is new and is not in the spec. Filed #3449 for a durable reader.

- Severity: low
- Concern: the spec lists four waits (1, 5, 15, 30 minutes) and "about 4 attempts". The ticket
  check says "at most 4 acts". Four tries have three waits, so the 30 minute wait never
  happens.
- Mitigation: the cap follows the ticket check. The 30 is kept in the list and applies only
  if the cap is raised.

- Severity: low
- Concern: the cap counts a try when the compose fails. If orion-actions is killed while a
  compose is running, that try opened an act and is not counted. One extra act for each kill.
- Mitigation: none in this patch. Rare, and bounded by the number of restarts.

- Severity: low
- Concern: a job that is refused (not failed) is looked at again every 45 seconds for up to
  6 hours. It was about 1.5 hours before, until midnight. Each look is 5 or 6 small reads.
- Mitigation: no act is opened and the refusal audit is already throttled.

- Severity: low
- Concern: the title for a trigger with no date uses the UTC date. An entry written at 22:30
  Mountain gets the next day's date. Walkway entries are not affected; their date comes from
  the trigger.
- Mitigation: none. Small and visible.

- Severity: low
- Concern: a schedule time set inside a daylight-saving hour (01:00 to 03:00) misbehaves on
  the two change days. The default 22:30 is correct on both.
- Mitigation: none.

- Severity: low
- Concern: on the first run after deploy, if the last night never closed under the old code,
  the new code writes "night_expired" records for it once. That is true, and it is expected.
- Mitigation: none needed.

UNVERIFIED until deploy (tracked in #3450):

- Three of three walkway entries have a title on the first night.
- The records file has one line for each job for the night.
- A forced compose failure opens at most 4 acts for each job.
- A night held past local midnight lands under its own date.

Issues opened: #3449 (audit rail has no durable reader), #3450 (live checks for this patch),
#3451 (orion-actions has no evals).
Issues closed by this PR: #3384, #3385, #3386.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3452
