# PR report: release the outreach seat when Hub is done or refuses; store the workflow on the turn trace (#3255)

Patch E of `docs/superpowers/specs/2026-11-15-day-letter-findings-fix-design.md` (PR #3439).
`Refs #3255`: the ticket also covers `fcc_timeout`, which this PR does not fix.

## Summary

- After a curiosity run, Orion keeps its GPU seat so Hub can write the message to Juniper under it.
  Hub tells durable-runs "I am done with the seat" in every case, also when it refuses to write
  (quiet hours, daily cap, strained). That call never arrived. This PR makes it arrive.
- Cause: the web route for that call was declared after a catch-all route with the same shape.
  The catch-all answered first with "400 action must be pause, resume or cancel".
- Each turn trace row now says what kind of turn it was (chat, reading, curiosity, outreach, ...).
  One new nullable column, `harness_turn_trace.workflow`.
- One additive migration for that column. It fits the post-pull auto-apply rules.

## Outcome moved

Every kept seat in the last 30 days sat idle for 1,800 s and ended `outreach_timeout`. Not one
ended `outreach_done`. That is 4 to 7 seats a day, about 2 to 3.5 idle agent-lane hours a day by
this count (the ticket counts 3.7 to 4.9 h on 11-12..11-14). After deploy the seat is released the
moment Hub is done or refuses.

Turn failures such as `fcc_timeout` can now be split by kind with one query.

## Current architecture

- durable-runs `finish` keeps the pool hold when the run asked to reach out
  (`admitted_graph.py`), heartbeats it (`admission_runtime._beat_outreach`), and ends it at
  `DURABLE_RUNS_OUTREACH_HOLD_MAX_SEC` (1,800 s) if nobody releases it.
- Hub `_maybe_reach_out` (`services/orion-hub/scripts/curiosity_investigation.py`) posts
  `/runs/{id}/release-outreach-lease` in a `finally`, so a refusal releases too. This was already
  true before this PR.
- `services/orion-durable-runs/app/main.py` declared `/runs/{run_id}/{action}` before
  `/runs/{run_id}/release-outreach-lease`. FastAPI matches in declaration order.
- `harness_turn_trace` held four JSON columns and no label for the kind of turn.

## Contradiction with the ticket and spec

The ticket says Hub never releases when it refuses. That is not what the code or the logs show.
Hub does release. durable-runs rejected the call. Live Hub log, 2026-11-15 08:35:17Z:

```text
curiosity_outreach_not_for_juniper run=b948ee77f8c1 line=investigate why=line:investigate
curiosity_outreach_lease_release_failed run=b948ee77f8c1 status=400 body={"detail":"action must be pause, resume or cancel"}
```

Live counts (`durable_resource_events`, `resource.lease_released`, last 30 days):
`outreach_timeout` 89, `outreach_done` 0. `run.outreach_pending` per day 11-07..11-14:
3, 4, 6, 7, 4, 5, 6, 6. The same counts per day ended `outreach_timeout`.

So the fix is in durable-runs' route table, not in Hub and not in `admission_runtime.py`. It also
fixes the case where Hub did write the message: that seat was held 1,800 s too.

## Architecture touched

- durable-runs: route order in `app/main.py`.
- Contract (additive): `HarnessRunRequestV1.workflow`, `HarnessRunV1.workflow`.
- Hub: `turn_workflow_label` in `orion/hub/turn_orchestrator.py` sets the label on the request.
- harness-governor: copies the label onto the run artifact it publishes.
- sql-writer: stores the label in the new column.

## Files changed

- `services/orion-durable-runs/app/main.py`: the release route is declared before the catch-all.
- `services/orion-durable-runs/tests/test_release_outreach_route.py`: the call reaches the release.
- `services/orion-durable-runs/README.md`: the behaviour and a live check.
- `orion/schemas/harness_finalize.py`: `workflow` on the request and on the run artifact.
- `orion/hub/turn_orchestrator.py`: `turn_workflow_label`, set on the request.
- `services/orion-harness-governor/app/bus_listener.py`: label copied onto every artifact.
- `services/orion-sql-writer/app/models/harness_turn_trace.py`: the column.
- `services/orion-sql-writer/app/harness_turn_trace_persist.py`: writes the column when a label is there.
- `services/orion-sql-db/manual_migration_harness_turn_trace_workflow_v1.sql`: the column.
- Tests in hub, harness-governor and sql-writer.

## Schema / bus / API changes

- Added: `HarnessRunRequestV1.workflow: str | None`, `HarnessRunV1.workflow: str | None`,
  column `harness_turn_trace.workflow text NULL`.
- Removed: none.
- Renamed: none.
- Behavior changed: `POST /runs/{id}/release-outreach-lease` now reaches its handler.
- Compatibility notes: both models ignore unknown keys (no `extra="forbid"`), so Hub, governor and
  sql-writer can deploy in any order. An older Hub or governor leaves the column NULL: unknown, not
  guessed. An artifact without a label never overwrites a stored label.

Label values: `chat` when Juniper typed it, else the caller's own `source` tag when it set one (`endogenous_outreach`,
`curiosity_investigation`, `curiosity_outreach`, `curiosity_self_inquiry`, ...), else `reading` for a reading turn, else NULL.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- skipped keys requiring operator action: none.

## Migration

`services/orion-sql-db/manual_migration_harness_turn_trace_workflow_v1.sql`

- Header: `-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer`.
- One statement: `ALTER TABLE harness_turn_trace ADD COLUMN IF NOT EXISTS workflow text;`
- I did not apply it. Dry run of the applier against the live database (read-only):
  `NOT APPLIED ... would apply (single transaction, lock_timeout=10s)`.
- No boot-time DDL was added.

## Tests run

```text
cd /mnt/scripts/Orion-Sapienform-outreach-seat-release-on-refusal
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD; PY=/mnt/scripts/Orion-Sapienform/.venv/bin/python
(cd services/orion-durable-runs && $PY -m pytest tests -q -p no:cacheprovider)        944 passed, 144 skipped
(cd services/orion-sql-writer && $PY -m pytest tests -q -p no:cacheprovider)          1305 passed, 227 skipped
(cd services/orion-harness-governor && $PY -m pytest tests -q -p no:cacheprovider)    106 passed, 1 failed
(cd services/orion-hub && $PY -m pytest tests/test_turn_workflow_label.py tests/test_curiosity_gpu_lease.py -q -p no:cacheprovider)   15 passed
$PY -m pytest tests/test_no_personal_terms.py tests/test_post_pull_apply_migrations.py -q -p no:cacheprovider   143 passed
$PY scripts/check_migration_required_by.py     OK (1 new migration file)
$PY scripts/check_metric_lineage.py --gate     PASS
git diff --check                               clean
```

The one governor failure is `tests/test_fcc_warm_pool_wiring.py::test_start_passes_settings_into_pool_config`.
It fails the same way in the primary checkout on `main`, so it is not from this change.

Old code check: with the old `main.py`, `test_hub_release_call_releases_the_seat_at_once` and
`test_release_route_maps_runtime_errors` fail (the call gets 400).

## Evals run

```text
None. No eval harness covers the Door-A seat hand-back or the turn trace writer. The route test
and the live checks below cover this change.
```

## Docker/build/smoke checks

```text
Not run. No deploy was allowed for this task. Live checks are UNVERIFIED until deploy.
```

## Review findings fixed

Review ran in a subagent on the branch diff against `origin/main`. Verdict: nothing material.
It confirmed by reading: no other shadowed route; Hub cannot get 409 in the normal flow
(`finish_projection` writes `run.completed` and `terminal='completed'` in one transaction, and Hub
calls only on that event); all eight artifact publish paths carry the label; no reader breaks.

- Finding: a chat client could set the stored label. A chat turn's payload is the client's own
  frame, and a `source` key in it won over "chat".
  - Fix: a turn Juniper typed is "chat" first; `source` is read only for other callers.
  - Evidence: `test_turn_workflow_label.py` (a `source` on a Juniper turn still gives "chat").
- Finding: now that releases really happen, the hold heartbeat could beat a hold Hub released a
  moment before, then log a false "lost" and release it twice.
  - Fix: `_beat_outreach` skips an entry that is no longer the live one, before and after the beat.
  - Evidence: `services/orion-durable-runs/app/admission_runtime.py`; inferred race, not reproduced.
- Finding: `release_outreach` answered `released: true` even when the pool call failed.
  - Fix: it answers `released: false, reason: pending_retry` (the release is retried as before).
  - Evidence: `test_release_says_released_only_when_the_pool_confirmed`.
- Finding: the Hub refusal test passes on old code, so it is not proof of the fix.
  - Fix: its docstring now says it is a contract pin and names the route test as the regression test.
- Finding (no code change): sql-writer deployed before the migration would fail the whole
  `run_artifact` write for a labelled turn.
  - Fix: none. The deploy gate refuses sql-writer until the migration is applied. Do not deploy it
    with `ORION_ALLOW_UNAPPLIED_MIGRATION=1`.
- Finding (no code change): a relayed external-room turn has no label (NULL), while the live
  activity view calls it "chat".
  - Fix: documented in `turn_workflow_label`. NULL means unknown.

## Restart required

Apply the migration first (the post-pull rebuild does it; the deploy gate refuses sql-writer until
it is applied). Then, from the clean primary `main` checkout:

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-harness-governor up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

The seat fix needs only durable-runs. The label needs the other three.

## Risks / concerns

- Severity: medium
- Concern: UNVERIFIED live. Acceptance check for #3255: `outreach_timeout` holds per day and
  lane-hours lost under 0.5 a day (was 4 to 5).
- Mitigation: after deploy run the query in `services/orion-durable-runs/README.md` (Door-A). It
  must show `outreach_done` and no new `outreach_timeout`.

- Severity: low
- Concern: UNVERIFIED live. `harness_turn_trace.workflow` is filled only after Hub, governor and
  sql-writer are all on this code.
- Mitigation: `select workflow, count(*) from harness_turn_trace where created_at > '<deploy time>' group by 1`.

- Severity: low
- Concern: `fcc_timeout` (the other half of #3255) is not fixed here. The label only makes it
  possible to split those turns by kind.
- Mitigation: the ticket stays open (`Refs #3255`).

- Issues opened: none. Issues closed: none.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3479
