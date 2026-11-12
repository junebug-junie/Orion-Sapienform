## Summary

- durable-runs' LangGraph tables keep every checkpoint, every pending write and every old channel value forever. This adds a pruning job that keeps only what a thread needs to resume: its newest checkpoint, that checkpoint's pending writes, and the stored value each channel's newest version points at.
- It ships in **dry run**. In the service, an hourly pass logs what it would delete and shows it on `/health`. By hand, `scripts/prune_checkpoints.py` reports and deletes nothing unless given `--apply`.
- Nothing was deleted by this work. The apply command is below for Juniper.
- No index migration: the indexes LangGraph needs already exist. The slowness is the amount of history, not a missing index.

## Outcome moved

What was slow, measured read-only on 2026-11-12 ~03:30 UTC:

- **The resume sweep.** Every 120 s (`DURABLE_RUNS_RESUME_SWEEP_SEC`) the runner lists every checkpoint (`runner.py` `alist(None)`). That read covers 53.8k checkpoints in 5.3k threads. EXPLAIN ANALYZE: **16.4 s warm, ~4.0M buffer hits**. The three `temporal_self` day threads alone are **9.0 s and 2.8M buffer hits** of that.
- **Why temporal_self costs so much.** LangGraph's checkpoint read joins the value table on `thread_id` only and hashes every value row the thread ever wrote. For `temporal_self:orion:2026-11-11` that is 5,321 rows hashed to return 10 (15.6 ms warm, 1,139 pages). The sweep repeats that once per checkpoint row: 2,208 temporal_self checkpoints x ~5,500 rows.
- **Why this delays temporal_self's own step.** The saver runs every call behind ONE asyncio lock (`AsyncPostgresSaver._cursor`: `async with self.lock`). temporal_self, situation and every durable run share that saver. While the sweep's listing runs, temporal_self's checkpoint read waits.
- **The backup window.** On 2026-11-11 the nightly backup ran 12:00-12:44 UTC. temporal_self steps had gaps of 347 s (12:33) and 330 s (12:39) instead of 120 s. The same pattern shows on 11-09 (11:21, 11:28, 12:25) and 11-10 (12:15, 12:31).

What the dry run says it would remove (live, read-only, 2026-11-12):

```text
threads: 5335 (to prune: 5335)
checkpoints: 53795 total, 48460 older than their thread's newest
writes:      267964 total, 267955 on those older checkpoints
blobs:       75403 total, 52212 superseded (16.1 MB of blob payload)
temporal_self:orion:2026-11-09   blobs 5968 -> 10   checkpoints 760 -> 1
temporal_self:orion:2026-11-10   blobs 5841 -> 10   checkpoints 739 -> 1
temporal_self:orion:2026-11-11   blobs 5367 -> 10   checkpoints 709 -> 1
situation:juniper:2026-11-11     blobs 1917 -> 9    checkpoints 213 -> 1
```

UNVERIFIED:

- **The ">80 s checkpoint read during pg_dump" itself.** durable-runs restarted at 02:56 UTC on 11-12, so its logs from the backup window are gone. Postgres has `log_min_duration_statement=-1` and `log_lock_waits=off`. The lock-wait explanation above comes from the library source plus the measured 16 s sweep. Nothing recorded it happening.
- **The time saved after an apply.** That is measured only after Juniper applies.

## Current architecture

- `services/orion-durable-runs/app/main.py` opens one `AsyncPostgresSaver` on a pool (public schema). The runner, the situation driver, the temporal_self driver and the admission runtime all use it.
- The temporal_self and situation drivers write one checkpoint per step (`durability="exit"`). They delete whole day threads after 2 days (`adelete_thread`), so each day thread grows ~700 checkpoints and ~5-6k value rows a day.
- The runner's resume sweep lists all checkpoints every 120 s and keeps the newest per thread.
- No code reads history. There is no `aget_state_history`, and no config carries a `checkpoint_id`. The only `alist` keeps the newest per thread. All 53.8k checkpoints are format `v` 4, so the parent checkpoint is never read.

## Architecture touched

- New `app/checkpoint_prune.py`: the plan SQL, the delete SQL, `run_once` and `PrunePoller`.
- New `scripts/prune_checkpoints.py`: the operator CLI. It is a dry run unless given `--apply`.
- `app/main.py`: starts the poller (it is not started when mode is `off`), `/health` -> `checkpoint_prune`, and cancels it on shutdown.
- `app/settings.py`, `.env_example`, `docker-compose.yml`: two new keys.

The prune rule, per `(thread_id, checkpoint_ns)`:

- Delete checkpoints with `checkpoint_id` lower than the newest.
- Delete writes on those checkpoints.
- Delete value rows whose integer version part is lower than the version the newest checkpoint names for that channel.

Never deleted:

- anything at or above the newest. Channel versions only grow, so a writer racing the job only adds rows the job keeps.
- channels the newest checkpoint does not name.
- versions that do not parse.

Each DELETE re-reads the newest checkpoint itself. The job uses its own connection, never the saver's lock. It skips a pass while a `pg_dump` runs.

## Files changed

- `services/orion-durable-runs/app/checkpoint_prune.py`: the prune rule, the dry-run plan, the batched apply and the poller.
- `services/orion-durable-runs/scripts/prune_checkpoints.py`: the CLI.
- `services/orion-durable-runs/app/main.py`: poller lifecycle and `/health`.
- `services/orion-durable-runs/app/settings.py`: `checkpoint_prune_mode`, `checkpoint_prune_interval_sec`.
- `services/orion-durable-runs/.env_example`, `docker-compose.yml`: the keys.
- `services/orion-durable-runs/README.md`: the "Checkpoint pruning" section under Resume semantics.
- `services/orion-durable-runs/tests/test_checkpoint_prune.py`: pure tests, real-Postgres lifecycle tests, poller tests, the DeltaChannel guard.
- `orion/memory/referents/backfill.py`: falls back to the newest checkpoint's `answer_text` (review finding 1).
- `orion/memory/referents/tests/test_referents_pg.py`: the post-prune backfill test.
- `docs/superpowers/pr-reports/2026-11-12-durable-checkpoint-prune-pr.md`: this report.

## Schema / bus / API changes

- Added: `/health` field `checkpoint_prune` (mode, interval, passes, last totals and deletions, last error).
- Removed: none.
- Renamed: none.
- Behavior changed: none while mode is `dry_run` (the shipped default). It only reads, about 3 s an hour.
- Compatibility notes: no SQL migration. No bus change.

## Env/config changes

- Added keys: `DURABLE_RUNS_CHECKPOINT_PRUNE_MODE=dry_run` (`off` | `dry_run` | `apply`), `DURABLE_RUNS_CHECKPOINT_PRUNE_INTERVAL_SEC=3600.0`.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (both keys present in `services/orion-durable-runs/.env`).
- skipped keys requiring operator action: none.

## Tests run

```text
tests/test_checkpoint_prune.py against a throwaway postgres:16 container (ORION_ADMISSION_TEST_DSN): 15 passed
full services/orion-durable-runs/tests in the durable-runs image + CI deps: 1000 passed, 14 skipped
orion/memory/referents/tests (postgres:16 + falkordb:6.0.1): 20 passed (7 pg tests incl. the new backfill one)
mutation checks (each must fail a test):
  delete the referenced blob version (< -> <=)       -> failed (state lost)
  delete writes of the newest checkpoint (< -> <=)   -> failed (interrupt lost)
  drop the version-format guard                      -> failed
python scripts/check_service_env_compose_parity.py orion-durable-runs -> OK
python scripts/check_env_template_parity.py -> PASS
git diff --check -> clean
```

## Evals run

```text
Live dry run against the production DB (read-only SELECTs): numbers in "Outcome moved".
No eval harness change: the real-Postgres lifecycle test is the behaviour proof (state identical, interrupt kept, both threads resume).
```

## Docker/build/smoke checks

```text
No image build or deploy (task said do not deploy). Tests ran inside the current orion-durable-runs image.
```

## Review findings fixed

Review ran in a subagent. It read the installed library source and ran read-only live queries.

- Finding (must-fix): `apply` would delete the only source the referents backfill read. `orion/memory/referents/backfill.py` read `answer_text` from `checkpoint_writes`. For all 65 `episode_distill_run` threads that write sits on an older checkpoint.
  - Fix: the backfill now falls back to the newest checkpoint's inline `answer_text`, which the prune keeps (live: a string in 65 of 65). The module docstring and README list repo-wide history readers.
  - Evidence: new `test_checkpoint_backfill_reads_the_newest_state_after_a_checkpoint_prune`. referents pg tests: 7 passed (Postgres 16 + FalkorDB).
- Finding: there was no guard against `DeltaChannel`. A delta channel is rebuilt from parent checkpoints and their writes, so the prune would truncate it.
  - Fix: `test_no_durable_runs_graph_uses_a_delta_channel` fails if any durable-runs graph uses `DeltaChannel`, `add_messages` or `MessagesState`.
  - Evidence: the test passes (no graph uses one today).
- Finding: the regex guard and the numeric cast were siblings in one AND. Postgres may evaluate them in any order, so the cast could run on a version the guard rejects.
  - Fix: the comparison is now `CASE WHEN guard THEN cast-compare ELSE false END`.
  - Evidence: the "weird" version test and the guard mutation still behave as expected.
- Finding: the hourly dry run detoasted every checkpoint's jsonb.
  - Fix: the query picks the newest ids first, then joins back once per thread.
  - Evidence: live dry run is 2.3 s end to end, including container start.
- Finding: shutdown could hang behind an apply pass running in a worker thread.
  - Fix: `PrunePoller.request_stop()` sets a threading event. `apply_plan` checks it between batches. `main.py` calls it on shutdown.
  - Evidence: `test_apply_stops_between_batches_when_asked`.
- Finding: test gaps. There was no `update_state(as_node=...)` path, no racing pending write above the newest checkpoint, and no check of the deleted-writes count.
  - Fix: all three were added.
  - Evidence: `test_update_state_threads_read_back_and_resume_after_prune`. The lifecycle test now asserts the racing write survives and that `writes` deleted equals `writes_del`.
- Finding (nit): the rule never deletes whole threads, so the sweep still grows with thread count.
  - Fix: written down in the README and the docstring as the next follow-up. It is not built here, because deleting finished runs is a separate decision.
  - Evidence: n/a.
- Finding (nit): `/health` showed `null` when the mode is off.
  - Fix: it now reports `{"mode": "off"}`.
  - Evidence: code.

## Restart required

Deploy (from the clean primary main checkout after merge):

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

Then, for Juniper (this deletes):

```bash
docker exec orion-athena-durable-runs python /app/scripts/prune_checkpoints.py
docker exec orion-athena-durable-runs python /app/scripts/prune_checkpoints.py --apply
```

The first command is the dry run, to read before deleting. The second deletes. Do not run either between 12:00 and 12:45 UTC (the nightly backup); the job refuses while a pg_dump runs.

To keep it pruned every hour afterwards, set `DURABLE_RUNS_CHECKPOINT_PRUNE_MODE=apply` in `services/orion-durable-runs/.env` and redeploy.

Check after apply: `curl -fsS localhost:8124/health | jq .checkpoint_prune`, and re-run the sweep EXPLAIN in this report (expect far under 16 s).

## Risks / concerns

- Severity: medium
  - Concern: deleting history removes time travel (reading an older checkpoint by id). Nothing in durable-runs uses it today.
  - Mitigation: the nightly backup still holds full history. Dry run is the default.
- Severity: low
  - Concern: freed space is reused, but the table files do not shrink. Use VACUUM FULL only by hand, outside the backup window.
  - Mitigation: none needed for speed. The speed-up comes from fewer rows per thread.
- Severity: low
  - Concern: the GPU pool's own checkpoint schema (`gpu_pool`, 5.8 GB, 615k checkpoints) is the biggest thing the backup dumps. It is not covered here, because its resume rules were not checked.
  - Mitigation: a separate look at that schema.

## PR link

TBD
