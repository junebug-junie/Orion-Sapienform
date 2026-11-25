## Summary

- The GPU pool's event history lost every event published while orion-sql-writer was restarting. The pool sent its events only over Redis pub/sub, and pub/sub keeps nothing for a subscriber that is away.
- The pool now also writes each event to a Redis Stream (`orion:stream:gpu_pool:event`). A stream keeps its entries until they are read.
- sql-writer now reads that stream through a consumer group (`cg:sql-writer`). The group remembers where the writer stopped, so the events from a restart are written when it comes back.
- sql-writer no longer reads the pub/sub channel. Pub/sub stays for the live callers that wake on `granted`.
- A new eval counts lease grants that have no row in `gpu_pool_events`, and tags each one with the sql-writer rebuild it fell in.

Fixes #3920

## Outcome moved

Failure mode removed: a sql-writer rebuild (17-24 s, plus ~20 s of boot) no longer drops pool events. Before this patch, 11-24 00:00Z to 11-25 07:00Z: 12,412 grants, 69 with no `granted` row, 68 of them inside a sql-writer `up -d --build` window. This is what made the lane-door report call real lease ids "made up" (#3495 comment, 46 of 47).

## Current architecture

- Producer: `services/orion-gpu-pool/app/runtime.py` `_emit` -> `_publish` -> `OrionBusAsync.publish` -> `redis.publish` (`orion/core/bus/async_service.py:577`). Fire-and-forget pub/sub.
- Consumer: orion-sql-writer's `Hunter` subscribes `orion:gpu_pool:event` (in `SQL_WRITER_SUBSCRIBE_CHANNELS`, also force-added by `settings.effective_subscribe_channels`). Route `gpu_pool.event.v1` -> `GpuPoolEventSQL`, insert-only on `event_id`.
- No other persister. When the sql-writer container is down, Redis delivers the message to nobody, and it is gone.

### Root cause evidence (live, 2026-11-25)

1. Missing grants vs rebuilds. `gpu_pool_leases` (the pool's own projection, written directly) is ground truth for "a grant happened". `services/orion-sql-writer/evals/gpu_pool_event_gaps.py --since 2026-11-24 --until 2026-11-25T07:00`:
   ```text
   grants (gpu_pool_leases): 12412
   sql-writer rebuild windows in range: 15
   missing 'granted' rows in gpu_pool_events: 69 (inside a rebuild window: 68, outside: 1)
     2026-11-25T00:55:18.957553+00:00 outside  metacog  fast  5d5c3983ddb44aa59f33a7295ab9a782
   verdict: FAIL
   ```
   The window runs from `START` to `END + 40 s`. `END` is when compose returns. The new container subscribes ~20-25 s later (missing grants landed up to 22 s after `END`).
2. The 06:35:50Z rebuild, per second. `gpu_pool_events` rows stop at 06:35:56 and restart at 06:36:16. `gpu_pool_state_history` (another pool pub/sub channel that sql-writer persists) has the same hole: 06:35:54 -> 06:36:22. In the same minutes, the pool's own lease table holds 21 fast grants with no event row.
3. The 00:53-00:57Z hole. That window was mostly genuinely quiet: the pool touched one lease between 00:52:54 and 00:58:00 (`vision-council`, 00:55:18-00:55:32), so "no pool events" was real for most of it. That one lease's events are missing. It is not in `post_pull_rebuild.log` (the 00:46/00:47 sql-writer attempts failed in 1 s and did not replace the container). It is a manual redeploy: `~/.orion/agent-board.jsonl` has `deploy:orion-sql-writer` ("docker compose up -d --build for services/orion-sql-writer") at 00:55:12Z from session `4cef05bc`, between `deploy:orion-thought` 00:55:04 and `deploy:orion-prediction` 00:55:29. `gpu_pool_state_history` has a 23 s hole at exactly 00:55:15 -> 00:55:38, and `grammar_events` writes pause 00:55:15 -> 00:55:35. So all 69 are the same cause.
4. Not a Redis buffer drop: the pool logged 0 `gpu_pool_publish_failed` since 00:00Z, and nothing landed in `bus_fallback_log` for `gpu_pool.event.v1`.

## Architecture touched

- orion-gpu-pool: `PoolRuntime` takes an optional `event_stream` sink. `_emit` publishes as before, then XADDs the same envelope (`_stream_event`, phase `event_stream`). Best-effort: a Redis error is logged as `gpu_pool_event_stream_failed`, never raised into scheduling. It still streams when the pub/sub publish fails. `main.py` builds the sink with `RedisStreamWorkQueue` (MAXLEN ~40000, a quiet logger, socket timeouts, and a 2 s `wait_for` bound, because it runs inside the runtime lock).
- orion-sql-writer: new `app/durable_stream_intake.py` (`DurableStreamIntake`): XREADGROUP -> `handle_envelope` -> XACK. Entries left unacked are reclaimed by XAUTOCLAIM after 30 s idle, and at boot at once. They are dead-lettered only when older than 12 h. Undecodable entries go straight to the DLQ. The group is created at id `0`, so the first boot also writes what the pool streamed before it (duplicates skip on the primary key). Started in `main.py` lifespan next to the Hunter. It is cancelled and closed on shutdown.
- `effective_subscribe_channels` removes `orion:gpu_pool:event` while the intake is on, even from a stale operator list. New `effective_stream_channels` names the streams read.
- Same shape as the live world-pulse stream consumer (`orion/spark/concept_induction/wp_stream_consumer.py`, `orion:stream:world_pulse:run:result`). It has no in-memory run_id guard here, because the table's primary key is the guard.

## Files changed

- `orion/schemas/gpu_pool.py`: `GPU_POOL_EVENT_STREAM`, `GPU_POOL_EVENT_STREAM_DLQ` constants.
- `orion/bus/channels.yaml`: stream + dlq entries (`kind: stream`). sql-writer removed from the pub/sub channel's consumers.
- `config/integration/organs.yaml`: `exempt_channels` for the stream and dlq (a transport copy of an existing payload, no new experience).
- `config/metrics/metric_definitions.lock.json`: re-lock (routing change on the pub/sub channel, 2 added stream channels).
- `services/orion-gpu-pool/app/runtime.py`, `app/main.py`, `app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: producer side.
- `services/orion-sql-writer/app/durable_stream_intake.py` (new), `app/main.py`, `app/settings.py`, `.env_example`, `README.md`: consumer side.
- `services/orion-sql-writer/evals/gpu_pool_event_gaps.py` (new): the live gap eval.
- `services/orion-gpu-pool/tests/test_event_stream.py` (new), `services/orion-sql-writer/tests/test_gpu_pool_event_stream_intake.py` (new): regression tests.
- `services/orion-sql-writer/tests/test_gpu_pool_event_sql_shape.py`: asserts stream intake, not pub/sub.
- `services/orion-sql-writer/tests/test_subscription_matches_declared_consumers.py`: streams read through a consumer group count as subscribed.
- `scripts/sync_local_env_from_example.py`: `SQL_WRITER_GPU_POOL_EVENT_STREAM_` prefix (otherwise the keys never reach a live `.env`).
- `.github/workflows/orion-gpu-pool-tests.yml`, `.github/workflows/orion-sql-writer-tests.yml`: install `fakeredis` (test-only).

## Schema / bus / API changes

- Added: Redis Stream `orion:stream:gpu_pool:event` (`GpuPoolEventV1`, `gpu_pool.event.v1`, producer orion-gpu-pool, consumer orion-sql-writer). Added: `orion:stream:gpu_pool:event:dlq` (producer orion-sql-writer).
- Removed: orion-sql-writer as a consumer of pub/sub `orion:gpu_pool:event`.
- Renamed: none.
- Behavior changed: `gpu_pool_events` is written from the stream. Rows written during a sql-writer restart now arrive late (when it returns) instead of never.
- Compatibility notes: the payload and table are unchanged. Deploy the pool before or with sql-writer (see Risks).

## Env/config changes

- Added keys: orion-gpu-pool `GPU_POOL_EVENT_STREAM_ENABLED=true`, `GPU_POOL_EVENT_STREAM_KEY=orion:stream:gpu_pool:event`, `GPU_POOL_EVENT_STREAM_MAXLEN=40000`, `GPU_POOL_EVENT_STREAM_TIMEOUT_SEC=2.0`. orion-sql-writer `SQL_WRITER_GPU_POOL_EVENT_STREAM_ENABLED=true`, `SQL_WRITER_GPU_POOL_EVENT_STREAM_KEY`, `SQL_WRITER_GPU_POOL_EVENT_STREAM_GROUP=cg:sql-writer`, `SQL_WRITER_GPU_POOL_EVENT_STREAM_DLQ`, `SQL_WRITER_GPU_POOL_EVENT_STREAM_DLQ_AFTER_SEC=43200`.
- Removed keys: none. `orion:gpu_pool:event` removed from the `SQL_WRITER_SUBSCRIBE_CHANNELS` value in the template.
- Renamed keys: none.
- `.env_example` updated: yes, both services.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes. The 4 pool keys and 5 sql-writer keys were added.
- skipped keys requiring operator action: none. The live `SQL_WRITER_SUBSCRIBE_CHANNELS` still lists `orion:gpu_pool:event` (diverged value, not overwritten). The code filter drops it while the intake is on, so no action is needed.

## Tests run

```text
cd services/orion-gpu-pool && python -m pytest tests -q            -> 208 passed, 15 skipped
PYTHONPATH=.:services/orion-sql-writer python -m pytest services/orion-sql-writer/tests -q
                                                                   -> 1373 passed, 285 skipped
test_gpu_pool_event_stream_intake.py (7) + test_event_stream.py (3) -> all pass
mutation: no boot recovery -> crash test FAILS; old pub/sub force-add -> 2 tests FAIL
python scripts/check_integration_seam.py --base origin/main       -> OK (2 new exemptions listed)
python scripts/check_definition_drift.py --gate                   -> PASS after re-lock
orion-static-gates.yml, every run step locally                     -> all PASS after re-lock
```

## Evals run

```text
python3 services/orion-sql-writer/evals/gpu_pool_event_gaps.py --since 2026-11-24 --until 2026-11-25T07:00
  pre-fix baseline: 69 missing, 68 inside a rebuild window, 1 outside (manual redeploy 00:55:12Z). verdict: FAIL
  post-deploy: UNVERIFIED (not deployed). Gate: 0 missing over a window that contains a sql-writer rebuild.
```

## Docker/build/smoke checks

```text
docker compose ... -f services/orion-gpu-pool/docker-compose.yml config -> GPU_POOL_EVENT_STREAM_* resolve (true / key / 40000)
sql-writer reads the new keys through env_file (.env); no compose change.
Image builds not run (the worktree has no .env, so safe_docker_build.sh refuses). Both images COPY the whole orion/ package; work_queue.py and redis are already present.
Live path: UNVERIFIED (not deployed, per task).
```

## Review findings fixed

Review: `orion-repo-agent` subagent on `git diff origin/main...HEAD`. No blocker. Four should-fix items, all fixed:

- Finding: the pool's XADD runs inside the runtime lock. The queue's Redis client had no socket timeout, so a half-open connection could hang every lease verb.
  - Fix: the sink uses its own client with `socket_timeout`/`socket_connect_timeout`, and a hard `asyncio.wait_for` bound (`GPU_POOL_EVENT_STREAM_TIMEOUT_SEC=2.0`). The client is closed on shutdown.
  - Evidence: `tests/test_event_stream.py` now also feeds a `TimeoutError` sink; the grant still succeeds and pub/sub still fires.
- Finding: the pool's stream flag and sql-writer's flag are coupled. The pool README offered the pool flag alone as "off", which would leave `gpu_pool_events` empty.
  - Fix: both `.env_example` files and both READMEs now say the flags are coupled: turn sql-writer's off first, and deploy the pool first. Deploy order is also in Risks. Not taken: keeping pub/sub as a second intake. It would double every insert (~33k duplicate rows a day, each logged) to cover a mis-set flag.
  - Evidence: `services/orion-gpu-pool/.env_example`, `services/orion-sql-writer/.env_example`.
- Finding: 20 deliveries x 30 s dead-lettered every event of a ~10 minute Postgres outage into a DLQ nothing reads. That is the same loss, later.
  - Fix: dead letters now go by age (`SQL_WRITER_GPU_POOL_EVENT_STREAM_DLQ_AFTER_SEC=43200`, inside the ~1 day MAXLEN horizon). Undecodable entries still go to the DLQ at once.
  - Evidence: `test_an_outage_is_waited_out_and_only_an_entry_older_than_dlq_after_is_dead_lettered`: 30 failed passes, no DLQ; past the age, one DLQ entry and an empty PEL.
- Finding: the failure paths were asserted by comment. A crash mid-batch only recovered after 30 s.
  - Fix: new tests drive the real `handle_envelope` with Postgres down (row and fallback writes both fail): the entry stays pending, then is written. At boot the intake now takes back its unacked entries at once (autoclaim with idle 0), and a test covers it.
  - Evidence: mutation check. Disabling boot recovery fails the crash test. Restoring the old pub/sub force-add fails 2 tests.
- Nit fixed: the envelope is built inside the publish `try` again, and a build failure skips the stream write. One traceback per failed batch, not per entry. The consumer name falls back to `athena` when `NODE_NAME` is empty.
- Nit not changed: the first-boot replay from id `0` writes duplicate outcomes to the storage-write organ in one window. In the normal deploy (pool then sql-writer in the same run), the stream holds only minutes of entries by then. Redis memory per entry is still an estimate; measure it with `MEMORY USAGE` after deploy.

## Restart required

Pool first, then sql-writer (both on athena, from the clean primary `main` checkout after merge):

```bash
scripts/safe_docker_build.sh orion-gpu-pool up -d --build
scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

Check:

```bash
redis-cli -u redis://100.92.216.81:6379/0 XINFO GROUPS orion:stream:gpu_pool:event
python3 services/orion-sql-writer/evals/gpu_pool_event_gaps.py --since <deploy time + 1 h>
```

The group should show `lag` near 0 and `pending` 0. After the next sql-writer rebuild, the eval should read 0 missing.

## Risks / concerns

- Severity: medium. Concern: deploy order. If sql-writer is rebuilt before the pool, sql-writer stops reading pub/sub while nothing writes the stream yet. Events are then lost until the pool is rebuilt. Mitigation: the post-pull rebuild runs changed services in name order (`orion-gpu-pool` before `orion-sql-writer`), and the manual commands above do the same. One gap remains: when the pool's rebuild fails and is retried in a later run, sql-writer goes first. Then rebuild the pool by hand, or set `SQL_WRITER_GPU_POOL_EVENT_STREAM_ENABLED=false` until it is up. The group starts at id `0`, so whatever the pool streamed before sql-writer came up is written.
- Severity: low. Concern: Redis memory. 40000 entries of ~1 KB is about 40-50 MB on a bus with 340 MB used and no maxmemory. Mitigation: `GPU_POOL_EVENT_STREAM_MAXLEN`. One peak day (11-24: 32,564 events) fits. A sql-writer outage longer than ~1 day trims the oldest unread entries.
- Severity: low. Concern: the first boot replays the whole stream, so up to 40000 duplicate inserts log one INFO line each. This happens once.
- Severity: low. Concern: other sql-writer pub/sub channels (`orion:gpu_pool:state`, `orion:grammar:event`, and more) lose events on restart in the same way. This patch fixes the one channel the lane-door report depends on. Follow-up: #3935.
- Issues: closes #3920. Opened: #3935 (the other sql-writer pub/sub channels have the same restart loss).

## PR link

PR_LINK

🤖 Generated with [Claude Code](https://claude.com/claude-code)
