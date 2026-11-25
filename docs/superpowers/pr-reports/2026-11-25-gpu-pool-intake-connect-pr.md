## Summary

- Hotfix for #3938 (#3920). After that deploy (2026-11-25 08:29Z), sql-writer's new pool-event stream reader failed on every loop with `RedisStreamWorkQueue not connected. Call await connect().` (6,915 errors in about 2 h).
- sql-writer had already stopped reading pool events over pub/sub, so no pool event reached `gpu_pool_events` from 08:29Z until this fix deploys.
- The intake now connects its own queue in `ensure_ready()`. `connect()` is only a ping when the queue is already connected. A Redis outage at boot now raises inside the retry loop, instead of the loop failing forever on "not connected".
- Nothing was lost. The pool kept writing to `orion:stream:gpu_pool:event` (2,334 entries, first one 08:26Z, cap 40,000 = about a day). The consumer group starts at id `0`, so the whole gap is written on the first run, and the insert-only `event_id` primary key drops the duplicates.

## Outcome moved

Pool events reach `gpu_pool_events` again. The 08:29Z+ gap is backfilled from the stream.

## Current architecture

`main.py` builds `RedisStreamWorkQueue(settings.orion_bus_url)` with no client and never calls `connect()`. The #3920 tests handed the intake a queue that already had a fakeredis client, so they never ran the URL-only path.

## Architecture touched

- `services/orion-sql-writer/app/durable_stream_intake.py`: `ensure_ready()` connects the queue before creating the group.

## Files changed

- `services/orion-sql-writer/app/durable_stream_intake.py`: the connect call.
- `services/orion-sql-writer/tests/test_gpu_pool_event_stream_intake.py`: a regression test that builds the queue from a URL alone, exactly as `main.py` does.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
pytest services/orion-sql-writer/tests/test_gpu_pool_event_stream_intake.py -q
8 passed
Same tests against origin/main's intake: 1 failed (RuntimeError: RedisStreamWorkQueue not connected), 7 passed.
```

## Evals run

`services/orion-sql-writer/evals/gpu_pool_event_gaps.py` (from #3938) is to be run after deploy. See Restart required.

## Docker/build/smoke checks

Live evidence of the bug: `docker logs orion-athena-sql-writer` shows `durable_intake_loop_error stream=orion:stream:gpu_pool:event`, and `XINFO GROUPS orion:stream:gpu_pool:event` returns no group.

## Review findings fixed

See the PR comment from the review subagent.

## Restart required

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

Check: `redis-cli -u redis://100.92.216.81:6379/0 XINFO GROUPS orion:stream:gpu_pool:event` shows `cg:sql-writer` with lag near 0 and pending 0, and `gpu_pool_events` has rows from 08:29Z onward.

## Risks / concerns

- Severity: low. On the first run, about 2.3k old entries are replayed. They are insert-only on the `event_id` primary key, so the existing rows are skipped.
- Issues: reopens nothing. #3920 stays closed once this is live-verified.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
