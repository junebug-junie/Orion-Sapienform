## Summary

- Grammar lane queue depth is now a setting, `SQL_WRITER_GRAMMAR_QUEUE_MAXSIZE`, default unchanged at 512. Deliberately NOT raised: the queue is in memory and lost on restart, so overflow goes straight to the durable fallback table and the drain replays it.
- Per-lane `high_water` added to `grammar_queue_snapshot()`.
- New drain loop replays events shed with error `grammar queue full` from `bus_fallback_log` back into the grammar ledger while all lanes are idle.

## Outcome moved

2026-09-29: 722 `grammar.event.v1` rows landed in `bus_fallback_log` (alert threshold 720). Cause, confirmed from live rows + logs: four `harness_motor` traces from `orion-harness-governor` (~900 events / ~10 s) overflowed the 512-deep queue of lane 0 (a trace's events all hash to one lane). Nothing ever replayed them. Persist speed was fine (64 events / 0.2-0.3 s, no timeouts).

## Current architecture

`handle_envelope` -> `_spawn_grammar_persist` -> per-shard bounded `asyncio.Queue` -> shard worker -> `persist_grammar_trace_batch`. Overflow -> `_write_fallback(..., "grammar queue full")`, never retried.

## Architecture touched

`services/orion-sql-writer` only. No bus/schema/channel change.

## Files changed

- `app/worker.py`: queue size from settings; high-water; shed-error constant.
- `app/grammar_fallback_drain.py` (new): the drain.
- `app/main.py`, `app/settings.py`, `.env_example`, `docker-compose.yml`: wiring + 3 keys.
- `tests/test_grammar_fallback_drain.py` (new).

## Schema / bus / API changes

None. `grammar_queue_snapshot()` shards gain a `high_water` field (additive).

## Env/config changes

- Added keys: `SQL_WRITER_GRAMMAR_QUEUE_MAXSIZE=512`, `SQL_WRITER_GRAMMAR_DRAIN_INTERVAL_SEC=30` (0 disables), `SQL_WRITER_GRAMMAR_DRAIN_BATCH=200`.
- `.env_example` updated: yes. Local `.env`: the sync script skipped these (`SQL_WRITER_` is outside its synced prefixes), added by hand.

## Tests run

```text
tests/test_grammar_fallback_drain.py: 9 passed
services/orion-sql-writer/tests full suite: 12 failures, all also failing on unmodified main (main: 13); 0 new
check_env_template_parity / check_env_key_single_source / check_compose_no_relative_mounts: PASS
```

## Evals run

None; sql-writer has no eval harness. Follow-up: a live smoke (below).

## Docker/build/smoke checks

`docker compose config` not runnable from the worktree (gitignored `.env` absent); `check_service_env_compose_parity` reports N/A (service uses `env_file`). Live drain of the real 722 rows: **UNVERIFIED** until deployed.

## Review findings fixed

- Finding: drain deleted fallback rows when `persist_grammar_trace_batch` returned 0 without raising (rolled-back conflict / cancelled query), losing them.
  - Fix: delete only rows whose `event_id` is confirmed present in `grammar_events`.
  - Evidence: `test_returning_zero_without_a_stored_row_does_not_delete`, `test_deduped_rows_are_deleted_...`.
- Finding: drain timeout counted queue-wait and could cancel a live lane query.
  - Fix: cancel only if the drain's own job started.
  - Evidence: `test_timeout_while_queued_does_not_cancel_a_live_query`.
- Finding: a persistently failing trace pinned the oldest 200 rows forever.
  - Fix: id cursor advances past scanned rows; resets when a sweep comes up empty.
  - Evidence: `test_cursor_advances_past_failing_rows_and_resets_when_empty`.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform-grammar-queue-drain && scripts/safe_docker_build.sh orion-sql-writer up -d --build
```
Then confirm: `docker logs orion-athena-sql-writer | grep grammar_drain` and
`SELECT count(*) FROM bus_fallback_log WHERE error='grammar queue full';` trends to 0.

## Risks / concerns

- Low: queued events (<=512/lane) are in memory only and lost on restart -- unchanged from before this PR. Overflow is durable.
- Low: the governor still emits ~900 events per run; this hides the symptom, not the emission. Follow-up: look at the producer.
- Low: pre-existing sql-writer test failures on main are unrelated.

## PR link

(see PR)
