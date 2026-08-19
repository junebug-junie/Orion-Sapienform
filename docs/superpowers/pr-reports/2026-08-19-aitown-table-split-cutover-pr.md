## Summary

Juniper: *"why not hit this shit now and fix all of it so no dual write... im not using ai town at the moment."* Verified live: AI Town's own Convex backend is confirmed dead (`AitownClientError: Connection refused`, every call), last real `chat_history_log` write 18 days stale. Zero concurrent-write risk, so the phased dual-write bridge PR #1734 built for a live AI Town world was pure unneeded complexity. Retired it same day, replaced with routing + a real, live, one-time move of the historical data.

- **Producer**: `orion-sql-writer` no longer dual-writes AI-Town rows to both tables. `_resolve_chat_history_model_cls()` decides which ONE table a row belongs to, once. Ships **on by default** (`SQL_WRITER_AITOWN_ROUTING_ENABLED=true`) -- unlike the retired dual-write flag, routing carries no first-time-in-production risk for non-AI-Town rows.
- **Historical data**: moved, not copied. `scripts/backfill_aitown_chat_history_move_to_split_table.py`, snapshot-first, atomic INSERT+DELETE in one transaction. Ran live: 1,577 AI-Town rows moved out of `chat_history_log` (1,747 -> 170 rows), into `aitown_chat_history_log` (0 -> 1,577).
- **Code review (2 rounds)** found and fixed a real, confirmed race condition in the routing fallback logic, closed with a Postgres advisory lock -- **live-proved via two genuinely concurrent sessions** (see below), plus 6 other findings (env sync, backfill robustness, missing index, protocol-compliance artifacts, JSON codec).

## Outcome moved

`chat_history_log` now holds only real Orion/Juniper conversation. AI Town content lives in its own table, cleanly partitioned (an id lives in exactly one table, never both), enabling `orion-recall`'s companion PR to actually see AI-Town content again without polluting the primary table.

## Current architecture

`orion-sql-writer`'s chat-history write path: `upsert_chat_history_row()`/`_chat_history_conflict_updates()` already took a `model_cls` parameter (PR #1734), used here to route instead of duplicate.

## Files changed

- `services/orion-sql-writer/app/worker.py`: `_maybe_dual_write_aitown_chat_history`/`_maybe_dual_patch_aitown_spark_meta` (additive, Phase 1) removed entirely. Replaced with `_resolve_chat_history_model_cls()` (routing decision) + `_lock_chat_history_row()` (advisory lock closing the cross-session race). `_apply_spark_meta_patch` now checks both tables (a patch carries no `client_meta` to reclassify from) and picked up a pre-existing lost-update-race fix (`.with_for_update()`) on the primary-table side while already being touched.
- `services/orion-sql-writer/app/settings.py`: `sql_writer_aitown_dual_write_enabled` retired, replaced with `sql_writer_aitown_routing_enabled` (default `True`).
- `services/orion-sql-writer/.env_example`: same rename.
- `services/orion-sql-writer/app/models/aitown_chat_history_log.py`: docstring updated for the routing model.
- `services/orion-sql-db/manual_migration_aitown_chat_history_log_v1.sql`: added a `created_at` index (code review -- every `orion-recall` chat query filters/orders on it).
- `scripts/backfill_aitown_chat_history_move_to_split_table.py` (new): one-off, snapshot-first, atomic move. Review-fixed: mismatch-check no longer aborts on a legitimate pre-existing overlap (only a genuine anomaly), writes `report.md`/`before_after.csv` per AGENTS.md section 14, registers a JSON codec so `spark_meta`/`client_meta` land in the snapshot as real nested JSON, not escaped strings.
- `services/orion-sql-writer/tests/test_aitown_chat_history_dual_write.py`: rewritten for routing semantics + a new `TestLockChatHistoryRow`.
- `services/orion-sql-writer/tests/test_spark_meta_patch.py`, `test_llm_uncertainty_spark_meta.py`: pre-existing fake sessions updated for the new `.with_for_update()`/advisory-lock call shapes.

## Schema / bus / API changes

- Added: `aitown_chat_history_log.created_at` index.
- Removed: `SQL_WRITER_AITOWN_DUAL_WRITE_ENABLED` setting (fully retired, no fallback).
- Renamed: → `SQL_WRITER_AITOWN_ROUTING_ENABLED`.
- Behavior changed: AI-Town rows land in exactly one table now, not both. Zero change for non-AI-Town rows.
- Compatibility notes: `upsert_chat_history_row`'s `model_cls` parameter is unchanged (PR #1734); `test_chat_history_turn_coalesce.py`'s 11 regression tests pass unmodified.

## Env/config changes

- Removed keys: `SQL_WRITER_AITOWN_DUAL_WRITE_ENABLED`.
- Added keys: `SQL_WRITER_AITOWN_ROUTING_ENABLED=true`.
- `.env_example` updated: yes.
- local `.env` synced: yes -- code review caught this was missed on the first pass; fixed (`services/orion-sql-writer/.env` now has the new key, old key removed).

## Tests run

```
.venv/bin/python3 -m pytest services/orion-sql-writer/tests/ -q
  → 336 passed, 7 failed (pre-existing, order-dependent, confirmed identical on unmodified main), 3 skipped
```

## Evals run

No eval harness exists for this pipeline.

## Docker/build/smoke checks

Rebuilt and redeployed `orion-athena-sql-writer` live 4 times across two review rounds, against real Postgres:

**Routing (happy path + wrong-table-prevention):**
```
AI-Town turn -> mirror table only (primary correctly absent)
client_meta-less message-path event for the same id -> correctly found the row already
  routed to the mirror table via the id-lookup fallback, merged into it (no stray duplicate)
spark_meta patch -> correctly fell back to the mirror table
Non-AI-Town turn -> primary table only (mirror correctly absent)
```

**The race fix, real concurrency proof (2 genuinely concurrent threads, real Postgres sessions):**
```
turn-path:    acquired lock at T+0.084s, held until T+1.085s (commit)
message-path: BLOCKED from T+0.15s, only acquired at T+1.083s -- ~0.93s of
              real, measured blocking, releasing the instant turn-path committed
```

**Historical data move (live, one-time):**
```
before: chat_history_log=1,747 (1,577 AI-Town), aitown_chat_history_log=0
after:  chat_history_log=170 (0 AI-Town), aitown_chat_history_log=1,577
verified via direct psql query post-move, not just script output
```

**Backfill robustness fix, synthetic overlap test:**
```
Inserted the same id into BOTH tables by hand, ran the real script:
  moved: {"inserted": 0, "deleted": 1, "already_mirrored": 1}
  verdict: ok
(Before the fix: this would have raised RuntimeError and aborted the whole batch.)
```

## Review findings fixed

- Finding (CONFIRMED, most severe): `_resolve_chat_history_model_cls`'s fallback lookup races across separate sessions -- the turn-path and message-path writes for the same id run in independent transactions with no serialization point, so a message-path write could miss an uncommitted turn-path insert and route to the wrong table.
  - Fix: `_lock_chat_history_row()`, a `pg_advisory_xact_lock` keyed on the row id, acquired before routing on both call sites.
  - Evidence: real two-thread concurrency test above, ~0.93s of measured blocking.
- Finding (CONFIRMED): `.env` not synced -- primary checkout still had the old `SQL_WRITER_AITOWN_DUAL_WRITE_ENABLED` key.
  - Fix: replaced with `SQL_WRITER_AITOWN_ROUTING_ENABLED=true`.
- Finding (PLAUSIBLE, confirmed real by a live precondition check): the backfill's mismatch guard aborted the entire batch on any pre-existing overlap (e.g. from earlier dual-write testing), not just a genuine anomaly.
  - Fix: only `inserted > deleted` is now treated as fatal; `inserted < deleted` (rows already correctly mirrored) is expected and handled.
  - Evidence: synthetic overlap test above.
- Finding: backfill script's artifacts didn't match AGENTS.md section 14's `report.md`/`before_after.csv` shape.
  - Fix: both now written alongside `report.json`.
- Finding: snapshot JSON double-encoded `spark_meta`/`client_meta` as escaped strings.
  - Fix: registered an asyncpg JSON codec.
- Finding: `aitown_chat_history_log` had no `created_at` index despite every `orion-recall` query filtering/ordering on it.
  - Fix: added. (Noted: `chat_history_log` itself has the same pre-existing gap, not introduced here, not fixed here -- out of scope for this migration file.)
- Finding: routing removes the resilience the additive dual-write had (a transient DB error writing to the mirror no longer leaves the row safely in a "primary" table, since there's no primary/mirror distinction under real partitioning).
  - Not fixed: this is the direct, understood consequence of the architectural decision this PR implements, not a bug. A transient write failure to *either* table under real partitioning is a general reliability concern equivalent to a transient failure writing to `chat_history_log` itself ever had -- not something the old dual-write protected against for the primary table either.

## Restart required

Already done live on this session's Athena host as part of verification -- `orion-athena-sql-writer` is running the new image now.

## Risks / concerns

- Severity: low
- Concern: the companion `orion-recall` PR (separate PR) was built and pushed before this PR merged, so its code currently describes invariants ("id lives in exactly one table") that depend on THIS PR merging. Both PRs should merge together or in this order (sql-writer first).
- Mitigation: noted explicitly in both PR descriptions; the recall PR's code is written to be correct either way (two-query merge / accepted bounded duplicate risk), not dependent on strict merge ordering for correctness, only for the comments' accuracy.

## PR link

(opened via `gh pr create`, see final message)
