## Summary

Track B Phase 1 of `docs/superpowers/specs/2026-08-18-aitown-concept-graph-split-and-atlas-readability-design.md` -- the physical `chat_history_log` table split Juniper asked for ("wtf, i had no idea chat history log has all these ai town... htese should have their own table!"), explicitly deferred at the time pending a fresh go-ahead per phase given the real risk surface found in `orion-sql-writer`'s concurrency-hardened write path. This PR is that first phase, and only that phase.

- New `aitown_chat_history_log` table, column-for-column mirror of `chat_history_log`.
- Additive dual-write in `orion-sql-writer`: when a row's `client_meta` carries the canonical AI Town platform tag, it's written to both tables. **Zero consumer-visible change** -- `chat_history_log`'s own write path, and every one of its ~50 existing readers, is completely untouched.
- Ships with the new write path **disabled by default** (`SQL_WRITER_AITOWN_DUAL_WRITE_ENABLED=false`), matching this repo's established convention for a new write path exercised for the first time in production.
- Code review caught 4 real issues, all fixed: a missing schema-parity gate, a lost-update race on the mirror table's spark_meta patch path, an unsynced `.env`, and (most importantly) the SAVEPOINT containment guarantee this whole design rests on was only proven against a fake session -- fixed by live-verifying it against a genuine Postgres-level transaction abort.

## Outcome moved

`aitown_chat_history_log` exists and can now start accumulating real AI Town rows the moment the flag is flipped on, with a live-proven guarantee that a mirror-write failure can never take the real `chat_history_log` write down with it. Phase 2 (per-consumer audit of the ~50 `chat_history_log` readers) can build on this without touching the primary write path again.

## Current architecture

`services/orion-sql-writer/app/worker.py::upsert_chat_history_row()` is a single atomic `INSERT ... ON CONFLICT DO UPDATE` that replaced a real historical race condition ("roughly one Hub turn in five" lost its prompt or response to a SELECT-then-INSERT race across three parallel chassis tasks). `_apply_spark_meta_patch()` separately updates an existing row's `spark_meta` by `correlation_id`. Both are the two places this PR had to thread through carefully, per the design doc's own risk assessment.

## Architecture touched

`orion-sql-writer`'s chat-history write path (2 new helper functions, `upsert_chat_history_row`/`_chat_history_conflict_updates` generalized to take a `model_cls` parameter), a new Postgres table, one new settings flag.

## Files changed

- `services/orion-sql-db/manual_migration_aitown_chat_history_log_v1.sql` (new): `create table if not exists aitown_chat_history_log`, column types verified against the live `chat_history_log` schema via `\d chat_history_log`, not guessed. Applied live to this session's Athena Postgres.
- `services/orion-sql-writer/app/models/aitown_chat_history_log.py` (new): `AitownChatHistoryLogSQL`, column-for-column identical to `ChatHistoryLogSQL`.
- `services/orion-sql-writer/app/models/__init__.py`: registered the new model.
- `services/orion-sql-writer/app/settings.py`: `sql_writer_aitown_dual_write_enabled` (default `False`).
- `services/orion-sql-writer/app/worker.py`:
  - `upsert_chat_history_row()`/`_chat_history_conflict_updates()` gain a `model_cls` parameter (default `ChatHistoryLogSQL`, fully backward compatible) so the same concurrency-hardened merge logic serves both tables from one implementation instead of a hand-copied second one.
  - `_is_aitown_client_meta()`: the canonical `client_meta.external_room.platform == 'aitown'` signal, reimplemented locally (not cross-imported from `orion-recall`, matching this repo's service-boundary convention).
  - `_maybe_dual_write_aitown_chat_history()`: wired into both existing `upsert_chat_history_row()` call sites (turn path and message-path fill-only upsert).
  - `_maybe_dual_patch_aitown_spark_meta()`: wired into `_apply_spark_meta_patch()`, patch-only (never inserts a row on its own say-so).
- `services/orion-sql-writer/.env_example`: `SQL_WRITER_AITOWN_DUAL_WRITE_ENABLED=false`.
- `services/orion-sql-writer/tests/test_aitown_chat_history_dual_write.py` (new): 22 tests.

## Schema / bus / API changes

- Added: `aitown_chat_history_log` table (Postgres, additive).
- Removed: none.
- Renamed: none.
- Behavior changed: none for any existing consumer. `chat_history_log` itself, and every current reader of it, is unaffected regardless of the new flag's state.
- Compatibility notes: `upsert_chat_history_row()`'s new `model_cls` parameter defaults to the exact prior behavior; the 11 pre-existing regression tests in `test_chat_history_turn_coalesce.py` pass unmodified.

## Env/config changes

- Added keys: `SQL_WRITER_AITOWN_DUAL_WRITE_ENABLED=false`.
- `.env_example` updated: yes.
- local `.env` synced: yes, `SQL_WRITER_AITOWN_DUAL_WRITE_ENABLED=false` added directly to the primary checkout's `services/orion-sql-writer/.env`.
- skipped keys requiring operator action: none.

## Tests run

```
.venv/bin/python3 -m pytest services/orion-sql-writer/tests/test_aitown_chat_history_dual_write.py \
  services/orion-sql-writer/tests/test_chat_history_turn_coalesce.py -q
  → 33 passed

# Full service suite for context (pre-existing, unrelated failures --
# confirmed identical on unmodified main, none in any file that imports
# worker.py):
.venv/bin/python3 -m pytest services/orion-sql-writer/tests/ -q
  → 324 passed, 11 failed, 3 skipped (11 failures pre-existing on main,
    order-dependent test-suite pollution unrelated to this change --
    verified by running the same failing files standalone on both main
    and this branch: identical results either way)
```

## Evals run

No eval harness exists for this pipeline.

## Docker/build/smoke checks

Rebuilt and redeployed `orion-athena-sql-writer` live three times (initial dual-write verification, savepoint-containment verification after review, final revert to shipping defaults), against the real Postgres instance, not mocked:

**Dual-write, happy path** (flag temporarily enabled):
```
_write_row(ChatHistoryLogSQL, {...aitown turn...})   → primary + mirror rows both written, same prompt/response
_ensure_chat_history_from_message(..., client_meta=aitown)  → fill-only contribution merges into both tables
_apply_spark_meta_patch({correlation_id, spark_meta})  → patch applied to both tables' spark_meta
_write_row(ChatHistoryLogSQL, {...hub turn, non-aitown client_meta...})  → primary row only, mirror correctly absent
```

**SAVEPOINT containment, real Postgres abort** (the review finding this needed to survive):
```
upsert_chat_history_row(sess, good_values, model_cls=ChatHistoryLogSQL)      # primary, clean
_maybe_dual_write_aitown_chat_history(sess, bad_mirror_values, incoming_wins=True)  # mirror, llm_low_margin_token_count="not-a-number"
sess.commit()

→ psycopg2.errors.InvalidTextRepresentation: invalid input syntax for type integer: "not-a-number"
  (logged as aitown_chat_history_dual_write_failed, swallowed)
→ outer commit() succeeded
→ primary row exists: True (prompt intact)
→ mirror row exists: False (rolled back to the savepoint, as designed)
```

All synthetic test rows deleted after each verification pass. Flag reverted to `false` and redeployed before finishing -- the shipped state matches `.env_example`'s default.

## Review findings fixed

- Finding: no schema-parity gate between `ChatHistoryLogSQL` and `AitownChatHistoryLogSQL` -- a future column added to `chat_history_log` alone would silently stop all mirror writes forever (caught only as a logged warning).
  - Fix: deterministic test gate (`TestMirrorTableSchemaParity`) asserting column names, types, and primary key match exactly.
  - Evidence: `test_column_names_match_exactly`, `test_column_types_match`, `test_primary_key_matches`.
- Finding: `_maybe_dual_patch_aitown_spark_meta` did a plain SELECT-then-UPDATE, the exact lost-update race `upsert_chat_history_row`'s own redesign exists to avoid.
  - Fix: `.with_for_update()` row lock inside the existing SAVEPOINT.
  - Evidence: `test_locks_the_mirror_row_before_merging`.
- Finding: `.env` not synced.
  - Fix: added directly to the primary checkout's live `.env`.
  - Evidence: `grep SQL_WRITER_AITOWN_DUAL_WRITE_ENABLED services/orion-sql-writer/.env` now returns it.
- Finding: the SAVEPOINT containment guarantee was only tested against a hand-written fake session, never a real Postgres abort -- couldn't actually catch a broken `begin_nested()` usage.
  - Fix: no code change (the mechanism was already correct) -- live-verified against a genuine `psycopg2.errors.InvalidTextRepresentation` mid-transaction, confirming the outer commit survives.
  - Evidence: transcript above.
- Finding (side effect of the race fix): the existence check was inside the SAVEPOINT, paying for one on every no-op call.
  - Fix: reordered so the cheap existence check runs un-transacted first.
  - Evidence: `test_noop_when_no_mirror_row_exists`'s `nested_calls == 0` assertion.
- Investigated, REFUTED: `_is_aitown_client_meta` not handling a JSON-string `client_meta` form the way sibling implementations do. Traced `orion/core/bus/codec.py`'s `decode()` and confirmed `client_meta` is already a native dict by the time it reaches every call site in this diff -- the sibling helpers need the string branch for a different code path (reading `client_meta` back out of Postgres via a driver that doesn't auto-decode JSONB), not this one.

## Restart required

Already done live on this session's Athena host as part of verification -- `orion-athena-sql-writer` is running the new image now, with `SQL_WRITER_AITOWN_DUAL_WRITE_ENABLED=false` (shipping default).

## Risks / concerns

- Severity: low (by design -- Phase 1's whole point was to keep the real risk contained)
- Concern: `_apply_spark_meta_patch`'s *primary*-table SELECT-then-UPDATE has the same pre-existing lost-update race the review flagged on the mirror side. Not introduced by this PR, not fixed by this PR -- explicitly out of scope.
- Mitigation: flagged here as a real follow-up; low-frequency path (one spark_meta patch event per turn, not the high-concurrency three-parallel-writers path `upsert_chat_history_row` itself was hardened against), so not urgent, but real.
- Severity: low
- Concern: dual-write ships disabled. Flipping it on is a separate, explicit decision -- this PR does not turn it on in production.
- Mitigation: intentional, matches the design doc's phased approach and this repo's established convention for new write paths.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1728
