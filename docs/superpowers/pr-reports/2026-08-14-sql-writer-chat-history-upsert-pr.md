# Fix the cause: stop dropping chat_history_log writes as idempotent duplicates

## Summary

- Roughly **one Hub unified-turn in five (20.5%)** was persisted as a half-row —
  Juniper's prompt with no response, or Orion's response with no prompt.
  PR #1649 recovered the 16 turns that were still recoverable. **This fixes the
  cause.**
- `chat_history_log` writes are now a single atomic
  `INSERT … ON CONFLICT (id) DO UPDATE` with a per-column merge rule, replacing a
  SELECT-then-INSERT that three parallel threads could all lose.
- `_ensure_chat_history_from_message` runs in its **own** session, so a chat-history
  conflict can no longer roll back the `chat_message` row it was riding on.
- A 23505 on `chat_history_log` is now a loud error, never an "idempotent duplicate".
- Proven against a real Postgres and end-to-end on the live bus, not asserted.

## Outcome moved

| | before | after |
| --- | --- | --- |
| concurrent same-PK writers, 3 contributors, forced contention | **13/25** complete | **25/25** |
| writes silently swallowed as duplicates | 50 | **0** |
| live bus → deployed container, 5 turns | — | **5/5** complete |
| `chat_message` rows surviving per turn | 1 of 2 | **2 of 2** |

## Current architecture

One Hub turn publishes three events that all write the **same**
`chat_history_log` primary key, each carrying a different subset of columns:

| event | carries |
| --- | --- |
| `chat.history.message.v1` (user) | `prompt` |
| `chat.history.message.v1` (assistant) | `response` |
| `chat.history.turn` | **`source`** + both halves + `spark_meta` |

They are dispatched as independent chassis tasks
(`orion/core/bus/bus_service_chassis.py:520-530`) and executed in **parallel OS
threads** via `asyncio.to_thread` (`worker.py:1539`) with **thread-local**
`scoped_session`s (`db.py:54`).

## Root cause

All three could SELECT-miss together, all three would INSERT, and two would take
a Postgres 23505 that `_write_row` discarded:

```python
if ... pgcode == '23505':
    logger.info("Duplicate entry ... skipping (idempotent write).")
    return False
```

Correct for append-only tables. Wrong here — `chat_history_log` is a row
*progressively assembled*, so "we already have this id" never means "we already
have this content". The turn event is the only one carrying `source`, so it was
the reliable casualty: **every** corrupted row had `source IS NULL` and exactly
one of prompt/response.

Second-order damage: `_ensure_chat_history_from_message` shared the caller's
session with the `chat_message` write, so that rollback destroyed the
`chat_message` row too — while the log line blamed `chat_message`, a table that
never had a duplicate. That is what made this read as dropped bus traffic for
months.

## Architecture touched

- `services/orion-sql-writer/app/worker.py` — the write path only. No schema, no
  channel, no envelope, no config.

## Files changed

- `services/orion-sql-writer/app/worker.py`:
  - **added** `upsert_chat_history_row()` + `_chat_history_conflict_updates()`,
    with a per-producer policy. Turn path (`incoming_wins=True`):
    `coalesce(nullif(excluded.c,''), t.c)`. Message path (`incoming_wins=False`):
    `coalesce(t.c, nullif(excluded.c,''))` — fill-only. `nullif` is skipped for
    JSONB/scalars, since `coalesce(spark_meta,'')` does not typecheck. `''`
    counts as absent for text because `ChatHistoryTurnV1` declares
    `prompt`/`response` as required `str`, so the turn always carries both keys
    and either can legitimately be empty.
  - **changed** `_ensure_chat_history_from_message()` — own session via
    `session_factory()`. Note `get_session()` would **not** have worked: it is a
    thread-local `scoped_session` and would hand back the caller's own session,
    leaving the transactions shared and letting `remove_session()` tear the
    caller's session out from under it.
  - **changed** the `IntegrityError` handler — 23505 on this table propagates.
  - **deleted** `_coalesce_chat_history_turn_fields()` — its whole job was the
    Python half of the read-modify-write that raced, and it had no callers left.
- `services/orion-sql-writer/evals/test_chat_history_concurrent_write_eval.py`: new.
- `services/orion-sql-writer/tests/test_chat_history_turn_coalesce.py`: replaced the
  dead function's test with 11 covering the real ON CONFLICT semantics and the
  per-producer merge policies.
- `services/orion-sql-writer/tests/test_llm_uncertainty_spark_meta.py`: fake session
  now captures `execute(stmt)` as well as `merge(obj)`.

## Schema / bus / API changes

None. Added / Removed / Renamed: none. Behavior changed: a concurrent write to an
existing `chat_history_log` row now merges its columns instead of being dropped.
Compatibility: no migration — same table, same columns, same primary key.

## Env/config changes

None. No `.env_example` touched, so no sync was required.

## Tests run

```text
pytest services/orion-sql-writer/tests -q
10 failed, 244 passed, 3 skipped

Baseline on unmodified main: 10 failed, 240 passed, 3 skipped
Failure sets diffed and IDENTICAL -- zero regressions. All 10 are pre-existing
DB-dependent grammar/notify tests (test_grammar_truth x4,
test_journal_entry_payload_boundary x1, test_notify_attention_ack x3,
test_notify_attention_escalate x2).

pytest services/orion-sql-writer/tests/test_chat_history_turn_coalesce.py -q
11 passed
```

## Evals run

```text
pytest services/orion-sql-writer/evals -q
2 passed

Mutation check -- swapped the upsert for the old SELECT-then-INSERT:
  1 failed: "10/15 turns lost a contributor's columns to the race"
  and it reproduced the EXACT live fingerprints:
    ('the user prompt', '', '')                          <- prompt-only
    ('', 'the assistant response', '')                   <- response-only
    ('the user prompt', 'the assistant response', '')    <- both, source empty
  matching the 10 prompt-only / 6 response-only split seen in production.
Restored -> 2 passed.
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-sql-writer build   -> Image Built
scripts/safe_docker_build.sh orion-sql-writer up -d   -> Started, Up, no new errors
  (the grammar_retention QueryCanceled at boot is pre-existing -- present in the
   pre-deploy logs too)

Real-Postgres concurrency proof, 3 writers past a barrier, 25 trials:
  OLD (select+insert)  complete 13/25   swallowed writes: 50
  NEW (on conflict)    complete 25/25   swallowed writes: 0

Live end-to-end through the real bus into the deployed container:
  published the real 3-event burst x5 -> complete rows: 5/5
  cleanup removed 15 rows (5 chat_history_log + 10 chat_message), confirming
  BOTH message halves now survive -- the second-order bug is fixed too.

Post-deploy: 0 new source-IS-NULL rows; 0 synthetic rows left behind.
```

## Review findings fixed

- **Finding (MEDIUM):** the message path flipped from "fill only if empty" to
  "last writer wins". The old `_ensure_chat_history_from_message` guarded every
  field (`if not existing.prompt`, `if session_id and not existing.session_id`,
  ...); my single merge rule dropped that. Because the assistant message is
  published *after* the turn on the WS path (`websocket_handler.py` ~1802 then
  ~1911), a message could clobber the turn's canonical
  `client_meta`/`memory_status`/`memory_tier`. Latent today (the WS path passes
  identical values) but genuinely unprotected.
  - **Fix:** `upsert_chat_history_row(..., incoming_wins=...)`. The turn path
    keeps `coalesce(nullif(excluded.c,''), c)` — it is authoritative and its
    caller has already folded prior state into `spark_meta`/`thought_process`.
    The message path uses `coalesce(c, nullif(excluded.c,''))` — fill-only,
    reproducing the old per-field guards exactly.
  - **Evidence:** 5 new tests pin the distinction, including one asserting the
    two policies actually differ and one that fill-only still *fills* an empty
    column. Live smoke re-run after the change: 5/5 complete, `source` intact.
- **Finding (LOW):** the docstring justifying `nullif(…, '')` cited seeding this
  same diff deleted, so a future reader could remove it and re-introduce the
  clobber.
  - **Fix:** rewritten to the real reason.
  - **Evidence:** `ChatHistoryTurnV1` declares `prompt: str = Field(...)` and
    `response: str = Field(...)` — both required, so the turn always carries the
    keys and either can be `''`.
- **Finding (LOW):** the conflict target narrowed from `id OR correlation_id` to
  `id` alone.
  - **Fix:** documented in `upsert_chat_history_row`'s docstring rather than
    changed. Verified every current producer sets `id == correlation_id`
    (`scripts/chat_history.py` passes `turn_id` == the correlation id at all call
    sites; `_write_row` back-fills `id` from `correlation_id` when absent).
  - **Evidence:** no live producer sends a divergent `id`; noted so a new one
    does not silently re-create the orphan shape.
- **Finding (LOW/MEDIUM) — NOT ACCEPTED:** "raising on a 23505 turns a dropped
  write into a hot retry loop on every redelivery."
  - **Why not:** this transport has no redelivery. `bus_service_chassis.py`
    catches handler exceptions and calls `_publish_error`; there is no ack,
    requeue, or retry anywhere in it — it is fire-and-forget pub/sub. A raise
    produces exactly one error event, which is the intended loudness, with no
    loop. Verified by reading the handler and grepping the chassis for
    ack/redeliver/requeue/retry (no matches). Left as-is.

## Restart required

Already applied on athena during verification. To redeploy elsewhere:

```bash
scripts/safe_docker_build.sh orion-sql-writer build
scripts/safe_docker_build.sh orion-sql-writer up -d
```

## Risks / concerns

- **Severity: low — Concern:** the turn path's rule is "a non-empty incoming
  value wins", so if two *turn* events ever carried different non-empty text for
  one column, the later would win. Cannot happen with one turn event per
  correlation id. The message path is fill-only, so it cannot overwrite at all.
  **Mitigation:** both policies documented and pinned by tests in
  `_chat_history_conflict_updates`.
- **Severity: low — Concern:** `ChatGptLogSQL` still uses `sess.merge()` and the
  old duplicate-swallow. It has the same table shape but no evidence of a
  concurrent multi-event producer, so it was left alone rather than changed
  speculatively. **Mitigation:** revisit if that path ever grows one.
- **Severity: informational:** the bus remains at-most-once
  (`orion/core/bus/async_service.py:271`), and `orion_chat` holds 4 documents.
  Both are unrelated to this race and still open — see PR #1649's risk list.

## PR link

<to be filled after push>
