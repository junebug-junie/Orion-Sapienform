# Chat history backfill: recover 16 turns lost to the sql-writer turn-write race

## Summary

- Root-caused a long-lived, intermittent chat-history logging regression: roughly
  **one unified-turn in five (20.5%)** was persisted as a half-row — the user's
  prompt with no response, or Orion's response with no prompt.
- The cause is a **concurrent same-primary-key write race inside
  orion-sql-writer**, not the Hub and not bus delivery. Two of the three writes
  that assemble a `chat_history_log` row are discarded as "duplicate".
- **Restored all 16 recoverable turns** from copies preserved in
  `bus_fallback_log`, including the "new model on your 3rd v100" exchange the
  Hub-rehydrate PR (44ff583b2) reported Orion had lost the thread of.
- Shipped `scripts/backfill_chat_history_from_bus_fallback.py` (dry-run by
  default, fill-only, idempotent) plus tests.
- **The writer defect itself is not fixed by this PR** — see Risks.

## Outcome moved

| metric | before | after |
| --- | --- | --- |
| `hub_orion` complete rows | 62 | 76 |
| `source IS NULL` half-rows | 23 | 7 |
| recoverable half-rows | 16 | 0 |
| messages the live 48h rehydrate yields for `orion_journal` | 35 | 44 |

Nine of the 24 rows in Hub's live rehydrate window were half-rows contributing a
dangling user turn or a reply with no question. They are now proper pairs, so a
Hub restart reconstructs a coherent thread instead of a lopsided one.

## Current architecture

One Hub turn publishes four bus events in a ~10ms burst
(`orion/hub/turn_orchestrator.py:638-670`):

1. `chat.history.message.v1` — user
2. `chat.history.message.v1` — assistant
3. `chat.history.turn` — the turn row (**the only event carrying `source` and
   both halves**)
4. a legacy raw dict

Events 1-3 all write the **same `chat_history_log` primary key**: #1 and #2 via
`_ensure_chat_history_from_message` piggy-backed onto the `chat_message` write
(`services/orion-sql-writer/app/worker.py:1061,1262`), #3 via `_write_row`'s
SELECT-then-`merge` (`worker.py:1314-1330`).

## Architecture touched

Nothing in the runtime path. This PR adds a recovery script and its tests; the
data repair was applied out-of-band against the live `conjourney` database.

## Root cause

`_write_row` runs under `asyncio.to_thread` (`worker.py:1539`) with a
thread-local `scoped_session` (`services/orion-sql-writer/app/db.py:54`), and
`SQL_WRITER_CONCURRENT_HANDLERS=true` (live-confirmed in the running container)
dispatches each event as an independent chassis task
(`orion/core/bus/bus_service_chassis.py:520-530`). So the three writes execute in
**parallel OS threads, in independent transactions, against one primary key**.
Each does a non-atomic read-modify-write: SELECT, miss, INSERT. Whichever commits
first wins; the losers raise Postgres 23505 and hit:

```python
except IntegrityError as e:
    sess.rollback()
    if ... pgcode == '23505':
        logger.info("Duplicate entry ... skipping (idempotent write).")
        return False        # worker.py:1332-1342
```

Correct for append-only tables. Wrong here: `chat_history_log` is a row
*progressively assembled* by three events, so the colliding write is carrying
exactly the fields the row does not yet have. The turn write is the reliable
casualty, which is why every corrupted row has `source IS NULL`.

Second-order damage: because `_ensure_chat_history_from_message` shares the
session with the `chat_message` write, that rollback also discards the
`chat_message` row — and logs the wrong table (`"Duplicate entry for
chat_message"` when `chat_message` never had a duplicate). That is what made this
look like dropped bus traffic for months.

## Evidence (how the two rival explanations were eliminated)

**Bus loss — ruled out.** The surviving message write and the *last-published*
event of the burst were processed within **−21ms to +29ms** of each other across
all 16 turns, and several gaps are **negative** — the event published fourth was
processed before the event published first. Redis pub/sub delivers in order on a
single connection and drops by disconnecting (losing a contiguous suffix), never
by skipping the middle. All four events were delivered; two vanished inside
sql-writer.

**Publish-side schema validation — ruled out.** All 48 envelopes (16 turns × 3)
were rebuilt from the preserved payloads through the *real* hub builders and
re-validated exactly as `OrionBusAsync._validate_payload` does. All 48 pass. (An
earlier run that appeared to show `reasoning_trace.model` failures was an
artifact of the test bypassing `_normalize_reasoning_trace`, which injects that
field; corrected.)

**Not usable as a witness:** `orion-vector-host` consumes the same channel, but
its `orion_chat` collection holds **4 documents total** and `orion_chat_turns`
holds **1**. Flagged below.

## Files changed

- `scripts/backfill_chat_history_from_bus_fallback.py`: new. Dry-run by default;
  fill-only (never overwrites stored text); re-asserts the fill-only guard inside
  the `UPDATE` so a concurrent writer cannot be clobbered; snapshots the plan to
  `/tmp/chat-history-backfill/` before writing.
- `tests/test_backfill_chat_history_from_bus_fallback.py`: new. 8 tests covering
  the fill-only contract, whitespace-only stored values, payloads missing a half,
  JSON-encoded payloads, and undecodable payloads.
- `docs/superpowers/pr-reports/2026-08-14-chat-history-turn-write-race-backfill-pr.md`: this report.

## Schema / bus / API changes

None. No channel, schema, or envelope was added, removed, or changed.

## Env/config changes

None. No `.env_example` touched, so no sync was required.

## Tests run

```text
pytest tests/test_backfill_chat_history_from_bus_fallback.py -q
8 passed in 0.11s

# mutation check: the fill-only guard was inverted to confirm the tests can fail
2 failed, 6 passed   (test_never_overwrites_text_that_is_already_stored,
                      test_row_needing_nothing_is_dropped_from_the_plan)
# guard restored -> 8 passed
```

## Evals run

```text
No eval harness exists for scripts/ backfills. The behavioural check that
matters here is the live before/after in the Outcome table plus the
idempotency re-run below; a synthetic eval would not add signal.
```

## Docker/build/smoke checks

```text
No runtime service changed, so no image rebuild or restart is involved.

Live data verification:
  hub_orion complete rows      62 -> 76
  source IS NULL half-rows     23 -> 7
  second dry run after apply:  "rows with at least one field to restore: 0"
                               (idempotent -- restored rows no longer match
                                the source IS NULL predicate)
  spot-checked 3 restored pairs by hand; each restored half is the correct
  counterpart of the half already stored.
```

## Remaining 7 half-rows — all correct or genuinely unrecoverable

| rows | cause | recoverable |
| --- | --- | --- |
| 3 | "Run your dream cycle." — dream workflow is card-only metadata; `services/orion-hub/scripts/websocket_handler.py:1561` sets `orion_response_text = ""` | no response ever existed |
| 2 | Hub error path during the llamacpp 400 window — `websocket_handler.py:1590-1598` `continue`s past both publishes | no response ever produced |
| 2 | Endogenous outreach — Orion speaking unprompted, assistant-only by design | not broken |

## Restart required

```text
No restart required.
```

## Risks / concerns

- **Severity: high — Concern:** the write race is still live and still corrupting
  roughly one unified-turn in five. This PR repairs history; it does not stop the
  bleeding. **Mitigation:** follow-up PR — make the `chat_history_log` write an
  atomic `INSERT ... ON CONFLICT (id) DO UPDATE` with per-column
  `COALESCE(NULLIF(excluded.x,''), chat_history_log.x)`; stop treating 23505 on
  this table as "already have it"; give `_ensure_chat_history_from_message` its
  own session so a chat-history conflict cannot roll back a `chat_message` row.
- **Severity: medium — Concern:** recovery was only possible because the legacy
  raw-dict publish happens to be an unregistered kind and lands in
  `bus_fallback_log` as `"Unknown kind"`. That is an accident, not a designed
  safety net, and nothing ever reads it. If it is tidied up before the writer is
  fixed, future losses become permanent. **Mitigation:** fix the writer first;
  treat the dead-letter path as load-bearing until then.
- **Severity: medium — Concern:** `orion_chat` holds 4 documents and
  `orion_chat_turns` holds 1, so semantic recall over chat history has almost
  nothing to search. Unrelated to this race and not investigated here.
  **Mitigation:** separate investigation.
- **Severity: low — Concern:** the bus is genuinely at-most-once
  (`orion/core/bus/async_service.py:271`, plain `redis.publish`, return value
  ignored) and Redis reports `client_output_buffer_limit_disconnections: 5` over
  3 days. Not the cause of these 16 rows, but a real independent loss channel.
  **Mitigation:** separate investigation.

## PR link

<to be filled after push>
