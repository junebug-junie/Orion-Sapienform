# sql-writer: publish each chat turn to memory once, not twice

## Summary

- **Each finished chat turn now reaches the memory service once.**
  - The database writer used to announce every Hub turn twice on `orion:memory:turn:persisted`: once from the turn envelope and once from the assistant message.
  - Both arrival orders happen live (2026-10-02 08:44 turn first; 08:48 messages first).
- **The turn envelope's copy wins.**
  - It carries the turn's `spark_meta`, including the new `conversation_phase` stamp. The message copy is read back from the row and does not.
  - Before, whichever copy arrived first was the one judged, so the stamp could be lost.
- **Flows with no turn envelope still work.** The Collapse Mirror reply (a user message, then an assistant reply) still publishes once, from the row, after a 5 s wait.
- This is the source fix that review item 3b of #2479 / #2484 asked for. The consumer-side guards stay for now (see below).

## Outcome moved

One memory turn per chat turn at the source, and the copy that carries the conversation clock is the one consolidation sees. Before: two publishes per Hub turn, in either order.

## Current architecture

- `services/orion-sql-writer/app/worker.py` publishes `memory.turn.persisted.v1` from three places:
  - **A** (`chat.history` turn envelope, with prompt and response): publishes the envelope's own copy;
  - **B** (`chat.history` without prompt or response): reads the row;
  - **C** (every assistant `chat.history.message.v1`): reads the row.
- A Hub turn sends both A and C. The Collapse Mirror reply sends only messages, so C is its only route.

## Every subscriber of `orion:memory:turn:persisted` (none relies on two copies)

| Subscriber | How | What it does with it | Needs two? |
|---|---|---|---|
| orion-memory-consolidation | exact channel (`CHANNEL_MEMORY_TURN_PERSISTED`) | judges and windows the turn | No: the second copy was the bug |
| orion-bus-mirror | wildcard `orion:*` | mirrors every message for observability | No: a duplicate only doubles a mirror row |
| orion-signal-gateway | wildcard `orion:memory:*` | runs adapters; none matches `memory.turn.persisted.v1` (checked `orion/signals/adapters/*`), so it is dropped | No |
| orion-bus-tap | wildcard `orion:*` | debug tap | No |
| `orion/cognition/hub_gateway/bus_harness.py`, `scripts/bus_probe.py` | wildcard | dev and test tools | No |

`orion/signals/registry.py` lists the channel as organ metadata only; it is not a subscriber.

## The existing consumer-side guards (this is the third layer, plus one producer-side check)

1. **orion-memory-consolidation `handle_memory_turn_persisted`:** skips a payload that already carries an ok `turn_change_appraisal`. **Keep** (cheap; also covers bus redelivery).
2. **`WindowStore.append_turn` collapse** (2026-08-20): replaces duplicate entries for one correlation id within a window. **Keep** as the idempotence guard on the window write.
3. **`WindowStore.find_windowed_turn`** (#2479, Fix 2): skips classifying a turn already in a window. **Can be retired after deploy** once `memory_turn_duplicate_skipped` logs zero for a week. Until then it is the guard that keeps the score single. Recommendation: keep it anyway as the redelivery guard; it is one indexed lookup.
4. **Producer-side, sql-writer `_fetch_chat_turn_for_memory_emit`:** returns None when the row already has an ok appraisal. **Keep.**

## Architecture touched

orion-sql-writer only. No schema, channel or payload change.

## Files changed

- `services/orion-sql-writer/app/worker.py`:
  - `_claim_memory_turn_emit` (an in-process claim per correlation id, 1 h TTL, bounded);
  - `_emit_memory_turn_from_envelope_once` (path A);
  - `_schedule_memory_turn_from_row` / `_emit_memory_turn_from_row_deferred` (path C, delayed);
  - path B also claims.
- `services/orion-sql-writer/app/settings.py`, `.env_example`: `SQL_WRITER_MEMORY_TURN_ROW_EMIT_DELAY_SEC=5.0`.
- `services/orion-sql-writer/tests/test_memory_turn_persisted_once.py` (new).

## Schema / bus / API changes

- Added: none. Removed: none.
- **Behavior changed:** at most one `memory.turn.persisted.v1` per correlation id per sql-writer process per hour. A message-only turn publishes about 5 s later than before.
- **Compatibility:** consumers are unchanged and already tolerate one copy.

## Env/config changes

- **Added:** `SQL_WRITER_MEMORY_TURN_ROW_EMIT_DELAY_SEC=5.0` (orion-sql-writer). It reaches the container through `env_file` (compose parity N/A).
- `.env_example` updated: yes. **Local `.env` synced** with `python scripts/sync_local_env_from_example.py --all-keys orion-sql-writer` (primary checkout).

## Tests run

```text
PYTHONPATH=.:services/orion-sql-writer pytest services/orion-sql-writer/tests
  branch: 730 passed, 12 failed, 3 skipped      main: 726 passed, the same 12 failed, 3 skipped
New test_memory_turn_persisted_once.py (4 tests): 3 fail on main, all 4 pass here
  (turn then message, message then turn, Collapse Mirror message-only flow, redelivered turn envelope).
check_metric_lineage --gate PASS; check_definition_drift --gate PASS; check_env_template_parity PASS.
```

## Evals run

No eval harness covers sql-writer publishing. Live evidence for the bug: sql-writer logs on 2026-10-02 show both arrival orders, and the #2479 replay found 86 of 87 closing turns judged twice. Post-deploy check: `memory_turn_persisted_duplicate_suppressed` appears once per Hub turn, and consolidation's `memory_turn_duplicate_skipped` drops to zero.

## Docker/build/smoke checks

```text
Not built or deployed (instructed).
```

## Review findings fixed

The orchestrator runs the review.

## Restart required

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

Independent of #2484's order. Deploying it before or after orion-memory-consolidation is safe: consolidation dedups either way.

## Risks / concerns

- **Severity:** low. **Concern:** the claim is in-process. Two sql-writer replicas, or a restart inside the 5 s delay, could still double-publish or drop a message-only publish. **Mitigation:** one replica live (UNVERIFIED that it is always one); the consumer guards stay; the degraded-classify retry appraises any row never classified.
- **Severity:** low. **Concern:** if a turn envelope arrives more than 5 s after its assistant message, the row copy (without `spark_meta`) wins. **Mitigation:** the live gap is milliseconds; the delay is configurable.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2487

🤖 Generated with [Claude Code](https://claude.com/claude-code)
