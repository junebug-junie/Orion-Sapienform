# PR: Chat Grammar Lane V1

**Branch:** `worktree-feat+substrate-signal-bridge-v1`  
**Base:** `main`  
**Head:** `b4da3068` (7 commits on top of signal bridge)

## Summary

Closes the chat lane gap. Before this PR, every meaningful lane in Orion compressed through the substrate grammar pipeline except the most semantically rich signal — what users actually say. Chat turns produced in-memory molecules for repair pressure appraisal but never emitted `GrammarEventV1` events, never reached a reducer, and never perturbed the attention field.

After this PR, a chat turn flows:

```text
orion-hub chat turn (websocket_handler.py)
  → GrammarEventV1 on orion:grammar:event  (hub.chat:{node_id}:{turn_id})
  → orion-substrate-runtime  chat_grammar_consumer cursor
  → orion/substrate/chat_loop/ reducer
  → StateDeltaV1(target_kind=chat_turn)
  → substrate_reduction_receipts
  → orion-field-digester
  → node conversation_load / repair_pressure channels
```

## Non-goals respected

- No new `GrammarEventKind`, `AtomType`, or `RelationType` literals
- No raw user text in any grammar event (utterance atom uses `payload_ref` only)
- No modifications to `substrate_effect_pipeline.py` or `appraisal/`
- No live bus subscription added (same stance as signal bridge V1)
- Default-off at every layer (`PUBLISH_HUB_CHAT_GRAMMAR=false`, `ENABLE_CHAT_GRAMMAR_REDUCER=false`)

## Architecture

```
[Hub emitter]         services/orion-hub/scripts/grammar_emit.py
[Hub publisher]       services/orion-hub/scripts/grammar_publish.py
[Hub wiring]          services/orion-hub/scripts/websocket_handler.py  (after substrate_effect_pipeline)
[Hub settings]        services/orion-hub/app/settings.py  (+2 fields)
[Schema]              orion/schemas/chat_projection.py
[Reducer package]     orion/substrate/chat_loop/  (constants, grammar_extract, reducer, pipeline)
[SQL migration]       services/orion-sql-db/manual_migration_chat_substrate_loop.sql
[Runtime store]       services/orion-substrate-runtime/app/store.py
[Runtime worker]      services/orion-substrate-runtime/app/worker.py
[Runtime settings]    services/orion-substrate-runtime/app/settings.py
[Field digester]      services/orion-field-digester/app/ingest/state_deltas.py
[Lattice]             config/field/biometrics_lattice.yaml + orion_field_topology.v1.yaml
```

### Trace shape

```
trace_id: hub.chat:{node_id}:{turn_id}

trace_started
  atom: session_context   (entity)      — session_id, payload_ref=hub.session:{sid}
  atom: user_utterance    (observation) — text_value=None, payload_ref=hub.chat:{sid}:{turn_id}
  atom: repair_signal     (signal)      — level/confidence from appraisal (if has_repair_signal)
  edge: user_utterance → derived_from → session_context
  edge: repair_signal  → derived_from → user_utterance  (if repair_signal)
trace_ended
```

### Pressure hints → field lattice

| Hint | Formula | Channel |
|------|---------|---------|
| `conversation_load` | `min(1.0, word_count / 150.0)` | `node_channels` |
| `repair_pressure` | `repair_pressure_level` (from appraisal) | `node_channels` |
| `topic_coherence` | `max(0.0, 1 - repair_pressure)` | internal only |

### Worker addition

`_grammar_reducer_poll_loop` cursor-advance dispatch was a binary `if EXECUTION / else TRANSPORT`; promoted to 3-way `if/elif/else` with `CHAT_GRAMMAR_CURSOR_NAME → advance_chat_cursor`. Everything else is a straight pattern match of the execution reducer.

## Files changed

| File | Change |
|------|--------|
| `services/orion-hub/scripts/grammar_emit.py` | New — pure builder, 7 events per turn |
| `services/orion-hub/scripts/grammar_publish.py` | New — fail-open async publisher |
| `services/orion-hub/app/settings.py` | +`PUBLISH_HUB_CHAT_GRAMMAR`, `GRAMMAR_EVENT_CHANNEL` |
| `services/orion-hub/scripts/websocket_handler.py` | Wire grammar publish after substrate_effect_pipeline |
| `orion/schemas/chat_projection.py` | New — `ChatTurnStateV1`, `ChatSessionProjectionV1` |
| `orion/substrate/chat_loop/__init__.py` | New — package marker |
| `orion/substrate/chat_loop/constants.py` | New — cursor name, projection ID, reducer ID, etc. |
| `orion/substrate/chat_loop/grammar_extract.py` | New — extract turn state + pressure hints from events |
| `orion/substrate/chat_loop/reducer.py` | New — `reduce_chat_trace_events` → StateDeltaV1 |
| `orion/substrate/chat_loop/pipeline.py` | New — `process_chat_grammar_events` |
| `services/orion-sql-db/manual_migration_chat_substrate_loop.sql` | New — DDL + cursor seed |
| `services/orion-substrate-runtime/app/settings.py` | +`ENABLE_CHAT_GRAMMAR_REDUCER`, `CHAT_GRAMMAR_BATCH_LIMIT` |
| `services/orion-substrate-runtime/app/store.py` | +fetch/advance/load/save for chat + cursor registry |
| `services/orion-substrate-runtime/app/worker.py` | +chat poll loop + 3-way cursor dispatch |
| `services/orion-substrate-runtime/.env_example` | +2 new flags |
| `services/orion-field-digester/app/ingest/state_deltas.py` | +chat_turn block → perturbations |
| `config/field/biometrics_lattice.yaml` | +`conversation_load`, `repair_pressure` to node_channels |
| `config/field/orion_field_topology.v1.yaml` | +same two channels |
| `services/orion-field-digester/tests/conftest.py` | New — sys.path for service tests |
| `services/orion-field-digester/tests/test_field_chat_perturbations.py` | New — 5 tests |
| `services/orion-hub/tests/test_hub_grammar_emit.py` | New — 8 tests |
| `services/orion-hub/tests/test_hub_grammar_publish_fail_open.py` | New — 5 tests |
| `tests/test_chat_substrate_reducer.py` | New — 10 tests |
| `tests/test_chat_substrate_pipeline.py` | New — 3 tests |

## Commits

```
7edea207 feat(hub): add chat turn grammar event emitter
781b6a7e feat(substrate): add chat projection and cursor DDL
927ee00c feat(substrate): add chat turn reducer and pipeline
1724b7d4 feat(hub): add chat grammar publisher and wire into chat handler
9496b100 feat(substrate-runtime): wire chat grammar reducer loop
6cadbba8 feat(field-digester): map chat_turn deltas to conversation_load and repair_pressure
b4da3068 fix(tests): move field-digester tests to service dir with conftest
```

## Tests

```bash
# Root (reducer, pipeline, signal bridge)
/mnt/scripts/Orion-Sapienform/.orion_dev/bin/pytest \
  tests/test_chat_substrate_reducer.py \
  tests/test_chat_substrate_pipeline.py \
  tests/test_substrate_signal_bridge.py \
  tests/test_substrate_signal_bridge_e2e.py -q
# 21 passed

# Hub (grammar emit + publish) — hub has its own pytest.ini; run separately
/mnt/scripts/Orion-Sapienform/.orion_dev/bin/pytest \
  services/orion-hub/tests/test_hub_grammar_emit.py \
  services/orion-hub/tests/test_hub_grammar_publish_fail_open.py -q
# 13 passed

# Field digester
/mnt/scripts/Orion-Sapienform/.orion_dev/bin/pytest \
  services/orion-field-digester/tests/test_field_chat_perturbations.py -q
# 5 passed
```

Total: **39 tests, 39 passed**.

> Note: hub tests must run separately from root tests — the hub has its own `pytest.ini` and a conftest that guards `scripts` namespace to prevent `services/orion-hub/scripts/` from being shadowed by the repo-level `scripts/`. This is pre-existing behavior, not a regression.

## Rollout

1. Apply SQL migration: `psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_chat_substrate_loop.sql`
2. Deploy hub with `PUBLISH_HUB_CHAT_GRAMMAR=false` (default — no-op until ready)
3. Deploy substrate-runtime with `ENABLE_CHAT_GRAMMAR_REDUCER=false` (default)
4. When ready to enable: set `PUBLISH_HUB_CHAT_GRAMMAR=true` on hub, then `ENABLE_CHAT_GRAMMAR_REDUCER=true` on substrate-runtime

## Test plan

- [x] `build_chat_turn_grammar_events` returns valid GrammarEventV1 — all events validate
- [x] `user_utterance` atom has `text_value=None` — no raw user text in any trace
- [x] Stable event IDs — same inputs produce same IDs
- [x] `repair_signal` atom present only when `has_repair_signal=True`
- [x] Publisher fail-open — exception in `publish_grammar_event` does not propagate
- [x] `PUBLISH_HUB_CHAT_GRAMMAR=false` (default) — zero Redis calls
- [x] Reducer: noop on empty events, wrong prefix, wrong source service
- [x] StateDeltaV1 has stable `delta_id` for idempotent re-reduction
- [x] Field digester: `chat_turn` delta → `conversation_load` + `repair_pressure` perturbations
- [x] Field digester: `topic_coherence` never reaches the lattice
- [x] Field digester: `noop` operation → no perturbations
- [x] `ENABLE_CHAT_GRAMMAR_REDUCER=false` default — safe to deploy before hub is publishing
- [ ] Live: chat turn → grammar event on `orion:grammar:event` → cursor advances → field perturbed (requires `PUBLISH_HUB_CHAT_GRAMMAR=true`)
