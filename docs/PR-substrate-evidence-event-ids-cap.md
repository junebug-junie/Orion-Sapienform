# PR: cap evidence_event_ids to prevent O(N) blowup in pressure reducer

**Branch:** `fix/substrate-evidence-event-ids-cap`  
**File:** `orion/substrate/biometrics_loop/pressure_reducer.py`

---

## Problem

`ActiveNodePressureStateV1.evidence_event_ids` was append-only with no upper bound. Every processed event merged its evidence IDs into the list:

```python
node_state.evidence_event_ids = sorted(set(node_state.evidence_event_ids + evidence))
```

After processing the biometrics backlog (~11K events at ~8 evidence IDs each), `athena.evidence_event_ids` had **412,511 entries**.

`reduce_node_pressure_candidates` calls `deepcopy(projection)` at the top of every invocation. Deepcopying a Pydantic model with a 412K-element list triggers **4 million recursive `copy.deepcopy` calls per event** — measured at ~2.5 seconds per event.

Downstream effects:
- A 500-event batch took ~21 minutes instead of ~5 seconds
- `last_tick_at` appeared frozen (the loop was alive but stuck inside `asyncio.to_thread`)
- `/grammar/truth` classified biometrics as `dead_no_heartbeat` (false alarm — 120s heartbeat threshold exceeded by a 21-minute batch)
- `save_pressure` and `save_receipt` were also slow: the 412K list bloated the projection JSON and the state delta `before`/`after` fields in every receipt

## Fix

Cap `evidence_event_ids` to the 200 most-recent IDs after each merge:

```python
all_ids = sorted(set(node_state.evidence_event_ids + evidence))
node_state.evidence_event_ids = all_ids[-200:]
```

The field tracks which evidence contributed to the current pressure state. Retaining 200 entries is sufficient for operational tracing; full history is in the grammar events table.

## Result

| Metric | Before | After |
|--------|--------|-------|
| 50-event batch time | 127s | ~2.5s |
| Per-event time | ~2,500ms | ~50ms |
| 11K backlog drain time | ~7.7 hours | ~3 minutes |
| `evidence_event_ids` per node | 412,511 | ≤ 200 |
| biometrics classification | `dead_no_heartbeat` | `healthy` |

## Deployment note

The stored pressure projection had 412K entries at the time of the fix. The projection row was manually deleted from `substrate_active_node_pressure_projection` before restarting the container so the first batch wasn't still slow. Future restarts don't need this — the cap kicks in on the first write after any projection load.

## Acceptance checks

- [ ] `classification: healthy` in `/grammar/truth` for biometrics reducer
- [ ] `pending_backlog` drains to 0 within a few minutes of restart
- [ ] `evidence_event_ids` per node ≤ 200 in loaded pressure projection
- [ ] No `dead_no_heartbeat` false positives during steady-state (empty queue, 1s poll interval)
