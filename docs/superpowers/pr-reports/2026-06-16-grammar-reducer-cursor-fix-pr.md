# PR: Fix stale grammar reducer cursors (execution + transport)

## Summary

`./scripts/grammar_production_truth.sh` correctly failed because `transport_grammar_reducer` was ~16h behind with **72k+ pending** `bus.transport:*` grammar events. Root cause was **throughput starvation**: all reducers shared one serial poll loop at **50 events / 5s**, so transport could not catch up while biometrics and execution consumed the loop.

This PR splits reducers into **independent poll loops**, raises the transport batch limit (**500**), adds **reducer health classification** to `/grammar/truth`, improves the truth smoke output, and adds poison-event quarantine with bounded retries.

## Root cause

| Finding | Evidence |
|---------|----------|
| Transport cursor frozen while worker alive | `updated_at` recent, `last_event_created_at` stuck at `2026-06-15T11:36:46Z` |
| Massive backlog | `SELECT COUNT(*) …` → **75,456** pending transport events |
| Fetch worked; serial loop starved transport | Manual `_transport_tick` advanced cursor; live poll advanced only ~minutes of events over hours |
| Execution mildly behind | ~400s wall lag (within 6h threshold) |

## Behavior before / after

| Aspect | Before | After |
|--------|--------|-------|
| Poll architecture | Single loop: biometrics → execution → transport | **3 independent asyncio loops** |
| Transport batch | 50 / tick | **500 / tick** (configurable) |
| Truth payload | `cursor_lag_by_reducer` only | + `stream_lag_by_reducer`, `pending_backlog_by_reducer`, `reducer_health_by_name` |
| Truth script output | Generic degraded reasons | **Grouped reasons** + reducer health block |
| Poison events | Batch failure blocked cursor | Per-event isolation; quarantine after `REDUCER_POISON_MAX_RETRIES` |
| Live catch-up (deployed image) | ~16h lag static | Transport cursor advanced **11:36 → 12:25+** within first minute of new build |

## Files changed

- `services/orion-substrate-runtime/app/worker.py` — parallel loops, poison quarantine, cursor commit logging
- `services/orion-substrate-runtime/app/reducer_health.py` — **new** in-process health snapshots
- `services/orion-substrate-runtime/app/grammar_truth.py` — backlog/stream lag/health in truth payload
- `services/orion-substrate-runtime/app/store.py` — `grammar_cursor_metrics()`
- `services/orion-substrate-runtime/app/settings.py` — batch limits, heartbeat/poison settings
- `services/orion-substrate-runtime/docker-compose.yml`, `.env_example`, `README.md`
- `scripts/grammar_truth_gate.py`, `scripts/grammar_production_truth.sh`
- Tests: `test_reducer_health.py`, `test_worker_reducer.py`, `test_grammar_truth_reducer_health.py`, `test_grammar_truth_gate.py`

## Verification

```bash
# Unit tests (exit 0, 12 passed)
PYTHONPATH=. ./venv/bin/python -m pytest \
  services/orion-substrate-runtime/tests/test_reducer_health.py \
  services/orion-substrate-runtime/tests/test_worker_reducer.py \
  services/orion-substrate-runtime/tests/test_grammar_truth_reducer_health.py \
  tests/test_grammar_truth_gate.py -q

# Live stack — rebuilt substrate-runtime container
PROJECT=orion-athena docker compose --env-file services/orion-substrate-runtime/.env up -d --build

# Truth smoke — still FAIL until transport backlog clears (expected, not weakened)
./scripts/grammar_production_truth.sh
# substrate-runtime: degraded (cursor_lag:transport_grammar_reducer, reducer_stream_lag:transport_grammar_reducer)
# transport_bus: class=alive_behind backlog≈72k heartbeat=recent
```

## Operator actions

1. **Deploy** rebuilt `orion-substrate-runtime` (already restarted on host with new image).
2. **Wait for catch-up** (~72k events at 500/5s ≈ **12–15 min** if ingress rate stays moderate) **or** operator reset:
   ```bash
   curl -X POST -H "X-Orion-Operator-Token: $SUBSTRATE_CURSOR_RESET_OPERATOR_TOKEN" \
     'http://127.0.0.1:8115/grammar/cursor/reset?cursor_name=transport_grammar_reducer&mode=tail'
   ```
   (`mode=tail` marks truth degraded for skipped history — use only if catch-up time is unacceptable.)
3. Re-run `./scripts/grammar_production_truth.sh` after cursor stream lag < 6h.

## Remaining risks

- Transport truth will remain **degraded until backlog clears**; this is intentional.
- In-process reducer health resets on container restart (heartbeat history is not persisted).
- No Redis consumer-group integration — reducers poll Postgres `grammar_events`, not Redis streams directly.

## Non-goals

- Did not raise `SUBSTRATE_CURSOR_LAG_RESYNC_HOURS` or remove `cursor_lag:*` checks.
- Did not change bus channel/schema contracts.
