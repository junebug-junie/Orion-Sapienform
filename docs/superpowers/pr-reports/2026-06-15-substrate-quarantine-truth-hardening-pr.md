# PR: Substrate poison quarantine production-truth hardening

## Summary

PR #707 fixed reducer starvation by splitting substrate reducers into independent poll loops, but poison-event quarantine remained **semantically weak**: quarantine state lived only in-process (`reducer_health.py`), reset on container restart, and `/grammar/truth` could become healthy after history was skipped unless operators had no durable signal.

This PR makes quarantine **production-truth honest** by persisting quarantine rows in Postgres, degrading truth until operator acknowledgement, and exposing bounded quarantine detail in the truth payload and smoke gate.

## Root cause of hardening gap

- Quarantine was recorded via in-process `record_quarantine()` plus a `ReductionReceiptV1` (`quarantine:{reducer_key}:{event_id}`).
- `/grammar/truth` never consulted durable quarantine state; only ephemeral reducer health snapshots.
- Receipts expire (default 6h error retention) and can be pruned — not a reliable ack contract.
- After restart, in-process quarantine lists cleared → truth could falsely pass.

## Changes

### Durable model

New table `substrate_reducer_quarantine` (migration `manual_migration_substrate_reducer_quarantine_v1.sql`):

| Column | Purpose |
|--------|---------|
| `quarantine_id` | `quarantine:{reducer_key}:{event_id}` |
| `reducer_key`, `cursor_name`, `event_id` | Reducer identity |
| `trace_id`, `reason` | Diagnosis |
| `quarantined_at` | When poison was isolated |
| `acknowledged_at`, `acknowledged_by` | Operator ack audit (row retained) |

### New truth payload fields

- `quarantine_by_reducer` — unack count + bounded recent examples per reducer
- `unacknowledged_quarantine_count_by_reducer`
- `reducer_health_by_name[*].unacknowledged_quarantine_count`
- `quarantine_recovery` — endpoint docs in truth payload

### New degraded reason

- `reducer_quarantine_present:<cursor_name>` — emitted when unacknowledged quarantine exists for that cursor

Existing reasons unchanged (`cursor_lag:*`, `reducer_stream_lag:*`, etc.).

### Operator acknowledgement

`POST /grammar/quarantine/ack` — same auth as cursor reset (`X-Orion-Operator-Token` / `SUBSTRATE_CURSOR_RESET_OPERATOR_TOKEN`):

```bash
# Single event
curl -X POST \
  -H "X-Orion-Operator-Token: $SUBSTRATE_CURSOR_RESET_OPERATOR_TOKEN" \
  "http://127.0.0.1:8115/grammar/quarantine/ack?cursor_name=transport_grammar_reducer&event_id=gev_x"

# All unacked for cursor
curl -X POST \
  -H "X-Orion-Operator-Token: $SUBSTRATE_CURSOR_RESET_OPERATOR_TOKEN" \
  "http://127.0.0.1:8115/grammar/quarantine/ack?cursor_name=transport_grammar_reducer&ack_all=true"
```

Ack sets `acknowledged_at` / `acknowledged_by`; does **not** delete rows. Clears degraded quarantine reason when no unacked rows remain.

### Smoke gate

`grammar_truth_gate.py` classifies quarantine under `reducer_quarantine=[reducer_quarantine_present:...]`. Reducer health summary includes `unack_quarantine` count.

## Files changed

- `services/orion-sql-db/manual_migration_substrate_reducer_quarantine_v1.sql` (new)
- `services/orion-substrate-runtime/app/store.py`
- `services/orion-substrate-runtime/app/quarantine_ack.py` (new)
- `services/orion-substrate-runtime/app/grammar_truth.py`
- `services/orion-substrate-runtime/app/worker.py`
- `services/orion-substrate-runtime/app/main.py`
- `services/orion-substrate-runtime/app/reducer_health.py`
- `scripts/grammar_truth_gate.py`
- `services/orion-substrate-runtime/.env_example`
- `services/orion-substrate-runtime/README.md`
- Tests: `test_quarantine_truth.py`, `test_worker_independent_reducers.py`, updates to existing reducer/truth/gate tests

## Tests added

| Test file | Coverage |
|-----------|----------|
| `test_quarantine_truth.py` | Truth degrades on unack quarantine; survives health reset; ack auth + audit |
| `test_worker_independent_reducers.py` | Transport advances while biometrics blocked; independent poll tasks |
| `test_worker_reducer.py` | Poison quarantine calls `save_quarantine` |
| `test_grammar_truth_gate.py` | Quarantine degraded group classification |

## Verification

```bash
PYTHONPATH=. ./venv/bin/python -m pytest \
  services/orion-substrate-runtime/tests/test_reducer_health.py \
  services/orion-substrate-runtime/tests/test_worker_reducer.py \
  services/orion-substrate-runtime/tests/test_grammar_truth_reducer_health.py \
  services/orion-substrate-runtime/tests/test_quarantine_truth.py \
  services/orion-substrate-runtime/tests/test_worker_independent_reducers.py \
  tests/test_grammar_truth_gate.py -q
```

Result: **22 passed**

```bash
PYTHONPATH=. ./venv/bin/python -m compileall services/orion-substrate-runtime/app scripts/grammar_truth_gate.py -q
```

Result: **exit 0**

## Deploy notes

1. Apply migration before or with deploy:
   ```bash
   psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_substrate_reducer_quarantine_v1.sql
   ```
2. No new env keys (ack reuses `SUBSTRATE_CURSOR_RESET_OPERATOR_TOKEN`).
3. Restart `orion-substrate-runtime` after migration.

## Non-goals (preserved)

- PR #707 independent reducer loops unchanged
- Cursor lag thresholds unchanged
- No silent deletion of quarantine evidence
