# PR: Substrate receipt retention + compaction

**Branch:** `feat/substrate-receipt-retention-v1`  
**Base:** `main`

## Summary

Containment fix for unbounded `substrate_reduction_receipts` growth (~318 GB incident). Success receipts are stored metadata-only (keeping `state_deltas` for field digestion), errors/debug samples retain full JSON with longer TTL, and a 15-minute prune loop deletes only expired rows whose deltas are already applied (or error receipts). Emergency metadata-only mode and emergency prune run from the pruner loop under disk/table pressure — not on every insert.

**Plan:** `docs/superpowers/plans/2026-05-29-substrate-receipt-retention-v1.md`

## Files changed

| File | Change |
|------|--------|
| `services/orion-sql-db/manual_migration_receipt_retention_v1.sql` | Retention columns + indexes + legacy backfill |
| `orion/substrate/receipts/retention.py` | Classify, compact, hash, expires_at helpers |
| `orion/substrate/receipts/__init__.py` | Package exports |
| `services/orion-substrate-runtime/app/store.py` | Retention-aware `save_receipt()` |
| `services/orion-substrate-runtime/app/receipt_pruner.py` | Safe + emergency prune, pressure cache |
| `services/orion-substrate-runtime/app/worker.py` | 15m prune loop |
| `services/orion-substrate-runtime/app/settings.py` | `ORION_RECEIPT_*` settings |
| `services/orion-substrate-runtime/.env_example` | Documented defaults |
| `services/orion-substrate-runtime/docker-compose.yml` | Env passthrough |
| `tests/test_receipt_retention.py` | Classification + insert param tests |
| `tests/test_receipt_pruner.py` | Safe/emergency SQL + cooldown tests |

## Operator steps (required before/after merge)

### 1. Apply migration

```bash
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_receipt_retention_v1.sql
```

### 2. Add env keys to `services/orion-substrate-runtime/.env`

Copy from `.env_example` (already applied on athena host):

```bash
ORION_RECEIPT_RETENTION_SUCCESS_HOURS=48
ORION_RECEIPT_RETENTION_ERROR_DAYS=7
ORION_RECEIPT_FULL_PAYLOAD_SUCCESS=false
ORION_RECEIPT_FULL_PAYLOAD_SAMPLE_RATE=0.01
ORION_RECEIPT_MAX_TABLE_GB=25
ORION_RECEIPT_WARN_TABLE_GB=15
ORION_RECEIPT_CRITICAL_TABLE_GB=20
ORION_RECEIPT_EMERGENCY_METADATA_ONLY=true
ORION_RECEIPT_PRUNE_INTERVAL_SEC=900
ORION_RECEIPT_POSTGRES_DATA_PATH=/mnt/postgres
ORION_RECEIPT_DISK_CRITICAL_PCT=85
```

### 3. Rebuild substrate-runtime

```bash
docker compose -f services/orion-substrate-runtime/docker-compose.yml up -d --build
```

## Test plan

- [x] `PYTHONPATH=. pytest tests/test_receipt_retention.py tests/test_receipt_pruner.py tests/test_execution_substrate_pipeline.py tests/test_biometrics_pipeline.py -v` — 15 passed
- [x] Migration applied on athena Postgres
- [x] Live deploy: new rows `is_full_payload=false`, no `accepted_event_ids` in JSON
- [x] Field digester advancing (`substrate_field_applied_deltas` growing)
- [x] Safe prune SQL executes; emergency prune off insert hot path (post-review fix)
- [ ] Monitor table size over 24h (target: stay under 25 GB)

## Verification SQL

```sql
-- Recent receipt shape
SELECT receipt_id, receipt_kind, is_full_payload, payload_bytes,
       (receipt_json ? 'accepted_event_ids') AS has_event_lists,
       expires_at
FROM substrate_reduction_receipts
ORDER BY created_at DESC LIMIT 10;

-- Table size
SELECT pg_size_pretty(pg_total_relation_size('substrate_reduction_receipts'));
```

## Known risks / follow-ups

- **Execution receipts still ~7 MB** — compact mode keeps fat `state_deltas.before/after`; biometrics rows ~1.6 KB. Post-MVP: trim execution delta blobs.
- **Legacy backfill** — migration sets `expires_at = created_at + 48h` on existing rows; safe prune protects unapplied deltas but first prune on huge tables may be heavy.
- **25 GB hard cap** — advisory only today (warn 15 / critical 20 GB); enforce in follow-up if needed.
- **Disk pressure in container** — `/mnt/postgres` not mounted in compose; table-size pressure still works via SQL.

## Diff summary

- Metadata-only success writes with `expires_at`, denormalized columns, payload hash/bytes
- Safe prune: expired + (error OR all deltas applied)
- Emergency prune: pruner loop only, cooldown, applied-delta guard on success/debug deletes
- No cognition redesign; field digester unchanged
