## Summary

- Adds a `created_at` index to `gpu_pool_events`. sql-writer's 60 s retention filters on `created_at`, but every secondary index was on `generated_at`, so each cycle read the whole table twice.
- Migration `services/orion-sql-db/manual_migration_gpu_pool_events_created_at_index.sql`: `CREATE INDEX CONCURRENTLY IF NOT EXISTS`, so there is no write lock on the live table. The post-pull applier auto-applies it.
- The same index is on the ORM model, so fresh databases get it from `create_all`.
- Fixes #3763.

## Outcome moved

The retention delete and the debt count become index range scans instead of full-table scans. Measured live before the patch (2026-11-25): about 876k rows (30-day retention), about 150 ms per parallel seq scan, about 2 scans a minute, all from cache. The CPU cost is small today, but it grows with table size and with the retention window.

## Current architecture

`grammar_truth._batch_delete_sql` / `_count_debt_sql` both use `WHERE created_at < :cutoff`. The live indexes were: pkey(event_id), (generated_at), (lease_id, generated_at), (work_class, generated_at).

## Files changed

- `services/orion-sql-db/manual_migration_gpu_pool_events_created_at_index.sql`: new additive migration.
- `services/orion-sql-writer/app/models/gpu_pool_event.py`: model index.
- `services/orion-sql-writer/tests/test_gpu_pool_event_sql_shape.py`: test that both the model and the migration carry the index.

## Schema / bus / API changes

- Added: index `idx_gpu_pool_events_created`. No bus or schema change.

## Env/config changes

None.

## Tests run

```text
sql-writer tests/test_gpu_pool_event_sql_shape.py: 5 passed
tests/test_sql_migration_drift_gate.py: 90 passed
(reviewer) drift gate + post-pull applier + deploy migration gate: 280 passed; gpu-pool schema drift gate: 4 passed
scripts/check_migration_required_by.py --base origin/main: OK
```

## Evals run

None. This is an index-only change and no behavior changes. The live check after deploy is the evidence (below).

## Docker/build/smoke checks

```text
apply_post_pull_migrations.py --dry-run: NOT APPLIED -> would apply (no transaction wrapper: CREATE INDEX CONCURRENTLY, lock_timeout=10s)
```

## Review findings fixed

- Finding: `none` reason in the REQUIRED-BY header lost its closing paren when parsed (cosmetic).
  - Fix: rewrote the header without parentheses.
  - Evidence: no other findings. The reviewer confirmed the planner uses the btree for the delete's ORDER BY created_at LIMIT shape, that `create_all` does not touch existing-table indexes, and that all gates pass.

## Restart required

No restart required. The post-pull applier creates the index. Verify with:

```bash
docker exec orion-athena-sql-db psql -U postgres -d conjourney -c "select indexname from pg_indexes where tablename='gpu_pool_events'"
```

Then confirm `seq_scan` on `gpu_pool_events` stops climbing (`pg_stat_user_tables`).

## Risks / concerns

- Severity: low. Concern: an interrupted CONCURRENTLY build leaves an INVALID index. Mitigation: the applier refuses that state for a human, and the file says to drop and re-run.
- Issues closed: #3763. Issues opened: none.
