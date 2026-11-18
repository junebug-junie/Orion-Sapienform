## Summary

- The unapplied-migration gate (#2455) replayed `*_rollback.sql` files as if they were applied, so the expected schema lost every object a rollback drops. Since 2026-10-04 the 10-minute substrate ladder watch was RED on every run (300+ consecutive) with "should have been dropped" for 4 rollback files.
- Fix: `orion/sql_migration_drift.py` skips files named `*_rollback.sql` (`is_rollback_file`). A rollback is applied only when backing a migration out, never part of the expected schema.
- The noise hid one real miss: `manual_migration_memory_confirmation_v1.sql` (merged 2026-10-06) had 2 of 4 indexes missing. Applied live 2026-10-07 (tables of 14 and 40 rows).

## Outcome moved

The watch goes back to meaning something: live run with the fix reports 122 applied, 0 missing (was 4 false RED + 1 real RED buried among them).

## Current architecture

`load_files()` globbed every `*.sql` in services/orion-sql-db and replayed them in merge order.

## Architecture touched

`orion/sql_migration_drift.py` (shared by `scripts/check_sql_migrations_applied.py` and `make substrate-ladder-watch`), its test.

## Files changed

- `orion/sql_migration_drift.py`: `is_rollback_file()`; `load_files()` skips rollbacks.
- `tests/test_sql_migration_drift_gate.py`: regression test (forward + rollback pair → only forward replayed).
- this report.

## Schema / bus / API changes

- None.

## Env/config changes

- None.

## Tests run

```text
pytest tests/test_sql_migration_drift_gate.py tests/scripts/test_substrate_ladder_liveness.py -> 126 passed
```

## Evals run

```text
Live read-only run of scripts/check_sql_migrations_applied.py against production Postgres:
  before fix (main):  4 false RED (rollback files) + 1 real RED (memory_confirmation_v1)
  after fix + applying memory_confirmation_v1: 131 replayed, 122 applied, 0 missing
```

## Docker/build/smoke checks

```text
None needed: host-side script run by cron from the primary checkout.
```

## Review findings fixed

- Not run through a review subagent: 5-line filter plus a regression test with the live before/after above.

## Restart required

```text
No restart. After merge: git pull in the primary checkout; the existing cron picks it up next tick.
```

## Risks / concerns

- Severity: low. Concern: relies on the `_rollback.sql` naming convention (all 5 existing rollback files follow it and start with "-- Rollback of"). A rollback named differently would still be replayed and alarm loudly, which is the safe direction.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2527
