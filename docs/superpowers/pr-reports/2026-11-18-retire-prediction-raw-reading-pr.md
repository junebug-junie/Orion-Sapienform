Needs Juniper: apply by hand after review

Apply (one line, from the primary checkout after merge):

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 < services/orion-sql-db/manual_migration_retire_prediction_raw_reading_drop.sql
```

Snapshot (16,100 rows, 346,437 bytes, sha256 `18146ca75d6ddd92be0f698dd0cc4ad630d8d302880c1f46358d2f99c11754c3`):

- `/mnt/storage-warm/orion-backups/substrate_prediction_raw_reading_2026-11-18.dump` (durable)
- `/tmp/substrate_prediction_raw_reading_retire/snapshot.dump` (same file; /tmp clears on reboot)

Restore:

```bash
docker exec -i orion-athena-sql-db pg_restore -U postgres -d conjourney < /mnt/storage-warm/orion-backups/substrate_prediction_raw_reading_2026-11-18.dump
```

Fixes #3505

## Summary

- Stages the drop of `substrate_prediction_raw_reading`, a table nothing has written or read since 2026-11-03.
- New hand-applied migration, marked `-- DESTRUCTIVE` with `REQUIRED-BY: none`, so the post-pull applier refuses it and no deploy waits on it.
- Took a read-only `pg_dump` snapshot first and recorded the restore command in the migration header.
- Updates the substrate runtime README line that said the table is kept.
- Nothing was written to the database. The drop runs only when Juniper applies it.

## Outcome moved

A dead table leaves the schema once Juniper applies the file. Until then nothing changes at runtime.

## Current architecture

The rung 6 expectation tuner wrote raw prediction-error readings here (execution, chat, street_handoff). PR for the tuner retirement (2026-11-03) stopped the writer and deleted the reader, and kept the table. Issue #3505 flagged it as dead.

Deadness evidence (re-checked 2026-11-18):

- `rg -n substrate_prediction_raw_reading` over the whole repo (all file types): only `services/orion-sql-db/manual_migration_prediction_raw_reading_v1.sql`, the substrate runtime README, and historical docs/PR reports. No code, YAML, config, dbt/semantic layer or Grafana reference (no dbt/semantic/grafana directory references it).
- Live rows: execution 14,209 (last 2026-11-03 09:44:50Z), chat 1,879 (last 09:28:30Z), street_handoff 12 (last 2026-11-02 19:12:56Z).
- `pg_stat_user_tables`: n_tup_ins 29,295, n_tup_upd 0, n_tup_del 13,195 (cumulative), last_autoanalyze 2026-11-03 00:48Z. The counter did not move across this session.
- `pg_depend`: only the row type, a column default and the primary key. No view (`information_schema.view_table_usage` empty), no foreign key, no trigger, no function body, no materialized view mentions it. No live query in `pg_stat_activity`.

## Architecture touched

- `services/orion-sql-db` migration corpus only. No service code, bus, schema or env change.

## Files changed

- `services/orion-sql-db/manual_migration_retire_prediction_raw_reading_drop.sql`: the DESTRUCTIVE drop, with evidence, snapshot path, check, apply and restore commands.
- `services/orion-substrate-runtime/README.md`: "kept, not dropped" now points at the staged drop.
- `docs/superpowers/pr-reports/2026-11-18-retire-prediction-raw-reading-pr.md`: this report.

`manual_migration_prediction_raw_reading_v1.sql` stays. The repo keeps a retired table's create file and lets the later drop supersede it (same as the feedback/consolidation rung retirement). The drift checker confirms it reads the old file as `superseded`.

## Schema / bus / API changes

- Added: none
- Removed: table `substrate_prediction_raw_reading` (when applied)
- Renamed: none
- Behavior changed: none at runtime
- Compatibility notes: no reader or writer exists

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
python3 scripts/check_migration_required_by.py
  migration REQUIRED-BY declarations OK (1 new migration file(s) vs origin/main)
python -m pytest -q tests/test_deploy_migration_gate.py tests/test_sql_migration_drift_gate.py tests/test_post_pull_apply_migrations.py
  278 passed
python scripts/apply_post_pull_migrations.py --dry-run --repo . services/orion-sql-db/manual_migration_retire_prediction_raw_reading_drop.sql
  -- DESTRUCTIVE file: operator-approved only; read it and decide by hand
python scripts/check_sql_migrations_applied.py --file manual_migration_prediction_raw_reading_v1.sql
  superseded (dropped later by manual_migration_retire_prediction_raw_reading_drop.sql)
```

## Evals run

```text
None. A table drop has no eval surface; the checks above are the gate.
```

## Docker/build/smoke checks

```text
No build: no service code changed.
Read-only snapshot: pg_dump -Fc, pg_restore --list shows TABLE, TABLE DATA, pkey, index; 16,100 data rows restored to stdout.
```

## Review findings fixed

- None. Migration-only change; the header/applier/drift gates above are the review for this shape.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low
- Concern: after merge and before the hand apply, `check_sql_migrations_applied.py` (and the watch that runs it) shows this file RED as `DROP_NOT_APPLIED`. That is expected and clears once applied.
- Mitigation: apply soon after merge, or leave it red as the reminder.
- Issues opened: none. Issues closed: #3505 (on merge).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3706

🤖 Generated with [Claude Code](https://claude.com/claude-code)
