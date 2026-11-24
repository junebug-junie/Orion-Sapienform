## Summary

Follow-up to #3800 (PR #3781 Patch 1), answering Orion's review questions Q2, Q9, Q10 and Q12 on #3781.

- Adds an explicit read grant, so Orion's read-only role can read the new signal history table. The grant lives in `scripts/sql/`, like every other `orion_readonly` grant, so the migration stays auto-applicable.
- Adds two tests. One checks that the grant file opens the table. One checks that the read-both count and event-time queries count each (node_id, observed_at) once.
- Points the migration comment at the grant file.

## Outcome moved

The grant no longer depends on the database default privileges. Live, those defaults already give `orion_readonly` SELECT (`has_table_privilege` = t on 2026-11-24). A rebuilt database would lose them without this file.

## Current architecture

#3800 is merged and live. The table exists, and the dual-write is writing (at the check: 56 `scene_change` rows, 84 `alarm` rows). Nothing reads the table yet.

## Architecture touched

- Postgres grant script only. No change to service code, the schema, the bus or env.

## Files changed

- `scripts/sql/2026-11-24_grant_orion_readonly_node_signal_history.sql`: the explicit GRANT SELECT, with undo and verify lines.
- `services/orion-sql-db/manual_migration_node_signal_history_v1.sql`: comment only. It now names the grant file. The DDL is unchanged and the file still reads as applied.
- `orion/substrate/tests/test_signal_history.py`: the grant test (Q12) and the dedupe test (Q2).

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: none.
- Compatibility notes: none.

## Env/config changes

None.

## Answers to the review questions

- **Q2. Do the read-both reads dedupe?** Yes. The union is `DISTINCT ON (node_id, observed_at)`, and the count and event-time queries wrap that deduped union. A new test pins this. A scratch-Postgres run in #3800 showed the dedupe working: a key in both tables came back once, with the new table's value.
- **Q9. Does strain's event row collide with that tick's sample?** Yes, and that is safe. `_write_prediction_error_node` stamps `observed_at` with the same `now` that the event row uses. The sampler's later row for that reading therefore has the same key, and `ON CONFLICT DO NOTHING` keeps one row with value 1.0. This is the same as in the PE table today.
  - Live check: 14 strain rows with value > 0 in 7 days, and 0 near-duplicate pairs (two rows less than 5 s apart).
  - The backfill copies values unchanged. So the `value > 0` event times the restart seed reads come out the same from read-both as from the PE table.
- **Q10. Is the migration where the gates find it?** Yes, it is in `services/orion-sql-db/`. Both `check_sql_migrations_applied.py` and the post-pull applier found it. The applier applied it on deploy, and the checker now reads it as `applied`.
- **Q12. Can `orion_readonly` read the table?** Yes. Live it already can, through default privileges. This PR adds the explicit grant.

## Tests run

```text
pytest orion/substrate/tests/test_signal_history.py tests/test_curiosity_urgent_prompt.py tests/test_deploy_migration_gate.py tests/test_sql_migration_drift_gate.py -> 168 passed
scripts/check_migration_required_by.py -> OK
auto_apply_refusals(migration, existing_tables=set()) -> []
check_sql_migrations_applied.py --file manual_migration_node_signal_history_v1.sql -> applied
git diff --check -> clean
```

## Evals run

```text
None. The change is a grant and tests. The live read-only checks are under "Answers to the review questions".
```

## Docker/build/smoke checks

```text
None needed. No image changes.
```

## Review findings fixed

- No new code path. The #3800 review findings were fixed in #3800.

## Restart required

```text
No restart required.
```

Apply the grant once (it is idempotent, and today it is a no-op):

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 < scripts/sql/2026-11-24_grant_orion_readonly_node_signal_history.sql
```

## Risks / concerns

- Severity: low. Concern: none beyond the risks already listed in #3800.
- Issues opened: none. Issues closed: none. Refs #3781.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3804

🤖 Generated with [Claude Code](https://claude.com/claude-code)
