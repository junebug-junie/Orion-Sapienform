## Summary

- The vision-split migration's comment said the split table keeps state "so scripts/undo_vision_split.py can put it back". A reader (and, on 2026-10-16, the post-pull auto-apply) took that as "apply this file with that script". That script only reverses one split.
- The header now says plainly: `scripts/undo_vision_split.py` is the undo tool, not the apply route, and the file is applied with plain psql (the existing `Apply:` line).
- Comment-only change. No SQL changed.

## Outcome moved

A human or tool reading the header no longer gets the wrong apply route. The parser fix from 2026-10-16 (`orion/sql_migration_drift.py::header_apply_script`) already reads this file as "no apply script"; that verdict is unchanged: `(None, None)` before and after.

## Current architecture

`services/orion-sql-db/manual_migration_vision_split_v1.sql` is already applied on the live DB (`vision_individual_split` exists; `vision_crop_observation.individual_id` exists). The post-pull applier applies a changed migration only when it is NOT APPLIED, so this edit triggers no apply.

## Architecture touched

- One migration file's comments.

## Files changed

- `services/orion-sql-db/manual_migration_vision_split_v1.sql`: header text only.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: none
- Compatibility notes: no SQL change; file already applied.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
python -c "header_apply_script(<file>)"  -> (None, None)
pytest -q tests/test_sql_migration_drift_gate.py tests/test_post_pull_apply_migrations.py tests/test_deploy_migration_gate.py
278 passed
python scripts/check_migration_required_by.py -> OK
```

## Evals run

```text
None. Comment-only change; no behaviour to evaluate.
```

## Docker/build/smoke checks

```text
None needed. No runtime change.
```

## Review findings fixed

- None. Comment-only change; review subagent not run for this PR.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low
- Concern: none known. The corpus test in tests/test_sql_migration_drift_gate.py still pins this file as naming no apply script.
- Mitigation: n/a

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3214
