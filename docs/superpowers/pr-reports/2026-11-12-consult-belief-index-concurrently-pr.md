## Summary

- The consult belief migration (#3246) built its index without `CONCURRENTLY` on a table that already exists. The post-pull applier refused it (correctly: that would block the table's writers while it builds), and the deploy wrapper then refused to rebuild orion-hub.
- The index is now `CREATE INDEX CONCURRENTLY IF NOT EXISTS`, so the file is on the auto-apply allowlist.

## Outcome moved

Deploy of 759d268f5 (2026-11-12 21:30Z): `REFUSED ... index on existing table orion_consult without CONCURRENTLY`, then `orion-hub up -d --build rc=1`. The migration was later applied by hand; the live index is valid (`indisvalid = t`). This file now passes `auto_apply_refusals(text, {"orion_consult"})` with no refusal, so a fresh database applies it on pull.

## Current architecture

`services/orion-sql-db/manual_migration_orion_consult_belief_v1.sql`: 5 `ADD COLUMN IF NOT EXISTS` plus one plain `CREATE INDEX`.

## Architecture touched

That migration file only.

## Files changed

- `services/orion-sql-db/manual_migration_orion_consult_belief_v1.sql`: index built CONCURRENTLY.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: none live (already applied; `IF NOT EXISTS` makes it a no-op).
- Compatibility notes: the applier runs a CONCURRENTLY file outside a single transaction.

## Env/config changes

- None. `.env_example` not changed; no sync needed.

## Tests run

```text
auto_apply_refusals(file, {"orion_consult"}) -> []
check_sql_migrations_applied.py --service orion-hub -> all 18 required migration(s) applied
```

## Evals run

```text
None: a one-line DDL change.
```

## Docker/build/smoke checks

```text
Not run. Hub still needs the rebuild that the refusal skipped (see Restart required).
```

## Review findings fixed

- No review subagent: a one-statement change checked by the applier's own rule function.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low. Concern: none live. Cause of the miss: the PR's dry-run was run before the index line was added and not re-run.

## PR link

PR_LINK

🤖 Generated with [Claude Code](https://claude.com/claude-code)
