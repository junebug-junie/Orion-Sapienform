## Summary

- topic-foundry's startup schema step (`ensure_tables`) ran 15 `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` on every start. Each takes an ACCESS EXCLUSIVE lock even when the column already exists.
- It now reads `information_schema.columns` and only ALTERs columns that are missing. With a current schema, startup takes no exclusive lock.

## Outcome moved

- **Live 2026-10-31 04:32 UTC:** a 27-minute `pg_dump` (`COPY substrate_perception_embedding_baseline`) held a read lock. The redeployed topic-foundry sat at its first ALTER for 11+ minutes, reading "unhealthy", and every later reader of `topic_foundry_runs` queued behind its waiting ALTER.
- **Throwaway Postgres, with a session holding a read lock on `topic_foundry_runs`:**
  - new `ensure_tables`: 0.03 s
  - `main`: blocked until the 15 s test timeout

## Current architecture

`app/storage/repository.py:ensure_tables` runs CREATE TABLE IF NOT EXISTS, 15 ADD COLUMN IF NOT EXISTS and 3 CREATE INDEX IF NOT EXISTS at every start (lifespan).

## Architecture touched

`services/orion-topic-foundry/app/storage/repository.py` only.

## Files changed

- `services/orion-topic-foundry/app/storage/repository.py`:
  - `ADDED_COLUMNS` (the same 15 columns)
  - `missing_columns(cur)`
  - `ensure_tables` alters only the missing ones; indexes come after the columns
- `services/orion-topic-foundry/tests/test_startup_ddl_lock.py`: a current schema runs no ALTER; a missing column is still added.

## Schema / bus / API changes

- None. Same columns, same indexes.

## Env/config changes

- None.

## Tests run

```text
tests/test_startup_ddl_lock.py   2 passed
tests (--continue-on-collection-errors): 117 passed, 1 failed, 7 collection errors
main, same command:                      115 passed, 1 failed, 7 collection errors
  (same pre-existing failure; collection errors are a missing local sklearn)
Throwaway Postgres 16: first start creates the schema in 0.12 s;
  under a held read lock the new code takes 0.03 s and main blocks (killed at 15 s)
```

## Evals run

```text
n/a (schema startup)
```

## Docker/build/smoke checks

```text
Deploy: orion-topic-foundry post-pull rebuild on athena.
```

## Review findings fixed

- See the review summary added after the subagent review.

## Restart required

```bash
# athena: post-pull rebuild of orion-topic-foundry (automatic on git pull of primary main)
```

## Risks / concerns

- Severity: low. Concern: a genuinely new column still needs the exclusive lock, so it can wait behind a backup once, on the deploy that adds it. Mitigation: that is a one-time schema change, not every start.

## PR link

(this PR)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
