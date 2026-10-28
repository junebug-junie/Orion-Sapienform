## Summary

- orion-prediction now keeps only the action-bet columns it knows when it reads a bet row (`PostgresStore._bet`).
- Why: the post-pull rebuild applies a migration before it rebuilds the service. The next PR adds a `shadow` column to `prediction_action_bet`. With `RETURNING *` / `SELECT *` and `ActionBetV1(extra="forbid")`, the old service would have answered every place, settle and list call with a 500 until its rebuild finished, or for good if the rebuild failed.
- Ship this first. The world-model shadow PR (next PR) then adds the column safely.

## Outcome moved

A new column on `prediction_action_bet` no longer breaks the running service.

## Current architecture

Bet reads passed every database column to `ActionBetV1`, which forbids unknown fields.

## Architecture touched

orion-prediction store only.

## Files changed

- `services/orion-prediction/app/store.py`: `_bet` filters to `ACTION_BET_COLS`.
- `services/orion-prediction/tests/test_action_bets.py`: a row with an unknown column validates after filtering.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
PYTHONPATH=<repo> pytest services/orion-prediction/tests/test_action_bets.py tests/test_schemas.py -q -> 24 passed
```

## Evals run

```text
None: storage-shape change only.
```

## Docker/build/smoke checks

```text
None. The post-pull rebuild covers orion-prediction.
```

## Review findings fixed

- Raised by the review of the world-model shadow branch (HIGH 2): a migration-first deploy breaks reads in the old service. This PR is the fix, shipped before the migration.

## Restart required

```bash
git pull   # rebuilds orion-prediction
```

## Risks / concerns

- Severity: low
  - Concern: none known. Reads drop only columns this code never used.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2938

🤖 Generated with [Claude Code](https://claude.com/claude-code)
