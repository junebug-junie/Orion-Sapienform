# SQL writer: keep the grammar ledger census off `/health`

## Summary

- Split SQL writer's bounded grammar operational state from its detailed operator diagnostics.
- Route `GET /health` through the operational snapshot only.
- Preserve every existing `GET /grammar/truth` field, including exact fallback counts and `latest_by_source_service`.
- Document the health/truth boundary.

## Why

`GET /health` previously built the complete grammar truth payload. The
`latest_by_source_service` diagnostic performs an exact `COUNT(*)` and
`MAX(created_at)` grouped over every retained `grammar_events` row. Live investigation
found roughly 1.34 million retained rows and a cold/load-sensitive query that could exceed
the grammar connection's statement timeout, turning an operator census into a false health
failure.

Upstream producers and downstream reducers do not consume this aggregate. Producers keep
publishing the existing `GrammarEventV1` contract, SQL writer keeps persisting it, and
substrate-runtime keeps reading source/trace-filtered batches from its existing cursors.

## Contract

- No bus, event, database schema, channel, producer, reducer, cursor, or env changes.
- `/health` keeps its existing response fields and degradation semantics for configuration,
  subscription, queue pressure, required indexes, and retention state.
- `/grammar/truth` keeps all required fields and exact diagnostic semantics.
- No new metric, cache, materialized table, or source-liveness definition is introduced.

## Verification

- `pytest -q services/orion-sql-writer/tests`: **1169 passed, 184 skipped**.
- `pytest -q tests/test_grammar_truth_gate.py`: **6 passed**.
- Focused grammar truth/retention/fallback/routing suite: **97 passed**.
- `python3 -m compileall -q services/orion-sql-writer/app services/orion-sql-writer/tests/test_grammar_truth.py`: passed.
- `scripts/safe_docker_build.sh orion-sql-writer build`: passed using temporary worktree-local
  links to the checked-in env templates; no live env or service was changed.
- `git diff --check`: passed.
- Independent code review: no findings.

## Deployment smoke

After merge and deploy from clean synchronized `main`:

1. Call `/health` repeatedly immediately after restart and confirm it remains fast and does
   not produce the ledger-wide source census in PostgreSQL activity/logs.
2. Call `/grammar/truth` and confirm `grammar_fallbacks` and
   `latest_by_source_service` remain populated.
3. Confirm grammar queue depth, recent fallback counts, and substrate reducer cursors remain
   healthy.
