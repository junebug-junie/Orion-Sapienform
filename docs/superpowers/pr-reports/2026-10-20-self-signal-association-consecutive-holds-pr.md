# Self-signal associations: a clause renders only after 6 consecutive hourly holds

## Summary

- Orion's "when this is high, my next hour had X Nx as often" clause is recomputed every hour. The interval was corrected for many pairs at once (Bonferroni) but not for looking at the same pair again every hour, so a fluke could eventually be shown. Now a clause is shown only when its verdict (same direction) held on each of the last 6 consecutive hourly computations. A missing hour or a different verdict resets the run.
- One rule, shared: `held_latest_by_pair` / `index_by_signal(rows, holds=, fresh_since=)` plus the shared 3 h freshness `ASSOCIATION_MAX_AGE` in `orion/attention/self_signal_outcomes.py`. Both readers use it: equilibrium's episode open rows (`SelfSignalOutcomeRollup.renderable()`) and orion-thought's self-readings clause (`refresh_associations`).
- One shared query (`association_history_sql`): the newest row per (signal, outcome, UTC hour), so each reader loads exactly one row per hourly computation (live: 240 rows for 24 pairs x 10 h, ~4 ms); a LIMIT cut drops the oldest hours first.
- Equilibrium seeds the run from persisted rows at start, so a restart does not reset a run that was already going. An unpublished current row never falls back to last hour's row.
- sql-writer: bounded 30-day retention for `self_signal_outcome_association` on the existing periodic retention loop. Cycle budget 70 -> 75 s keeps the documented >= 5 s per table across 15 tables.
- `scripts/analysis/self_signal_outcome_truth.py --persisted` now counts "renderable" with the same rule and prints each pair's current run.

## Outcome moved

Repeated looks over time no longer let a one-hour fluke reach Orion's prompt or an episode open row. Live today: **0 pairs render under the new rule** (`--persisted`: "24 latest associations, 0 with a clause this hour, 0 renderable (held 6 consecutive hours)"). No persisted row has ever had a `tends_more`/`tends_less` verdict (3,858 rows: 2,972 below_min_n, 656 excluded_circular, 230 no_clear_difference), so nothing visible changes today; the rule is in place before the first clause appears.

## Current architecture

PR #2671 (live since 10-14): equilibrium computes ~24 association rows per hour into `self_signal_outcome_association` (append-only, no retention). orion-thought read the newest row per pair (`DISTINCT ON (signal_id, outcome)`, `LIMIT 200`, within 3 h) and equilibrium used only its in-memory latest computation. Any single hour whose Bonferroni interval excluded 1 rendered immediately.

## Architecture touched

- `orion/attention/self_signal_outcomes.py`: `DEFAULT_CONSECUTIVE_HOLDS=6`, `CLAUSE_VERDICTS`, `hourly_runs`, `consecutive_holds`, `held_latest_by_pair`, `best_per_signal`, `index_by_signal(holds, fresh_since)`, `ASSOCIATION_COLUMNS`, `association_history_sql`, `association_history_limit`.
- orion-equilibrium-service: rollup keeps `history` (seeded via new `fetch_association_rows`, extended by each published cycle, trimmed to K+1 hours); new setting.
- orion-thought: reader uses the shared query and rule; new setting.
- orion-sql-writer: retention table registration + setting; cycle budget 75 s.

## Files changed

- `orion/attention/self_signal_outcomes.py`: shared holds rule and history query.
- `services/orion-equilibrium-service/app/self_signal_outcome_rollup.py`: history seed/extend/trim, `renderable()` via the rule.
- `services/orion-equilibrium-service/app/{settings.py,service.py}`, `.env_example`, `docker-compose.yml`, `README.md`: `EQUILIBRIUM_SELF_SIGNAL_OUTCOME_CONSECUTIVE_HOLDS=6`.
- `services/orion-thought/app/self_readings_cue.py`: history fetch, rule with freshness.
- `services/orion-thought/app/settings.py`, `.env_example`, `README.md`: `SELF_SIGNAL_OUTCOME_CONSECUTIVE_HOLDS=6`.
- `services/orion-sql-writer/app/{grammar_truth.py,grammar_retention_loop.py,settings.py,models/self_signal_outcome.py}`, `.env_example`, `docker-compose.yml`, `README.md`: retention.
- `scripts/analysis/self_signal_outcome_truth.py`: `--persisted` renderable = held rule.
- Tests: `tests/test_self_signal_outcomes.py`, `services/orion-equilibrium-service/tests/test_self_signal_outcome_rollup.py`, `services/orion-thought/tests/test_stance_self_readings.py`, `services/orion-sql-writer/tests/test_self_signal_outcome_sql_shape.py`.

## Schema / bus / API changes

- Added: none (no schema, channel, or table change).
- Removed: none.
- Renamed: none.
- Behavior changed: a clause renders only after K consecutive hourly holds; association rows older than 30 days are pruned by sql-writer.
- Compatibility notes: readers only; producer rows unchanged. `index_by_signal` gained keyword args (default K=6).

## Env/config changes

- Added keys: `EQUILIBRIUM_SELF_SIGNAL_OUTCOME_CONSECUTIVE_HOLDS=6` (equilibrium), `SELF_SIGNAL_OUTCOME_CONSECUTIVE_HOLDS=6` (thought), `SELF_SIGNAL_OUTCOME_ASSOCIATION_RETENTION_DAYS=30` (sql-writer).
- Changed: `GRAMMAR_RETENTION_PERIODIC_MAX_CYCLE_SEC` 70 -> 75 (sql-writer; settings default, `.env_example`, compose default).
- Removed / renamed keys: none.
- `.env_example` updated: yes (3 services).
- local `.env` synced: the worktree has no `.env`; the keys were added to the primary checkout's `services/*/.env` directly (missing-key add, same policy as `scripts/sync_local_env_from_example.py`), and sql-writer's local `GRAMMAR_RETENTION_PERIODIC_MAX_CYCLE_SEC` was moved 70 -> 75.
- skipped keys requiring operator action: none.

## Tests run

```text
PYTHONPATH=. pytest tests/test_self_signal_outcomes.py tests/test_attention_self_readings.py orion/metacog/tests/test_evidence_map*.py   -> 214 passed
services/orion-equilibrium-service: tests/test_self_signal_outcome_rollup.py tests/test_self_signal_episode_gate.py evals/test_self_signal_outcome_replay_eval.py   -> 34 passed
services/orion-thought: tests/test_stance_self_readings.py evals/test_self_readings_truth_eval.py   -> 35 passed, 3 skipped (skips are pre-existing template checks)
services/orion-sql-writer: tests/test_self_signal_outcome_sql_shape.py tests/test_grammar_retention_periodic.py tests/test_gpu_pool_state_history.py   -> 78 passed, 1 skipped
python scripts/check_env_template_parity.py   -> PASS (95 services)
```

New cases: K holds renders; K-1 does not; verdict flip and direction flip reset; missing hour resets; same-hour recompute is one computation; per-pair; stale newest dropped; equilibrium == shared rule over the same rows (so both readers agree); restart seeds the run; unpublished current row never falls back; equilibrium drops a clause whose newest computation is > 3 h old, like thought; thought reader asks for >= 3 h + 6 h of history.

## Evals run

```text
services/orion-equilibrium-service/evals/test_self_signal_outcome_replay_eval.py   -> pass (in the 34 above)
services/orion-thought/evals/test_self_readings_truth_eval.py   -> pass (in the 35 above)
python scripts/analysis/self_signal_outcome_truth.py --persisted (live, read-only)
  327 persisted hours, 24 latest associations, 0 with a clause this hour, 0 renderable (held 6 consecutive hours)
  truth-to-rows: PASS
```

## Docker/build/smoke checks

```text
Not run: no Dockerfile, dependency, or port change. Shared history SQL validated read-only on live Postgres
(EXPLAIN ANALYZE: 3.7 ms, 240 rows = 24 pairs x 10 hours).
```

## Review findings fixed

Code review ran in a subagent over `git diff origin/main...HEAD`. No must-fix findings; window/off-by-one, fail-closed behavior, and SQL were verified (including read-only against live Postgres).

- Finding (should): the readers disagreed on stale data. orion-thought drops a clause whose newest computation is > 3 h old; equilibrium had no age check, so a stalled rollup would keep the clause on open rows forever.
  - Fix: `ASSOCIATION_MAX_AGE` (3 h) moved into the shared module; `renderable(now)` applies it via `fresh_since`; both service.py callers pass `now`.
  - Evidence: `test_a_stale_computation_stops_the_open_row_clause_like_thought` (2 h 59 m renders, 3 h 1 m does not).
- Finding (should): thought's "missing-hour" test passed for the wrong reason (removing an hour from 6 left 5 rows, which fails even with no gap rule).
  - Fix: `_held(_ASSOC_ROW, 7, gap_at=3)` keeps 6 rows with the gap inside the run; added a positive case (gap older than the run still renders).
  - Evidence: `test_a_clause_that_has_not_held_six_hours_renders_nothing[missing-hour]`, `test_a_gap_older_than_the_run_does_not_matter`.
- Finding (nit): a LIMIT cut dropped whole pairs from the end of the alphabet, and the limit allowed holds+4 hour buckets while thought's window can span holds+5.
  - Fix: query wraps the per-hour `DISTINCT ON` and orders by `computed_at DESC`, so a cut drops the oldest hours across all pairs; limit uses holds+5; docstring corrected.
  - Evidence: query-shape test; live re-run 240 rows / 24 pairs.
- Finding (nit): equilibrium counts published rows; thought counts persisted rows.
  - Fix: documented in `renderable()` (differs only if sql-writer drops a row, until the next restart re-seeds).
- Finding (nit): `--persisted` counted stale runs as renderable.
  - Fix: passes `fresh_since=now - ASSOCIATION_MAX_AGE`.
- Finding (nit, no change): a failed association-history read on first load now also skips the cycle. Kept: same fail-closed policy as the hour-row read, covered by `test_a_read_failure_skips_the_cycle_and_writes_nothing`.
- Finding (reminder): the local sql-writer `.env` needed 70 -> 75 by hand. Done (primary checkout's `services/orion-sql-writer/.env`).

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && for s in orion-sql-writer orion-equilibrium-service orion-thought; do docker compose --env-file .env --env-file services/$s/.env -f services/$s/docker-compose.yml up -d --build || break; done
```

## Risks / concerns

- Severity: low. Concern: K is two keys (one per reader service); they can drift. Mitigation: same default 6 in code, `.env_example`, and compose; both READMEs name the sibling key.
- Severity: low. Concern: an equilibrium outage of one hour resets every run (6 more hours before a clause can return). Mitigation: intended (fail closed).
- Severity: low. Concern: the first retention pass deletes rows older than 30 days. Today the oldest row is 2026-10-14, so nothing is deleted until 2026-11-13. No DELETE was run against prod by this work.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
