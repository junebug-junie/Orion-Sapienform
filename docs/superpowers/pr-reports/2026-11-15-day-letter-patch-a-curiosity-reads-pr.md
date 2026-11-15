# Patch A: curiosity reads and the SQL guard

Part of the fix plan in PR #3439 (`docs/superpowers/specs/2026-11-15-day-letter-findings-fix-design.md`).
Fixes #3398. Fixes #3394. Fixes #3401.

## Summary

- The list of "your last runs" that starts every curiosity run now shows the newest runs. It had been stuck on runs from late September.
- An idle curiosity tick now says it was idle. Before, it saved an empty row with no reason, and Orion read those rows as decisions that left no trace.
- Orion's SQL is now stopped when a Postgres regex uses `\b`. In Postgres `\b` is a backspace, so the pattern matches nothing and the count reads as 0. The block message says to use `\y`.
- The dig help text has one line about `\y`.

## Outcome moved

- Run menu: read live through the new code on 2026-11-15, the newest run listed is `d3f6ab8e4143`, written 2026-11-14 22:19Z. The old query on the same graph kept the oldest 200 of 752 joined rows.
- Supervisor run order: 529 of 529 runs now get a timestamp. Before, every run past the first 200 rows sorted as "unknown".

## Current architecture

- `orion/curiosity/worldview.py` read run outcomes joined to their priors with `LIMIT 200` and no `ORDER BY`, then sorted in Python. The sort was right; the 200 rows it sorted were the wrong ones.
- `orion/curiosity/supervisor.py:build_run_order` reused the same read to order all of history.
- `services/orion-substrate-runtime/app/worker.py` saved `[]` with no gate when a tick had no seeds.
- Orion's turns run plain `psql` through Bash. A pre-tool hook (`orion/fcc/repeat_failure_breaker.py`) already runs before every tool call.

## Architecture touched

- Curiosity run read: now two reads. Read 1 is one row per run outcome, `ORDER BY t.written_at DESC LIMIT 2000`, no join. Python still decides the real order, because 8 of 551 live stamps are ISO strings. Read 2 fetches the claims of only the runs shown.
- Idle tick: `gate={"gate_result": "idle_no_seeds"}` on the no-seeds branch.
- Pre-tool hook: one more rule, checked before the transcript rules. It needs no transcript.

## Files changed

- `orion/curiosity/worldview.py`: ordered outcome read, `recent_run_claims_cypher`, `read_recent_runs`, truncation warning.
- `orion/curiosity/supervisor.py`: docstring for the new read. Behavior comes from the query change.
- `services/orion-substrate-runtime/app/worker.py`: idle gate value.
- `scripts/analysis/eval_system_one_appraisal.py`: counts idle ticks apart, so they do not inflate "seeded ticks with a gate".
- `orion/fcc/repeat_failure_breaker.py`: `decide_psql_backspace`, `psql_backspace_message`, wired into `run_hook`.
- `orion/dig/__main__.py`: one HELP line (the help text has a 3,750 character cap; it still fits).
- Tests: `tests/test_curiosity_worldview.py`, `tests/test_system_one_appraisal_idle_gate.py`, `orion/fcc/tests/test_repeat_failure_breaker.py`, two files in `services/orion-substrate-runtime/tests/`.

## Schema / bus / API changes

- Added: `gate_json.gate_result = "idle_no_seeds"` (a new value in an existing JSON field).
- Removed: the `claim` column from `RECENT_RUNS_CYPHER`. Claims now come from `recent_run_claims_cypher`.
- Renamed: none.
- Behavior changed: `RECENT_RUNS_LIMIT` is 2000 run outcomes (was 200 joined rows).
- Compatibility notes: no bus channel, table, or schema registry entry changed.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed, no template changed.
- skipped keys requiring operator action: none.

## Tests run

```text
.venv/bin/python -m pytest tests/test_curiosity_worldview.py tests/test_curiosity_supervisor.py \
  orion/fcc/tests/test_repeat_failure_breaker.py orion/harness/tests/test_fcc_motor_repeat_failure_breaker.py \
  orion/dig services/orion-cortex-exec/tests/test_situation_curiosity_reverie_context.py
  -> all pass except orion/fcc/tests/test_claude_spawn.py::test_no_undiscovered_root_callers_of_claude_permission_argv,
     which fails the same way on the untouched main checkout.
.venv/bin/python -m pytest tests/test_system_one_appraisal_idle_gate.py tests/test_system_one_appraisal.py -> 9 passed
services/orion-substrate-runtime: pytest tests --continue-on-collection-errors
  -> 555 passed. 28 failures and errors, the same 28 test names as on the untouched main checkout (local environment).
```

The regression test `test_the_old_join_shape_would_have_shown_the_oldest_runs` runs the old query on the same 400-run fixture and shows it returns run 99 of 400 as "newest".

## Evals run

```text
None. These are read and guard fixes; there is no eval harness for the run menu or the hook.
Live read-only check instead: read_recent_runs() and build_run_order() against orion_worldview on 2026-11-15 (numbers above).
```

## Docker/build/smoke checks

```text
Not run. Nothing is deployed from a worktree.
```

Live checks that stay UNVERIFIED until deploy:

- #3398: the run menu inside a real curiosity kickoff prompt (`cockpit_turn_sighting`) lists a run from the last 24 h.
- #3394: `select count(*) from substrate_endogenous_curiosity_candidates where gate_json is null and created_at > <deploy>` returns 0.
- #3401: one blocked `\b` command in a real turn trace.

## Review findings fixed

REVIEW_PENDING

## Restart required

```bash
# After merge, from the clean primary main checkout: git pull runs the post-pull rebuild.
# Services that carry this code: orion-hub, orion-substrate-runtime, orion-cortex-exec.
```

## Risks / concerns

- Severity: low
- Concern: read 1 is capped at 2000 run outcomes. Past that, the oldest runs and any run with an ISO-string stamp are cut first.
- Mitigation: a warning is logged when the cap is hit (`curiosity_worldview_recent_runs_truncated`, `curiosity_supervisor_run_order_truncated`). There are 551 outcomes today.

- Severity: low
- Concern: the SQL guard matches on text. A psql command that holds `\b` for another reason next to a `~` operator is blocked too.
- Mitigation: the block message says what to change, and the rule only fires when `psql`, a regex operator and `\b` are all present.

Issues opened: none. Issues closed by this PR: #3398, #3394, #3401.

## PR link

PR_LINK_PENDING
