## Summary

- The drift preamble check from #2601 was not broken. It ran on every investigate turn and correctly found nothing to report, but it wrote no log line when nothing had moved, so "checked, nothing moved" looked exactly like "never ran".
- `prior_drift_preamble()` now writes exactly one line per call: `curiosity_prior_drift_checked ... offered=N moved=0`, `curiosity_prior_drift_preamble ... offered=N moved=M`, or `curiosity_prior_drift_skipped reason=reader_missing|no_prior_in_prompt|no_offered_priors|no_priors_parsed`. The existing `_read_failed` and `_unknown` lines are unchanged.
- `no_priors_parsed` logs at WARNING, and only when a `[confidence=` menu line is present. It means the prompt prints `prior_id:` lines but `_OFFERED_PRIOR_RE` parses none of them, so the preview format has drifted and drift can never be detected.
- `_prompt_for_attempt` now calls the check before its own `reader is None` early return, so a missing reader is logged as a skip instead of leaving no trace.

## Outcome moved

You can now tell from journald whether the drift check ran on a given turn, how many priors it compared, and why it skipped if it did.

## Current architecture

`_turn_result_for` -> `_prompt_for_attempt` -> `prior_drift_preamble` -> `read_prior_states` + `build_prior_drift_preamble`. Only the moved, read-failed and unreadable paths logged anything.

### Investigation (live data, 2026-10-13)

The suspected bug: in 5 of 14 runs, an offered prior's `last_tested_at` fell between `durable_admission_runs.created_at` and `fct_curiosity_runs.lifecycle_started_at`, yet no preamble was logged.

- `lifecycle_started_at` is not when the turn starts. For 206e69124ed4 it equals `completed_at` (09:17:01, `lifecycle_minutes=0`). The turn request actually arrived at 08:11:19.6, 0.3 s after admission (hub log `curiosity_turn_request`).
- The prior movements in that window were written by the runs themselves. Example: `PriorRevision run_id=206e69124ed4 prior=self:pool_heavy_composition_reverts_within_a_day_20261009 0.62 -> 0.70`. The same holds for 384ddfdb9132, c09226cea005 and e5eb9cfcf7ea.
- Sweep of all 28 turn requests since the logs begin (2026-10-11 23:23): the longest wait between admission and turn request was 742 s. Zero offered priors had a `PriorRevision` written between admission and turn request. So at every turn start there was no drift to report.
- Offline replay of the real stored prompt for 206e69124ed4 against today's prior state: `offered_priors_in_prompt` parses all 16 offered priors, and `build_prior_drift_preamble` produces an 8-prior preamble (for example `confidence 0.62 -> 0.70, tested 2 -> 3` for pool_heavy, and `0.65 -> 0.08, tested 0 -> 1, status now refuted` for chat_seat_demand_collapse). The parser and comparison are correct. They would fire if a brief actually waited.

## Architecture touched

orion-hub only: `scripts/curiosity_investigation.py`. No contract changes.

## Files changed

- `services/orion-hub/scripts/curiosity_investigation.py`: one log line per drift-check outcome. `_prompt_for_attempt` routes the reader-missing case through the check.
- `services/orion-hub/tests/test_curiosity_fresh_priors_and_answers.py`: 6 regression tests, one per outcome.
- `docs/superpowers/pr-reports/2026-10-13-prior-drift-preamble-silent-pr.md`: this report.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: new log lines only. The `curiosity_prior_drift_preamble` line gains `offered=`.
- Compatibility notes: prompt bytes are unchanged on every path.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
pytest services/orion-hub/tests/test_curiosity_fresh_priors_and_answers.py -q
  before fix: 5 failed, 19 passed
  after fix:  25 passed
pytest services/orion-hub/tests -q -k curiosity
  469 passed, 3 skipped
```

## Evals run

```text
Offline replay of the live run 206e69124ed4 brief against live FalkorDB orion_worldview
(read-only GRAPH.RO_QUERY): 16 offered priors parsed, 8 moved since then, preamble text produced.
Sweep of 28 live turns: 0 with drift present at turn start.
```

## Docker/build/smoke checks

```text
Not deployed (per instruction). After restart, expect one curiosity_prior_drift_* line per investigate turn.
```

## Review findings fixed

- Finding: the format-drift WARNING would fire on healthy briefs whose only `prior_id:` lines come from RECENTLY SETTLED or Orion's Day corrections (no confidence bracket).
  - Fix: WARN only when a `[confidence=` line is present. Otherwise log INFO `reason=no_offered_priors`.
  - Evidence: `test_settled_only_prior_lines_are_a_quiet_skip_not_a_warning` passes.
- Reviewer confirmed: returned prompt unchanged on every path, no new exception path, single caller.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low
- Concern: the original detector query (`lifecycle_started_at` used as the turn start) will keep producing false positives. Any future drift audit should compare against the `curiosity_turn_request` log time, or the new `curiosity_prior_drift_checked` line.
- Mitigation: documented above. The new log line makes that audit a journald grep.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2625
