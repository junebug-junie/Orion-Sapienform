# Curiosity replica, patch 5: the three-arm readout

## Summary

- Curiosity orders its offer three ways, picked at random per run: the old uncertainty rule, the value rule, or the
  replica's plan. This adds the script that scores the three against each other, as the design pre-registered it.
- The scorecard is read once, at the stop: every arm has 250 finished runs, or 180 days have passed. It counts only
  runs drawn before that fixed moment. The replica wins only if it beats both rules (one-sided permutation tests,
  10,000 shuffles, both p below 0.05).
- Before the stop, the script shows counts only and hides each arm's outcome numbers, so nobody steers the test by
  peeking. `--unblind` shows them, for checking the pipe, never for deciding.
- It runs the harm check every 30 days. Does the replica arm fail turns, or test nothing, more often than the two
  rules together (p below 0.01)? A run with no outcome a day after its draw counts as failed.
- It counts the last 14 days of fallbacks by reason and flags it if more than 30% of replica runs were unfunded. It
  also reuses the answers-to-Prime readout for the refusal alarm.
- Read-only. Exit 3 when something must go to Juniper, exit 2 when Postgres can't be read.

## Outcome moved

Before this patch, nothing could tell whether the replica is winning, losing, or starving. The arm comparison had to
be done by hand in SQL, and the hand count got it wrong: the 11-04 rows, drawn while the replica arm was off, were
mixed in. Now one command gives the pre-registered answer and tells Juniper when to step in.

## Current architecture

`curiosity_offer_decisions` stores the arm and its propensities (patch 3). `curiosity_run_outcomes` stores the
realized nats, the tested count and the moved count. `curiosity_offer_plans` stores each plan's status. The only
readout was `replay_curiosity_realized_nats.py` (two arms, normal approximation) plus the answers readout (patch 4).

## Architecture touched

Analysis only. No service, contract, env or schema change.

## Files changed

- `scripts/analysis/curiosity_three_arm_readout.py`: the readout (pure functions + one read-only query).
- `scripts/analysis/tests/test_curiosity_three_arm_readout.py`: 14 tests on fixture data.
- `.github/workflows/orion-prime-tests.yml`: CI step and path triggers.
- this report.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none. Behavior changed: none.
- Compatibility notes: reads existing tables only.

## Env/config changes

- None. No `.env_example` touched, so no sync needed.

## Tests run

```text
pytest scripts/analysis/tests/test_curiosity_three_arm_readout.py scripts/analysis/tests/test_curiosity_replica_answers_readout.py
16 passed in 0.47s
```

The tests cover:
- too few runs: no verdict;
- a clear winner at 250 per arm;
- beating one rule but not the other: a loss;
- the 180-day clock stop;
- runs drawn after the stop: verdict unchanged;
- hidden numbers before the stop;
- the harm check: it pauses an arm that keeps failing, it does not move between looks, and lost runs count as failed;
- an arm with too few scored runs at the stop: inconclusive;
- fallbacks counted by reason, with the unfunded flag;
- the 0-propensity exclusion;
- unreadable answers: sent to Juniper;
- bad propensity text: no crash;
- the Fisher test checked against a value worked out by hand.

## Evals run

```text
Live run, 2026-11-11, read-only (POSTGRES_URI=postgresql://postgres:postgres@localhost:55432/conjourney):
replica plan 16 completed of 16 drawn; value rule 9 of 9; uncertainty rule 17 of 17; 0 failed, 0 unfinished
Stop rule: started 2026-11-05, day 6 of 180. RUNNING: no verdict.
Harm check: no look yet; first 2026-12-05.
Fallbacks, 14 days: 1 of 16 (invalid:not_a_candidate), unfunded 0%.
Answers to Prime: 2, both support. Needs Juniper: nothing. exit=0
```

## Docker/build/smoke checks

```text
Not applicable: analysis script, no runtime change.
```

## Review findings fixed

The code review ran in a subagent. It found no statistical mistakes in the p-value direction, the Fisher tail, the
join or the time zones. It found three material gaps and four minor ones:

- Finding: runs drawn after the stop still counted, so the verdict depended on the day someone read it.
  - Fix: one fixed stop moment (the earlier of start + 180 days, or the draw that gave the last arm its 250th run);
    only runs up to it are in the test.
  - Evidence: `test_runs_drawn_after_the_stop_never_change_the_verdict`.
- Finding: the readout showed each arm's outcome numbers every day before the stop (peeking).
  - Fix: before the stop, counts only; `--unblind` for pipe checks.
  - Evidence: `test_before_the_stop_the_outcome_numbers_are_hidden`.
- Finding: the harm check ran on every read, so reading daily turned one look per 30 days into 30 looks.
  - Fix: read only on the runs before the latest 30-day mark; "no look yet" before day 30.
  - Evidence: `test_the_harm_look_only_moves_at_a_scheduled_look_and_lost_runs_count_as_failed`.
- Finding (minor): too few runs at the clock stop read as "retire it". Fix: INCONCLUSIVE, goes to Juniper.
- Finding (minor): a run that never wrote an outcome was invisible to the harm check. Fix: after a day, it counts as
  failed.
- Finding (minor): unreadable answers exited 0. Fix: `answers_unknown` goes to Juniper (exit 3).
- Finding (minor): bad propensity text crashed the script. Fix: it is read as propensity 0.
- Not changed: "a run with no plan row may be in flight". Checked: Hub awaits the plan before it writes the offer
  row (`curiosity_investigation.py` `_investigate`), so a missing plan row is a real missing record.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low.
  - Concern: "moved a belief" uses the stored moved count, not nats above 0. One uncertainty run moved a belief
    but has no nats number (`no_scorable_test`). So the uncertainty arm reads 8 moved, where a nats-based count
    gives 7.
  - Mitigation: the run is shown as "nats unknown 1".
- Severity: low.
  - Concern: the value arm has 9 runs and the other two have 16 and 17. This is within chance for 1/3 draws, but
    worth watching.
  - Mitigation: the per-arm draw counts are on every readout.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3183

🤖 Generated with [Claude Code](https://claude.com/claude-code)
