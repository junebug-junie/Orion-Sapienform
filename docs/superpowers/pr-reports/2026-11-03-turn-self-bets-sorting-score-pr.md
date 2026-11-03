## Summary

- The turn self-bet grade now includes a **sorting score**: within each caller, does stance give a higher probability to the turns where the thing happened? It is reported as an AUC, where 0.5 is a coin flip and 1.0 is perfect sorting, with a one-sided p-value.
- **Fix:** a turn that crashed or was lost no longer counts as a failed attempt on a bet stance never made. Since #3044, every crashed turn wrote "failed" rows for the retired step and reply-length range bets. The self-monitor (#3063) then told stance: "My last 26 guesses about how many steps my turns take all failed to start." That sentence was false.
- The spec's keep/drop rule now gates on the sorting score. It is judged on 11-21, over bets placed from 11-07 on.

## Outcome moved

- **Base-rate shifts can no longer wreck the grade.** On 10-31 Orion's replies doubled in length, and the 7-day "usual" lagged behind. That made 73% of turns count as "longer than usual", so the Brier skill punished stance for a change it could not see. The sorting score does not move with such a shift.
- **First live sorting scores, 11-03:**
  - longer reply: 0.73, p 0.001;
  - more steps: 0.58, p 0.11 (too few pairs yet);
  - fetches a page: 0.58, p < 0.001;
  - **asks a question: 0.49, p 0.68.** Inside each caller, stance cannot tell which turns will end on a question. Its earlier positive Brier skill came from knowing the caller.
- **The false "failed to start" sentence stops for new turns.** 80 rows already written still carry it; see Risks.

## Current architecture

- `_grade_p` reported Brier, skill against a global base rate, and skill against each caller's own rate, plus calibration bins.
- `settle()` checked "turn crashed or expired" before "bet stated", so a void turn marked every faculty `failed_to_execute` or `expired_unsettled`, stated or not.

## Architecture touched

- Shared: `orion/prediction/grading.py` gains `sorting_score` (additive; `grade()` is unchanged).
- orion-prediction: `_grade_p` and `settle()`.
- Spec: the decision rule.

## Files changed

- `orion/prediction/grading.py`: `sorting_score` and `_avg_ranks` (tie-averaged ranks, tie-corrected Mann-Whitney variance, pooled within groups). Refuses None or NaN input and mismatched lengths.
- `services/orion-prediction/app/turn_self_bets.py`:
  - `_grade_p` adds `sort_auc`, `sort_pairs` and `sort_p`;
  - `settle()` checks "stated" before the crash and expiry voids.
- `services/orion-prediction/tests/test_grading.py`: 5 sorting-score tests.
- `services/orion-prediction/tests/test_turn_self_bets.py`:
  - a regression test for crashed and lost v2 turns;
  - the crash and expiry tests are scoped to the faculties a v1 bet states;
  - a test for the per-caller sorting score.
- `docs/superpowers/specs/2026-11-02-turn-self-bets-relative-size-design.md`: the decision rule, why it changed, and the first live read.

## Schema / bus / API changes

- Added: `sort_auc`, `sort_pairs` and `sort_p` in each probability faculty's grade at `/turn-self-bets/live`.
- Removed: none
- Renamed: none
- Behavior changed: an unstated faculty on a crashed or expired turn settles `ungradeable_no_stated`, not `failed_to_execute` or `expired_unsettled`.
- Compatibility notes: the self-monitor adapter already skips `ungradeable_no_stated`, so it needs no change.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no change needed
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: no env change
- skipped keys requiring operator action: none

## Tests run

```text
services/orion-prediction: pytest tests/test_turn_self_bets.py tests/test_grading.py -q -> 44 passed
orion/self_monitor -q                                                                    -> 94 passed
services/orion-prediction: pytest tests -q -> 20 failed / 244 passed; the same 20 fail on main in
  this venv (diffed failure lists: identical), untouched here
git diff --check -> clean
```

## Evals run

```text
sorting_score over the live 7-day turn self-bets (numbers in "Outcome moved"). Read-only.
Review agent: sorting_score against scipy.stats.mannwhitneyu (asymptotic, greater, no continuity
correction) on 500 single-group and 300 four-group cases with heavy ties: max gap 1.1e-16.
```

## Docker/build/smoke checks

```text
UNVERIFIED until deploy. After pull: /turn-self-bets/live shows sort_auc per probability faculty,
and new crashed v2 turns write ungradeable_no_stated on turn.self.steps / turn.self.reply_length.
```

## Review findings fixed

- Code review in a subagent: nothing material. The maths matches scipy, and the edge cases (n = 1, all ties, single-class groups, empty input) are handled.
  - Finding (low): a None or NaN probability sorts as the highest. Fix: refuse it with ValueError. Evidence: `test_sorting_score_refuses_input_it_would_misread`.
  - Finding (low): mismatched `groups` length gave an unclear IndexError. Fix: an explicit length check, covered by the same test.
- Self-found while checking how #3063 renders the new bets: the false "failed to start" streak. Fix and regression test above.

## Restart required

```text
Post-pull rebuild handles it (orion-prediction; orion-thought and orion-durable-runs import
orion/prediction/grading.py through the self-monitor and rebuild too). No migration.
```

## Risks / concerns

- Severity: medium
  - Concern: 80 rows already written still say a never-made bet failed. They are 68 rows on the retired range bets and 12 older probability rows where stance stated nothing. Until they are corrected or age out (about 11-10), the self-monitor keeps telling stance "my last 26 guesses about how many steps ... failed to start".
  - Mitigation: a one-statement data fix, which needs Juniper's OK because it writes to production. The exact SQL is in the session report.
- Severity: medium
  - Concern: #3063's track-record cue went live on 11-03 at about 18:25 UTC, so stance now sees its own grades while the relative bets are being tested.
  - Mitigation: none needed. That is the intended feedback loop, so the 11-21 decision measures stance with feedback.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3077

🤖 Generated with [Claude Code](https://claude.com/claude-code)
