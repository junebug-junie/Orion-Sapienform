## Summary

- New hourly forecast in orion-prediction: "will Juniper send Orion a message in the next hour?" (prediction-error suite rebuild, family B, item 1).
- Served model: logistic regression where each of Juniper's rules (asleep 3-8, weekend mornings, evenings, Mondays, kid pickup, typing to Claude) is a starting belief that the data moves only as far as the evidence pushes it. The state model and the base rate are graded beside it, never instead of it.
- Each forecast is settled an hour later against `chat_message` and graded per local day: loss gap, standard error, and the chance the model is really better. There is no 0.05 line; Juniper: "0.05 is arbitrary".
- It ships failing at the hard case, by Juniper's call ("we can be honest that it fails and still roll w it"). It forecasts mid-conversation well. It cannot yet forecast her coming back after a quiet stretch.
- No consumer and no bus event. Outreach timing comes later and must state its own bar first.

## Outcome moved

Orion gets a scored forecast of Juniper's reachability, with an honest live grade. The replay on typing-covered hours (08-14..11-10) found:

- The rules model has 26 % lower loss than the base rate overall (P(really better) 0.95), tied with the state model.
- In cold hours (no message in the prior 3 h), it is about level with the base rate (P 0.25).
- It is somewhat timid at the top: stated 0.24 -> 0.40 observed.

## Current architecture

orion-prediction forecast only physical series (cabinet, office), plus action bets and turn self-bets. Nothing forecast Juniper.

## Architecture touched

- orion-prediction: new `app/reachable.py`, with a 60 s loop that issues in the first 10 min of each hour and settles 10 min after the hour ends.
- `GET /reachable`, plus a `reachable` block in `/health`.
- `backtest.paired_gap`, split out of `paired_win_p` with unchanged behaviour.
- Reads `chat_message` and `juniper_affective_state_log`, both read-only.
- Writes the new table `prediction_reachable_forecast`.

## Files changed

- `services/orion-prediction/app/reachable.py`: models, features, coverage mask, grade, `Reachable` loop
- `services/orion-prediction/app/backtest.py`: `paired_gap` (mean, SE, n) behind `paired_win_p`
- `services/orion-prediction/app/store.py`: Juniper message, typing and coverage reads; reachable table I/O; boot schema check
- `services/orion-prediction/app/main.py`, `app/settings.py`: loop, endpoint, health, flag
- `services/orion-prediction/docker-compose.yml`: `PREDICTION_REACHABLE_ENABLED`, plus the two turn-self-bet keys compose was missing (review catch)
- `services/orion-prediction/.env_example`, `README.md`
- `services/orion-sql-db/manual_migration_prediction_reachable_forecast_v1.sql`: one new table and one index, additive, REQUIRED-BY orion-prediction
- `services/orion-prediction/tests/test_reachable.py`: 19 tests
- `services/orion-prediction/evals/run_reachable_replay_eval.py`: held-out hourly replay on live data
- `docs/superpowers/specs/2026-11-10-juniper-reachable-forecast-design.md`: design, metric gate, replay

## Schema / bus / API changes

- Added: table `prediction_reachable_forecast`; `GET /reachable`.
- Removed: none.
- Renamed: none.
- Behavior changed: none for existing paths. `paired_win_p` returns identical values.
- Compatibility notes: boot checks the new table. The post-pull run applies the migration before rebuilding (additive allowlist, REQUIRED-BY header).

## Env/config changes

- Added keys: `PREDICTION_REACHABLE_ENABLED=true`
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: yes
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes; the key is present in `services/orion-prediction/.env`
- skipped keys requiring operator action: none

## Tests run

```text
orion-prediction suite in its image: 332 passed (earlier run); test_reachable.py 19 passed after the priors change
scripts/check_metric_lineage.py --gate: PASS
scripts/check_migration_required_by.py: OK
scripts/check_service_env_compose_parity.py orion-prediction: OK (35 keys)
scripts/check_env_template_parity.py: PASS
git diff --check: clean
```

## Evals run

```text
evals/run_reachable_replay_eval.py (live DB, read-only, 07-31..11-10, 2,457 hourly origins)
all origins:    all  rules 0.0786 / state 0.0774 / base 0.1066 log loss; P(beats base) 0.95, P(beats state) 0.40
                cold rules 0.0497 / state 0.0500 / base 0.0484;           P(beats base) 0.25, P(beats state) 0.46
typing-covered: all  rules 0.0906 / state 0.0892 / base 0.1250;           P(beats base) 0.95
                cold rules 0.0560 / state 0.0563 / base 0.0550;           P(beats base) 0.22
calibration: stated 0.030 -> 0.063 observed; 0.091 -> 0.108; 0.244 -> 0.40; 0.471 -> 0.588
```

## Docker/build/smoke checks

```text
Tests ran in the orion-prediction-prediction image. Live smoke after deploy: UNVERIFIED.
Check: one row per hour in prediction_reachable_forecast, settled ~70 min after its origin; GET :8132/reachable.
```

## Review findings fixed

- Finding: `PREDICTION_REACHABLE_ENABLED` never reached the container (compose lists env vars explicitly).
  - Fix: added to compose, plus 2 missing turn-self-bet keys.
  - Evidence: parity check OK.
- Finding: live training started at the first message while the replay started at a fixed window.
  - Fix: both use a fixed 120-day window.
  - Evidence: same code path in both.
- Finding: hours before the typing log existed, or during its outages, were trained as "not typing".
  - Fix: coverage mask; an uncovered origin serves the state model.
  - Evidence: tests `test_uncovered_typing_log_is_not_read_as_not_typing` and `test_hours_before_the_log_existed_are_left_out_of_training`; replay reports 423 uncovered origins.
- Finding: SE recovered through `t.ppf` was capped at |t| about 10, and wrong for all-equal differences.
  - Fix: `backtest.paired_gap` returns SE directly.
  - Evidence: `test_gap_se_is_direct_not_capped_and_matches_paired_win_p`.
- Finding: logistic overflow warnings and an unchecked optimiser result.
  - Fix: `logaddexp`/`expit`; the state model is served on failure.
  - Evidence: tests.
- Finding: the asleep test passed only through the evening feature.
  - Fix: messages now also fall at daytime hours, and the test compares 05:00 to 10:00.
  - Evidence: `test_asleep_hours_read_lower_than_daytime`.
- Finding: a sign test on daily differences called a worse model significant.
  - Fix: the grade uses the mean gap.
  - Evidence: documented in the spec.

## Restart required

Automatic: Juniper's `git pull` on main runs the post-pull rebuild, which applies the migration and then rebuilds orion-prediction.

## Risks / concerns

- Severity: medium. Concern: no cold-hour skill. Mitigation: no consumer. The grade reports cold `prob_better` live; outreach must set its bar on it first.
- Severity: low. Concern: under-confident probabilities. Mitigation: documented; revisit with live rows rather than refitting on the replay.
- Severity: low. Concern: prior strengths and the 60-day half-life are judgment calls, not fit. Mitigation: they are Juniper's rules made explicit; the live grade shows if any rule is wrong, and data moves each one.
- Feedback addressed (Juniper 2026-11-11): an earlier draft dropped the pickup rule and served the plain state model on a lowest-loss contest, sidelining her rules. Rules are now priors and always served.
- Severity: low. Concern: the Juniper-message filter assumes every non-`orion_*` user turn is hers. Mitigation: checked against `chat_history_log` (1 mismatch in 121).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3180

🤖 Generated with [Claude Code](https://claude.com/claude-code)
