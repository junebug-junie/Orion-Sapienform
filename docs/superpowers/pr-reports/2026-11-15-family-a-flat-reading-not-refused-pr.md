## Summary

Family A (the forecaster for every body and room reading) stopped forecasting at 2026-11-14 17:50Z. The office air was clean, so the PM2.5 sensor read 0 for a full day. The check that runs before each 5-minute forecast saw "one value for 24 h", called the sensor broken, and refused the whole bucket. Every bucket since then was refused (146 rows at 23:58Z).

- The per-bucket check no longer tests for "one value". That test is a model-selection decision, and it still runs at selection time over 14 days.
- The per-bucket check now refuses only for real invalidity: input missing (`absent`, `not_loaded`), input stopped sending (`stale`), or input outside the range it was trained on (`left_trained_band`).
- A reading resting at its training minimum counts as in range, for targets too, even when that minimum sits below the band's 10th percentile. (Before, only covariates got this.)
- New eval `evals/run_family_a_gate_replay_eval.py` replays the gate on live history and compares it with the stored refusals.

## Outcome moved

Live replay of the last 24 h of real data (2026-11-14 00:10Z to 2026-11-15 00:10Z, the serving run's bands at each bucket):

| gate | buckets passed | buckets refused | reason |
|---|---|---|---|
| stored by the live loop | 212 scored | 75 | `{"office.pm25_ug_m3": ["one_value"]}` |
| main (before) | 212 | 76 | the same reason (one more bucket, not yet issued live) |
| this branch (after) | 288 | 0 | none |

The old gate in the replay refuses the same buckets the live loop refused, so the replay is faithful. The new gate refuses none.

Raw data: `office_env_sample.office_pm25_ug_m3` read 0 every hour from 2026-11-14 00:00Z on (about 120 samples per hour, `stale=false`). The trained band for PM2.5 is `[0.0, 6.6]`, the training minimum is 0.0, and 0.0 is also its most common training value (22.6 %). So 0 is inside the trained range.

## Current architecture

`FamilyALive.issue` runs `family_a.gate_named_inputs` on the last 24 h before each forecast. For windows under 7 days it called `_short_window_reasons`. That function refused when one value covered more than 85 % of the window (`one_value`), except for event-like covariates resting at their floor. It also refused when no value touched the training band. Any refusal on any input refused the whole bucket: no forecasts, one `refused` joint row per horizon.

## Architecture touched

- `services/orion-prediction/app/family_a.py`, `_short_window_reasons`: the `one_value` check is removed. The floor exemption now covers every input, not only covariates. The unused `status` argument is removed.
- No schema, bus, env, or compose change.

Scoring with a flat reading is safe. Each residual is divided by the training standard deviation stored in the manifest (PM2.5: 3.82), never by the 24 h window's own spread. `fit_standardizer` also floors the std at 1e-9. The covariance and the quantile offsets come from validation rows. The new live-loop test checks that `d2`, `excess`, `p` and `abs_error` are finite on scored buckets while a target is flat.

Whole-bucket refusal is kept for the remaining reasons. Every input is a lagged feature of every target's ridge forecast, so a stale or out-of-range input makes all targets' forecasts invalid, not only its own. The joint calibration tail is also fitted with every target observed. Dropping one target and shrinking `n_observed` would change the calibration, and that needs its own calm check. This is noted in #3370.

## Files changed

- `services/orion-prediction/app/family_a.py`: the per-bucket gate no longer refuses for degeneracy; the floor rest counts as in regime for targets too.
- `services/orion-prediction/app/family_a_live.py`: the docstring matches the new gate.
- `services/orion-prediction/tests/test_family_a.py`: regression tests (flat PM2.5 at 0 does not refuse; flat target still refuses when stale, absent, or out of band; a target at its training floor below the band does not refuse). The old test that expected `one_value` on a target is replaced.
- `services/orion-prediction/tests/test_family_a_live.py`: a live-loop regression. A target that is flat for 23 h before selection and after it is issued and scored, never refused.
- `services/orion-prediction/evals/run_family_a_gate_replay_eval.py`: new read-only replay eval.
- `services/orion-prediction/README.md`: gate description and eval command.

## Schema / bus / API changes

- Added: none
- Removed: none. `one_value` can no longer appear in a per-bucket `refusal`. It still appears in selection verdicts.
- Renamed: none
- Behavior changed: per-bucket refusals are now limited to `absent`, `not_loaded`, `stale`, and `left_trained_band`.
- Compatibility notes: the 146+ stored `refused` rows stay as they are.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change)
- skipped keys requiring operator action: none

## Tests run

```text
cd services/orion-prediction && PYTHONPATH=../..:. python -m pytest tests/test_family_a.py tests/test_family_a_live.py -q -p no:cacheprovider
68 passed in 41.02s

New regression tests against main's family_a.py (to show they catch the bug):
FAILED test_flat_target_at_rest_inside_its_trained_band_does_not_refuse  (HygieneRefused: office.pm25_ug_m3: one_value)
FAILED test_flat_target_still_refuses_when_stale_absent_or_out_of_band   (('one_value','stale') != ('stale',))
FAILED test_a_target_resting_flat_inside_its_band_is_issued_and_scored_not_refused

Full service suite on the host venv: 20 failures in test_backtest.py, test_causal.py, test_events.py, and
test_heat_balance.py. None of these files imports family_a, and this branch changes only family_a code.
These failures come from the host venv, not from this change. CI is the authority.
```

## Evals run

```text
POSTGRES_URI=... PYTHONPATH=../..:. python evals/run_family_a_gate_replay_eval.py --hours 24 --compare-module <main's family_a.py>
window [2026-11-14 00:10Z, 2026-11-15 00:10Z), stored joint rows (5 min horizon): {'refused': 75, 'scored': 212}
current: pass=288 refused=0 reasons={}
compare: pass=212 refused=76 reasons={'{"office.pm25_ug_m3": ["one_value"]}': 76}
PASS
```

## Docker/build/smoke checks

```text
No compose, Dockerfile, or requirements change. No build was run. The live effect is UNVERIFIED until orion-prediction is rebuilt from main after merge.
```

## Review findings fixed

- Finding: the docstring said a frozen sensor is dropped "at the next select". Select drops it only after it is one value for more than 85 % of 14 days (about 12 days). Until then, a sensor frozen inside its band reads as calm.
  - Fix: the docstring now states the real trade-off. A per-bucket check against the training's own flattest day is filed as #3370.
  - Evidence: `family_a.py` `_short_window_reasons` docstring.
- Finding: a target resting at its training minimum still refused as `left_trained_band` when the minimum sat below the band's 10th percentile. A dirtier training fortnight could make PM2.5 refuse again.
  - Fix: the floor exemption now applies to targets too.
  - Evidence: `test_target_resting_at_its_training_floor_below_the_band_does_not_refuse`. When the training floor is above the flat value, the bucket still refuses (asserted in `test_flat_target_still_refuses_when_stale_absent_or_out_of_band`).
- Finding (nit): the `absent` case passes on old code too.
  - Fix: a comment now says it guards against reordering the checks.

## Restart required

Rebuild after merge, from the clean primary main checkout synced with origin/main:

```bash
scripts/safe_docker_build.sh orion-prediction up -d --build
```

Verify: after the restart, the first family A buckets have `status='issued'` in `prediction_joint_surprise`, and no new `refused` rows name `one_value`.

## Risks / concerns

- Severity: medium
- Concern: a sensor frozen at a constant inside its band, while it still reports, reads as calm for up to about 12 days.
- Mitigation: #3370 proposes a per-bucket check grounded in training data.
- Severity: low
- Concern: the #3309 7-day calm check must leave out the refused gap, from 2026-11-14 17:50Z to the deploy time. There are no scored rows in that gap, and the refusal rate is inflated. This gap is noted on #3309.
- Issues opened: #3370. Issues closed: none.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3371

🤖 Generated with [Claude Code](https://claude.com/claude-code)
