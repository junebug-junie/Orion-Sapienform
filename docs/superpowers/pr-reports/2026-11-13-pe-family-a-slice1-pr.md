# Family A slice 1: raw-reading hygiene gate, baseline rungs, joint surprise (eval only)

## Summary

- Orion now has a checked list of every raw body and room reading (45 candidates: cabinet, machines,
  AC, office, closet) and a gate that refuses bad inputs instead of training on them.
  (`config/prediction/family_a.v1.yaml`, `services/orion-prediction/app/family_a.py`)
- Three forecasters are measured side by side at 5 and 15 min ahead: "same as now", one small model per
  reading, and one model that sees every reading at once. The all-readings model wins at every seed,
  with a significance test, but only by a little (loss x0.94-0.96 at 5 min, x0.96-0.98 at 15 min).
- A joint surprise score is computed: one number for "how surprising is the whole body right now",
  where readings that move together count once. It rests at 0 on the last 24 h of live data and spikes
  on a real event (athena load 18 -> 60 on 11-12 09:15-11:00 UTC). Eval output only.
- Nothing is served, stored, or published. No new table, bus channel, node, or env key.
- No neural network yet: only 9 days hold every reading. The spec asks for 21 (2026-11-26 earliest).

## Outcome moved

Family A ("what will my body and room read next?") moved from spec to a measured baseline on live data.
The ladder rung above per-series AR is now proven to add real signal (significant at both horizons,
3 seeds), and the gate that kept failing the mood-arc encoder (dead, pinned, renamed, duplicated
inputs) is a tested function that refuses, with reasons, on live data today.

## Current architecture

orion-prediction forecasts two readings (cabinet and office temperature) one at a time, gates models by
backtest vs baselines, gives conformal ranges, and publishes per-reading surprise into
`node:prediction.body` / `node:prediction.room`. Every other raw reading (GPU, fans, watts, closet CO2,
office air and light) was stored but forecast by nothing. The only multivariate "anomaly" model was the
mood-arc encoder, trained on pressure / hint channels.

## Architecture touched

- orion-prediction only, as new pure modules plus an eval. Not wired into `app/main.py` or the engine loop.
- Reuses `orion/mood_arc/fit_encoder.py::block_purged_temporal_split` (public function, called on origin
  indices), `app/backtest.py::paired_gap` / `paired_win_p` / `overlap_lag`, `app/live.py::holm`.
- No contract change: no schema, channel, table, or env key.

## Files changed

- `config/prediction/family_a.v1.yaml`: candidate inputs (raw tables only) and the fixed hygiene thresholds.
- `services/orion-prediction/app/family_a.py`: config parser (refuses hint channels and non-raw tables),
  hygiene gate, train-only standardiser, split, rungs, significance, joint surprise, manifest.
- `services/orion-prediction/app/family_a_load.py`: read-only 5-min bucket loader (keys as bind parameters).
- `services/orion-prediction/evals/run_family_a_baselines_eval.py`: live eval (read-only).
- `services/orion-prediction/tests/test_family_a.py`: 43 gate tests.
- `services/orion-prediction/README.md`: Family A section + eval command.
- `.github/workflows/orion-prediction-tests.yml`: also runs when `orion/mood_arc/fit_encoder.py` changes
  (the split is imported from there).

## Raw-reading inventory (live Postgres, 2026-11-13)

| source | readings | history |
|---|---|---|
| `orion_biometrics_summary` athena (`measurements` only) | cabinet temp / humidity / baro / light / sound / vibration / lidar | since 08-29 (76 d) |
| same | chassis, CPU, GPU-total watts; fan %; load; net / disk rate | since 08-14 (91 d) |
| same | GPU0/1 temp | since 09-30 (44 d) |
| same | board temp | since 10-09 (35 d) |
| same | per-GPU watts, GPU memory | since 10-29 (15 d) |
| `orion_biometrics_summary` circe | chassis watts / fan | since 09-08 (66 d); CPU watts, load since 08-14 |
| same | GPU0-3 temp / board temp / per-GPU watts | since 09-30 / 10-09 / 10-29 |
| `orion_biometrics_cluster` hecate | chassis watts | since 10-14; this table keeps 30 days (sql-writer retention) |
| `home_cooling_sample` | cabinet AC watts | since 09-26 (48 d) |
| `office_env_sample` | temp, humidity | since 10-28 (15 d) |
| same | eCO2, TVOC | since 11-03 (10 d) |
| same | PM2.5, lux | since 11-05 (8 d) |
| same (`payload_json.measurements`) | sound dBA | since 11-07 (6 d) |
| `closet_env_sample` | CO2 (SCD40), temp, humidity | since 11-05 (8 d) |

No retention job was found on the summary, office, closet, or cooling tables, so the raw corpus already
persists in Postgres. The 30-day limit on `orion_biometrics_cluster` touches only hecate's watts, which the
gate refuses anyway. No new corpus writer is needed; the gate's staleness check (30 min) is the run-time
health check.

## Hygiene result (training part of 2026-10-30..11-12)

- **Targets (41)**: every athena and circe machine reading in the config (watts, temps, fan, load,
  net / disk rate); cabinet temp, humidity, baro, sound, vibration, lidar; AC watts; office temp,
  humidity, eCO2, PM2.5, lux; closet CO2, temp, humidity.
- **Covariate only (1)**: `athena.cabinet_light_raw` (event-like: rests at its floor 57 % of buckets).
- **Refused (3)**:
  - `hecate.chassis_watts`: one value 98.6 %, 44 distinct (hecate is on demand).
  - `office.tvoc_ppb`: duplicate of `office.eco2_ppm` (corr 0.99999; the CCS811 computes one from the other).
  - `office.sound_dba`: only 38 % present (starts 11-07). Passes presence around 11-14 in a 14-day window;
    its level (75-81 dBA in a quiet office) should be checked before it is trusted.

## Backtest (14 days, 5 blocks, purged; joint loss = mean squared standardised error over 41 targets)

| horizon | seed | persistence | AR | best baseline | ridge_var | ratio | p paired t | p block bootstrap | win |
|---|---|---|---|---|---|---|---|---|---|
| 5 min | 0 | 0.353 | 0.216 | 0.216 | 0.208 | 0.962 | 5e-14 | 0.0005 | yes |
| 5 min | 1 | 0.388 | 0.224 | 0.224 | 0.209 | 0.935 | 3e-18 | 0.0005 | yes |
| 5 min | 2 | 0.377 | 0.218 | 0.218 | 0.205 | 0.940 | 5e-21 | 0.0005 | yes |
| 15 min | 0 | 0.516 | 0.332 | 0.332 | 0.324 | 0.977 | 0.005 | 0.001 | yes |
| 15 min | 1 | 0.562 | 0.335 | 0.335 | 0.321 | 0.956 | 1e-4 | 0.0005 | yes |
| 15 min | 2 | 0.539 | 0.330 | 0.330 | 0.316 | 0.959 | 9e-07 | 0.0005 | yes |

0.0005 is the bootstrap floor (1/2001). AR beats persistence everywhere (p < 1e-44). Window 2026-10-30 00:40 .. 11-13 00:40 UTC. The best baseline is
chosen per target on the held-out set itself, which favours the baseline.

Where the other readings help most (seed 0, ridge / best baseline at 5 / 15 min, Holm p): office temp
0.66 (1e-10) / 0.72 (1e-4); cabinet temp 0.69 (2e-10) / 0.82 (0.002); athena board temp 0.77 (1e-8) /
0.91 (1e-4); circe board temp 0.88 (1e-4); athena chassis watts 0.80 (7e-5) at 5 min. Most GPU, network, and disk readings: about 1.0 (no help).

**Ladder verdict:** keep `ridge_var` over AR. Do not build the small dense network yet: only 9 days hold
every target (closet, PM and lux start 11-05); the spec asks for 21 days.

## Metric quality gate: joint surprise (`family_a.joint_surprise`)

1. **Provenance.** `ridge_var` residuals (`run_horizon`: truth minus AR forecast minus cross-reading ridge)
   on standardised 5-min means of the raw tables above. Covariance from fit-row residuals (10 % shrink to
   the diagonal, then eigen-clipped so it is positive definite); d^2 on the readings present; chi-square tail; conformal p against validation rows
   (`conformal_p`); excess = -ln p - `calm_point(n_cal)`.
2. **Independence vs `node:prediction.body` / `.room`.** Those nodes score cabinet and office temperature
   from the univariate champion. Both readings are 2 of the 41 targets here. On the last 24 h they carry
   3 % of the joint score on average; the largest shares are athena load, board temp, CPU watts, chassis
   watts (4-8 % each). Shared chain: same sensors for those two readings only. Not redundant.
3. **Theory anchor.** Mahalanobis distance of a forecast residual vector is the standard multivariate
   outlier statistic (chi-square under a Gaussian residual); the conformal step makes p uniform under
   exchangeability whatever the real residual shape, so E[-ln p] is a known constant and calm is 0 by
   theory. Whitening is what makes one event on four coupled readings one surprise (tested: d^2 9.4 vs
   36 unwhitened).
4. **Live rest check.** Model trained on data before the last 24 h, scored on the last 24 h:

   | horizon | n | mean excess | 95 % CI | p <= 0.05 | hours at rest (calibrated null) | hourly range | longest run > 0.5 |
   |---|---|---|---|---|---|---|---|
   | 5 min | 287 | -0.03 | [-0.26, +0.35] | 3.8 % | 0.68 (0.54) | -0.86..+1.70 | 3 h |
   | 15 min | 285 | -0.18 | [-0.38, +0.11] | 3.9 % | 0.72 (0.55) | -0.89..+1.17 | 2 h |

   It returns to rest, and it is not pinned at 0: it rose to +3.9 at 11-12 09:20 UTC, carried 84 % by
   athena load (load 18 -> 60, CPU 190 -> 315 W, 09:15-11:00), and came back after. It reads slightly
   too calm (68-72 % of hours at or below 0 vs 55 % for a calibrated score), the same pattern the
   one-width cabinet ranges had before adaptive ranges. On held-out backtest blocks the mean excess CI
   contains 0 in 6 of 6 runs (-0.10..+0.20), but the share of p <= 0.05 is 3.5-9.3 %: tail calibration across blocks is off
   by up to 2x on some splits.
5. **Existing mechanism.** None: `app/surprise.py` is per reading; the mood-arc encoder is reconstruction
   error on hint channels. This reuses the existing split and paired test.
6. **Reversibility.** Pure functions and an eval. No schema, table, node, channel, or default. Delete
   the files to remove it.

Not wired to any consumer, per spec rule 5 ("no consumer until variance at known physical events is shown").
One known physical event is shown here; the spec's double-count check on a live AC-off event is UNVERIFIED.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: none in the running service.
- Compatibility notes: none.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no env template changed).
- skipped keys requiring operator action: none.
- New config file: `config/prediction/family_a.v1.yaml` (baked into the image with `config/prediction/`; read by the eval only).

## Tests run

```text
cd services/orion-prediction
PYTHONPATH=../..:. python -m pytest tests -q -p no:cacheprovider     (in the orion-prediction image)
394 passed, 2 warnings in 261.76s (after review fixes; host load ~330)
PYTHONPATH=../..:. python -m pytest tests/test_family_a.py -q -p no:cacheprovider
43 passed in 5.72s (after review fixes)
```

## Evals run

```text
POSTGRES_URI=... PYTHONPATH=../..:. python evals/run_family_a_baselines_eval.py --days 14 --seeds 0 1 2 --out <scratch>
(orion-prediction image on app-net, read-only; window 2026-10-30 00:40 .. 2026-11-13 00:40 UTC.
 37 s on a quiet host; 913 s on the post-fix run because host load average was 330 on 96 cores)
-> tables above
```

## Docker/build/smoke checks

```text
No build needed: no runtime path changed. The eval and tests ran inside the existing
orion-prediction-prediction image (python 3.12, pinned requirements) with the worktree mounted read-only.
```

## Review findings fixed

Code review ran in a subagent on the full diff. No blockers. It also ran 24 independent null histories:
0 false ridge wins.

- Finding: residual covariance from pairwise-complete data was not always positive definite (min
  eigenvalue -0.20 under blocky missingness), so d^2 could go negative and read as calm.
  - Fix: eigen-clip after shrinking; a target without a finite variance from >= 10 fit rows is left out
    of the joint score instead of getting a 1e-9 variance.
  - Evidence: `test_covariance_stays_positive_definite_under_blocky_missingness`.
- Finding: the rows that chose the ridge penalty also calibrated the conformal p (biases excess upward).
  - Fix: penalty is chosen on a second purged split of the fit rows, then refit on all fit rows; the
    validation rows are used only for calibration.
  - Evidence: `test_held_out_rows_never_touch_any_training_decision` (held rows shifted by +1000: alphas,
    fit and calibration residuals bit-identical); held-out mean excess moved from -0.06..+0.33 to
    -0.10..+0.20.
- Finding: refused targets counted toward the "half the targets observed" rule and could blank the score
  (and crash the eval).
  - Fix: count usable targets only; eval guards empty results.
  - Evidence: `test_joint_surprise_counts_only_usable_targets`.
- Finding: a quiet day of an event-like covariate (resting at its floor) refused the whole live run.
  - Fix: for a covariate resting on its training floor, `one_value` / band checks do not apply; absent and
    stale still do. Event-like = floor share in (0.50, 0.85] documented in the config.
  - Evidence: `test_quiet_day_of_an_event_like_covariate_does_not_refuse_the_run`.
- Finding: a fractional `--days` put the grid off the 5-min buckets (everything read absent).
  - Fix: start floored to the grain.
- Finding: the standardiser test did not test leakage; the purge test checked one direction.
  - Fix: replaced with the held-row shift test; purge test now checks both directions for the outer and
    both inner splits.
- Nits fixed: NaN-safe duplicate correlation; hint pattern widened (`hint`, `headroom`, `composite`,
  `anomaly`, `_score`) and applied to the cooling role; two inputs on one source refused; purge docstring.
- Nits left: mask channel is lag-0 only (missing at lags 1-12 reads as the training mean, not a raw 0.0);
  the input screen sees the whole training window including backtest held-out shape (not predictability);
  bootstrap blocks can straddle held-out stretches. Low impact; listed for the wiring patch.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low. Concern: the joint score reads slightly too calm live and up to 2x hot in the tail on
  some backtest splits. Mitigation: eval only; the next patch must pass the same live calm check over 7 days
  before any consumer, and should try adaptive (recent-miss-scaled) calibration as `app/adaptive.py` does.
- Severity: low. Concern: heavy-tailed readings (network / disk bytes per second) are standardised raw.
  Mitigation: they gain nothing from ridge today; a log transform per input is a candidate for the next patch.
- Severity: low. Concern: `office.sound_dba` reads 75-81 dBA. Mitigation: refused today (absent); check the
  level before it passes presence (~11-14).

## Recommended next patch

Wire the gate and `ridge_var` into orion-prediction's select -> issue -> score loop as one multivariate
series (store per-reading forecasts + the joint p per bucket, scored when due), with the run-time gate on
every issue, and re-run the 7-day live calm check on those stored rows. Only after that: a
`prediction.surprise.v1` joint field and a consumer. Small dense network rung: re-run this eval after
2026-11-26.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
