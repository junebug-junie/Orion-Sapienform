## Summary

Orion now has a sensor that can see circe's heat. The prediction service reads the closet-top board (above the cabinet, where circe's and hecate's hot air goes). It fits "degrees of closet-top warming per 100 W" for each machine, beside the cabinet fit it already had. This is Patch 1 of the air-path design (#3234, `docs/superpowers/specs/2026-11-12-orion-air-path-design.md`). It only measures. Nothing acts on the number.

- New source kind `closet_env` reads `closet_env_sample`. Only the `closet_temp_c` column is allowed. Stale or offline samples are never read.
- New series `closet.top_temp_c` (role `covariate`). It is never forecast, and it is not an input to the cabinet forecast.
- A second `HEAT_BALANCE` entry: closet top vs office, watts circe / hecate / athena, the portable AC as a control.
- `GET|POST /effects/heat-per-watt?series_id=closet.top_temp_c` now works. Stored rows go under their own `series_id`.
- New read-only eval `evals/run_air_path_eval.py`: lead/lag, the one-hour forecast with and without closet heat, slopes, and the daily residual (the stuck-air watch).

## Outcome moved

Before this patch, no sensor in Orion's prediction or regulation saw circe's heat (cabinet local slope +0.01 C per 100 W, not different from zero). Now, live over 21 days: circe warms the closet top by **+0.27 C per 100 W (95 % +0.21..+0.33)**. hecate is correctly refused as "flat" because it has run only since 2026-11-14 20:00 UTC. When it has run at least 10 % of the window's hours, the same GET returns its slope, with no code change.

## Current architecture

`app/heat_balance.py` declared one target (`athena.cabinet_temp_c`). `Effects.heat_per_watt` accepted only forecast targets. The closet board's data was read only by Hub panels. The registry comment called circe and hecate "machines in the cabinet". They are above it.

## Architecture touched

Only `orion-prediction` and its registry. No bus channel, no schema registry entry, no table, and no env key changed.

## Files changed

- `services/orion-prediction/app/series.py`: `CLOSET_ENV_KEYS = ("closet_temp_c",)` and the `closet_env` source parser.
- `services/orion-prediction/app/store.py`: the `closet_env` branch of `points()`. It uses the same `_read_only` path and timeout as `office_env`. The filter is `sensor_online AND NOT stale AND closet_temp_c IS NOT NULL`.
- `services/orion-prediction/app/heat_balance.py`: the second `HEAT_BALANCE` entry, exactly as the design gives it. "cabinet" became "target" where the text is generic. The internal frame column `cabinet` became `target`. `TARGET_LABEL` keeps the cabinet's stored words unchanged. `load_frame` folds change events only for forecast targets, so a closet fit does not add an empty series to `/health`.
- `services/orion-prediction/app/effects.py`: `heat_per_watt` accepts any registered series declared in `HEAT_BALANCE`, whatever its forecast role.
- `services/orion-prediction/app/main.py`: GET docstring.
- `config/prediction/series.v1.yaml`: the `closet.top_temp_c` series, and the corrected "machines in the cabinet" comment.
- `services/orion-prediction/README.md`: the closet top in the covariate table and the heat-per-watt section, plus the eval command.
- `services/orion-prediction/tests/test_closet_heat_balance.py` (new), plus `test_series.py`, `test_inputs.py`, and `test_heat_balance.py` / `evals/run_heat_per_watt_eval.py` (fixture column rename).
- `services/orion-prediction/evals/run_air_path_eval.py` (new).

## Schema / bus / API changes

- Added: the `series_id=closet.top_temp_c` value for the existing heat-per-watt route. Its rows land in `prediction_effect_estimate` under that series_id, with `details.what` = "closet top minus room" / "closet top, room free".
- Removed / renamed: none.
- Behavior changed: a registered series that has no heat balance (for example `athena.chassis_watts`) is now refused with the same 422 status, but with the `{"error": "refused", ...}` body (HeatBalanceRefused) instead of the string "unknown target series". An unknown id still gets the old string 422.
- Compatibility: the cabinet answer and its stored rows are unchanged. A golden test pins them to values computed on origin/main.

## Metric gate for `closet.top_temp_c` (full findings: the design doc, "Number 1")

1. **Provenance.** The pan-1 SCD40 board (`sensor_id` scd40-closet). The sample is built in `services/orion-office-env/app/sample.py` with `ENV_ROOM=closet`. sql-writer flattens it in `_normalize_closet_env_sample_payload` into `closet_env_sample`. A stale sample has NULL readings and is also filtered by flag.
2. **Independence.** It is not a copy of the cabinet (gap correlation 0.60 in level, 0.31 in 5-min changes). It is not a copy of the room (0.11). As a load signal it is redundant with circe's watts. Its new content is the physical response at circe's exhaust, which the cabinet cannot see. So it is used only as a heat-balance target, never as a load input.
3. **Theory anchor.** Steady-state air heat balance: air warms by P / (m_dot c_p) when it passes a heat source. This is the same anchor as the cabinet fit.
4. **Live sanity.** Range -0.29..+3.92 C. It is not flat or saturated. At light load it returns to its floor within one 5-min bucket. It is a difference of two live readings, so it cannot be a decay artifact. Caveat: a possible fixed offset between the SCD40 and the SHT40 (open question 6), so use slopes only.
5. **Existing mechanism.** This reuses `heat_balance.py` and adds no new estimator. `heat_sensitivity.yaml` is a different quantity, and this fit does not feed it.
6. **Reversibility.** Revert one series entry and one dict entry. The stored rows are inert.

## Env/config changes

- Added / removed / renamed keys: none.
- `.env_example` updated: no. Local `.env` sync: not needed. Skipped keys: none.

## Tests run

```text
cd services/orion-prediction && PYTHONPATH=../..:. python -m pytest tests -q -p no:cacheprovider   -> 478 passed (re-run on the review-fix commit: 478 passed)
PYTHONPATH=$PWD python -m pytest -q scripts/analysis/tests/test_body_economy_replay_and_fit.py        -> 20 passed
python scripts/check_env_template_parity.py                                                            -> PASS
```

## Evals run

```text
evals/run_causal_calibration_eval.py       -> RESULT: PASS
evals/run_heat_per_watt_eval.py            -> RESULT: PASS
evals/run_cabinet_replay_eval.py           -> RESULT: PASS
evals/run_turn_surprise_replay_eval.py --fixture evals/fixtures/turn_self_bets_chat_2026-11-13.json --check -> CHECK PASS
evals/run_air_path_eval.py --days 30 (live DB, read-only, 2026-11-15 00:30 UTC):
WINDOW 2026-10-16 00:00..2026-11-15 00:00 UTC asked (30 d); closet-top readings from 2026-11-05 00:45 (2871 5-min buckets)
LEAD_LAG closet top vs cabinet (+ = closet top first): 5-min changes peak at lag +0 min (r 0.54; lag 0 r 0.54; -5/+5 min r 0.27/0.43; n 2870); 15-min changes peak at lag +0 min (r 0.75; lag 0 r 0.75; -15/+15 min r 0.46/0.56; n 956)
FORECAST one-hour cabinet change, test 11-12 00:50..11-14 22:55 UTC (842 points): MAE without closet heat 0.632 C, with 0.632 C; paired one-sided p (with < without) 0.197
SLOPES closet top minus office (21 d window, 238 h, R^2 0.60): circe +0.27 C per 100 W (95% +0.21..+0.33, p 1.64e-17, mean 982 W); hecate refused: flat: only 2% of hours more than 20 W from its median 0 W (need 10%)
RESIDUAL daily mean of the closet-top gap fit (C; * = above +0.5 C, the stuck-air watch; fit on circe, athena, athena.cabinet_cooling_watts, 238 h): 11-05 -0.00, 11-06 -0.40, 11-07 +0.87*, 11-08 +0.05, 11-09 -0.08, 11-10 -0.14, 11-11 -0.19, 11-12 -0.17, 11-13 +0.04, 11-14 +0.06. Cabinet 32 C crossing days: 2, residual high in the 6 h before: 1
GATE second patch (closet heat into the cabinet forecast / cooling act): NOT MET (paired p 0.197 vs < 0.05; residual-led 32 C days 1 vs >= 3)
```

These numbers agree with the design. The closet top does not lead the cabinet. It does not improve the cabinet forecast. circe's slope sits inside the design's +0.16..+0.32 range. 11-07 is still the only stuck-air day. The window asked for 30 days, but closet data starts 2026-11-05, so about 10 days are real. The office-sound line from the design is dropped, because Juniper said office noise is not a cost.

## Docker/build/smoke checks

```text
Not run: no Dockerfile, dependency, or compose change. Deploy is a plain rebuild (below). The live check after deploy is listed under Restart required.
```

## Review findings fixed

The code review ran in a subagent. No must-fix findings.

- Finding: the "cabinet unchanged" test passed for the wrong reason (the monkeypatch had no effect).
  - Fix: a golden cabinet result (slopes, intervals, r2, hours, summary, refusals) computed on origin/main af420fb7b, asserted exactly.
- Finding: a closet fit added an empty `closet.top_temp_c` entry to `/health` through `Engine.effective`.
  - Fix: `load_frame` folds change events only for forecast targets. A test asserts the closet never joins the engine state.
- Finding: the eval's "6 h before a 32 C crossing" window included the hour containing the crossing.
  - Fix: only hours that end by the crossing count.
- Finding: README sentence about `/experiments/guard` was spliced onto the new paragraph.
  - Fix: moved back.
- Nits fixed: the forecast test skips 12 rows after the train/test cut (no target leak). The Newey-West lag is documented as approximate. A day that starts hot is no longer a "crossing". RESIDUAL exits non-zero with under 24 complete hours, and the docstring matches. The `main.py` docstring says "target". The yaml numbers carry their window. The 422-body change is listed above.

## Restart required

```bash
scripts/safe_docker_build.sh orion-prediction up -d --build
```

Run it from the clean primary `main` checkout after merge. Live check after deploy (read-only):

```bash
curl -fsS "http://localhost:8132/effects/heat-per-watt?series_id=closet.top_temp_c&days=7" | jq '{n_hours, circe: .slopes["circe.chassis_watts"] | {per_100w, ci_lo_per_100w, ci_hi_per_100w}, refused}'
curl -fsS "http://localhost:8132/effects/heat-per-watt?days=7" | jq '.summary'   # the cabinet answer, same shape as before
```

Pass: circe's 95 % interval overlaps +0.16..+0.32, and `refused` names hecate while hecate has run fewer than 10 % of hours.

## Risks / concerns

- Severity: low. Concern: a later patch reads the closet-top slope as cabinet heat (~18x larger for circe) and sheds circe work against the cabinet's 32 C line. Mitigation: separate `series_id`. `latest_heat_fit` filters by series_id (tested). Comments in the yaml, `HEAT_BALANCE`, and the README forbid feeding `heat_sensitivity.yaml` or the heat quote (#3194).
- Severity: low. Concern: a fixed SCD40-vs-SHT40 offset is read as "Orion adds 0.7 C at rest". Mitigation: slopes only, documented. Open question 6 is still open.
- Severity: info. hecate has run only since 2026-11-14 20:00 UTC, so its closet-top slope is not fittable yet. Re-check in its first full week, and re-run the eval at 30 days (~12-05).
- Issues opened: #3410 (`verify-live`: post-deploy GET, the hecate-week recheck, the 12-05 eval re-run). Issues closed: none. Design: #3234.
