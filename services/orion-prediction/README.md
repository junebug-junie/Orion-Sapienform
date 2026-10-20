# orion-prediction

Forecasts Orion's telemetry, grades every forecast against what really happened, and estimates
cause and effect ("if X happens, how much does the cabinet warm?"). First series: the Athena
cabinet temperature.

Spec: `docs/superpowers/specs/2026-10-15-orion-prediction-service-design.md`.

## What it does, in order

1. **Select** (every 6 h). Backtest every model from 40 past starting points. At each forecast
   distance (5 min, 15 min, 1 h, 6 h, 24 h), pick the model that beats the better of two
   baselines and whose error ranges are calibrated. If none does, serve the baseline there.
2. **Issue** (every completed 5-minute bucket). Fit the chosen models, stitch their forecasts,
   attach error ranges built from their own past errors (split-conformal), and compute the chance
   of crossing each declared threshold (34 °C shed, 32 °C experiment guard). Store it, publish it.
3. **Score**. When the real bucket arrives, write the actual value, the error, and whether it fell
   inside the 80 % and 95 % ranges. A bucket with no readings is scored `missing`, never guessed.
4. **Monitor**. Alerts in `/health` and `/series/{id}/live` (and one log line when they change):
   live coverage out of band, worse than baseline over 24 h, stale input, stale forecast,
   degenerate output (refused, never stored), failed selection.

## Models

| model | role |
|---|---|
| `naive` (same as now), `seasonal_naive` (same time yesterday) | baselines; the better one per horizon is the bar |
| `window_average_1h` | short-horizon smoother |
| `mstl_ets`, `mstl_arima` | MSTL decomposition (cycles up to 1 day) + ETS / ARIMA trend |
| `prophet` | changepoints + every declared seasonality as Fourier terms (incl. weekly, custom) |
| `direct_ar` | one ridge regression per lead time from the temperature's own recent history (`app/direct.py`) |
| `direct_arx` | the same plus the series' covariates (chassis watts, AC watts) as known at the origin -- never a future value |

`direct_ar` vs `direct_arx` is a standing ablation: the scoreboard shows what the covariates add.

A seasonality is used only when training holds 2 full cycles (monthly is refused today). MSTL
skips cycles longer than a day: STL at the weekly period costs ~26 s per fit vs ~2.3 s.

## Change events

Known moments the world changed under a series, stored in `prediction_change_event`:

| effect | what the forecaster does |
|---|---|
| `exclude` (start, end) | masks the window out of training and backtest actuals (e.g. the cabinet door open) |
| `break` (a moment) | Prophet fits a level shift there at once; after 10 days of data past it, every model trains only on data after it, and the selection reruns immediately |

`POST /change-events` (`scope` = a series id or `*`), `GET /change-events`; `/health` shows each
series' pending and adopted breaks. Seeded 2026-10-20 with the athena T10 swap (break, 10-15 18:57 UTC)
and the vision bring-up (cabinet open 10-16 21:30 -> 10-17 00:35 excluded; sensor geometry changed:
break at 00:35).

## Cause and effect: three levels, always labelled (`claim_kind`)

| level | endpoint | what it can claim |
|---|---|---|
| `conditional` | `POST /scenario` | "if chassis watts follow this path, temp is forecast to…" -- correlation. Live example: more AC power *goes with* a hotter cabinet, because the AC works harder when it is hot. |
| `observational_effect` | `POST /interventions` then `POST /effects/estimate` | interrupted time series; interval and p-value from placebo runs at earlier times |
| `experimental_effect` | `POST /experiments/design`, log blocks, `POST /experiments/{id}/analyze` | randomised ON/OFF switchback; the only causal answer |

### Does event A make event B more likely? (`POST /effects/event-pair`)

Always `observational_effect`. Reads `vision_events` read-only (10 s statement timeout, same
`conjourney` database as the telemetry). Body:

```json
{"a": {"event_type": "doorbell_ring"},
 "b": {"event_type": "person_presence", "stream_id": "walkway"},
 "start": "2026-10-06T00:00:00Z", "end": "2026-10-20T00:00:00Z",
 "lag_min_sec": 0, "lag_max_sec": 300, "n_placebo": 999, "seed": 0, "placebo": "same_time_of_day"}
```

Each A opens a window `[a + lag_min, a + lag_max)`; overlapping windows merge, so no B counts twice.
`rate_ratio` = B seen in the windows / (B's rate over the whole range x window length). The p-value
comes from shifting all of A by one random offset many times (`p = (1 + #placebo >= real) / (N + 1)`).
`same_time_of_day` (default) shifts by whole days +/- 1 h, so two streams that only share the daily
rhythm read as no effect (needs >= 7 days; the range is cut to whole days); `circular` shifts anywhere (38 % false positives on daytime-only
independent streams, ~5 % on rhythm-free ones). Too few A (< 5) or no B -> `422` with
`detail.error = "refused"` and the counts. Each answer is stored in `prediction_effect_estimate`
(`action = event_pair:<A>`, `series_id = <B>`, `effect` = rate ratio, details hold every count).
Counts are rows, not visits.

Live (2026-10-20, 14 days): no `doorbell_ring` rows exist yet, so the doorbell pair refuses;
walkway presence -> walkway caption (0-60 s) 32x, p = 0.006.

`GET /effects/{action}?series_id=` returns the best available level. `GET /experiments/guard`
says whether it is safe to start an ON block (P(≥ 32 °C within 30 min) < 5 %).

**No dispatcher exists.** Nothing here switches a workload on or off. Whoever performs the action
logs each block (`POST /interventions` with `experiment_id` + `arm`) and re-checks the guard
before each ON block. The 34 °C reflex in hardware-watch stays the hard stop.

## Contracts

- Bus: `orion:prediction:forecast` (`PredictionForecastV1`, `orion/schemas/prediction.py`).
  Consumer: orion-hardware-watch shows the shed probability beside each reflex decision
  (observe-only).
- Tables: `services/orion-sql-db/manual_migration_prediction_v1.sql`, `manual_migration_prediction_change_events_v1.sql`.
- Series registry: `config/prediction/series.v1.yaml` (baked into the image).

## Run

```bash
# apply the migration first (additive):
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_prediction_v1.sql
scripts/safe_docker_build.sh orion-prediction up -d --build      # from clean primary main only
curl -fsS http://localhost:8132/health
curl -fsS http://localhost:8132/series/athena.cabinet_temp_c/models
```

## Tests and evals

```bash
cd services/orion-prediction
PYTHONPATH=../..:. python -m pytest tests -q
PYTHONPATH=../..:. python evals/run_causal_calibration_eval.py          # planted effects, many seeds
PYTHONPATH=../..:. python evals/run_cabinet_replay_eval.py --windows 16 # real cabinet history
POSTGRES_URI=... PYTHONPATH=../..:. python evals/run_event_pair_live_eval.py  # live vision pairs, read-only
python evals/export_cabinet_fixture.py --until <iso>                     # refresh the fixture
```
