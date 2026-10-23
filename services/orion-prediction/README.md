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

## Covariates

| series | what | source |
|---|---|---|
| `athena.chassis_watts` | athena power (iLO) | biometrics |
| `circe.chassis_watts` | circe power (PDU, read by athena) | `orion_biometrics_cluster` |
| `hecate.chassis_watts` | hecate power (BMC, read by athena); 0 before 2026-10-04 | `orion_biometrics_cluster` |
| `athena.cabinet_cooling_watts` | portable AC power (Shelly plug) | `home_cooling_sample` |
| `athena.central_ac_proxy` | outdoor F above 75, 3 h earlier: the central AC's switch | `prediction_weather_hourly` |

Weather comes from the shared Open-Meteo client (`orion/weather/openmeteo.py`, also used by the
situation layer) at the situation layer's location (`ORION_SITUATION_WEATHER_LAT/LON`): refreshed hourly
(past 7 days + next 48 h), backfilled from the archive on first start, observed values replacing
forecasts. `evals/run_house_heat_balance_eval.py` re-checks the proxy against the house meter.

A seasonality is used only when training holds 2 full cycles (monthly is refused today). MSTL
skips cycles longer than a day: STL at the weekly period costs ~26 s per fit vs ~2.3 s.

## Shadow scoring: live evidence for the gate

Every model that beat the baseline somewhere in the backtest (plus whatever serves) also forecasts at
every issue, unserved, into `prediction_shadow_forecast`; the reducer scores those rows like the served
ones. Per model and horizon, `app/live.py` pairs them with the baseline forecast for the same origin,
keeps one origin per horizon-long time block (independent, anchored to the clock), and with >= 20 of
them gives a verdict: `better` (Holm-adjusted p < 0.05 across every model x horizon tested), `worse`
(p < 0.05), `unclear`, or `pending`. Each horizon looks back long enough to collect 20 (25 horizons,
at least 7 days: 24 h -> 25 days); shadow rows are kept 35 days. The gate then:

- never serves a model that is live-`worse`;
- serves a live-`better` model that also beats the baseline in the backtest, even on a borderline p;
- otherwise uses the backtest: p < 0.05 to start serving, p < 0.10 to keep serving (no flicker);
- ranks live winners before backtest-only winners.

Shadow models are fitted after the served forecast is stored and saved in their own transaction, so
they never delay or cost it; failures show in `/health` (`shadow_failed`).

`worse_than_baseline` alerts on the SERVING model's own live verdict. `/live` lists every verdict.
Cost: ~13 s of CPU per 5-minute issue with four contenders (measured 2026-10-22). Rollback lever:
`PREDICTION_SHADOW_ENABLED=false` -- which also turns off `worse_than_baseline` (it reads live verdicts).

## Grading the overheat warning

Every issued forecast stores each threshold's probability (`prediction_exceedance`). Once its window
has passed it is scored two ways: `hit_raw` (any single reading crossed -- what the 34 C reflex acts on)
and `hit_mean` (a 5-minute mean crossed -- what the forecast models). `GET /series/{id}/live` returns,
per threshold and per outcome: Brier score, the base-rate Brier, skill (above 0 = knows *when*, not just
*how often*) and calibration bins. `worse_than_baseline` now compares only forecasts where champion and
baseline differ, and needs >= 5 % worse with p < 0.05.

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

## Action bets

An action bet is a yes/no prediction about one of Orion's own body actions: "this action, in this
situation, at this place, works with probability p". The body adapter (the Hub for Zelda) asks for a
quote on each possible move, places a bet on the move it starts, and settles the bet when the move
ends. The service only sees opaque strings (`skill`, `context` values, `place`); what a skill means
and when it succeeded is the body's business. Spec:
`docs/superpowers/specs/2026-10-23-embodied-experience-learning-design.md` ("the action bet").
Contracts: `ActionBet*V1` in `orion/schemas/prediction.py` (HTTP only, no bus channel). Code:
`app/action_bets.py`.

| endpoint | does |
|---|---|
| `POST /action-bets/quote` | `{domain, body, options: [{option_id, skill, context, place?}]}` (at most 20) -> `{domain, body, served_model, quotes: [{option_id, model, p, p_baseline, interval80, eig_nats, level, counts}]}`. Read-only, one grouped count query. |
| `POST /action-bets` | place: `{bet_id, domain, body, skill, context, place?, action, chosen_rank?, session_id?}` -> the stored bet. The service computes `p` and `p_baseline`; a caller cannot send them (422). Idempotent by `bet_id`: a repeat returns the stored bet unchanged. |
| `POST /action-bets/{bet_id}/settle` | `{result: success\|failure\|void, outcome, detail?}` -> the stored bet. Open -> settled only: the same result again is 200, a different one 409 (`detail.bet` is the stored bet), an unknown bet 404. |
| `GET /action-bets/live?domain=&body=` | per-skill and per-situation posteriors (counts, mean, interval80, eig), the top 20 place beliefs, grading of the served and baseline p, the live verdict, open / never-settled / void counts (24 h), the last 20 bets |
| `GET /action-bets?session_id=` | that session's bets, oldest first, at most 500 |

**Model** `action_beta_backoff` (served): a Beta-Bernoulli rate with three levels of backoff, strength
k = 4, over settled `success` / `failure` bets (void never counts). Skill: `Beta(1+s, 1+f)` over every
domain and body. Situation (same domain, skill, exact context): prior = the skill mean with strength k.
Place (same domain, place): prior = the situation mean with strength k. `p` is the mean of the deepest
level with at least one settled bet (`level`); no data -> `prior`, p = 0.5. `interval80` is the
10 %/90 % Beta quantile of that level; `eig_nats` the expected information gain of one more try
(0.193 at no data, 0.022 at 10 of 20). With k = 4, five failures at a place read p < 0.2 only when the
situation mean is below about 0.45.

**Baseline** `action_skill_rate`: the skill mean alone, stored on every bet as `p_baseline`. Each bet
also stores its counts at placing, so the model's p is recomputable for every bet even while the
baseline serves.

**Gate**: squared error of the model's p vs the baseline's, one pair per session (a bet without a
session is its own unit), the paired test from `app/live.py`/`backtest.paired_win_p`: `pending` under
20 sessions, then `better` / `worse` / `unclear` at p < 0.05. Global over every domain, cached 60 s.
When the model is live-`worse`, quotes and new bets serve the baseline (`model` / `served_model` say
`action_skill_rate`). Grading of served and baseline p uses `app/grading.py` `grade` (`enough` at
n >= 50; `reportable` false under 10 bets).

**Expiry**: a bet still open 1 h after placing is closed `void` with outcome `never_settled` (checked
at most every 30 s, on quote / place / live). Never graded; counted in `/health`
(`action_bets: {open, never_settled_24h, served_model}`) and `/live`. Log lines: `action_bet_placed`,
`action_bet_settled`, `action_bet_expired`.

**Table** `prediction_action_bet`, operator migration (also auto-applied by the post-pull rebuild;
the service checks it at boot like every other prediction table):

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_prediction_action_bet_v1.sql
```

## Contracts

- Bus: `orion:prediction:forecast` (`PredictionForecastV1`, `orion/schemas/prediction.py`).
  Consumer: orion-hardware-watch shows the shed probability beside each reflex decision
  (observe-only).
- Tables: `services/orion-sql-db/manual_migration_prediction_v1.sql`, `manual_migration_prediction_change_events_v1.sql`, `manual_migration_prediction_action_bet_v1.sql`.
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
