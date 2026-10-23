# Body economy: heat sensitivity and price replay — findings (patch A4)

Date: 2026-10-23. Design: `docs/superpowers/specs/2026-10-23-organism-body-economy-design.md` (PR #2855).
Everything here is read-only evidence. Nothing live reads these numbers yet.

## The headline in plain words

**Yes, Orion's own load moves the cabinet temperature, but only a little, and it is not why the cabinet has been hot.**

- **circe** (the big GPU host): when circe's work steps up from its quiet level to its busy level, the cabinet warms by about **1.3 °C within two hours** (95% interval 0.8 to 1.8 °C). This effect is clearly real. Over a whole day it is smaller and less certain: about 0.6 °C, and the interval includes zero.
- **athena** (the host the sensor sits on): athena's work barely moves. Its busy-versus-quiet swing is only about 50 W, and that swing is worth about **0.3 to 0.4 °C**.
- **hecate**: about 0.5 °C per 100 W, but hecate was awake for only about 23 hours in the window. Treat this as a hint, not a number.
- **Weather does not explain the cabinet either.** Across 27 days, the daily cabinet average has **no correlation with outdoor temperature** (r = +0.05). The hottest cabinet days (10-18..22, 32–34 °C) had cool outdoor days (13–17 °C).
- **Something physical changed in the cabinet on 10-16, around 15:00–19:00 MDT** (21:00–01:00 UTC). The cabinet's distance sensor swung 131 → 92 → 36 → 88 mm and settled at 94 mm, down from about 116 mm. Humidity stepped down. The cabinet's overnight floor rose about 5 °C. Outdoor temperature and Orion's load did not change. The likely cause is that someone worked at the cabinet then: a sensor moved, a door was closed, or the airflow changed. `UNVERIFIED`: only Juniper can confirm what happened.

**Does this overturn the design's Fact 2?** Partly.

- Fact 2 says **"Heat is mostly weather."** That is **overturned**. Outdoor temperature explains almost nothing, either day to day or hour to hour. The day-level heat comes from something that is not on record. The 10-16 cabinet change is the main suspect.
- Fact 2 says **"Orion's biggest GPU host is weakly tied to the cabinet number."** That is **half right**. Over a day, the tie is weak. Within two hours, circe's load measurably warms the cabinet.
- Fact 2's explanation of the **0.10 vs 0.22** circe correlation is **wrong**. The difference is not chassis watts against energy-rollup watts. Both sources give the same value in a given window. The two passes used **different 14-day windows** (see below).
- Fact 2's conclusion still holds. Shedding Orion's work cannot cool a cabinet whose level is set by something else. A learned sensitivity is still the right instrument.

## Numbers: heat sensitivity

Script: `scripts/analysis/fit_heat_sensitivity.py`. Window: 2026-09-26 .. 10-23. Outdoor temperature first appears in `prediction_weather_hourly` on 09-26. The data is 15-minute bins, n = 2,559.

**Regressor.** Each host's CPU + GPU watts (`cpu_watts_total + gpu_watts_total`), not chassis watts. Chassis watts include the fans, and athena's fan duty tracks the cabinet temperature (r = 0.78). A hot cabinet raises chassis watts, so the arrow also points backwards. hecate reports only chassis watts.

**Controls.**
- outdoor temperature at 0, 1 and 3 h;
- AC watts;
- three hour-of-day harmonics;
- one dummy per day.

**Sensitivity.** The sum of the 0/15/30/60-minute lag coefficients, in °C per 100 W held steady.

**Intervals.** A bootstrap that resamples whole days, with a Newey-West interval beside it.

| model | athena °C/100 W (95% CI) | circe °C/100 W (95% CI) | n | R² |
|---|---|---|---|---|
| **primary** (day dummies) | **+0.54** (−0.18..+1.01) | **+0.18** (−0.11..+0.43) | 2,559 | 0.89 |
| before the 10-16 cabinet change only | +0.51 (−0.14..+1.04) | +0.17 (−0.12..+0.46) | 1,974 | 0.83 |
| primary, no outdoor temperature | +0.53 (−0.15..+0.99) | +0.17 (−0.11..+0.42) | 2,562 | 0.89 |
| primary + hecate (from 10-04) | +0.39 (−0.62..+0.95) | +0.23 (−0.17..+0.59) | 1,805 | 0.91 — hecate **+0.52 (+0.30..+0.75)** |
| no day dummies (confounded) | +3.15 (+1.02..+6.18) | +0.26 (−0.29..+0.77) | 2,559 | 0.25 |
| chassis watts, no day dummies | +3.21 (+1.57..+5.18) | +0.05 (−0.19..+0.30) | 2,549 | 0.35 |
| **hourly first differences** (2 h step response) | **+0.73** (+0.14..+1.13) | **+0.35** (+0.23..+0.48) | 638 h | — |

**What a typical swing is worth.** The swing is each host's p10 → p90 compute watts.
- circe swings 366 W:
  - over a day: +0.64 °C (CI −0.41..+1.59);
  - within 2 h: **+1.29 °C**.
- athena swings 51 W:
  - over a day: +0.27 °C;
  - within 2 h: +0.37 °C.

**Why the primary model has day dummies.** Without them, the residuals form a slow random walk (lag-1 r = 0.98, R² = 0.25). That drift loads onto any regressor that also trends. athena's estimate then jumps to an implausible 3 °C per 100 W. The day dummies absorb the day-level drift (R² 0.25 → 0.89). The cost is that they also absorb any real day-level effect of Orion's load. The first-difference model checks this from the other side: it removes slow drift by construction and still finds circe clearly positive.

**Outdoor temperature.** Within a day, the outdoor coefficients sum to about −0.03 °C per °C. Across days, r = +0.05. Dropping outdoor temperature changes no estimate by more than 0.02.

**AC.** The AC coefficient is +0.57 °C per 100 W, which is backwards for a cooler. The AC draws more because the cabinet is hot. Its hourly standard deviation is about 32 W, so, as the design warned, it controls for very little.

### The correlation discrepancy (circe 0.10 vs 0.22)

Exact SQL: `CORR_SQL` in `fit_heat_sensitivity.py`. It averages `orion_biometrics_cluster` per UTC hour and joins it to `orion_energy_hourly` on the same UTC hour.

| window | circe (chassis / energy) | athena (chassis / energy) | cooling |
|---|---|---|---|
| 10-07..10-20 (the design's stated 14 days) | **0.218 / 0.218** | 0.475 / 0.414 | 0.214 |
| 10-09..10-22 (the last 14 days as of 10-23) | **0.099 / 0.098** | 0.583 / 0.569 | 0.455 |
| 10-07..10-20, energy hour joined 6 h late | 0.218 / 0.111 | 0.475 / 0.394 | 0.077 |

- **Which value is right?** Both are right for their own window. The 0.10 / 0.58 / 0.46 pass used 10-09..10-22. The 0.22 / 0.41–0.47 / 0.21 pass used 10-07..10-20.
- **What this shows.** Moving the window by two days swings these correlations by up to 0.24, so they are not a sensitivity.
- **What it is not.** It is not a chassis-versus-rollup difference. Both sources agree to three decimals within each window.

## Numbers: price replay

Script: `scripts/analysis/replay_body_economy.py`. Price math: `orion/physiology/prices.py`.

**Setup.**
- Replay window: 10-07 00:00 .. 10-23 00:00 UTC, hourly.
- Soft line: 31 °C (the design's proposal). Horizon: 6 h.
- Rule: dual ascent with `eta = 0.25`, `max_step = 0.25`, `budget_floor = 0.25 °C`, and ceiling 1.0.

**Sensitivity used.** The design's rule applies: use the upper bound when the interval includes 0. That gives athena 1.01, circe 0.43 and hecate 0.52 °C per 100 W, applied to watts above each host's p10 baseline.

**The ceiling is not data-sized.** `size_ceiling()` exists, but it needs graded want values and per-act costs. No `WantV1` exists, and `grammar_acts.cost_json` is null, so 1.0 is a declared placeholder.

**The replay is open-loop.** Nothing in it lowers Orion's load when the price rises.

**Forecasts.** Two forecast modes are replayed:
- `oracle`: the realized mean over the next 6 h. This grades the rule alone.
- `persistence`: the last hour. Any live system could run this today.

| check | verdict | numbers |
|---|---|---|
| **R1** heat price ~0 under the line, >0 over it, unmeetable hours flagged | **FAIL as written. PASS outside the decay tail.** | Over the line: 100% of hours priced (176–179 h). Unmeetable with zero Orion load: 133/133 (oracle) and 137/137 (persistence) flagged `exogenous_bind`, none above the ceiling. Under the line: 91–95% of 205–206 h priced ~0. **Every** under-line miss (11–18 h) is the price still falling in the 6 h after a binding hour. There are no other misses. |
| **R3** back to 0 within a horizon, bounded oscillation, no unflagged ceiling run | **FAIL with the design's literal rule. PASS for one variant.** | Return to 0: at most 4–6 h in every variant (pass). Unflagged ceiling runs: 15 h (oracle) and 7 h (persistence) with upper-bound sensitivities. The cause is the open loop: Orion's share stays over a small budget and nothing backs off. Oscillation: a 0 → 1.0 jump when `exogenous_bind` starts, because the design freezes the price *at* the ceiling. The one real case is the 10-16 cabinet-opening event. With hysteresis on the bind flag, a bounded ramp into it (`bind_ramp`) and point sensitivities, oracle passes (max 4 h to 0, ceiling run 2 h, oscillation 0). Persistence still swings exactly 0.50 at the 10-16 event, against a declared amplitude of < 0.50. |
| **R4** month-end energy forecast within ±10% | **PASS** | Realized Orion cost 09-23..10-22: **$197.17** at $0.1213/kWh. The forecast (spent so far + trailing-7-day rate × hours left) ran 384 hourly forecasts: worst −6.1%, mean \|error\| 1.8%. The harder rate-only forecast (30 d × trailing 7-day rate): worst −9.3%, mean 4.1%, also within ±10%. 5.8% of the realized kWh is gap-filled with each source's median hour: cooling had no data before 09-26 and hecate none before 10-04. |
| R2 | NOT YET EVALUABLE | Needs non-GPU wants with values and costs. `WantV1` and `want_ledger` do not exist (patch C), and `cost_json` is null. |
| R5 | NOT YET EVALUABLE | Needs a posted lane price and bid-before-lease (patch E). |
| R6 | NOT YET EVALUABLE | Needs the reservation rule and a bid stream (patches B/C). |
| R7 | NOT YET EVALUABLE | Needs per-act downstream-use rates per family. These are not recorded. |

**Energy price dynamics.** All envelopes below are hypothetical, because Juniper has not set one.
- $180: the fixed load alone exceeds it, so the constraint is flagged `exogenous_bind` for 325 of 384 h and the price holds at the ceiling.
- $197 (about the realized cost): binding for 140 h, back to 0 within 2 h of clearing, no oscillation.
- $230: never binds.

R3 passes for every envelope.

### What the replay says Patch B should change

1. **Do not jump to the ceiling on `exogenous_bind`.** Ramp up at `max_step`; this is `PriceParams.bind_ramp`. The literal "freeze at ceiling" is an unbounded one-tick step, and it breaks the design's own bounded-step rule.
2. **Add hysteresis to the bind flag.** Use `exogenous_bind_with_hysteresis`, with 0.5 °C as replayed. Otherwise a forecast that wanders across the line chatters between the ceiling and the integrator.
3. **Rewrite R1 to excuse one horizon of decay after a binding hour, or accept a faster decay.** An integrator cannot read 0 in the hour a constraint clears. `slack_decay_floor = ceiling / horizon` shortens the tail (14 → 11 h) without removing it.
4. **Revisit the "upper bound when the interval spans 0" rule for athena.** With athena at 1.01 instead of 0.54, Orion's own share is inflated enough to hold the price at the ceiling for 15 h with no flag (open loop). With point sensitivities, oracle R3 passes.
5. **R3's "no unflagged ceiling run" clause cannot be graded open-loop.** It needs patch C's shadow bids, so that load can respond to the price.

## Caveats

- **This is a passive fit.** It is not a causal estimate. The design's randomized holdback pulses (Organ 4 / Organ 6 part 2) are what can settle it. Treat these intervals as the passive prior that the pulses should overrule.
- **The sensor sits on athena.** athena's coefficient may reflect local heat near the sensor rather than the cabinet as a whole.
- **Day dummies have a cost.** They remove any day-level effect of Orion's load, so the "over a day" numbers are lower bounds on what a sustained change of the daily average would do. The first-difference model covers the short run, but not multi-day accumulation.
- **The hecate estimate is weak.** It rests on about 23 awake hours across 5 days (10-04, 10-08, 10-09, 10-11, 10-15).
- **The weather source changes.** `prediction_weather_hourly` uses Open-Meteo `historical_forecast` rows up to 10-13 and live `recent` rows from 10-14. No level jump is visible at the seam (49 °F → 48 °F across midnight).
- **The 10-16 cabinet change is inferred, not confirmed.** The evidence is the lidar, humidity and temperature steps. If the sensor was moved, readings before and after 10-16 are not comparable. The fit restricted to before 10-16 gives the same sensitivities.
- **R4 has one window.** Only one fully observed 30-day window exists (09-23..10-22). The forecast also already knows at least 14 of the 30 days, which makes it easy. The rate-only variant is the stricter test, and it also passes.

## Reproduce

Both scripts are read-only. They run SELECT in a `read only` transaction through `docker exec orion-athena-sql-db psql`. Override the command with `ORION_ANALYSIS_PSQL`.

```bash
.venv/bin/python scripts/analysis/fit_heat_sensitivity.py --out-json /tmp/heat_fit.json --out-md /tmp/heat_fit.md
.venv/bin/python scripts/analysis/replay_body_economy.py --fit-json /tmp/heat_fit.json --out-json /tmp/replay.json --out-md /tmp/replay.md
```

Live inputs, as of the 10-23 ~04:50 UTC run:
- `orion_biometrics_cluster`: 171,930 rows, 09-23..10-23;
- `orion_energy_hourly`;
- `prediction_weather_hourly`: 648 observed hours from 09-26.
