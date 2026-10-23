# Body economy: heat sensitivity and price replay — findings (patch A4, corrected)

Date: 2026-10-23. Design: `docs/superpowers/specs/2026-10-23-organism-body-economy-design.md` (PR #2855).
All of this is read-only evidence. Nothing live reads these numbers yet.

**Correction (supersedes the version merged in #2861).** The first version had two problems:
- It modelled weather as a straight-line outdoor-temperature term.
- It treated the 10-16/17 cabinet event as unexplained "physical change" heat.

orion-prediction had already worked out both:
- Weather acts on the cabinet through the house's central AC. That is the `athena.central_ac_proxy` covariate (#2829), checked against the house meter.
- The 10-17 00:35 event is a recorded change in the sensor's position, not heat. It is change event #2780: "cabinet sensor geometry changed … cabinet reads ~2 C higher after". The vision bring-up with the cabinet open (10-16 21:30 → 10-17 00:35) is an exclusion window.

This version reuses those definitions, rows and windows exactly.

## The headline in plain words

**The cabinet reading rose 4.2 °C, from 29.1 °C (09-26..10-10) to 33.3 °C (10-18..10-22). About half the rise is the sensor reading higher, not heat. Most of the rest is real warming that began mid-October, in the same days the house's central AC went quiet. Orion's own load explains only about 0.1–0.4 °C.**

How the 4.2 °C splits (primary model):

| what | °C of the 4.2 °C rise | how sure |
|---|---|---|
| **Sensor reads higher** after it was moved on 10-17 (a measurement offset, not heat) | **+2.0** | Taken from #2780's recorded ~2 °C, not estimated here. If the fit estimates it freely, it takes 3.3–3.7 °C, because it also absorbs the warming that began two days earlier. |
| **Warming from mid-October that no continuous signal explains** | **+1.5 to +2.1** | Real heat. In the fit it lands on the 10-15 T10-swap step (+2.09, CI +1.03..+2.93). Without that step, +1.54 stays unexplained. |
| …of which the T10 card's own extra power can explain | at most about **+0.9** | The swap raised athena's chassis draw by 44 W. Even at the fan-inflated chassis sensitivity, that is ≤0.9 °C. At the compute sensitivity it is about 0.1. |
| …the rest, about **+1 to +2 °C**: the house's central AC going quiet | best-supported reading, **by timing** | The house meter shows the house's afternoon load outside the cabinet falling from 2.6–5.6 kWh/h (through 10-10) to 1.2–1.6 kWh/h from 10-16. |
| **Orion's own load** (athena + circe) | **+0.13** (up to +0.46 if the T10 step is removed) | athena's compute draw rose 23 W and circe's 16 W. |
| **Portable cabinet AC** | +0.14 | It draws more when the cabinet is hot. This is a mirror of the heat, not a cause. |

**Where this confirms Juniper's correction, and where the data stops short.**

- **It confirms:** the extra heat is not Orion's load, and it is not a linear effect of outdoor temperature. The remaining real warming starts exactly when the central AC stops running in the afternoons.
- **It stops short:** neither central-AC signal explains the warming when it is fitted directly.
  - The `central_ac_proxy` coefficient is +0.02 °C per °F (CI −0.15..+0.17).
  - The house-meter AC load contributes +0.01..+0.02 °C.
- **Why it cannot.** The proxy reads 0 from 10-11 01:00Z onward. It cannot see a change on 10-15..17. The house meter counts only whole kWh, carries all other household load, and lags about 3 days. The AC stopping, the T10 swap (10-15 18:57) and the sensor move (10-17 00:35) all fall within the same ~2 days.
- **So:** "weather through the central AC" is the leading explanation for the remaining 1–2 °C, but on this data it rests on timing. It is not a fitted effect. A controlled check would settle it: run the central AC for a few warm-ish afternoons with Orion's load held steady.

**Is the daily outdoor correlation of +0.05 a contradiction?** No. The corrected model expects it. On warm days the central AC runs and cools the house. On mild days it is off and the house warms. On cold days the house is cool. Daily cabinet temperature therefore does not track outdoor temperature in a straight line.

**Does this overturn the design's Fact 2?**
- "Heat is mostly weather" **stands**, with a mechanism: weather acts through the central AC, not directly. Add a caveat: about 2 °C of the "hot spell" is the sensor reading higher.
- "Orion's biggest GPU host is weakly tied" **stands**.
- The doc's explanation of the circe 0.10 vs 0.22 discrepancy is **wrong**. The discrepancy comes from the window, not from the watts source (see below).

## How much Orion's load moves the cabinet, per host

Script: `scripts/analysis/fit_heat_sensitivity.py`. The data is 15-minute bins from 09-26 to 10-23, n = 2,550, with the #2780 exclusion window masked.

**Primary model:**
- each host's CPU + GPU watts at 0/15/30/60-minute lags;
- orion-prediction's `athena.central_ac_proxy`, read through its own registry parser (outdoor °F above 75, 3 h earlier, held for 60 min);
- the portable AC (`home_cooling_sample`, as `athena.cabinet_cooling_watts`);
- three hour-of-day harmonics;
- the #2780 sensor break as a **fixed 2.0 °C measurement offset** (the reading after the break is lowered by 2 °C);
- the T10-swap break as a free step.

The sensitivity is the sum of the lag coefficients, in °C per 100 W held steady. The interval is a bootstrap that resamples whole days.

| model | athena °C/100 W (95% CI) | circe °C/100 W (95% CI) | R² |
|---|---|---|---|
| **primary** | **+0.40** (−1.04..+1.47) | **+0.28** (−0.08..+0.67) | 0.38 |
| sensor break size estimated (free step) | +0.48 (−1.04..+1.53) | +0.31 (−0.06..+0.70) | 0.60 |
| + linear outdoor term | +0.52 (−1.03..+1.51) | +0.43 (+0.10..+0.74) | 0.54 |
| house-meter AC load instead of the proxy | +0.29 (−1.25..+1.29) | +0.32 (−0.09..+0.69) | 0.34 |
| one dummy per day | +0.50 (−0.16..+1.01) | +0.24 (−0.01..+0.48) | 0.87 |
| before the sensor break only | +0.43 (−0.96..+1.42) | +0.37 (−0.01..+0.76) | 0.27 |
| + hecate (from 10-04) | +0.13 (−1.78..+1.36) | +0.15 (−0.30..+0.69) | 0.35, hecate +0.55 (−0.03..+1.67) |
| **hourly differences (2 h step response)** | **+0.62** (+0.18..+0.95) | **+0.35** (+0.24..+0.48) | — |

**What a typical swing is worth** (each host's p10 → p90 compute watts):

| host | swing | over a day | within 2 h |
|---|---|---|---|
| circe | 366 W | +1.0 °C (CI −0.3..+2.4) | **+1.3 °C** (CI +0.9..+1.8) |
| athena | 51 W | +0.2 °C | +0.3 °C |

- The 2-hour response is the robust number. It removes slow drift by construction, and its interval clears 0 for both hosts.
- The day-level numbers have wide intervals. The residual still drifts slowly (lag-1 r = 0.98).

**hecate** was awake for only about 23 hours, so its estimate is a hint, not a number.

### The correlation discrepancy (circe 0.10 vs 0.22)

Exact SQL: `CORR_SQL` in `fit_heat_sensitivity.py`. It averages each UTC hour and joins it to `orion_energy_hourly` on the same hour.

| window | circe (chassis / energy) | athena (chassis / energy) | cooling |
|---|---|---|---|
| 10-07..10-20 | **0.218 / 0.218** | 0.475 / 0.414 | 0.214 |
| 10-09..10-22 | **0.099 / 0.098** | 0.583 / 0.569 | 0.455 |

- Both values are right for their own window. Chassis watts and rollup watts agree to three decimals.
- Moving the window by two days swings these correlations by up to 0.24, so they are not a sensitivity.

## Price replay (rerun: the sensitivities moved)

Script: `scripts/analysis/replay_body_economy.py`. Price math: `orion/physiology/prices.py`.

**Setup.**
- Replay window: 10-07..10-22, hourly.
- Soft line: 31 °C. Horizon: 6 h.
- Rule: `eta = 0.25`, `max_step = 0.25`, ceiling 1.0. The ceiling is a placeholder. It is not sized from data, because no wants or act costs exist yet.

**Sensitivities used.** The design's rule applies: use the upper bound when the interval includes 0. That gives athena 1.47, circe 0.67 and hecate 1.67 °C per 100 W. In the first version these were 1.03, 0.44 and 0.48. The intervals are wider now, because the central-AC proxy and the change-event steps take over day-level variation that the old outdoor term had attributed to load.

**The replay is open-loop.** Nothing in it lowers Orion's load when the price rises.

**The cabinet reading is raw.** The replay uses the raw reading, as the live reflex does. After 10-17 that reading is about 2 °C above the earlier sensor's scale, so the soft line binds earlier than it would have before the move.

| check | verdict | numbers |
|---|---|---|
| **R1** | **FAIL with the literal rule. PASS with bind hysteresis (oracle).** | Over the line: 100% priced. Under the line: 88–95% priced ~0. The misses are the decay tail, plus 4 h in one variant. Unmeetable hours flagged, judged on the realized reading: 88–96%; the gate is ≥ 90%. The oracle + 0.5 °C hysteresis variants pass outright. |
| **R3** | **FAIL with the literal rule and with upper-bound sensitivities. PASS with point sensitivities (oracle).** | With the larger upper bounds, Orion's estimated share holds the price at the ceiling, unflagged, for 18–20 h. With point sensitivities, oracle passes (5 h to 0, ceiling run 2 h). Persistence mode still oscillates 0.50–1.00 around the 10-16 event. |
| **R4** | **PASS** | Realized $197.17 for 09-23..10-22. Worst error −6.1%; rate-only forecast −9.3%. No-imputation check on athena + circe: worst −7.1%. (Unchanged: R4 does not use the sensitivities.) |
| R2, R5, R6, R7 | NOT YET EVALUABLE | Each needs wants, bids, lane prices or per-act downstream use. None of these exist yet. |

### What this means for Patch B

1. **Ramp into `exogenous_bind` at `max_step`** (`bind_ramp`). Do not jump straight to the ceiling.
2. **Add hysteresis to the bind flag** (0.5 °C as replayed).
3. **Excuse one horizon of price decay in R1.**
4. **Correct the heat level for #2780's measurement offset** before comparing it with the soft line. Otherwise the line silently moves 2 °C after a sensor change. Patch B should read the same change-events table.
5. **Reconsider "use the upper bound when the interval spans 0."** With the corrected fit the upper bounds are 3–4× the point estimates. That makes Orion's own share look large and holds the price at its ceiling.

## Caveats

- **Timing confound.** The AC going quiet, the T10 swap and the sensor move fall within the same ~2 days in mid-October. On this data, their split comes from timing and from the T10 card's measured watts, not from separate variation.
- **The sensor offset is fixed, not estimated.** The fit takes #2780's "~2 C" as given. If the true offset is 1.5 or 2.5 °C, the warming share moves by the same amount, in the opposite direction.
- **The proxy has had no signal since 10-11.** It cannot carry any October change after that. Re-run `services/orion-prediction/evals/run_house_heat_balance_eval.py` when warm weather returns. On 10-23 it gives +0.30 kWh/h per °F above 75 °F (p = 3e-16, 604 h).
- **This is a passive fit.** The randomized holdback pulses described in the design should overrule it.
- **The sensor sits on athena.**
- **Each window happens only once.** There is one 30-day window for R4 and one hot spell for the rise split.

## Reproduce

Both scripts are read-only: SELECT in a `read only` transaction through `docker exec orion-athena-sql-db psql`.

```bash
.venv/bin/python scripts/analysis/fit_heat_sensitivity.py --out-json /tmp/heat_fit.json --out-md /tmp/heat_fit.md
.venv/bin/python scripts/analysis/replay_body_economy.py --fit-json /tmp/heat_fit.json --out-json /tmp/replay.json --out-md /tmp/replay.md
```

The fit reads these sources:
- `orion_biometrics_cluster`;
- `prediction_weather_hourly`;
- `home_cooling_sample`;
- `energy_usage_interval` (the house meter);
- `prediction_change_event` (non-retracted, scope `*` or `athena.cabinet_temp_c`);
- `config/prediction/series.v1.yaml`, through `services/orion-prediction/app/series.py`.
