# Orion's air path, read as one path

Status: **design, proposal mode** (touches regulation inputs). Docs only. No implementation until
Juniper says go. Asked for by Juniper on 2026-11-12.
Live numbers below were pulled read-only on 2026-11-12 ~06:50 UTC from the live database
(`closet_env_sample`, `office_env_sample`, `orion_biometrics_summary`, `orion_biometrics_cluster`,
`home_cooling_sample`) over 2026-11-05 00:45 .. 2026-11-12 06:50 UTC (2,090 five-minute buckets), and
from `GET /effects/heat-per-watt` on `orion-prediction` (port 8132, the read-only form; nothing stored).

## For Juniper (read this, skip the rest if you like)

**What we checked.** Orion's air moves in one path: office air goes into the cabinet (athena) and
past circe, the hot air rises to the top of the closet, and some of it comes back into the office.
Three sensors sit on that path: the office board, the cabinet board, and the closet-top board. We
tested three numbers built from them.

**What the data says.**

1. **Heat Orion adds (closet top minus office): keep it, but not for the cabinet forecast.** The
   closet top is the only place where circe's heat is visible. The cabinet sensor cannot see circe at
   all (+0.01 °C per 100 W, not different from zero, 21 days). The closet top sees circe clearly:
   about **+0.25 °C per 100 W** of circe power (95 % range +0.16..+0.32), and it moves in the same
   five minutes as circe's power. When hecate comes online, this is the number that will show its
   heat. It is not useful for the cabinet forecast: it does not move before the cabinet, and adding
   it to a one-hour cabinet forecast did not reduce the error (0.581 -> 0.583 °C).
2. **Stuck air (closet top warming faster than the office, or CO2 rising): drop it as proposed.**
   The warming-rate part is only the change in number 1, so it adds nothing new. The CO2 part
   cannot work: the machines make no CO2, and the office has no real CO2 sensor to compare with (the
   office "eCO2" is a guess from other gases). The closet CO2 follows the house (people, windows),
   not Orion. One honest version survives as a thing to watch, not to wire: on 11-07 the closet top
   ran 0.8 °C hotter for a whole day than circe's power explains. That is what slow, stuck air would
   look like. One day is not evidence.
3. **Orion's effect on the office: the warmth part already exists; the sound part is real but has
   no user here.** How much Orion warms the office is already measured (the "room free" heat fit in
   `orion-prediction`). The office sound is the surprise: it follows **circe's fan speed almost
   exactly** (correlation 0.93). Office sound goes from about 76 dBA when circe is light to about
   80 dBA when circe is busy. Most of the office noise is circe. But nothing in the heat path would
   use that number, so it is not part of this design. It is a question for you below.

**One recommendation:** the first patch only measures. Teach `orion-prediction` to read the closet
top and fit "heat added per 100 W" there for circe and hecate, beside the cabinet fit it already
has. No change to the cooling act, the forecast, or any price. The heat quote (#3194) keeps using
the cabinet. A replay eval re-checks "does closet-top heat warn early?" at 30 days and when hecate
runs, and only a pass there opens a second patch.

**Your questions (details at the end):** is the closet board really on athena (the code says
pan-1)? Where is the portable AC? Was the closet door shut on 11-07? Do you want Orion to weigh
office noise as a cost? Is there a closet-top temperature you care about for circe and hecate?

---

## Arsonist summary

The pitch "read the air path as one path" is right about the body and wrong about the forecast.

- The cabinet sensor sits in athena's air. Circe's exhaust does not reach it. The cabinet's local
  heat fit (`services/orion-prediction/app/heat_balance.py`) already says so: circe local slope
  +0.014 °C per 100 W, 95 % CI -0.029..+0.057 (21 days, live GET 11-12). So today Orion has **no
  sensor that sees circe's heat directly**. The closet top is that sensor. That is the real gap, and
  it is a perception gap, not a forecast gap.
- The closet top does **not** help the cabinet forecast. It moves at the same time as the cabinet,
  not before (both follow the office air). Adding it to a one-hour cabinet-change model changed the
  test error from 0.581 to 0.583 °C. The tempting move, "add closet heat as a covariate to the
  cabinet forecast and the heat quote", would add a slow copy of circe's watts and nothing else.
  Circe's watts are already in that model.
- "Stuck air" as proposed is a keyword in a lab coat. Its warming-rate half is the derivative of
  number 1. Its CO2 half has no source to trace (servers make no CO2) and no matched reference (the
  office board has no CO2 sensor, only a VOC-based estimate).
- "How loud Orion gets in the office" is the most surprising true finding, and it has no consumer in
  the heat path. Wiring it into the cooling act because it correlates would be the exact failure
  this repo bans.

So the patch is small: one source kind, one series entry, one heat-balance entry, one eval. It gives
Orion a measured answer to "how much heat do circe and hecate put into the closet", and it gives the
hecate arrival a number that will move.

## Current architecture

Physical layout (Juniper, 2026-11-12): office -> server closet -> server cabinet -> hardware. athena
sits inside the cabinet. circe and hecate sit **above** the cabinet, in the closet's top half. hecate
is on demand and was off all week (`hecate.chassis_watts` = 0 W in every bucket, 11-05..11-12).

| sensor | what it reads | host / clock | producer -> table | cadence (7 d rows) | read by today |
|---|---|---|---|---|---|
| office board (pan-1 Nano: SHT40, CCS811, PMSA003I, TSL2591, SEN0232 sound) | office air just outside the closet | pan-1, NTP synced (0.6 ms) | `services/orion-office-env` -> `orion:office:env:sample` -> sql-writer `office_env_sample` | 30 s (20,134) | `orion-prediction` series `office.temp_c` (target, and covariate of the cabinet), heat-per-watt room side; situational brief (`orion/situational/office_sound_reader.py`, `context.py:2611`) |
| closet-top board (SCD40: temp, RH, real CO2) | top half of the closet, circe/hecate exhaust | pan-1 (second Nano), same clock as office | `orion-office-env` (`ENV_ROOM=closet`) -> `orion:closet:env:sample` -> `closet_env_sample` | 30 s (20,135) | Hub panels only (`services/orion-hub/scripts/closet_env_routes.py`, `cabinet_climate_routes.py`) |
| cabinet board (athena Nano ESP32) | cabinet air at athena | athena, NTP synced | biometrics -> `orion_biometrics_summary.measurements->cabinet_temp_c` | ~31 s (19,330) | heat forecast (`athena.cabinet_temp_c`), cooling act (`services/orion-durable-runs/app/cooling_act.py`, `orion/autonomy/shed_act.py`), thermal lines (`orion/autonomy/thermal_gate.py`) |
| machine power | chassis watts per host | athena reads all three | `orion_biometrics_cluster.measurements_by_node` | 15 s (40,224) | forecast covariates, heat-per-watt, body economy heat cost (`config/physiology/heat_sensitivity.yaml`) |

Note: the code and the 2026-11-05 PR report say the closet SCD40 is on **pan-1's second Nano**, not
on athena (`closet_env_sample.node = 'pan-1'`, docstring "pan-1 SCD40"). The layout note says "a
sensor array from athena". Open question 1.

Existing heat machinery this design must fit with, not duplicate:

- `orion-prediction` heat per watt (`app/heat_balance.py`): local fit (cabinet minus room on each
  machine's watts) and total fit (cabinet, room free). One target declared: `athena.cabinet_temp_c`,
  room `office.temp_c`, watts athena/circe/hecate, control `athena.cabinet_cooling_watts`.
- S4 -> S3 heat quote (PR #3194, `docs/superpowers/specs/2026-11-11-s4-s3-heat-quote-design.md` on
  branch `docs/s4-s3-heat-quote`): forecast with/without work steers the cooling act. Uses the
  cabinet total fit.
- Body economy heat cost: `config/physiology/heat_sensitivity.yaml` (cabinet degrees per 100 W per
  host; circe 0.31), read by `orion/physiology/heat_config.py`. Thermal lines owned by
  `thermal_gate.py` (commit 8a1e29367).
- Strain loop (#2800): reads cabinet heat, not the closet.
- `config/prediction/series.v1.yaml` comment calls athena, circe and hecate "the three machines in
  the cabinet". Per the layout, only athena is in the cabinet. The patch fixes the comment.

## The three numbers: gate findings

Data alignment for all three: closet and office come from the same host (pan-1), with the same
30 s cadence and the same clock, so they align to the sample. The cabinet and the machine watts
come from athena (NTP synced). Producer sample age is under 1 s on both pan-1 boards. Everything
below is on 5-minute means (the forecast grain), so clock skew is far below the grain. Gaps: office
temperature missing in 0.6 % of buckets (105 stale samples), closet 0 %; office sound starts
2026-11-07 01:45 UTC and its first 20 readings (11-07 01:00-02:00) are the old flat 37.4 dBA bug,
excluded below (filter > 60 dBA).

### Number 1: heat Orion adds = closet top minus office — PASSES (as a measurement)

1. **Provenance.** `closet_temp_c`: SCD40 on pan-1, firmware `firmware/pan-1-office-nano`, sample
   built in `services/orion-office-env/app/sample.py`, flattened by
   `services/orion-sql-writer/app/worker.py` `_normalize_closet_env_sample_payload`. Office:
   SHT40, same producer, `_normalize` office path. Stale samples carry NULL readings, never 0.
2. **Independence.** It is not a copy of the cabinet temperature: the cabinet minus office gap and
   the closet minus office gap correlate 0.60 in level and 0.31 in 5-minute changes. It is not a
   copy of the room covariate either (gap vs office temperature 0.11). It **is** close to circe's
   watts: 5-minute changes correlate 0.65 (0.78 over 15 min); an hourly fit on circe, athena and AC
   watts explains 54 % of it. So as a *load* signal it is redundant (we read circe's watts
   directly). Its new content is the physical *response* to those watts at circe's exhaust, which no
   other sensor sees (cabinet local slope for circe is zero).
3. **Theory anchor.** Steady-state air heat balance, already the anchor of `heat_balance.py`: air
   passing a heat source warms by P / (m_dot c_p). The office is the intake air; the closet top is
   air that circe (and later hecate) has heated, plus the warm air that rises there. The slope is
   heat per watt; its size depends on air flow. Measured hourly, lags 0..2 h: circe
   **+0.25 °C per 100 W** (day-block bootstrap 95 % +0.16..+0.32); lag 0 carries it (+0.23, then
   +0.03, -0.01); athena -0.15 (athena is in the cabinet, below; not its plume).
4. **Live sanity.** Range -0.29..+3.92 °C, median 1.59, not flat, not saturated, no NULL runs.
   Rest state: Orion's body never switches off, so "calm" means "light load", not zero. At circe's
   lightest tenth the gap is 0.81 °C (median 0.70); at its busiest tenth 1.71 (median 1.84). It
   falls back to the light-load floor within one 5-minute bucket when circe drops. It is not a
   decay artifact: it is a difference of two live readings refreshed every 30 s.
   Two caveats. (a) The two boards are different sensors (SCD40 vs SHT40). The floor of ~0.7 °C
   may hold a fixed sensor offset. Their moisture agrees (vapour pressure closet minus office
   +0.09 hPa, sd 0.15, on 8.3 hPa), so it is the same air with no moisture source, but that check
   cannot see a temperature offset. Use slopes, not the absolute level, until a side-by-side check
   (open question 6). (b) The portable AC ran only on 11-05 and early 11-06 (119 W, 14 % of
   buckets). With it on the gap drops to 0.1-0.8 °C at every circe load. One and a half days is too
   short to trust that coefficient (-1.15 °C per 100 W of AC); keep the AC as a control, do not
   quote its effect.
   hecate: 0 W for the whole week, so no hecate slope can be measured yet. The existing fit refuses
   such a machine on its own (`MIN_SPREAD_W` = 20 W, `MIN_ACTIVE_SHARE` = 10 %), which is the right
   answer today.
5. **Existing mechanism.** `heat_balance.py` already fits exactly this shape (target minus room on
   per-machine watts, HAC errors, refusals) for the cabinet. Reuse it with a second declared target;
   build nothing new. `heat_sensitivity.yaml` is a different quantity (cabinet degrees per 100 W)
   and must not be fed from this fit.
6. **Reversibility.** One series entry, one `HEAT_BALANCE` entry, one source kind. Stored fits go
   to `prediction_effect_estimate` under a different `series_id`, so nothing that reads the cabinet
   fit sees them. Removal is a revert.

Variant rejected: closet top minus **cabinet**. The cabinet reads 1.10 °C above the office on
average (athena's air, not clean inflow), the two boards are on different hosts, and the cabinet
already has its own fit. The office is the clean boundary for both.

**Lead or lag (decides the forecast question).** Cross-correlation of 5-minute and 15-minute
changes over the 7 days:

| pair | peak | at lag 0 | at ±15 min |
|---|---|---|---|
| closet top vs cabinet | lag 0 | r 0.57 (5 min), 0.73 (15 min) | ≤ 0.24 / ≤ 0.49, symmetric |
| closet top vs office | lag 0 | 0.50 / 0.70 | ≤ 0.26 / ≤ 0.53 |
| cabinet vs office | lag 0 | 0.87 / 0.93 | ≤ 0.42 / ≤ 0.64 |
| heat added vs circe watts | lag 0 | 0.65 / 0.78 | ≈ 0 after |
| cabinet vs circe watts | +5 min (circe first) | 0.07 / 0.14 | 0.22 / 0.26 peak |

The closet top does not lead the cabinet. Both follow the office air at the same time. Circe moves
the closet top within 5 minutes and barely moves the cabinet. Forecast test (one-hour cabinet change,
fitted on the first 70 % of the week, tested on the last 30 %, 613 test points; base inputs: cabinet,
office, office trend, athena and circe watts, AC watts): base MAE 0.581 °C; + heat added 0.583;
+ closet top 0.583; + closet CO2 0.579. No gain. **Closet-top heat does not help the cabinet
forecast or the cooling act today.** The heat quote (#3194) stays on the cabinet.

Circe's own intake: circe's board temperature follows the office (1.01 °C per °C, R² 0.90), and
the closet top adds nothing once the office is in the fit (+0.07). So circe breathes office-level
air, not its own trapped exhaust. No recirculation is visible this week. That is why no
"circe is cooking itself" line is proposed.

### Number 2: stuck air = closet top warming / CO2 rising faster than the office — FAILS

- **Warming-rate part: fails independence (step 2).** It is the time change of number 1. A
  derivative of a number already in the design is not new signal. And the data shows no build-up to
  detect at the 5-minute or hourly grain: the gap answers circe in the same bucket, and the lag
  1 h and 2 h coefficients are ~0 (+0.03, -0.01 per 100 W).
- **CO2 part: fails the theory anchor (step 3) and provenance of a reference (step 1).** Servers
  make no CO2, so closet CO2 has no Orion source. The office board has no CO2 sensor: `office_eco2_ppm`
  is the CCS811's estimate from VOCs (425..6,806 "ppm" this week, mean 818), a different measurand
  from the SCD40's real CO2 (400..616 ppm). Closet CO2 runs opposite to the room's warmth
  (r -0.81 with office temperature): highest ~14:00 UTC (07:00 MDT, the house closed overnight),
  lowest in the afternoon, and it rose slowly all week (429 -> 511 ppm daily mean). That is the
  house's people and windows, not Orion's air.
- **The surviving idea, recorded, not wired.** By the same heat balance, if air flow out of the
  closet top drops (door shut, more machines, summer), the same watts give a bigger gap. So stuck
  air would show as the fit's **residual** staying high for hours. This week the daily residual
  was near 0 on 6 days (-0.46..+0.06) and **+0.82 °C on 11-07** (08:00-16:00 UTC: gap 3.1-3.7 °C
  at 1,000-1,140 W, versus ~2 °C on other days at that load). 11-07 is also a day the cabinet
  passed 32 °C (~20:00 UTC). One day, possible door or sensor cause, AC confound on 11-05: not
  evidence. The replay eval (below) reports the daily residual so 30 days can say whether it is a
  real, recurring state. No number, no consumer until then.

### Number 3: Orion's effect on the office — warmth part EXISTS, sound part has NO CONSUMER here

- **Office warmth when busy vs idle: fails the existing-mechanism check (step 5).** The heat-per-watt
  *total* fit already measures it (machine watts -> cabinet with the room free to warm, which is the
  office path), and `heat_sensitivity.yaml` prices it. A second version would drift from the first.
- **Office sound when busy vs idle: real, but dropped from this design.**
  - Provenance: DFRobot SEN0232 on pan-1's office Nano -> `services/orion-office-env/app/sample.py`
    `_sound_reading` -> `office_env_sample.payload_json->measurements->office_sound_dba`.
  - Live: 1,498 good buckets since 11-07; 74.1..81.5 dBA; not flat. It tracks **circe's fan**
    (`fan_pct_max`) at r 0.93 in level and 0.76 in 5-minute changes; circe watts 0.87 / 0.72; athena
    watts 0.16 / -0.05. By circe fan fifth: 26 % fan -> 75.9 dBA, 50 % fan -> 80.2 dBA. With hour of
    day held fixed: +0.63 dB per 100 W of circe. Theory anchor: fan sound power rises with fan
    speed (fan laws, ~50·log10 of the speed ratio), and circe's fans speed up with its load.
  - Rest state: circe never went below 536 W this week, so "Orion idle" sound was never seen. The
    quietest reading is circe's light-load fan, not a quiet office.
  - Independence: as a *busy signal* it is redundant (circe's fan and watts are read directly). As
    *Orion's footprint on Juniper's room* it is a real outcome. But this design is the heat path,
    and no heat consumer needs it. Per the no-keyword-cathedral rule it is not added. It goes to
    Juniper as a question (open question 4). The cheapest real consumer, if wanted, already exists:
    the situational brief's office-sound line could say that most of the office sound follows
    circe's fans.

## Missing questions

Answered here:

- *Does the closet top lead the cabinet?* No (table above). It does not enter the cabinet forecast.
- *Office or cabinet as the boundary?* Office: same host and clock as the closet, clean intake air.
- *Does circe breathe its own exhaust?* Not this week (circe board temperature follows the office).
- *Is a sensor offset hiding in the level?* Unknown; moisture agrees; use slopes until checked.
- *Should this feed body-economy prices?* No. Heat cost is about the cabinet line (32 °C). The
  closet top has no line today.

For Juniper:

1. The closet board: the repo says it is pan-1's second Nano (`closet_env_sample.node = pan-1`),
   not athena. Is the repo right? (It changes nothing in this design, but the docs should agree.)
2. Where is the portable AC (`cabinet_cooling` plug), and why did it stop after 11-06? With it on,
   most of the closet-top heat went away.
3. Was the closet door shut on 11-07? That day the closet top ran 0.8 °C hotter than circe's power
   explains.
4. Do you want Orion to treat its office noise as a cost (for example, background circe work during
   your working hours costs more because it is louder)? Yes would be its own design, with this
   week's numbers as the start.
5. Is there a closet-top temperature you care about for circe and hecate? Without one, the closet
   top informs but never triggers an act.
6. Can you put the two pan-1 boards side by side for an hour once? That measures the sensor offset
   and makes the absolute "heat added" level usable.
7. When hecate comes online, will it sit in the closet top beside circe?

## Proposed schema / API changes

No bus channel, no new table, no schema registry change, no env key. All inside `orion-prediction`.

- `services/orion-prediction/app/series.py`: new source kind `closet_env` with an allowlist
  `CLOSET_ENV_KEYS = ("closet_temp_c",)` (the key becomes a SQL column name, same rule as
  `OFFICE_ENV_KEYS`).
- `services/orion-prediction/app/store.py` `points()`: `closet_env` branch reading
  `closet_env_sample` with `sensor_online AND NOT stale AND closet_temp_c IS NOT NULL`, through the
  same `_read_only` path and timeout as `office_env`.
- `config/prediction/series.v1.yaml`: new series `closet.top_temp_c` (5 min, mean, max_fill 15 min,
  `train_from` 2026-11-05T00:45Z), consumers `[heat-per-watt]`. Role `covariate` if `load_frame`
  accepts a covariate as the heat-balance target (it reads `engine.effective` and
  `latest_complete_bucket`; the patch checks). Otherwise `target`, which adds forecasting and
  scoring of the closet top with no decision consumer; prefer covariate. Fix the "three machines in
  the cabinet" comment.
- `services/orion-prediction/app/heat_balance.py` `HEAT_BALANCE`: second entry
  `"closet.top_temp_c": {"room": "office.temp_c", "watts": ("circe.chassis_watts",
  "hecate.chassis_watts", "athena.chassis_watts"), "controls": ("athena.cabinet_cooling_watts",)}`.
  athena stays in as a control-like regressor (its slope here was -0.15; leaving it out would push
  athena's room effect into circe's slope). The docstring's "cabinet" wording becomes "target".
- Read: `GET /effects/heat-per-watt?series_id=closet.top_temp_c` (no new route). Store:
  `POST` with the same body. Rows land in `prediction_effect_estimate` with
  `series_id = closet.top_temp_c`, so the cabinet consumers (`effects.py` experiment prediction,
  the #3194 quote) never read them.

## Files likely to touch

- `services/orion-prediction/app/series.py`, `app/store.py`, `app/heat_balance.py` (one entry +
  wording), `README.md` (heat per watt section: second target).
- `config/prediction/series.v1.yaml`.
- `services/orion-prediction/tests/` : source-kind allowlist test, store SQL shape test, a
  heat-balance fit test on a synthetic closet frame (circe slope recovered; hecate at 0 W refused
  with the spread reason; stale closet rows never read).
- `services/orion-prediction/evals/run_air_path_eval.py` (new, read-only): over the last N days,
  re-runs this doc's checks and prints one line each: closet-top vs cabinet lead/lag, one-hour
  cabinet forecast MAE with and without closet heat, circe and hecate slopes at the closet top,
  daily fit residual (stuck-air watch), and office dBA vs circe fan. Fixture export beside
  `evals/export_cabinet_fixture.py`.

## Non-goals

- No change to the cabinet forecast inputs, the heat quote (#3194), the cooling act, the 34 °C
  reflex, the strain loop, or any body-economy price.
- No new thermal line. The closet top gets a line only if Juniper names one (question 5).
- No CO2 number. No "stuck air" number. No noise cost.
- No new bus channel or table. The Hub panels stay as they are.
- No use of `office_eco2_ppm` as CO2.

## Acceptance checks

- Unit: `closet_env` source refused for any key outside the allowlist; stale/offline closet rows
  never reach a series; the synthetic fit recovers a known circe slope within its interval; hecate at
  constant 0 W is refused with the spread reason, not fitted as 0.
- Live, after deploy (read-only GET): `GET /effects/heat-per-watt?series_id=closet.top_temp_c&days=7`
  returns a circe slope whose 95 % interval overlaps +0.16..+0.32 °C per 100 W, and hecate under
  `refused` while hecate is off. The cabinet answer (`series_id` default) is byte-for-byte the same
  shape as before.
- Hecate arrival: once hecate runs at least 10 % of hours with 20 W of spread, the same GET returns
  a hecate slope (no code change). That is the "hecate visible as more heat added" proof.
- Eval: `python services/orion-prediction/evals/run_air_path_eval.py --days 30` prints the six lines
  above. Gate for any second patch that feeds the cooling act: closet heat must lower the one-hour
  cabinet MAE in a paired test (`backtest.paired_win_p`, p < 0.05), or the daily residual must lead
  cabinet 32 °C crossings on at least 3 separate days. Neither holds on this week's data.

## Proposal mode

- **What capability changes.** Orion's prediction service gains a measured answer to "how much heat
  do circe and hecate put into the closet top, per 100 W". Today no sensor in Orion's cognition,
  prediction or regulation sees circe's heat at all. Nothing acts on it.
- **Data touched.** Read: `closet_env_sample` (temperature only), `office_env_sample`
  (temperature), `orion_biometrics_cluster` watts, `home_cooling_sample`. Written: only on `POST
  /effects/heat-per-watt`, rows in `prediction_effect_estimate`. No people data, no camera, no chat.
  No privacy boundary is involved.
- **What trace proves it worked.** The live GET above with a circe slope and interval, and stored
  rows `SELECT action, effect, details->>'what' FROM prediction_effect_estimate WHERE series_id =
  'closet.top_temp_c' ORDER BY created_at DESC LIMIT 3`. The eval output file with the six lines.
- **Dangerous failure mode.** A later patch reads the closet-top slope as if it were cabinet heat
  (it is ~18x the cabinet's local circe slope) and prices or sheds circe work against the cabinet's
  32 °C line. That would hold back background work for heat the cabinet never sees. Guards: a
  separate `series_id`; `heat_sensitivity.yaml` and the #3194 quote read only the cabinet target
  (test: the cabinet fit result and stored rows are unchanged by the new entry). Second danger: a
  sensor offset read as "Orion adds 0.7 °C at rest". Guard: the design uses slopes only until
  question 6 is answered.
- **How to disable or roll back.** Revert the series entry and the `HEAT_BALANCE` entry (config
  plus one dict entry). Nothing else depends on them. Stored estimate rows are inert and can stay.

## Recommended next patch

**Patch 1: closet top as a second heat-balance target, plus the air-path eval.** About half a day.
`closet_env` source kind, `closet.top_temp_c` series, one `HEAT_BALANCE` entry, tests, the read-only
eval, README. Deploy is a rebuild of `orion-prediction` only. It moves one real thing: Orion can
measure circe's heat (and hecate's, once it runs) where that heat actually goes.

Then wait. Re-run the eval at 30 days (about 12-05) and the first week hecate runs. Only if it
passes the forecast gate above does a second patch add closet heat to the cabinet forecast or the
heat quote. If Juniper names a closet-top line (question 5) or says yes to a noise cost
(question 4), each of those is its own short design, built on this week's numbers.
