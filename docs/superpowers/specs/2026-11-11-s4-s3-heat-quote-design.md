# S4 -> S3: the heat quote (a forecast that changes an act, and an act that settles the forecast)

Status: **design, proposal mode** (autonomy change). No implementation until Juniper says go.
Brief: `docs/superpowers/handoffs/2026-11-11-handoff-self-model-ladder-and-s4-s3-design.md` (PR #3191).
Date: 2026-11-11. Live numbers below were pulled 2026-11-11 ~04:40 UTC from the live database and
`orion-prediction` (port 8132).

## For Juniper (read this, skip the rest if you like)

**What it does.** Before Orion's cooling act lets background GPU work run, it asks the forecast one
question: "if I let this work run now, how likely is the cabinet to pass 32 °C in the next hour?"
The forecast answers with two numbers: with the work, and without it. If the "with" number is high,
the cooling act holds the background work back *before* the cabinet heats, instead of after (today
it only acts once the cabinet is already warm and rising). An hour later the forecast grades both
numbers against what the cabinet really did. The cooling act also tells the forecast every time it
held work back, so the forecast stops learning from a cabinet that Orion was cooling on purpose.

That is the two-way argument the portrait says is missing: the forecast changes what control does,
and what control did changes the forecast.

**One recommendation:** build it in three small patches. The first patch is a replay test on the
last 30 days. If the replay shows the "with the work" question does not warn earlier than today's
forecast, stop there and switch to the demand loop (below). Do not build the act on a forecast that
cannot see heat coming.

**Honest risk, up front.** The live numbers make this a real bet, not a sure thing:

- Today's 1-hour cabinet forecast almost never sees a rise coming. From a cool start (below 31 °C),
  the cabinet reached 32 °C within the hour 46 times since 10-20. The forecast's upper line warned on
  7 of those 46, and warned falsely 87 times. It cannot know that work is about to start. Telling it
  the plan is the fix this design bets on.
- One background act on circe is small: its median measured draw is 19-81 W above rest (7 days,
  `grammar_acts.cost_json.wh_by_host`). Circe's measured heat effect on the cabinet is +0.33 °C per
  100 W held 2 h (95 % range +0.20..+0.45, 21 days). So one act moves the cabinet about 0.1-0.3 °C.
  The loop only matters when several acts stack. The replay test measures whether they do.
- It is November. The cabinet passed 32 °C on 2 days in the last 2 weeks (10-29, 11-07). Holds
  will be rare until spring. The *grading* still happens every hour, all winter, because every
  background admission gets a quote whether or not anything is held back.

**Your yes/no decisions (4):**

1. **Heat loop first?** Yes = build this. No = build the demand loop instead (pre-warm or release
   GPU lanes from a forecast of chat and curiosity load). Recommend: **yes**.
2. **Let it act on its own once it passes?** After the replay test passes and 14 days of
   "would hold" records look sane, the forecast-triggered hold binds through the same market as the
   cooling act today, with no second ask. Recommend: **yes**.
3. **Keep the coin flip?** Half of the forecast warnings get no hold (the same 50 % control group the
   cooling act already uses). That is how Orion learns whether holding work back actually cools the
   cabinet. Recommend: **yes**.
4. **Let the portrait judge the loop by its trace, not its wiring?** The portrait's gap line today
   clears on any channel between two services. Change it to clear only when the trace shows real
   quote -> decision -> graded rows in the last 7 days. Recommend: **yes** (otherwise a plain HTTP
   call clears it, which is the thing the brief says not to do).

**Kill switch:** one env flag in durable-runs turns the forecast trigger off; the cooling act goes
back to acting on warm-and-rising only. A second flag in orion-prediction turns quotes off. The loop
also demotes itself to "would hold" (shadow) on its own if its grades go bad (see Failure modes).

---

## Arsonist summary

The gap is real and the obvious fix is fake. A channel from orion-world-pulse to
orion-gpu-lane-controller would clear the portrait line and change nothing: world-pulse reads news,
and the lane controller swaps models. Neither has anything the other could use.

The parts that *can* argue already exist and already sit next to each other without talking:

- `orion-prediction` forecasts the cabinet every 5 minutes and computes the chance of crossing
  32 °C and 34 °C in the next 30 minutes (`prediction_exceedance`, 6,101 rows since 10-20).
- `orion-hardware-watch` reads that forecast but is observe-only by design ("it never sheds, and
  never stops a shed", `services/orion-hardware-watch/app/watcher.py` `note_forecast`).
- The cooling act (`services/orion-durable-runs/app/cooling_act.py`) holds background GPU work
  through the pool's shed lever, through the market, with a 50 % control group. It triggers only on
  `shed_act.is_warming`: already warm *and* already rising. It never reads the forecast.
- `orion-prediction` already has the pieces for a "what if" answer: heat per watt
  (`app/heat_balance.py`, `/effects/heat-per-watt`), interventions (`/interventions`,
  `prediction_intervention`), and self-scoring of exceedance probabilities.
- No control part ever tells the forecast what it did. The cabinet forecast trains on hours where
  Orion was shedding on purpose and does not know it.

So the smallest real loop is not a new service or a new channel. It is one question control asks
the forecast before it acts, one record control sends back after it acts, and one grade the forecast
writes when the hour is over. Everything else is reuse.

Where it can fail: the replay may show that background work is too small a heat source for the
"with the work" quote to beat today's forecast. That is why Patch 0 is a go/no-go test, not a
formality.

## Current architecture

- **S4 (ahead):** `orion-prediction`. Champion cabinet forecast per horizon (5/15/60/360/1440 min).
  Live 8-day mean error at 60 min: 0.60 °C (baseline 0.86); at 6 h: 2.39 °C (baseline 2.73). Useful
  lead is about 1 hour, not 6. Exceedance windows are 30 minutes. `POST /scenario` exists but is
  correlational (it says so: more AC power goes with a hotter cabinet).
- **S3 (control now):** the cooling act in `orion-durable-runs` (tick 60 s) -> market bid
  (`self_regulate.shed`, tier 1) -> `orion-gpu-pool` shed RPC -> background grants withheld
  (`gpu_pool_orion_shed`). Last sheds: 10-15..10-25 (8 rows, all `expired`). The 34 °C reflex in
  `orion-hardware-watch` -> pool is the hard floor and is not touched by this design.
- **Background GPU load:** 397 background `agent` leases in 3 days, mean hold 19 min, ~43
  lease-hours per day (`gpu_pool_events`). Measured energy per act is on `grammar_acts.cost_json`.
- **Heat per watt (21 d, total fit, room free to warm):** athena +0.80 °C/100 W (+0.32..+1.28),
  circe +0.33 (+0.20..+0.45). The 7-day local fit currently **refuses** (R² 0.06 < 0.10). The quote
  must handle a refusal.
- **Portrait gap rule:** `orion/body_atlas/capacities.py` `s4_s3_links` clears `gap:s4-to-s3` on
  any non-census channel, HTTP call or shared Redis key each way between a confirmed S4 and S3 part.
  Claimed parts today: S4 = orion-world-pulse, S3 = orion-gpu-lane-controller.

## Missing questions (answered here, so Juniper does not have to)

- *Which forecast?* Not a new model. The champion cabinet forecast plus the heat-per-watt slopes,
  conditioned on a plan control states. Standing rule: forecasts live in `orion-prediction`.
- *Which threshold?* 32 °C (`thermal_gate.DEFAULT_HOT_C`), the band Orion's learned shed owns. 34 °C
  stays with the reflex; this design never touches it.
- *Which act?* The existing cooling act's shed. No new actuator, no new tier, no new family.
- *Who settles?* `orion-prediction` grades its own quotes from the cabinet series, the same way it
  scores exceedance today. Control does not self-grade.
- *Does a shed spoil the grade?* No, because each quote carries two scenarios (with the work,
  without it). Whichever one really happened is the one that counts for calibration; the
  intervention record says which. The coin flip makes the with/without difference a fair estimate
  of what holding work back does.
- *Privacy?* None touched. Machine telemetry, room temperature and GPU lease counts only. No people
  data, no camera, no chat content.

## Proposed schema / API changes

**orion-prediction (HTTP only, like action bets; no bus channel):**

```text
POST /heat-quotes
  body:  { series_id: "athena.cabinet_temp_c", threshold_name: "experiment_guard",
           window_min: 60,
           plan: [ { host: "circe", watts: 140.0, minutes: 60 } ],   # control's stated plan
           requested_by: "orion-durable-runs.cooling_act", correlation_id }
  reply: { quote_id, issued_at, forecast_issued_at, model,
           p_with, p_without,                    # chance of >= threshold within window
           delta_c_mean, delta_c_lo, delta_c_hi, # the plan's added degrees, from heat per watt
           slope_fit_id | null, refused: null | "<reason>" }
GET  /heat-quotes/live    # grades: Brier with/without vs exceedance baseline, coverage, counts
```

- `p_without`: the current champion forecast's exceedance over the window (same paths the
  exceedance code already draws, extended from 30 to 60 min).
- `p_with`: the same paths shifted by the plan's added degrees, drawn per path from the slope's
  interval (lag coefficients applied over the window, `heat_balance` total fit). Theory: steady-state
  air heat balance, already anchored in `heat_balance.py`.
- Slope fit refused or older than 24 h -> `refused` set, `p_with = null`. Control must treat a
  refusal as "no forecast trigger", never as "safe".

**New table** `prediction_heat_quote` (migration with
`-- ORION-MIGRATION-REQUIRED-BY: orion-prediction`, additive DDL only): quote fields above + plan
JSON + `settled_at`, `max_actual_c`, `hit`, `held` (bool, from the intervention), `n_readings`.
Settled by the existing score step once `issued_at + window` has passed; no readings -> `missing`.

**orion-durable-runs cooling act:**

- New trigger beside `is_warming`: when no want is open, the cabinet is not warming, and background
  work is granted or queued, ask for a quote -- at most one per new forecast (every 5 min), not one
  per 60 s tick. Plan watts = (granted + queued background leases on
  each host) x that host's median measured background-act watts over the last 7 days
  (`grammar_acts.cost_json.wh_by_host` / duration).
- `p_with >= ORION_COOLING_ACT_FORECAST_P` (default 0.5) and `p_without` below it -> post the same
  `self_regulate.shed` want, `source = "cabinet_forecast"`, `event_ref = "heat_quote:<quote_id>"`.
  Same market, same 50 % draw, same shed RPC, same settlement, same episode row.
- Every drawn arm (treated *and* control) -> `POST /interventions` with
  `action = "cooling_act.shed"`, `arm = on|off`, the episode id in `notes`. Reactive sheds get the
  same record. This is the control -> forecast link.
- Shadow mode (`ORION_COOLING_ACT_FORECAST_MODE=shadow|bind`): in shadow the quote is asked and
  recorded, the want is not posted, and a `would_hold` line is logged with the quote id.

**orion-prediction cabinet forecast:** intervention windows with `action = cooling_act.shed` and
`arm = on` are masked from training like the existing exclusions, so the forecast stops learning
"normal" from deliberately cooled hours.

**Portrait (decision 4):** `gap:s4-to-s3` clears when, in the last 7 days, at least
`S4_S3_MIN_SETTLED` (5) quotes were asked by a confirmed S3 part, settled by a confirmed S4 part,
and at least one fed a market decision. Wiring alone prints "linked, not yet used". Orion re-claims
S4 = orion-prediction and S3 = orion-durable-runs (cooling act); the existing code check decides.

## Metric gate (the quote, `p_with`)

1. **Provenance.** Champion paths: `services/orion-prediction/app/engine.py` (issue step, exceedance).
   Slopes: `app/heat_balance.py` total fit. Plan watts: `orion/energy/act_energy.py` via
   `grammar_acts.cost_json`. All traced to producing code, not schema comments.
2. **Independence.** `p_with` is not new signal about the cabinet; it is the existing forecast plus
   one new input, the plan. The plan is the only new information, and it is the point of the
   design. `p_without` is the existing exceedance on a longer window. Do not wire either into any
   other model as an extra feature.
3. **Theory anchor.** Air heat balance: exhaust warms by P/(m·c_p); held power raises the cabinet
   by the measured slope over a 0-2 h response. Named, already in the codebase, already fitted.
4. **Live data.** Rest state: the 32 °C exceedance reads near zero most of the time (4,624 of 6,095
   scored rows in the lowest decile, mean p 0.001, hit rate 0.004), so it can return to calm.
   Calibration of `p_without`'s parent is honest at the ends and loose in the middle (p ~0.33 -> hit
   0.57; p ~0.44 -> 0.71). Lead time from a cool start is weak (7 of 46 caught, 87 false). Slope:
   significant at 21 days, refused at 7 days. **UNVERIFIED:** whether adding the plan catches more
   rises. Patch 0 answers it before anything binds.
5. **Existing mechanisms.** Reused: exceedance, heat per watt, interventions, cooling act, market,
   holdback, settlement, act energy. Not reused: `/scenario` (correlational, wrong anchor) and
   action bets (binary success of an act, not a physical quantity).
6. **Reversibility.** One table, two flags, one trigger branch. Removing it drops nothing else.

## Failure modes (proposal mode)

| danger | guard |
|---|---|
| A biased-high quote holds background work off forever | Holds reuse the shed's 900 s TTL; at most `ORION_COOLING_ACT_FORECAST_MAX_HOLDS_PER_DAY` (default 6) forecast holds a day; 50 % control group always runs the work. |
| Grades go bad and it keeps acting | Auto-demote to shadow when the 7-day Brier of `p_with` is not better than the exceedance baseline (paired test, the existing `backtest.paired_win_p`), or 80 % coverage of the added degrees falls below 0.6. Log line + `/health` alert. |
| It starves Juniper's chat | The shed lever only withholds `background` priority grants (`TARGET_ID = pool:background_gpu`). Test: a forecast hold never blocks `interactive` or `system` grants. |
| Slope refused, quote guesses | Refusal -> no trigger. Test with the live refusal shape (R² 0.06). |
| Stale forecast | Quote refuses when the champion forecast is older than 10 min. |
| Self-fulfilling grade (holds make the forecast look wrong) | Two scenarios per quote; calibration uses the one that happened; treated windows masked from forecast training. |
| It and the reactive trigger double-fire | One open want per event window, as today; the forecast trigger only runs when no want is open and the cabinet is not already warming. |

Rollback: `ORION_COOLING_ACT_FORECAST_MODE=off` (durable-runs) and `PREDICTION_HEAT_QUOTE_ENABLED=false`
(prediction). Flags ship ON per standing rule; the act ships in `shadow` until the gates below pass,
then the code switches itself to `bind` only if Juniper said yes to decision 2.

## Files likely to touch

- `services/orion-prediction/app/heat_quote.py` (new: quote + settle), `app/main.py`, `app/store.py`,
  `app/engine.py` (60-min window, intervention mask), `settings.py`, `.env_example`, README, tests,
  `evals/run_heat_quote_replay_eval.py` (new).
- `services/orion-sql-db/manual_migration_prediction_heat_quote.sql` (new).
- `orion/schemas/prediction.py` (`HeatQuoteRequestV1`, `HeatQuoteV1`), `orion/schemas/registry.py`.
- `orion/autonomy/shed_act.py` (forecast trigger rule, pure), `services/orion-durable-runs/app/cooling_act.py`,
  its `settings.py`, `.env_example`, tests.
- `orion/body_atlas/capacities.py` (decision 4 only).

## Non-goals

- No new service, no bus channel, no change to the 34 °C reflex or to hardware-watch.
- No control of the AC (the Shelly plug can switch; that is a bigger autonomy change, a later ask).
- No scheduling of durable runs hours ahead: the 6 h forecast error (2.4 °C) is too wide.
- No world-pulse involvement.
- No body-economy price change; the hold goes through the existing family and tier.

## Acceptance checks

- **Patch 0 (replay, go/no-go):** over the last 30 days, rebuild a quote at every background grant
  from the stored forecasts, the 21-day slope and the measured act watts. Pass = the `p_with`
  trigger catches at least twice as many of the cool-start 32 °C rises as today's forecast (7 of
  46) at no more false alarms per true catch. Fail = stop; recommend the demand loop.
- **Live, Patch 1 (quotes + settle, no act):** at least 100 quotes a day settled; `p_without`
  matches the stored exceedance within rounding on the shared 30-min window; zero quotes on a
  refused slope.
- **Live, Patch 2 (shadow 14 days, then bind):** every forecast hold has a trace row:
  `prediction_heat_quote.quote_id` -> want `event_ref` -> episode arm -> `prediction_intervention`
  row -> quote `settled_at`. One SQL join returns the whole chain. Interactive grants unchanged
  during holds (`gpu_pool_events`). Auto-demote tested by forcing a bad slope.
- **Portrait:** `python -P -m orion.dig portrait` stops printing the S4 gap line only after the trace
  rows exist, not after the wiring does.

## Recommended next patch

**Patch 0: the replay eval**, `services/orion-prediction/evals/run_heat_quote_replay_eval.py`,
read-only, no service change, about a day of work. It answers the one question the whole design
rests on (does telling the forecast the plan make it see heat coming?) before any act changes.
Then Patch 1 (quote + settle + intervention records, live, no act), then Patch 2 (trigger in shadow,
binding on Juniper's decision 2).

### Fallback if Patch 0 fails: the demand loop

Forecast chat, curiosity and field-note load for the next hour (a count series in `orion-prediction`)
-> `orion-gpu-lane-controller` keeps a lane warm or releases it -> settled by real wait times and cold
spawns (a cold claude spawn took ~58 s on 11-04). Daily events all year, but the lane controller has
no market seat today, so it is a larger patch. Not designed further here.

## Notes

- Querying `POST /effects/heat-per-watt` for this spec wrote estimate rows to
  `prediction_effect_estimate` (the endpoint stores every fit it runs). Harmless; noted for honesty.
