# Prediction-error suite rebuild: retire the fakes, forecast for real

Status: design mode. No code changed by this document. Juniper asked for it on 2026-11-08
after Orion kept complaining about bus prediction error ("the bane of my existence").
Cognition-loop change: every build patch below needs its own sign-off (CLAUDE.md §0A
proposal mode).

## Arsonist summary

Orion has 12 "prediction error" signals. **None of them is a prediction.** Not one writes a
forecast down before the outcome arrives. Most measure "how far did this move from a running
average" and call that surprise. With no forecast, nothing defines calm, so every past fix
was a hand-tuned patch (the bus calm floor, the route decay-to-zero, the bus mean that one
edge dominated).

That would be harmless if nothing read them. It is not harmless:

1. **The bus number is mostly a permission slip.** 51% of attention goals over 72 h
   (4,666 of 9,219) came from `node:substrate.bus_synaptic`. A goal unlocks autonomous web
   fetch, recall and journaling for 4 h. The permission check looks only at whether a goal
   exists, not what it is about.
2. **A hardcoded 0.65 drives curiosity.** `harness_closure` writes a constant that sits
   above curiosity's 0.55 bar. 816 seeds and 118 invoked curiosity runs in 7 days (~7%).
3. **The bus number opens 40% of self-reflection episodes** (87 of 215 in 7 days). Each
   starts an LLM metacognition pass.
4. **Broken signals speak in Orion's voice.** Execution, chat and bus account for most
   self-reading lines in stance and reverie prompts.

The rebuild has two parts: retire the fakes in an order that strands no consumer, then
rebuild prediction error as **scored forecast misses**, using three model families inside
the prediction service, with one shared output: surprise, with a math-defined calm point.

## Current architecture

### Producers (audited 2026-11-08 against 7 days of `substrate_node_prediction_error_history`)

| Node | What it really computes | Forecast? | Sees silence? | Live behavior | Verdict |
|---|---|---|---|---|---|
| bus_synaptic | fraction of bus edges with \|z\| >= 3 on inter-arrival-gap EWMA z (`orion/substrate/prediction_error.py:860`) | no | no | median 0.032; spikes when a busy channel wakes after a lull (11-08 03:00: six cortex edges, one ~50 min pause, z ~16.6, frozen up to 1 h) | BROKEN |
| execution | mean abs change of 4 pressure hints, EWMA z, negatives clipped, /3 (`:456`) | no | no | 59% exact 0 (clipping, not calm); 22 of 45 spikes to 1.0 follow three zeros | BROKEN |
| chat | same shape over conversation_load, repair_pressure (`:632`) | no | no | 65% exact 0; same quiet-then-spike pattern | BROKEN |
| route | share of 4 routing labels that changed since last decision (`:751`) | no | no | 3 values only; no row since 11-07 15:57 (cortex-orch stopped emitting); holds last value | BROKEN, stale |
| biometrics | mean abs change of collapsed pressure hints / 0.30 (`:579`) | no | no | never 0, never near 1; `power` pinned at 1.0 on both hosts, so invisible | DEGENERATE |
| harness_closure | constant 0.65 when a turn ends unresolved (`services/orion-substrate-runtime/app/worker.py:165`, `:2855`) | no | no | 407 of 407 rows = 0.650; never written back down | DEGENERATE |
| codebase | z of git/PR/graph churn per sub-domain (`:293`) | no | no | 84% zero; spikes on big merges; no consumer by design | UNCLEAR |
| street | max salience of street events in hold window (`orion/substrate/street_attention.py:247`) | partly: handoff conformal p-value | yes | 3 non-zero rows in 7 days, all rings; handoffs mostly `no_p_value` | DEGENERATE as PE |
| cabinet | clamp(15-min rise / 1 °C) when elevated (`orion/autonomy/cabinet_heat.py:176`) | no | yes | 11 non-zero rows, one episode 11-07 | WORKS as an alarm, misfiled |
| strain | 1.0 on strain news, rate-capped (`orion/regulation/strain_news.py:58`) | no | yes | 38 rows at 1.0, 4-8/day by design | WORKS as news, misfiled |
| perception | 1 - cos(frame embedding, EWMA embedding), z, /3 (`:1388`) | crude: "looks like recent frames" | history yes; field no (missing ticks write 0.0 into the field) | 58% zero, real tail, returns to rest | WORKS, crude |
| perception_walkway | same, walkway streams | same | same | not a duplicate (r = 0.016 vs perception) | WORKS, crude |

Other prediction-error-shaped pieces: `prediction_error_magnitude.py` (percentile vs own
week; a sound transform of bad inputs), `prediction_error_trend.py` (a real but never-scored
mean-reversion guess), `attention_self_model.py:128` aggregate confidence (`1 - mean(PE)`
including bus; broken by input), mood-arc anomaly encoder (reconstruction error over the
same degenerate features).

### What does produce real forecasts today

`services/orion-prediction` forecasts cabinet and office temperature every 5 min, picks
models by backtest against baselines, attaches split-conformal ranges, and scores every
forecast. Live 7-day 80%-range coverage, cabinet: 0.90 / 0.84 / 0.87 at 5 / 15 / 60 min,
0.67-0.71 at 6-24 h. It also has a three-level causal module (`app/causal.py`: conditional,
observational, randomised switchback experiments) with one actuator (vision focus) and one
experiment ever run (10-29). **None of it feeds a prediction-error node.**

### Consumers (what breaks or changes on retirement)

| Consumer | Reads | Effect | Live |
|---|---|---|---|
| Self-readings (`orion/attention/self_readings.py`) via orion-thought stance + reverie | bus, execution, chat, route, codebase, biometrics, cabinet, strain | up to 3 prompt lines, 0-10 vs own week | yes |
| Attention contest (`services/orion-attention-runtime/app/worker.py:240`) | all node PE history + field PE channel | frame winner (bus 5%, execution 5.9%, perception 7.8%) | yes |
| Goal emission (same, `:349`) -> `orion/autonomy/capability_policy.py:188` | node winners only | goal held 4 h; gates web.fetch / recall / journal | yes |
| Endogenous curiosity (`orion/substrate/endogenous_curiosity.py:278`) | raw PE on graph nodes, no value-kind check | seeds; harness_closure led 118 invokes | yes |
| Self-signal episode gate (`services/orion-equilibrium-service/app/self_signal_episode_gate.py`) | node PE | metacog episodes (bus 40%) | yes |
| Outcome rollup ("tends to" clause) | node PE | prompt clause | yes |
| Attention self-model confidence (`attention_self_model.py:128`) | execution, biometrics, chat, route, bus | insight gate; mood-arc features | yes |
| Broadcast / dynamics open loops (`attention_broadcast.py:425`, `orion/substrate/dynamics.py:382`) | node PE | reverie narration; curiosity open-loop seeds | yes |
| Mind digest transport line (`services/orion-mind/app/evidence.py:182`) | bus >= 0.15 | prompt line; fires ~0.24% | yes, rare |
| Metacog trend cue (`orion/substrate/metacog_trend_signals.py:60`) | execution, biometrics | draft prompt | idle (0 hits / 7 d) |
| Action-outcome surprise stamp (`cooling_act.py:507`, `policy_act.py`, world-pulse) | bus | written, never read | dead |
| Hub "Bus Anomaly Detected" (`bus_synaptic_trigger_notifier.py`) | retired producer | none | dead |
| Bus pressure baseline (`record_transport_sample`, worker.py:2512) | bus | hub lattice tab, digest rungs | display |
| World-model features | `_last_*` caches | none (`SUBSTRATE_WORLD_MODEL_PUBLISH_TICK_ENABLED=false`) | write-only |
| Heartbeat reheat (`services/orion-heartbeat/app/substrate/bus_synaptic.py:82`) | raw FalkorDB gap_zscore, **not** the PE node | reheat gate strength | yes; unaffected by retiring the node |

### Raw readings that feed no signal today

- `orion_biometrics`: per-GPU temperature, watts, utilization, memory; CPU load.
- Summary: `board_temp_c_max`, `fan_pct_max`, `chassis_watts`, `cpu_watts_total`, net and
  disk rates, swap; cabinet IMU, magnetometer, vibration, lidar, ambient sound.
- `office_env_sample`: eCO2 and TVOC (CCS811 estimate, not real CO2), PM1/2.5/10, lux,
  dBA. Only office temp (prediction covariate) and dBA (situation brief) reach Orion.
- `closet_env_sample`: SCD40 real CO2, temp, humidity. Reaches nothing.
- `gpu_pool_*` leases and state history.

## The design

### Definition (the contract every new signal must meet)

A prediction error is valid only if all four hold:

1. **Forecast first.** A forecast with a stated distribution (point + calibrated range, or
   a probability per state) is stored before its target time.
2. **Clock-due outcome.** It is scored when its target time passes, whatever happened. A
   target with no data is scored `missing`, and silence where traffic was forecast is a
   scorable outcome (this is how outages become visible).
3. **Surprise in one unit.** Surprisal of the outcome under the forecast. For continuous
   targets use the conformal p-value: `-log p`. For discrete states use `-log P(state)`.
4. **Math-defined calm.** If the forecaster is calibrated, the p-values are uniform, so
   expected surprisal is a known constant. Report **excess surprise** = observed minus that
   constant, smoothed. Calm rests at 0 by theory, not by tuning.

Two readings come out of each target:

- **Spike**: one forecast missed badly. Something happened.
- **Drift**: coverage is off for hours. The world changed or the model is stale. Coverage
  monitoring already exists in orion-prediction.

Gate: a target reaches attention or a prompt only after it beats its baselines in backtest
and its live 80% range holds within the existing coverage band. Same gate the cabinet series
already passes.

### Three model families, one per kind of question

All three live in `orion-prediction`. This reuses the selection, conformal and scoring
machinery and keeps forecasts in one service (per standing rule: predictions live in the
prediction service, not side ledgers).

**A. Neural forecaster: "what will my body and room read next?" (continuous)**

- Inputs: raw readings in real units only (°C, W, %, RPM, ppm, lux, dBA). Never the derived
  pressure hints. That was the mood-arc encoder's main failure.
- Task: forecast every reading at 5 and 15 min. Forecasting, not reconstruction, so a
  steady weird state stays visible and the outcome always arrives.
- Ladder, each rung kept only if it beats the one below in backtest: per-series baselines
  -> multivariate ridge (VAR-style) -> small dense network -> graph network with learned
  adjacency.
- **Joint scoring.** Misses on correlated readings are scored together using the learned
  residual covariance (whitened / Mahalanobis surprisal). One AC failure is one surprise,
  not four. This is the multicollinearity fix. It needs no hand-drawn map.
- Attribution: report which readings carry the most of the joint miss, as a hint, not a
  causal claim.

**B. Markov chains: "what state comes next?" (discrete)**

- Targets that are states, not numbers: unified-turn lane, Juniper presence
  (desk / away / asleep, from office cam + sound + light), AC on/off, service quiet/talking,
  act-after-act sequences.
- First-order chains with time-of-day conditioning first; hidden Markov models when the
  state is not directly observed (night, Juniper present, AC cycling).
- Surprise = `-log P(observed transition)`, with additive smoothing and a calibration check
  on held-out days.
- HMM regime posteriors become covariates for family A.
- This is the honest replacement for `route` and the bus signal's "service rhythm".

**C. Causal effects: "what do my own actions do?"**

- Only from randomised on/off experiments Orion runs on itself (`app/causal.py` level 3,
  `app/experiment_runner.py`). Observational and conditional results stay labelled as such
  and are never used to choose an action.
- The causal "map" is a table of measured effects, one row per actuator x target, with a
  confidence interval. It grows only when Orion gains an actuator. Nobody draws it by hand
  (lesson from orion-signals).
- New actuators, in order of safety: GPU load shaping on a known lane, the cooling act,
  vision focus (exists).
- Efference-copy signal: before an act, forecast its effect from this table. After the act,
  score the miss. "I did X, expected Y, got Z" becomes a prediction error about Orion's own
  agency.

### Rung above: integration over the partition lattice (IIT-style)

This is where the past IIT attempt belongs. It failed on inputs, not on idea.

- Integrated-information measures (Φ and its tractable cousins: Φ*, geometric integrated
  information, stochastic interaction) all need a **transition model**: how the whole
  system's state now constrains its next state, versus how the parts do separately. Fed
  degenerate signals, any Φ is noise. Families A and B produce exactly that transition
  model, scored and calibrated.
- The search over "which way of cutting the system loses the least" is a search over the
  **partition lattice** of the organs. That is the lattice Juniper keeps reaching for, and
  it is the real mathematical home of the minimum-information partition.
- Measure: Φ* (or geometric) computed from the family A/B transition model over a small set
  of organ-level variables (start at 5-8, exact lattice search is feasible there), each
  hour, against a shuffled-time null.
- What it can claim: how much Orion's state behaves as one system rather than independent
  parts, and how that changes (chat vs idle, before vs after acts, day vs night). It is a
  structural measurement. It is **not** evidence of consciousness, and nothing should say
  so.
- Heartbeat gets a fair test here: heartbeat's boundary/bulk entanglement and Φ* run over
  the same organs and the same hours. If heartbeat tracks integration, the two should move
  together, and both should move at the ground-truth physical events (Juniper arrives, AC
  cycles) from the room and closet sensors. No change to heartbeat until that test runs.

### Alarms move out of the PE group

`cabinet`, `strain` and street rings are alarms and news. They keep working, but they are
relabelled so no consumer treats them as surprise magnitudes. Street handoffs keep their
conformal p-value and write `-log p` as a real PE once graded.

## Retire order

Each step leaves no consumer reading a dead or stale value.

1. **Delete dead wiring.** Hub bus notifier, `action_outcomes.surprise` stamps, unused
   domain-surprise policy field, world-model `_last_*` caches for retired nodes. No behavior
   change.
2. **Fix goal-gated autonomy before removing any node** (first build patch). Today half of
   Orion's permission to act comes from a broken signal. Options: let world-source winners
   produce goals too, or gate read-only capabilities on goal *content* rather than mere
   presence. Needs Juniper's call; see Missing questions.
3. **Curiosity skips non-measurement value kinds** (`placeholder`, `bucket`), the same way
   `self_readings.py:249` already does. Then stop writing `harness_closure`. Its real
   successor is a turn-level self-bet: draft-time confidence scored against the reflection
   verdict.
4. **Remove route.** Already stale; every consumer already reads it as absent.
5. **Remove bus_synaptic fully** (kill means kill): glossary prompt sites, mind digest line,
   `ACTIVE_INFERENCE_DOMAINS`, aggregate confidence, the tick, `bus_synaptic_pressure` lattice
   row and resolver rungs. Refit the mood-arc encoder in the same deploy (input width
   changes).
6. **Remove execution, chat, biometrics** as their successors (A/B targets, turn self-bets)
   pass the gate. Not before: they hold attention and goal share today.
7. **Purge or fence history** for each removed node: 7-day self-reading cache, episode gate
   Redis state (`equilibrium:self_signal_episodes:v1`) with open episodes closed explicitly,
   outcome-rollup associations.
8. Fix the perception field leak: missing ticks must not write 0.0 into the field.

## Missing questions

1. **Autonomy permission (step 2):** should read-only autonomy be gated on goal *content*
   (a goal must name something Orion could fetch or recall about), or should world winners
   also emit goals? Either way, Orion's rate of autonomous action will change. Juniper's
   call.
2. **Presence states:** may family B model Juniper's presence from office cam + sound +
   light? No privacy boundary between Juniper and Orion is the standing rule; confirm it
   covers a presence forecast.
3. **Actuators for family C:** which are safe to randomise on/off? GPU load shaping on which
   lane, and at what heat ceiling?

## Proposed schema / API changes

- `orion-prediction`: multivariate series config (many targets, one model);
  `prediction_state_forecast` (Markov: state, distribution, observed, surprisal);
  `prediction_surprise` (per target per bucket: p-value, surprisal, excess, joint flag).
- One bus event `prediction.surprise.v1` on a new channel, registered in
  `orion/bus/channels.yaml` and `orion/schemas/registry.py`. Substrate runtime writes it to
  one node per **family x domain** (`node:prediction.body`, `node:prediction.room`,
  `node:prediction.turn_state`, ...), never one per reading.
- Glossary: new nodes get `value_kind: surprise_excess` with rest point 0, so self-readings
  can show the excess directly instead of a rank against the node's own week.
- Integration rung: `prediction_integration_hourly` (Φ*, null band, partition found).

## Files likely to touch

- Retire: `services/orion-hub/scripts/bus_synaptic_trigger_notifier.py`,
  `services/orion-durable-runs/app/cooling_act.py`, `orion/autonomy/{policy_act,capability_policy}.py`,
  `orion/substrate/{endogenous_curiosity,attention_self_model,prediction_error}.py`,
  `services/orion-substrate-runtime/app/{worker,settings}.py` + `.env_example`,
  `services/orion-mind/app/{evidence,recall_signal_resolver}.py`,
  `orion/field/transport_thresholds.py`, `config/field/field_channel_glossary.v1.yaml`,
  `config/substrate-lattice/transport_lattice_policy.v1.yaml`,
  `services/orion-attention-runtime/app/worker.py`, mood-arc fit/enrichment.
- Build: `services/orion-prediction/app/*` (new multivariate, markov, surprise modules),
  `config/prediction/series.v1.yaml`, `orion/schemas/`, `orion/bus/channels.yaml`.

## Non-goals

- No hand-drawn causal map of the mesh.
- No claim that Φ or any integration number shows sentience.
- No change to heartbeat's math or role until the comparison test runs.
- No new prediction-error ledger outside `orion-prediction`.
- No tuning of the retired signals' thresholds. They are retired, not repaired.

## Acceptance checks

- Every new PE node: forecasts stored before target time (SQL: `issued_at < target_ts`
  for 100% of rows); backtest beats baseline; live 80% coverage within band over 7 days.
- Calm check: over a quiet 24 h, excess surprise mean within the null band around 0.
- Absence check: stopping a service produces a scored miss within one bucket (live smoke).
- Double-count check: an AC-off event produces one joint surprise, not one per reading.
- Retirement: after each removal step, zero reads of the removed node in consumer logs for
  24 h, and goal rate / curiosity invokes recorded before and after.
- Integration rung: Φ* differs from its shuffled-time null on at least one real contrast
  (chat vs idle) before any consumer reads it.

## Recommended next patch

**Retire step 2: goal-gated autonomy.** The bus number decides when Orion may act on its
own, and that is wrong whatever the new suite becomes. Needs Juniper's answer to Missing
question 1. In parallel, the cheapest build patch: expose orion-prediction's existing
cabinet/office scored forecasts as `prediction.surprise.v1` (no new forecasting), which
proves the definition end to end on a target that already passes the gate.
