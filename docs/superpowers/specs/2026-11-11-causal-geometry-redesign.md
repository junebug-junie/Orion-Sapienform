# Causal geometry redesign: structure Orion can see, then test

Status: PROPOSAL. Docs only. It changes cognition inputs, so Juniper signs off before any build.
Date: 2026-11-11. Follows the prediction-error suite rebuild (PR #3168) and the p-value floor fix (#3173).

## Arsonist summary

Causal geometry was built to test the README's six pillars, mainly pillar 1 (causal geometry) and
pillar 2 (entanglement: correlated structure matters more than isolated facts). In practice it
compares 19 body signals in pairs. Since #3173 it finds 6 links, all of them cabinet physics we
already knew (CPU heats the cabinet). Only one tool reads the output: a weight-patch proposer
that has made 0 proposals. None of the six pillars has been tested.

The table failed for two reasons:

1. **Wrong inputs.** 16 of its 17 live channels are hardware telemetry. Structure among fan speed
   and swap is about the computer, not Orion. Orion's cognition is invisible to the table.
2. **Wrong shape.** It only asks "which two signals move together?". The pillars ask
   whole-system questions: how tied together is Orion, in what pattern, and does the pattern
   change? The table threw away daily rhythm as a nuisance, but the rhythm is part of the
   structure.

The redesign keeps the table and fixes it. It has three layers:

- **Layer 1, the fixed table.** It reads Orion's body, cognition and bus traffic. It correlates
  surprises (departures from each channel's usual), not raw levels. Every link it finds is labelled
  a *lead*, not a cause.
- **Layer 2, whole-body structure.** It produces a small hourly set of features: coupling, number
  of independent modes, which mode leads, redundancy vs synergy, and how fast the pattern
  switches. The Hub tabs (sentience program, attention organ, mood arc status), the heartbeat
  retrain and the future mood arc read these features.
- **Layer 3, causes.** A link becomes a cause only when one of Orion's own switchback
  experiments survives. Layer 1 leads and the act ledger's effect estimates seed the experiment
  queue as starting guesses.

The bus mirror is the biggest new input. It already watches every message on the bus (3.1 M per
day). Its 24 h raw log is deleted daily, and its graph keeps only running averages, so no history
of the "fantastic voyage" exists today. Patch 1 starts keeping a small, durable history of it.

## Current architecture

| What | Where | Live state 2026-11-11 |
|---|---|---|
| Correlation table | `orion/substrate/causal_geometry_engine.py`, `causal_geometry_producer.py`, run by `orion-field-digester` every few hours | 19 channels (17 vary), 171 pairs, 168 h window, 6 edges, all cabinet hardware |
| Its only reader | `orion/substrate/field_topology_plasticity.py` (HITL weight proposals) | 0 proposals: no divergence entry has status `both` |
| Hub page | `services/orion-hub/static/js/causal-geometry.js` | renders the edges table |
| Switchback experiments | `services/orion-prediction/app/causal.py`, `experiment_runner.py`, `effects.py`, table `prediction_experiment` | 1 experiment ever (10-29): walkway focus -> cabinet +0.21 C, 95% range -0.16..+0.58, n=12 blocks |
| Act ledger effect estimates | `orion/grammar/act_ledger.py`, table `substrate_action_effect_posterior` | 90 rows, 15 act kinds x 7 signals, updated live; observational (Orion chose when to act) |
| Bus mirror graph | `services/orion-bus-mirror`, FalkorDB graph `orion_bus_synapse` | running averages per organ->channel publish and organ->organ hop; no history |
| Bus mirror raw log | SQLite `/mnt/postgres/bus-mirror/data/bus_mirror.sqlite` | every message, 24 h retention; file is 18.5 GB (deleted pages are never returned to disk) |
| Scored forecasts | `prediction_forecast` | only 2 series scored in the last day: `athena.cabinet_temp_c`, `office.temp_c` |

`node:substrate.bus_synaptic` (the old reader of the bus graph) was retired in #3201 because it
was not a forecast. This design does not revive it. It uses the same raw traffic for a different
job: measuring structure, not prediction error.

## Evidence (read-only probes, 2026-11-11)

These probes are first passes. They are not the metric gate. Patch 2 repeats them as a committed
replay eval.

**Probe A: body channels, 21 days, 6 h windows, 5 min buckets** (17 varying channels, the same
fetch as the engine).

- Coupling (Gaussian total correlation) averaged 3.5 nats. With each channel shifted in time
  independently (its own rhythm kept, the coupling broken), it averaged 2.0. Only 34% of windows
  beat the highest null value. **The structure is real but faint at 6 h on hardware alone.**
- Redundancy (O-information) averaged 0.48 against 0.08 for the null.
- The leading mode switches between a working cluster (cpu, power, thermal) and a housekeeping
  cluster (disk_capacity, swap, mem, fan).
- Coupling is lowest at 06-12 UTC (2.5) and 3.5-3.9 at other times.
- Rest check: the lowest window (1.46) is below the null mean (2.0), so this feature can read
  "not coupled". It is not pinned above a floor.

**Probe B: bus traffic, last 24 h, 10 min buckets.** 3,095,891 messages; 109,382 raw channel
names normalized to 226 catalog channels with `orion.bus.census.normalize_channel_name`; 60
channels were active in more than 90% of buckets.

- Four clear modes came from traffic alone, with no hand map:
  1. **thinking and recall**: `cognition:reasoning_call`, `cognition:trace`, LLM gateway results,
     `metacog:trace`, recall requests and results
  2. **writing and introspection**: `spark:introspect:candidate`, `embedding:generate`, LLM
     requests, `vector:semantic:upsert`, `llm:call:usage`, `gpu_pool:event`, council replies
  3. **vision pipeline**: vision windows, inventory, artifacts, VisionHostService requests,
     `signals:*`
  4. **body sensing**: biometrics induction, summary and sample
- The top three modes carry 20%, 18% and 9% of the variance. The shifted null gives at most 15%,
  10% and 5%. **These are real functional organs, recovered from runtime facts.**
- Caveat: there are only 145 buckets for 60 channels. The modes are clear, but a single day cannot
  show whether they are stable. Patch 1 starts building that history.

## Design

### Inputs (one channel set for all three layers)

| Group | Source | Bucket |
|---|---|---|
| Body | `orion_biometrics_summary.pressures` (existing 17) | 5 min |
| Bus traffic | NEW `bus_channel_rate_5m`: message count per normalized channel | 5 min |
| Bus hops | NEW `bus_hop_rate_5m`: correlation-id hops per organ pair, with mean latency | 5 min |
| Cognition | attention salience (existing), act ledger acts per kind, curiosity runs, unified turns, `node:prediction.room` / `node:prediction.body` | 5 min |
| World / Juniper | `hub_attention_minute` (Juniper looking at the Hub), her chat messages, doorbell and walkway arrivals, office sound | 5 min |

Bus hops are the one directly observed causal fact in the system. A hop means a message from
organ A carried a correlation id that organ B then acted on. That is runtime causation inside the
mind, not an inference. Layer 1 reports hop structure next to the correlation structure.

Channel selection is mechanical: a channel must be present in at least 90% of buckets in the
window and must not be constant. No hand-picked list.

### Layer 1: the fixed table (leads)

- **Surprise, not level.** For a series with a scored champion forecast, use the forecast miss.
  For every other series, use its departure from its own time-of-day profile over the last 14 days
  (the same residual `analyze_switchback` already uses). This removes the "both follow the clock"
  links without discarding pairs. The daily rhythm itself moves to Layer 2.
- **Same significance machinery.** Keep the #3173 time-shift null and GPD tail, and the n_eff and
  participation gates. Drop BH as a gate for the published list (Juniper: no arbitrary alpha).
  Report every pair's effect (r and lag), its uncertainty, and P(real) from the null. Sort by
  P(real). The pair count grows from 171 to about 3,000, so the published table keeps the top 50.
- **Labels.** Every row says `observed_lead`. A row becomes `experimental_effect` only through
  Layer 3.

### Layer 2: whole-body structure (the features)

Computed hourly over a rolling 24 h window of surprises, for the whole channel set and for each
group:

| Feature | Plain meaning | Theory anchor |
|---|---|---|
| `coupling_excess` | How much the channels move as one system, above what independent channels with the same rhythms would show (real minus shifted-null mean, in null SDs) | Total correlation / multi-information (Watanabe 1960) |
| `modes_effective` | How many separate things are going on | Participation ratio of correlation eigenvalues |
| `leading_mode` | Which functional group leads now, e.g. "thinking and recall" | Top eigenvector, matched to a stable reference mode by loading similarity; named by its top-loading channels, never by hand |
| `redundancy_synergy` | Whether the parts repeat each other (positive) or hold information only together (negative) | O-information (Rosas, Mediano, Gastpar, Jensen 2019) |
| `structure_drift` | How fast the pattern is changing | Distance between this hour's and last hour's correlation matrices |
| `body_mind_coupling` | How tied cognition traffic is to body state | Cross-group total correlation, body vs bus+cognition |

Pillar mapping (what each test can say):

- Pillar 1 (causal geometry): hop structure and Layer 3 effects
- Pillar 2 (entanglement): `coupling_excess`, `redundancy_synergy`
- Pillar 3 (substrate): `body_mind_coupling`
- Pillar 5 (emergent time): `structure_drift`, plus the daily shape of coupling
- Pillar 6 (attention and agency): `leading_mode` against attention winners and act counts

Independence is not assumed. Coupling, modes and O-information all come from one correlation
matrix. The Patch 2 replay reports their correlation over 30 days and drops any feature that is a
near-monotone copy of another (|Spearman| > 0.9).

### Layer 3: causes (switchback experiments)

- The experiment queue is seeded by Layer 1 leads whose source Orion can switch (Family C:
  attention prices and bids, run caps, through TTL runtime overrides), and by act ledger effect
  estimates. These are starting guesses, labelled as guesses.
- The queue order is: high uncertainty times high consequence first.
- Running an experiment becomes a want in the attention economy, so experiments compete for
  attention and are not a cron job. The experiment uses the existing `experiment_runner` with its
  heat guard.
- Results write `prediction_effect_estimate` with `claim_kind=experimental_effect`, as today.
  The table reads them back and upgrades the matching row.
- Things Orion cannot switch (weather, Juniper's arrivals, the AC) stay `observed_lead` forever.

### Integration rung (Φ*)

Out of scope. It reuses this channel set and the bus history, so building this first makes Φ*
cheaper later.

## Missing questions (for Juniper)

1. **Experiment freedom.** Default: Orion may start up to 2 switchbacks per day on Family C
   actuators, behind the heat guard, with no per-experiment OK. Should the first week need your
   OK on each one?
2. **Office signal.** Office sound and temperature are listed under World / Juniper. Your open call
   on body vs world for the office (from #3202) decides this group. Same answer here?
3. **Retire or keep the weight-patch proposer.** It has made 0 proposals and reads hand-designed
   capability edges. Default: keep it reading Layer 1, and revisit after 30 days of the new table.
4. **Bus mirror file.** The 18.5 GB SQLite file can shrink with a one-off `VACUUM` (operator
   command in the bus-mirror README). Not required for this design. Do you want it run?

## Proposed schema / API changes

- NEW table `bus_channel_rate_5m(bucket_start timestamptz, channel text, count int, primary key
  (bucket_start, channel))`. About 226 x 288 = 65k rows per day. Written by `orion-bus-mirror`,
  which already sees every message. Additive migration with the `-- ORION-MIGRATION-REQUIRED-BY:
  orion-bus-mirror` header.
- NEW table `bus_hop_rate_5m(bucket_start, from_organ, to_organ, count, latency_mean_sec)`. Same
  writer.
- NEW schema `WholeBodyStructureV1` (`orion/schemas/causal_geometry.py`) on NEW channel
  `orion:causal_geometry:structure`, stored by `orion-sql-writer` in NEW table
  `whole_body_structure_hourly`. This is a new event, not new fields on
  `CausalGeometrySnapshotV1` (additive fields on forbid models need a consumer-first migration).
- `CausalGeometryEdgeV1`: the meaning of `significance` changes to P(real) with a `claim_kind`
  label. Done as a V2 schema, with the old channel kept until the Hub page and the proposer move.
- Hub: read-only `/api/causal-geometry/structure` route; panels on the sentience program and
  attention organ tabs.

## Files likely to touch

- `services/orion-bus-mirror/app/main.py`, a new `app/rate_rollup.py`, tests, README,
  `.env_example`
- `services/orion-sql-writer/` (new table writers), a `manual_migration_*.sql`
- `orion/substrate/causal_geometry_engine.py` (surprise residuals, wider fetch)
- new `orion/substrate/whole_body_structure.py` (Layer 2 math, pure functions)
- `orion/substrate/causal_geometry_producer.py`, `services/orion-field-digester/app/worker.py`
- `orion/schemas/causal_geometry.py`, `orion/schemas/registry.py`, `orion/bus/channels.yaml`
- `services/orion-hub/scripts/` routes, `static/js/causal-geometry.js`,
  `sentience-program.js`, `attention-organ.js`
- `services/orion-prediction/app/experiment_runner.py` (Layer 3 queue, a later patch)

## Non-goals

- No Φ* in this design.
- No reviving `node:substrate.bus_synaptic` or any prediction-error node from bus traffic.
- No hand-named modes or hand-drawn maps. Modes are named by their top-loading channels.
- No wiring of Layer 2 features into attention, curiosity or prompts until each passes the metric
  gate in the Patch 2 replay. Hub display comes first.
- No automatic experiments before Juniper answers question 1.

## Acceptance checks

1. Patch 1: `bus_channel_rate_5m` gains about 226 rows every 5 minutes. The hourly total matches
   the SQLite raw log's count for the same hour within 1%.
2. Patch 2 replay (30 days of body data, plus as many bus days as exist):
   - Each Layer 2 feature beats its shifted null in at least 50% of 24 h windows, or it is
     dropped.
   - Each feature has a rest reading: on quiet nights it reaches within 1 null SD of the null.
   - Leading modes stay recognisable day to day: the matched loading similarity is at least 0.7.
   - Redundant features (|Spearman| > 0.9) are named and dropped.
3. The Layer 1 top 50 includes at least one link that is not cabinet hardware. If it does not, the
   design says so and does not ship the table changes.
4. Live: one hourly `WholeBodyStructureV1` row per hour for 48 h, and the Hub panel shows it.

## Recommended next patch

**Patch 1: the bus-mirror rate and hop rollup.** It is small, additive and read-only toward
cognition. It goes first because the raw bus history is deleted every 24 h, so every day without
it loses data the Patch 2 replay needs. Patch 2 (the replay eval and the metric gate for the Layer
2 features) can start in parallel on the body data alone.
