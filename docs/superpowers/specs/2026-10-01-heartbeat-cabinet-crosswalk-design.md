# Heartbeat x cabinet temperature crosswalk — design (no runtime code)

Status: design only. Depends on PR #2451 (time-window organ occupancy,
`fix/heartbeat-time-window-dark-seats`) and on the attend→act action from
`2026-09-29-attend-to-act-loop-design.md` actually shipping.

## Arsonist summary

Orion's heartbeat (a small quantum-style tensor model that tracks which
organs are talking and how tangled their signals are) has never been asked
whether it says anything about the real world. The one real-world number we
care about right now is the server-cabinet temperature. Question: does the
heartbeat's picture move in a way that lines up with the cabinet cooling or
heating, specifically around the moments Orion acts on a hot cabinet?

Today the answer is "it can't, structurally": heartbeat never sees a cabinet
temperature. This doc proposes a **logging-only experiment**: snapshot
heartbeat `/h1` just before and after each attend→act trigger and store it
next to the real temperature outcome. A null result is an acceptable result.
Heartbeat publishes nothing and gets no new channel.

It fails the independence and theory-anchor parts of the metric gate today
(findings below). That is why this is an experiment to *measure*, not a
signal to wire into anything.

## Current architecture

Verified in code on `main` @ f44dc1127:

- Heartbeat absorbs only `GrammarEventV1` atoms from `orion:grammar:event`.
  Per atom it uses `source_service` (picks one of five organ sites,
  `ORGAN_SITE_MAP` in `services/orion-heartbeat/app/substrate/routing.py`),
  `atom_type` (picks the operator), and `confidence/salience/uncertainty`
  (strength). It reads no payload, no temperature, no free text.
- Cabinet temperature is `cabinet_temp_c` (`orion/telemetry/cabinet_sensors.py:38,109`).
  `compute_cabinet_pressures` folds it, with humidity, pressure and gas
  resistance, into ONE baseline-relative 0–1 number via `max(...)`:
  `cabinet_climate_activity` (`cabinet_sensors.py:177-197`).
- `orion-biometrics` turns that into the grammar atom
  `cabinet_climate_activity_signal` (`services/orion-biometrics/app/grammar_emit.py:31-32`,
  `atom_type="signal"`, `salience`=activity). `source_service="orion-biometrics"`
  (`grammar_emit.py:182`), so heartbeat DOES absorb it — as an anonymous
  "signal" on site 1 (biometrics), indistinguishable from the dozens of other
  biometrics atoms (~11k events/h). A temperature value, or even "this atom is
  the cabinet one", is not recoverable from what heartbeat keeps.
- `cabinet.ambient.spike.v1` (`services/orion-biometrics/app/main.py:~348-361`)
  is **audio** (`AMBIENT_AUDIO_SPIKE_ENABLED`, `CabinetAmbientSpikeV1`), not
  temperature. Its grammar atom (`cabinet_ambient_audio_spike_signal`) is built
  by `orion/substrate/cabinet_ambient_spike_consumer.py:45` with
  `source_service="orion-substrate-runtime"` (line 24), which is NOT in
  heartbeat's five-organ allowlist, so heartbeat skips it
  (`events_skipped_organ`).
- Real temperature outcome: `orion_biometrics_summary.measurements->>'cabinet_temp_c'`
  (node athena, ~30 s cadence; documented in the attend→act amendment).
- Heartbeat has one other real-world input: the reheat driver reads
  FalkorDB bus-synapse `raw mean(|gap_zscore|)`
  (`app/substrate/bus_synaptic.py`, `app/service.py:340-352`), which is bus
  timing, not temperature.
- `/h1` (after #2451) reports `dark_seats`, `organ_fire_counts`,
  `organ_distinctness`, `smear/smeared`, `verdict`, `mean_ratio`,
  `std_ratio`, plus per-organ `organ_seconds_since_last_fire` over a
  wall-clock window (default 300 s).
- The attend→act design defines the trigger (elevated AND rising cabinet,
  AC healthy, reflex idle), a precommitted outcome (`substrate_action_outcomes`
  with `predicted_delta`/`observed_delta`), a correlation chain keyed on
  `open_loop_id`/`broadcast_log_id`, and a randomized holdback control arm
  (amended: per-template `holdback_fraction: 0.5`, draw only among eligible
  decisions, `overlap:reflex` rows excluded in both arms).

## Missing questions

1. Does the attend→act action ship, and when? Without treated/control events
   there is nothing to align to. (Also depends on hardware-watch reflex
   shipping first, per the amendment.)
2. Reheat-driver floor (open item, see below): is the heartbeat's baseline
   reheat a real signal or a permanent offset?
3. How many eligible episodes exist? The amendment measures ~5.6 episodes/day
   (≥0.5 °C rise) and ~1.6/day (≥1.0 °C). At a 0.5 holdback that is a few
   control rows per day — enough to see a large effect in weeks, not a small
   one. State the minimum detectable effect before looking.
4. Is there any known mechanism by which heartbeat state would covary with
   cabinet temperature? Candidate (weak): GPU/biometrics load raises both
   biometrics-site event rate and temperature. That is a common cause, not a
   finding.

## Metric quality gate (findings recorded here)

Metric under consideration: heartbeat `/h1` organ/entanglement summary as a
predictor or correlate of cabinet temperature change.

1. **Provenance.** Traced above: heartbeat inputs are atom_type + three
   floats + source_service. No cabinet reading reaches it except as one
   unlabeled biometrics-site atom. PASS (traced), and the trace shows the
   linkage is thin.
2. **Independence.** FAIL. Biometrics-site occupancy and the cabinet
   temperature share upstream causes (GPU/CPU load -> heat and -> more
   biometrics atoms), and `cabinet_climate_activity_signal` is literally a
   monotone transform of the temperature among its inputs. Anything
   heartbeat shows about the biometrics site is not independent of the
   thermometer it is being compared with.
3. **Theory anchor.** FAIL. No named theory says a tensor-network entropy
   profile over five organ sites should lead or track cabinet temperature.
   Per the gate, "no real theory -> don't build a detector". This design
   therefore builds **no detector**, only a passive before/after log whose
   honest expected outcome is null.
4. **Live-data sanity.** NOT YET DONE for the crosswalk. Required before the
   analysis: confirm `/h1` fields are non-degenerate after #2451 deploys
   (including whether `smear` ever leaves its saturated state), and the
   reheat-floor test below.
5. **Existing mechanism.** The attend→act outcome tables and the heartbeat
   AST/HOT thread (`2026-07-29-heartbeat-into-ast-hot-design.md`) already
   carry heartbeat state into the self-model; no second consumer is added.
6. **Reversibility.** High: one extra log table/rows, no schema in heartbeat,
   no publisher. Delete the table to remove.

Conclusion: do not wire heartbeat into any decision. Run the log; analyze
offline; null result is a valid, recorded outcome.

## Proposed schema / API changes

Smallest seam that works. Heartbeat stays read-only.

- **No change to heartbeat's publishing.** No new bus channel. Heartbeat
  already serves `GET /h1` (and `/health`).
- The attend→act dispatcher/settlement code (whichever service owns
  `substrate_action_outcomes`) fetches `GET http://orion-heartbeat:7251/h1`
  at two moments per episode, treated AND control arm alike:
  - `t0`: at the dispatch eligibility snapshot (same time the eligibility
    snapshot is taken).
  - `t1`: at settlement (`t0 + 20 min`, the attend→act settle point).
- Stored as an additive JSON column/side table on the action-outcome row,
  e.g. `heartbeat_h1_before` / `heartbeat_h1_after` (the raw /h1 dict,
  including `generated_at`, `fire_window_sec`, `organ_seconds_since_last_fire`).
  Failure to fetch stores `null` plus a reason; never blocks the action
  (fail-open here is fine since it is observation only, but the null must be
  visible, not defaulted to zero).
- Heartbeat `/h1` is only recomputed every 30 s; the stored snapshot must
  include `generated_at` and the analysis must reject snapshots older than
  ~90 s at capture.
- Eval: extend the attend→act eval fixture mode with a check that both arms
  carry the pair of snapshots (or an explicit null reason).

## Files likely to touch

(For the later implementation, not this patch.)

- `orion/autonomy/evals/run_attend_act_loop_eval.py` — snapshot-pair check
- the dispatch/settlement code named in the attend→act design D4 chain
  (`substrate_action_outcomes` writer) — fetch and store the pair
- one sql migration for the additive column (if the table is Postgres)
- `docs/superpowers/pr-reports/…`, a short analysis script under
  `scripts/analysis/` that reads treated-vs-control and reports effect sizes
  with intervals
- NOT `services/orion-heartbeat/` (read-only; nothing to change after #2451)

## Non-goals

- No heartbeat publishing, no new bus channel, no schema registry entry.
- No heartbeat-driven action, gating, or trigger. No feeding the
  temperature into heartbeat as a new atom (would make it a thermometer by
  construction and defeat the independence check).
- No claim that heartbeat predicts temperature. No new heartbeat metric.
- No AC control, no change to the attend→act action itself.

## Open item: the reheat-driver floor

Heartbeat's baseline reheat is `reheat_prob = reheat_prob_scale * min(1, raw/3.0)`
with `raw = mean(|gap_zscore|)` over reliable bus-synapse edges
(`service.py:340-344`; saturation 3.0). Live notes in `bus_synaptic.py` put
`raw` around 1.0–1.1 in normal operation. A pure-noise (calm) z-score
population has `E|z| = sqrt(2/pi) ≈ 0.798`
(the same permanent floor AGENTS.md's metric gate describes for
`bus_synaptic_prediction_error`). So:

- `raw ≈ 1.1` -> signal 0.37; calm floor `0.8` -> signal 0.27. The reheat
  term never reaches 0, and roughly 0.8/1.1 ≈ 70% of its value may be the
  structural floor, not activity. The ensemble may therefore never reach true
  rest from this driver alone, whether or not anything is happening.
- The module docstring deliberately keeps the floor as a "baseline hum".
  That is a design choice, but it means the driver varies only over
  ~0.27–0.37 of its range; any crosswalk against it is crosswalking noise
  around a constant.

How to test (offline, read-only, before trusting any heartbeat variation):

1. Pull the history of `raw_mean_abs_gap_zscore` (heartbeat records it in the
   reheat state shown on `/health`; also persisted via the existing
   health/telemetry rows if present — check before adding anything) for
   several days.
2. Compare its minimum/low percentiles with `sqrt(2/pi)`. If the 1st
   percentile sits near 0.8 and the median near 1.1, the floor is real and
   the excess (raw − 0.8) is the only candidate signal.
3. Recompute the signal as `min(1, max(0, raw − 0.798)/(3.0 − 0.798))` on the
   stored history and check it can actually reach ~0 in quiet periods.
4. Null check: shuffle the per-edge z-scores in a frozen FalkorDB snapshot
   and confirm the shuffled mean|z| lands ~0.8 (the floor), not ~1.1.
5. Any definition change to the reheat driver needs Juniper's approval or an
   adjacent metric (standing rule), so this item ends in a recommendation,
   not a patch.

## Acceptance checks

- Gate findings above are copied into the eventual PR.
- A fixture proves treated and control episodes both get a before/after
  `/h1` pair (or an explicit null reason); a missing pair on one arm fails
  the eval.
- Snapshots older than 90 s at capture are rejected by the analysis script.
- The analysis reports effect size, interval and episode count for both
  arms, and states the pre-registered minimum detectable effect; an
  underpowered result is labelled UNDERPOWERED, not "no effect".
- The reheat-floor test above has been run and its numbers recorded before
  any heartbeat variable is interpreted.
- Heartbeat service diff for this work is empty.
- Live: after deploy, pull real episodes from Postgres and show the stored
  pairs. Until then UNVERIFIED.

## Recommended next patch

Nothing to build yet. In order:

1. Merge/deploy #2451 and confirm `/h1` recency fields and window are live.
2. Run the reheat-floor test (read-only analysis script + one paragraph of
   findings).
3. When the attend→act A1 action (and the hardware-watch reflex before it)
   ships, add the snapshot-pair capture and eval check as one small patch.
4. Collect for the number of weeks the amendment's volume math implies, then
   run the offline analysis once and record the result, null or not.
