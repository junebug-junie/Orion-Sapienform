## Summary

This is step 0 of the heartbeat-seats / field-demon design (PR #3837, ticket #3838). It overhauls the heartbeat's inputs before anything is built on its seats.

- **Input rules.** Each input to the heartbeat (`services/orion-heartbeat`) now has one explicit rule per service and role. *Measured* means a real value that can read calm. *Presence* means an event-driven atom whose fields are stamped constants. *Skip* means fixed-clock atoms, duplicates, or gauges that never read calm. Stamped constants no longer set gate strength.
- **Reheat removed.** The bus_synaptic "reheat" is gone, together with its FalkorDB reader, its settings and env keys, the `last_reheat` block on `/health`, and the Hub panel that showed it. Decay alone still brings the substrate to rest.
- **Seats changed.** cortex-orch now shares the execution seat. The agent harness (`orion-harness-governor`, about 18,500 event atoms a day) takes the freed seat. Every other catalogued producer is either routed or named with a reason.
- **Evals and CI.** There is a new input audit eval that can run as a gate, and a before/after replay eval. A new CI workflow runs the heartbeat tests; until now nothing in CI ran them.
- **Probes re-run.** Both pre-registered probes were re-run against their original fail lines, and dated amendments were added to the pre-registration docs.

## Outcome moved

The heartbeat now moves with how busy Orion is. Before the overhaul it did not.

On replayed real atoms (two 2-hour windows on 2026-11-24, one night and one day), the old build's H1 numbers did not track the number of atoms absorbed: every Spearman ρ was within ±0.10. The new build's bulk depth tracks that count with ρ = 0.69 in both windows, and its trajectory spread falls as activity rises (ρ −0.32 and −0.37).

On the same 2 hours of live data, the old build now **fails** the neighbour-links probe: it reads as a thermometer (delta 0.004). The new build **holds** (delta 0.78).

## Current architecture (before)

- Five producers each had a seat: hub, biometrics, cortex-exec, bus, cortex-orch.
- Every atom's confidence × salience and confidence × (1 − uncertainty) set gate strength, with a minimum strength of 0.05.
- A two-site "reheat" ran on a 2 s timer. It was driven by the raw mean |gap_z| of the FalkorDB `orion_bus_synapse` graph.
- 13 catalogued producers were counted but not routed.

## Input audit: which metrics are dead

The audit covers 3 days of `grammar_events` (2026-11-21 22:18 to 11-24 22:51 UTC, 583,780 atoms).

- The table comes from `services/orion-heartbeat/evals/audit_heartbeat_inputs.py`, which exits 1 if live data contradicts a rule.
- "Night" means 07:00–13:00 UTC, which is 00:00–06:00 Mountain.
- "Fixed clock" means the hourly count CV is below 0.35. Fixed-clock emitters measure about 0.21; event-driven emitters measure 0.6 or more.
- The varying uncertainty on biometrics clockwork atoms is one per-tick telemetry value. The emitter stamps it on all 22 atoms of each tick, so on these atoms it is not a separate measurement.

**Answer to "which metrics are dead":**

- **12 inputs carry nothing at all.** They fire on a fixed clock with constant fields, or they are alarms pinned at 0:
  - bus: tick started, tick completed, health, census;
  - biometrics: node_context, telemetry_sample, capability_surface, node_availability, cabinet UV, both staleness alarms;
  - hub: session_context.
- **All cortex-exec, cortex-orch and harness atoms are stamped constants.** Only *when* they fire is real.
- **The reheat driver was dead** (see below).
- **8 biometrics gauges vary but can never read calm** (structural floors): memory, fan, power, disk capacity, and the cabinet activity gauges.
- **8 inputs are live and can read calm:**
  - biometrics gpu, thermal, disk io, and stability (inverted);
  - bus activity |z| (calm-floor corrected);
  - hub repair pressure, repair signal, and stance.
- **Correction to the design doc:** the hub is not "mostly fixed". Repair pressure varies across about 400 values. Stance is a real two-state reading.

| Input (service / role) | Producer | Per day | Fixed clock? | Varies? (distinct conf/sal/unc) | Reads calm at night? | Verdict | Rule now |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| biometrics / body_state | `services/orion-biometrics/app/grammar_emit.py:236` | 5253 | yes | 1/15739/15735 | - | redundant (strain composite of the pressure gauges (not independent); floor 0.12) | skip |
| biometrics / cabinet_ambient_audio_activity_signal | `services/orion-biometrics/app/grammar_emit.py:412` | 2632 | yes | 1/7883/7885 | - | varies, never calm (floor: night p05 0.11) | skip |
| biometrics / cabinet_ambient_audio_staleness_signal | `services/orion-biometrics/app/grammar_emit.py:446` | 2632 | yes | 1/1/7885 | - | DEAD (alarm pinned at 0.0 for 3 days) | skip |
| biometrics / cabinet_climate_activity_signal | `services/orion-biometrics/app/grammar_emit.py:412` | 2632 | yes | 1/7885/7885 | - | varies, never calm (floor: night p05 0.075) | skip |
| biometrics / cabinet_em_activity_signal | `services/orion-biometrics/app/grammar_emit.py:412` | 2632 | yes | 1/7885/7885 | - | varies, never calm (floor: night p05 0.11) | skip |
| biometrics / cabinet_proximity_activity_signal | `services/orion-biometrics/app/grammar_emit.py:412` | 2632 | yes | 1/7885/7885 | - | varies, never calm (floor: night p05 0.16) | skip |
| biometrics / cabinet_sensor_staleness_signal | `services/orion-biometrics/app/grammar_emit.py:429` | 2632 | yes | 1/1/7885 | - | DEAD (alarm pinned at 0.0 for 3 days) | skip |
| biometrics / cabinet_uv_activity_signal | `services/orion-biometrics/app/grammar_emit.py:412` | 2632 | yes | 1/1/7885 | - | DEAD (pinned at 0.0 for 3 days) | skip |
| biometrics / cabinet_vibration_activity_signal | `services/orion-biometrics/app/grammar_emit.py:412` | 2632 | yes | 1/7885/7885 | - | varies, never calm (floor: night p05 0.12) | skip |
| biometrics / capability_surface | `services/orion-biometrics/app/grammar_emit.py:249` | 5253 | yes | 1/1/15735 | - | DEAD (clockwork: fixed cadence and constant fields) | skip |
| biometrics / disk_capacity_pressure_signal | `services/orion-biometrics/app/grammar_emit.py:358` | 5253 | yes | 1/81/81 | - | varies, never calm (fill gauge pinned at ~0.81, not activity) | skip |
| biometrics / disk_pressure_signal | `services/orion-biometrics/app/grammar_emit.py:324` | 5253 | yes | 1/15732/15731 | yes: night p05 0.001, at rest 49% | live | measured |
| biometrics / fan_pressure_signal | `services/orion-biometrics/app/grammar_emit.py:371` | 5253 | yes | 1/104/3606 | - | varies, never calm (floor: night p05 0.36) | skip |
| biometrics / gpu_pressure_signal | `services/orion-biometrics/app/grammar_emit.py:285` | 5253 | yes | 1/324/10973 | yes: night p05 0.0, at rest 36% | live | measured |
| biometrics / memory_pressure_signal | `services/orion-biometrics/app/grammar_emit.py:298` | 5253 | yes | 1/15711/15727 | - | varies, never calm (floor: never below 0.207) | skip |
| biometrics / node_availability | `services/orion-biometrics/app/grammar_emit.py:265` | 5253 | yes | 1/1/15735 | - | DEAD (clockwork: fixed cadence and constant fields) | skip |
| biometrics / node_context | `services/orion-biometrics/app/grammar_emit.py:209` | 5253 | yes | 1/1/15735 | - | DEAD (clockwork: fixed cadence and constant fields) | skip |
| biometrics / power_pressure_signal | `services/orion-biometrics/app/grammar_emit.py:345` | 5253 | yes | 1/14624/14935 | - | varies, never calm (floor: night p05 0.10, at rest on 3% of night ticks) | skip |
| biometrics / stability_signal | `services/orion-biometrics/app/grammar_emit.py:393` | 5253 | yes | 1/15739/15735 | yes: night p05 0.008, at rest 41% | live | measured |
| biometrics / telemetry_sample | `services/orion-biometrics/app/grammar_emit.py:223` | 5253 | yes | 1/1/15735 | - | DEAD (clockwork: fixed cadence and constant fields) | skip |
| biometrics / thermal_pressure_signal | `services/orion-biometrics/app/grammar_emit.py:311` | 5253 | yes | 1/36/10749 | yes: night p05 0.0, at rest 34% | live | measured |
| bus / bus_activity_zscore_computed | `services/orion-bus/app/grammar_emit.py:265` | 7958 | yes | 1/1/23787 | yes: night p05 0.0, at rest 53% | live | measured |
| bus / bus_census_computed | `services/orion-bus/app/grammar_emit.py:227` | 7958 | yes | 1/1/9 | - | DEAD (clockwork: drift fraction spans 0-0.023 (9 values) | skip |
| bus / bus_health_observed | `services/orion-bus/app/grammar_emit.py:152` | 7958 | yes | 1/1/1 | - | DEAD (clockwork: fixed cadence and constant fields) | skip |
| bus / bus_observer_tick_completed | `services/orion-bus/app/grammar_emit.py:118` | 7958 | yes | 1/1/1 | - | DEAD (clockwork: fixed cadence and constant fields) | skip |
| bus / bus_observer_tick_started | `services/orion-bus/app/grammar_emit.py:96` | 7958 | yes | 1/1/1 | - | DEAD (clockwork: fixed cadence and constant fields) | skip |
| bus / rpc_transport_timeout | `orion/core/bus/async_service.py:847` | 162 | no (events) | 1/1/0 | yes: silent when idle | stamped constant; timing is real | presence |
| cortex-exec / exec_plan_started | `services/orion-cortex-exec/app/grammar_emit.py:143-383` | 940 | no (events) | 1/1/1 | yes: silent when idle | stamped constant; timing is real | presence |
| cortex-exec / exec_recall_gate_observed | `services/orion-cortex-exec/app/grammar_emit.py:143-383` | 940 | no (events) | 1/1/1 | yes: silent when idle | stamped constant; timing is real | presence |
| cortex-exec / exec_request_received | `services/orion-cortex-exec/app/grammar_emit.py:143-383` | 940 | no (events) | 1/1/1 | yes: silent when idle | stamped constant; timing is real | presence |
| cortex-exec / exec_result_assembled | `services/orion-cortex-exec/app/grammar_emit.py:143-383` | 941 | no (events) | 1/1/1 | yes: silent when idle | stamped constant; timing is real | presence |
| cortex-exec / exec_result_emitted | `services/orion-cortex-exec/app/grammar_emit.py:143-383` | 941 | no (events) | 1/1/1 | yes: silent when idle | stamped constant; timing is real | presence |
| cortex-exec / exec_step_completed | `services/orion-cortex-exec/app/grammar_emit.py:143-383` | 1985 | no (events) | 1/1/1 | yes: silent when idle | stamped constant; timing is real | presence |
| cortex-exec / exec_step_failed | `services/orion-cortex-exec/app/grammar_emit.py:143-383` | 10 | no (events) | 1/1/1 | yes: silent when idle | stamped constant; timing is real | presence |
| cortex-exec / exec_step_started | `services/orion-cortex-exec/app/grammar_emit.py:143-383` | 1994 | no (events) | 1/1/1 | yes: silent when idle | stamped constant; timing is real | presence |
| cortex-orch / route_arbitration_decided | `services/orion-cortex-orch/app/grammar_emit.py:156` | 69 | no (events) | 1/1/1 | yes: silent when idle | stamped constant; timing is real | presence |
| gpu-pool / gpu_lease_aborted | `services/orion-gpu-pool/app/runtime.py:1829` | 3 | no (events) | 1/1/0 | - | not routed: live but small (~90 event atoms/day, stamped fields); no free boundary seat without changing N_SITES | unrouted |
| gpu-pool / gpu_lease_backlogged | `services/orion-gpu-pool/app/runtime.py:1829` | 18 | no (events) | 1/1/0 | - | not routed: live but small (~90 event atoms/day, stamped fields); no free boundary seat without changing N_SITES | unrouted |
| gpu-pool / gpu_lease_dead_lettered | `services/orion-gpu-pool/app/runtime.py:1829` | 1 | no (events) | 1/1/0 | - | not routed: live but small (~90 event atoms/day, stamped fields); no free boundary seat without changing N_SITES | unrouted |
| gpu-pool / gpu_lease_expired | `services/orion-gpu-pool/app/runtime.py:1829` | 19 | no (events) | 1/1/0 | - | not routed: live but small (~90 event atoms/day, stamped fields); no free boundary seat without changing N_SITES | unrouted |
| gpu-pool / gpu_lease_recalled | `services/orion-gpu-pool/app/runtime.py:1829` | 21 | no (events) | 1/1/0 | - | not routed: live but small (~90 event atoms/day, stamped fields); no free boundary seat without changing N_SITES | unrouted |
| gpu-pool / gpu_lease_swap_failed | `services/orion-gpu-pool/app/runtime.py:1829` | 0 | no (events) | 1/1/0 | - | not routed: live but small (~90 event atoms/day, stamped fields); no free boundary seat without changing N_SITES | unrouted |
| gpu-pool / gpu_lease_swap_requested | `services/orion-gpu-pool/app/runtime.py:1829` | 6 | no (events) | 1/1/0 | - | not routed: live but small (~90 event atoms/day, stamped fields); no free boundary seat without changing N_SITES | unrouted |
| gpu-pool / gpu_lease_unavailable | `services/orion-gpu-pool/app/runtime.py:1829` | 22 | no (events) | 1/1/0 | - | not routed: live but small (~90 event atoms/day, stamped fields); no free boundary seat without changing N_SITES | unrouted |
| harness-governor / exec_plan_started | `orion/harness/grammar_emit.py:165-338` | 148 | no (events) | 1/1/0 | yes: silent when idle | stamped constant; timing is real | presence |
| harness-governor / exec_recall_gate_observed | `orion/harness/grammar_emit.py:165-338` | 142 | no (events) | 1/1/0 | yes: silent when idle | stamped constant; timing is real | presence |
| harness-governor / exec_request_received | `orion/harness/grammar_emit.py:165-338` | 148 | no (events) | 1/1/0 | yes: silent when idle | stamped constant; timing is real | presence |
| harness-governor / exec_result_assembled | `orion/harness/grammar_emit.py:165-338` | 148 | no (events) | 1/1/0 | yes: silent when idle | stamped constant; timing is real | presence |
| harness-governor / exec_result_emitted | `orion/harness/grammar_emit.py:165-338` | 137 | no (events) | 1/1/0 | yes: silent when idle | stamped constant; timing is real | presence |
| harness-governor / exec_step_completed | `orion/harness/grammar_emit.py:165-338` | 5586 | no (events) | 1/1/0 | yes: silent when idle | stamped constant; timing is real | presence |
| harness-governor / exec_step_failed | `orion/harness/grammar_emit.py:165-338` | 17 | no (events) | 1/1/0 | yes: silent when idle | stamped constant; timing is real | presence |
| harness-governor / exec_step_started | `orion/harness/grammar_emit.py:165-338` | 5586 | no (events) | 1/1/0 | yes: silent when idle | stamped constant; timing is real | presence |
| harness-governor / exec_tool_provenance_mismatch | `orion/harness/grammar_emit.py:165-338` | 34 | no (events) | 1/1/0 | yes: silent when idle | stamped constant; timing is real | presence |
| harness-governor / harness_fcc_step | `orion/harness/grammar_emit.py:165-338` | 6534 | no (events) | 0/0/0 | yes: silent when idle | stamped constant; timing is real | presence |
| hub / repair_pressure_reading | `services/orion-hub/scripts/grammar_emit.py:226` | 139 | no (events) | 359/400/359 | yes: silent when idle | live | measured |
| hub / repair_signal | `services/orion-hub/scripts/grammar_emit.py:183` | 11 | no (events) | 34/34/34 | yes: silent when idle | live | measured |
| hub / session_context | `services/orion-hub/scripts/grammar_emit.py:129` | 155 | no (events) | 1/1/1 | - | DEAD (duplicate: fires once per turn with user_utterance, all fields stamped) | skip |
| hub / stance_disposition | `services/orion-hub/scripts/grammar_emit.py:270` | 155 | no (events) | 1/2/2 | yes: silent when idle | live | measured |
| hub / user_utterance | `services/orion-hub/scripts/grammar_emit.py:145` | 155 | no (events) | 1/27/1 | yes: silent when idle | stamped constant; timing is real | presence |
| llm-gateway / llm_gateway_window_completed | `services/orion-llm-gateway/app/grammar_emit.py:599` | 1373 | yes | 1/1/0 | - | not routed: fixed-cadence window atoms with stamped fields; real counts exist only in summary text | unrouted |
| llm-gateway / llm_inference_window_observed | `services/orion-llm-gateway/app/grammar_emit.py:599` | 1129 | yes | 1/1/0 | - | not routed: fixed-cadence window atoms with stamped fields; real counts exist only in summary text | unrouted |
| sql-writer / storage_write_window_observed | `services/orion-sql-writer/app/write_health.py:573` | 26431 | yes | 1/1/0 | - | not routed: fixed-cadence window atoms with stamped fields; real counts exist only in summary text | unrouted |
| sql-writer / storage_writer_window_completed | `services/orion-sql-writer/app/write_health.py:573` | 1370 | yes | 1/1/0 | - | not routed: fixed-cadence window atoms with stamped fields; real counts exist only in summary text | unrouted |
| substrate-runtime / cabinet_ambient_audio_spike_signal | `orion/substrate/cabinet_ambient_spike_consumer.py:73` | 32 | no (events) | 1/96/0 | - | not routed: all-null bounded-input snapshots plus an audio spike derived from the biometrics cabinet sensor (not independent) | unrouted |
| substrate-runtime / system_one_bounded_input_state | `orion/substrate/system_one_appraisal.py:414` | 347 | yes | 0/0/0 | - | not routed: all-null bounded-input snapshots plus an audio spike derived from the biometrics cabinet sensor (not independent) | unrouted |
| vision-frame-router / vision_organ_window_completed | `services/orion-vision-frame-router/app/grammar_emit.py:250` | 1374 | yes | 1/1/0 | - | not routed: fixed-cadence window atoms with stamped fields; real counts exist only in summary text | unrouted |
| vision-frame-router / vision_stream_window_observed | `services/orion-vision-frame-router/app/grammar_emit.py:250` | 5495 | yes | 1/1/0 | - | not routed: fixed-cadence window atoms with stamped fields; real counts exist only in summary text | unrouted |

## Seats and reheat: what changed

- **Seats** (N_SITES 10 and the cut at 5 are unchanged):

  | Site | Seat | Services |
  | --- | --- | --- |
  | 0 | hub | orion-hub |
  | 1 | biometrics | orion-biometrics |
  | 2 | execution | orion-cortex-exec, orion-cortex-orch |
  | 3 | bus | orion-bus |
  | 4 | agent harness | orion-harness-governor |

  - cortex-orch is merged into execution. It fires about 70 times a day, and all of its fields are stamped.
  - gpu-pool is left out. It is live but small (about 90 atoms a day), and adding it would need an 11th site. That would break the probe geometry and the band calibration.
  - The other unrouted producers, with reasons, are listed in `routing.UNROUTED_PRODUCERS` and on `/health` under `unrouted_producer_reasons`.
- **Gate strength.** Each atom gates with one value, `level`: the rotation is 0.45 × level. A measured value at rest (level 0) applies no gate. Presence atoms gate at level 0.5. That equals the old neutral strength.
- **Bus seat.**
  - The only real bus measurement is `bus_activity_zscore_computed`. Its |z| is recovered exactly from the emitted uncertainty (uncertainty = |z|/3), then the calm floor √(2/π) is subtracted.
  - On real night data it reads 0 on 53% of ticks. That matches the theory: P(|z| < 0.80) = 0.575 for a calm z.
- **Reheat.** Removed outright. The table below shows before and after on real data.

  | Window | Build | concentrated / mixed / redundant | bulk p50 | coupling mean / std / bulk | mean_ratio after 30 min silence |
  | --- | --- | --- | ---: | --- | ---: |
  | night | before, reheat 0.0035 (as shipped) | 28 / 68 / 3 % | 0.85 | 0.05 / 0.01 / −0.01 | 0.012 |
  | night | before, reheat 0 | 29 / 71 / 1 % | 0.85 | 0.02 / −0.03 / 0.02 | 0.000 |
  | night | after | 28 / 71 / 1 % | 0.80 | 0.20 / −0.37 / 0.69 | 0.000 |
  | day | before, reheat 0.0035 (as shipped) | 25 / 75 / 0 % | 0.85 | 0.05 / 0.06 / 0.00 | 0.000 |
  | day | before, reheat 0 | 20 / 80 / 0 % | 0.86 | −0.05 / −0.08 / −0.10 | 0.000 |
  | day | after | 8 / 92 / 0 % | 0.57 | 0.59 / −0.32 / 0.69 | 0.000 |

  - Turning reheat off changed the verdict mix by no more than seed noise. (A third of verdicts flip with the seed alone; see the README.) Its only visible effect was a floor of about 0.01 under mean_ratio during long silence.
  - Without reheat, every build still relaxes to rest within 30 minutes. So the 07-28 thermalization problem does not come back.
  - Live evidence that the driver was dead: `/health` `last_reheat` on 2026-11-24 23:07 UTC showed raw mean |gap_z| = 0.473 and reheat_prob = 0.00315. That |gap_z| is below the 0.80 calm floor.

## Probe results (window 2026-11-24 20:51–22:51 UTC, fail lines unchanged)

**Lattice vs thermometer (09-19).**

- The "thermometer" call needs both fail conditions to hold. One holds, so the call is **not thermometer**.
- The result moved from "lattice" to the pre-declared **mixed** outcome: "shuffle moved but kick smeared".

| Build | rel_shuffle | Shuffle null? | Kick near / far | Smear | Decision |
| --- | ---: | --- | --- | ---: | --- |
| old, today's data | 1.03 | no | 0.49 / 0.006 | 0.012 | lattice |
| new | 1.19 | no | 0.48 / 0.32 | 0.67 | mixed (not thermometer) |

Why the kick now smears: in the old build the far cuts already sat at about 0.88 before the kick, which is the same level they reach after a kick. The old build's far end was full, so a kick could not move it. In the new build the far cuts rest at 0.52–0.66, so a strong pulse does travel. The kick test cannot tell "the pulse stays local" apart from "the far end is already saturated". The old "lattice" call on this arm partly depended on saturation. The fail line was not moved.

**Neighbour links v2 (09-20).**

| Build | Deciding pair exec–bus: co-fire / elsewhere (n) | delta (need ≥ 0.05) | Decision |
| --- | --- | ---: | --- |
| old, today's data | 29 / 191 | 0.004 | **thermometer** |
| new | 29 / 203 | 0.78 | **relational (holds)** |

Pair 1 (biometrics–execution) is inverted in both builds, as in September. That is geometry: pulses only travel rightward, so this pair cannot decide the probe. There is one deviation from the pre-reg: a level-0 atom still counts as a seat fire. That most likely lowers the co-fire mean. The amendments are in both pre-registration docs.

## Architecture touched

- `services/orion-heartbeat`: routing, the MPS update, the ensemble, the service loop, settings, compose, `.env_example`, the README, tests and evals.
- `services/orion-hub`: the attention-organ panel (reheat display removed, seat count fixed).
- `scripts/analysis/measure_heartbeat_*.py`: all four harnesses moved to the new API.
- `docs/research/preregistration/`: two dated amendments.
- `.github/workflows/orion-heartbeat-tests.yml`: new CI workflow.

## Files changed

- `services/orion-heartbeat/app/substrate/routing.py`: seats, per-role input rules, `level`, the unrouted-producer reasons.
- `services/orion-heartbeat/app/substrate/mps_state.py`: the level gate, `decay_tick`; reheat removed.
- `services/orion-heartbeat/app/substrate/ensemble.py`: decay only; level-0 atoms skip the spread recompute.
- `services/orion-heartbeat/app/substrate/bus_synaptic.py`, `tests/test_bus_synaptic.py`: deleted.
- `services/orion-heartbeat/app/service.py`: the new routing call; skip, kind and unlisted-role counters; decay loop; `/health` fields.
- `services/orion-heartbeat/app/settings.py`, `.env_example`, `docker-compose.yml`: reheat and FalkorDB keys removed; the interval key renamed.
- `services/orion-heartbeat/app/substrate/proprioception.py`, `lattice_probe.py`, `neighbor_links.py`: work per seat; the shuffle moves the routed site.
- `services/orion-heartbeat/evals/audit_heartbeat_inputs.py`: new input audit and gate.
- `services/orion-heartbeat/evals/replay_overhaul_before_after.py`: new before/after replay.
- `services/orion-heartbeat/evals/replay_organ_map_options.py`: deleted. It answered an old question and used the old API.
- `services/orion-heartbeat/evals/memory_horizon/*`: moved to the new API; the README notes the change.
- `services/orion-heartbeat/tests/*`: rewritten for the rules; new relaxation, level-0, audit and `/health` tests.
- `services/orion-hub/static/js/attention-organ.js`, `scripts/attention_organ_routes.py`, `tests/test_attention_organ_page.py`: reheat surface removed, harness short name added, seat count fixed.
- `scripts/analysis/measure_heartbeat_{lattice_vs_thermometer,neighbor_links,surprise,ensemble_calibration}.py`: new API, no reheat query.
- `docs/research/preregistration/2026-09-19-*.md`, `2026-09-20-heartbeat-neighbor-links-v2.md`: amendments.
- `.github/workflows/orion-heartbeat-tests.yml`: new.

## Schema / bus / API changes

- **Added** to `/health`:
  - `seat_order`, `unrouted_producer_reasons`;
  - `events_skipped_input` and `events_skipped_input_by_reason`;
  - `inputs_absorbed_by_kind`, `inputs_unlisted_role`.
- **Removed** from `/health`: `last_reheat`; `config.reheat_strength`, `config.reheat_prob_scale` and `config.decay_reheat_interval_sec`.
- **Renamed:** `config.decay_reheat_interval_sec` is now `config.decay_interval_sec`.
- **Behaviour changed:**
  - `allowlisted_organs` and `organ_site_map` now list 6 services on 5 seats.
  - `/h1` `dark_seats` and `organ_fire_counts` are keyed by seat: hub, biometrics, cortex-exec, bus, harness-governor.
- **Compatibility:** no bus channels or schemas changed. AST/HOT reads `dark_seats` generically. The Hub panel is updated in the same PR.

## Env/config changes

- **Added keys:** `HEARTBEAT_DECAY_INTERVAL_SEC`.
- **Removed keys:** `HEARTBEAT_REHEAT_STRENGTH`, `HEARTBEAT_REHEAT_PROB_SCALE`, `HEARTBEAT_DECAY_REHEAT_INTERVAL_SEC`, `FALKORDB_URI` and `FALKORDB_BUS_GRAPH` (heartbeat only).
- **Renamed keys:** `HEARTBEAT_DECAY_REHEAT_INTERVAL_SEC` is now `HEARTBEAT_DECAY_INTERVAL_SEC`.
- **`.env_example` updated:** yes.
- **Local `.env` synced:** yes, with `python scripts/sync_local_env_from_example.py orion-heartbeat --all-keys`, which added the new key. The 5 retired keys were removed by hand from the primary checkout's `services/orion-heartbeat/.env`. A backup was kept in the session scratchpad.
- **Skipped keys:** none.

## Tests run

```text
CI: all checks pass, including the new 'orion-heartbeat tests' workflow
PYTHONPATH=. pytest services/orion-heartbeat/tests -q        -> 178 passed
PYTHONPATH=. pytest services/orion-hub/tests/test_attention_organ_page.py services/orion-hub/tests/test_field_attention_operator_panel.py -q -> passed
```

## Evals run

```text
evals/audit_heartbeat_inputs.py --csv <3-day dump>                    -> exit 0 (no rule contradicted by data)
evals/replay_overhaul_before_after.py, 2 windows x {before rp0.0035, before rp0, after} -> table above
scripts/analysis/measure_heartbeat_lattice_vs_thermometer.py (old + new) -> table above
scripts/analysis/measure_heartbeat_neighbor_links.py (old + new)        -> table above
```

## Docker/build/smoke checks

```text
Not run: no rebuild or deploy from a worktree (CLAUDE.md section 8). Live behavior after deploy is UNVERIFIED -> #3871.
```

## Review findings fixed

The review was run in a subagent. It found no blockers.

- **Finding:** the pre-registration docs were not amended to match the changed substrate.
  - **Fix:** added a dated amendment to each doc. The fail lines are unchanged.
  - **Evidence:** the two pre-registration diffs.
- **Finding:** the neighbour probe counts level-0 atoms as seat fires.
  - **Fix:** recorded as a deviation in the amendment, with the direction of its bias. The proprioception docstring now states that a calm reading counts as lit.
  - **Evidence:** amendment text; `proprioception.py` docstring.
- **Finding:** the before arm of the replay eval defaulted to no reheat.
  - **Fix:** `--reheat-prob` is now required for an old build. The runs used both 0.0035 and 0.
  - **Evidence:** the replay table.
- **Finding:** the verdict bands are percentiles from the old input regime.
  - **Fix:** removed the "bands still apply" claim from the code comment, and filed #3869.
  - **Evidence:** day bulk p50 fell from 0.85 to 0.57.
- **Finding:** the junk-field test could pass for the wrong reason.
  - **Fix:** it now asserts the error is not a `SkippedInputError`.
  - **Evidence:** `test_routing.py`.
- **Finding:** a bus tick with no baseline was reported as "measured field missing".
  - **Fix:** it now has its own skip reason, "no baseline yet".
  - **Evidence:** `test_routing.py`.
- **Finding:** `inputs_absorbed_by_kind` counted queued atoms, not absorbed ones.
  - **Fix:** it is now counted after `absorb()` succeeds.
  - **Evidence:** `service.py`.
- **Finding:** level-0 atoms recomputed the ensemble spread for nothing.
  - **Fix:** the ensemble returns early when level ≤ 0. The physics is identical.
  - **Evidence:** `ensemble.py`.
- **Finding:** the Hub panel showed a wrong "sites" count.
  - **Fix:** it now shows the services and seats separately.
  - **Evidence:** `attention-organ.js`.
- **Finding:** the calibration harness skipped atoms without counting them.
  - **Fix:** it now counts them.
- **Finding:** the audit was one-directional.
  - **Fix:** added a `SKIP_NOW_VARIES` flag and a test for it.
- **Finding (CI, static gates):** the deleted `bus_synaptic.py` was still registered in `config/substrate/prediction_error_readers.yaml`.
  - **Fix:** removed the stale entry.
  - **Evidence:** `check_pe_alarm_readers: OK`; Static repo gates now pass.
- **Finding, noted and not changed:** the second bus tick after a restart saturates to level 1.0. This is a transient in the emitter. It is now documented on the level function.

## Restart required

Merge the PR first. Then, from the clean primary `main` checkout synced with `origin/main`:

```bash
scripts/safe_docker_build.sh orion-heartbeat up -d --build
```

Hub static JS changed. A Hub rebuild picks it up:

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- **Severity: medium.**
  - **Concern:** the verdict classes (concentrated / mixed / redundant) are calibrated on the old regime. During the day replay, concentrated fell from about 20–25% to 8%.
  - **Mitigation:** the verdict is ledger-only and nothing consumes it. Recalibration is tracked in #3869.
- **Severity: medium.**
  - **Concern:** the lattice probe's kick arm now smears. The probe still says "not thermometer", but organ identity now sits in the mix, not in a local pulse.
  - **Mitigation:** reported as the pre-declared "mixed" outcome. No consumer is wired to it.
- **Severity: low.**
  - **Concern:** the skipped atoms are still emitted. Only the heartbeat ignores them. Other consumers, such as the field digester reading biometrics pressures, still read them. The `orion_bus_synapse` graph also lives on for other readers. The heartbeat's own reheat reader and loop are fully gone.
- **Severity: low.**
  - **Concern:** power, memory and fan pressure are real body signals, and they were dropped from the heartbeat because they have floors.
  - **Mitigation:** to get them back, feed them in as change-from-baseline. That would be a new metric and needs its own gate.
- **Issues:**
  - opened #3869 (band recalibration), #3870 (window producers' counts exist only in summary text), #3871 (verify-live);
  - refs #3838 (left open until #3871 is verified).

## Open questions for Juniper

1. **Seats per service, or per body-atlas organ?** This PR keeps one seat per service, which is reversible. Body-atlas organs would put several services on one organ, or split a service across organs. That is a new seat map, and it would need a new band calibration.
2. **Keep the MPS?** After the overhaul, the heartbeat moves with activity: bulk depth ρ 0.69. But much of that is timing plus a few gauges. The per-seat counts and levels alone might be the product, with the MPS kept as one reader. Step 1 (the demon replay) could test this: compare seat-level forecasts with and without the MPS.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3873

🤖 Generated with [Claude Code](https://claude.com/claude-code)
