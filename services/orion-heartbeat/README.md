# orion-heartbeat (v0)

A small, real tensor-network substrate (matrix product state, via `quimb`) that
tests whether Orion's grammar-event stream exhibits holographic-style
boundary/bulk entanglement structure. Additive. Since 2026-10-07 it publishes
a bounded H1 verdict trace back onto `orion:grammar:event` (see "What
heartbeat emits") -- a ledger record only; no reducer or field channel reads it.

Full design record, including the pivots that got here (three prior brainstorm
rounds, a shelved 2026-05-01 research charter, and why the first two attempts
at "Phase 0" were corrected before landing on this):
`docs/superpowers/specs/2026-07-24-spark-field-holographic-lattice-design.md`.

**2026-07-28: N-trajectory dissipation ensemble.** The original v0 substrate
never rested — no dissipation channel, so continuous entangling gates from
real organ traffic thermalized it to a permanent near-ceiling boundary/bulk
ratio (`verdict=redundant` on essentially every tick, confirmed live: 16+
consecutive ticks, `ratio` moving 0.73-0.98 but never crossing the
`_HIGH_RATIO` threshold downward). Root-caused as quantum-chaotic
thermalization (Page's-theorem territory), not a threshold-tuning bug — see
`docs/superpowers/specs/2026-07-28-precision-weighted-attention-organ-and-
heartbeat-discrimination-design.md`. The single `HeartbeatSubstrate` is now
wrapped by an `EnsembleSubstrate` of N independent trajectories with a real
relaxation mechanism, described below.

**2026-11-24: input overhaul (#3838).** A live audit found that most of what
the heartbeat read as gate strength were constants that the emitters stamp
per atom type, and that the bus_synaptic "reheat" sat below its own calm
floor. Each input now has an explicit rule (measured / presence / skip), the
reheat is gone, cortex-orch shares the execution seat, and the agent harness
has a seat. See "Seats and inputs" and "Reheat removed" below.

## What it does

1. Subscribes to the existing `orion:grammar:event` stream (`GrammarEventV1`)
   — no bespoke per-organ ingestion, reuses what
   `services/orion-substrate-runtime` already standardized. Message intake
   is decoupled from the N-trajectory absorb cost via a bounded
   `asyncio.Queue` + separate worker task (`HEARTBEAT_ABSORB_QUEUE_MAXSIZE`)
   — a burst degrades to queue backlog (and, if sustained, dropped-and-
   counted events), not a stalled bus consumer.
2. Routes each `atom_emitted` atom to one of five **seats** by the emitting
   service (`app/substrate/routing.py`): chat (`orion-hub`), biometrics,
   execution (`orion-cortex-exec` + `orion-cortex-orch`), transport
   (`orion-bus`), agent harness (`orion-harness-governor`). Seats are MPS
   sites 0-4 (boundary). Sites 5-9 are bulk, touched only through
   entanglement.
3. Applies the input rule for that (service, semantic_role) -- see "Seats and
   inputs". A measured atom gates with its real value as `level` (0 = rest,
   no gate). A presence atom gates at a fixed level. A skipped atom is
   counted and not absorbed. The gate is a local absorb-and-entangle update
   on **every trajectory in the ensemble**, bond dimension capped at 4.
4. On its own wall-clock timer (`HEARTBEAT_DECAY_INTERVAL_SEC`, independent
   of message arrival), each trajectory relaxes toward **its own
   seed-derived target** (not a shared target, so the trajectories stay
   distinct), at a rate suppressed when trajectories disagree
   (`app/substrate/ensemble.py::spread_gate`). Under real silence this
   brings the substrate to rest. There is no reheat any more.
5. Periodically computes the ensemble's mean boundary/bulk entanglement
   ratio (across all N trajectories) and its cross-trajectory spread — the
   headline H1 result, classified `redundant` / `concentrated` / `mixed`.

All access to the shared ensemble state (absorb, decay, H1
computation, `/health` stats) is serialized via a single `asyncio.Lock`
(`HeartbeatService._ensemble_lock`) — the actual quimb computation still runs
off the event loop thread (`asyncio.to_thread`), but never concurrently with
another read/write of the same tensor network. Found live: without this,
`asyncio.to_thread`-offloaded calls racing each other or a same-thread
`/health` read could corrupt the shared state or spuriously starve unrelated
async I/O (confirmed: a real absorb() backlog once produced a false
"FalkorDB timeout" — FalkorDB itself was fine, the event loop was just
starved of a chance to service that socket).

## What it deliberately does not do (v0 scope)

- No active-inference free-energy minimization (the 2026-05-01 charter's
  original update rule) — the relaxation mechanism above is a real
  dissipation channel but an honest, disclosed heuristic (documented choice,
  not a rigorously-derived open-quantum-system map), same spirit as
  `app/substrate/mps_state.py`'s other hand-picked constants
  (`_HOP_DECAY`/`_MIN_STRENGTH`/`_MAX_STRENGTH`). Not the charter's full
  variational free-energy machinery.
- No literal "partial trace + max-entropy completion + quantum fidelity" (the
  charter's literal H1 formula) — confirmed to be either near-tautological
  for a pure global MPS state (boundary/bulk reduced density matrices share
  an identical spectrum) or computationally too expensive for a tick loop at
  reasonable subset sizes. Uses the MPS's native, cheap bipartite
  entanglement entropy instead; see `app/substrate/reconstruction.py`'s
  module docstring for the full reasoning.
- No H2 (cross-organ mutual information), H3 (intervention propagation), H4
  (predictive surprise), shadow-comparison against `orion/spark/orion_tissue.py`,
  ablation baseline, or formal pre-registration process.
- No modification to `FieldStateV1`, `orion-field-digester`, or
  `orion/spark/orion_tissue.py` — this is a wholly separate, additive
  consumer of an existing stream.
- No `SelfStateV1` dependency anywhere.
- No field channel, substrate node or prior. The only output is the bounded
  `heartbeat.h1:` grammar trace below (plus the pre-existing `/h1` that
  orion-substrate-runtime's AST/HOT tick polls). Wiring the verdict into a
  real downstream consumer is still deferred -- see "Why nothing reaches the
  field" for the live evidence.

## What heartbeat emits

One kind of `GrammarEventV1` atom on `orion:grammar:event`
(`app/substrate/verdict_atoms.py`), trace prefix `heartbeat.h1:<node>:`,
`source_service=orion-heartbeat`, persisted to `grammar_events` by sql-writer:

- `h1_hourly_summary` -- one per `HEARTBEAT_VERDICT_SUMMARY_INTERVAL_SEC`
  (3600): per-class tick counts, raw flips, mean/min/max of std_ratio,
  mean_ratio and bulk depth, and per-producer counts of grammar atoms
  heartbeat could not route. Sent even when every H1 computation in the
  window failed (`h1_ticks=0`, confidence 0). A dead H1 loop shows up as
  missing summary rows. Every summary, and `/h1`, carries two labels:
  `verdict_meaning=noise_no_better_than_shuffled_input_2026-11-08` and
  `mean_ratio_meaning=last_3_events_sequence_meter_2026-11-08`.

**What the verdict class and `mean_ratio` mean (2026-11-08 replay).** The
memory-horizon replay (`evals/memory_horizon/`, and
`docs/superpowers/specs/2026-11-08-prediction-error-suite-rebuild-design.md`
"Heartbeat: what it measures today") found:

- The gates are fixed random unitaries. The tensor network encodes nothing
  learned.
- The verdict class is noise. Real input and shuffled input agree on it as
  often as two seeds do (67-70%).
- `mean_ratio` is reproducible across seeds (r = +0.91), but it remembers
  about the last 3 atoms (held-out R^2 0.88). It is a short sequence meter,
  not an integrator.

So the **`h1_verdict_transition` atom was retired** (2026-11-24, #3838). It
published settled class changes at confidence 1.0, which presented noise as a
finding. Its keys `HEARTBEAT_VERDICT_SETTLE_TICKS` and
`HEARTBEAT_VERDICT_MAX_TRANSITIONS_PER_HOUR` are gone. `/h1` still returns
`verdict`, because AST/HOT records it, but it is labelled. The real next step
is the rebuild the 11-08 doc recommends: a **trained** MPS (a Born machine fit
to Orion's state, #3296), so that entanglement across a cut means real mutual
information. Input tuning on random gates cannot get there. The input
overhaul here only removes dead inputs and makes the seats honest for that
rebuild.

Heartbeat subscribes to the same channel, so its own atoms come back; they
are skipped before routing (`events_skipped_self` on `/health`).

### Why nothing reaches the field

No field channel, substrate reducer or prior was added. Checked against the
metric gate (CLAUDE.md §0A) on the same 7 days:

- **No rest state.** `mean_ratio` never went below 0.668 (p5 0.80); the
  "true silence" branch (`mean_ratio <= 0.2`) fired 0 times. Every
  `concentrated` tick (2,792 of 2,792) came from the bulk-depth band, whose
  edges are percentiles of heartbeat's own past output -- a self-calibrated
  quantile, so it reads ~17% concentrated by construction.
- **No coupling to activity.** `concentrated` share sits at 14-20% in every
  UTC hour of the day. `std_ratio` vs total organ fires in the window:
  r = -0.008; bulk depth vs fires: r = 0.022. The verdict is not tracking how
  busy the organs are.
- **A third of it is dice.** Replaying the same 3 h of real atoms with only
  the random seed changed (42 -> 142 / 242) gives a different verdict on 31-35%
  of ticks (organ-map table below).
- **Already consumed.** AST/HOT already pulls `/h1` every tick
  (`SUBSTRATE_HEARTBEAT_H1_URL`), so the verdict is already visible to the
  self-model; a field channel would be a second copy of a signal that has
  not yet shown it measures anything.

So the atom is an observable trace and nothing more. Revisit if a
replay shows the verdict moving with something real (the cabinet crosswalk
experiment, `docs/superpowers/specs/2026-10-01-heartbeat-cabinet-crosswalk-design.md`).

### Seats and inputs (2026-11-24 overhaul, #3838)

**Why.** Juniper said the heartbeat ran on dead metrics. A 3-day audit of
`grammar_events` (`evals/audit_heartbeat_inputs.py`, table in the PR report)
confirmed it. For cortex-exec, cortex-orch and most of the bus, every atom of
a given type carried the same confidence, salience and uncertainty. The
emitter stamps those numbers; it does not measure them. The bus observer and
biometrics also fire on a fixed clock (hourly CV about 0.21), so for their
constant atoms even the timing says nothing.

**Seats** (`SEAT_ORDER`). N_SITES stays 10 and the boundary/bulk cut stays at
5, so the verdict bands and the probe geometry still apply.

| Site | Seat | Services |
| --- | --- | --- |
| 0 | orion-hub | orion-hub (chat turns, including Orion's own curiosity and day-letter turns) |
| 1 | orion-biometrics | orion-biometrics |
| 2 | orion-cortex-exec | orion-cortex-exec and orion-cortex-orch (about 70 atoms a day, all stamped, so merged) |
| 3 | orion-bus | orion-bus |
| 4 | orion-harness-governor | orion-harness-governor (agent harness, about 18,500 event atoms a day) |

**Input rules** (`ROLE_RULES`, one entry per live role).
`SERVICE_DEFAULT_RULES` catches a new role and counts it on `/health`
`inputs_unlisted_role`.

- **measured**: the value varies on live data, and it reads 0 on a quiet
  night often enough to count as calm. Biometrics `gpu_pressure_signal`,
  `thermal_pressure_signal`, `disk_pressure_signal` and `stability_signal`
  (inverted: 1 - stability). Bus `bus_activity_zscore_computed`: the |z| of
  the mesh publish rate minus its calm floor sqrt(2/pi), so a calm bus reads
  0 on about 53% of night ticks. Hub `repair_pressure_reading`,
  `repair_signal` and `stance_disposition`.
- **presence**: event-driven, every field stamped. All cortex-exec,
  cortex-orch and harness-governor roles. Bus `rpc_transport_timeout`. Hub
  `user_utterance` (its salience is word count, saturated at 1.0 on 78% of
  turns). These seats read calm by going silent.
- **skip**: fixed-clock atoms with constant fields (bus tick started,
  tick completed, health and census; biometrics node_context,
  telemetry_sample, capability_surface and node_availability). Gauges with a
  structural floor that never read calm (memory, fan, power, disk capacity,
  the cabinet activity gauges). `body_state`, a composite of the pressure
  gauges. Alarms pinned at 0 for 3 days (cabinet UV and both staleness
  signals). Hub `session_context`, a duplicate of `user_utterance`.

**Not routed** (`UNROUTED_PRODUCERS`, also on `/health`
`unrouted_producer_reasons`):

- gpu-pool: live, but about 90 event atoms a day, and there is no free seat
  without a change to N_SITES.
- llm-gateway, sql-writer, vision-frame-router: fixed-clock window atoms with
  stamped fields. The real counts exist only in the summary text.
- substrate-runtime: all-null snapshots, and an audio spike derived from the
  biometrics cabinet sensor (not independent).
- actions, dream, durable-runs, embodiment, mind, thought, topic-foundry:
  catalogued, but no atoms in 3 days.

Re-run the audit after any emitter change. It exits 1 when the data
contradicts a rule: a measured input that never varies or cannot rest, a
presence input on a fixed clock, an unlisted role, or an unnamed producer.

### Reheat removed (2026-11-24, #3838)

The reheat was a two-site entangling gate driven by the raw mean |gap_z| of
the `orion_bus_synapse` FalkorDB graph. It was dead as a driver:

- Live raw mean |gap_z| was 0.53. That is below the calm floor of a single
  |z|, sqrt(2/pi) ~= 0.80. reheat_prob was about 0.0035 per bond per tick: a
  near-constant hum.
- The 10-02 check found busy and quiet bus edges not significantly different
  (shuffle p=0.16).

Removing it needed an answer to the 07-28 problem: the substrate must still
relax. Without reheat, decay alone brings a busy trajectory to rest
(`tests/test_mps_state.py::test_decay_tick_relaxes_a_busy_state_to_rest`).
Replay of real atoms, before and after (`evals/replay_overhaul_before_after.py`):

Two 2-hour windows of real atoms on 2026-11-24 (night 07:00-09:00 UTC, day
18:00-20:00 UTC). Seed 42, 30 min warm-up dropped, then 30 min of pure
silence. "Coupling" is the Spearman correlation between an H1 number and the
number of atoms absorbed in the same 30 s window: does H1 move with activity?

| Window | Build | concentrated / mixed / redundant | bulk p50 | coupling: mean / std / bulk | mean_ratio after 30 min silence |
| --- | --- | --- | ---: | --- | ---: |
| night | before, reheat 0.0035 (as shipped) | 28% / 68% / 3% | 0.85 | 0.05 / 0.01 / -0.01 | 0.012 |
| night | before, reheat 0 | 29% / 71% / 1% | 0.85 | 0.02 / -0.03 / 0.02 | 0.000 |
| night | after | 28% / 71% / 1% | 0.80 | 0.20 / -0.37 / 0.69 | 0.000 |
| day | before, reheat 0.0035 (as shipped) | 25% / 75% / 0% | 0.85 | 0.05 / 0.06 / 0.00 | 0.000 |
| day | before, reheat 0 | 20% / 80% / 0% | 0.86 | -0.05 / -0.08 / -0.10 | 0.000 |
| day | after | 8% / 92% / 0% | 0.57 | 0.59 / -0.32 / 0.69 | 0.000 |

Read it this way:

- Turning the reheat off in the old build changed the verdict mix by no more
  than seed noise (README "Why nothing reaches the field": a third of
  verdicts flip with the seed alone). Its only visible effect was a floor of
  about 0.01 under mean_ratio in long silence. It was not doing work.
- Both builds relax to rest in 30 min of silence, so removing reheat did not
  bring back the 07-28 thermalization problem.
- The old build did not move with activity (every coupling within +-0.10).
  The overhauled build does: bulk depth rises with the atoms absorbed
  (rho 0.69 in both windows) and trajectory spread falls (rho -0.32 / -0.37).
- The verdict bands are percentiles of the old regime. Day bulk p50 fell
  from 0.85 to 0.57, so the bands need recalibration before the verdict
  classes mean anything (follow-up issue in the PR report).

Bus activity now reaches the substrate through the bus seat
(`bus_activity_zscore_computed`), which can read calm.

### Earlier history: organ-map replay (2026-10-07)

Before the overhaul, six newer producers were counted but not routed. A
replay showed that folding them onto the fixed 10 sites shifted the verdict
distribution beyond seed noise. That replay (`evals/replay_organ_map_options.py`)
was retired with the overhaul: the old API it used is gone, and the per-role
rules above answer its question.

## Run

```bash
cp services/orion-heartbeat/.env_example services/orion-heartbeat/.env
python scripts/sync_local_env_from_example.py orion-heartbeat
```

Then via `scripts/safe_docker_build.sh` (per CLAUDE.md; do not call `docker
compose` directly from the shared checkout):

```bash
scripts/safe_docker_build.sh orion-heartbeat up -d --build
curl -fsS http://localhost:7251/health
curl -fsS http://localhost:7251/h1
```

`/h1` returns `{"ok": false, "reason": "no_h1_computed_yet"}` until
`HEARTBEAT_H1_INTERVAL_SEC` (default 30s) has elapsed since start.

## Configuration

Ensemble/dissipation settings (`app/settings.py`, `.env_example`) — defaults
are sweep-derived, not guessed, from
`scripts/analysis/measure_heartbeat_ensemble_calibration.py` (offline
synthetic + real historical `grammar_events` replay; re-run that script
after changing any of these to validate against fresh live data before
deploying a change):

| Var | Default | Purpose |
| --- | --- | --- |
| `HEARTBEAT_N_TRAJECTORIES` | `8` | Ensemble size. Measured ~118ms/tick compute cost at N=8 — comfortably under this system's real average tick interval, with headroom for bursts but not unlimited. |
| `HEARTBEAT_DECAY_GAMMA` | `0.2` | Fraction each relaxation application contracts a site toward its own seed-derived target. |
| `HEARTBEAT_BASE_DECAY_PROB` | `0.15` | Base per-site decay probability before spread-gating. |
| `HEARTBEAT_DECAY_SPREAD_SENSITIVITY` | `4.0` | How sharply cross-trajectory disagreement suppresses decay. |
| `HEARTBEAT_DECAY_INTERVAL_SEC` | `2.0` | Wall-clock cadence for the decay loop, independent of message arrival. (Was `HEARTBEAT_DECAY_REHEAT_INTERVAL_SEC`; the `HEARTBEAT_REHEAT_*` and `FALKORDB_*` keys were removed with the reheat.) |
| `HEARTBEAT_ABSORB_QUEUE_MAXSIZE` | `10000` | Bound on the message-intake→absorb queue; sustained overflow drops-and-counts (`events_dropped_queue_full`) rather than blocking intake or growing unbounded. |
| `HEARTBEAT_VERDICT_ATOMS_ENABLED` | `true` | Publish the `heartbeat.h1:` grammar trace (see "What heartbeat emits"). |
| `HEARTBEAT_VERDICT_SUMMARY_INTERVAL_SEC` | `3600.0` | Summary atom cadence. |

**Verdict bands** (`app/substrate/reconstruction.py`) are percentiles of
heartbeat's own 48 h output (2026-09-01). On 7 days of live AST/HOT samples
(2026-09-30..10-07): mixed 80%, concentrated 17% (all via the bulk-depth band;
the `mean_ratio <= 0.2` silence branch never fired), redundant 3%. See "Why
nothing reaches the field" above.

## Debug surfaces

- `GET /health` — service status, absorption/queue/skip counters
  (`events_seen`/`events_queued`/`events_absorbed`/
  `events_dropped_queue_full`/`events_skipped_*`), ensemble size and seeds
  (`n_trajectories`/`seeds`, for forensic replay), and substrate health
  (`max_bond`/`norm`, aggregated across all trajectories). Also
  `events_skipped_self` (own atoms echoed back), `events_skipped_organ_by_source`
  (per-producer unrouted counts), `catalog_loaded` /
  `catalog_producers_unrouted` (channels.yaml producers heartbeat does not
  route; `null` if the catalog could not be read) / `uncatalogued_sources_seen`
  (sources on the wire the catalog does not list), `unrouted_producer_reasons`
  (why each catalogued producer has no seat), `seat_order`,
  `events_skipped_input` / `events_skipped_input_by_reason` (atoms an input
  rule skipped), `inputs_absorbed_by_kind` (measured / presence),
  `inputs_unlisted_role` (roles no rule names yet), and `verdict_atoms`
  (published / failed counts, confirmed verdict, current window, last atom).
- `GET /h1` — latest ensemble H1 result. Headline proprioception:
  `dark_seats` (organs with zero fires in the last `HEARTBEAT_ORGAN_FIRE_WINDOW_SEC`
  seconds, default 300, pruned by wall clock; an empty window reads unknown --
  `dark_seats=[]`, `organ_fire_counts={}`, `organ_distinctness=null` -- never
  all-dark; `organ_last_fired_at` / `organ_seconds_since_last_fire` /
  `fire_window_sec` tell a dark organ from a merely rare one, null = never seen
  since boot), `smear`/`smeared`
  (far/near entropy on the current profile; both null when the near end is
  dead -- carries less than a tenth of the far end's entanglement,
  `SMEAR_DEAD_RATIO` in `app/substrate/proprioception.py`, derived from the
  live distribution's trough -- rather than a 1e4..1e6 ratio), `organ_distinctness` (occupancy
  concentration). Also `verdict`, `mean_ratio`/`std_ratio` (mean saturates
  under real traffic — secondary), `bulk_penetration_depth`, `tick_count`,
  and the seeds that produced this reading.
