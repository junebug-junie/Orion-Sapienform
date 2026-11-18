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

## What it does

1. Subscribes to the existing `orion:grammar:event` stream (`GrammarEventV1`)
   — no bespoke per-organ ingestion, reuses what
   `services/orion-substrate-runtime` already standardized. Message intake
   is decoupled from the N-trajectory absorb cost via a bounded
   `asyncio.Queue` + separate worker task (`HEARTBEAT_ABSORB_QUEUE_MAXSIZE`)
   — a burst degrades to queue backlog (and, if sustained, dropped-and-
   counted events), not a stalled bus consumer.
2. Filters to five confirmed-live organs (chat/`orion-hub`,
   biometrics/`orion-biometrics`, execution/`orion-cortex-exec`,
   transport/`orion-bus`, route/`orion-cortex-orch`) — see
   `app/substrate/routing.py`'s `ORGAN_SITE_MAP`.
3. Routes each `atom_emitted` event's atom onto one of 10 MPS sites (5
   boundary, one per organ; 5 bulk, touched only via entanglement
   propagation) and applies a small local absorb-and-entangle update to
   **every trajectory in the ensemble**, bond-dimension capped at 4.
4. On its own wall-clock timer (`HEARTBEAT_DECAY_REHEAT_INTERVAL_SEC`,
   independent of message arrival — real organ traffic essentially never
   goes fully quiet, confirmed live: 4 of 5 organs stay continuously active
   regardless of chat activity, so decay/reheat can't wait on message
   ticks), each trajectory:
   - Relaxes toward **its own seed-derived initial state** (not a shared
     target across trajectories — preserves cross-trajectory diversity
     instead of collapsing to clones), at a rate suppressed when
     trajectories currently disagree (`app/substrate/ensemble.py::
     spread_gate` — disagreement means something's still unresolved).
   - Gets a small **two-site entangling "reheat"** gate, strength driven by
     real live `bus_synaptic` activity (`orion_bus_synapse` FalkorDB graph's
     raw `gap_zscore` data — the RAW mean, not the calm-floor-corrected
     anomaly-detection version; see `app/substrate/bus_synaptic.py`'s module
     docstring for why that distinction matters here). A one-site reheat
     gate was tried first and is **mathematically incapable** of moving
     entanglement entropy across any cut (basic invariance theorem) — must
     be two-site.
5. Periodically computes the ensemble's mean boundary/bulk entanglement
   ratio (across all N trajectories) and its cross-trajectory spread — the
   headline H1 result, classified `redundant` / `concentrated` / `mixed`.

All access to the shared ensemble state (absorb, decay/reheat, H1
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
  original update rule) — the relaxation/reheat mechanism above is a real
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

Two kinds of `GrammarEventV1` atom on `orion:grammar:event`
(`app/substrate/verdict_atoms.py`), trace prefix `heartbeat.h1:<node>:`,
`source_service=orion-heartbeat`, persisted to `grammar_events` by sql-writer:

- `h1_verdict_transition` -- the verdict class changed and the new class held
  for `HEARTBEAT_VERDICT_SETTLE_TICKS` (3) consecutive H1 ticks (~90 s).
  Summary carries `from`/`to`, `held_ticks`, `held_since`, `mean_ratio`,
  `std_ratio` (ensemble spread), `mean_std_ratio_while_held`,
  `bulk_penetration_depth`, `suppressed_since_last`. Capped at
  `HEARTBEAT_VERDICT_MAX_TRANSITIONS_PER_HOUR` (12) per rolling hour.
- `h1_hourly_summary` -- one per `HEARTBEAT_VERDICT_SUMMARY_INTERVAL_SEC`
  (3600): per-class tick counts, raw flips, settled/suppressed/failed
  transitions, mean/min/max of std_ratio, mean_ratio and bulk depth, and
  per-producer counts of grammar atoms heartbeat could not route. Sent even
  when every H1 computation in the window failed (`h1_ticks=0`, confidence 0).
  A dead H1 loop shows up as missing summary rows.

Why debounce: on 7 days of the verdict as AST/HOT recorded it
(`substrate_attention_self_model`, 16,642 samples, 2026-09-30..10-07) the class
changed 31.8 times an hour. `redundant` runs had median length 1 and never
exceeded 3 samples; `concentrated` median 1, max 6. Emitting on every flip
would be ~760 atoms/day of threshold noise. Settle ticks 2 -> 5.0/h, 3 ->
1.1/h, 4 -> 0.24/h on the same history.

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

### Organ map: unrouted producers are counted, not routed

The MPS has 10 fixed sites with the boundary/bulk cut at 5; H1 reads that
cut. `routing.ORGAN_SITE_MAP` routes 5 organs onto boundary sites 0-4. Six
newer catalogued producers (gpu-pool, harness-governor, llm-gateway,
sql-writer, substrate-runtime, vision-frame-router; ~40k atoms/day, ~21% of
the grammar atoms heartbeat sees) are dropped. They are now **counted per producer**
(`/health` `events_skipped_organ_by_source`, the hourly atom's
`unrouted_atoms=`), and the bus catalog names the gap
(`catalog_producers_unrouted`, `uncatalogued_sources_seen`). The map itself
is unchanged.

Replay (`evals/replay_organ_map_options.py`, real `grammar_events`, the
service's own ensemble code, reheat held at 0.0054, first 30 min warm-up
dropped, ~300 H1 ticks per run):

| Window (3 h) | Option | concentrated | mixed | redundant | ticks agreeing with current |
| --- | --- | --- | --- | --- | --- |
| 10-06 21:14 - 10-07 00:15 UTC | current (5 organs, seed 42) | 25.4% | 74.3% | 0.3% | -- |
| | fold extras onto boundary 0-4 | 20.5% | 71.6% | 7.9% | 62.7% |
| | extras onto bulk 5-8 | 19.5% | 77.6% | 3.0% | 68.6% |
| | **control: current, seed 142** | 24.8% | 73.9% | 1.3% | 69.3% |
| | **control: current, seed 242** | 25.7% | 71.3% | 3.0% | 65.3% |
| 10-05 21:14 - 10-06 00:13 UTC | current | 28.0% | 70.7% | 1.3% | -- |
| | fold extras onto boundary 0-4 | 28.0% | 69.3% | 2.7% | 67.3% |
| | extras onto bulk 5-8 | 24.0% | 74.0% | 2.0% | 69.3% |

Read it this way:

- Changing only the random seed already flips about a third of individual
  verdicts. Remapping flips about the same share, so per-tick agreement
  cannot tell the two apart.
- At distribution level, folding onto the boundary pushed `redundant` from
  0.3% to 7.9% and `concentrated` from 25% to 20% in one window. That is
  outside the seed spread (0.3-3.0% and 24.8-25.7%). In the other window it
  barely moved.
- Putting organs on the bulk changes what "bulk" means. Bulk is supposed to
  be reached only through entanglement. Site 9 cannot take an organ at all:
  `absorb()` needs a right-hand neighbour.

So remapping would change what H1 means in a way the replay cannot rule
out, and the gain would be organs whose atoms would land on another organ's
seat. Decision: keep the map, count the drops. A larger N_SITES would be a
new instrument and needs its own calibration; it is out of scope here.

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
| `HEARTBEAT_REHEAT_STRENGTH` | `0.08` | Two-site reheat gate rotation angle. |
| `HEARTBEAT_REHEAT_PROB_SCALE` | `0.02` | Scales real `bus_synaptic` activity into a per-bond reheat probability. |
| `HEARTBEAT_DECAY_REHEAT_INTERVAL_SEC` | `2.0` | Wall-clock cadence for the dissipation loop, independent of message arrival. |
| `FALKORDB_URI` / `FALKORDB_BUS_GRAPH` | `redis://orion-athena-falkordb:6379` / `orion_bus_synapse` | Real live reheat driver — same graph `services/orion-substrate-runtime` already reads, additive read-only consumer. |
| `HEARTBEAT_ABSORB_QUEUE_MAXSIZE` | `10000` | Bound on the message-intake→absorb queue; sustained overflow drops-and-counts (`events_dropped_queue_full`) rather than blocking intake or growing unbounded. |
| `HEARTBEAT_VERDICT_ATOMS_ENABLED` | `true` | Publish the `heartbeat.h1:` grammar trace (see "What heartbeat emits"). |
| `HEARTBEAT_VERDICT_SETTLE_TICKS` | `3` | Consecutive H1 ticks a new verdict must hold before a transition atom. |
| `HEARTBEAT_VERDICT_SUMMARY_INTERVAL_SEC` | `3600.0` | Summary atom cadence. |
| `HEARTBEAT_VERDICT_MAX_TRANSITIONS_PER_HOUR` | `12` | Rolling-hour cap on transition atoms; extras counted as suppressed. |

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
  (sources on the wire the catalog does not list), and `verdict_atoms`
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
