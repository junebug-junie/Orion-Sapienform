# orion-heartbeat (v0)

A small, real tensor-network substrate (matrix product state, via `quimb`) that
tests whether Orion's grammar-event stream exhibits holographic-style
boundary/bulk entanglement structure. Read-only, additive, publishes nothing
to any existing consumer.

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
- No publishing anywhere, still — read-only research consumer, unchanged by
  the ensemble/dissipation work. Wiring this into any real downstream
  consumer (e.g. CollapseMirror's "insight" trigger) is explicitly deferred
  until the verdict thresholds are re-validated against live ensemble
  behavior, not just offline calibration (see Configuration below).

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

**Verdict thresholds (`_HIGH_RATIO`/`_LOW_RATIO` in `app/substrate/
reconstruction.py`) have not been re-validated against sustained live
ensemble behavior** — they carry over from offline calibration. Real
multi-organ silence is rare in current production (a 60h audit found 4 of 5
organs continuously active regardless of chat activity), so the
`concentrated` band specifically hasn't been observed live yet, only in
offline synthetic/replay calibration. Busy-state behavior (`redundant`,
`mean_ratio~0.79-0.91` observed live) is validated both offline and live.

## Debug surfaces

- `GET /health` — service status, absorption/queue/skip counters
  (`events_seen`/`events_queued`/`events_absorbed`/
  `events_dropped_queue_full`/`events_skipped_*`), ensemble size and seeds
  (`n_trajectories`/`seeds`, for forensic replay), and substrate health
  (`max_bond`/`norm`, aggregated across all trajectories).
- `GET /h1` — latest ensemble H1 result: `mean_ratio`/`std_ratio` across all
  N trajectories, `tick_count`, the seeds that produced this reading, and an
  explicit verdict (`redundant` / `concentrated` / `mixed`) — see
  Configuration above for threshold-validation status.
