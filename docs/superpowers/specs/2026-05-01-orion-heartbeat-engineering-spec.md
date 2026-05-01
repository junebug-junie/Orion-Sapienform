# Design: Orion Heartbeat — Tensor-Network Substrate Service

**Date:** 2026-05-01
**Status:** Draft v1, pre-implementation
**Scope:** New service `services/orion-heartbeat/` + `orion/heartbeat/` library + measurement harness in `scripts/heartbeat_research/` + reducer touch points in mature organ services.
**Companion document:** `docs/research/2026-05-01-orion-heartbeat-research-charter.md`

---

## Summary

A new dedicated service, `orion-heartbeat`, that runs continuously and owns a small tensor-network substrate (matrix product state, N=24 sites, bond dimension χ=4) implementing active-inference-style update dynamics. The substrate ingests `SurfaceEncoding v2` records produced by reducers wrapping events from mature mesh organs, evolves under variational free-energy minimization on each tick, derives a low-dimensional self-field φ that is broadcast on the bus for downstream consumers, emits a tick-ahead forecast, and computes prediction surprise on the subsequent tick. State is persisted with crash-safe writes. The service is additive: ablation = disable, mesh reverts to current behavior with no functional loss.

The existing `orion/spark/orion_tissue.py` (v0 tissue) continues to run inside `orion-spark-introspector` during a 4–6 week shadow comparison window. After shadow comparison, a deprecation decision is made.

This spec is the engineering counterpart to the research charter; the *why* lives there, the *what and how* lives here.

---

## Scope & Non-goals

### In scope (v1)

- New service `orion-heartbeat` running continuously.
- Tensor-network substrate via `quimb`, MPS state at N=24, χ=4, physical dimension d=4.
- `SurfaceEncoding v2` schema (substrate-agnostic boundary representation).
- Reducer registry covering 13 mature organ event types (see §6).
- Variational free-energy update dynamics.
- φ broadcast on bus.
- Forecast and surprise computation.
- Crash-safe persistence (per-tick snapshot, restart-tolerant).
- Measurement harness for hypothesis H1 (boundary reconstruction fidelity).
- Skeleton harnesses for H2, H3, H4 wired but with v1.5 / v2 measurement campaigns.
- Ablation baseline runner.
- Shadow comparison reporting against v0 tissue.
- HTTP debug surfaces and bus RPC for inspection.
- Service-local tests and integration tests.

### Out of scope (v1)

- Larger substrates (PEPS, MERA, learned tensor-network heads). Reserved for v2.
- Wiring stubbed organs (`orion-self-experiments`, `orion-discussion-window`). They join when they exist.
- Deprecation of v0 tissue in `orion-spark-introspector`. Decided after shadow comparison.
- Modifying any existing organ's functional behavior. Reducers are read-only consumers of existing bus events.
- Hub UI changes for the heartbeat. Inspect surfaces are read-only HTTP debug endpoints in v1.
- Frontier inference scaling. The substrate is intentionally small.
- Any modification to autonomy policy gates, social bridge, or other safety-critical control surfaces.

---

## Service Architecture: `orion-heartbeat`

### Directory layout

```
services/orion-heartbeat/
  Dockerfile
  docker-compose.yml
  requirements.txt
  .env_example
  README.md
  app/
    __init__.py
    main.py
    settings.py
    service.py
    pipeline/
      __init__.py
      tick.py
      ingest.py
      broadcast.py
      persistence.py
    reducers/
      __init__.py
      registry.py
      chat.py
      biometrics.py
      equilibrium.py
      recall.py
      vision.py
      social.py
      autonomy.py
      planner.py
      agent_chain.py
      world_pulse.py
      state_journaler.py
      journaler.py
      spark_introspector.py
    substrate/
      __init__.py
      mps_state.py
      free_energy.py
      forecast.py
      surprise.py
      reconstruction.py
    schemas/
      __init__.py
      surface_encoding_v2.py
      tick_event.py
      phi_v1.py
    rpc/
      __init__.py
      debug_endpoints.py
    observability/
      __init__.py
      stats.py
      logging_setup.py
  tests/
    __init__.py
    conftest.py
    test_service_health.py
    test_substrate_mps.py
    test_reducers.py
    test_tick_lifecycle.py
    test_persistence.py
    test_forecast_surprise.py
    test_h1_reconstruction.py
    test_ablation_runner.py
    fixtures/
      sample_envelopes.py
      golden_tick_states.py
```

The companion measurement harness lives in `scripts/heartbeat_research/` (separate from the service so it can be run against historical or replayed bus data without touching live state). Per-hypothesis pre-registration documents live in `docs/research/preregistration/`.

### Process model

Single async Python process (`asyncio` event loop) running:

1. Bus subscriber loop — reads from allowlisted bus channels into a bounded ingest queue.
2. Tick loop — runs at configurable rate (default 1 Hz; debounced down to 100 ms minimum interval if many events arrive in succession).
3. Snapshot writer task — flushes substrate state to disk on every Nth tick (default N=10) and on graceful shutdown.
4. Stats publisher task — emits service health on `orion:heartbeat:stats` every 5 seconds.
5. HTTP debug server (FastAPI) on port `7250`.

### Tick lifecycle

```
tick T:
  1. drain_ingest_queue() -> List[SurfaceEncodingV2]
  2. for encoding in encodings:
       update_op = project_to_substrate(encoding)
       substrate.absorb(update_op)
  3. substrate.minimize_free_energy(steps=variational_steps)
  4. phi = substrate.derive_phi()
  5. forecast_T1 = substrate.forecast_next_tick()
  6. if T > 0:
       surprise = kl_divergence(forecast_from_T-1, observed_boundary_at_T)
  7. broadcast(phi, forecast_T1, surprise) -> bus
  8. record_tick_event(T, phi, surprise, encoding_count, ...)
  9. if T % snapshot_interval == 0:
       persistence.write_snapshot(substrate.state)
```

### Ports and bus channels

| Resource | Value |
|---|---|
| HTTP debug port | `7250` |
| Bus subscribe (input) | dynamic via `HEARTBEAT_INPUT_ALLOWLIST_PATTERNS` (default: chat events, biometrics, equilibrium, recall results, vision events, social turns, autonomy state, planner outputs, agent chain results, world-pulse signals, state-journaler frames, journaler entries, spark-introspector output) |
| Bus subscribe (denylist) | `orion:heartbeat:*` to prevent feedback loops |
| Bus publish: φ broadcast | `orion:heartbeat:phi` (kind `heartbeat.phi.v1`) |
| Bus publish: tick event | `orion:heartbeat:tick` (kind `heartbeat.tick.v1`) |
| Bus publish: forecast | `orion:heartbeat:forecast` (kind `heartbeat.forecast.v1`) |
| Bus publish: surprise | `orion:heartbeat:surprise` (kind `heartbeat.surprise.v1`) |
| Bus publish: stats | `orion:heartbeat:stats` (kind `heartbeat.stats.v1`) |
| Bus RPC (debug introspection) | `orion:heartbeat:rpc:request` |
| Snapshot path | `${HEARTBEAT_SNAPSHOT_PATH:-/mnt/storage-lukewarm/orion/heartbeat/state.npz}` |

---

## Substrate: Tensor-Network State via quimb

### Representation

A matrix product state (MPS) implemented as `quimb.tensor.MatrixProductState`:

- `N` sites, default 24 (configurable via `HEARTBEAT_SUBSTRATE_N`).
- Bond dimension `chi`, default 4 (configurable via `HEARTBEAT_SUBSTRATE_CHI`).
- Physical dimension `d`, default 4 (configurable via `HEARTBEAT_SUBSTRATE_D`).
- State normalized to unit norm.

The MPS state is the substrate's bulk representation. Boundary state at any partition is computed via partial trace over the complementary partition, yielding a reduced density matrix ρ_boundary.

### Initialization

On first start (no snapshot present), the state is initialized to a small random MPS with bond dim χ=4 and unit norm, seeded with a deterministic RNG (seed configurable; default 42 for reproducibility during early development). Random initialization is preferred over a "zero state" because pure-zero states have degenerate entanglement spectra and are pathological for the dynamics.

### Update dynamics

Each tick, the substrate absorbs new evidence and minimizes a variational free-energy objective.

**Free-energy objective** (initial v1 form):

```
F(state | evidence) = E[log q(state) - log p(state, evidence)]
                    = D_KL(q(state) || p(state | evidence)) - log p(evidence)
```

where:
- `q(state)` is the substrate's current variational distribution (the MPS state interpreted as amplitudes over a discrete configuration space).
- `p(state, evidence)` is the joint generative model: `p(evidence | state) * p(state)`.
- `p(state)` is the substrate's prior, initially the previous tick's state with a small temperature for exploration.
- `p(evidence | state)` is the likelihood of observed boundary states given the substrate's current state, parameterized by the encoding-to-update operator (see §5).

**Practical implementation:** because exact free-energy minimization on an MPS is computationally heavy, v1 uses a finite number (default 4) of variational sweeps per tick with single-site or two-site updates (DMRG-style), each minimizing local free energy contribution. quimb provides primitives for these sweeps.

**Hyperparameters (v1, hand-tuned for tractability):**

- `variational_steps` per tick: 4 (configurable)
- `temperature` of prior carryover: 0.1 (configurable)
- `evidence_precision`: 1.0 (configurable; precision-weighting on observation likelihood)

These are documented as choices, not derivations. Sensitivity analysis on these parameters is part of v1.5 / v2.

### State persistence

Substrate state is persisted as a quimb-compatible numpy archive on every Nth tick (default N=10) and on graceful shutdown. Snapshot file format:

```
state.npz:
  tensors_*.npy       (one per MPS site, indexed)
  metadata.json       (N, chi, d, tick_count, last_tick_timestamp, schema_version)
```

Restart procedure: if `state.npz` exists at the configured path, load it; verify metadata matches expected substrate dimensions; if mismatch, fail loudly rather than silently re-initialize. Operator must explicitly delete or move the snapshot to start fresh.

A backup snapshot (`state.npz.bak`) is written before each new snapshot, allowing one-step rollback on corruption.

---

## `SurfaceEncoding v2` Schema

`SurfaceEncoding v2` is the substrate-agnostic boundary representation of an organ event. Unlike v1 (`orion/spark/surface_encoding.py`), it does not assume a 2D+channels grid topology; it carries enough information for any substrate to interpret an organ event.

### Schema

```python
class SurfaceEncodingV2(BaseModel):
    schema_version: Literal["surface_encoding.v2"] = "surface_encoding.v2"
    encoding_id: str                      # stable hash of source event id
    source_organ: str                     # e.g. "biometrics", "social_room", "recall"
    source_event_kind: str                # e.g. "biometrics.snapshot.v1"
    timestamp: datetime
    channel_assignments: list[ChannelAssignment]
    precision: float = Field(default=1.0, ge=0.0, le=10.0)
    semantic_tags: list[str] = Field(default_factory=list)
    feature_vec: list[float] | None = None
    raw_event_ref: str | None = None      # bus envelope id for traceability
    meta: dict[str, Any] = Field(default_factory=dict)


class ChannelAssignment(BaseModel):
    site_index: int                       # which substrate site is affected
    operator_kind: Literal["amplitude", "phase", "rotation", "projection"]
    operator_params: list[float]          # parameters for the chosen operator
```

### Channel assignment

Each organ's reducer is responsible for declaring which substrate sites its events affect. The mapping is pre-declared in the reducer's docstring and committed to the engineering spec; ad-hoc reassignment is forbidden.

The mapping is documented in `app/reducers/SITE_ASSIGNMENT_TABLE.md` (created at v1 ship). Initial assignment plan:

| Organ | Sites | Rationale |
|---|---|---|
| chat (operator) | 0–3 | Operator-facing surface; high-bandwidth |
| chat (Orion-emitted) | 4–7 | Orion's response surface |
| recall | 8–11 | Memory/context surface |
| social | 12–13 | Relational surface |
| autonomy state | 14–15 | Internal state surface |
| equilibrium | 16–17 | Homeostatic surface |
| biometrics | 18–19 | Embodied infra surface |
| vision | 20 | Embodied perception |
| journaler | 21 | Autobiographical compression |
| world-pulse | 22 | External-environment perception |
| planner / agent chain / spark-introspector | 23 | Reflective/planning surface |

This is a v1 hand-assignment, not derived. Assignments may be reorganized in v1.5 based on early operational data and the H2 mutual-information measurement.

### Reducer contract

Each reducer is a function:

```python
async def reduce_<organ>(envelope: BaseEnvelope) -> SurfaceEncodingV2 | None:
    ...
```

Returns `None` if the event should not produce a substrate update (e.g., heartbeat-self-loop guard, malformed payload). Returns a `SurfaceEncodingV2` otherwise.

Reducers must be deterministic and side-effect-free.

---

## Organ Wiring Matrix

The thirteen v1 reducers, with their input bus channels, expected event kinds, and site assignments:

| Reducer | Input channel(s) | Event kind | Site(s) | Notes |
|---|---|---|---|---|
| `chat.py` | `orion:chat:history:turn`, `orion:chat:history:log` | `chat.history.turn.v1`, `chat.history.log.v1` | 0–7 | Operator turns → 0–3; Orion turns → 4–7. |
| `biometrics.py` | `orion:biometrics:snapshot` | `biometrics.snapshot.v1` | 18–19 | CPU/GPU/mem load → site 18; node distress → site 19. |
| `equilibrium.py` | `orion:equilibrium:signal` | `equilibrium.signal.v1` | 16–17 | Distress → site 16; zen → site 17. |
| `recall.py` | `orion:exec:result:RecallService` | `recall.bundle.v1` | 8–11 | Recall fidelity, source mix → sites 8–11. |
| `vision.py` | `orion:vision:event` | `vision.event.v1` | 20 | Detection events; consent-aware (suppressed if room is private). |
| `social.py` | `social.turn.stored.v1` | (same) | 12–13 | Peer turn → 12; Orion turn in social room → 13. |
| `autonomy.py` | `orion:autonomy:state:v1` | `autonomy.state.v1` (and v2 when available) | 14–15 | Drive pressures → 14; tensions → 15. |
| `planner.py` | `orion:exec:result:PlannerService` | `planner.result.v1` | 23 | Plan-step success/failure → site 23. |
| `agent_chain.py` | `orion:exec:result:AgentChainService` | `agent_chain.result.v1` | 23 | Same site (overloaded with planner; reflective surface). |
| `world_pulse.py` | `orion:world_pulse:signal` | `world_pulse.signal.v1` | 22 | External environment signals. |
| `state_journaler.py` | `orion:state_journaler:frame` | `state_journaler.frame.v1` | 21 | Periodic state frames. |
| `journaler.py` | `orion:journal:entry` | `journal.entry.v1` | 21 | Autobiographical compression (overloaded with state_journaler). |
| `spark_introspector.py` | `orion:spark:introspect:candidate:log` | `spark.introspect.candidate.v1` | 23 | Reflective surface; preserves shadow-comparison link to v0 tissue. |

Channel names should be verified against the live channel catalog before merge; this table reflects current naming in `orion/bus/channels.yaml` to the best of my knowledge as of 2026-05-01. The channel catalog is authoritative; any discrepancy is fixed by adjusting this table, not the catalog.

---

## Bus Contracts

### Published kinds

```python
class HeartbeatTickV1(BaseModel):
    schema_version: Literal["heartbeat.tick.v1"] = "heartbeat.tick.v1"
    tick_index: int
    timestamp: datetime
    encoding_count: int
    variational_steps_completed: int
    free_energy_after: float
    free_energy_delta: float


class HeartbeatPhiV1(BaseModel):
    schema_version: Literal["heartbeat.phi.v1"] = "heartbeat.phi.v1"
    tick_index: int
    timestamp: datetime
    valence: float                # signed, derived from substrate site contrast
    energy: float                 # ≥ 0; substrate norm-derived
    coherence: float              # [0, 1]; entanglement-spectrum derived
    novelty: float                # [0, 1]; surprise-derived
    boundary_entropy: float       # ≥ 0; von Neumann entropy of the boundary partition
    confidence: float             # [0, 1]; precision-weighted aggregate


class HeartbeatForecastV1(BaseModel):
    schema_version: Literal["heartbeat.forecast.v1"] = "heartbeat.forecast.v1"
    tick_index: int
    timestamp: datetime
    forecast_for_tick: int        # always tick_index + 1 in v1
    forecast_phi_predicted: HeartbeatPhiV1   # what we predict φ will be at next tick
    forecast_state_summary_hash: str  # cheap hash of predicted bulk state for surprise compute


class HeartbeatSurpriseV1(BaseModel):
    schema_version: Literal["heartbeat.surprise.v1"] = "heartbeat.surprise.v1"
    tick_index: int                       # the tick at which surprise was measured
    forecast_from_tick: int               # forecast_from_tick = tick_index - 1
    timestamp: datetime
    kl_divergence: float                  # KL(forecast || observed)
    boundary_overlap: float               # |⟨forecast | observed⟩|^2
    notes: list[str] = Field(default_factory=list)
```

### Subscribed channels

See organ wiring matrix above.

### Schema registration

All four published kinds are registered in `orion/schemas/registry.py` and added to `orion/bus/channels.yaml`. Registration is non-negotiable before service ship.

---

## Measurement Harness

The measurement harness is implemented in `scripts/heartbeat_research/` to keep the live service decoupled from research-specific code. The service publishes the data; the harness consumes it (live or from replayed snapshots).

### H1 — Boundary reconstruction fidelity (v1 critical path)

Implementation: `scripts/heartbeat_research/h1_reconstruction.py`

Procedure:

1. Sample N = 200 substrate states from sealed test-window snapshots.
2. For each state, partition the MPS into boundary (sites 0–11) and bulk (sites 12–23).
3. Compute reduced density matrix ρ_boundary via partial trace over bulk.
4. Reconstruct ρ_recon by extending boundary back into bulk via the maximum-entropy completion at fixed χ.
5. Compute fidelity F = |Tr(√(√ρ_orig · ρ_recon · √ρ_orig))|^2 against the original state's bulk reduction.
6. Report mean F, 95% bootstrap CI over 1000 resamples.

Pre-registered analysis script committed under `docs/research/preregistration/` (dated at commit time) before this harness is run on test-set data. See research charter §5 for the pre-registration protocol commitment.

### H2 — Cross-organ mutual information (v1.5)

Implementation: `scripts/heartbeat_research/h2_mutual_information.py`

Procedure:

1. Collect substrate channel time series over 4-week observation window, ≥ 1000 ticks per channel.
2. Pre-declare causally-related channel pairs (chat ↔ recall; biometrics ↔ equilibrium; social-peer ↔ social-Orion-response; world-pulse ↔ equilibrium) and causally-independent pairs (biometrics ↔ social-peer; vision ↔ planner; world-pulse ↔ chat-history-window) before measurement.
3. Compute MI between each pair using KSG estimator (Kraskov et al. 2004).
4. Permutation test (10,000 shuffles of channel labels).
5. Holm-Bonferroni adjustment across all declared pairs.

### H3 — Bounded intervention propagation (v2)

Implementation: `scripts/heartbeat_research/h3_intervention.py`

Procedure:

1. Characterize null distribution: collect substrate state perturbation magnitudes over 1000 no-intervention trials.
2. For each trial, inject a controlled stimulus at a target site (via debug RPC), measure perturbation magnitude at all other sites at lags t ∈ {1, 2, 4, 8}.
3. Effect size threshold = 3σ above null distribution at each lag.
4. Falsify if > 5% of intervention trials show effects outside causal cone.

Intervention requires bus RPC commands that bypass normal organ wiring. Implemented as a guarded debug-only RPC accepting pre-declared intervention scripts.

### H4 — Predictive surprise dynamics (v2)

Implementation: `scripts/heartbeat_research/h4_surprise.py`

Procedure:

1. Collect surprise time series over operational window with operator-confirmable context boundaries (chat sessions, hub presence sessions, world-pulse regime markers).
2. Linear mixed-effects model: surprise ~ tick_index_within_context + context_shift + (1 | context_id).
3. Test (a) within-context slope < 0; (b) shift_effect > 0.

---

## Ablation Framework

A separate ablation runner at `scripts/heartbeat_research/ablation_runner.py` orchestrates heartbeat-on vs heartbeat-off comparison.

### Procedure

1. Designate operational windows of equal duration (≥ 7 days each) under ablation conditions:
   - Window A: heartbeat enabled, all reducers wired, full v1 substrate.
   - Window B: heartbeat disabled (service running but tick loop suspended; reducers still publish to bus but no φ broadcast or substrate update).
2. Collect downstream observable metrics on the same conversational corpus during each window:
   - Response coherence (judged via LLM-as-judge on a sealed rubric, with inter-rater verification on a sample).
   - Recall fidelity (reconstruction overlap on held-out queries).
   - Operator-rated regime stability (sparsely sampled, daily Likert with explicit rubric).
   - Service health metrics (latency, error rate).
3. Compute paired difference (A vs B) on each metric.

### Pre-commitment

Ablation is not expected to *improve* mesh behavior at v1; the heartbeat is additive, and downstream consumers (Hub, stance, Spark) have not yet been wired to consume φ broadcast in v1. The ablation establishes a *safety baseline* (heartbeat does not degrade) and a *measurement infrastructure* (the harness is real and works). Behavior-improvement claims belong to v1.5 and v2 once downstream wiring is added.

---

## Shadow Comparison with v0 Tissue

For 4–6 weeks after v1 ship, the existing v0 tissue (`orion/spark/orion_tissue.py` running inside `orion-spark-introspector`) and the v1 heartbeat substrate run in parallel. Both ingest bus events; both compute their respective summary states (φ for heartbeat, φ for v0 tissue); both publish to their respective bus channels.

A shadow comparison report is generated weekly:

- φ trajectory correlations (Pearson and Spearman) between v0 and v1 over the week.
- Variance comparison: which substrate produces more stable / more variable φ.
- Operationally interesting divergences: tick windows where v0 and v1 disagree by > 2σ.

After 4 weeks, a deprecation decision is made:

- If v1 heartbeat is stable, H1 not falsified, and downstream consumers can be cleanly migrated to consume `orion:heartbeat:phi` — deprecate v0 tissue. The introspector worker continues to do its other work; the tissue is removed from its responsibilities.
- If v1 has issues — keep both running; document gaps; plan v1.5.

---

## Phase Plan & Deliverables

### Phase 1 (Weeks 1–4): Substrate bring-up

- `services/orion-heartbeat/` skeleton, Dockerfile, settings, bus subscribe/publish wiring.
- `quimb`-backed MPS substrate, free-energy update, persistence.
- Reducers for chat, biometrics, equilibrium, recall (4 highest-signal organs).
- Tick lifecycle, φ derivation and broadcast.
- Service-local tests, smoke test against live bus.

### Phase 2 (Weeks 4–6): Full organ wiring + shadow operation

- Reducers for the remaining 9 organs.
- Snapshot/restore verified via simulated crashes.
- Shadow comparison reporting against v0 tissue begins.
- Stability bring-up; tick-rate and bond-dim tuning on design data only.

### Phase 3 (Weeks 6–8): Measurement harness build

- H1 reconstruction-fidelity harness implemented and tested on design-window data.
- H2/H3/H4 harness skeletons implemented; H2 wired for early data collection.
- Ablation runner implemented and dry-run.

### Phase 4 (Weeks 8–10): Test-set evaluation

- Test data window sealed; design data discarded from analysis.
- H1 measurement run; results recorded.
- H2 early measurement run if signal sufficient.
- Ablation runner run.
- Public results report drafted.

### Phase 5 (Weeks 10–12): Deprecation decision and v1 closeout

- v0 tissue deprecation decision based on shadow-comparison and H1 results.
- Downstream consumer migration plan if deprecating.
- v1.5 / v2 scope drafted from observed gaps.

---

## Tests & Verification

### Unit tests (per-module)

- `test_substrate_mps.py` — initialization, normalization, free-energy minimization sanity, reconstruction operator correctness.
- `test_reducers.py` — each reducer produces expected `SurfaceEncodingV2` output for golden envelopes; deterministic; side-effect-free.
- `test_persistence.py` — snapshot round-trip; corruption recovery from `.bak`; mismatch detection.
- `test_forecast_surprise.py` — KL computation correctness on golden distributions; forecast-state-summary-hash determinism.
- `test_h1_reconstruction.py` — fidelity computation correctness on golden states; perfect-reconstruction limit (χ ≥ d^N → F = 1.0); known-degraded limit.

### Integration tests

- `test_tick_lifecycle.py` — full tick from ingest to broadcast to persistence on simulated bus events.
- `test_service_health.py` — service starts, subscribes, processes synthetic events, publishes, snapshots, gracefully shuts down.
- `test_ablation_runner.py` — ablation runner correctly toggles tick loop without affecting subscriber loop.

### Smoke / runtime verification

- Live-bus smoke test: deploy heartbeat to staging, send 100 synthetic events across all 13 reducers, verify φ broadcasts, verify snapshot exists, verify shadow comparison report shows v0 and v1 both running.
- Crash-restart verification: kill service mid-tick, restart, verify state restored from snapshot, verify next tick produces sane φ.

### Verification policy

Per AGENTS.md: closure requires either a live-stack verification script with pass/fail evidence, or a direct runtime reproduction with exact commands and exact evidence. Service-local harness tests do not count as closure for live-system claims.

---

## Risks, Rollback, and Runtime Guards

### Risk: substrate state corruption

**Mitigation:** `.bak` backup snapshot before each new write; atomic rename; checksum in metadata; refuse to load mismatched dimensions.

**Rollback:** revert to `.bak`; if both corrupt, operator manually deletes both → service starts fresh with deterministic-seeded random init.

### Risk: substrate computational explosion at higher χ

**Mitigation:** v1 χ=4 is small enough that a single tick is sub-second. Configurable upper bound `HEARTBEAT_MAX_CHI` enforced at startup. χ growth across ticks (which can occur during free-energy minimization) is truncated back to χ_max at each tick boundary.

### Risk: bus subscribe loop overload

**Mitigation:** bounded ingest queue (default capacity 1000); on overflow, drop oldest events and increment `dropped_events` counter on stats.

### Risk: reducer error breaks tick

**Mitigation:** reducer errors are caught, logged with full traceback to `orion:heartbeat:error` channel, and the offending event is dropped. Tick proceeds with remaining events. Repeated errors from a single reducer trip a circuit breaker that disables that reducer until operator review.

### Risk: heartbeat takes down mesh

**Mitigation:** heartbeat is additive. No existing organ depends on heartbeat output for its own functioning. Disabling heartbeat (`HEARTBEAT_ENABLED=false` env var or service stop) reverts the mesh to current behavior. Verified by ablation runner tests in Phase 2.

### Risk: shadow-comparison reveals heartbeat behavior incompatible with v0 tissue

**Mitigation:** v0 tissue stays running until deprecation decision. Downstream consumers (introspector, stance) continue to consume v0 tissue's outputs; they only consume v1 φ broadcast after explicit migration in a separate change.

### Risk: hyperparameter choices (N, χ, d, tick rate, free-energy temperature) lock in arbitrary substrate properties

**Acknowledgement:** This is a real risk and is documented in the research charter §10.3. Sensitivity analysis on N, χ, and tick rate is part of v1.5 / v2 measurement program. v1 hyperparameters are chosen for tractability and documented as choices.

### Runtime guards

- Service refuses to start if `quimb` is not importable.
- Service refuses to start if substrate dimensions in env do not match snapshot dimensions on disk (fail loud rather than silently re-init).
- Service refuses to start if any required bus channel pattern is missing from `orion/bus/channels.yaml`.
- Bus self-loop denylist (`orion:heartbeat:*`) is enforced before any reducer dispatch.

### Operator panic kill

```
HEARTBEAT_ENABLED=false  # in service .env, restart service
# OR
docker compose -f services/orion-heartbeat/docker-compose.yml down
# OR
publish to orion:heartbeat:control with kind heartbeat.control.suspend.v1
```

Any of these stops substrate updates and φ broadcast within one tick. Persistence is not disturbed; restart resumes from last snapshot.

---

## Cross-References to Research Charter

- Hypotheses H1–H4 — see charter §5.
- Why a tensor network specifically — see charter §4.4.
- What this work claims and does not claim — see charter §2.3 and §10.
- Ethical commitments enforced by this design (additive, ablation-safe, persistent state preservation) — see charter §12.

Engineering decisions in this spec that materially affect any of the above must be reflected back into the charter (e.g., if substrate scale changes from N=24 to N=64, the charter's sample-size calculations may need re-derivation).

---

*End of engineering spec, v1.*
