# Design: Orion Organ Signal Gateway

**Date:** 2026-05-01
**Status:** Draft — pending user review
**Scope:** New service `services/orion-signal-gateway/` + `orion/signals/` shared library + OTEL collector sidecar config + Hub inspect endpoint designs

---

## What this is

A signal normalization gateway that sits between raw organ telemetry and every downstream consumer of Orion's internal state. Every organ in the mesh produces something — but what it produces ranges from raw hardware metrics to structured state objects to request-response results. This design defines:

1. A common signal schema (`OrionSignalV1`) with a dimensional representation that every organ will eventually emit
2. A causal DAG registry mapping which organs' signals flow into which others
3. A gateway service that adapts raw organ bus events into hardened signals
4. An OTEL integration design that replaces ad-hoc per-organ correlation IDs with unified trace propagation
5. Hub inspect surface designs for signal observability

## What this is NOT

- Not an implementation of signal hardening for each organ. That is the next phase. This spec defines the framework, schema, registry, and adapter contracts. Each organ's adapter is specified but not implemented.
- Not a replacement for any existing organ service. The gateway is read-only against all organs. Ablation = disable the gateway. Every existing organ continues working unchanged.
- Not a wearable biosignal implementation. The OTEL custom receiver for wearables is designed here but implemented next phase.
- Not a Hub UI implementation. The Hub inspect endpoints are designed here but not built this phase.
- Not a new persistence layer. The gateway maintains an in-memory signal window; nothing is written to the graph or SQL.

---

## Relationship to Phase 2 (errata pointer)

Adapter implementations, Hub inspect routes, milestone sequencing, and **first-pass** production contracts (multi-instance model A/B, bus semantics, canonical test layout, Hub trace cache constants, OTEL dimension allowlist, deterministic **`signal_id`** as **64-hex** full SHA-256 when preimage-based, `world_pulse` / `spark_introspector` registry alignment) are specified in **[Organ Signal Gateway — Phase 2](2026-05-01-organ-signal-gateway-phase-2-design.md)** and the **[offboarding / inventory guide](../guides/2026-05-01-organ-signal-gateway-offboarding.md)**. Where Phase 2 explicitly closes contradictions (e.g. file map, test roots), treat Phase 2 as superseding ambiguous or stale rows in this document for implementation and CI layout.

---

## Background: the multicollinearity problem is the integration surface

Orion's organs are causally chained. GPU usage telemetry flows through equilibrium into collapse mirror metacog processing, which feeds journals, which produces recall events, which shape chat stance, which influences recall output, which feeds back into the next chat turn. Feeding all of these as independent inputs to a downstream tensor network (e.g., the heartbeat substrate) would be treating the same causal event as six independent signals.

This is not simply a statistical problem to be corrected. In frameworks like IIT, the irreducibility of correlations across partitions is the measure of integration — the multicollinearity IS the integration surface. The design goal is not to decorrelate organs but to **explicitly model which correlations are causal and which are structural**, so that:

- A single GPU spike propagating through a known 6-hop causal chain is attributed as one integrated event, not six coincidences
- The heartbeat substrate's site topology can mirror the actual causal DAG rather than treating causally chained channels as independent inputs
- The operator can trace lineage: "this chat stance posture was shaped by an infra event 3 hops back"

The causal DAG registry and OTEL trace propagation are the two mechanisms that make this tractable.

---

## Architecture

### Overview

```
Bus (existing organ channels)
  → orion-signal-gateway
      → per-organ adapter  (from orion/signals/adapters/)
      → OrionSignalV1      (normalized, dimensioned, OTEL-traced)
  → bus signals.* channel namespace
      → orion-hub          (signal inspect surfaces)
      → orion-heartbeat    (substrate inputs)
      → orion-cortex-exec  (evidence refs for autonomy reducer)
      → research harness   (measurement pipelines)
```

The gateway subscribes to existing organ bus channels. It applies per-organ adapters from the shared `orion/signals/` library to transform raw events into `OrionSignalV1` records, then re-emits those on the `signals.*` channel namespace.

The gateway is **read-only against organ services** — it never writes back to them, never modifies their channels, and never sits in their request path.

### Self-hardening graduation path

An organ that wants to self-harden simply begins emitting `OrionSignalV1` directly to `signals.*`. The gateway detects signals on that channel, validates them against the schema and registry entry, and re-emits without adapting. Organs graduate out of the gateway one by one; no coordination is required.

### Shared library: `orion/signals/`

Adapter logic does not live inside the gateway service — it lives in a shared library so that any service (Hub, heartbeat, autonomy) can import the contracts.

```
orion/signals/
  __init__.py
  models.py           ← OrionSignalV1, OrionOrganRegistryEntry, OrganClass
  registry.py         ← static causal DAG registry, all organ entries
  normalization.py    ← EwmaBand, clamp01, NormalizationContext, dimensional helpers
  adapters/
    __init__.py
    base.py           ← OrionSignalAdapter ABC
    biometrics.py     ← reference implementation
    collapse_mirror.py
    equilibrium.py
    recall.py
    spark.py
    autonomy.py
    world_pulse.py
    social_memory.py
    social_room_bridge.py
    vision.py
    agent_chain.py
    planner.py
    dream.py
    state_journaler.py
    topic_foundry.py
    concept_induction.py
    graph_cognition.py
    chat_stance.py
    journaler.py
    power_guard.py
    security_watcher.py
```

### Service layout: `services/orion-signal-gateway/`

```
services/orion-signal-gateway/
  Dockerfile
  docker-compose.yml
  requirements.txt
  .env_example
  README.md
  otel/
    collector-config.yaml     ← OTEL collector sidecar configuration
  app/
    __init__.py
    main.py
    settings.py
    service.py                ← gateway bus chassis
    processor.py              ← signal processing loop
    normalization_state.py    ← per-organ NormalizationContext registry
    signal_window.py          ← in-memory recent signals by organ_id
    passthrough.py            ← validator for self-hardened organ signals
    tests/
      test_adapter_biometrics.py
      test_adapter_equilibrium.py
      test_causal_parent_resolution.py
      test_passthrough_validation.py
      test_otel_propagation.py
```

---

## 1. Signal schema (`orion/signals/models.py`)

### `OrganClass`

```python
class OrganClass(str, Enum):
    exogenous  = "exogenous"   # root signal: hardware, user input, environment
    endogenous = "endogenous"  # derived from other organs' signals
    hybrid     = "hybrid"      # partially derived, partially independent
```

The exogenous/endogenous distinction is the mechanism that resolves the multicollinearity concern for downstream consumers. Exogenous signals are independent inputs. Endogenous signals are the integration surface — their correlations with exogenous parents are the measure of integration, not confounds to suppress.

### `OrionSignalV1`

```python
class OrionSignalV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Identity
    signal_id: str
    # Deterministic when source_event_id is present:
    #   hashlib.sha256(f"{organ_id}:{source_event_id}".encode()).hexdigest()[:16]
    # If source_event_id is None, fall back to str(uuid4()) — not deterministic,
    # but avoids collision when the source event carries no stable ID.

    organ_id: str          # e.g. "biometrics", "collapse_mirror", "recall"
    organ_class: OrganClass

    # Signal type
    signal_kind: str
    # Organ-specific. Canonical values defined per organ in the registry.
    # Examples: "gpu_load", "cognitive_collapse", "recall_result", "chat_stance"

    # Dimensional representation — the hardened signal
    dimensions: dict[str, float]
    # Keys follow conventions across organs:
    #   level:       current intensity              [0.0, 1.0]
    #   trend:       direction of change            [-1.0, 1.0]
    #   volatility:  rate of change                 [0.0, 1.0]
    #   valence:     positive/negative charge       [-1.0, 1.0]  (where applicable)
    #   confidence:  signal reliability             [0.0, 1.0]
    #   arousal:     activation level               [0.0, 1.0]  (spark/affect signals)
    #   coherence:   internal consistency           [0.0, 1.0]
    #   novelty:     deviation from baseline        [0.0, 1.0]
    #   salience:    attentional weight             [0.0, 1.0]
    # Per-drive pressure keys (autonomy organ only):
    #   pressure_coherence, pressure_continuity, pressure_relational,
    #   pressure_autonomy, pressure_capability, pressure_predictive   [0.0, 1.0]

    # Causal provenance
    causal_parents: list[str] = []
    # signal_ids of OrionSignalV1 records this was derived from.
    # Populated by adapters using the gateway's prior_signals window.
    # Empty for exogenous signals.

    source_event_id: str | None = None
    # Original bus event ID / correlation_id from the source organ.
    # Preserved as otel span attribute for migration compatibility.

    # OTEL trace context
    otel_trace_id:       str | None = None
    otel_span_id:        str | None = None
    otel_parent_span_id: str | None = None
    # otel_trace_id propagates across the causal chain:
    # exogenous signals start a new trace; endogenous signals inherit
    # the trace_id of their causal_parents (first parent's trace_id wins
    # if parents disagree — rare, logged as a note).

    # Temporal
    observed_at: datetime   # when the source event occurred
    emitted_at:  datetime   # when the gateway produced this signal
    ttl_ms: int = 15_000

    # Human-readable audit
    summary: str | None = None
    notes:   list[str] = []     # max 5; gateway may append adapter warnings
```

### `OrionOrganRegistryEntry`

```python
class OrionOrganRegistryEntry(BaseModel):
    organ_id: str
    organ_class: OrganClass
    service: str                      # e.g. "orion-biometrics"
    signal_kinds: list[str]           # canonical signal_kind values this organ emits
    canonical_dimensions: list[str]   # dimension keys this organ populates
    causal_parent_organs: list[str]   # organ_ids structurally upstream in the causal DAG
    bus_channels: list[str]           # bus channels the gateway subscribes to for this organ
    notes: list[str] = []
```

---

## 2. Causal DAG registry (`orion/signals/registry.py`)

The registry is a static dict `ORGAN_REGISTRY: dict[str, OrionOrganRegistryEntry]` imported by adapters, the gateway, and any downstream consumer.

### Full organ inventory

| organ_id | organ_class | service | signal_kinds | canonical_dimensions | causal_parent_organs |
|---|---|---|---|---|---|
| `biometrics` | exogenous | orion-biometrics | gpu_load, cpu_load, memory_pressure, thermal_state, network_load, disk_io, power_state | level, trend, volatility, confidence | — |
| `vision` | exogenous | orion-vision-council | scene_state, person_detected, object_event, visual_context | level, valence, novelty, confidence | — |
| `social_room_bridge` | exogenous | orion-social-room-bridge | social_turn, room_event | level, valence, confidence | — |
| `power_guard` | exogenous | orion-power-guard | power_state, ups_event | level, trend, confidence | — |
| `security_watcher` | exogenous | orion-security-watcher | security_event | level, confidence | — |
| `equilibrium` | hybrid | orion-equilibrium-service | mesh_health, service_distress, zen_state | level, trend, confidence | biometrics |
| `world_pulse` | hybrid | orion-world-pulse | situation_state, time_context, environmental_context | level, valence, confidence | *(external feeds, chat turn context — see note)* |
| `social_memory` | hybrid | orion-social-memory | social_bond_state, relationship_continuity, social_repair_event | level, valence, trend | social_room_bridge |
| `collapse_mirror` | endogenous | orion-collapse-mirror | cognitive_collapse, metacog_event | level, valence, confidence | biometrics, equilibrium |
| `recall` | endogenous | orion-recall | recall_result, recall_gap, recall_quality | level, trend, confidence | autonomy, social_memory |
| `concept_induction` | endogenous | orion-spark-concept-induction | concept_salience, topic_formation, concept_drift | salience, novelty, confidence | chat_stance, recall, vision |
| `spark_introspector` | endogenous | orion-spark-introspector | phi_field, tissue_state, spark_signal | level, valence, arousal, coherence, novelty | biometrics, equilibrium, recall, collapse_mirror, vision |
| `graph_cognition` | endogenous | orion-cortex-exec (library) | metacog_perception, coherence_state, goal_pressure | coherence, tension, goal_pressure, confidence | social_memory, recall |
| `autonomy` | endogenous | orion-cortex-exec | autonomy_state, drive_pressure, tension_state | pressure_coherence, pressure_continuity, pressure_relational, pressure_autonomy, pressure_capability, pressure_predictive, confidence | graph_cognition |
| `chat_stance` | endogenous | orion-cortex-exec | chat_stance, turn_effect, metacog_residue | coherence, valence, confidence | recall, autonomy, equilibrium, social_memory, spark_introspector |
| `journaler` | endogenous | orion library (`orion/journaler/`) — bus events emitted from cortex-exec | journal_entry, recall_event | level, novelty, valence | collapse_mirror, chat_stance, agent_chain |
| `state_journaler` | endogenous | orion-state-journaler | state_frame, state_transition | level, coherence, confidence | autonomy, equilibrium, recall |
| `dream` | endogenous | orion-dream | dream_cycle_output, memory_consolidation | level, coherence, novelty | recall, social_memory, state_journaler |
| `agent_chain` | endogenous | orion-agent-chain | action_outcome, tool_execution, capability_event | level, success, surprise | planner, autonomy, chat_stance |
| `planner` | endogenous | orion-planner-react | plan_state, goal_progress | level, confidence, surprise | autonomy, agent_chain |
| `topic_foundry` | endogenous | orion-topic-foundry | topic_state, topic_drift | salience, novelty, coherence | concept_induction, chat_stance |

**Registry accuracy note.** The `causal_parent_organs` entries above are first-pass structural approximations derived from code inspection. The implementation phase must verify each organ's actual bus channel subscriptions and cross-check against the adapters before treating the registry as authoritative. `bus_channels` (which channels the gateway subscribes to per organ) are also to be specified per-organ during implementation; they are omitted from the table above for readability.

### The primary causal chain (explicitly modeled)

The canonical example chain is modeled structurally as:

```
biometrics [exogenous]
  → equilibrium [hybrid]
    → collapse_mirror [endogenous]
      → journaler [endogenous]
        → recall [endogenous]    ← also receives: autonomy, social_memory
          → chat_stance [endogenous]   ← also receives: autonomy, equilibrium, social_memory, spark
            → (chat turn output)
              → recall [next turn]   ← temporal feedback loop
```

**Temporal feedback loops.** The chat_stance → recall dependency is real but not a cycle in the static DAG — it operates across time steps. Within any single OTEL trace, the chain is acyclic: the GPU spike at t=0 starts a trace; collapse at t=1 is a child span; recall degradation at t=2 is a grandchild. Temporal ordering resolves what looks like circularity. The static registry records the structural relationship; OTEL records the temporal one.

---

## 3. Adapter design (`orion/signals/adapters/`)

### Base class

```python
class OrionSignalAdapter(ABC):
    organ_id: ClassVar[str]

    @abstractmethod
    def can_handle(self, channel: str, payload: dict) -> bool:
        """Return True if this adapter should process this bus event."""

    @abstractmethod
    def adapt(
        self,
        channel: str,
        payload: dict,
        registry: dict[str, OrionOrganRegistryEntry],
        prior_signals: dict[str, OrionSignalV1],
        norm_ctx: NormalizationContext,
    ) -> OrionSignalV1 | None:
        """Transform a raw bus event into a hardened signal. Return None to drop."""
```

### Design rules for all adapters

**`prior_signals` for causal parent resolution.** The gateway passes its recent signal window (most recent `OrionSignalV1` per organ_id, bounded to 30 seconds). Endogenous adapters use this to populate `causal_parents`. Example: the collapse_mirror adapter looks up `prior_signals.get("biometrics")` and `prior_signals.get("equilibrium")` to find their signal_ids.

**`NormalizationContext` for EWMA state.** Adapters do not hold their own state. EWMA bands, volatility trackers, and spike detectors live in `NormalizationContext`, which is owned by the gateway and passed into each `adapt()` call. This keeps adapters pure and independently testable.

**Adapters are stateless.** Side effects (logging, metrics) are the only acceptable exception.

**Graceful degradation.** If an adapter cannot extract a required field from a payload, it returns a signal with `confidence = 0.1` and a descriptive `notes` entry rather than raising or returning None. Dropping is reserved for genuinely irrelevant events (wrong channel, wrong payload type).

**OTEL parent context.** Endogenous adapters extract `otel_trace_id` and `otel_span_id` from the first entry in `causal_parents` (resolved via `prior_signals`) and set those as the OTEL parent context for the new span. Exogenous adapters start a new trace.

### Reference implementation: biometrics adapter

The biometrics adapter wraps the existing `BiometricsInductionV1` / `InductionTracker` / `EwmaBand` logic from `orion/telemetry/biometrics_pipeline.py`. Its job is to:

1. Extract GPU/CPU/memory/thermal sub-payloads from the raw biometrics event
2. Run each through the `NormalizationContext`'s tracker for that metric key
3. Map `InductionMetricState.level`, `.trend`, `.volatility` to `dimensions`
4. Emit one `OrionSignalV1` per signal_kind (gpu_load, cpu_load, etc.) or a combined signal depending on adapter configuration

This makes the existing biometrics pipeline the template every other adapter follows for the `level/trend/volatility` dimensional convention.

### Hardest adapters

**`chat_stance` adapter.** Chat stance is assembled per-turn inside `orion-cortex-exec`, not emitted as a standalone bus event. The adapter subscribes to the cortex router output channel and extracts the autonomy/stance payload from the turn metadata. Its `causal_parents` list draws from all prior_signals within the turn window: recall, autonomy, equilibrium, social_memory, spark_introspector signal_ids.

**`recall` adapter.** Recall is request-response, not a bus event. The adapter subscribes to the recall result channel (emitted by `orion-recall` after each retrieval) and normalizes the fusion pipeline's quality and relevance scores into `level`/`trend`/`confidence` dimensions.

---

## 4. Normalization conventions (`orion/signals/normalization.py`)

All adapters share the same normalization primitives:

```python
class EwmaBand:
    """Adaptive normalization band using EWMA mean and deviation.
    Inherited directly from orion/telemetry/biometrics_pipeline.py."""
    alpha: float = 0.1
    mean: float | None = None
    dev:  float | None = None

    def update(self, value: float) -> None: ...
    def normalize(self, value: float) -> float: ...  # returns [0, 1]

def clamp01(v: float) -> float: ...
def clamp11(v: float) -> float: ...   # for trend/valence dimensions

class NormalizationContext:
    """Per-organ EWMA state, owned by the gateway, passed to adapters."""
    def get_band(self, organ_id: str, metric_key: str) -> EwmaBand: ...
    def get_tracker(self, organ_id: str, metric_key: str) -> InductionTracker: ...
```

The `EwmaBand` and `InductionTracker` implementations are moved from `orion/telemetry/biometrics_pipeline.py` into `orion/signals/normalization.py` and re-exported from the telemetry module for backward compatibility.

---

## 5. OTEL integration

### What OTEL replaces

Current state: each organ threads its own `correlation_id` through its events. Reconstructing the GPU→collapse→recall→stance chain after the fact requires joining across 4+ tables/channels by matching correlation IDs that were never designed to be the same value.

With OTEL: the gateway starts one trace when it processes an exogenous signal. Every endogenous signal derived from it inherits the same `trace_id` and creates a child span. The entire causal chain becomes one trace — queryable in any OTEL-compatible backend. Existing correlation IDs are preserved as span attributes (`correlation_id` key) during migration.

### Gateway instrumentation

```python
with tracer.start_as_current_span(
    f"signal.{organ_id}.{signal_kind}",
    context=parent_otel_context,     # from causal_parents' otel_span_ids
) as span:
    span.set_attribute("organ_id", organ_id)
    span.set_attribute("organ_class", organ_class.value)
    span.set_attribute("signal_kind", signal_kind)
    span.set_attribute("correlation_id", source_event_id or "")
    for k, v in dimensions.items():
        span.set_attribute(f"dim.{k}", v)
    signal.otel_trace_id       = _format_trace_id(span.get_span_context().trace_id)
    signal.otel_span_id        = _format_span_id(span.get_span_context().span_id)
    signal.otel_parent_span_id = parent_span_id
```

### OTEL collector sidecar (`otel/collector-config.yaml`)

The collector is a standard OpenTelemetry Collector Contrib binary — no custom code. Configuration only.

**Receivers:**

| Receiver | Signals unlocked |
|---|---|
| `otlp` | Spans and metrics from the gateway itself |
| `hostmetrics` | CPU per-core, memory pages, disk latency, network error rates — finer-grained than current biometrics `nvidia-smi` polling |
| `nvidia/dcgm` (dcgmreceiver) | GPU SM utilization, tensor core activity, memory bandwidth, NVLink — richer GPU telemetry |
| `prometheus` | Any existing Prometheus endpoints exposed by mesh services |
| Custom wearable receiver | (Designed here, implemented next phase) — bridge service consuming HRV/GSR/HR from wearable API, emitting OTEL metrics |

**Exporters:** Prometheus (for Grafana), OTLP gRPC (for Jaeger/Tempo if deployed), logging exporter for development.

**Wearable biosignal receiver design (next-phase implementation):**
A thin `orion-wearable-bridge` service polls a wearable device API (configurable endpoint, e.g., Garmin Connect, Apple HealthKit via companion app, or direct BLE) and emits OTEL metrics:
- `orion.biosignal.hr` — heart rate (bpm)
- `orion.biosignal.hrv_rmssd` — HRV (ms)
- `orion.biosignal.gsr` — galvanic skin response (µS, if available)
- `orion.biosignal.spo2` — blood oxygen (%, if available)

These feed into the biometrics adapter as additional `signal_kind` values (`biometric_hr`, `biometric_hrv`, etc.) with the same `level/trend/volatility` dimensional convention.

---

## 6. Hub inspect surfaces (designed, not implemented this phase)

### `GET /api/signals/active`

Returns the most recent `OrionSignalV1` per organ_id. The Hub maintains an in-memory window (latest per organ_id) by subscribing to `signals.*`. No database query.

Response shape:
```json
{
  "as_of": "2026-05-01T21:00:00Z",
  "signals": {
    "biometrics": { /* OrionSignalV1 */ },
    "collapse_mirror": { /* OrionSignalV1 */ },
    ...
  }
}
```

### `GET /api/signals/trace/{trace_id}`

Returns all signals sharing an OTEL trace_id, ordered by `observed_at`. The Hub maintains a rolling trace cache (last N traces by trace_id). Gives the operator a full causal chain explorer: hand it a trace_id from any signal and see the complete GPU→collapse→recall→stance lineage.

Response shape:
```json
{
  "trace_id": "...",
  "chain": [
    { "organ_id": "biometrics", "signal_kind": "gpu_load", "observed_at": "...", "dimensions": {...} },
    { "organ_id": "equilibrium", "signal_kind": "mesh_health", "observed_at": "...", "causal_parents": ["..."] },
    ...
  ]
}
```

---

## 7. Testing

### A. Adapter unit tests (`orion/signals/adapters/tests/`)

- Each adapter: given a representative raw payload, produces a valid `OrionSignalV1` with correct `organ_id`, `organ_class`, `signal_kind`, and non-empty `dimensions`.
- `biometrics` adapter: EWMA-normalized `level` is in [0, 1]; `trend` is in [-1, 1]; repeated identical input converges to stable level.
- `collapse_mirror` adapter: when `prior_signals` contains a recent biometrics signal, `causal_parents` contains that signal's `signal_id`.
- Graceful degradation: malformed payload produces a signal with `confidence = 0.1` and a `notes` entry, not an exception.
- Biometrics adapter is the reference; its test suite is the template for all others.

### B. Causal parent resolution tests

- Gateway with two sequential events (biometrics → collapse) within the 30-second window: collapse signal's `causal_parents` contains biometrics signal_id.
- Events outside the 30-second window: `causal_parents` is empty; a `notes` entry records the missed link.
- OTEL trace_id propagates: biometrics signal and its derived collapse signal share `otel_trace_id`.

### C. Pass-through validation tests

- A valid `OrionSignalV1` on `signals.*` passes through without re-adapting.
- A signal with an unknown `organ_id` (not in registry) is rejected with a warning logged; not re-emitted.
- A signal with missing required fields is rejected; not re-emitted.

### D. Normalization context tests

- `EwmaBand` with `alpha=0.1`: after 50 identical inputs, `normalize(value)` returns a stable value near 0.5.
- `clamp01` and `clamp11` enforce range bounds.
- `NormalizationContext` returns the same band instance for repeated (organ_id, metric_key) lookups.

---

## 8. Known limitations

1. **Chat stance and recall adapters require cortex cooperation.** Chat stance is assembled per-turn inside `orion-cortex-exec` and not emitted as a standalone bus event. The adapter subscribes to the cortex router output channel. If the router payload does not carry the expected fields, the adapter degrades gracefully (low-confidence signal with notes).

2. **`causal_parents` resolution is time-bounded.** The gateway's `prior_signals` window is 30 seconds. If an endogenous event arrives more than 30 seconds after its causal parent, the link is lost and `causal_parents` is empty. This is a known approximation; the 30-second default is configurable.

3. **Temporal feedback loops are not represented in the static DAG.** The registry records structural relationships. The `chat_stance → recall` dependency on the next turn creates a real feedback loop that the static DAG cannot represent. OTEL trace ordering handles the temporal dimension; the registry handles the structural one.

4. **OTEL wearable biosignal receiver is designed but not implemented.** The `orion-wearable-bridge` service and the `orion.biosignal.*` OTEL metric definitions are specified here. Implementation is a next-phase item dependent on wearable hardware selection.

5. **Self-hardening graduation is not enforced.** Nothing prevents an organ from emitting both its raw event on its existing channel and an `OrionSignalV1` on `signals.*`. The gateway processes both. Operators should track graduation status via the registry's `notes` field per organ.

6. **Exogenous/endogenous classification is static.** The registry classification is a structural judgment, not a dynamic measurement. An organ re-classified over time (e.g., `world_pulse` gaining more exogenous inputs) requires a registry update and a gateway redeploy.

---

## Out of scope (next phase)

- Per-organ adapter implementation (each organ's adapter is specified above; code is not written this phase)
- Hub inspect endpoint implementation
- `orion-wearable-bridge` service implementation
- OTEL backend deployment (Jaeger, Grafana Tempo)
- Signal persistence layer (graph or SQL storage of historical signals)
- Signal-based alerting or policy gates
- Migration of existing correlation ID infrastructure to full OTEL adoption

---

## File map

| File | Purpose |
|---|---|
| `orion/signals/models.py` | `OrionSignalV1`, `OrionOrganRegistryEntry`, `OrganClass` |
| `orion/signals/registry.py` | `ORGAN_REGISTRY` dict, all organ entries |
| `orion/signals/normalization.py` | `EwmaBand`, `InductionTracker` (moved from telemetry), `NormalizationContext`, `clamp01`, `clamp11` |
| `orion/signals/adapters/base.py` | `OrionSignalAdapter` ABC |
| `orion/signals/adapters/*.py` | One file per organ (specified, not implemented) |
| `services/orion-signal-gateway/app/service.py` | Bus chassis, subscription management |
| `services/orion-signal-gateway/app/processor.py` | Signal processing loop, adapter dispatch |
| `services/orion-signal-gateway/app/normalization_state.py` | Per-organ `NormalizationContext` registry |
| `services/orion-signal-gateway/app/signal_window.py` | In-memory recent signals by organ_id (30s window) |
| `services/orion-signal-gateway/app/passthrough.py` | Validator for self-hardened organ signals on `signals.*` |
| `services/orion-signal-gateway/otel/collector-config.yaml` | OTEL collector sidecar configuration |
| `orion/telemetry/biometrics_pipeline.py` | Re-exports `EwmaBand`, `InductionTracker` from new location for backward compat |
