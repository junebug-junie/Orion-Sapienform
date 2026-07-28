# Contracts: Cognition Trace & Spark Integration

## Overview

This document defines the contracts for the "Cognition Trace Memory" system, which captures agentic thoughts/actions and feeds them into:
1.  **Persistence Layers:** SQL (Relational), RDF (Graph), Vector (Embedding).
2.  **Spark/Tissue:** An inner field simulation (Valence/Energy/Novelty) that evolves based on trace stimuli.

## 1. Cognition Trace Channel

*   **Channel Env Var:** `CHANNEL_COGNITION_TRACE_PUB`
*   **Default Value:** `orion:cognition:trace`
*   **Publisher:** `orion-cortex-exec` (at the end of plan execution)
*   **Envelope Kind:** `cognition.trace`
*   **Schema:** `orion.schemas.telemetry.cognition_trace.CognitionTracePayload`

### Schema Payload (`CognitionTracePayload`)

| Field | Type | Description |
| :--- | :--- | :--- |
| `correlation_id` | `UUID` | The unique ID of the conversation/trace. |
| `mode` | `str` | Execution mode (`brain`, `agent`, `council`). |
| `verb` | `str` | The high-level verb executed (e.g., `chat`, `plan`). |
| `packs` | `List[str]` | The cognitive packs active during execution. |
| `options` | `Dict` | Runtime options passed to the executor. |
| `final_text` | `str` | The final output text (if any). |
| `steps` | `List[StepExecutionResult]` | List of executed steps including thoughts, tools, and artifacts. |
| `timestamp` | `float` | Unix timestamp of trace completion. |
| `source_service` | `str` | Service producing the trace (`cortex-exec`). |
| `source_node` | `str` | Node name. |
| `recall_used` | `bool` | Whether RAG/Recall was accessed. |
| `recall_debug` | `Dict` | Debug info for recall (sources, scores). |
| `metadata` | `Dict` | Additional request context (request_id, status). |

## 2. Spark Telemetry

*   **Channel Env Var:** `CHANNEL_SPARK_TELEMETRY`
*   **Default Value:** `orion:spark:telemetry` (legacy `orion:spark:introspection:log` may still appear)
*   **Publisher:** `orion-spark-introspector` (after processing a trace)
*   **Envelope Kind:** `spark.telemetry` (legacy `spark.introspection.log` may still appear)
*   **Schema:** `orion.schemas.telemetry.spark.SparkTelemetryPayload`

**Spark contract canonical source:** The single source of truth for Spark telemetry and snapshot schemas is
`orion/schemas/telemetry/spark.py`. All services should import Spark schemas from that module (or a compatibility
shim that re-exports it). This prevents drift across duplicate definitions.

### Schema Payload (`SparkTelemetryPayload`)

| Field | Type | Description |
| :--- | :--- | :--- |
| `correlation_id` | `UUID` | Links back to the trace. |
| `phi` | `float` | The calculated "Integrated Information" or "Coherence" of the tissue. |
| `novelty` | `float` | Novelty of the trace stimulus (0.0 to 1.0). |
| `trace_mode` | `str` | Trace mode (for aggregation). |
| `trace_verb` | `str` | Trace verb (for filtering). |
| `stimulus_summary` | `str` | Debug string of the stimulus encoding (e.g., "v=0.8 a=0.5"). |
| `metadata` | `dict` | Full tissue stats (valence, energy, etc.). |

**Phi semantics:** `SparkStateSnapshotV1.phi` is a dict of components (valence/energy/coherence/novelty, etc.). If
`SparkTelemetryPayload.phi` is present, it represents a scalar coherence-style score derived from those components.

## 3. Consumer Behavior

**As of 2026-07-28, this section reflects live reality, not the original design intent** — see
`docs/superpowers/specs/2026-07-28-cognition-trace-signal-gateway-consumer-audit.md` for the
full audit. Several consumers described in earlier versions of this doc were never real or have
since been deliberately killed.

### SQL Writer — real, live, the only full-fidelity record
*   **Table:** `cognition_traces` (via `CognitionTraceSQL` model)
*   **Subscription:** `cognition.trace` -> `CognitionTraceSQL` mapping.
*   **Storage:** Stores full JSON of steps/options + structured metadata columns.
*   Live-confirmed 2026-07-28: 7393+ rows, newest seconds old at query time. This is the one
    channel of real archival value in this whole tree — do not remove without a real replacement.

### RDF Writer — killed 2026-07-22
The `orion:CognitionRun`/`orion:CognitionStep` Fuseki ontology this section used to describe was
retired as part of the Fuseki decommission campaign: live traffic was ~750 writes/6h of pure
redundancy against SQL Writer's `cognition_traces` (61k+ rows at the time). See
`services/orion-rdf-writer/app/settings.py`'s subscribe-list comment and
`docs/superpowers/specs/2026-07-22-tags-enriched-fuseki-kill-spec.md`. Do not re-add without a
real Falkor/Postgres-gap reason.

### Vector Writer — dead code, removed 2026-07-28
The `cognition.trace` -> `orion_cognition` Chroma-embedding branch this section used to describe
was never reachable: `VECTOR_WRITER_SUBSCRIBE_CHANNELS` (default and live `.env`) has never
included `orion:cognition:trace`. No envelope of this kind ever reached the branch. Removed
rather than wired up — if semantic search over past `final_text` is wanted, that's a fresh
feature to scope, not a restoration.

### Signal Gateway — removed 2026-07-28
`CognitionTraceAdapter` derived `cognition_run`/`cognition_step` `OrionSignalV1` objects (spec
§5.3) from this channel. Its only real consumer was `orion-hub`'s debug-only
`SignalsInspectCache` (a UI inspection cache, not a live cognition/drive loop).
`orion-substrate-runtime`'s `execution_trajectory` reducer already builds an equivalent,
causally-linked per-turn structure from `GrammarEventV1` (the doctrinal "substrate trace",
see `docs/context-engineering/00_substrate_trace_doctrine.md`) — and that one has a real
consumer: `orion-spark-introspector`'s `inner_state.py` HTTP-polls
`/projections/execution_trajectory` to build `InnerStateFeaturesV1`. Removed as a redundant
third encoding of the same "how did this turn go" concept.

### Spark Introspector — real subscription, but the derived signal is discarded before use
*   **Logic (`handle_trace()` in `worker.py`):**
    1.  Receives `cognition.trace`.
    2.  Computes a heuristic `valence`/`arousal` from trace success/fail counts ("Basic
        heuristics" block) — **live-confirmed this result is immediately overwritten** by
        `_get_phi_stats()`'s own internal state a few lines later, before it's ever published.
        The step-derived signal from `cognition_trace.steps` does not actually reach phi.
    3.  What does reach phi: `mode`/`verb` bookkeeping and whatever `spark_meta`/`turn_effect`
        is nested in `trace.metadata`.
    4.  Publishes `spark.telemetry` (legacy `spark.introspection.log`).
*   Separately, a synthetic `mode="heartbeat"` variant of this same channel (published by
    `orion-equilibrium-service`'s idle-keepalive loop, `EQUILIBRIUM_SPARK_HEARTBEAT_ENABLE`,
    default `false`) takes a different branch in `handle_trace()` that decays existing phi state
    rather than deriving from real trace content — unrelated to landing-pad's spark data despite
    the shared name.
*   This whole subscription's future is tracked separately in
    `docs/superpowers/specs/2026-07-28-spark-introspector-retirement-and-honest-substrate-convergence.md`
    (phi/EKG output found to be largely theater independent of this audit) — do not re-fix the
    discarded-heuristic bug in isolation; let that retirement track decide this consumer's fate.
