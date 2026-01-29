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

### SQL Writer
*   **Table:** `cognition_traces` (via `CognitionTraceSQL` model)
*   **Subscription:** `cognition.trace` -> `CognitionTraceSQL` mapping.
*   **Storage:** Stores full JSON of steps/options + structured metadata columns.

### RDF Writer
*   **Graph:** `orion:cognition`
*   **Ontology:**
    *   `orion:CognitionRun` (The trace)
    *   `orion:CognitionStep` (The steps)
    *   `orion:hasStep` / `orion:nextStep` (Sequence)
    *   `orion:hasThought` (Step reasoning)
    *   `orion:hasEvidenceRef` (Artifact links)

### Vector Writer
*   **Collection:** `orion_cognition`
*   **Embedding:** `final_text` (content)
*   **Metadata:** `correlation_id`, `verb`, `mode`.

### Spark Introspector
*   **Logic:**
    1.  Receives `cognition.trace`.
    2.  Derives `SurfaceEncoding` (Valence/Arousal) from trace success/fail/complexity.
    3.  Maps to `OrionTissue` stimulus.
    4.  Updates Tissue (Predictive Coding + Diffusion).
    5.  Calculates `Novelty` and `Phi`.
    6.  Publishes `spark.telemetry` (legacy `spark.introspection.log`).
