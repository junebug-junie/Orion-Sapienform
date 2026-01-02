# Orion Titanium Contracts

This document is the **canonical source of truth** for the Orion Titanium Contract Stack. It defines the message envelopes, taxonomy, and channel conventions that all services must adhere to.

**Core Directives:**
*   **Schema-First:** All payloads must be defined in `orion/schemas/*`. No "dict soup".
*   **Typed Envelopes:** All messages must use `BaseEnvelope` (or a subclass) with `kind`, `source`, `correlation_id`, and `causality_chain`.
*   **Explicit Routing:** Channels are configured via environment variables and `settings.py`. No hardcoded channel names in code.
*   **Caller-Owned Replies:** The caller generates a unique reply channel (e.g., `...:reply:<uuid>`) and passes it in the `reply_to` field.

---

## 1. Envelope & Correlation

All communication happens via **Redis Pub/Sub** using the `orion.core.bus` library.

**Envelope Schema:** `orion.envelope` v2.0.0
**Python Model:** `orion.core.bus.bus_schemas.BaseEnvelope`

### Key Fields

| Field | Type | Description |
| :--- | :--- | :--- |
| `schema_id` | `Literal["orion.envelope"]` | Identifies this as a Titanium envelope. |
| `schema_version` | `str` | Version of the envelope schema (e.g., "2.0.0"). |
| `id` | `UUID` | Unique message ID. |
| `correlation_id` | `UUID` | **Stable** ID for the entire request/flow. Must be preserved across service hops. |
| `kind` | `str` | Canonical message kind (e.g., `cortex.orch.request`). |
| `source` | `ServiceRef` | Identity of the producer (`name`, `node`, `version`). |
| `reply_to` | `str` (Optional) | Ephemeral channel for the response. Caller-owned. |
| `causality_chain` | `List[CausalityLink]` | Append-only lineage for debugging and tracing. |
| `payload` | `Dict` (or Typed Model) | The actual data, strictly validated against a schema. |

### The "No Dict Soup" Rule
Services must **decode and validate** payloads immediately upon receipt.
*   **Producers:** Must serialize typed Pydantic models (from `orion/schemas/*`).
*   **Consumers:** Must deserialize into the expected Pydantic model. If validation fails, the message is rejected (or dead-lettered).

---

## 2. Channel & Kind Registry

This registry maps logical flows to specific channels, message kinds, and payload schemas.

### Cortex / Cognition Runtime

| Flow | Channel (Env Var) | Kind (Req → Res) | Payload Schema |
| :--- | :--- | :--- | :--- |
| **Client → Orch** | `orion-cortex:request` | `cortex.orch.request` → `cortex.orch.result` | `orion.schemas.cortex.contracts.CortexClientRequest` |
| **Orch → Exec** | `orion-cortex-exec:request` | `cortex.exec.request` → `cortex.exec.result` | `orion.schemas.cortex.schemas.PlanExecutionRequest` |
| **Exec → LLM** | `orion-exec:request:LLMGatewayService` | `llm.chat.request` → `llm.chat.result` | `orion.schemas.chat.ChatRequestPayload` |
| **Exec → Recall** | `orion-exec:request:RecallService` | `recall.query.request` → `recall.query.result` | `orion.schemas.recall.RecallRequest` |
| **Exec → Planner** | `orion-exec:request:PlannerReactService` | `agent.planner.request` → `agent.planner.result` | `orion.schemas.agents.schemas.PlannerRequest` |
| **Exec → AgentChain**| `orion-exec:request:AgentChainService` | `agent.chain.request` → `agent.chain.result` | `orion.schemas.agents.schemas.AgentChainRequest` |
| **Exec → Council** | `orion:agent-council:intake` | `council.request` → `council.result` | `orion.schemas.agents.schemas.DeliberationRequest` |

### Ingestion & Writes

| Flow | Channel (Env Var) | Kind | Payload Schema |
| :--- | :--- | :--- | :--- |
| **Collapse Intake** | `orion:collapse:intake` | `collapse.mirror.entry` | `orion.schemas.collapse_mirror.CollapseMirrorEntry` |
| **Collapse Triage** | `orion:collapse:triage` | `collapse.mirror` | (Same as Entry) |
| **Meta Tags Output**| `orion:tags:enriched` | `tags.enriched` | `orion.schemas.telemetry.meta_tags.MetaTagsPayload` |
| **SQL Write** | `orion:collapse:sql-write` | `sql.write` (or specific kinds) | `orion.schemas.sql.schemas.SqlWriteRequest` |
| **Vector Write** | `orion:vector:write` | `vector.write` (or specific kinds) | `orion.schemas.vector.schemas.VectorWriteRequest` |
| **RDF Enqueue** | `orion:rdf-collapse:enqueue` | `rdf.write.request` | `orion.schemas.rdf.RdfWriteRequest` |
| **Biometrics** | `orion:telemetry:biometrics` | `biometrics.telemetry` | `orion.schemas.telemetry.biometrics.BiometricsPayload` |
| **Dream Trigger** | `orion:dream:trigger` | `dream.trigger` | `orion.schemas.dream.DreamRequest` |
| **TTS Request** | `orion:tts:intake` | `tts.synthesize.request` | `orion.schemas.tts.TTSRequestPayload` |

---

## 3. Mismatch Report (Known Divergences)

The following discrepancies exist between the contracts defined here and the current implementation.

### Service Behavior vs. Contract

1.  **Orion Collapse Mirror Output:**
    *   *Contract:* Expected to publish to `orion:collapse:sql-write`.
    *   *Actual:* Publishes to `orion:collapse:intake` (re-entry) and `orion:collapse:triage`. SQL Writer must subscribe to `orion:collapse:sql-write` AND `orion:collapse:triage` (or be routed correctly) to capture data.

2.  **Orion SQL Writer Config:**
    *   *Contract:* Uses explicit write channels.
    *   *Actual:* Subscribes to a list of channels (`orion:tags:enriched`, `orion:collapse:sql-write`, etc.) and routes based on `kind`. This is a valid pattern ("Hunter") but differs from a simple "Queue" model.

3.  **Biometrics Channel Name:**
    *   *Contract:* `orion:telemetry:biometrics`.
    *   *Actual:* Code defaults to `orion:biometrics:telemetry` in `settings.py`, but respects `TELEMETRY_PUBLISH_CHANNEL` env var. Ensure `.env` is set correctly.

4.  **Meta Tags Output Channel:**
    *   *Contract:* `orion:tags:enriched`.
    *   *Actual:* Code defaults to `orion:tags:raw`. Ensure `CHANNEL_EVENTS_TAGGED` env var is set to `orion:tags:enriched` for downstream consumers (SQL Writer) to see it.

5.  **Agent Council Channel:**
    *   *Contract:* `orion:agent-council:intake`.
    *   *Actual:* Code defaults vary (`orion-exec:request:CouncilService` vs `orion:council:intake`). Ensure `CHANNEL_COUNCIL_INTAKE` is consistently set in `.env`.

---

## 4. Configuration & Provenance

Environment variables drive the bus topology. The hierarchy is:
1.  `.env` (local overrides)
2.  `docker-compose.yml` (orchestration defaults)
3.  `settings.py` (Pydantic defaults)

**Do not hardcode channel names.** Use the configured variables (e.g., `settings.CHANNEL_EXEC_REQUEST`).

---

## 5. Observability

*   **Correlation ID:** Must be passed to every child message.
*   **Causality Chain:** Producers append a `CausalityLink` before sending a child message.
*   **Logging:** Every hop logs `kind`, `correlation_id`, and `elapsed_time`.
*   **Errors:** Failures are published to `system.error` (or returned to the caller) with `ok=False` and structured error details.

---

## 6. Development

### Acceptance Harness
Use `scripts/bus_harness.py` to test flows.

```bash
# Brain mode
python scripts/bus_harness.py brain "hello world"

# Agent mode
python scripts/bus_harness.py agent "plan a trip"

# Tap (monitor all traffic)
python scripts/bus_harness.py tap
```

### Adding a New Service
1.  Define schemas in `orion/schemas/<service>.py`.
2.  Add config in `settings.py` and `docker-compose.yml`.
3.  Implement `Service` using `orion.core.bus.service.OrionBus` (or `async_service`).
4.  Update this contract document.
