# Orion Cognitive Bus Contracts (Authoritative)

This document is the single source of truth for cross-service routing, payloads, and correlation rules for the Cortex stack. All clients and services **must** follow these contracts—no ad-hoc dict envelopes, no fallback paths.

---

## Envelope & Correlation Rules

* Transport: Redis pub/sub via `orion.core.bus` (Titanium envelope only).
* Envelope: `orion.core.bus.bus_schemas.BaseEnvelope`
  * `schema_id`: `"orion.envelope"` (alias `schema`)
  * `schema_version`: `"2.0.0"`
  * `id`: unique per hop, auto-generated.
  * `correlation_id`: **stable per request**; propagated unchanged across hops.
  * `causality_chain`: append-only lineage (parent hop metadata).
  * `reply_to`: required for RPC hops; callee publishes the response to this channel.
  * `source`: `ServiceRef` describing the producer.
* Encoding: `orion.core.bus.codec.OrionCodec` (never raw JSON outside the codec).
* Error handling: Failures are returned as structured payloads with `ok=false` (no fallbacks to direct LLM or HTTP).

---

## Channel Map (canonical)

| Role | Channel | Kind (request → result) | Notes |
| --- | --- | --- | --- |
| Client → Cortex-Orch | `orion-cortex-orch:request` | `cortex.orch.request` → `cortex.orch.result` | Single public intake for brain/agent/council. |
| Cortex-Orch → Cortex-Exec | `orion-cortex-exec:request` | `cortex.exec.request` → `cortex.exec.result` | Orch never calls workers directly. |
| Exec → LLM Gateway | `orion-exec:request:LLMGatewayService` | `llm.chat.request` → `llm.chat.result` | **Only** intake for LLMs. |
| Exec → Recall | `orion-exec:request:RecallService` | `recall.query.request` → `recall.query.result` | Augmentation step; never called by clients. |
| Exec → Agent Chain (ReAct) | `orion-exec:request:AgentChainService` | `agent.chain.request` → `agent.chain.result` | Agentic workflow worker. |
| Exec → Planner-React (direct) | `orion-exec:request:PlannerReactService` | `agent.planner.request` → `agent.planner.result` | Reserved; Agent Chain proxies by default. |
| Exec → Council (stub) | `orion-exec:request:CouncilService` | `council.request` → `council.result` | Supervisor/worker path placeholder. |
| Health | `system.health` | `system.health` | Heartbeats from all services. |
| Errors | `system.error` | `system.error` | Structured failures with correlation preserved. |

Reply channels are ephemeral per request and **must** be provided by the caller (e.g., `orion-cortex-orch:result:<uuid>`).

---

## Cortex Client Contract (Hub/UI → Orch)

**Kind:** `cortex.orch.request`  
**Payload model:** `orion.schemas.cortex.contracts.CortexClientRequest`

Fields:
* `mode`: `"brain" | "agent" | "council"` (required)
* `verb`: logical verb name (e.g., `chat_general`)
* `packs`: list of capability packs (agent path uses this for tool selection)
* `options`: free-form tuning (temperature, max_tokens, etc.)
* `recall`: `RecallDirective`  
  * `enabled` (default true), `required` (fail-fast if recall fails), `mode` (`hybrid`|`deep`), `time_window_days`, `max_items`
* `context`:
  * `messages`: `List[LLMMessage]` (required history)
  * `session_id`, `user_id`, `trace_id`
  * `metadata`: dict for client-specific tags

**Result kind:** `cortex.orch.result`  
**Payload model:** `CortexClientResult`
* `ok`: bool
* `mode`, `verb`
* `final_text`: normalized reply string
* `steps`: `List[StepExecutionResult]` (including recall/agent/llm)
* `memory_used`: bool
* `recall_debug`: counts/mode/window/error (no raw fragments unless explicitly requested)
* `error`: optional structured error

---

## Exec Contract (Orch → Exec)

**Kind:** `cortex.exec.request`  
**Payload model:** `orion.schemas.cortex.schemas.PlanExecutionRequest`

Required args (embedded in `PlanExecutionArgs.extra`):
* `mode`: `"brain" | "agent" | "council"`
* `packs`: list[str]
* `recall`: `RecallDirective` as above
* `options`: llm/agent tuning
* `trace_id`: correlation string for downstream logs

**Result kind:** `cortex.exec.result`  
**Payload:** `PlanExecutionResult` with:
* `status`: `"success" | "partial" | "fail"`
* `final_text`: string or `None`
* `memory_used`: bool
* `recall_debug`: dict
* `steps`: ordered `StepExecutionResult`
* `error`: optional string

---

## Worker Contracts

### LLM Gateway
* Request kind: `llm.chat.request`, payload `ChatRequestPayload`
  * `model`, `profile`, `messages`, `options`, `user_id`, `session_id`
* Response kind: `llm.chat.result`, payload `ChatResultPayload`
  * Normalized `content` + `usage` + `spark_meta`

### Recall
* Request kind: `recall.query.request`, payload `RecallRequestPayload`
  * `query_text`, `session_id`, `user_id`, `mode`, `time_window_days`, `max_items`, `packs`, `trace_id`
* Response kind: `recall.query.result`, payload `RecallResultPayload`
  * `fragments[]` (normalized) + `debug` (mode/window/count/error)
  * Exec surfaces counts + debug; raw fragments stay server-side unless explicitly requested.

### Agent Chain (ReAct worker)
* Request kind: `agent.chain.request`, payload `AgentChainRequest`
  * `text`, `mode`, `messages`, `session_id`, `user_id`, `packs`, `tools`
* Response kind: `agent.chain.result`, payload `AgentChainResult`
  * `text`, `structured`, `planner_raw`

### Council (stub)
* Request kind: `council.request`
* Response kind: `council.result`
* Currently returns a structured error envelope; path reserved for supervisor/worker.

---

## Acceptance Harness

`python scripts/bus_harness.py brain "hello world"`  
* Publishes a `cortex.orch.request` to `orion-cortex-orch:request`
* Waits on a correlation reply channel and prints the envelope + step trace

Other modes:
* `python scripts/bus_harness.py agent "plan a trip"` (agent chain path)
* `python scripts/bus_harness.py tap` (pattern-subscribe to `orion:*` for live traffic)

All harness requests use the canonical envelopes above; no bespoke payloads.

---

## Observability Requirements

* Every service logs: service name, kind, correlation_id, request channel, reply channel, and elapsed timing per step.
* Exec returns `StepExecutionResult.logs` with worker hops and timing.
* Errors are emitted to `system.error` with the same `correlation_id`.

---

## Hard Rules

1. No direct Client → LLM Gateway calls. The only path is Client → Orch → Exec → LLM Gateway.
2. No fallback paths. Failures bubble up as structured errors with preserved correlation.
3. Only the channels above are valid. Settings **must** expose them; no magic defaults.
4. Payloads must be typed via shared `orion/schemas` (never loose dicts).

