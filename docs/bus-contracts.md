# Orion Cognitive Bus Contracts (Authoritative)

This document is the **only** source of truth for cross-service routing, payloads, and correlation for the Cortex stack. All clients and services must adhere—no bespoke dict envelopes, no fallback HTTP calls, no side channels.

---

## Envelope & Correlation

* Transport: Redis pub/sub via `orion.core.bus` using `BaseEnvelope` + `OrionCodec`.
* Envelope: `schema_id="orion.envelope"`, `schema_version="2.0.0"`.
* `correlation_id`: **stable across the entire request**; do not regenerate downstream.
* `causality_chain`: append-only lineage; preserve when replying.
* `reply_to`: required for RPC hops. Caller owns reply channel creation.
* Errors: returned as structured payloads with `ok=false` (never bypass to LLM).

---

## Canonical Channels

| Flow | Request channel | Kind (req → res) | Reply prefix (caller-owned) |
| --- | --- | --- | --- |
| Client → Cortex-Orch | `orion-cortex:request` | `cortex.orch.request` → `cortex.orch.result` | `orion-cortex:result` |
| Orch → Exec | `orion-cortex-exec:request` | `cortex.exec.request` → `cortex.exec.result` | `orion-cortex-exec:result` |
| Exec → LLM Gateway | `orion-exec:request:LLMGatewayService` | `llm.chat.request` → `llm.chat.result` | `orion-exec:result:LLMGatewayService` |
| Exec → Recall | `orion-exec:request:RecallService` | `recall.query.request` → `recall.query.result` | `orion-exec:result:RecallService` |
| Exec → Agent Chain (ReAct) | `orion-exec:request:AgentChainService` | `agent.chain.request` → `agent.chain.result` | `orion-exec:result:AgentChainService` |
| Exec → Planner-React (direct) | `orion-exec:request:PlannerReactService` | `agent.planner.request` → `agent.planner.result` | `orion-exec:result:PlannerReactService` |
| Exec → Council (stub) | `orion-exec:request:CouncilService` | `council.request` → `council.result` | `orion-exec:result:CouncilService` |
| Health | `system.health` | `system.health` | n/a |
| Errors | `system.error` | `system.error` | n/a |

Reply channels are ephemeral per request and **must** be provided by the caller (example: `orion-cortex:result:<uuid>`).

---

## Cortex Client Contract (Client/Hub → Orch)

* **Kind:** `cortex.orch.request`
* **Payload model:** `orion.schemas.cortex.contracts.CortexClientRequest`
* **Fields:**
  * `mode`: `"brain" | "agent" | "council"`
  * `verb`: logical verb (e.g., `chat_general`)
  * `packs`: list[str] (tool packs for agent flows)
  * `options`: tuning dict (`temperature`, `max_tokens`, etc.)
  * `recall`: `RecallDirective` (`enabled`, `required`, `mode`, `time_window_days`, `max_items`)
  * `context`: `CortexClientContext` (`messages` [required], `session_id`, `user_id`, `trace_id`, `metadata`)
* **Result kind:** `cortex.orch.result`
* **Result model:** `CortexClientResult`
  * `ok`, `status`, `mode`, `verb`, `final_text`
  * `steps: List[StepExecutionResult]`
  * `memory_used`, `recall_debug`
  * `error` (structured), `correlation_id`

---

## Exec Contract (Orch → Exec)

* **Kind:** `cortex.exec.request`
* **Payload:** `PlanExecutionRequest` (`plan`, `args`, `context`)
* Required `args.extra` entries:
  * `mode`, `packs`, `recall: RecallDirective`, `options`, `trace_id`, `session_id`, `verb`
* **Result kind:** `cortex.exec.result`
* **Payload:** `PlanExecutionResult` (`status`, `final_text`, `steps`, `memory_used`, `recall_debug`, `error`)

Exec is the **only** orchestrator of workers. Orch must never call LLM/Recall/Agent directly.

---

## Worker Contracts

### LLM Gateway
* `llm.chat.request` → `llm.chat.result`
* Request: `ChatRequestPayload` (`model`, `profile`, `messages`, `options`, `user_id`, `session_id`)
* Response: `ChatResultPayload` (`content`, `usage`, `spark_meta`, `raw`)

### Recall
* `recall.query.request` → `recall.query.result`
* Request: `RecallRequestPayload` (`query_text`, `session_id`, `user_id`, `mode`, `time_window_days`, `max_items`, `packs`, `trace_id`)
* Response: `RecallResultPayload` (`fragments[]`, `debug`)
* Exec surfaces only `memory_used` + `recall_debug` to the client unless debug is explicitly requested.

### Agent Chain (ReAct worker)
* `agent.chain.request` → `agent.chain.result`
* Request: `AgentChainRequest` (`text`, `mode`, `messages`, `session_id`, `user_id`, `packs`, `tools`)
* Response: `AgentChainResult` (`text`, `structured`, `planner_raw`)

### Council (stub)
* `council.request` → `council.result`
* Currently returns a structured error envelope; reserved for supervisor/worker path.

---

## Settings Provenance

* `.env_example` → `docker-compose.yml` → `settings.py`
* Every channel above has a surfaced setting in its service:
  * Orch: `channel_cortex_request`, `channel_cortex_result_prefix`, `channel_exec_request`, `channel_exec_result_prefix`
  * Exec: `channel_exec_request`, `exec_request_prefix`, `exec_result_prefix`, `channel_llm_intake`, `channel_recall_intake`, `channel_agent_chain_intake`, `channel_planner_intake`, `channel_council_intake`
  * Agent Chain: `agent_chain_request_channel`, `agent_chain_result_prefix`, `planner_request_channel`, `planner_result_prefix`
  * Planner-React: `planner_request_channel`, `planner_result_prefix`
  * Recall: `RECALL_BUS_INTAKE`
  * LLM Gateway: `channel_llm_intake`

No service may invent a channel name that is not represented in its settings.

---

## Acceptance Harness

* `python scripts/bus_harness.py brain "hello world"` publishes `cortex.orch.request` to `orion-cortex:request` and waits on `orion-cortex:result:<uuid>`.
* Other modes: `python scripts/bus_harness.py agent "plan a trip"` or `python scripts/bus_harness.py tap` (pattern subscribe to `orion:*`).
* Harness envs (`scripts/.env_example`): `ORCH_REQUEST_CHANNEL`, `ORCH_RESULT_PREFIX`, `HARNESS_RPC_TIMEOUT_SEC`.

---

## Observability

* Every hop logs: service name, kind, correlation_id, intake channel, reply channel, elapsed time.
* Exec returns `StepExecutionResult.logs` with worker hops and timing.
* Structured failures also publish to `system.error` with the same `correlation_id`.

---

## Hard Rules

1. No direct Client → LLM Gateway calls (only Client → Orch → Exec → LLM).
2. No fallback paths. Failures bubble as structured errors with correlation preserved.
3. Only the canonical channels above are valid, and they must be reflected in settings/env.
4. Payloads must use shared `orion/schemas`; never ship loose dict envelopes.
