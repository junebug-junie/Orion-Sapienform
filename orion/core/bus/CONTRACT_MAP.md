# Orion Cognitive Bus Contract (Authoritative)

This file is the single source of truth for cognitive bus contracts. All services **must** use the channel and kind constants defined in [`contracts.py`](./contracts.py).

## Channels

| Flow | Channel (intake) | Reply prefix |
| ---- | ---------------- | ------------ |
| Client → Cortex-Orch | `orion-cortex:request` | `orion-cortex:reply:` |
| Orch → Cortex-Exec | `orion-cortex-exec:request` | `orion-cortex-exec:reply:` |
| Exec → LLM Gateway | `orion-exec:request:LLMGatewayService` | caller-provided |
| Exec → Recall | `orion-exec:request:RecallService` | caller-provided |
| Exec → Agent Chain | `orion-exec:request:AgentChainService` | caller-provided |
| Exec → Planner React | `orion-exec:request:PlannerReactService` | caller-provided |
| Exec → Agent Council | `orion-exec:request:AgentCouncilService` | caller-provided |

Reply channels **must** be `reply_prefix + correlation_id`.

## Envelope kinds

| Kind | Purpose |
| ---- | ------- |
| `cortex.orch.request` / `cortex.orch.result` | Client ↔ Orch RPC |
| `cortex.exec.request` / `cortex.exec.result` | Orch ↔ Exec RPC |
| `llm.chat.request` / `llm.chat.result` | Exec ↔ LLM Gateway |
| `recall.query.request` / `recall.query.result` | Exec ↔ Recall |
| `agent.chain.request` / `agent.chain.result` | Exec ↔ Agent Chain |
| `agent.planner.request` / `agent.planner.result` | Exec ↔ Planner React |
| `agent.council.request` / `agent.council.result` | Exec ↔ Council |

## Correlation + reply rules

- Every envelope carries `id`, `correlation_id`, `reply_to`, and `causality_chain`.
- Callers **must** set `reply_to` on the ingress envelope. Services publish responses to that channel; they **must not** invent new reply destinations.
- Services derive child envelopes by preserving the parent `correlation_id` and extending `causality_chain`.

## Payload contracts (high level)

- **LLM Gateway** (`llm.chat.request`): `ChatRequestPayload` (model/messages/options/user_id/session_id). Response: `ChatResultPayload`.
- **Recall** (`recall.query.request`): `RecallRequestPayload` (query_text, max_items, time_window_days, mode, tags, user_id, session_id, trace_id). Response: `RecallResultPayload` (`ok`, `error`, `fragments`, `debug`).
- **Agent Chain** (`agent.chain.request`): `AgentChainRequest` / `AgentChainResult`.
- **Planner React** (`agent.planner.request`): `PlannerRequest` / `PlannerResponse`.
- **Council** (`agent.council.request`): opaque dict for now; maintain `correlation_id` + reply channel discipline.

## Logging / observability expectations

- Every hop logs `channel`, `kind`, `correlation_id`, `reply_to`, and `source`.
- Exec returns per-step `StepExecutionResult` (status, soft_fail flag, latency, logs, error).

## Acceptance harness (manual)

- Use `scripts/bus_harness.py` to publish `brain` or `agent` requests with proper envelopes.
- Use `redis-cli psubscribe orion:*` to observe traffic end-to-end (correlation ids should align with harness output).

