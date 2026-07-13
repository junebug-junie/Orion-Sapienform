# Orion Cortex Orchestrator

The **Cortex Orchestrator** (Orch) is the entry point for the Cognitive Runtime. It accepts high-level client requests (via `orion-cortex:request`), manages the session state, and delegates execution planning to **Cortex Exec**.

## Contracts

### Consumed Channels
| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion-cortex:request` | `ORCH_REQUEST_CHANNEL` | `cortex.orch.request` | Client requests (Brain, Agent, Council modes). |

### Published Channels
| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion-cortex-exec:request` | `CORTEX_EXEC_REQUEST_CHANNEL` | `cortex.exec.request` | Delegation to Cortex Exec. |
| (Caller-defined) | (via `reply_to`) | `cortex.orch.result` | Final result sent back to client. |
| `orion:grammar:event` | `GRAMMAR_EVENT_CHANNEL` | `grammar.event.v1` | Shadow route-arbitration trace (lane pick, mind-gate decision, output mode) for the substrate-runtime `route_grammar` reducer. Off by default. |

### Route arbitration visibility

`call_verb_runtime()` computes, per turn: which execution lane was picked and why (`resolve_execution_lane`), whether "mind" projection fired or was skipped and why, and the output mode. These facts are:

1. Always attached to the returned `VerbResultV1.output["_route_metadata"]` (no flag — always on, zero schema/bus cost) and merged into `main.py`'s `final_meta["route_metadata"]` on the client-facing response.
2. Optionally shadow-published as a `GrammarEventV1` trace (`trace_id` prefix `orch.route:{node}:{correlation_id}`, `source_service=orion-cortex-orch`) when `PUBLISH_CORTEX_ORCH_GRAMMAR=true`, for the substrate-runtime `route_grammar` reducer to materialize into `active_route_arbitration`. See `docs/superpowers/specs/2026-07-12-orch-route-grammar-lane-design.md`.

### Environment Variables
Provenance: `.env_example` → `docker-compose.yml` → `settings.py`

| Variable | Default (Settings) | Description |
| :--- | :--- | :--- |
| `ORCH_REQUEST_CHANNEL` | `orion-cortex:request` | Input channel. |
| `CORTEX_EXEC_REQUEST_CHANNEL` | `orion-cortex-exec:request` | Output channel to Exec. |
| `REDIS_URL` | ... | Redis connection. |
| `PUBLISH_CORTEX_ORCH_GRAMMAR` | `false` | Shadow-publish route arbitration as a `GrammarEventV1` trace. Fire-and-forget; a publish failure never affects the chat response. |
| `GRAMMAR_EVENT_CHANNEL` | `orion:grammar:event` | Channel used for the route-arbitration grammar trace above. |

## Running & Testing

### Run via Docker
```bash
docker-compose up -d orion-cortex-orch
```

### Smoke Test
Use the bus harness in "Brain" mode.
```bash
python scripts/bus_harness.py brain "hello world"
```
