# Bus RPC subscriber hardening

## Summary

After the June 16 pub/sub socket-timeout fix, services that combined **long-lived `subscribe()` loops** with **inline `rpc_request()`** on the same `OrionBusAsync` instance began losing RPC replies (stolen by trace/biometrics caches or torn down by overlapping `listen()`).

This PR introduces a shared `fork_rpc_client()` helper and routes outbound RPC through a **forked bus + RPC worker** while keeping Hunter/Rabbit/cache intake on the parent bus.

## Services hardened

| Service | Change |
|---------|--------|
| **orion-hub** | Single `rpc_bus` for cortex gateway, TTS/STT, and crystallization bus embed |
| **orion-context-exec** | `ContextExecRunner.rpc_bus` for LLM/recall/synthesis tool RPC |
| **orion-cortex-orch** | `_bus_for_rpc()` for DecisionRouter, workflow, verb dispatch |
| **orion-cortex-exec** | `_rpc_bus` for plan execution and verb runtime nested RPC |
| **orion-actions** | `_actions_rpc_bus` for cortex orch/exec RPC from Hunter handlers |
| **orion-chat-memory** | `embed_rpc_bus` for embedding RPC from Hunter handler |
| **orion-vision-council** | `_rpc_bus` for LLM RPC while dual intake listeners run |
| **orion-agent-chain** | `fork_rpc_client` per intake request (worker enabled) |

## Pattern

```python
from orion.core.bus.rpc_fork import fork_rpc_client

rpc_bus = await fork_rpc_client(listener_bus)
# listener_bus: subscribe/publish intake
# rpc_bus: all rpc_request() calls
```

Documented in `services/orion-bus/AGENT_CONTEXT.md`.

## Test plan

- [x] `tests/test_bus_rpc_fork_client.py` — helper delegates to `fork(start_rpc_worker=True)`
- [x] `tests/test_bus_pubsub_timeout.py` — pubsub socket_timeout=None unchanged
- [x] `services/orion-agent-chain/tests/test_bus_listener_rpc.py` — fork mock updated (4 passed)
- [x] `services/orion-cortex-gateway/tests/test_gateway_consumer_resilience.py` — intake/RPC fork (1 passed)
- [x] `services/orion-context-exec/tests/test_llm_profile_binding.py` — runner smoke (8 passed)
- [ ] Rebuild/restart Hub + gateway after merge; confirm Hub logs show `[rpc-fork] worker started` and `path=worker`
- [ ] Smoke `memory_graph_suggest` end-to-end (180s budget, no 63s cancel)

## Env / config

No new environment variables. `.env_example` unchanged; `sync_local_env_from_example.py` reported no changes needed.

## Remaining risks

- **orion-cortex-orch / exec**: module-level `_rpc_bus` initialized in `main()` — import-time callers still fall back to `svc.bus` (same as before fork existed).
- **Per-request fork in agent-chain**: one worker per intake message; acceptable for low QPS, revisit if hot-path latency matters.
- **Hub TTS + cortex share one `rpc_bus`**: worker multiplexes reply channels; fine under concurrent RPC.

## Deploy notes

Rebuild and restart affected containers after merge:

```bash
# Hub (confirms path=worker in logs)
docker compose -f services/orion-hub/docker-compose.yml build hub-app
docker compose -f services/orion-hub/docker-compose.yml up -d hub-app

# Cortex stack
docker compose -f services/orion-cortex-gateway/docker-compose.yml build
docker compose -f services/orion-cortex-orch/docker-compose.yml build
docker compose -f services/orion-cortex-exec/docker-compose.yml build
```
