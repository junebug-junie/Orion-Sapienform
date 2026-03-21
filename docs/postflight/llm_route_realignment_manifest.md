# LLM Route Realignment Manifest

## Scope

This change set realigns Orion's llama.cpp routing so that active LLM serving is Atlas-centered:

- `chat` remains on the current Atlas chat path.
- `metacog` moves from Athena to an Atlas-only dedicated lane.
- `agent` is introduced as a dedicated heavy Atlas route for PlannerReact and AgentChain reasoning.

## Files touched

- `config/llm_profiles.yaml`
- `docs/postflight/llm_route_realignment_manifest.md`
- `scripts/smoke_llm_gateway_routes.py`
- `services/orion-planner-react/app/api.py`
- `services/orion-agent-chain/app/tool_executor.py`
- `services/orion-agent-chain/tests/test_tool_executor_routes.py`
- `services/orion-llm-gateway/.env_example`
- `services/orion-llm-gateway/README.md`
- `services/orion-llm-gateway/docker-compose.yml`
- `services/orion-llm-gateway/tests/test_llm_backend.py`
- `services/orion-llamacpp-host/.env_example`
- `services/orion-llamacpp-host/README.md`
- `services/orion-llamacpp-host/docker-compose.yml`
- `services/orion-llamacpp-host/docker-compose.atlas-workers.yml`
- `tests/test_planner_react_contract.py`

## Resulting route intent

- `chat` → existing Atlas chat worker endpoint (`atlas-worker-1`)
- `metacog` → Atlas dedicated metacog worker (`atlas-worker-2`)
- `agent` → Atlas heavy Qwen worker (`atlas-worker-agent-1`)

## Architecture notes

- Routing remains gateway-centered and env-first through `LLM_GATEWAY_ROUTE_TABLE_JSON`.
- No new route enum or service-discovery layer was introduced.
- `served_by` remains observability metadata; route isolation is enforced by explicit route → URL mapping plus one profile per host container.
- The gateway models `chat` as the existing Atlas chat endpoint only; any internal Atlas lane-sharing remains behind that endpoint and is not represented as separate gateway routes.
- Default/no-route callers intentionally still resolve through `LLM_ROUTE_DEFAULT` (normally `chat`), while explicit bad named routes fail closed whenever route-table mode is active.

## Validation notes

- PlannerReact now sends `route="agent"` for its gateway calls.
- AgentChain delegated LLM-only tools now send `route="agent"`.
- Metacog callers continue using explicit `route="metacog"`.
- Gateway tests cover both fail-closed behavior for unknown explicit routes and preserved default-chat behavior for no-route callers under route-table mode.
