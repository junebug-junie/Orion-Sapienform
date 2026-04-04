# LLM Route Realignment Manifest (Merged Chat+Agent Default + Helper Lane)

## Scope

This change set keeps Orion's schema-level and internal route contracts intact while changing the default Atlas physical mapping:

- `chat` remains on Atlas chat lane (`8011`).
- `agent` remains a distinct logical route, but defaults to the same Atlas chat lane (`8011`) in merged mode.
- `metacog` remains a dedicated Atlas lane (`8012`), unchanged behavior.
- `helper` is added as a new Atlas lane (`8013`) for cheap bounded internal transforms.
- Dedicated heavy `agent` split (`8014`) remains available through configuration only (`agent-split` compose profile + route table update).

## Files touched

- `config/llm_profiles.yaml`
- `docs/postflight/llm_route_realignment_manifest.md`
- `scripts/smoke_llm_gateway_routes.py`
- `services/orion-cortex-exec/app/executor.py`
- `services/orion-agent-chain/tests/test_tool_executor_routes.py`
- `services/orion-cortex-exec/tests/test_chat_general_route_mapping.py`
- `services/orion-llm-gateway/.env_example`
- `services/orion-llm-gateway/README.md`
- `services/orion-llm-gateway/docker-compose.yml`
- `services/orion-llm-gateway/tests/test_llm_backend.py`
- `services/orion-llamacpp-host/.env_example`
- `services/orion-llamacpp-host/README.md`
- `services/orion-llamacpp-host/docker-compose.yml`
- `services/orion-llamacpp-host/docker-compose.atlas-workers.yml`
- `tests/test_planner_react_contract.py`

## Resulting route intent (default merged mode)

- `chat` → existing Atlas chat worker endpoint (`atlas-worker-1`)
- `agent` → existing Atlas chat worker endpoint (`atlas-worker-1`) [logical route preserved]
- `metacog` → Atlas dedicated metacog worker (`atlas-worker-2`)
- `helper` → Atlas helper worker (`atlas-worker-helper-1`)

## Optional split mode (config-only)

- `agent` → Atlas heavy Qwen worker (`atlas-worker-agent-1`)
- Enabled by route-table configuration and docker compose `agent-split` profile.

## Architecture notes

- Routing remains gateway-centered and env-first through `LLM_GATEWAY_ROUTE_TABLE_JSON`.
- No new route enum or service-discovery layer was introduced.
- `served_by` remains observability metadata; route isolation is enforced by explicit route → URL mapping plus one profile per host container.
- Schema-level `route` was not collapsed or renamed (`chat`, `agent`, `metacog` remain valid; `helper` added).
- Default/no-route callers intentionally still resolve through `LLM_ROUTE_DEFAULT` (normally `chat`), while explicit bad named routes fail closed whenever route-table mode is active.

## Validation notes

- PlannerReact now sends `route="agent"` for its gateway calls.
- AgentChain delegated LLM-only tools now send `route="agent"`.
- `chat_general.synthesize_chat_stance_brief` now sends `route="helper"`.
- `chat_general.llm_chat_general` now sends `route="chat"` explicitly.
- Metacog callers continue using explicit `route="metacog"`.
- Gateway tests cover merged and split-mode `agent` route tables, helper route loading, fail-closed behavior for unknown explicit routes, and preserved default-chat behavior for no-route callers under route-table mode.
- Typed agent RPC contracts (`agent.planner.request`, `agent.chain.request`) and Orch mode routing remain unchanged.
