# PR: Hub LLM route override + Agent → context-exec lane

**Branch:** `feat/hub-llm-route-override`  
**Worktree:** `.worktrees/feat/hub-llm-route-override`

## Summary

Implements the design contract **mode decides behavior; route decides compute**:

1. **orion-llm-gateway** exposes `GET /routes` backed by `LLM_GATEWAY_ROUTE_TABLE_JSON`, with cached upstream health probes and `default_route=chat`.
2. **orion-hub** adds a Route selector (`chat`, `quick`, `agent`, `metacog`) beside mode buttons, proxies `GET /api/llm-routes`, and sends `llm_route` on chat/voice payloads (wired to `options.llm_route` for cortex paths).
3. **Hub Agent mode** is reclaimed for **context-exec**: HTTP + WS call `POST /context-exec/run` when `HUB_AGENT_CONTEXT_EXEC_ENABLED=true`, passing selected route as `llm_profile`. No AgentChain/ReAct bypass; proposal ledger/review unchanged.
4. **Down routes** trigger explicit operator choice (Use chat / Try anyway / Cancel) — no silent fallback.

## Files changed (high signal)

| Area | Paths |
|------|-------|
| Gateway catalog | `services/orion-llm-gateway/app/route_catalog.py`, `app/main.py`, `tests/test_route_catalog.py` |
| Hub clients/UI | `services/orion-hub/scripts/llm_gateway_client.py`, `context_exec_client.py`, `context_exec_agent_bridge.py`, `templates/index.html`, `static/js/app.js` |
| Hub routing | `scripts/cortex_request_builder.py`, `api_routes.py`, `websocket_handler.py` |
| Schema | `orion/schemas/context_exec.py` (`llm_profile`) |
| Context-exec | `services/orion-context-exec/app/runner.py` (runtime_debug) |
| Env/compose | `services/orion-hub/.env_example`, `docker-compose.yml`, `app/settings.py`, `scripts/sync_local_env_from_example.py` |
| Smoke | `scripts/smoke_llm_gateway_routes.py`, `scripts/repl/orion_fresh_main_smoke.sh` (worktree `.git` file) |

## Test plan

- [x] `services/orion-llm-gateway/tests/test_route_catalog.py` — 2 passed
- [x] `services/orion-hub/tests/test_llm_route_selector.py` — 7 passed
- [x] `services/orion-hub/tests/test_proposal_review_hub.py` — 20 passed
- [x] `ORION_PY=orion_dev/bin/python bash scripts/context_exec_beta_gate.sh` — **BETA GATE PASS**
- [x] `ORION_PY=orion_dev/bin/python STORE=/tmp/orion-proposals.json ./scripts/repl/orion_fresh_main_smoke.sh` — **PASS=20 FAIL=0**

## Operator notes

- Synced to host `services/orion-hub/.env` (gitignored): `HUB_LLM_GATEWAY_*`, `HUB_AGENT_CONTEXT_EXEC_*`, `HUB_CONTEXT_EXEC_*`.
- Restart Hub + LLM gateway containers after pulling env changes.
- `HUB_LLM_GATEWAY_URL` host dev default: `http://127.0.0.1:8210`.

## Remaining risks

- `llm_profile` is recorded on context-exec runs; RLM engines do not yet route LLM RPC by profile (compute binding is a follow-up).
- Route-down UX uses browser `confirm()` dialogs (functional, not a styled modal).

## Non-goals

- No new route name keys beyond `chat`, `quick`, `agent`, `metacog`.
- No mutation / write loosening on context-exec.
- No changes to proposal review API semantics.
