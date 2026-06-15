# PR: feat(context-exec): bind llm_profile to RLM engine route selection

## Summary

Hub Agent mode already passes `llm_profile` on `ContextExecRequestV1` (from `feat/hub-llm-route-override`). This PR makes that profile **operational** in context-exec: validates trusted route ids, resolves effective profile/route, records explicit `runtime_debug` fields, and binds RLM `llm.subcall` / bus LLM RPC to the selected gateway route key.

Permissions, proposal gating, and read-only posture are unchanged.

## Files changed

| Area | Path |
|------|------|
| Schema | `orion/schemas/context_exec.py` |
| Schema tests | `orion/schemas/tests/test_context_exec_llm_profile.py` |
| Resolver | `services/orion-context-exec/app/llm_profile_resolver.py` (new) |
| LLM RPC | `services/orion-context-exec/app/llm_tools.py` (new) |
| Runner | `services/orion-context-exec/app/runner.py` |
| Organs | `services/orion-context-exec/app/organ_runtime.py` |
| Settings | `services/orion-context-exec/app/settings.py` |
| Env / compose | `services/orion-context-exec/.env_example`, `docker-compose.yml` |
| Docs | `services/orion-context-exec/README.md` |
| Tests | `tests/test_llm_profile_resolver.py`, `tests/test_llm_profile_binding.py`, `tests/conftest.py` |

## Route/profile semantics

| Input | Behavior |
|-------|----------|
| `llm_profile` omitted | Default `CONTEXT_EXEC_DEFAULT_LLM_PROFILE` (`chat`) |
| `chat` / `quick` / `agent` / `metacog` | Selected profile = route_used for LLM RPC |
| Invalid id (e.g. `http://evil`, `circe-32b`) | Pydantic validation error (422) |
| Route down / not in gateway catalog | Fail closed with `LLMProfileUnavailableError` |
| Route down + `CONTEXT_EXEC_LLM_PROFILE_FALLBACK_ENABLED=true` | Fallback to default profile; `fallback_used=true`, `fallback_reason` set |

**Runtime debug fields** (every run):

- `llm_profile_requested`, `llm_profile_selected`, `route_used`
- `engine_requested`, `engine_selected` (existing engine fields preserved)
- `fallback_used`, `fallback_reason` (profile route fallback; engine fallback still uses `fallback_engine`)

No arbitrary URLs — route selection is by trusted id only via `ChatRequestPayload.route`.

## Before / after

| Scenario | Before | After |
|----------|--------|-------|
| Agent + chat | `llm_profile` stored in `runtime_debug.llm_profile` only | RPC route=`chat`; debug fields populated |
| Agent + quick | Same (no compute binding) | RPC route=`quick` |
| Agent + agent | Required agent route even for chat/quick workloads | Only uses `agent` route when selected |
| Invalid profile | Accepted at HTTP layer | Rejected at schema validation |
| Down route | Silent pass-through | Fail closed (or explicit fallback if enabled) |

## Test results

```bash
PYTHONPATH=. orion_dev/bin/python -m pytest orion/schemas -q
# 20 passed

PYTHONPATH=. orion_dev/bin/python -m pytest services/orion-context-exec/tests -q
# 147 passed, 1 xfailed

PYTHONPATH=. orion_dev/bin/python -m pytest services/orion-hub/tests/test_llm_route_selector.py -q
# 7 passed

PYTHONPATH=. orion_dev/bin/python -m pytest services/orion-hub/tests/test_proposal_review_hub.py -q
# 20 passed

PYTHONPATH=. orion_dev/bin/python -m pytest services/orion-llm-gateway/tests/test_route_catalog.py -q
# 2 passed

ORION_PY=orion_dev/bin/python bash scripts/context_exec_beta_gate.sh
# BETA GATE PASS

ORION_PY=orion_dev/bin/python STORE=/tmp/orion-proposals.json ./scripts/repl/orion_fresh_main_smoke.sh
# PASS=20 FAIL=0
```

## Remaining risks

- AlexZhang/fake engines remain organ-heuristic today; `llm.subcall` binding is wired and tested via probe engine + `llm_tools`, but most modes do not yet invoke model RPC on every path.
- Gateway `/routes` probe is skipped when `CONTEXT_EXEC_LLM_GATEWAY_URL` is empty (dev/test); production should set the URL for fail-closed availability checks.
- `CONTEXT_EXEC_ALLOWED_LLM_PROFILES` is documented in env but schema uses fixed allowlist (`chat/quick/agent/metacog`) — intentional match with gateway catalog.

## Base branch

Stacks on `feat/hub-llm-route-override` (Hub route selector + `llm_profile` request field).
