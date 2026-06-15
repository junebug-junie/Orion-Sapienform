# PR: fix(context-exec): live Agent route hotfixes after synthesis deploy

**No GitHub PR** — landed as direct commit on `main`: [`61a90d22`](https://github.com/junebug-junie/Orion-Sapienform/commit/61a90d22)  
**Stacks on:** [#696](https://github.com/junebug-junie/Orion-Sapienform/pull/696) (agent synthesis)

## Summary

Commits the live hotfixes required after deploying Hub Agent route-bound synthesis. These fixes make the live Hub → Agent → context-exec route probe work against rebuilt containers, without changing mutation posture, proposal gating, or route names.

## Fixes

- Map non-UUID `ctxcorr_*` correlation ids to UUIDv5 before building `BaseEnvelope`
- Use valid `AgentTraceStepV1.effect_kind="read_only"`
- Return `llm_response` and `operator_summary` from Hub Agent API responses
- Make route probe accept `llm_response` or fallback `text`
- Fix metacog synthesis crash by catching `json.JSONDecodeError`
- Align `CONTEXT_EXEC_LLM_GATEWAY_URL` default to port `8210` (settings, compose, `.env_example`)

## Files changed

| Area | Path |
|------|------|
| LLM RPC | `services/orion-context-exec/app/llm_tools.py` |
| Synthesis | `services/orion-context-exec/app/agent_synthesis.py` |
| Settings / env | `services/orion-context-exec/app/settings.py`, `.env_example`, `docker-compose.yml` |
| Tests | `services/orion-context-exec/tests/test_llm_profile_binding.py` |
| Hub bridge | `services/orion-hub/scripts/context_exec_agent_bridge.py` |
| Hub API | `services/orion-hub/scripts/api_routes.py` |
| Probe | `scripts/context_exec_agent_route_probe.sh` |

## Verification

| Command | Result |
|---------|--------|
| `context_exec_beta_gate.sh` | BETA GATE PASS |
| `orion_fresh_main_smoke.sh` | PASS=20 FAIL=0 |
| Live `quick` route probe | PROBE_OK |
| Live `metacog` route probe | PROBE_OK |
| Live `chat` route | ROUTE_DOWN (Atlas worker down) |
| Live `agent` route | ROUTE_DOWN (Atlas worker down) |
| Trace autopsy via `quick` | PROBE_OK |

## Safety

- No real mutation enabled
- No proposal execution buttons added
- No arbitrary URL routing
- No silent fallback
- Pending Decisions remains the review surface

## Env parity

`.env_example` matches operator `.env` for Agent route, synthesis, gateway port `8210`, and proposal review (`PROPOSAL_REVIEW_API_ENABLED=true`, `PROPOSAL_LEDGER_STORE_PATH=/tmp/orion-proposals.json`). Ledger intake remains disabled by default.
