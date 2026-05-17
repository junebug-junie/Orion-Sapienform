# PR: Hub presence grounding hardening

**Branch:** `feature/hub-presence-grounding`  
**Worktree:** `.worktrees/hub-presence-grounding`  
**Base:** `master` (`a06463a8`)  
**Commit:** `7b64048e`

## Summary

Session-scoped **audience presence** (solo / kid_present / family / etc.) already existed across Hub API, UI, chat metadata, cortex-exec situation, and chat prompts. This PR proves the chain, fixes the main stale-cache seam, unifies HTTP/WebSocket injection, improves observability, and adds tests plus a smoke script.

**Answer:** Presence **does work** end-to-end when the session id is consistent. The main bug was cortex-exec `build_situation_for_ctx` caching by `session_id` only, so a mid-session presence change could keep solo grounding until TTL expiry.

## Classification (preflight)

**E — Multiple seams, mostly D + C:**

| Seam | Severity | Status in PR |
|------|----------|--------------|
| Situation cache ignored presence changes | **C** | Fixed (fingerprint in cache key) |
| `loadPresenceContext()` before `initSession()` on first visit | Minor | Fixed (reload after session init) |
| HTTP vs WS presence inject duplicated | Minor | Fixed (`presence_session.py` helper) |
| Tests / smoke / debug flags | **D** | Added |
| UI modal / API store | Already worked | No redesign |

## Consumers (verified)

| # | Consumer | Evidence |
|---|----------|----------|
| 1 | Hub UI / API | `index.html` Presence button; `app.js` modal save/clear/chip; `api_routes.py` GET/POST/DELETE `/api/presence`; `PresenceContextStore` in `main.py` |
| 2 | Hub `/api/situation/brief` | `api_situation_brief` reads same store (`api_routes.py` ~1018–1049) |
| 3 | Hub chat builder / WS | `inject_session_presence` + `build_chat_request` → `metadata.presence_context`; WS + HTTP both call helper |
| 4 | cortex-exec situation | `executor.py` → `build_situation_for_ctx`; affordance `kid_friendly_explanation` when kid_present / child companion |
| 5 | chat_stance | `build_chat_stance_inputs` / `build_chat_stance_debug_payload` situation + `final_prompt_contract` |
| 6 | `chat_quick.j2` / `chat_general.j2` | `situation_prompt_fragment` block; `test_situation_prompt_integration` |
| 7 | Attention / autonomy / detectors | **Not presence consumers** (no wiring found in this pass) |

## NOT consumers (unless proven otherwise)

- Autonomy mutation loop
- Social / durable memory (unless `persist_to_memory` + `orion_presence_persist_allowed`)
- notify / actions / dreams workflows
- `presence_state` websocket connection counter (separate from audience presence)

## Files changed

| File | Change |
|------|--------|
| `services/orion-cortex-exec/app/situation.py` | Cache key includes presence fingerprint (audience, companions, requestor, privacy, timestamps) |
| `services/orion-cortex-exec/app/router.py` | `_situation_grounding_metadata_from_ctx`; response metadata debug flags |
| `services/orion-hub/scripts/presence_session.py` | **New** — shared `inject_session_presence` |
| `services/orion-hub/scripts/api_routes.py` | Use inject helper on HTTP chat |
| `services/orion-hub/scripts/websocket_handler.py` | Use inject helper on WS chat |
| `services/orion-hub/scripts/cortex_request_builder.py` | `routing_debug`: `presence_context_present`, `audience_mode`, `situation_fragment_present` |
| `services/orion-hub/static/js/app.js` | `loadPresenceContext()` after `initSession()`; remove eager pre-session load |
| `scripts/smoke_presence_grounding.py` | **New** — API + optional chat smoke |
| Hub tests | `test_presence_api`, `test_presence_session`, `test_presence_chat_injection`, `test_situation_request_builder` |
| Cortex tests | `test_situation_provider` (cache + kid_present), `test_router_autonomy_payload_export` (metadata) |

No `.env_example` / docker-compose changes (no new settings).

## Tests run

```bash
# Hub (from worktree, with services/orion-hub/.env)
cd .worktrees/hub-presence-grounding/services/orion-hub
set -a && source /path/to/services/orion-hub/.env && set +a
PYTHONPATH=".:scripts:../../.." orion_dev/bin/pytest \
  tests/test_presence_api.py \
  tests/test_presence_session.py \
  tests/test_situation_request_builder.py \
  tests/test_presence_chat_injection.py -q
# 10 passed

# Cortex-exec
cd .worktrees/hub-presence-grounding
PYTHONPATH="services/orion-cortex-exec:." orion_dev/bin/pytest \
  services/orion-cortex-exec/tests/test_situation_provider.py \
  services/orion-cortex-exec/tests/test_router_autonomy_payload_export.py::test_situation_grounding_metadata_from_ctx -q
# 7 passed
```

## Smoke commands

```bash
# Presence + situation brief only (no cortex chat)
python3 scripts/smoke_presence_grounding.py --hub-base http://127.0.0.1:8080 --skip-chat

# Full path (requires running Hub + cortex-exec)
python3 scripts/smoke_presence_grounding.py --hub-base http://127.0.0.1:8080
```

Manual curl sequence (same session id throughout):

```bash
SID="presence-smoke-$(uuidgen | tr -d '-')"
H=(-H "X-Orion-Session-Id: $SID" -H "Content-Type: application/json")
curl -s "${H[@]}" http://127.0.0.1:8080/api/presence | jq .audience_mode
curl -s -X POST "${H[@]}" http://127.0.0.1:8080/api/presence \
  -d '{"audience_mode":"kid_present","requestor":{"display_name":"Juniper"},"companions":[{"display_name":"Kid","relationship":"child","role":"listener","age_band":"child"}]}' | jq .audience_mode
curl -s "${H[@]}" http://127.0.0.1:8080/api/situation/brief | jq .presence.audience_mode
```

## Acceptance

- [x] Hub UI can set/clear presence; chip reflects mode
- [x] Store is session-scoped with TTL (`PresenceContextStore`)
- [x] `/api/situation/brief` matches stored presence for same `X-Orion-Session-Id`
- [x] Chat HTTP/WS inject store when client omits meaningful `presence_context`
- [x] cortex-exec rebuilds situation when presence changes (cache key)
- [x] chat_stance / prompts receive fragment (existing path + tests)
- [x] Debug: `routing_debug` + cortex `metadata` flags for inspect/smoke
