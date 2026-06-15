# PR: feat(hub): Mode + Compute dropdowns (Agent apparatus model)

**Branch:** `feat/hub-mode-compute-dropdowns`  
**Base:** `main`

## Grounding report (before fix)

| Area | What the repo actually did |
|------|----------------------------|
| Mode UI | Inline **button chip row** in `index.html` (`mode-btn` buttons). `thought-process.js` rewrote the Brain button label to **Grounded Small** at runtime and injected `brainDeepModeBtn`. |
| Compute UI | Inline **ROUTE** chip radiogroup (`#llmRouteSelector`, `.llm-route-btn` for `chat`/`quick`/`agent`/`metacog`). |
| State | `selectedLlmRoute` in `app.js`; persisted in `localStorage` key `orion_llm_route`. |
| Default compute | **`chat`** (`selectedLlmRoute \|\| 'chat'`; backend `cortex_request_builder` default `chat`). |
| Catalog fetch | `loadLlmRouteCatalog()` → `GET /api/llm-routes` once on page load (no poll). |
| Down route | `confirmDownRouteOrProceed` suggested **`chat`** fallback via `window.confirm`. |
| Payload wiring | Chat/voice send `llm_route` on payload → `build_chat_request` → `options.llm_route`. |
| Agent routing | `should_use_context_exec_agent_lane` when `mode=agent`; `run_hub_agent_via_context_exec` maps `options.llm_route` → `llm_profile`. |
| Tests | Backend/route behavior only; **no UI regression** guarding chip rows. |
| User-facing label | **ROUTE** (not COMPUTE). |

**Problem:** `Agent` (mode) and `agent` (compute lane) appeared as peer chip choices; compute lanes rendered as pills despite the requested dropdown model.

## Summary of UI changes

- Replaced mode chip row with **`MODE` dropdown** (`#hubModeSelect`).
- Replaced route chip row with **`COMPUTE` dropdown** (`#hubComputeSelect`) showing lane metadata (`quick ● up — atlas-worker-fast-1 / llama.cpp`).
- Removed `#llmRouteSelector` radiogroup and all `.llm-route-btn` elements.
- Added 30s polling for `/api/llm-routes` catalog refresh.
- Down-lane dialog now suggests **Use quick** (not chat).

## Conceptual model implemented

```text
MODE     [ Agent ▾ ]        ← behavior / apparatus
COMPUTE  [ quick ▾ ]        ← GPU/model lane (chat | quick | agent | metacog)

Agent (mode) ≠ agent (compute lane id)
Default compute = quick
Agent + quick  → context-exec llm_profile=quick
Agent + agent  → context-exec llm_profile=agent (only when explicitly selected)
```

**Agent** remains the mode label. No ReAct/AgentChain restoration.

## Files changed

| Path |
|------|
| `services/orion-hub/templates/index.html` |
| `services/orion-hub/static/js/app.js` |
| `services/orion-hub/static/js/thought-process.js` |
| `services/orion-hub/scripts/cortex_request_builder.py` |
| `services/orion-hub/scripts/context_exec_agent_bridge.py` |
| `services/orion-hub/tests/test_llm_route_selector.py` |
| `services/orion-hub/tests/test_chat_stance_debug_panel.py` |
| `services/orion-hub/README.md` |

No `.env_example`, docker-compose, bus channel, or schema registry changes required.

## Tests run

| Command | Exit | Result |
|---------|------|--------|
| `pytest services/orion-hub/tests/test_llm_route_selector.py -q` | 0 | 16 passed |
| `pytest services/orion-hub/tests/test_proposal_review_hub.py -q` | 0 | 14 passed |
| `pytest services/orion-hub/tests/test_chat_stance_debug_panel.py -q` | 0 | (included above) |
| `cd services/orion-llm-gateway && PYTHONPATH=. pytest tests/test_route_catalog.py -q` | 0 | 2 passed |
| `pytest services/orion-context-exec/tests/test_llm_profile_binding.py -q` | 0 | 8 passed |
| `ORION_PY=orion_dev/bin/python bash scripts/context_exec_beta_gate.sh` | 0 | BETA GATE PASS |
| `ORION_PY=orion_dev/bin/python STORE=/tmp/orion-proposals.json ./scripts/repl/orion_fresh_main_smoke.sh` | 0 | PASS=20 FAIL=0 |

## Final UI (text)

```text
MODE     [ Grounded Small ▾ ]
COMPUTE  [ quick ● up — atlas-worker-fast-1 / llama.cpp ▾ ]

Mode options: Auto, Grounded Small, Brain, Quick, Story, Agent, Council
Compute options: chat, quick, agent, metacog (with status metadata in menu)
```

## Confirmations

- **Agent remains Agent** in the mode dropdown.
- **Default compute is `quick`** (UI, localStorage invalid fallback, backend default when `llm_route` omitted).
- **Compute dropdown passes selected lane** through chat/voice payloads as `llm_route` → context-exec `llm_profile`.
- **No backend safety posture changed** (permissions, mutation, proposal gating untouched).
- **Regression test** asserts no ROUTE radiogroup/chip row for compute lanes.

## Code review

Subagent review: **APPROVED** (30 Hub tests green; minor dead-code cleanup applied pre-merge).

## Env parity

No `.env_example` keys added or changed in this PR.
