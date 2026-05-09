# Hub: Orion Mind tab + memory graph reliability

## Summary

This branch delivers **Orion Mind** visibility in the Hub (tab, modal drill-down, recent-run analytics) and **memory graph** hardening across recall/GraphDB paths, client-side suggest UX, and Hub-side failover when the default LLM lane times out. It also wires **Mind** correctly through the chat stack (session headers, optional `mind_enabled` metadata into cortex requests) and documents local **orion-mind** connectivity in orch’s example env.

> **Note:** Relative to `origin/main`, this branch’s history also includes several **design specs**, a **social-room-bridge** webhook hardening commit, and an earlier **memory/recall** guardrails fix. If the goal is a narrowly scoped Hub-only merge, consider rebasing onto `main` and dropping unrelated commits before merge.

## Orion Mind (Hub + orch)

- **Mind tab UI:** Recent Mind runs, filters, safe analytics presentation, modal for run detail (`feat(hub): implement Mind tab…`).
- **API correctness:** `X-Orion-Session-Id` is read as a proper **header** on Mind list/get/correlation routes so browser `fetch` matches FastAPI expectations (`fix(hub): correct Mind route ordering…`).
- **Cortex request path:** When the client sends `context.metadata.mind_enabled === true`, Hub forwards **`mind_enabled: true`** into cortex chat metadata so orch can enable Mind for that turn.
- **Configuration:** `.env_example` sets **`ORION_MIND_BASE_URL=http://orion-mind:6611`** as the documented default for compose-style deployments.

## Memory graph

- **Hub `memory_graph_suggest`:** After the first cortex chat result, if the effective text looks like a **timeout** and the client did not pin **`llm_route`**, Hub **retries once** with **`options.llm_route = "quick"`** and annotates routing debug (`memory_graph_retry_quick`, reason snippet). Reduces pain when the default **chat** lane on Circe is slow or down while **quick** is healthy.
- **Main chat UI (`app.js`):** Memory graph bridge builder uses **input budgets** (total + per-turn caps), a **client-side timeout** for the suggest `fetch`, and clearer handling when the model returns empty or clipped input.
- **Memory subview (`memory.js`):** Standalone memory graph suggest uses **bounded prompt length**, **AbortController** timeout, and user-visible notes when input was clipped.
- **Markup:** Minor **`index.html`** wrapper fix so the Mind runs modal sits in a valid DOM tree.

## Tests

- `test_handle_chat_request_turn_effect.py` — `memory_graph_suggest` timeout-shaped responses trigger quick retry; no retry when `llm_route` is already set; fixtures satisfy `CortexClientResult` shape.
- `test_memory_graph_bridge_ui.py` — extended coverage for bridge / suggest behavior.
- `test_mind_hub_tab.py` — Mind tab expectations updated for current UI/API.

## How to verify

1. Hub: open **Mind** tab, confirm recent runs load with session header; open a run in the modal.
2. Memory graph: from chat **Memory graph** bridge or Memory subview, run **Propose draft** with a long transcript — observe clipping note if applicable; with default lane timing out, confirm a second attempt can succeed on **quick** (check Hub logs for `memory_graph_suggest_retry_quick`).
3. With dev venv per `AGENTS.md`:  
   `PYTHONPATH=. ./orion_dev/bin/python -m pytest tests/test_handle_chat_request_turn_effect.py tests/test_memory_graph_bridge_ui.py tests/test_mind_hub_tab.py -q`

## Risk / rollout

- **Quick retry** only runs when timeout-like text is detected and **`llm_route` is unset**; explicit client routing is respected.
- **Orchestrator** must reach **orion-mind** where Mind is used; update real `.env` from `.env_example` as needed (do not commit secrets).
