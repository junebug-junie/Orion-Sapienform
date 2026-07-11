# Turn-change classify hardening — Phase 1 design

**Status:** approved  
**Date:** 2026-06-22

## Problem

Turn-change appraisal scores four-line LLM output via **logprob token parsing**. Thinking models emit a long reasoning prefix before the answer; even when gateway strips `content`, the **logprob stream** can include think tokens and mis-score labels (same failure class as BPE-split `NOVEL:`).

Chat compute may run with thinking enabled (`enable_thinking: true` on Qwen-style models). That lane must not own turn-change classify.

## Decision

Route turn-change classify to **`metacog`** by default: atlas-worker-2, `llama-3-8b-instruct-q4_k_m`, instruct-only, no thinking.

Classify remains a **separate async RPC** off the chat hot path (`memory:turn:persisted` → consolidation worker → gateway).

Every classify request sets `chat_template_kwargs: {"enable_thinking": false}` regardless of route (belt-and-suspenders).

## Phase 1 scope

### A — Classify contract

| Item | Notes |
|------|-------|
| `TURN_CHANGE_CLASSIFY_ROUTE` | Default `metacog`; optional `quick` for bake-off |
| Thinking models | **Out of scope** for classify; no escalation tier |
| BPE-safe logprob parser | Already on `main` (`boundary.py`) |
| `reconcile_novelty_with_shift` | Already on `main` |
| Golden turn suite | CI regression from live chat fixtures (follow-up) |
| Audit on patch | `turn_change_classify_route` on spark_meta patch |

### B — Delivery (follow-up)

| Item | Notes |
|------|-------|
| Spark telemetry re-emit | When `spark_meta:patch` lands, refresh trace `novelty` |
| `novelty_source` | Optional field distinguishing tissue vs appraisal |

### Non-goals

- Thinking-model escalation on metacog
- Routing classify through `chat` compute lane
- Hub/mind/cortex latency reduction
- `memory_graph_suggest` JSON rail (separate fix)

## Env contract

| Key | Default | Purpose |
|-----|---------|---------|
| `TURN_CHANGE_CLASSIFY_ROUTE` | `metacog` | Gateway route for classify RPC |
| `MEMORY_CLASSIFY_TIMEOUT_SEC` | `8.0` | RPC timeout (unchanged in Phase 1) |
| `TURN_CHANGE_CONFIDENCE_MARGIN` | `0.15` | Session-window reappraisal threshold |
| `TURN_CHANGE_SUBSTRATE_THRESHOLD` | `0.65` | Substrate signal emit floor |
| `TURN_CHANGE_WINDOW_TURNS` | `3` | Session-window baseline depth |

## Acceptance

1. Classify RPC uses `route=metacog` and `enable_thinking: false` by default.
2. Patch includes `turn_change_classify_route: "metacog"`.
3. Unit tests assert gateway payload route and thinking disabled.
4. Existing classify tests pass unchanged (mocked bus).

## Risks

- Metacog worker saturation if classify volume grows (acceptable; async).
- `quick` bake-off may show faster but less accurate scores — env switch only, no code fork.
