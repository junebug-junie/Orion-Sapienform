# PR: Spark introspection Phase 3 — LLM / GPU lane isolation

## Branch

- **Head:** `feat/spark-introspection-phase3-llm-lane-isolation` (pushed to `origin`)
- **Suggested base:** `feat/spark-introspection-phase2-exec-lane-isolation` (Phase 3 builds on Phase 2 exec lanes). If Phase 2 is already merged, rebase onto `main` and retarget the PR.

Create PR: https://github.com/junebug-junie/Orion-Sapienform/pull/new/feat/spark-introspection-phase3-llm-lane-isolation

---

## Summary

Isolates **LLM gateway / model routes** from chat so spark, background, and agent traffic cannot silently consume the **chat** worker. Routing is **opt-in** via `LLM_LANE_ROUTING_ENABLED` (default `false`). Cortex-exec and spark-introspector stamp **`llm_lane`**, **`execution_lane`**, **`allow_chat_fallback`**, and related options; the gateway resolves a route-table key before calling the provider and returns **`llm_route_unavailable`** when non-chat routes are missing and chat fallback is not allowed.

---

## What changed

| Area | Change |
|------|--------|
| **orion-llm-gateway** | `lane_routes.py`: deterministic `resolve_llm_lane_route`. `run_llm_chat`: lane resolution, `llm_gateway_lane_route` / `llm_gateway_lane_rejected` logs, emergency chat warning, `raw.error=llm_route_unavailable` + `client_route` in details. Settings + `.env_example` for lane flags and route labels. |
| **orion-cortex-exec** | `llm_lane.py`: `resolve_llm_lane_for_step` (verb / execution_lane / explicit `llm_lane`); **`allow_chat_fallback`** from `ctx` / `options` when set. Executor: merge lane metadata into LLM `options`, `exec_llm_lane_decision` log; remove duplicate LLM budget resolution block. Metacog, journal pageindex, collapse-mirror paths stamp background lane. `_resolve_llm_chat_max_tokens` includes stance-brief branch. Tests + `pytest_sessionstart` token caps when service `.env` uses large `LLM_CHAT_*`. |
| **orion-spark-introspector** | Settings + worker: `SPARK_INTROSPECTION_LLM_LANE`, `ALLOW_CHAT_FALLBACK`, `MAX_TOKENS`; orch `options`; `spark_introspection_dispatch` log includes `llm_lane`. `.env_example` updated. |
| **Docs** | Plan: `docs/superpowers/plans/2026-05-13-spark-introspection-phase3-llm-gpu-lane-isolation-implementation.md` (operator notes + **per-service** pytest commands). |

---

## How to enable (operators)

1. **Gateway:** set `LLM_LANE_ROUTING_ENABLED=true` and ensure `LLM_GATEWAY_ROUTE_TABLE_JSON` includes the keys you need (`spark`, `background`, and/or `metacog` as background-class).
2. **Optional routes:** `LLM_ROUTE_SPARK_SERVED_BY`, `LLM_ROUTE_BACKGROUND_SERVED_BY`, `LLM_ROUTE_AGENT_SERVED_BY` for `served_by` table matching.
3. **Emergency chat (discouraged):** `LLM_ALLOW_BACKGROUND_TO_CHAT_FALLBACK=true` **and** per-request `allow_chat_fallback: true` (e.g. `SPARK_INTROSPECTION_ALLOW_CHAT_FALLBACK=true` flows through cortex when orch passes options into exec).
4. **Rollback:** `LLM_LANE_ROUTING_ENABLED=false`.

Copy new keys from each service **`.env_example`** into local **`.env`** (`.env` is gitignored).

---

## Verification (ran locally)

```bash
cd services/orion-llm-gateway && PYTHONPATH=. ../../venv/bin/python -m pytest tests/ -q --tb=short
# 32 passed

cd ../orion-cortex-exec && PYTHONPATH=. ../../venv/bin/python -m pytest tests/test_llm_lane_propagation.py tests/test_chat_general_route_mapping.py -q --tb=short
# 15 passed

cd ../orion-spark-introspector && PYTHONPATH=. ../../venv/bin/python -m pytest tests/ -q --tb=short
# 1 passed
```

**Not run:** live Hub / `hub_quick_playwright_dual-fast` (stack-dependent).

---

## Risk / follow-up

- PR stacks on Phase 2 branch; align base with whatever is merged first.
- Unstaged local work (e.g. `orion-hub/*`, `scripts/git-stash-table.sh`) was **not** included in this branch’s push.

---

## Suggested PR title

**feat(llm-gateway): Phase 3 LLM/GPU lane isolation (opt-in routing + cortex/spark metadata)**
