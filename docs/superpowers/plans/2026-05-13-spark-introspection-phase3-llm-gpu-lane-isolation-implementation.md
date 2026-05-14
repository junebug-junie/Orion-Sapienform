# Spark Introspection Phase 3 — LLM / GPU lane isolation (implementation)

**Status:** Implemented in repo (2026-05-13).

**Design reference:** User-provided Phase 3 spec (2026-05-13): LLM/GPU lanes `chat`, `spark`, `background`, `agent`; no silent non-chat → chat fallback; explicit `llm_route_unavailable` when routes missing.

## Operator notes

- **`LLM_LANE_ROUTING_ENABLED`** defaults to **`false`**. Set **`true`** on `orion-llm-gateway` when the route table defines the lanes you want isolated (add `spark` / `background` keys to `LLM_GATEWAY_ROUTE_TABLE_JSON`, or rely on `metacog` as legacy **background**-class key).
- **Rollback:** `LLM_LANE_ROUTING_ENABLED=false` restores pre–Phase-3 routing (route key from payload only).
- **Emergency chat sharing (discouraged):** `LLM_ALLOW_BACKGROUND_TO_CHAT_FALLBACK=true` **and** per-request `allow_chat_fallback: true`. Upstream may set `allow_chat_fallback` in `ctx.options` / `ctx` (e.g. `SPARK_INTROSPECTION_ALLOW_CHAT_FALLBACK`); cortex-exec passes it through to the gateway unless overridden.

## Tasks (executable)

1. **Plan file** — This document (spec captured; steps below).
2. **Gateway settings + `.env_example`** — `LLM_ROUTE_*_SERVED_BY` extensions, `LLM_ALLOW_BACKGROUND_TO_CHAT_FALLBACK`, `LLM_LANE_DEFAULT`, `LLM_LANE_ROUTING_ENABLED`.
3. **`lane_routes.py` + `tests/test_lane_routes.py`** — Deterministic `resolve_llm_lane_route`; map lanes to route-table keys (`background` → `background` or `metacog`); chat fallback only when both flags true.
4. **`llm_backend.run_llm_chat`** — If routing enabled and route table non-empty: resolve lane → set effective route key → on failure return `raw.error=llm_route_unavailable`; structured log `llm_gateway_lane_route` / `llm_gateway_lane_rejected`.
5. **Cortex-exec** — `app/llm_lane.py` with `resolve_llm_lane_for_step`; executor merges lane options into `ChatRequestPayload.options`; log `exec_llm_lane_decision`; `tests/test_llm_lane_propagation.py`.
6. **Spark-introspector** — `SPARK_INTROSPECTION_LLM_LANE`, `SPARK_INTROSPECTION_ALLOW_CHAT_FALLBACK`, `SPARK_INTROSPECTION_MAX_TOKENS`; stamp orch `options`; log `spark_introspection_dispatch` with `llm_lane`; `tests/test_spark_llm_lane_metadata.py`.
7. **Verification** — Run pytest **from each service directory** (multiple services define a top-level `app` package; a single root `PYTHONPATH=.` collect often fails).

```bash
cd /path/to/Orion-Sapienform/services/orion-llm-gateway
PYTHONPATH=. ../../venv/bin/python -m pytest tests/test_lane_routes.py tests/test_llm_lane_run_llm_chat.py -q --tb=short

cd ../orion-cortex-exec
PYTHONPATH=. ../../venv/bin/python -m pytest tests/test_llm_lane_propagation.py tests/test_chat_general_route_mapping.py -q --tb=short

cd ../orion-spark-introspector
PYTHONPATH=. ../../venv/bin/python -m pytest tests/test_spark_llm_lane_metadata.py -q --tb=short
```

Or use `./scripts/test_service.sh <service>` per `AGENTS.md`.

## Acceptance (Phase 3)

- With routing **on** and spark/background routes absent, spark-lane requests fail fast with `llm_route_unavailable` (no HTTP call to chat).
- With routing **off**, behavior matches Phase 2 route selection.
- Logs: `exec_llm_lane_decision`, `llm_gateway_lane_route` / `llm_gateway_lane_rejected`, `spark_introspection_dispatch` includes `llm_lane`.
