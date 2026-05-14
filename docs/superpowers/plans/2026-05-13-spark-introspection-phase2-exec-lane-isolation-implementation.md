# Spark Introspection Phase 2 — Exec lane isolation (implementation)

**Status:** Implemented in repo (2026-05-13).

**Design reference:** `docs/superpowers/specs/2026-05-13-spark-introspection-lane-isolation-design.md` (high-level lane program); detailed Phase 2 exec contract was the Cursor prompt in the parent chat.

## Operator notes

- **`EXEC_LANE_ROUTING_ENABLED`** defaults to **`false`** on cortex-orch so existing single `cortex-exec` stacks keep working. Set **`true`** when running one exec consumer per lane (`cortex-exec-chat`, `cortex-exec-spark`, `cortex-exec-background`).
- **Chat** traffic stays on **`orion:verb:request`** (only `EXEC_LANE=chat` or `legacy` exec processes subscribe). **Spark** and **background** use **direct PlanExecution RPC** to `orion:cortex:exec:request:{spark|background}`.
- **`orion:cortex:exec:request:chat`:** With current orch, **Hub chat is not published on this channel**; chat remains verb-driven. A `cortex-exec-chat` consumer on `:chat` is only needed if another producer sends PlanExecution there or you add a future direct-chat orch path—do not assume three symmetric “orch → PlanExecution” lanes.
- **Rollback:** `EXEC_LANE_ROUTING_ENABLED=false` and a single exec on `CHANNEL_EXEC_REQUEST=orion:cortex:exec:request`.

## Verification

See repo **Verification** in the agent final message (pytest + `docker compose config`).
