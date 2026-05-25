# PR: Orion Bus Substrate Trace — Transport Legibility v1

**Branch:** `feat/orion-bus-substrate-trace-v1`  
**Base:** `main`  
**Worktree:** `/mnt/scripts/Orion-Sapienform/.worktrees/feat-orion-bus-substrate-trace-v1`  
**Commits:** `322ec78b`, `ee810551`

## Summary

Adds optional `bus-observer` sidecar under `services/orion-bus` for **bounded periodic** transport `GrammarEventV1` rollups. Redis `bus-core` and `bus-exporter` are unchanged.

```text
bus-observer (poll) → PING + XLEN → GrammarEventV1 → orion:grammar:event (when enabled)
```

**Not a packet log.** Default publish off.

## service role

`transport_infrastructure`

## native contract

Redis core + Redis exporter (+ optional bus-observer Python sidecar)

## substrate trace stance

Bounded transport rollups and anomalies only — health, stream depth, backpressure, uncataloged configured streams.

## implemented roles

- `bus_observer_tick_started`
- `bus_health_observed`
- `bus_stream_depth_observed`
- `bus_backpressure_observed`
- `bus_configured_stream_uncataloged`
- `bus_observer_tick_completed`
- `bus_observer_tick_failed`

## deferred roles

- `bus_stream_lag_observed`
- `bus_schema_validation_failed`
- `bus_delivery_anomaly_observed`
- `bus_metrics_scrape_observed` / `bus_metrics_scrape_failed` / `bus_memory_pressure_observed`

## tests run

```text
PYTHONPATH=services/orion-bus:. pytest services/orion-bus/tests/ -q
7 passed

PYTHONPATH=. python -m compileall services/orion-bus -q
ok
```

## live proof

`not_verified_live` — operator may enable `PUBLISH_ORION_BUS_GRAMMAR=true` and run:

```sql
select created_at, source_service, trace_id,
  event_json::jsonb #>> '{atom,semantic_role}' as semantic_role
from grammar_events
where source_service = 'orion-bus'
  and trace_id like 'bus.transport:%'
order by created_at desc limit 30;
```

## downstream follow-up

`bus_transport_reducer` → `StateDeltaV1(target_kind=transport_bus)` → transport pressure → field digestion (Layer 3–4 deferred).

## Files changed

| Path | Role |
|------|------|
| `services/orion-bus/app/*` | Observer, grammar emit/publish |
| `services/orion-bus/AGENT_CONTEXT.md` etc. | Context-engineering |
| `orion/bus/channels.yaml` | `orion-bus` grammar producer |
| `scripts/smoke_orion_bus_substrate_trace.sh` | Smoke |

## Code review

Subagent review: **APPROVED** (9/9 plan gates).
