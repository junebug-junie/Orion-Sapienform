# orion-bus — Agent Context

## Role
`orion-bus` is **transport_infrastructure** (mesh nervous-system conduit), not an organ.

## Native contracts
- Redis Streams / pub-sub (`redis_transport`)
- Prometheus metrics via `redis_exporter` (`redis_exporter_metrics`)
- Operator `redis-cli` inspection (`operator_stream_inspection`) — no substrate trace by default

## Substrate trace stance
Emit **bounded periodic transport rollups** only:
health, stream depth, backpressure, uncataloged configured streams, and a
bounded per-stream schema-validation sample (`bus_schema_validation_failed`:
counts only -- `mismatch_count`/`sampled_count` from a small `XREVRANGE`
sample of cataloged streams checked against their declared `schema_id`, see
`BUS_OBSERVER_SCHEMA_SAMPLE_COUNT`).
Never emit full message payloads or per-packet traces.

## Implementation
- `bus-core` + `bus-exporter` — unchanged Redis stack
- `bus-observer` (optional) — Python sidecar emitting `GrammarEventV1` on `orion:grammar:event` when enabled

## Publishing
Default `PUBLISH_ORION_BUS_GRAMMAR=false`. Fail-open on publish errors.

## RPC + long-lived subscribers
Services that run Hunter/Rabbit `subscribe()` loops **and** outbound `rpc_request()` on the same
`OrionBusAsync` instance can lose replies (stolen by trace caches or torn down by overlapping
`listen()`). Pattern:

```python
from orion.core.bus.rpc_fork import fork_rpc_client

rpc_bus = await fork_rpc_client(listener_bus)  # dedicated worker pubsub
# keep listener_bus for intake/publish; route all rpc_request via rpc_bus
```

Hub, cortex-gateway, cortex-orch, cortex-exec, context-exec, actions, chat-memory, vision-council,
and agent-chain follow this split as of the bus RPC hardening pass.

## Downstream (deferred)
`bus_transport_reducer` → `StateDeltaV1(target_kind=transport_bus)` → field pressure hints.
