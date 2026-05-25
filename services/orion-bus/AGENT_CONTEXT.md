# orion-bus — Agent Context

## Role
`orion-bus` is **transport_infrastructure** (mesh nervous-system conduit), not an organ.

## Native contracts
- Redis Streams / pub-sub (`redis_transport`)
- Prometheus metrics via `redis_exporter` (`redis_exporter_metrics`)
- Operator `redis-cli` inspection (`operator_stream_inspection`) — no substrate trace by default

## Substrate trace stance
Emit **bounded periodic transport rollups** only:
health, stream depth, backpressure, uncataloged configured streams.
Never emit full message payloads or per-packet traces.

## Implementation
- `bus-core` + `bus-exporter` — unchanged Redis stack
- `bus-observer` (optional) — Python sidecar emitting `GrammarEventV1` on `orion:grammar:event` when enabled

## Publishing
Default `PUBLISH_ORION_BUS_GRAMMAR=false`. Fail-open on publish errors.

## Downstream (deferred)
`bus_transport_reducer` → `StateDeltaV1(target_kind=transport_bus)` → field pressure hints.
