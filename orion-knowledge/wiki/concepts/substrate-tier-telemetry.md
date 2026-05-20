# Substrate tier telemetry

Compiled view of accepted claims for the substrate tier telemetry persistence design.

## Persistence service

`orion-substrate-telemetry` subscribes to `orion:substrate:tier_outcomes` and append-only persists tier outcome rows to Postgres (`claim:orion:substrate-telemetry:0001`).

## Orch integration

`orion-cortex-orch` optionally HTTP-fetches persisted substrate telemetry and merges into `MindRunRequestV1.snapshot_inputs.facets.substrate_telemetry` before calling Mind (`claim:orion:substrate-telemetry:0002`).

## Execution spec

See `spec:substrate-tier-telemetry-v1` for requirements and acceptance tests.
