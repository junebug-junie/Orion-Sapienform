# Context Pack: orion-substrate-telemetry

## Goal
Verify substrate tier telemetry persistence matches spec

## Current accepted facts
- orion-substrate-telemetry subscribes to orion:substrate:tier_outcomes and append-only persists tier outcome rows to Postgres. (`claim:orion:substrate-telemetry:0001`)
- orion-cortex-orch optionally HTTP-fetches persisted substrate telemetry and merges into MindRunRequestV1.snapshot_inputs.facets.substrate_telemetry before calling Mind. (`claim:orion:substrate-telemetry:0002`)

## Required behavior
- Subscribe to orion:substrate:tier_outcomes via BaseChassis
- Validate kind substrate.tier_outcomes.v1 before insert
- Expose GET /v1/substrate/tier-outcomes/latest and /history by correlation_id
- Orch merges facet when ORION_SUBSTRATE_TELEMETRY_BASE_URL is set

## Non-goals
- Hub Redis bus subscription for tier outcomes
- Idempotent dedupe of duplicate bus deliveries in v0

## Relevant files
- `services/orion-substrate-telemetry/app/main.py`
- `services/orion-cortex-orch/app/mind_runtime.py`
- `orion/bus/channels.yaml`

## Acceptance tests
- PYTHONPATH=. ./venv/bin/python -m pytest services/orion-substrate-telemetry/tests/ -q
- PYTHONPATH=. ./venv/bin/python -m pytest services/orion-cortex-orch/tests/test_substrate_telemetry_orch.py -q

## Known traps
- Duplicate bus deliveries create multiple rows; dedupe is deferred
- Do not block BaseChassis loop on bad frames
