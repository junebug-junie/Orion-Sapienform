# orion-self-experiments

Typed self-experiment registry (intake + validation only).

Experiments used to compile to a `ContextExecRequestV1` and dispatch to
`orion-context-exec`. That service was retired 2026-10-10 and the dispatch/retry
routes went with it, so experiments are validated and stored but never executed.

Also publishes a bus-native `SystemHealthV1` heartbeat to `orion:system:health` every
`HEARTBEAT_INTERVAL_SEC` (default 10s) via its own independent Redis connection when
`ORION_BUS_ENABLED=true`.

## Responsibilities

- Accept legacy `skill_id` probes and typed `SelfExperimentCreateRequestV1` payloads
- Validate against the deterministic experiment registry (no keyword routing)
- Store lifecycle state

## API

- `POST /v1/experiments` — create candidate
- `GET /v1/experiments/{id}` — fetch record
- `GET /v1/experiments` — list with filters
- `POST /v1/experiments/{id}/discard` — discard

## Config

Copy `.env_example` to `.env`.

## Tests

```bash
PYTHONPATH=. ./venv/bin/python -m pytest services/orion-self-experiments/tests -q
```
