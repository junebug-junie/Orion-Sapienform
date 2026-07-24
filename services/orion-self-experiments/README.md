# orion-self-experiments

Typed self-experiment registry and context-exec dispatcher.

Also publishes a bus-native `SystemHealthV1` heartbeat to `orion:system:health` every
`HEARTBEAT_INTERVAL_SEC` (default 10s) via its own independent Redis connection when
`ORION_BUS_ENABLED=true`.

## Responsibilities

- Accept legacy `skill_id` probes and typed `SelfExperimentCreateRequestV1` payloads
- Validate against the deterministic experiment registry (no keyword routing)
- Compile `ContextExecRequestV1` and dispatch to context-exec (bus or HTTP)
- Store lifecycle state, artifacts, and proposal linkage

## API

- `POST /v1/experiments` — create candidate
- `POST /v1/experiments/{id}/dispatch` — compile + dispatch (when enabled)
- `GET /v1/experiments/{id}` — fetch record
- `GET /v1/experiments` — list with filters
- `POST /v1/experiments/{id}/discard` — discard
- `POST /v1/experiments/{id}/retry` — retry failed/queued dispatch

## Config

Copy `.env_example` to `.env`. Dispatch is **disabled by default** (`SELF_EXPERIMENTS_DISPATCH_ENABLED=false`).

## Smoke

```bash
./scripts/self_experiment_context_exec_smoke.sh
```

## Tests

```bash
PYTHONPATH=. ./venv/bin/python -m pytest services/orion-self-experiments/tests -q
```
