# orion-signal-gateway

Normalizes raw organ-bus events into `OrionSignalV1` objects and publishes them to the signal bus.

## What it does

Each organ publishes raw telemetry to the Orion bus (Redis pub/sub). This gateway subscribes,
routes each event through the matching adapter, normalizes dimensions via EWMA bands, and emits
a typed `OrionSignalV1` to `orion:signals` (and organ-specific sub-channels).

## Architecture

```
Orion Bus (Redis)
  └─ organ event (e.g. biometrics.induction.v1)
       └─ AdapterRegistry.route()
            └─ BiometricsAdapter.adapt()  →  OrionSignalV1
                 └─ SignalWindow.put()
                      └─ publish → orion:signals.*
```

Causal parents are resolved from the `SignalWindow` — the most recent non-stale signal per organ
is available for endogenous/hybrid organs to reference.

## Running

### Docker Compose

```bash
cd services/orion-signal-gateway
docker-compose up --build
```

Requires the `orion-net` Docker network to exist. Service is reachable at `http://localhost:8090`.

### Local (uvicorn)

```bash
cd /path/to/repo
pip install -r services/orion-signal-gateway/requirements.txt
PYTHONPATH=. uvicorn app.main:app --host 0.0.0.0 --port 8000 \
  --app-dir services/orion-signal-gateway
```

## Key endpoints

| Method | Path             | Description                              |
|--------|------------------|------------------------------------------|
| GET    | `/health`        | Liveness check — returns `{"status":"ok"}` |
| GET    | `/signals/active`| All non-stale signals in the current window |

## Environment variables

Copy `.env_example` to `.env` and adjust:

| Variable               | Default                      | Description                       |
|------------------------|------------------------------|-----------------------------------|
| `SERVICE_NAME`         | `orion-signal-gateway`       | Reported in OTEL spans            |
| `SERVICE_VERSION`      | `0.1.0`                      |                                   |
| `NODE_NAME`            | `athena`                     | Physical node identifier          |
| `ORION_BUS_URL`        | `redis://orion-redis:6379/0` | Redis connection string           |
| `ORION_BUS_ENABLED`    | `true`                       | Set `false` to disable bus I/O    |
| `LOG_LEVEL`            | `INFO`                       |                                   |
| `SIGNAL_WINDOW_SEC`    | `30.0`                       | TTL for the in-memory signal window |
| `SIGNALS_OUTPUT_CHANNEL` | `orion:signals`            | Base pub/sub channel for output   |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | _(unset)_             | OTLP gRPC endpoint (e.g. `http://otel-collector:4317`) |
| `OTEL_CONSOLE_EXPORT`    | `false`                      | Log spans to stdout (dev only)    |

## OpenTelemetry

Adapter-emitted signals get a span named `signal.{organ_id}.{signal_kind}` with attributes
`organ_id`, `organ_class`, `signal_kind`, `correlation_id`, and `dim.*` (spec §5). Hybrid and
endogenous signals inherit `trace_id` from the first registry-listed parent organ that has
`otel_trace_id` / `otel_span_id` in the current `SignalWindow`. Exogenous signals start a new trace.

## Self-hardening graduation

An organ publishes a validated `OrionSignalV1` payload using envelope kind
`signal.{organ_id}.{signal_kind}` on `orion:signals:{organ_id}`. The gateway treats that as
passthrough (no adapter), re-emits it, and skips adapter spans. Events whose kind is not this
prefix (for example `spark.signal.v1`) continue through organ adapters.
