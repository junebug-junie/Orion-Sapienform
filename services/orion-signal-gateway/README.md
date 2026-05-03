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

Requires the `app-net` Docker network to exist (see `docker-compose.yml`).

The container runs Uvicorn on **port 8000**. Compose publishes it on the host as **`SIGNAL_GATEWAY_HTTP_PORT` → 8000** (default **8879** to stay clear of common **809x** ports). Override in `.env` if needed. Gateway URL: `http://localhost:8879` with the defaults.

### Observability stack (same compose file)

When you run `docker compose up` in this directory, the following optional services start alongside the gateway (all on Docker network **`app-net`**):

| Service | Role | Default host ports |
|--------|------|---------------------|
| **otel-collector** | Receives OTLP from the gateway app; forwards traces to Tempo; exposes Prometheus scrape on **`:8889`** | `4317` (gRPC), `4318` (HTTP), `8889` (metrics) |
| **otel-tempo** | Trace backend (Grafana Tempo), OTLP ingest | `3200` (HTTP query API), `14317` (OTLP gRPC on host — avoids clashing with the collector’s `4317`), `14318` (OTLP HTTP) |
| **otel-grafana** | Operator UI; Tempo datasource is provisioned as uid **`tempo`** | `3001` → container `3000` (`GRAFANA_HTTP_PORT`) |

**Gateway app OTLP:** Compose passes **`OTEL_EXPORTER_OTLP_ENDPOINT`** (default `http://otel-collector:4317`), **`OTEL_CONSOLE_EXPORT`**, and **`OTEL_DIMENSION_ALLOWLIST`** into **`orion-signal-gateway`** so spans export to the collector without extra shell wiring. Override values via `.env` next to this compose file.

**Hub deep links:** Configure Hub with **`HUB_OTEL_GRAFANA_BASE_URL`** (e.g. `http://127.0.0.1:3001`) and, if needed, **`HUB_OTEL_GRAFANA_ORG_ID`** (default `1` for single-org Grafana) so `/api/signals/trace/{trace_id}` and `/api/observability/grafana-tempo-trace/{trace_id}` return Grafana Explore URLs for the **32-character lowercase hex** `otel_trace_id`. See `services/orion-hub/.env_example`.

**Metrics:** this compose file does **not** run a Prometheus server. Scrape the collector on the host at **`http://127.0.0.1:8889/metrics`** from your own Prometheus (or add a `prometheus` service in a follow-up).

**Security / Grafana auth:** by default, compose sets **`GRAFANA_ANONYMOUS_ENABLED=true`** (via `.env`) for local use. For any shared or routed network, set **`GRAFANA_ANONYMOUS_ENABLED=false`**, set **`GF_SECURITY_ADMIN_PASSWORD`** (or your org’s auth), and set **`GF_AUTH_DISABLE_LOGIN_FORM=false`** so login is not disabled while anonymous is off. See [Grafana configuration](https://grafana.com/docs/grafana/latest/setup-grafana/configure-grafana/).

**Phase 1 smoke (spec §8.1):** With compose up and default host ports, run `bash scripts/smoke_otel_phase1.sh` (Tempo `/ready`, Grafana `/`, collector metrics). Then run `python3 scripts/e2e_otel_phase1.py` to **POST minimal OTLP/HTTP JSON** to `:4318` and assert the trace appears in Tempo (`/api/traces/...`). That covers collector → Tempo without live gateway traffic. Full §8.1 also includes gateway-emitted spans, **`OrionSignalV1.otel_trace_id`**, Hub `/api/signals/trace/{id}` as the **Orion causal chain** (not an OTEL span tree), and opening the Hub-generated Explore link in Grafana.

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
| `SIGNAL_GATEWAY_HTTP_PORT` | `8879`                   | Host port mapped to container `8000` (compose / curl) |
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
| `OTEL_DIMENSION_ALLOWLIST` | JSON list (see defaults in `app/settings.py`) | Keys permitted on spans as `dim.*`; metric keys ending in `_level`, `_trend`, `_volatility` are also emitted. Never attaches `summary`, raw payload fields, or `notes` to spans. |
| `TEMPO_HTTP_PORT` | `3200` | Compose only: host port for Tempo HTTP/query API. |
| `TEMPO_OTLP_GRPC_PORT` | `14317` | Compose only: host OTLP gRPC (differs from collector `4317` to avoid clashes). |
| `TEMPO_OTLP_HTTP_PORT` | `14318` | Compose only: host OTLP HTTP. |
| `GRAFANA_HTTP_PORT` | `3001` | Compose only: host port for Grafana. |
| `GRAFANA_ANONYMOUS_ENABLED` | `true` | Compose only: Grafana anonymous login (set `false` + admin password on shared networks). |

## OpenTelemetry

Adapter-emitted signals get a span named `signal.{organ_id}.{signal_kind}` with attributes
`organ_id`, `organ_class`, `signal_kind`, `correlation_id`, and `dim.*` (spec §5). Hybrid and
endogenous signals inherit `trace_id` from the **first registry-listed parent** (``causal_parent_organs`` order) that has
`otel_trace_id` / `otel_span_id` in the current `SignalWindow`. Exogenous signals start a new trace.
If multiple parents carry OTEL context and **trace_ids disagree**, the gateway keeps first-wins for span linkage, emits a structured **warning** log (`orion_signal_parent_trace_disagreement`), and appends a **note** to the signal when a slot is free under the five-note cap.

## Multi-instance deployment (operator checklist)

Supported models:

- **(A) Singleton writer:** one gateway replica consumes organ bus traffic and publishes `orion:signals:{organ_id}` (recommended default).
- **(B) N replicas:** only with **downstream dedupe** by `signal_id` **and** gateway-side partitioning or leader election so **at most one** replica adapts a given raw channel.

**Unsupported:** multiple full subscribers republishing the same raw events with no dedupe or partitioning (duplicate `signal_id` / conflicting parent resolution).

Compose / Helm comments should state which model applies. Do not advertise multi-replica HA without meeting **(B)**.

## Bus semantics (Redis)

Default wiring uses Redis **pub/sub** (see mesh compose). Expect **at-most-once** delivery per subscriber and **no cross-channel ordering** guarantee. Under load, the gateway processes envelopes inline; slow adapters increase lag and can affect **parent resolution** inside `SIGNAL_WINDOW_SEC` (stale or missing parents if the window is exceeded). Reordering across channels can break lineage expectations—size the window and organ publish rates accordingly.

## Signal window vs `ttl_ms` vs Hub

| Concept | Role |
|--------|------|
| `ttl_ms` on `OrionSignalV1` | Hint for consumers (default 15s in schema). |
| `SIGNAL_WINDOW_SEC` | Gateway in-memory horizon for `prior_signals` / parent lookup (default 30s). |
| Hub cache (phase 2b) | Should be **≥** gateway window or explicitly aligned so operators do not see newer gateway state than Hub. |

## Graduation / dedupe

First-party consumers should treat **`signal_id`** (and when present **`source_event_id`**) as the idempotency key. Set **`SUPPRESS_ADAPTED_WHEN_PASSTHROUGH=true`** (with **`PASSTHROUGH_DEDUPE_WINDOW_SEC`**) to skip adapter emits when a passthrough for the same **`(organ_id, source_event_id)`** was seen recently; otherwise consumers dedupe on ingest.

## Self-hardening graduation

An organ publishes a validated `OrionSignalV1` payload using envelope kind
`signal.{organ_id}.{signal_kind}` on `orion:signals:{organ_id}`. The gateway treats that as
passthrough (no adapter), re-emits it, and skips adapter spans. Events whose kind is not this
prefix (for example `spark.signal.v1`) continue through organ adapters.

**Collector vs organ-bus biometrics:** the mesh causal chain uses **gateway-emitted biometrics signals**; Prometheus / DCGM sidecars remain parallel infra telemetry unless explicitly wired as parents in the registry.
