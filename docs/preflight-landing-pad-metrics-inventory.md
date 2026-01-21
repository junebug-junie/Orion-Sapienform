# Landing Pad Metrics Inventory + Recommended Visualizations

## 1) Ingest inventory (envelope kinds, reducers, outputs)

### Ingest surfaces

Landing Pad subscribes to bus channels matching allowlist patterns and filters out pad loops:
- Allowlist patterns default to `orion:telemetry:*` and `orion:cortex:*`.
- Denylist patterns default to `orion:pad:*`.
- Any envelope kind arriving on allowed channels is ingested into the normalization pipeline, with explicit reducers for certain kinds and a fallback reducer for all others.【F:services/orion-landing-pad/app/settings.py†L25-L40】【F:services/orion-landing-pad/app/pipeline/ingest.py†L72-L132】

### Explicit reducer kinds

| Envelope kind | Reducer | File path | Output | Notes |
| --- | --- | --- | --- | --- |
| `telemetry.metric.v1` | `metric_reducer` | `services/orion-landing-pad/app/reducers/stubs.py` | `PadEventV1` (`type="metric"`) | Extracts metric name/value from payload and maps to `PadEventV1`.【F:services/orion-landing-pad/app/reducers/registry.py†L16-L23】【F:services/orion-landing-pad/app/reducers/stubs.py†L10-L27】 |
| `spark.state.snapshot.v1` | `snapshot_reducer` | `services/orion-landing-pad/app/reducers/stubs.py` | `PadEventV1` (`type="snapshot"`) | Maps snapshot payload to `PadEventV1` with subject derived from `source_node`/`node`/`subject`.【F:services/orion-landing-pad/app/reducers/registry.py†L16-L23】【F:services/orion-landing-pad/app/reducers/stubs.py†L30-L49】 |
| _Any other kind_ | `fallback_reducer` | `services/orion-landing-pad/app/reducers/fallback.py` | `PadEventV1` (`type="unknown"`) | Wraps payload under `{raw, kind}` to preserve the original kind when available.【F:services/orion-landing-pad/app/reducers/fallback.py†L8-L30】 |

### Normalized outputs (schemas)

Landing Pad emits the following Pydantic models (schema source of truth):
- `PadEventV1`, `StateFrameV1`, `TensorBlobV1`, `PadRpcRequestV1`, `PadRpcResponseV1` in `orion/schemas/pad/v1.py`.【F:orion/schemas/pad/v1.py†L8-L109】

## 2) Output inventory (channels + kinds)

| Bus channel | Kind | Schema | Publisher |
| --- | --- | --- | --- |
| `orion:pad:event` | event | `PadEventV1` | `orion-landing-pad`【F:orion/bus/channels.yaml†L520-L526】 |
| `orion:pad:frame` | event | `StateFrameV1` | `orion-landing-pad`【F:orion/bus/channels.yaml†L528-L534】 |
| `orion:pad:signal` | event | `GenericPayloadV1` | `orion-landing-pad`【F:orion/bus/channels.yaml†L536-L542】 |
| `orion:pad:stats` | telemetry | `GenericPayloadV1` | `orion-landing-pad`【F:orion/bus/channels.yaml†L544-L550】 |
| `orion:pad:rpc:request` | request | `PadRpcRequestV1` | `*` (consumed by landing pad)【F:orion/bus/channels.yaml†L552-L558】 |
| `orion:pad:rpc:reply:*` | result | `PadRpcResponseV1` | `orion-landing-pad`【F:orion/bus/channels.yaml†L560-L566】 |

## 3) Storage inventory (Redis keys/streams + schema)

### Redis storage locations

Landing Pad persists event and frame data to Redis via `PadStore`:
- Latest keys:
  - `${PAD_EVENTS_STREAM_KEY}:latest`
  - `${PAD_FRAMES_STREAM_KEY}:latest`
- Streams:
  - `${PAD_EVENTS_STREAM_KEY}`
  - `${PAD_FRAMES_STREAM_KEY}`
- Streams are capped with approximate trimming (`maxlen`) and latest keys are TTL-bound by `pad_event_ttl_sec`/`pad_frame_ttl_sec`.【F:services/orion-landing-pad/app/store/redis_store.py†L14-L41】【F:services/orion-landing-pad/app/settings.py†L54-L59】

### Persisted fields

Fields persisted are the full JSON model dumps of:
- `PadEventV1`: `event_id`, `ts_ms`, `source_service`, `source_channel`, `subject`, `type`, `salience`, `confidence`, `novelty`, `payload`, `links`.
- `StateFrameV1`: `frame_id`, `ts_ms`, `window_ms`, `summary`, `state`, `salient_event_ids`, `tensor`.

These fields are defined by `orion/schemas/pad/v1.py`, which is the serialization source for Redis storage.【F:orion/schemas/pad/v1.py†L37-L77】【F:services/orion-landing-pad/app/store/redis_store.py†L31-L62】

## 4) Query surfaces (read utilities)

### Redis store read methods

`PadStore` provides these read methods (used by HTTP endpoints, RPC, and the UI):
- `get_latest_frame`, `get_frames`, `get_salient_events`, `get_latest_tensor`.
- Stream payload accessors: `get_event_payloads`, `get_frame_payloads`, `range_event_payloads`, `range_frame_payloads` for recent sampling and time-window queries.【F:services/orion-landing-pad/app/store/redis_store.py†L43-L102】

### Pad RPC methods

`PadRpcServer` supports request methods:
- `get_latest_frame`
- `get_frames`
- `get_salient_events`
- `get_latest_tensor`

Each is routed to the Redis store and replies on `orion:pad:rpc:reply:*`.【F:services/orion-landing-pad/app/rpc/server.py†L123-L183】

## 5) Live data reconnaissance (script)

A new report script scans recent Redis stream samples and prints:
- distinct `PadEventV1.type` values
- distinct envelope `kind` values when preserved in payloads
- distinct metric names (`payload.metric` / `payload.name`)
- dimension keys + cardinalities (e.g., node/service/host/etc.)
- last-seen timestamps per stream, plus redacted sample payload excerpts

Script: `services/orion-landing-pad/scripts/pad_inventory_report.py`.【F:services/orion-landing-pad/scripts/pad_inventory_report.py†L1-L140】

## 6) Visualization recommendations (grounded in stored fields)

### Metric events (PadEventV1)

1) **Salience timeseries (line)**
   - **Fields:** `ts_ms`, `salience`, group by `source_service` or `type`
   - **Why:** shows attention spikes across producers.【F:orion/schemas/pad/v1.py†L37-L54】
2) **Novelty distribution (histogram)**
   - **Fields:** `novelty` (event payloads)
   - **Why:** indicates how often the system is surprised vs. steady-state noise.【F:orion/schemas/pad/v1.py†L37-L54】
3) **Confidence P95 over time**
   - **Fields:** `confidence`, `ts_ms`
   - **Why:** highlights if incoming signals trend toward uncertain data.【F:orion/schemas/pad/v1.py†L37-L54】
4) **Event rate (count per bucket)**
   - **Fields:** `ts_ms` with `pad.event.count`
   - **Why:** monitor ingestion bursts and potential backpressure load.【F:orion/schemas/pad/v1.py†L37-L54】

### Frames (StateFrameV1)

5) **Frame cadence (line)**
   - **Fields:** `ts_ms` from `StateFrameV1`
   - **Why:** verify frame ticker cadence and gaps.【F:orion/schemas/pad/v1.py†L80-L97】
6) **Salient-event density (line/histogram)**
   - **Fields:** `salient_event_ids` length per frame
   - **Why:** visualize “busy” cognitive windows vs. quiet windows.【F:orion/schemas/pad/v1.py†L80-L97】
7) **Tensor dimension watch**
   - **Fields:** `tensor.dim`
   - **Why:** ensure tensorizer configuration is consistent across frames.【F:orion/schemas/pad/v1.py†L57-L77】

### Signals + stats (bus outputs)

8) **Signal timeline table**
   - **Fields:** `orion:pad:signal` events containing `event_id` + `salience`
   - **Why:** track high-salience pulses and correlate to event timeline.【F:services/orion-landing-pad/app/service.py†L133-L146】
9) **Stats trend (queue depth / frames built)**
   - **Fields:** `PadStatsTracker` counters + gauges
   - **Why:** detect throttling and queue pressure over time.【F:services/orion-landing-pad/app/observability/stats.py†L8-L49】【F:services/orion-landing-pad/app/service.py†L193-L207】

## 7) Metrics Explorer UI (landing pad service)

The Landing Pad HTTP server now exposes a lightweight UI and API under both root and `/landing-pad`, mirroring the Spark Introspector dual-mount strategy for subpath hosting (works whether a reverse proxy strips or preserves the prefix). The routes are registered inside the Landing Pad FastAPI app and use Redis store read methods rather than raw Redis access in the API layer.【F:services/orion-landing-pad/app/main.py†L16-L20】【F:services/orion-landing-pad/app/web/main.py†L14-L24】【F:services/orion-landing-pad/app/web/api.py†L145-L239】

### API endpoints

- `GET /api/metrics` and `/landing-pad/api/metrics`
- `GET /api/dimensions` and `/landing-pad/api/dimensions`
- `GET /api/query` and `/landing-pad/api/query`
- `GET /healthz` and `/landing-pad/healthz`
- `GET /ui` and `/landing-pad/ui` (serves the Plotly UI)

All frontend fetches are relative paths (no leading `/`), so the UI works on both mounts.【F:services/orion-landing-pad/app/web/api.py†L145-L239】【F:services/orion-landing-pad/app/static/landing_pad/index.html†L120-L219】

## 8) Run instructions (Landing Pad service)

### Local run (uvicorn)

```bash
cd services/orion-landing-pad
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8370
```

Open:
- `http://localhost:8370/landing-pad/ui`
- Fallback if prefix stripped: `http://localhost:8370/ui`

### Docker Compose

```bash
cd services/orion-landing-pad
cp .env_example .env
docker compose up --build
```

### Tailscale serve (subpath)

```bash
sudo tailscale serve --https=443 /landing-pad http://127.0.0.1:8370
```

Open:
- `https://<node>.ts.net/landing-pad/ui`
- Fallback if prefix stripped by proxy: `https://<node>.ts.net/ui`
