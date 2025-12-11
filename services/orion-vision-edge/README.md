# Orion Vision Edge

## services/orion-vision-edge/README.md

```markdown
# 🛰 Orion Vision Edge (Athena)

The **Orion Vision Edge** service runs on Athena as a lightweight, always-on
perception node. It handles:

- RTSP / camera capture (e.g. Reolink, GoPro, etc.)
- Frame-level detectors:
  - Motion
  - Face detection (Haar cascade)
  - YOLO object detection (people, etc.)
- Event-level detectors:
  - Presence (e.g. “Juniper present / absent”)
- Optional HTML UI for local debugging:
  - MJPEG live stream with overlays
  - Snapshot endpoint
  - SSE events pane (for dev)

All detections are normalized into a common schema and published onto the
**Orion bus** for downstream consumers like:

- `orion-security-watcher` (visit + alert logic)
- Future `orion-vision-identity` (who is this?)
- Future `orion-vision-affect` (emotion / posture / vibes)
- Cortex Orchestrator perception verbs

This service is **edge-first**: it does just enough work on Athena to be useful,
then gets out of the way so heavy cognition can happen on GPU nodes.

---

## Architecture

High-level flow:

1. **CameraSource**
   - Connects to `SOURCE` (usually an RTSP URL).
   - Runs in a background thread, keeping `last_frame` updated.

2. **Detector Worker**
   - Async loop that:
     - Reads frames from `CameraSource`.
     - Every `DETECT_EVERY_N_FRAMES` frames:
       - Runs all **frame detectors** (`motion`, `face`, `yolo`, etc.).
       - Runs all **event detectors** (`presence`, future identity hooks).
     - Builds a structured `Event` object with:
       - `ts` (UTC)
       - `stream_id`
       - `frame_index`
       - list of `Detection` objects
       - optional `meta`
     - Publishes the event to a **bus channel**:
       - raw lane: `VISION_EVENTS_PUBLISH_RAW` (e.g. `orion:vision:edge:raw`)

3. **HTTP API & UI**
   - `/health` – basic health + config
   - `/snapshot.jpg` – latest frame as JPEG
   - `/stream.mjpg` – MJPEG live stream (for debugging)
   - `/events` – SSE dev stream of bus messages
   - `/detect` – upload an image and run the same detector pipeline

The bus-first pattern means **everything interesting flows through Redis**, not
direct HTTP calls from other services.

---

## Bus Channels

The Vision Edge service currently uses:

- **Publish**
  - `VISION_EVENTS_PUBLISH_RAW` (env)
    - Default: `orion:vision:edge:raw`
    - Payload: normalized vision `Event` with detections
  - `vision.detect.upload`
    - On-demand detection results for uploaded images (compat / dev)

- **Subscribe**
  - None for v1. This service is purely a producer.

Downstream examples:

- `orion-security-watcher` subscribes to `orion:vision:edge:raw`.
- Future: `orion-vision-identity`, `orion-vision-affect`, Cortex perception verbs.

---

## Event & Detection Schema

Pydantic models live in `app/schemas.py`:

```python
class Detection(BaseModel):
    kind: str                # "motion", "face", "yolo", "presence", etc.
    bbox: tuple[int, int, int, int]  # x, y, w, h
    score: float = 1.0
    label: str | None = None
    meta: dict[str, Any] | None = None

class Event(BaseModel):
    ts: datetime
    stream_id: str
    frame_index: int
    detections: list[Detection] = []
    meta: dict[str, Any] | None = None
```

On the bus, we publish a JSON-safe dict:

```json
{
  "ts": "2025-12-10T23:07:23.123456Z",
  "stream_id": "cam0",
  "frame_index": 1234,
  "detections": [
    {"kind": "face", "bbox": [x, y, w, h], "score": 0.99, "label": null},
    {"kind": "yolo", "bbox": [x, y, w, h], "score": 0.91, "label": "person"}
  ],
  "meta": {
    "source": "edge",
    "camera": "rtsp://..."
  }
}
```

---

## Configuration

All configuration is driven by environment variables via Pydantic Settings.

### Core camera & stream config

| Env var                 | Type    | Default | Description                                   |
|-------------------------|---------|---------|-----------------------------------------------|
| `SOURCE`                | str     | **req** | RTSP URL or device index                      |
| `WIDTH`                 | int     | 640     | Capture width in pixels                       |
| `HEIGHT`                | int     | 360     | Capture height in pixels                      |
| `FPS`                   | int     | 15      | Target frames per second                      |
| `DETECT_EVERY_N_FRAMES` | int     | 10      | Run detectors every N frames                  |
| `STREAM_ID`             | str     | `cam0`  | Logical name for this camera stream           |

### Detectors & thresholds

| Env var              | Type   | Default           | Description                                  |
|----------------------|--------|-------------------|----------------------------------------------|
| `DETECTORS`          | str    | `face,yolo`       | Comma list: `motion`, `face`, `yolo`, etc.   |
| `MOTION_MIN_AREA`    | int    | 2000              | Min contour area for motion detector         |
| `FACE_CASCADE_PATH`  | str    | `/app/haar/...`   | Haar cascade XML path                        |
| `FACE_SCALE_FACTOR`  | float  | 1.1               | Haar scale factor                            |
| `FACE_MIN_NEIGHBORS` | int    | 5                 | Haar neighbors                               |
| `FACE_MIN_SIZE`      | str    | `30,30`           | Min face size (w,h)                          |
| `ENABLE_PRESENCE`    | bool   | `true` / `false`  | Enable presence detector                     |
| `PRESENCE_TIMEOUT`   | int    | 60                | Seconds until “absent” after last detection  |
| `PRESENCE_LABEL`     | str    | `Juniper`         | Human-readable label for presence events     |
| `ENABLE_YOLO`        | bool   | `true` / `false`  | Toggle YOLO detector                         |
| `YOLO_MODEL`         | str    | `yolov8n.pt`      | YOLO checkpoint path                         |
| `YOLO_CLASSES`       | str    | `person`          | Comma list of class names to keep            |
| `YOLO_CONF`          | float  | 0.25              | Confidence threshold                         |
| `YOLO_DEVICE`        | str    | `cpu` or `0`      | `cpu` or CUDA device (e.g. `0`)              |
| `ANNOTATE`           | bool   | `true`            | Draw boxes on frames for UI stream           |
| `JPEG_QUALITY`       | int    | 90                | JPEG quality for snapshots / MJPEG           |

### UI & bus

| Env var                         | Type  | Default                      | Description                      |
|---------------------------------|-------|------------------------------|----------------------------------|
| `ENABLE_UI`                     | bool  | `true`                       | Enable built-in HTML UI          |
| `ORION_BUS_ENABLED`             | bool  | `true`                       | Enable bus publishing            |
| `ORION_BUS_URL`                 | str   | `redis://orion-redis:6379/0` | Redis URL                        |
| `VISION_EVENTS_PUBLISH_RAW`     | str   | `orion:vision:edge:raw`      | Raw events channel (publish)     |
| `VISION_EVENTS_SUBSCRIBE_RAW`   | str   | `orion:vision:edge:raw`      | Reserved (compat)                |
| `VISION_EVENTS_PUBLISH_NOTABLE` | str   | `orion:vision:edge:event:notable` | Future filtered events channel |

---

## Running with Docker Compose

Example service (Athena):

```yaml
services:
  orion-athena-vision-edge:
    build:
      context: ../..
      dockerfile: services/orion-vision-edge/Dockerfile
    container_name: orion-athena-vision-edge
    restart: unless-stopped
    environment:
      - SOURCE=${REOLINK_URL}
      - WIDTH=${WIDTH}
      - HEIGHT=${HEIGHT}
      - FPS=${FPS}
      - DETECTORS=${DETECTORS}
      - MOTION_MIN_AREA=${MOTION_MIN_AREA}
      - DETECT_EVERY_N_FRAMES=${DETECT_EVERY_N_FRAMES}
      - STREAM_ID=${STREAM_ID}
      - ENABLE_UI=${ENABLE_UI}
      - ENABLE_YOLO=${ENABLE_YOLO}
      - YOLO_MODEL=${YOLO_MODEL}
      - YOLO_CLASSES=${YOLO_CLASSES}
      - YOLO_CONF=${YOLO_CONF}
      - YOLO_DEVICE=${YOLO_DEVICE}
      - FACE_CASCADE_PATH=${FACE_CASCADE_PATH}
      - FACE_SCALE_FACTOR=${FACE_SCALE_FACTOR}
      - FACE_MIN_NEIGHBORS=${FACE_MIN_NEIGHBORS}
      - FACE_MIN_SIZE=${FACE_MIN_SIZE}
      - ANNOTATE=${ANNOTATE}
      - ENABLE_PRESENCE=${ENABLE_PRESENCE}
      - PRESENCE_TIMEOUT=${PRESENCE_TIMEOUT}
      - PRESENCE_LABEL=${PRESENCE_LABEL}
      - JPEG_QUALITY=${JPEG_QUALITY}
      - ORION_BUS_URL=${ORION_BUS_URL}
      - ORION_BUS_ENABLED=${ORION_BUS_ENABLED}
      - VISION_EVENTS_PUBLISH_RAW=${VISION_EVENTS_PUBLISH_RAW}
      - VISION_EVENTS_SUBSCRIBE_RAW=${VISION_EVENTS_SUBSCRIBE_RAW}
      - VISION_EVENTS_PUBLISH_NOTABLE=${VISION_EVENTS_PUBLISH_NOTABLE}
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    ports:
      - "7100:7100"
    networks:
      - app-net

networks:
  app-net:
    external: true
```

Then:

```bash
docker compose up -d orion-athena-vision-edge
```

---

## Tailscale Exposure

On Athena:

```bash
sudo tailscale serve reset

# Vision Edge on /
sudo tailscale serve --bg 7100
# → https://athena.<tailnet>.ts.net/ → Vision UI
```

---

## Debugging

- Check logs:

  ```bash
  docker logs -f orion-athena-vision-edge
  ```

- Health:

  ```bash
  curl http://localhost:7100/health
  ```

- Snapshot:

  ```bash
  curl -o snap.jpg http://localhost:7100/snapshot.jpg
  ```

- Bus:

  ```bash
  redis-cli -u "$ORION_BUS_URL" SUBSCRIBE orion:vision:edge:raw
  ```

You should see JSON events when you move in front of the camera.

---

## Roadmap

- Hook in **identity** (per-person embeddings + `person_id`)
- Hook in **affect** (emotion / posture)
- Mark static face regions (wall photos vs real people)
- Emit **PerceptualEpisode** events into Cortex Orchestrator
- Drive Hub’s **Vision** panel directly from this service
- Per-visit capture + snapshot archives for security / narrative
```

---

## services/orion-security-watcher/README.md

```markdown
# 🛡 Orion Security Watcher

The **Orion Security Watcher** is a small, focused service that sits between:

- Edge vision (`orion-vision-edge` on Athena)
- Notification channels (email now, SMS/push later)
- Higher-level cognition (Cortex, perception verbs)

Its job is to:

1. Subscribe to raw vision events from the bus.
2. Group them into **visits** (someone enters, moves around, leaves).
3. Decide whether a visit is **interesting or suspicious** based on:
   - Security mode (armed vs off)
   - Identities (future)
   - Presence detections
4. Rate-limit alerts so you don’t get spammed.
5. Capture image snapshots for evidence.
6. Send alerts (email) and publish structured alert events back onto the bus.

It also exposes a small **Tailscale-friendly UI** for arming/disarming and testing.

---

## Architecture

High-level flow:

1. **Bus Worker**
   - Subscribes to `VISION_EVENTS_SUBSCRIBE_RAW`
     - Typically `orion:vision:edge:raw`.
   - For each `Event`, parses into a `VisionEvent` model.
   - Passes it to `VisitManager` along with current `SecurityState`.

2. **VisitManager**
   - Maintains in-memory **visits**:
     - Tracks time windows of continuous human presence.
     - Aggregates detections (faces, people, motion, presence).
   - Produces:
     - `VisitSummary` objects (for logging / SQL / RDF later).
     - Optionally an `AlertPayload` when conditions are met and not rate-limited.

3. **Notifier**
   - When an `AlertPayload` is emitted:
     - Captures snapshots from `VISION_SNAPSHOT_URL`.
     - Saves them under `SECURITY_SNAPSHOT_ROOT`.
     - Sends email if `NOTIFY_MODE=inline`.
     - (Later: SMS/push, webhooks, etc.)

4. **Bus Integration**
   - Publishes visit summaries:
     - `CHANNEL_SECURITY_VISITS` (e.g. `orion:security:visits`)
   - Publishes alerts:
     - `CHANNEL_SECURITY_ALERTS` (e.g. `orion:security:alerts`)

5. **UI / API**
   - `GET /` – Tailwind HTML UI (via Tailscale) to arm/disarm and send test alert.
   - `GET /health` – status + config.
   - `GET /security/state` – JSON state (programmatic).
   - `POST /security/state` – set `armed` + `mode`.
   - `GET /state` – same as `/security/state` (UI-compat shortcut).
   - `POST /state` – same as `/security/state` (UI-compat shortcut).
   - `POST /security/test-alert` – synthetic alert for end-to-end testing.
   - `POST /test-alert` – shortcut alias for UI.

---

## Bus Channels

Configurable via env:

- **Subscribe**
  - `VISION_EVENTS_SUBSCRIBE_RAW`
    - Default: `orion:vision:edge:raw`
    - Source: `orion-vision-edge` events

- **Publish**
  - `CHANNEL_SECURITY_VISITS`
    - e.g. `orion:security:visits`
    - One message per visit, summarizing what happened.
  - `CHANNEL_SECURITY_ALERTS`
    - e.g. `orion:security:alerts`
    - One message per *alert* (after rate limiting), including snapshot paths.

Alerts are JSON-safe dicts derived from the `AlertPayload` model, using
`model_dump(mode="json")` to ensure datetimes are ISO strings.

---

## Models (Conceptual)

**VisionEvent**  
Normalized view of what came from Vision Edge. Typically includes:

- `ts`, `stream_id`, `frame_index`
- `detections`: list of `Detection` (humans, presence events, etc.)
- `meta`

**SecurityState**

- `armed: bool`
- `mode: str` – e.g. `off`, `vacation_strict`
- `updated_at: datetime | None`
- `updated_by: str | None`

Stored on disk (e.g. JSON file) so config survives restarts.

**VisitSummary**

Captures one contiguous human presence “episode”:

- `visit_id`
- `stream_id`
- `start_ts`, `end_ts`
- `duration_sec`
- `humans_present` / `identities` (future)
- raw detection counts, etc.

**AlertPayload**

Structured info for one alert:

- `ts`, `service`, `version`
- `alert_id`, `visit_id`
- `camera_id`
- `armed`, `mode`
- `humans_present`
- `best_identity`, `best_identity_conf` (future)
- `identity_votes` (future distribution)
- `reason` (e.g. `test_alert`, `unknown_human_while_armed`)
- `severity` (e.g. `low`, `high`)
- `snapshots: list[str]` – file paths for captured JPEGs
- `rate_limit`
  - `global_blocked`
  - `identity_blocked`
  - cooldown durations

---

## Configuration

All configuration is handled via Pydantic Settings.

### Core security settings

| Env var                        | Type  | Default           | Description                                   |
|--------------------------------|-------|-------------------|-----------------------------------------------|
| `SECURITY_ENABLED`             | bool  | `true`            | Enable bus worker + alert logic               |
| `SECURITY_MODE`                | str   | `vacation_strict` | Default mode                                  |
| `SECURITY_DEFAULT_ARMED`       | bool  | `false`           | Initial armed state when no state file exists |
| `SECURITY_GLOBAL_COOLDOWN_SEC` | int   | `300`             | Min seconds between any two alerts            |
| `SECURITY_IDENTITY_COOLDOWN_SEC` | int | `600`             | Min seconds between alerts for same identity  |

### Vision integration

| Env var                      | Type | Default                      | Description                                   |
|------------------------------|------|------------------------------|-----------------------------------------------|
| `VISION_EVENTS_SUBSCRIBE_RAW`| str  | `orion:vision:edge:raw`      | Bus channel to consume vision events          |
| `VISION_SNAPSHOT_URL`        | str  | `http://orion-athena-vision-edge:7100/snapshot.jpg` | URL for latest frame snapshot       |
| `SECURITY_SNAPSHOT_ROOT`     | str  | `/mnt/telemetry/orion-security/alerts` | Where to store snapshot JPEGs  |

### Bus

| Env var            | Type | Default                      | Description             |
|--------------------|------|------------------------------|-------------------------|
| `ORION_BUS_ENABLED`| bool | `true`                       | Enable bus usage        |
| `ORION_BUS_URL`    | str  | `redis://orion-redis:6379/0` | Redis URL               |
| `CHANNEL_SECURITY_VISITS` | str | `orion:security:visits` | Visit summary channel   |
| `CHANNEL_SECURITY_ALERTS` | str | `orion:security:alerts` | Alerts channel          |

### Notification

| Env var                       | Type  | Default     | Description                                |
|-------------------------------|-------|-------------|--------------------------------------------|
| `NOTIFY_MODE`                 | str   | `none`      | `none` or `inline` (email inside service)  |
| `NOTIFY_EMAIL_ENABLED`        | bool  | `false`     | Enable email notifications                 |
| `NOTIFY_EMAIL_SMTP_HOST`      | str   | –           | SMTP host                                  |
| `NOTIFY_EMAIL_SMTP_PORT`      | int   | 587         | SMTP port                                  |
| `NOTIFY_EMAIL_SMTP_USERNAME`  | str   | –           | SMTP username                              |
| `NOTIFY_EMAIL_SMTP_PASSWORD`  | str   | –           | SMTP password / app password               |
| `NOTIFY_EMAIL_FROM`           | str   | –           | From header, e.g. `Orion Security <...>`   |
| `NOTIFY_EMAIL_TO`             | str   | –           | Comma list of recipients                   |

---

## Running with Docker Compose

Example service:

```yaml
services:
  orion-athena-security-watcher:
    build:
      context: ../..
      dockerfile: services/orion-security-watcher/Dockerfile
    container_name: orion-athena-security-watcher
    restart: unless-stopped
    environment:
      - SECURITY_ENABLED=${SECURITY_ENABLED}
      - SECURITY_MODE=${SECURITY_MODE}
      - SECURITY_DEFAULT_ARMED=${SECURITY_DEFAULT_ARMED}
      - SECURITY_GLOBAL_COOLDOWN_SEC=${SECURITY_GLOBAL_COOLDOWN_SEC}
      - SECURITY_IDENTITY_COOLDOWN_SEC=${SECURITY_IDENTITY_COOLDOWN_SEC}
      - ORION_BUS_URL=${ORION_BUS_URL}
      - ORION_BUS_ENABLED=${ORION_BUS_ENABLED}
      - VISION_EVENTS_SUBSCRIBE_RAW=${VISION_EVENTS_SUBSCRIBE_RAW}
      - VISION_SNAPSHOT_URL=${VISION_SNAPSHOT_URL}
      - SECURITY_SNAPSHOT_ROOT=${SECURITY_SNAPSHOT_ROOT}
      - CHANNEL_SECURITY_VISITS=${CHANNEL_SECURITY_VISITS}
      - CHANNEL_SECURITY_ALERTS=${CHANNEL_SECURITY_ALERTS}
      - NOTIFY_MODE=${NOTIFY_MODE}
      - NOTIFY_EMAIL_ENABLED=${NOTIFY_EMAIL_ENABLED}
      - NOTIFY_EMAIL_SMTP_HOST=${NOTIFY_EMAIL_SMTP_HOST}
      - NOTIFY_EMAIL_SMTP_PORT=${NOTIFY_EMAIL_SMTP_PORT}
      - NOTIFY_EMAIL_SMTP_USERNAME=${NOTIFY_EMAIL_SMTP_USERNAME}
      - NOTIFY_EMAIL_SMTP_PASSWORD=${NOTIFY_EMAIL_SMTP_PASSWORD}
      - NOTIFY_EMAIL_FROM=${NOTIFY_EMAIL_FROM}
      - NOTIFY_EMAIL_TO=${NOTIFY_EMAIL_TO}
    volumes:
      - /mnt/telemetry/orion-security:/mnt/telemetry/orion-security
    ports:
      - "7120:7120"
    networks:
      - app-net

networks:
  app-net:
    external: true
```

Then:

```bash
docker compose up -d orion-athena-security-watcher
```

---

## Tailscale Exposure

On Athena:

```bash
# Vision Edge on /
sudo tailscale serve reset
sudo tailscale serve --bg 7100

# Security Watcher on /security
sudo tailscale serve --bg --set-path=/security 7120

# Check:
tailscale serve status
```

You should see:

```text
https://athena.<tailnet>.ts.net (tailnet only)
|-- /         proxy http://127.0.0.1:7100
|-- /security proxy http://127.0.0.1:7120
```

- `https://athena.<tailnet>.ts.net/` → Vision UI  
- `https://athena.<tailnet>.ts.net/security` → Security UI

---

## Usage

### 1. Arm / Disarm

From the `/security` UI (via Tailscale):

- Toggle **Armed** on/off.
- The current state is persisted and exposed via:
  - `GET /state`
  - `GET /security/state`

From CLI:

```bash
curl -X POST http://localhost:7120/state \
  -H 'Content-Type: application/json' \
  -d '{"armed": true, "mode": "vacation_strict"}'
```

### 2. Test Alert

To validate snapshots + email:

- Click **Send Test Alert** in the `/security` UI  
  or:

  ```bash
  curl -X POST http://localhost:7120/test-alert
  ```

You should see in logs:

- Snapshot capture
- Alert published on `CHANNEL_SECURITY_ALERTS`
- Email send (if enabled)

---

## Debugging

- Logs:

  ```bash
  docker logs -f orion-athena-security-watcher
  ```

- Health:

  ```bash
  curl http://localhost:7120/health
  ```

- Bus:

  ```bash
  redis-cli -u "$ORION_BUS_URL" SUBSCRIBE orion:security:alerts
  redis-cli -u "$ORION_BUS_URL" SUBSCRIBE orion:security:visits
  ```

---

## Roadmap

- Integrate **identity**:
  - "unknown human while armed" → higher severity
  - known family → lower severity / different rules
- Integrate **perception verbs** via Cortex:
  - Perceptual episodes flow into `orion:cortex:intake:perception`
- Per-identity, per-location rate limits & policies.
- SMS / push providers for faster alerts.
- UI for browsing recent visits + alerts, plus labeling unknown visitors.
```
