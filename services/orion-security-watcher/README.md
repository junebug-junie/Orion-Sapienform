# 🛡 Orion Security Watcher (Guard)

`orion-security-watcher` (Guard) consumes `VisionEdgeArtifact` events from `orion-vision-edge` and determines security state (Presence, Alerts).

It implements a rolling window logic to debounce detections and emit high-confidence signals.

---

## Architecture

1.  **Ingest**: Subscribes to `vision.artifacts`.
2.  **Buffer**: Maintains a rolling window (e.g., 30s) of artifacts per camera.
3.  **Analyze**: Checks for sustained human presence (YOLO `person` class).
4.  **Emit**:
    *   `vision.guard.signal`: Periodic status (presence/absent/unknown).
    *   `vision.guard.alert`: Immediate alert on sustained detection.

---

## Configuration

### Bus Channels

| Env var | Default | Notes |
|---|---|---|
| `CHANNEL_VISION_ARTIFACTS` | `vision.artifacts` | Input from Edge |
| `CHANNEL_VISION_GUARD_SIGNAL` | `vision.guard.signal` | Periodic status |
| `CHANNEL_VISION_GUARD_ALERT` | `vision.guard.alert` | Alert events |

### Guard Logic

| Env var | Default | Notes |
|---|---|---|
| `GUARD_WINDOW_SECONDS` | `30` | Rolling window size |
| `GUARD_EMIT_EVERY_SECONDS` | `5` | Signal emission frequency |
| `GUARD_PERSON_MIN_CONF` | `0.4` | Min confidence to count as person |
| `GUARD_SUSTAIN_SECONDS` | `2` | Duration of presence to trigger alert |
| `GUARD_ALERT_COOLDOWN_SECONDS` | `60` | Alert cooldown |

---

## Run & Test

Use `docker-compose up -d`.

To verify:
1.  Check logs: `docker logs -f orion-security-watcher`
2.  Monitor signals:
    ```bash
    redis-cli SUBSCRIBE vision.guard.signal
    ```
