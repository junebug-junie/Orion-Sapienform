# Orion Biometrics

The **Biometrics** service collects hardware telemetry (CPU, memory, GPU usage, power consumption) and publishes it to the bus for storage and monitoring.

## Contracts

### Published Channels
| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion:telemetry:biometrics` | `TELEMETRY_PUBLISH_CHANNEL` | `biometrics.telemetry` | Live hardware metrics. |

### Environment Variables
Provenance: `.env_example` → `docker-compose.yml` → `settings.py`

| Variable | Default (Settings) | Description |
| :--- | :--- | :--- |
| `TELEMETRY_PUBLISH_CHANNEL` | `orion:biometrics:telemetry` | Publish channel. (Recommended: `orion:telemetry:biometrics`) |
| `HEALTH_CHANNEL` | `system.health` | Health check channel. |

## Running & Testing

### Run via Docker
```bash
docker-compose up -d orion-biometrics
```

### Smoke Test
```bash
python scripts/bus_harness.py tap
# Expect kind="biometrics.telemetry" every few seconds.
```
