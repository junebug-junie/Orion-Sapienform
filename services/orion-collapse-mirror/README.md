# Orion Collapse Mirror

The **Collapse Mirror** service acts as the primary ingestion point for emergent events ("collapse entries"). It accepts raw observation data, wraps it in a canonical `BaseEnvelope`, and publishes it to the bus for downstream processing by writers (SQL, Vector, RDF) and enrichment services (Meta Tags).

## Contracts

### Consumed Channels
| Channel | Env Var | Description |
| :--- | :--- | :--- |
| `orion:collapse:intake` | `CHANNEL_COLLAPSE_INTAKE` | Raw intake from HTTP ingress or other services. |
| `orion:collapse:triage` | `CHANNEL_COLLAPSE_TRIAGE` | Triage channel (consumed for re-processing/logging). |

### Published Channels
| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion:collapse:intake` | `CHANNEL_COLLAPSE_INTAKE` | `collapse.mirror.entry` | Normalized entry published to bus. |
| `orion:collapse:triage` | `CHANNEL_COLLAPSE_TRIAGE` | `collapse.mirror` | Published after ingestion for triage/enrichment. |

### Environment Variables
Provenance: `.env_example` → `docker-compose.yml` → `settings.py`

| Variable | Default (Settings) | Description |
| :--- | :--- | :--- |
| `CHANNEL_COLLAPSE_INTAKE` | `orion:collapse:intake` | Ingestion channel. |
| `CHANNEL_COLLAPSE_TRIAGE` | `orion:collapse:triage` | Triage/Fanout channel. |
| `ORION_HEALTH_CHANNEL` | `orion:system:health` | Health check channel. |
| `ERROR_CHANNEL` | `system.error` | Error reporting channel. |

## Running & Testing

### Run via Docker
```bash
docker-compose up -d orion-collapse-mirror
```

### Smoke Test (HTTP Ingress)
Post a raw collapse entry to the HTTP endpoint, which publishes to the bus.

```bash
curl -X POST http://localhost:8087/api/log/collapse \
  -H "Content-Type: application/json" \
  -d '{
    "observer": "Tester",
    "trigger": "Smoke Test",
    "summary": "Verifying bus connectivity",
    "observer_state": ["testing"]
  }'
```

**Verify on Bus:**
```bash
python scripts/bus_harness.py tap
# Expect: kind="collapse.mirror" on orion:collapse:triage
```
