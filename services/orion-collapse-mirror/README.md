# Orion Collapse Mirror

The **Collapse Mirror** service acts as the primary ingestion point for emergent events ("collapse entries"). It accepts raw observation data, wraps it in a canonical `BaseEnvelope`, and publishes it to the bus for downstream processing by writers (SQL, Vector, RDF) and enrichment services (Meta Tags).

Active runtime wiring is `bus_runtime.start_services()` (async `Hunter` on intake + async `Rabbit` on exec-step RPC). The legacy threaded `exec_worker.py` path was removed; do not reintroduce a second subscriber on the same exec channel.

## Contracts

### Consumed Channels
| Channel | Env Var | Description |
| :--- | :--- | :--- |
| `orion:collapse:intake` | `CHANNEL_COLLAPSE_INTAKE` | Raw intake from HTTP ingress, cortex-exec loopback, or other services. |
| `orion:exec:request:CollapseMirrorService` | `EXEC_REQUEST_PREFIX` + service name | Cortex exec-step RPC target. |

### Published Channels
| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion:collapse:triage` | `CHANNEL_COLLAPSE_TRIAGE` | `collapse.mirror.entry` | Strict observers routed to enrichment pipeline. |
| `orion:collapse:sql-write` | `CHANNEL_COLLAPSE_SQL_WRITE` | `collapse.mirror` | Raw storage path for SQL writer. |
| `orion:collapse:intake` | `CHANNEL_COLLAPSE_INTAKE` | `collapse.mirror.intake` | Exec-step loopback publish. |

### Environment Variables
Provenance: `.env_example` → `docker-compose.yml` → `settings.py`

| Variable | Default (Settings) | Description |
| :--- | :--- | :--- |
| `CHANNEL_COLLAPSE_INTAKE` | `orion:collapse:intake` | Ingestion channel. |
| `CHANNEL_COLLAPSE_TRIAGE` | `orion:collapse:triage` | Triage/Fanout channel. |
| `CHANNEL_COLLAPSE_SQL_WRITE` | `orion:collapse:sql-write` | SQL writer channel. |
| `EXEC_REQUEST_PREFIX` | `orion:exec:request` | Prefix for exec-step RPC channels. |
| `ORION_HEALTH_CHANNEL` | `orion:system:health` | Health check channel. |
| `ERROR_CHANNEL` | `system.error` | Error reporting channel. |

## Running & Testing

### Run via Docker
```bash
docker-compose up -d orion-collapse-mirror
```

### Readiness
`GET /ready` reports NUMSUB-backed readiness for both the intake `Hunter` and exec-step `Rabbit` consumers.

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

**Verify intake fanout only (downstream organ):**
```bash
python scripts/smoke_juniper_collapse_fanout.py --redis "$ORION_BUS_URL"
```

**Verify live-path prerequisites (upstream subscribers + substrate truth):**
```bash
./scripts/collapse_mirror_live_path_truth.sh
```

This checks `PUBSUB NUMSUB` on equilibrium metacog trigger, cortex exec request, collapse mirror exec RPC, intake, and sql-write, then runs aggregate grammar production truth. A passing intake smoke with a failing live-path gate usually means upstream subscribers or substrate truth — not collapse-mirror fanout.

**Verify on Bus:**
```bash
python scripts/bus_harness.py tap
# Expect: kind="collapse.mirror" on orion:collapse:triage
```
