# Orion Biometrics

The **Biometrics** service collects hardware telemetry (CPU, memory, GPU usage, power consumption), normalizes it into bounded pressures, and publishes multiple payloads to the bus for storage and downstream cognition.

## Contracts

### Published Channels
| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion:telemetry:biometrics` | `TELEMETRY_PUBLISH_CHANNEL` | `biometrics.telemetry` | Raw hardware metrics (legacy payload). |
| `orion:biometrics:sample` | `BIOMETRICS_SAMPLE_CHANNEL` | `biometrics.sample.v1` | Expanded raw sample (CPU/mem/disk/net/temps/power). |
| `orion:biometrics:summary` | `BIOMETRICS_SUMMARY_CHANNEL` | `biometrics.summary.v1` | Normalized pressures/headroom/composites. |
| `orion:biometrics:induction` | `BIOMETRICS_INDUCTION_CHANNEL` | `biometrics.induction.v1` | EWMA level/trend/volatility/spikes. |
| `orion:biometrics:cluster` | `BIOMETRICS_CLUSTER_CHANNEL` | `biometrics.cluster.v1` | Role-weighted cluster aggregate (hub mode). |
| `orion:spark:signal` | `SPARK_SIGNAL_CHANNEL` | `spark.signal.v1` | Bounded resource signal from cluster strain (hub mode). |

### Environment Variables
Provenance: `.env_example` → `docker-compose.yml` → `settings.py`

| Variable | Default (Settings) | Description |
| :--- | :--- | :--- |
| `TELEMETRY_PUBLISH_CHANNEL` | `orion:telemetry:biometrics` | Raw publish channel (legacy payload). |
| `BIOMETRICS_SAMPLE_CHANNEL` | `orion:biometrics:sample` | Sample publish channel. |
| `BIOMETRICS_SUMMARY_CHANNEL` | `orion:biometrics:summary` | Summary publish channel. |
| `BIOMETRICS_INDUCTION_CHANNEL` | `orion:biometrics:induction` | Induction publish channel. |
| `BIOMETRICS_CLUSTER_CHANNEL` | `orion:biometrics:cluster` | Cluster publish channel (hub mode). |
| `BIOMETRICS_MODE` | `agent` | `agent` (node), `hub` (aggregate), or `both`. |
| `CLUSTER_ROLE_WEIGHTS` | `{"atlas":0.7,"athena":0.3,"other":0.5}` | Role weighting for cluster aggregate. |
| `THERMAL_MIN_C` | `50.0` | Temperature floor for normalization. |
| `THERMAL_MAX_C` | `85.0` | Temperature ceiling for normalization. |
| `DISK_BW_MBPS` | `200.0` | Disk bandwidth scale (MB/s). |
| `NET_BW_MBPS` | `125.0` | Network bandwidth scale (MB/s). |
| `ORION_HEALTH_CHANNEL` | `orion:system:health` | Health check channel. |

## Running & Testing

### Run via Docker
```bash
docker-compose up -d orion-biometrics
```

### Smoke Test
```bash
scripts/smoke_biometrics.sh
# Expects one message on sample/summary/induction + state-service reply.
```
