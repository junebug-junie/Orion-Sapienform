# orion-field-digester

Substrate field digestion worker that consumes committed reduction receipts (biometrics + cortex-exec execution trajectories) and compiles lattice field state:

```text
substrate_reduction_receipts → delta dedupe → perturb/decay/diffuse/suppress → substrate_field_state
```

## Setup

```bash
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_field_digester_v1.sql
cp services/orion-field-digester/.env_example services/orion-field-digester/.env
```

## Run

```bash
cd services/orion-field-digester
docker compose up --build
```

Health: `GET http://localhost:8116/health`

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_URI` | (required) | Postgres connection string |
| `LATTICE_PATH` | `config/field/orion_field_topology.v1.yaml` | Node/capability lattice YAML (canonical) |

`biometrics_lattice.yaml` is retained as a compatibility alias; `orion_field_topology.v1.yaml` is the canonical config. Operators may keep `LATTICE_PATH` pointed at either file.
| `RECEIPT_POLL_INTERVAL_SEC` | `2.0` | Receipt poll interval |
| `BIOMETRICS_FIELD_DECAY_RATE` | `0.92` | Per-tick pressure decay multiplier |
| `BIOMETRICS_FIELD_DIFFUSION_RATE` | `1.0` | Node→capability diffusion strength |
| `LOG_LEVEL` | `INFO` | Python log level |
| `FIELD_DIGESTER_PORT` | `8116` | Host port for `docker compose` (compose-only) |

v1 persists projections to Postgres only; bus emit is deferred (`orion/bus/channels.yaml` unchanged).
