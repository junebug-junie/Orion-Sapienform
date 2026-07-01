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

## Idle tick (pacemaker)

`FIELD_DIGESTER_IDLE_TICK_ENABLED` (default `true`) keeps `tick_id` advancing on every poll even when there are no new receipts to consume. On a quiet poll the worker still loads/reconciles field state, runs decay + diffusion + suppression with an empty perturbation set, mints a new `tick_id`, and persists it via `save_field` — but it does not advance the receipt cursor or write pending deltas (that only happens via `commit_digest_tick` when receipts were actually consumed). This lets downstream free-running consumers (`orion-attention-runtime`, `orion-self-state-runtime`) keep advancing off the latest tick during quiet periods. Set to `false` to restore the old behavior where `_tick()` returns immediately when there are no new receipts.

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_URI` | (required) | Postgres connection string |
| `LATTICE_PATH` | `config/field/orion_field_topology.v1.yaml` | Node/capability lattice YAML (canonical) |

`biometrics_lattice.yaml` is retained as a compatibility alias; `orion_field_topology.v1.yaml` is the canonical config. Operators may keep `LATTICE_PATH` pointed at either file.
| `RECEIPT_POLL_INTERVAL_SEC` | `2.0` | Receipt poll interval |
| `BIOMETRICS_FIELD_DECAY_RATE` | `0.92` | Per-tick pressure decay multiplier |
| `BIOMETRICS_FIELD_DIFFUSION_RATE` | `1.0` | Node→capability diffusion strength |
| `FIELD_DIGESTER_IDLE_TICK_ENABLED` | `true` | Keep minting ticks (decay/diffusion only) on quiet polls with no new receipts |
| `LOG_LEVEL` | `INFO` | Python log level |
| `FIELD_DIGESTER_PORT` | `8116` | Host port for `docker compose` (compose-only) |

v1 persists projections to Postgres only; bus emit is deferred (`orion/bus/channels.yaml` unchanged).
