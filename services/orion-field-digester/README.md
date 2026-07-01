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

`FIELD_DIGESTER_IDLE_TICK_ENABLED` (**default `false`**) keeps `tick_id` advancing on every poll even when there are no new receipts to consume. On a quiet poll the worker still loads/reconciles field state, runs decay + diffusion + suppression with an empty perturbation set, mints a new `tick_id`, and persists it via `save_field` — but it does not advance the receipt cursor or write pending deltas (that only happens via `commit_digest_tick` when receipts were actually consumed). This lets downstream free-running consumers (`orion-attention-runtime`, `orion-self-state-runtime`) keep advancing off the latest tick during quiet periods.

**Do not enable this in production yet without retention.** `save_field` mints a fresh `tick_id` every call, so every idle tick is a genuine new row in `substrate_field_state` — at the default 2s poll interval that's ~43k rows/day with no pruning anywhere in this service. The same growth cascades downstream: `orion-attention-runtime` writes one `substrate_attention_frame` row per new `tick_id`, and `orion-self-state-runtime` writes one `substrate_self_state` row per new frame — neither of those services prunes either. This is the same failure shape as a prior unbounded-Postgres-growth incident on this host. Enabling this flag is safe only once a retention/pruning story exists for all three tables (mirror the existing `receipt_pruner.py` pattern in `orion-substrate-runtime`); until then, leave it `false` — the mesh behaves exactly as it did before this feature shipped.

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_URI` | (required) | Postgres connection string |
| `LATTICE_PATH` | `config/field/orion_field_topology.v1.yaml` | Node/capability lattice YAML (canonical) |

`biometrics_lattice.yaml` is retained as a compatibility alias; `orion_field_topology.v1.yaml` is the canonical config. Operators may keep `LATTICE_PATH` pointed at either file.
| `RECEIPT_POLL_INTERVAL_SEC` | `2.0` | Receipt poll interval |
| `BIOMETRICS_FIELD_DECAY_RATE` | `0.92` | Per-tick pressure decay multiplier |
| `BIOMETRICS_FIELD_DIFFUSION_RATE` | `1.0` | Node→capability diffusion strength |
| `FIELD_DIGESTER_IDLE_TICK_ENABLED` | `false` | Keep minting ticks (decay/diffusion only) on quiet polls with no new receipts — see warning above before enabling |
| `LOG_LEVEL` | `INFO` | Python log level |
| `FIELD_DIGESTER_PORT` | `8116` | Host port for `docker compose` (compose-only) |

v1 persists projections to Postgres only; bus emit is deferred (`orion/bus/channels.yaml` unchanged).
