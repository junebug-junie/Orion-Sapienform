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

`FIELD_DIGESTER_IDLE_TICK_ENABLED` (**default `true`**) keeps `tick_id` advancing on every poll even when there are no new receipts to consume. On a quiet poll the worker still loads/reconciles field state, runs decay + diffusion + suppression with an empty perturbation set, mints a new `tick_id`, and persists it via `save_field` — but it does not advance the receipt cursor or write pending deltas (that only happens via `commit_digest_tick` when receipts were actually consumed). This lets downstream free-running consumers (`orion-attention-runtime`, `orion-self-state-runtime`) keep advancing off the latest tick during quiet periods.

**Retention, the applied-deltas pruner, and the health monitor are all prerequisites, and are all live.** `save_field` mints a fresh `tick_id` every call, so every idle tick is a genuine new row in `substrate_field_state` — at the default 2s poll interval that's ~43k rows/day. This is bounded by an hourly batched pruner (`FIELD_STATE_RETENTION_HOURS`, default 72h) that never deletes the newest row. The same cascade happens downstream in `orion-attention-runtime`/`orion-self-state-runtime`, each of which independently prunes its own tables on the same 72h/hourly pattern.

`substrate_field_applied_deltas` (the delta dedup ledger) has no natural "latest row" reader, so instead of a time-based cutoff it is pruned by receipt existence: a dedup row is deleted only once its source receipt is confirmed gone from `substrate_reduction_receipts`, at which point that `delta_id` can structurally never be redelivered by `fetch_new_receipts()`. `FIELD_APPLIED_DELTAS_PRUNE_MIN_AGE_HOURS` is only a small safety margin against racing the receipt pruner's own transaction, not a correctness bound.

A background health monitor (`FIELD_DIGESTER_HEALTH_CHECK_INTERVAL_SEC`, default 900s) watches for: `substrate_field_state`'s oldest row exceeding `FIELD_STATE_STALL_MULTIPLIER` × the retention window (pruner stalled), `substrate_field_applied_deltas` row count exceeding `FIELD_APPLIED_DELTAS_ALERT_ROW_COUNT`, and the `conjourney` database exceeding `FIELD_DIGESTER_DB_SIZE_ALERT_GB`. Checks are edge-triggered — an alert (via `orion-notify`'s `POST /attention/request`, surfacing in Hub's existing Pending Attention panel) fires only on a healthy→unhealthy transition, plus a lower-severity recovery note on the way back, so a persisting condition does not spam a fresh attention item every check.

This whole chain was originally left disabled after a prior unbounded-Postgres-growth incident on this host; the guardrails above (mirroring the existing `receipt_pruner.py` pattern in `orion-substrate-runtime`) are what made re-enabling it safe.

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_URI` | (required) | Postgres connection string |
| `LATTICE_PATH` | `config/field/orion_field_topology.v1.yaml` | Node/capability lattice YAML (canonical) |

`biometrics_lattice.yaml` is retained as a compatibility alias; `orion_field_topology.v1.yaml` is the canonical config. Operators may keep `LATTICE_PATH` pointed at either file.
| `RECEIPT_POLL_INTERVAL_SEC` | `2.0` | Receipt poll interval |
| `BIOMETRICS_FIELD_DECAY_RATE` | `0.92` | Per-tick pressure decay multiplier |
| `BIOMETRICS_FIELD_DIFFUSION_RATE` | `1.0` | Node→capability diffusion strength |
| `FIELD_DIGESTER_IDLE_TICK_ENABLED` | `true` | Keep minting ticks (decay/diffusion only) on quiet polls with no new receipts — see section above |
| `FIELD_STATE_RETENTION_HOURS` | `72.0` | `substrate_field_state` retention window (hourly batched prune) |
| `FIELD_STATE_PRUNE_INTERVAL_SEC` | `3600.0` | `substrate_field_state` prune cadence |
| `FIELD_APPLIED_DELTAS_PRUNE_MIN_AGE_HOURS` | `1.0` | Safety margin before pruning an applied-delta row whose receipt is already gone |
| `FIELD_DIGESTER_HEALTH_CHECK_INTERVAL_SEC` | `900.0` | Health-monitor check cadence |
| `FIELD_STATE_STALL_MULTIPLIER` | `1.5` | Alert if `substrate_field_state`'s oldest row exceeds this × retention hours |
| `FIELD_APPLIED_DELTAS_ALERT_ROW_COUNT` | `5000000` | Alert if `substrate_field_applied_deltas` row count exceeds this |
| `FIELD_DIGESTER_DB_SIZE_ALERT_GB` | `60.0` | Alert if the `conjourney` database exceeds this size (observed baseline ~37.5GB as of 2026-07-12; default leaves real headroom, not a round guess) |
| `NOTIFY_BASE_URL` | `http://orion-athena-notify:7140` | `orion-notify` base URL for health-monitor attention alerts |
| `NOTIFY_API_TOKEN` | (empty) | `orion-notify` auth token, if configured |
| `LOG_LEVEL` | `INFO` | Python log level |
| `FIELD_DIGESTER_PORT` | `8116` | Host port for `docker compose` (compose-only) |

v1 persists projections to Postgres only; bus emit is deferred (`orion/bus/channels.yaml` unchanged).
