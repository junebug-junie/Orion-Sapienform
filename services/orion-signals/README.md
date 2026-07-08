# Orion Signals — organ signal mesh launcher

`orion-signals` is **not** a runtime service. It is a thin orchestration layer — a causal-spine mesh launcher — that brings up the organ-signal stack around `orion-signal-gateway` and the `orion:signals:*` bus channels.

The roster (`roster.v1.yaml`) is the machine-readable source of truth for which compose services belong to each tier.

## Tier definitions

| Tier | Includes | Purpose |
|------|----------|---------|
| **core** | `orion-bus` (bus-core), `orion-signal-gateway` | Redis bus + signal gateway spine |
| **tier1** | core + organ producers | Biometrics, equilibrium, collapse-mirror, cortex-exec, recall, spark-introspector, memory-consolidation |
| **tier2** | tier1 + spark-concept-induction | Downstream consumer of `orion:signals:biometrics`, `:spark`, `:equilibrium` |
| **routing** | cortex-gateway, cortex-orchestrator, llm-gateway | Chat/cortex routing lane (additive — does not include tier1/tier2 unless you launch `full`) |
| **full** | all tiers | Complete organ-signal mesh + routing |

Tiers are **cumulative**: `tier1` includes `core`, `tier2` includes `tier1`, and `full` includes everything.

## Prerequisites

1. **Docker network** — `app-net` must exist as an external network:

   ```bash
   docker network create app-net
   ```

2. **Per-service `.env` files** — each service in the roster needs its own `services/<compose_dir>/.env` synced from `.env_example`. Run from repo root:

   ```bash
   python scripts/sync_local_env_from_example.py
   ```

3. **`ORION_BUS_URL`** — set in `services/orion-signals/.env`. For mesh operation this **must** be `redis://<tailscale-node-ip>:6379/0` (not `bus-core` hostname from the host).

4. **Hub is external** — `orion-hub` uses `network_mode: host` and is **not** started by this stack. Start Hub separately when you need the Organ Signals UI or operator surfaces.

## Redis dedupe (bus-core vs bundled redis)

When you run `orion-bus` `bus-core` (included in `core` tier), point all services at the shared bus Redis — **do not** also start signal-gateway's bundled `orion-redis`.

Default: `SIGNALS_USE_BUNDLED_REDIS=false` in `services/orion-signals/.env`.

With bundled redis disabled, `scripts/up.sh` starts:

- `otel-tempo`, `otel-collector`, `otel-grafana`
- `orion-signal-gateway` with `--no-deps` (skips the compose `depends_on` for `orion-redis`)

Set `ORION_BUS_URL` in both `services/orion-signals/.env` and `services/orion-signal-gateway/.env` to your mesh bus endpoint. Inside containers on `app-net`, services can use `redis://bus-core:6379/0` via the bus-core network alias.

To run signal-gateway standalone with its own Redis, set `SIGNALS_USE_BUNDLED_REDIS=true`.

## Launch

From repo root (or anywhere — scripts resolve `REPO_ROOT`):

```bash
# Default tier from SIGNALS_TIER in .env (core)
./services/orion-signals/scripts/up.sh

# Explicit tier
./services/orion-signals/scripts/up.sh tier1
./services/orion-signals/scripts/up.sh tier2
./services/orion-signals/scripts/up.sh routing   # routing only (additive launch)
./services/orion-signals/scripts/up.sh full
```

## Smoke check

```bash
./services/orion-signals/scripts/smoke.sh
```

Checks:

- Redis `PONG` via `bus-core` container (if running)
- `GET /health` on signal-gateway
- `GET /signals/active` on signal-gateway

## Stop

Reverse-order teardown:

```bash
./services/orion-signals/scripts/down.sh [core|tier1|tier2|routing|full]
```

## Restart

After `.env` or roster changes:

```bash
./services/orion-signals/scripts/down.sh full
./services/orion-signals/scripts/up.sh full
```

Restart individual services via their compose dir:

```bash
docker compose \
  --env-file services/orion-signal-gateway/.env \
  --env-file services/orion-signals/.env \
  -f services/orion-signal-gateway/docker-compose.yml \
  restart orion-signal-gateway
```

## Hub dependency

The Organ Signals UI in Orion Hub reads from signal-gateway HTTP endpoints. Hub must be running separately on the operator host (host networking). Ensure `SIGNAL_GATEWAY_HTTP_PORT` matches what Hub expects.

## Files

| File | Role |
|------|------|
| `roster.v1.yaml` | Tier → compose service mapping |
| `.env_example` | Launcher operator contract |
| `scripts/up.sh` | Cumulative tier launcher |
| `scripts/down.sh` | Reverse-order stop |
| `scripts/smoke.sh` | Bus + gateway health checks |
| `tests/` | Roster and script gate tests |
