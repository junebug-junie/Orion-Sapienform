# orion-gpu-pool

The one lease queue for every GPU on circe. Every GPU user asks it for a **lease** and gets a
grant, a place in line, a durable backlog entry, or a typed "unavailable" with a reason. It is the
only thing that knows what is loaded on each card and the only thing that decides who goes first.

Design: `docs/superpowers/specs/2026-09-24-gpu-pool-design.md`.

## Where things live

| what | where |
| --- | --- |
| rules: cards, roles, who may borrow what, priorities, grace, retry | `config/gpu_pool.yaml` (no model names) |
| model catalog | `config/llm_profiles.yaml` |
| which model is in each role *right now* | discovered live (below) |
| scheduling decisions | `orion/gpu_pool/scheduler.py` (pure function, one test per rule) |
| lease lifecycle | `orion/gpu_pool/lease_graph.py` (LangGraph, checkpointed in Postgres) |
| the only client | `orion/gpu_pool/client.py` (`async with gpu_lease(...)`) |
| live projection | `gpu_pool_leases`, `gpu_pool_cards` (`services/orion-sql-db/manual_migration_gpu_pool_v1.sql`) |
| lease history | `gpu_pool_events`, written by orion-sql-writer from `orion:gpu_pool:event` |

**Adding a card:** add it under `cards:` in the YAML, then add or extend the roles on it. **Adding
a service:** add a role (`kind: service` needs `slots` and `vram_gb`) and a class that uses it.
`python scripts/check_gpu_pool_config.py` checks the YAML against the llama.cpp compose files.

## Discovery: what is actually loaded

Each llama.cpp worker announces `{role, llm_profiles profile, host port}` on
`orion:llm:worker:announce` every 30s (`LLM_ROLE` / `LLM_ANNOUNCE_PORT` in
`services/orion-llamacpp-host/docker-compose.atlas-workers.yml`). Every 15s the pool compares that
with the server's own `GET /props`: the loaded file must equal the profile's `hf_filename`.

| status | meaning | grants |
| --- | --- | --- |
| confirmed | announcement, profile and loaded file agree | yes |
| mismatch | announced profile's file ≠ what the server loaded, wrong port, or unknown profile | no |
| silent | port answers but no fresh announcement | no |
| down | no answer | no |
| unloaded / evicted | a swap seat not loaded / a resident pushed off by a swap | no |

## Bus (everything except the llama.cpp edge)

| channel | what |
| --- | --- |
| `orion:gpu_pool:lease:request` | acquire / heartbeat / release / cancel (RPC; answers at once) |
| `orion:gpu_pool:event` | lease facts; callers wake on `granted` |
| `orion:gpu_pool:state` (+ `:state:request`) | whole-pool snapshot |
| `orion:gpu_pool:control:request` | operator: lend, unlend, hold, release, replay, cancel, backfill. No token (the pool trusts the bus like every Orion service); each verb is logged with its actor and published as a pool event. |
| `orion:llm:worker:announce` | worker → pool discovery |

HTTP is only `/health`, the read-only `GET /v1/pool` debug mirror, and
`GET /v1/leases/{id}/history` (the lease's path through the graph).

Transport rule: the lease RPC travels on a fresh correlation id (the turn's id rides in
`turn_correlation_id`), and the pool records queue wait under the `gpu_pool_wait` rpc_health
label, which orion-equilibrium-service excludes from its transport baseline. Waiting in line is
not transport.

## Stage 1: observe mode (this deploy)

The pool discovers, answers leases, publishes state and events, and persists history. **Nothing
depends on it yet**: the gateway, durable-runs, world-model and diffusion still use their old
paths until stages 3-5. Swap decisions are published as `swap_requested` events with
`actuated: false`; no model is loaded or unloaded. Backlogged leases replay when their role
returns, but re-dispatching a *request* on behalf of a caller that has gone away arrives with the
gateway cutover (stage 3).

## Deploy (athena)

```bash
# 1. once: the projection tables. The LangGraph checkpoint tables create themselves, in their
#    own `gpu_pool` schema (the pool connects with search_path=gpu_pool,public). They must not
#    share public.checkpoints with durable-runs: its resume sweep lists every row there. On boot
#    the pool moves any lease threads it finds in public into its schema, then deletes
#    released leases' history after GPU_POOL_CHECKPOINT_RETENTION_HOURS (default 7 days, the
#    reach of backfill replay and the Hub walker).
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_gpu_pool_v1.sql
# 2. equilibrium must exclude the queue-wait hop BEFORE the pool publishes rpc_health:
#    EQUILIBRIUM_TRANSPORT_EXCLUDE_LABELS=log_orion_metacognition,gpu_pool_wait
# 3. the pool
scripts/safe_docker_build.sh orion-gpu-pool up -d --build
curl -s localhost:8127/health && curl -s localhost:8127/v1/pool | jq '.roles[] | {role, status, profile_name}'
```

circe's llama.cpp workers start announcing after they are recreated with the new compose
(`LLM_ROLE`, `LLM_ANNOUNCE_PORT`); until then their roles read `silent`, which is correct.

## Tests and eval

```bash
python scripts/check_gpu_pool_config.py
python -m pytest orion/gpu_pool/tests -q
cd services/orion-gpu-pool && python -m pytest tests -q        # + GPU_POOL_TEST_POSTGRES_URI for the Postgres tests
python services/orion-gpu-pool/evals/run_pool_day_eval.py      # exits 1 on owner starvation, lost leases, spill-down
```
