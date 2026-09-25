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
| live projection | `gpu_pool_leases`, `gpu_pool_cards` (`services/orion-sql-db/manual_migration_gpu_pool_v1.sql`, then `_v2_holds.sql`) |
| swap actuation engine + guards | `app/runtime.py` (`_begin_actuation`, `on_actuate_result`, `_check_actuation`), `app/guards.py` |
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
| `orion:gpu_pool:lease:request` | acquire / heartbeat / release / cancel / attach / status (RPC; answers at once). See "Durable-run holds" below. |
| `orion:gpu_pool:actuate:request` / `:actuate:result` | pool ↔ host actuator (`GpuActuateV1` / `GpuActuateResultV1`). The pool sends only for seats in `GPU_POOL_ACTUATE_ROLES` (empty by default = never). |
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

## Durable-run holds (stage 4.3)

Spec: `docs/superpowers/specs/2026-09-25-gpu-pool-stage4-durable-runs-and-actuation.md`. Built in
4.3, **used by nobody until durable-runs moves onto them (4.5)**.

- **A hold** (`acquire kind=hold`, `orion.gpu_pool.client.acquire_hold`) keeps one run on one
  role for the whole run. It is placed like any lease, **at most one per role**, and reserves one
  slot. Heartbeat window `hold_lease_ttl_sec` (90 s); a hold nobody heartbeats expires, retries and
  re-queues with the **same lease_id** (new generation).
- **Each call under it attaches** (`verb=attach`, `client.attached_lease`) with the hold's
  `lease_id` + `generation`. The child is a request lease (`hold_lease_id` column; never
  `parent_lease_id`, which is backfill lineage) that runs **in the hold's slot**: only on the hold's
  role, ahead of that role's queue. A hold plus its running call is one slot, never two -- a call
  whose run holds the only slot would otherwise queue behind itself forever.
- **Gaps are shared** (Juniper, 2026-09-25): while no call of the run is in flight, a lease of
  **strictly higher** priority may use the slot. Equal/lower priority and other holds may not.
- **Recall**: a hold borrowing another class's role is recalled as soon as that owner has demand
  there; a swap seat with `max_hold_sec` (agent-gpu2: 3600 s) drains after being loaded that long.
  A recalled hold gets `hold_clawback_grace_sec` (600 s) to finish its current node, then is
  aborted and re-queued. No cap on a hold on its home role in stage 4.
- **`status`** is a read with no side effect: resume after restart, Door-A validation.
- Attach refusals (`hold_unknown`, `not_a_hold`, `hold_not_granted:<status>`,
  `stale_hold_generation:<n>`) name the hold only in `reason`, never in `lease_id` -- the client
  cancels a failed reply's `lease_id`.

## Swap actuation (stage 4.3, OFF by default)

`GPU_POOL_ACTUATE_ROLES` (comma list of swap seats) arms it; empty = every swap decision stays a
`swap_requested {actuated: false}` event, as before. For an armed seat the pool:

1. persists the action on `gpu_pool_cards` (`swap_state`, `swap_role`, `swap_generation`,
   `swap_action`), **then** sends `GpuActuateV1 {action_id, generation, launch_digest, deadline_at}`
   and emits `swap_started`;
2. `accepted` must arrive within `actuate_ack_sec` (10 s), else `swap_failed reason=actuator_unreachable`
   and a cooldown;
3. `succeeded` -> `swapped` (the seat is grantable once discovery confirms its worker); a failed
   load with the residents restored (or `restored=None` and the containers show them up) -> idle +
   `swap_cooldown_sec`; anything else -> **`fault`**: no grants on any role of that card, no
   auto-retry, until discovery sees the card consistent again (then a cooldown);
4. no terminal result by `deadline_at` (sum of the launch timeouts it touches) -> `status`. A reply
   with `in_flight=true` (or, from an actuator that predates that field, a `phase`) keeps polling
   every 30 s, never faulting; otherwise the pool adopts the observed containers (a half-done card
   is a fault). A status gets 90 s to answer (the circe actuator asks docker first).
5. On a pool restart, a card left `loading`/`unloading` is reconciled with `status`, never a
   second transition. A seat already loaded by the old path is **adopted** from observation (its
   idle and max-hold clocks start then), not reloaded.

Loads are blocked -- reported as `swap_requested {actuated: false, reason}` -- by `min_residency`
(after an unload, `swap_min_residency_sec`), `cooldown`, and the seat's `swap.guards`:
`guard:thermal` (cabinet sensor; a degraded or missing reading blocks) and `guard:visual_baseline`
(visual chain baseline overdue or an attempt running; stage-4-only). Guards are read every
`GPU_POOL_GUARD_REFRESH_SEC` outside the lease lock; a guard never read blocks.

The Hub GPU-pool panel shows each card's `swap_state` (fault in red), the action in flight or last
finished, cooldown/residency, which guard blocks, and every hold with the calls running in it.

## Deploy (athena)

```bash
# 1. once: the projection tables. The LangGraph checkpoint tables create themselves, in their
#    own `gpu_pool` schema (the pool connects with search_path=gpu_pool,public). They must not
#    share public.checkpoints with durable-runs: its resume sweep lists every row there. On boot
#    the pool moves any lease threads it finds in public into its schema. A lease that ended
#    (released/unavailable) is forgotten -- checkpoint history and gpu_pool_leases row -- after
#    GPU_POOL_LEASE_RETENTION_HOURS (default 7 days, the reach of backfill replay and the Hub
#    walker). Dead letters are kept. Historical traffic lives in gpu_pool_events (sql-writer).
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_gpu_pool_v1.sql
# 1b. stage 4.3 (additive, lock_timeout 5s, index CONCURRENTLY; a 4.3 pool refuses to boot
#     without it). If the index build fails it leaves an INVALID index: DROP INDEX
#     gpu_pool_leases_hold_idx; and re-run.
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_gpu_pool_v2_holds.sql
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

## When lease RPCs are slow

Every lease verb and the 1 s tick share one lock. `GET /v1/lock-stats` returns, per op, how many
times it took the lock and its worst wait / hold since the previous call (then resets). A wait or
hold over 250 ms is also logged as `gpu_pool_slow_lock op=... phases={...}`, with time per phase
(probe, live_leases, schedule, resume, start_thread, bus_publish, publish_state). On 2026-09-25 the
phases were all database commits, stalled behind Postgres I/O from the substrate reconcile sweeps.
