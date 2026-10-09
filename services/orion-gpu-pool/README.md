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
| live projection | `gpu_pool_leases`, `gpu_pool_cards` (`services/orion-sql-db/manual_migration_gpu_pool_v1.sql`, then `_v2_holds.sql`, then `_v3_actuation_pause.sql`) |
| swap actuation engine + guards | `app/runtime.py` (`_begin_actuation`, `on_actuate_result`, `_check_actuation`), `app/guards.py` |
| lease history | `gpu_pool_events`, written by orion-sql-writer from `orion:gpu_pool:event` |

**Adding a card:** add it under `cards:` in the YAML, then add or extend the roles on it. **Adding
a service:** add a role (`kind: service` needs `slots` and `vram_gb`) and a class that uses it.
`python scripts/check_gpu_pool_config.py` checks the YAML against the llama.cpp compose files.

## More than one GPU node

`host` is the pool's home node; `hosts:` lists the others by name and address. A card on another
node says `host: <name>`. The pool reaches a role at its card's node address, labels grants
`<node>-worker-<role>`, and sends each card's `host` in `gpu_pool.state` (Hub labels biometrics
cards per node, since `index` repeats across nodes). A role's cards must all sit on one node; an
actuator may live on any listed node but only launches cards on its own. First use: hecate's
`agent-deep` (see `services/orion-llamacpp-host/README.md`).

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
| `orion:gpu_pool:actuate:request` / `:actuate:result` | pool ↔ host actuator (`GpuActuateV1` / `GpuActuateResultV1`). The pool sends only for swap seats with a `launch:` block in `config/gpu_pool.yaml`, and nothing while actuation is paused (stage 5.7). |
| `orion:gpu_pool:event` | lease facts; callers wake on `granted` |
| `orion:gpu_pool:state` (+ `:state:request`) | whole-pool snapshot |
| `orion:gpu_pool:control:request` | operator: lend, unlend, hold, release, replay, cancel, backfill, clear_fault, pause_actuation, resume_actuation. No token (the pool trusts the bus like every Orion service); each verb is logged with its actor and published as a pool event. |
| `orion:llm:worker:announce` | worker → pool discovery |

HTTP is only `/health`, the read-only `GET /v1/pool` debug mirror, and
`GET /v1/leases/{id}/history` (the lease's path through the graph).

Transport rule: the lease RPC travels on a fresh correlation id (the turn's id rides in
`turn_correlation_id`), and the pool records queue wait under the `gpu_pool_wait` rpc_health
label, which orion-equilibrium-service excludes from its transport baseline. Waiting in line is
not transport.

## Mode and the emergency stop (stage 5.7: enforce is the end state)

Runbook: `docs/runbooks/2026-09-30-gpu-pool-stage5-7-enforce.md`.

**Which seats the pool loads is config, not a switch.** A swap seat is actuated if and only if it has
a `launch:` block in `config/gpu_pool.yaml` (`PoolConfig.actuated_seats()`); today that is
`agent-gpu2`. `GPU_POOL_ACTUATE_ROLES` is deleted, with no fallback.

`GPU_POOL_MODE` (default `enforce`; any other value than `enforce`/`observe` fails the boot). Both modes
actuate the same seats. They differ in exactly three ways:

| | enforce (default) | observe (the rollback) |
| --- | --- | --- |
| a seat loaded or unloaded outside the pool | at boot and on resume, one read-only `status` per idle seat asks circe's actuator what its cards hold; the pool adopts the answer (`swapped reason=adopted:boot`), never reloads. A half-done card faults (`reconcile_ambiguous:*`); an action the pool never sent faults (`reconcile_foreign_action:*`); an actuator that cannot say whether something runs, or no answer in 90 s, keeps the stored state. Until the answer, that seat neither swaps nor drains | no `status` asked. Only a seat the pool has **never** acted on is marked loaded when its worker answers; `agent-gpu2` has been actuated (generation 79), so for gpu2 observe adopts nothing changed by hand |
| operator `hold` | allowed for a seat that can load | refused `hold_refused_observe_mode` |
| everything else | identical | identical |

**Operator holds on a seat nothing can load are refused in every mode**, with
`not_actuatable:<role>`: today `experiment` (deferred, no launch block). The scheduler never drains a
seat's residents for a seat without a launch, so even a lease that got past the refusal could not
empty the cards (stage 5 spec, "Corrections from building 5.1" item 5). The Hub greys the button out
with that reason.

**Emergency stop -- the one way to stop every model load and unload at once:** Hub GPU pool panel,
"Emergency stop: pause all model loading/unloading" (or control verb `pause_actuation`; from a shell,
`scripts/gpu_pool_pause.py pause|resume`). While paused:

- nothing is sent to the actuator; every swap decision is published as
  `swap_requested {actuated: false, paused: true}` with `reason=actuation_paused`;
- no seat is drained or recalled for a swap (the loaded 27B keeps serving; diffusion waits for gpu2);
- an action already in flight is NOT stopped (the actuator owns it) -- the pool keeps following its
  result. To stop that too, stop `orion-circe-gpu-lane-controller` on circe;
- operator holds are refused `actuation_paused`;
- it is persisted on every `gpu_pool_cards` row (`actuation_paused_at/_by`,
  `manual_migration_gpu_pool_v3_actuation_pause.sql`), so a restart stays paused; state carries
  `actuation_paused {paused, since, by}` and `/health` carries `actuation`;
- events `actuation_paused` / `actuation_resumed` record who and when.

"Resume" (`resume_actuation`) first asks the actuator what each card holds (enforce), then acts on
queued demand.

### History: stage 1 observe mode

Stage 1 deployed the pool with nothing depending on it; swap decisions were only published. The
modes have since changed meaning (above).

## Stage 5.1: config + contracts

Spec: `docs/superpowers/specs/2026-09-29-gpu-pool-stage5-world-diffusion-generic-actuation.md`.

- **`serialize_with`** (role key): nothing is placed on a role while a lease is active on a role it
  serializes with, in either direction. `world: serialize_with: [diffusion]` keeps the old
  `/capacity` permit's world/diffusion mutex on gpu2. No recall or preemption across the pair; the
  waiting lease keeps its place in queue order (it reserves the blocking partner, so younger work
  cannot keep taking it) and its deadline, and the pool emits one `queued` event with
  `reason=serialized:<role>`, `detail.serialized=true` (edge-triggered per lease). Removing the
  YAML line removes the rule. Nothing takes world leases until 5.4, so this is dormant.
- **Class `world`** is `on_unavailable: wait` (was backlog): a world prediction replayed hours later
  is useless.
- **`launch.cuda_env` now names the compose interpolation variable** the actuator sets from the
  card's `index`. `scripts/check_gpu_pool_config.py` refuses any launch service whose
  `CUDA_VISIBLE_DEVICES[_OVERRIDE]`/`NVIDIA_VISIBLE_DEVICES` is not exactly `${cuda_env}` or
  `${cuda_env:-<index>}`, and any literal `device_ids` pin.
- **`launch.profile_var` / `launch.profiles`**: the variable the actuator sets to the chosen
  `llm_profiles.yaml` profile (the service's `LLM_PROFILE_NAME` must interpolate it) and the
  allow-list, first = default. Since 5.3 the pool sends `launch.profiles[0]` with every `load`
  (`PoolConfig.load_profile`; `None` for unloads and for roles without `profiles`). Choosing a
  different entry by `needs_vision` / `min_ctx_tokens` / VRAM is **not built** (future). The profile
  is recorded on the card's `actuation` and on `swap_started`/`swapped`/`swap_failed`
  `detail.profile`. A seat that still has bridge verbs may not list `profiles` (the bridge refuses
  any profile, so every load would fail); the validator refuses that shape.
- **Stage 5.3 cutover:** `agent-gpu2` has no `swap.load/unload` bridge verbs; circe's controller
  runs its `launch` block and diffusion's through `launch_exec`. Visible differences: the seat's
  ready wait is 900 s (was the bridge's 600 s), so the first actuation deadline is 900 + 600 =
  1500 s and the stuck ceiling 3000 s; controller failure reasons carry the role
  (`upstream_not_idle:agent-gpu2`, `model_readiness_timeout:agent-gpu2`, ...). The pool never
  parses a reason -- it acts on `status`, `restored` and `observed` -- so it only records them.
  Runbook: `docs/runbooks/2026-09-29-gpu-pool-stage5-3-cutover.md`. End-to-end test:
  `tests/test_stage5_3_cutover_e2e.py` (real runtime -> real controller, fake docker).
- **`experiment`** lost its dead bridge verbs; the validator exempts an operator-only seat with no
  launch and no bridge (deferred; nothing can load it). Since 5.7 its operator hold is refused
  `not_actuatable:experiment` in every mode.
- **Seat limit**: `agent-gpu2 max_hold_sec: 9000` (Juniper 2026-09-29).


## Stage 5.4: world-model and image generation on pool leases

- **world-model** (circe, gpu2) takes a `world` request lease around every CUDA forward pass
  (`priority=system`, deadline `WM_GPU_LEASE_DEADLINE_SEC`=2 s). Still queued at the deadline ->
  its task returns `error_code=gpu_contended`; this pool's event for it reads
  `queued reason=serialized:diffusion` when a diffusion lease/hold is why. Unreachable pool ->
  `gpu_pool_unreachable`. Class `world` is `on_unavailable: wait` (5.1).
- **Image generation**: a reverie-visual durable run's `diffusion` hold (`hold_routes.diffusion`)
  is the grant; orion-thought's generate step validates it and attaches a child lease for the
  diffusion call itself (no second wait). The child outlives a hold the run gives back mid-render,
  so world stays off the card until the render ends. A generate outside a durable run (run-once
  route, legacy worker) takes a `diffusion` request lease. world-model likewise keeps its lease for
  up to one more `WM_TIMEOUT_S` when a forward pass times out but is still running.
- **Rollback is image-only, never a YAML-only revert**: `SwapGuard` no longer accepts
  `visual_baseline`, so a pre-5.4 `gpu_pool.yaml` fails to load under 5.4 code.
- **Nothing calls durable-runs `/capacity` any more** (gate:
  `orion/gpu_pool/tests/test_stage5_4_no_capacity_callers.py`). Deleted in 5.6.
- **`visual_baseline` swap guard deleted**, with `GPU_POOL_VISUAL_ACTIVITY_URL` and the image's
  copy of `config/proposals/visual_baseline.v1.yaml`. The 27B load is guarded by `thermal` only; an
  overdue image baseline reclaims gpu2 through the queue (a diffusion hold, owner reclaim).
- Eval: `run_pool_day_eval.py` now fails on any second where a world and a diffusion lease are
  granted together (`world_diffusion_grant_overlap_sec`), or if `serialize_with` never came up.

## Durable-run holds (stage 4.3)

Spec: `docs/superpowers/specs/2026-09-25-gpu-pool-stage4-durable-runs-and-actuation.md`. Built in
4.3, **used by nobody until durable-runs moves onto them (4.5)**.

- **A hold** (`acquire kind=hold`, `orion.gpu_pool.client.acquire_hold`) keeps one run on one
  role for the whole run. It is placed like any lease, **at most one per role**, and reserves one
  slot. Heartbeat window `hold_lease_ttl_sec` (90 s); a hold nobody heartbeats expires, retries and
  re-queues with the **same lease_id** (new generation).
- **Each call under it attaches** (`verb=attach`, `client.gpu_lease(..., hold=ref)`) with the hold's
  `lease_id` + `generation`. The child is a request lease (`hold_lease_id` column; never
  `parent_lease_id`, which is backfill lineage) that runs **in the hold's slot**: only on the hold's
  role, ahead of that role's queue. A hold plus its running call is one slot, never two -- a call
  whose run holds the only slot would otherwise queue behind itself forever.
- **Gaps are shared** (Juniper, 2026-09-25): while no call of the run is in flight, a lease of
  **strictly higher** priority may use the slot. Equal/lower priority and other holds may not.
- **Holds per role** (stage 7.3, `docs/superpowers/specs/2026-09-30-gpu-pool-stage7-concurrency.md`):
  `roles.<r>.max_holds` (default 1) non-urgent holds at once, one slot each. The scheduler clamps it
  to the slots discovery reads from `/props`, and `reserve_one_off_slots` (default 0) keeps that many
  slots for one-off calls (never the first hold). `GET /health` → `holds` shows configured, slots,
  the limit in force and why it is lower; a clamp is also logged once (`gpu_pool_max_holds_clamped`).
  Live: `agent-gpu2: max_holds: 2` (Bonsai, 2 × 131072). Rollback: delete that line.
- **Gap pinning** (7.3): a one-off call in a gap is charged to ONE idle hold, rebuilt every tick
  (an idle run with no call waiting first, then the most recently granted). Only that run waits
  one call; another idle run's next call gets its own slot at once. With two runs on two slots,
  one-off calls have no free slot: they wait for a run's gap (the shortest of the running calls).
- **Two holds on a swap seat**: an owner reclaim or `max_hold_sec` drain recalls both holds in the
  same tick (two take-backs, each with the 600 s grace); the seat unloads once both have left.
- **Recall**: a hold borrowing another class's role is recalled as soon as that owner has demand
  there; a swap seat with `max_hold_sec` (agent-gpu2: 9000 s, ~2.5 h since stage 5.1) drains after being loaded that long.
  A recalled hold gets `hold_clawback_grace_sec` (600 s) to finish its current node, then is
  aborted and re-queued **in its original place, without spending an attempt** (2026-09-29; the
  same path as an urgent pause. Before, each abort spent one of `retry.max_attempts` and the third
  recall dead-lettered the hold -- `unavailable:recall_grace_exceeded`). A request lease's abort
  and a hold's lost heartbeat still spend one. No cap on a hold on its home role in stage 4.
- **`status`** is a read with no side effect: resume after restart, Door-A validation
  (`client.validate_hold_ref`).
- Client for durable-runs (4.5): `acquire_hold`, `heartbeat_lease`, `lease_status`, `release_lease`,
  `hold_ref`, `durable_run_holder`. A child is never retryable, and an attach whose request_id
  names a different lease is refused `request_id_conflict`.
- Attach refusals (`hold_unknown`, `not_a_hold`, `hold_not_granted:<status>`,
  `stale_hold_generation:<n>`) name the hold only in `reason`, never in `lease_id` -- the client
  cancels a failed reply's `lease_id`.

## Urgent priority

Spec: `docs/superpowers/specs/2026-09-28-urgent-curiosity-and-hardware-watch-design.md` Part 1.
Rules U1–U3 in the `schedule()` docstring.

An `urgent` lease goes ahead of every queue. If no slot is free, the pool **pauses** one running
background (then system) durable-run hold on a role urgent could use, most recently granted first:
recall with reason `urgent_preempt`, then abort after the grace. The paused hold goes back in line
in its **original place** (same `lease_id`, same `created_at`, no retry attempt spent), and
durable-runs replays the interrupted node when it is granted again.

- Never paused: chat/interactive, other urgent, operator, one-shot request and child leases.
  Urgent reaches the `chat` role only while gpu0 is lent, and chat's own requests may still use an
  urgent hold's gaps there.
- An urgent owner taking back its own role pauses the hold borrowing it whatever that hold's
  priority -- system as well as background, even when a background hold elsewhere could be paused
  instead. Same pause (5 s grace, re-queued in place); it counts as the one pause.
- A paused run's last call may still be running on the slot after the abort. That call is still
  the pause: urgent waits for it instead of pausing a second run. While urgent is enabled,
  durable-runs heartbeats a held run at least every `urgent_preempt_grace_sec`, so it sees the
  recall before the abort and cancels the call about a second after it.
- Urgent only pauses holds on roles its own class may use, and only urgent leases the cap has room
  for are owed a pause or count as a waiting owner.
- Urgent holds may stack past a role's `max_holds` (bounded by slots and the cap below).
- `queued` and `recall` replies carry `reason`, so a caller can tell a pause from any other recall.

Defaults (`config/gpu_pool.yaml`):

- `urgent_preempt_grace_sec: 5` -- how long a paused hold gets before it is aborted and re-queued.
- `urgent_max_concurrent: 3` -- urgent leases active at once; a 4th waits and pauses nothing.

**Rollback:** set `urgent_max_concurrent: 0` and redeploy **both** `orion-gpu-pool` and
`orion-durable-runs`. Each bakes its own copy of the yaml into its image; durable-runs reads
`urgent_max_concurrent` to decide whether urgent runs skip its driver cap. Urgent then behaves
exactly like background: no pauses, no stacking. The eval's urgent scenario checks this too.

**Known limitation:** a finished run's hold kept for Hub outreach compose (Door-A) is an ordinary
background hold, so urgent work can pause it. Durable-runs then ends that hold at its next
outreach heartbeat (the pool may briefly re-grant it first), so an outreach compose still in progress loses its GPU
hold.

**Deploy order:** `orion-gpu-pool` and the circe `orion-gpu-lane-controller` change together. The
lane-controller re-reads `config/gpu_pool.yaml` live from circe's checkout (mounted at `/repo`), but
parses it with the `orion` code baked into its image, whose config models refuse unknown keys. Pull
circe's checkout without rebuilding the lane-controller and every actuation fails
`config_unloadable`. So pulling circe's checkout and rebuilding the lane-controller is one step.
`orion-llamacpp-host` needs nothing. `orion-sql-writer` must be redeployed before Plan 3 sends
urgent work: it validates pool events against the priority list.

Nothing sends urgent work yet (Plan 3 adds the trigger). A live pause smoke is **UNVERIFIED**: it
would pause a real background run, so it waits for Juniper's approval.

## Orion's learned shed (`orion_self_shed`, attend-to-act loop A1)

One more named reason on the U4 shed lever, below the reflex: `orion_self_shed`, precedence 1,
blocks NEW `background` grants only (never system, interactive, urgent; running work finishes;
nothing recalled). Set by execution-dispatch's `shed_background_gpu` action over
`orion:gpu_pool:shed:request` (`GpuPoolShedReasonRequestV1`, set/clear/status; the schema refuses
`cooling_incident`). Code: `orion/gpu_pool/orion_shed.py`; ledger `public.gpu_pool_orion_shed`
(`app/orion_shed_store.py`, migration `services/orion-sql-db/manual_migration_gpu_pool_orion_shed_v1.sql`).

- Caps, enforced here: `GPU_POOL_ORION_SHED_MAX_TTL_SEC` (900), `..._MAX_SEC_PER_DAY` (3600, rolling
  24 h, survives a restart), `..._MIN_GAP_SEC` (900, end of one to start of the next), one at a time.
- Refused: `disabled` (`GPU_POOL_ORION_SHED_ENABLED=false`, the default), `lever_disabled`
  (`GPU_POOL_SHED_ENABLED=false`), `reflex_active`, `already_active`, `min_gap`, `daily_cap`,
  `ledger_unavailable` (fail closed).
- Terminal: `expired` (TTL), `cancelled` (clear verb, or the kill switch at boot),
  `preempted_by_reflex` (a cooling incident OPENED -- the AC is no longer healthy).
- Manipulation check on the record: `drained_at`, `grants_withheld`, `delayed_grant_sec`.
- `/health` and the pool state carry `shed.orion_self_shed` (caps, 24 h use, the active record).

## Swap actuation (stage 4.3; config-armed since 5.7)

Every swap seat with a `launch:` block is actuated (see "Mode and the emergency stop"). For such a
seat the pool:

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
   is a fault). A status gets 90 s to answer (the circe actuator asks docker first). A reply that
   cannot say whether the action runs is re-asked 4 times before the pool believes the containers,
   and any action still unfinished after 2x its timeout faults the card (`actuator_stuck`).
   `GpuPoolControlV1 verb=clear_fault card=<card>` (Hub button on a faulted card) reconciles with
   `status` and adopts the answer, then cools down; a fault discovery cannot clear needs this.
5. On a pool restart, a card left `loading`/`unloading` is reconciled with `status`, never a
   second transition. An idle seat is **adopted** from the actuator's `status` answer in enforce
   (from its worker's liveness in observe); its idle and max-hold clocks start then. Never reloaded.

Loads are blocked -- reported as `swap_requested {actuated: false, reason}` -- by `min_residency`
(after an unload, `swap_min_residency_sec`), `cooldown`, and the seat's `swap.guards`:
`guard:thermal` (cabinet sensor; a degraded or missing reading blocks). The stage-4
`guard:visual_baseline` was deleted in stage 5.4: a reverie-visual run's diffusion hold reclaims gpu2
through the queue (owner reclaim) instead. Guards are read every
`GPU_POOL_GUARD_REFRESH_SEC` outside the lease lock; a guard never read blocks.
A blocked swap (load or unload, including an unload held back by `cooldown` after a refused or
failed action) is reported **once per episode** -- same seat, action and reason -- not once per
tick. A new reason, or the block clearing and recurring, is a new episode and is reported again.
`swap_started`/`swapped`/`swap_failed` are never deduplicated (stage 6.1).

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
# 1b/1c. stage 4.3 and 5.7 (additive). Still the recommended order -- but no longer a boot
#     precondition: the pool adds these columns itself (see "Boot schema self-heal" below). Run
#     1b anyway for its index (gpu_pool_leases_hold_idx, CONCURRENTLY), which boot never builds.
#     If the index build fails it leaves an INVALID index: DROP INDEX gpu_pool_leases_hold_idx;
#     and re-run.
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_gpu_pool_v2_holds.sql
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_gpu_pool_v3_actuation_pause.sql
# 2. equilibrium must exclude the queue-wait hop BEFORE the pool publishes rpc_health:
#    EQUILIBRIUM_TRANSPORT_EXCLUDE_LABELS=log_orion_metacognition,gpu_pool_wait
# 3. the pool
scripts/safe_docker_build.sh orion-gpu-pool up -d --build
curl -s localhost:8127/health && curl -s localhost:8127/v1/pool | jq '.roles[] | {role, status, profile_name}'
```

### Boot schema self-heal (since 2026-09-30)

Every LLM call leases through this pool, so a pool that will not boot is a total LLM outage. That
happened twice (2026-09-26 v2 `hold_lease_id`, 2026-09-30 v3 `actuation_paused_at`): the image was
deployed before its additive migration and `check_schema` refused to start. Now, after taking the
leader lock (single writer), the pool runs the missing `ALTER TABLE .. ADD COLUMN IF NOT EXISTS`
statements itself -- `BOOT_ADDITIVE_COLUMNS` in `app/store.py`, one `ALTER TABLE` per table (atomic),
stopping at the first lock failure, under `SET LOCAL lock_timeout` (3 s at boot, 300 ms on retries:
a waiting ALTER queues the pool's own writes behind it) and `statement_timeout` 15 s.

- **Lock not granted in time** (a `pg_dump`, a long transaction): the pool does **not** exit. It
  serves **degraded**: the missing columns are held in memory for this process (holds, attach
  idempotency, swap state and the emergency stop all still work), it logs
  `gpu_pool_schema_degraded` at CRITICAL, retries in the background (5 s, 10 s, 30 s, 60 s, then
  every 120 s), and on success writes the held values through (`gpu_pool_schema_recovered`). Until
  then those values would not survive a restart. `/health` carries it:
  `curl -s localhost:8127/health | jq .schema` -> `state` (`ok`/`degraded`), `missing`,
  `in_memory`, `last_error`, `attempts`, `degraded_since`. Operator fix: run the migration named in
  the log (the pool's own retry then finds nothing to do).
- **Never applied at boot:** tables (v1), indexes, type changes, drops, or a missing column not in
  the boot list. Those raise `SchemaNotHealable` and the pool refuses to boot, loudly, as before.
- **Drift gate:** `tests/test_schema_drift_gate.py` fails if the boot list and the
  `manual_migration_gpu_pool_v*.sql` `ADD COLUMN`s differ in either direction or in type/default,
  if a column the store writes is created by neither v1 nor the boot list, or if a boot entry is not
  additive (NOT NULL without a constant default, etc.). A new additive migration therefore needs
  its boot entry in the same patch.

circe's llama.cpp workers start announcing after they are recreated with the new compose
(`LLM_ROLE`, `LLM_ANNOUNCE_PORT`); until then their roles read `silent`, which is correct.

## Tests and eval

```bash
python scripts/check_gpu_pool_config.py
python -m pytest orion/gpu_pool/tests -q
cd services/orion-gpu-pool && python -m pytest tests -q        # + GPU_POOL_TEST_POSTGRES_URI for the Postgres tests
python services/orion-gpu-pool/evals/run_pool_day_eval.py      # exits 1 on owner starvation, lost leases, spill-down, or an urgent pause/resume miss
```

## When lease RPCs are slow

Every lease verb and the 1 s tick share one lock. `GET /v1/lock-stats` returns, per op, how many
times it took the lock and its worst wait / hold since the previous call (then resets). A wait or
hold over 250 ms is also logged as `gpu_pool_slow_lock op=... phases={...}`, with time per phase
(probe, live_leases, schedule, resume, start_thread, bus_publish, publish_state). On 2026-09-25 the
phases were all database commits, stalled behind Postgres I/O from the substrate reconcile sweeps.
