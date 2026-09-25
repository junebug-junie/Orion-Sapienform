# GPU pool stage 4 — durable runs on pool leases, and one generic way to load a model on a card

Status: design, for Juniper's review. No service code in this PR.
Date: 2026-09-25
Parent spec: `docs/superpowers/specs/2026-09-24-gpu-pool-design.md` (stages 1–3 live, pool in observe mode).
Decisions this builds on (Juniper, 2026-09-25): stage 4 moves durable-runs onto pool leases and deletes its own
admission and elastic *decision* logic (kill means kill, no fallback to the old broker); **Option A**: whatever
physically loads the gpu2 27B today keeps working until stage 5 replaces it; loading must be **generic per card
and role, YAML-driven**, so adding a card is config plus an actuator that knows how to start that role there.

## Arsonist summary

Two schedulers hand out the same agent card and neither knows the other exists. The GPU pool places every single
LLM call. Durable-runs separately hands one run at a time an exclusive "agent seat" for the whole run (26 minutes
to 7.8 hours) and, on its own, decides when to swap gpu2 from diffusion to a second 27B. Today 13 runs are
queued behind that seat, the oldest for 4 days, and the pool cannot see any of them.

Stage 4 makes the pool the only decider:

- **A durable run asks the pool for a *hold*.** The hold keeps the run on one model for the whole run, as today.
  But when the run is between calls (in a tool phase, about 40% of the time, measured below), the pool may slip
  short, higher-priority single calls onto that card. The run's own calls go to its held card and never queue
  behind themselves.
- **The pool decides when to load gpu2; the existing circe controller still does the physical load.** The pool
  sends a generic "load role X on cards Y" message, and in stage 4 the controller answers it with the same
  drain/stop/start/rollback code it runs today. The only thing that changes is *who asks*.
- **That same message is the permanent seam.** In stage 5 the controller becomes a generic actuator that reads
  "how to start role X" from `config/gpu_pool.yaml`. Adding gpu4 then means YAML plus a compose service, with no
  pool code change.

## Current architecture (verified 2026-09-25)

### How gpu2 gets its 27B today, end to end

1. **The trigger is a waiting durable run.** Durable-runs' broker (`orion/durable_admission/broker.py`) looks at
   a pending demand whose `alternatives` include `agent-burst`. `activation_decision()` (`elastic.py:8-37`)
   suppresses the load unless all of these hold: widening is on, the run allows it, the run has waited
   ≥ `threshold` (live payload: `threshold_seconds: 1200`), the lane is declared compatible, and the physical
   environment says eligible. It also applies a cold-start hysteresis: the load has to be predicted to beat the
   home seat.
2. **The physical eligibility check** (`services/orion-durable-runs/app/elastic_runtime.py:31-69`
   `environment()`):
   - it reads the cabinet thermal sensor (`orion.autonomy.thermal_gate.thermal_state`);
   - it checks that the visual chain is not overdue for its baseline (`thought /visual-chain/activity` →
     `visual_baseline_urgent`);
   - it asks the controller whether gpu2 is enabled and holds diffusion or agent-burst.
3. **Intent goes into Postgres.** `ElasticStore.intent()` writes `durable_elastic_slot`: generation+1,
   `operation_id=gpu2:<gen>:agent-burst`, `state=requested`.
4. **The action.** `ElasticRuntime.actuate()` makes an HTTP call, `POST http://100.112.254.99:8090/v1/gpu-slots/activate
   {slot: circe-gpu2, target, operation_id, generation}`, to `orion-gpu-lane-controller` on circe (timeout 1200s).
5. **The controller** (`services/orion-gpu-lane-controller/app/gpu2.py` `transition()`):
   - **fences** by calling back `GPU2_AUTHORITY_URL/elastic/status` (durable-runs) and refusing a stale
     operation or generation, or any request while durable admissions are still open;
   - **drains diffusion** (`/v1/lifecycle/drain`, waiting for `in_flight=false`) and stops it;
   - **starts the 27B**: `docker compose -f services/orion-llamacpp-host/docker-compose.atlas-workers.yml up -d
     --no-build --no-deps atlas-agent-burst` (compose profile `agent-burst`, `LLM_ROLE=agent-gpu2`,
     `CUDA_VISIBLE_DEVICES_OVERRIDE=2`, port 8016);
   - **waits** for `/health`;
   - **rolls back on failure**: stops agent-burst, restarts diffusion, reports `restored`.
6. **Guards that keep it from flapping**, all in durable-runs:
   - `DURABLE_RUNS_ELASTIC_MIN_RESIDENCY_SEC=600`: after a restore, no reload for 600s. The live reason
     `diffusion_min_residency` comes from here.
   - `…_IDLE_GRACE_SEC=300`: an idle 27B is unloaded after this.
   - `…_MAX_BORROW_SEC=3600`: a hard cap on how long gpu2 stays borrowed.
   - The drain, transition and cold-load budgets are 300s, 60s and 600s.
7. **The unload** is the same path with `target=diffusion`. The controller checks the 27B's `/slots` are idle
   before stopping it.
8. **Live state:**
   - the slot is at generation 8; the last load was at 06:33 UTC and it was restored at 08:20 UTC;
   - the pool served 31 agent leases on `agent-gpu2` in that window, because its observe mode marks a swap seat
     loaded when the worker announces itself and answers `/props` (`runtime.py:174`);
   - the pool itself has only ever *published* two `swap_requested` events, and actuated nothing.

The pool already has this logic in pure form. `schedule()` emits `SwapLoad` and `SwapUnload`, with
`swap_after_wait_sec`, `swap_idle_unload_sec` and `swap_cooldown_sec`, plus `max_hold_sec` recall
(`orion/gpu_pool/scheduler.py:370-405`). It just never acts on them (`runtime.py:398` `_swap`).

### How a durable run gets and uses the agent card today

- Graph `resource_request` → `resource_wait` (`admitted_graph.py:48-61`) registers a demand, then interrupts
  until the broker writes an active row in `durable_resource_leases`. The unique indexes allow one active lease
  per run, per lane and per backend.
- `AdmissionRuntime.execute()` (`admission_runtime.py:188-228`) renews the lease every heartbeat while the
  harness turn runs, and cancels the harness if the lease is lost.
- The lease token rides every LLM call:
  - **FCC** puts it in the header `X-Orion-Resource-Lease` (`orion/harness/fcc_motor.py:730-743`,
    `orion/llm/resource_lease.py` `LEASE_HEADER`);
  - **bus calls** put it in `options.resource_lease`;
  - the gateway's `LeaseGuard` (`services/orion-llm-gateway/app/resource_lease.py`) validates it against
    `durable-runs /leases/validate` every check interval.
- **But since stage 3 the gateway also takes its own pool request lease for the same call** (`main.py:345-348`).
  The durable lease is only an admission token, so each run is effectively scheduled twice.
- The per-call turn id is derived from `lease_id` + `generation` (`graph.py:112-123`), so the harness can cancel
  the right subprocess.
- **Door-A:** after a run finishes with `reach_out`, the lease stays active while Hub composes the message.
  Hub validates it (`HUB_CURIOSITY_LEASE_VALIDATION_URL`) and releases it via `/runs/{id}/release-outreach-lease`.

### Live numbers (2026-09-25, `orion-athena-sql-db`)

| fact | value |
| --- | --- |
| agent role slots (`/props` :8015) | 1 slot, 131072 ctx |
| durable runs granted in last 2 days | 43; hold p50 **35 min**, max **466 min**, min 8 min |
| pending demands | 13; oldest created 2026-09-21 13:34 (~4.2 days) |
| last-24h durable hold time vs FCC GPU time | 1081 min held; `http:anthropic` agent leases busy 616 min → **~57% busy / ~43% idle**. UNVERIFIED attribution: `http:anthropic` is not durable-runs-only |
| newest suppression payload | `suppression_reason: visual_baseline_urgent`, `thermal_state: elevated`, both burst lanes `health_unknown_or_unavailable`, broker estimate for `agent` = 92000 s |
| durable events, last 2 days | 218 `run.lane_swap_suppressed`, 4 `resource.elastic_*` ops (2 failed then 2 completed at 08:19–08:20) |
| gateway permits (`durable_gateway_permits`) | LLM lanes stopped 03:55 UTC (stage 3 cutover); **diffusion permits still live** (06:18), world-model 09-24 |

The last row matters. World-model and visual chain still get their GPU permits from durable-runs `/capacity`
(`orion/durable_admission/capacity.py`, `capacity_client.py`). **`capacity.py`, `/capacity` and the tables it
joins must survive stage 4.** They go in stage 5, as the parent spec already orders.

## Decision 1: hold semantics — one run keeps one model, the card is shared between its calls

The three options:

| option | what a run holds | cost |
| --- | --- | --- |
| A. strict run hold (today) | the whole card for the whole run | ~43% of a 27B idles in tool phases. Every other agent-class call queues behind a 35-min-to-7.8h hold. That is **worse than today**, because today the durable lease is not a pool lease and those calls slip in. |
| B. per-inference leases only | nothing between calls | the run can hop models mid-turn (27B gpu1 → 35B gpu0 when lent), which breaks affinity. Every hop re-prefills a large FCC prompt (up to 131k ctx) on a V100. The turn-cancel identity loses its stable lease id. |
| **C. affinity hold with interleave (recommended)** | the *role* for the whole run, plus the slot only while one of its calls is in flight | a higher-priority single call may occupy the slot between the run's calls, so the run's next call waits at most one inference, and loses its prefix cache only then |

Rules for C:

1. **Acquire.** `resource_request` sends `acquire {kind: hold, work_class: agent, priority: background,
   holder: durable-runs:<run_id>, retryable: true, request_id: <run_id>:<attempt>, deadline_at}`. The class and
   priority come from the run's route via `gpu_pool.yaml routes`, with `ResourceRequirementV1.priority`
   overriding. The pool places a hold like any lease: home `agent` first, then `agent-gpu2` when loaded, then
   `chat` when lent. **At most one hold per role.**
2. **Calls under a hold are child leases.** Each of the run's LLM calls carries a `GpuLeaseRefV1`. The gateway
   sends `attach {lease_id, generation}` and the pool creates a request lease with `parent_lease_id = hold`
   (the column already exists in `gpu_pool_leases`). Children:
   - are placed only on the hold's role;
   - jump that role's queue;
   - release on call end;
   - get the normal heartbeat/recall/telemetry machinery for free.
3. **Interleave.** A hold with no active child reserves its role against other holds and against leases of equal
   or lower priority. A request lease of **strictly higher** priority may be granted the free slot. In practice
   that is a system/interactive `agent` call from cortex-exec, or system metacog spill. Background never
   interleaves.
4. **Heartbeat.** Durable-runs heartbeats the hold every `DURABLE_RUNS_LEASE_HEARTBEAT_SEC`, including while
   queued. `hold_lease_ttl_sec: 90` (exists). A dead durable-runs means the hold expires, goes to `retry_wait`,
   and is re-queued with the same `lease_id`.
5. **Recall.** Holds only get recalled when:
   - the seat is borrowed and its owner wants it back (gpu0 unlend, and in stage 5 diffusion reclaiming gpu2);
   - `max_hold_sec` passes;
   - an operator acts.

   Grace is a new default, `hold_clawback_grace_sec: 600`. During grace the run's durable-runs driver sees
   `recall` on the heartbeat reply and releases at its next graph-node boundary. After grace, the pool aborts
   the hold: durable-runs cancels the harness turn (the existing `_cancel_harness`) and `retry_wait` re-queues it.
6. **Resume after restart.** `lease_id` and `generation` live in the checkpointed graph state. On resume,
   `resource_wait` calls a new `status` verb for that `lease_id`:
   - `granted` → continue;
   - `queued` or `backlogged` → interrupt again;
   - `unknown_lease` → back to `resource_request` with a new `request_id`.

   Wakes come from `orion:gpu_pool:event granted` filtered on the holder, plus the existing reconcile tick.
7. **Door-A.** `finish` with `reach_out` keeps the hold active, and durable-runs keeps heartbeating it until Hub
   calls `release-outreach-lease`, which now becomes a pool `release`. Hub's validation becomes a pool `status`
   call.

**Why the gateway must attach, not lease.** The agent role has one slot. If the gateway took a normal request
lease for a call whose run already holds that slot, the call would queue behind its own run forever. This is the
single most important correctness rule of stage 4. Acceptance check 2 watches it.

No cap on a hold on the *home* `agent` seat in stage 4, which is like-for-like with today. See Missing question 2.

## Decision 2: one actuation contract, bridged in stage 4, general in stage 5

### Who decides and who acts

| concern | lives in | why |
| --- | --- | --- |
| *whether* to load or unload a seat: wait threshold, cooldown, min residency, idle unload, max hold, thermal, visual-baseline urgency | **pool policy** (scheduler + YAML) | one decider that sees every card. Thermal is cabinet-wide, so a hot cabinet should block loading *any* extra model on *any* card |
| *how* to do it safely: generation fence, config fence, drain before stop, upstream idle before stop, container state known, readiness wait, rollback | **actuator safety** | it refuses unsafe steps whoever asks, but never decides whether |

### YAML shape (backward compatible)

These are new optional keys. Today's `swap: {evicts, load, unload}` stays valid.

```yaml
defaults:
  hold_clawback_grace_sec: 600     # NEW: recalled holds finish their current node within this
  swap_min_residency_sec: 600      # NEW: after a seat unloads, the evicted residents stay this long (was DURABLE_RUNS_ELASTIC_MIN_RESIDENCY_SEC)
  actuate_ack_sec: 10              # NEW: no "accepted" within this -> actuator_unreachable

actuators:                         # NEW: one per host; the pool routes by name
  circe: {host: circe}

cards:
  gpu2: {vram_gb: 32, index: 2}    # NEW optional index: the actuator's CUDA device for this card

roles:
  diffusion:
    kind: service
    cards: [gpu2]
    owner: diffusion
    port: 8014
    slots: 1
    vram_gb: 24
    launch:                        # NEW: how to start/stop this role on its cards
      actuator: circe
      compose: services/orion-diffusion-host/docker-compose.yml
      env_file: services/orion-diffusion-host/.env
      service: diffusion-host
      cuda_env: CUDA_VISIBLE_DEVICES
      drain: {set: /v1/lifecycle/drain, status: /v1/lifecycle/status}
      ready: /ready
      timeout_sec: 600
  agent-gpu2:
    kind: llm
    cards: [gpu2]
    owner: agent
    port: 8016
    max_hold_sec: 3600             # was DURABLE_RUNS_ELASTIC_MAX_BORROW_SEC
    launch:
      actuator: circe
      compose: services/orion-llamacpp-host/docker-compose.atlas-workers.yml
      env_file: services/orion-llamacpp-host/.env
      service: atlas-agent-burst
      compose_profile: agent-burst
      cuda_env: CUDA_VISIBLE_DEVICES_OVERRIDE
      ready: /health
      timeout_sec: 900
    swap:
      evicts: [diffusion]
      load: gpu2/agent             # stage-4 bridge verb; deleted in stage 5
      unload: gpu2/restore         # stage-4 bridge verb; deleted in stage 5
      after_wait_sec: 1200         # NEW per-seat override of swap_after_wait_sec: keeps today's 1200s trigger
      guards: [thermal, visual_baseline]   # NEW: pool-side preconditions for a load
```

Validation additions (`orion/gpu_pool/config.py`, `scripts/check_gpu_pool_config.py`):

- A swap seat needs either `load` and `unload` (bridge) **or** a `launch` on itself and on every role it evicts.
- `launch.actuator` must be in `actuators`.
- Every `launch.compose` + `service` must exist in that compose file with the role's `LLM_ROLE` and port.
- Every card a `launch` role spans needs an `index`.
- `guards` may only name guards the scheduler implements (`thermal`, `visual_baseline`).
- The existing VRAM, eviction and port checks are unchanged.

### Contract

The channels `orion:gpu_pool:actuate:request` (pool → actuator) and `orion:gpu_pool:actuate:result`
(actuator → pool) are registered in `channels.yaml` and `registry.py`.

```text
GpuActuateV1          # replaces the stage-1 placeholder (target/role/cards only; no producer or consumer exists yet)
  action_id: str      # idempotency key; a replayed action_id returns its recorded result
  generation: int     # pool-issued, monotonically increasing per card set; actuator rejects <= last seen
  actuator: str       # which host actuator must act ("circe"); others ignore it
  role: str
  action: load | unload | status
  cards: list[str]
  profile: str | None # llm_profiles.yaml profile to start the role with; stage 4 always sends None (compose default)
  launch_digest: str  # sha256 of the role's launch + its evicted roles' launch; actuator refuses on mismatch
  deadline_at: datetime
  reason: str         # "demand", "idle", "max_hold", "operator", ...

GpuActuateResultV1
  action_id, generation, role, action
  status: accepted | progress | succeeded | failed | refused
  phase: draining | stopping | starting | ready_wait | rolling_back | None
  restored: bool | None   # on failed load: were the evicted residents put back?
  elapsed_ms, reason
  observed: dict[str, str]  # role -> running|exited|absent|unknown after the action (for reconcile)
```

**Semantics.**

- A **load** of a seat means: drain and stop each evicted role, start the seat, wait for ready.
- An **unload** means: check the seat is upstream-idle, stop it, then start each evicted resident and wait for
  ready.
- **`status`** returns the last action and the observed containers. The pool sends it on its own restart, and
  whenever a result is overdue.
- The actuator runs one action per card set at a time. A second one gets `refused reason=busy`.
- The actuator only ever touches compose services named in **its own checked-out copy** of `gpu_pool.yaml` under
  its own actuator name. The bus message names a role, never a container. This keeps today's "no generic
  remote-docker API" safety property.

**Timeouts.** No `accepted` within `actuate_ack_sec` gives `swap_failed reason=actuator_unreachable`. No terminal
result by `deadline_at` (the role's `launch.timeout_sec`) makes the pool send `status`. It adopts whatever that
reports; if the actuator is still unreachable, the card is marked `fault`.

**Pool swap state during a load** (`GpuCardStateV1.swap_state`, which gains `fault`):

```text
idle --SwapLoad--> loading      evicted roles get no new grants; their leases are recalled (stage 5; in stage 4
                                diffusion is not leased and the actuator's drain covers it); seat not grantable
loading --succeeded--> idle     seat added to swapped_in, but grantable only once discovery confirms the announced
                                profile against /props (the existing health gate)
loading --failed, restored=true--> idle      cooldown_until = now + swap_cooldown_sec
loading --failed, restored=false--> fault    no grants on any role of that card; operator event; no auto-retry
fault --operator clear or discovery sees residents healthy--> idle
```

The observation shortcut `_observe_swap_seats` stops for roles the pool actuates. For those, the pool's
intent is the truth. Discovery still checks it, so a 27B container that dies becomes `down` with no grants.

**Guards (pool side).**

- `thermal` uses the same `thermal_state()` call over the cabinet URL that `elastic_runtime.py:36-49` uses today.
- `visual_baseline` is the `thought /visual-chain/activity` urgency check. **It is stage-4-only and deleted in
  the stage 5 PR that puts the visual chain on diffusion leases.** From then on diffusion is a lease owner and
  reclaims gpu2 through the scheduler.
- A failing guard produces `swap_requested {actuated: false, reason: guard:<name>}` edge-triggered, the same
  shape as today's observe events.

### Stage 4 bridge (Option A)

`orion-gpu-lane-controller` gains a bus consumer for `orion:gpu_pool:actuate:request`, with `actuator == circe`
and the role having `swap.load` set:

- `agent-gpu2 load` → `gpu2.transition(target="agent-burst")`;
- `agent-gpu2 unload` → `gpu2.transition(target="diffusion")`.

The drain, stop, start, readiness and rollback code is reused unchanged. Only the fence (`authority()`) changes:
under `GPU2_AUTHORITY=pool` it checks the message's `generation` against the last one it accepted, persisted to
a small state file on the controller's volume, and `launch_digest` against its own YAML. It no longer calls back
to durable-runs `/elastic/status`. The HTTP `/v1/gpu-slots/*` routes stay for the GPU1 affect flip, which is out
of scope here. The gpu2 activate route is deleted in PR 4.6.

**No window where nobody can open gpu2.** The deciders switch with three env flips in one maintenance step (PR 4.5
runbook). The durable elastic decider goes shadow first (gpu2 frozen in its current state), then the controller
fence moves to the pool, then the pool arms actuation for `agent-gpu2`. The gap is one pool restart. When armed,
the pool seeds its seat state from observation, so a 27B already loaded by the old path is adopted, not reloaded.

### Stage 5 (for scale; not built here)

The controller becomes the generic actuator:

- `load` and `unload` of any role with a `launch` block, built from the compose path, service, profile, `cuda_env`
  set from the cards' `index`, drain hook and ready path;
- the `swap.load`/`unload` bridge verbs and `gpu2.py`'s fixed targets are deleted;
- world and visual chain lease, and the `visual_baseline` guard and `capacity.py` are deleted.

### Worked example: adding gpu4, which hosts either an 8B or a vision model

```yaml
cards:
  gpu4: {vram_gb: 32, index: 4}
roles:
  fast2:
    kind: llm
    cards: [gpu4]
    owner: [metacog, fast]
    port: 8017
    launch: {actuator: circe, compose: services/orion-llamacpp-host/docker-compose.atlas-workers.yml,
             env_file: services/orion-llamacpp-host/.env, service: atlas-fast2,
             cuda_env: CUDA_VISIBLE_DEVICES_OVERRIDE, ready: /health, timeout_sec: 300}
  vision4:
    kind: llm
    cards: [gpu4]
    owner: vision
    port: 8018
    launch: {actuator: circe, compose: services/orion-llamacpp-host/docker-compose.atlas-workers.yml,
             env_file: services/orion-llamacpp-host/.env, service: atlas-vision4, compose_profile: vision4,
             cuda_env: CUDA_VISIBLE_DEVICES_OVERRIDE, ready: /health, timeout_sec: 600}
    swap: {evicts: [fast2], guards: [thermal]}
classes:
  fast:    {roles: [fast, metacog, fast2, agent, agent-gpu2, chat], on_unavailable: wait}
  vision:  {roles: [vision4], on_unavailable: backlog}
```

What else it takes:

- **Compose:** two services, `atlas-fast2` (always on) and `atlas-vision4` (profile `vision4`), each with
  `LLM_ROLE`, `LLM_ANNOUNCE_PORT` and a profile from `llm_profiles.yaml`. `check_gpu_pool_config.py` refuses a
  mismatch.
- **Pool code:** none. The scheduler already handles a resident plus a swap seat on one card (gpu2's shape).
  VRAM is checked against the *discovered* profile.
- **Actuator code:** none in stage 5. In stage 4 a new role would need a bridge target, which is exactly why the
  bridge is kept to `agent-gpu2` and generalised in stage 5.
- **Vision requests** (`needs_vision`, which exists in `gpu_pool_leases`) queue as class `vision`. After
  `swap_after_wait_sec` the pool loads `vision4`, evicting `fast2`. When `vision4` idles past
  `swap_idle_unload_sec` it unloads and `fast2` comes back.

## Telemetry and readers that move in stage 4

| reader | reads today | after stage 4 |
| --- | --- | --- |
| field-digester `queue_contention` source `durable_demand_pending` (count + oldest-wait, PR #2337) | `durable_resource_demands WHERE status='pending'` | the union of that **and** `gpu_pool_leases WHERE kind='hold' AND status IN ('queued','backlogged')` (PR 4.4). Before cutover the second half is 0; after migration the first half is 0; the legacy half is dropped in 4.6. `gpu_pool_waiting` gains `AND kind='request'`, so holds are never double-counted and each source keeps its own calibrated expected-wait anchor. Same source keys and meaning, new table. The metric gate provenance and live-data check are recorded in the PR 4.4 report. |
| Hub curiosity views (`curiosity_run_store.py`, `orion/curiosity/run_story.py`) | `durable_admission_runs`, `durable_resource_events` | **unchanged tables.** Durable-runs keeps writing the run registry and its event outbox, and emits the same names from pool replies: `run.waiting_resource` (queued), `run.resource_granted` + `run.lane_assigned` (granted, lane = role), `resource.lease_released`, `resource.lease_expired`. Stopped: `run.lane_swap_suppressed`, `run.resource_eligibility_expanded`, `resource.elastic_*` (already ignored by run_story). |
| gpu2 swap history | `durable_resource_events resource.elastic_*` | `gpu_pool_events`: `swap_requested` (exists) + new `swap_started`, `swapped`, `swap_failed`, `actuate_refused` |
| queue wait vs transport | durable wait invisible to rpc_health | hold queue wait goes under the pool's `gpu_pool_wait` hop, which is already in `EQUILIBRIUM_TRANSPORT_EXCLUDE_LABELS`. The hold RPC uses a fresh correlation id, as all lease RPCs do. Durable-runs' `http:100.112.254.99:8090/*` and cabinet/thought hops disappear from its rpc_health (expected; no reader keys on them, grep-verified in PR 4.5). |
| grammar | none from durable admission | none new. Holds and children are pool leases, so they ride the pool's existing lease lifecycle events. |
| Hub turn orchestrator / curiosity Door-A validation | `/leases/validate` | pool `status` verb |
| cortex-orch `durable_runs.py`, Hub `curiosity_investigation.py`, cortex-exec `self_study.py` (producers of `DurableRunRequestV1.admission`) | set `preferred_lane`, `allow_elastic_activation` | unchanged wire type in stage 4. `ResourceRequirementV1` keeps `preferred_lane` (→ route), `priority`, `deadline_at`, `requirements.min_ctx_tokens`. `allow_elastic_activation`, `alternatives`, `pinned_lane`, `operator_override` are accepted and ignored until producers stop sending them (PR 4.6), because the model is `extra="forbid"`. |

## Proposed schema/API changes

- `orion/schemas/gpu_pool.py`:
  - `GpuLeaseRequestV1.verb` adds `attach` and `status`;
  - `GpuLeaseRequestV1` adds `parent_lease_id`, `parent_generation`;
  - `GpuActuateV1` and `GpuActuateResultV1` are replaced as above;
  - `GpuPoolEventV1.event` adds `swap_started`, `swap_failed`, `actuate_refused`;
  - `GpuCardStateV1.swap_state` adds `fault`;
  - new `GpuLeaseRefV1 {lease_id, generation, role, holder}`.

  These are all `extra="forbid"`, so consumers (pool, gateway, Hub panel) deploy before producers (the
  consumer-first rule).
- New constant `GPU_POOL_ACTUATE_RESULT_CHANNEL`. `channels.yaml` + `registry.py` entries for the result channel.
- `orion/llm/resource_lease.py`: a new header `X-Orion-Gpu-Lease` carrying `GpuLeaseRefV1`. `options.gpu_lease`
  on bus calls. `X-Orion-Resource-Lease` and `ResourceLeaseV1` are deleted in PR 4.6 across every importer
  (`thought.py`, `harness_finalize.py`, `durable_run.py`, `orion/harness/finalize.py`, cortex-exec
  `executor.py`/`self_study.py`, `orion/hub/turn_orchestrator.py`, Hub `curiosity_investigation.py`).
- `config/gpu_pool.yaml` + `orion/gpu_pool/config.py`: the keys above.
- Durable-runs HTTP: `/elastic/status`, `/elastic/target`, `/leases/validate` and `/admission` are deleted in
  PR 4.5. `/capacity/*` stays until stage 5.
- Postgres: nothing is dropped in stage 4.
  - `durable_resource_demands`, `durable_resource_leases` and `durable_elastic_slot` are frozen. `capacity.py`
    still joins the first two for world and diffusion permits, and they are dropped in stage 5.
  - The PR 4.5 migration step marks the pending demands `withdrawn` with `reason=migrated_to_gpu_pool` (13
    rows; snapshot to `/tmp/gpu-pool-stage4-migrate/` first; a production write, so it needs Juniper's go).

## Env/config changes

- Pool: `GPU_POOL_ACTUATE_ROLES` (empty by default = observe for swaps; `agent-gpu2` at cutover). Cabinet and
  thought URLs for guards: `GPU_POOL_CABINET_URL`, `GPU_POOL_VISUAL_ACTIVITY_URL` (the latter is deleted in stage 5).
- Controller: `GPU2_AUTHORITY=durable|pool` (transitional; the `durable` branch is deleted in 4.6),
  `GPU_POOL_ACTUATOR_NAME=circe`, `ORION_BUS_URL` (it already runs a heartbeat chassis).
- Deleted in 4.5/4.6:
  - `DURABLE_RUNS_ELASTIC_*`, `DURABLE_RUNS_ADMISSION_*` / `WIDENING_*` (the broker keys);
  - `GPU2_AUTHORITY_URL`, `LLM_GATEWAY_LEASE_VALIDATION_*`, `HUB_CURIOSITY_LEASE_VALIDATION_URL`,
    `HUB_CURIOSITY_ELASTIC_ACTIVATION_ENABLED`.
  - `DURABLE_RUNS_CAPACITY_*` stays until stage 5.
- Every `.env_example` change is synced with `scripts/sync_local_env_from_example.py` in its PR.

## Files likely to touch

- `config/gpu_pool.yaml`, `orion/gpu_pool/{config,scheduler,client,lease_graph}.py`, `orion/schemas/gpu_pool.py`,
  `orion/schemas/registry.py`, `orion/bus/channels.yaml`, `scripts/check_gpu_pool_config.py`
- `services/orion-gpu-pool/app/{runtime,store,settings}.py` + migration (the `gpu_pool_leases` index on
  `parent_lease_id`, and the persisted swap intent/generation per card), tests, evals
- `services/orion-gpu-lane-controller/app/{main,gpu2,settings}.py`, new `app/actuator_bus.py`, README, `.env_example`
- `services/orion-llm-gateway/app/{main,resource_lease,pool_placement,openai_passthrough,anthropic_passthrough,passthrough_proxy}.py`
- `services/orion-durable-runs/app/{admitted_graph,admission_runtime,main,settings,graph}.py`; delete
  `elastic_runtime.py`; `orion/durable_admission/{broker,policy,elastic}.py` deleted (with `store.py` trimmed to
  what `capacity.py` needs)
- `orion/harness/{fcc_motor,finalize}.py`, `orion/llm/resource_lease.py`, `orion/schemas/{durable_run,resource_admission,thought,harness_finalize}.py`
- `services/orion-cortex-exec/app/{executor,self_study}.py`, `orion/hub/turn_orchestrator.py`,
  `services/orion-hub/scripts/{curiosity_investigation,main}.py`, `services/orion-hub/app/settings.py`
- `services/orion-field-digester/app/store.py`, `orion/field/queue_contention.py` docs
- `.github/workflows/orion-durable-runs-tests.yml`, `gpu2-elastic-tests.yml` (their elastic/admission evals are deleted
  with the code), `config/metrics/metric_definitions.lock.json` (re-lock)

## Non-goals

- The generic compose actuator for arbitrary roles, experiment-seat actuation, world/diffusion leases, and
  deleting `capacity.py` and the durable tables (all stage 5).
- The GPU1 affect flip (stays as is; deleted by the parent spec's stage 5).
- Gateway per-call telemetry and grammar reducers (stage 6).
- Choosing which *model* a role runs. `profile` exists in the contract, but stage 4 never sets it.
- Fixing why 13 runs pile up. Stage 4 makes the queue visible to the one decider that can open a second seat.
  Capacity policy is Missing question 2.

## Corrections from building 4.1 (PR #2349)

These override the text elsewhere in this document.

1. **The gpu4 example needs one more line.** Add `fast2` to `classes.metacog.roles`: it's owned by `[metacog, fast]` but only class `fast` listed it, so the validator rejects the example as written.
2. **The fields that point a call at its run's hold are `hold_lease_id` / `hold_generation`, not `parent_*`.** `parent_lease_id` already means "replayed from" in the pool's stored request and projection. 4.3 records "child of hold" in its own projection column (`hold_lease_id`) and never reuses `parent_lease_id`.
3. **4.1 deploys the pool and sql-writer**, not "pool, gateway, Hub". sql-writer validates every `GpuPoolEventV1`, so it must accept the new events before a 4.3 pool emits them.
4. **The pool's lease handler used to treat any unknown verb as cancel.** 4.1 makes `attach` and `status` return `verb_not_supported` until 4.3 builds them.
5. **`cuda_env` only works if compose lets the device be set from outside.** `atlas-agent-burst` hard-codes `CUDA_VISIBLE_DEVICES_OVERRIDE=2`. Stage 5 must make it `${...}`, or the gate must require it. The 4.2 bridge is unaffected.
6. **The Hub must show `swap_state`, including `fault`**, in 4.3. Acceptance check 7 depends on it.

## Juniper's answers (2026-09-25)

1. **Gaps are shared.** While a run holds a card, higher-priority single calls may use it between the run's own calls. The run keeps its card and model for the whole run; at worst it waits one call's length.
2. **No hold cap in stage 4.** Recall and priority still apply. Decide a cap from a week of pool data.

## Acceptance checks (live, observable)

1. After PR 4.5:
   - zero new rows in `durable_resource_demands` / `durable_resource_leases`;
   - each accepted run has one `gpu_pool_leases` row, `kind='hold', holder='durable-runs:<run_id>'`.
2. **No self-deadlock:** zero `work_class='agent'` request leases with `parent_lease_id IS NULL` whose
   `turn_correlation_id` belongs to a run holding a hold (SQL join, run for 24h). FCC calls show as children of
   their run's hold.
3. **Interleave:** during a run's tool phase, a cortex-exec system `agent` call is granted on `agent` (pool
   event `granted`, `waited_ms` < one inference), and the run's next child waits at most one inference.
4. **gpu2 via the pool:**
   - with a hold queued ≥ 1200s and guards clear, the pool emits `swap_started` then `swapped`;
   - the controller logs `gpu2_transition operation=<pool action_id>`;
   - discovery marks `agent-gpu2` confirmed;
   - the queued hold is granted `role=agent-gpu2`.

   Nothing on athena calls `:8090/v1/gpu-slots/activate` (controller access log).
5. **Guards:**
   - an unload followed by demand within 600s gives `swap_requested reason=min_residency`;
   - a hot cabinet gives `reason=guard:thermal`;
   - an overdue visual baseline gives `reason=guard:visual_baseline`.
6. **Restart safety:**
   - kill durable-runs mid-hold → `expired` after 90s → re-queued; on restart the run resumes with the same
     `lease_id`;
   - kill the pool mid-load → on restart it sends `status`, adopts the result, and never issues a second
     transition.
7. **Failed load:** force a readiness timeout → `swap_failed restored=true`, diffusion running, cooldown set; with
   `restored=false` the card shows `fault` in the Hub panel.
8. **Field:** `queue_contention` `durable_demand_pending` equals the count of queued/backlogged holds;
   `gpu_pool_waiting` excludes holds.
9. **Hub:** the curiosity run view shows waiting → granted (with the role) → released for a post-cutover run, from
   the same tables.

## Recommended next patch: stage 4 as ordered PRs

Each PR is independently deployable, and the order matters.

| PR | what | deploy | behavior change |
| --- | --- | --- | --- |
| **4.1 contracts + config** | schemas above (additive), result channel, registry, YAML keys (`launch`, `index`, `actuators`, per-seat `after_wait_sec`/`guards`, `hold_clawback_grace_sec`, `swap_min_residency_sec`), validator + CI gate. Nothing produces the new fields. | pool, gateway, Hub (consumers first) | none |
| **4.2 circe actuator bridge** | controller bus consumer; pool-generation fence behind `GPU2_AUTHORITY` (default `durable`); `status` action; persisted last generation | circe controller | none (nobody sends) |
| **4.3 pool: holds + actuation engine** | hold placement, `attach` children, interleave, hold recall/grace, `status` verb; swap state machine, guards, min residency, actuate send/ack/timeout/reconcile, `fault`; tests + eval extended with 35-min/7.8h holds and a failed load | pool, `GPU_POOL_ACTUATE_ROLES=` empty | none for swaps; holds unused |
| **4.4 consumers accept the new lease ref** | gateway honours `X-Orion-Gpu-Lease` / `options.gpu_lease` via `attach` (old `LeaseGuard` path still live); FCC/harness/cortex-exec/Hub pass either ref through; Hub Door-A validates via pool `status` when given a pool ref; field-digester union source + `kind='request'` filter | gateway, cortex-exec, Hub, field-digester | none (no pool refs issued yet) |
| **4.5 cutover** | durable-runs `resource_request`/`resource_wait`/`execute` on pool holds; broker, policy, elastic, `elastic_runtime.py`, `/elastic/*`, `/leases/validate`, `/admission` deleted; stop emitting suppression/elastic events. **Runbook, in order:** (1) wait until `durable_resource_leases` has 0 active rows; (2) durable-runs `DURABLE_RUNS_ELASTIC_SHADOW=true` + restart (old decider stops, gpu2 frozen); (3) circe controller `GPU2_AUTHORITY=pool` + restart; (4) pool `GPU_POOL_ACTUATE_ROLES=agent-gpu2` + restart (adopts observed seat state); (5) deploy durable-runs 4.5; (6) with Juniper's go, snapshot then withdraw the frozen pending demands, since their runs re-register through the pool on resume | durable-runs, then env flips | **yes: the pool is the only decider** |
| **4.6 cleanup** | delete `ResourceLeaseV1`/`X-Orion-Resource-Lease`/`LeaseGuard` old path across importers; the controller `durable` authority branch and gpu2 activate route; the field-digester legacy half; deprecated `ResourceRequirementV1` fields; env keys listed above; elastic/admission evals and workflows | all touched | none |

Stage 5 then generalises the actuator from `launch:`, moves world and diffusion onto leases, and deletes
`capacity.py`, the `visual_baseline` guard, the bridge verbs and the frozen tables.
