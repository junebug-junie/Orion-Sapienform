# GPU pool — one lease queue for every GPU on circe

Status: DESIGN v2 (awaiting Juniper sign-off before build)
Date: 2026-09-24
Supersedes: `orion/durable_admission/` (broker, policy, capacity, elastic, capacity_client), gateway
`capacity.py` / `upstream_admission.py` / `priority_admission.py` / `lane_gate.py` /
`admission_ledger.py`, PR #2317 (`lane_contention.py`), the #2246 stance fallback chain, and
`orion-gpu-lane-controller`'s own decision-making.

## Arsonist summary

Right now nine different pieces of code decide which GPU a piece of work lands on, and four of
them have their own idea of priority. None of them sees the whole machine, and none of them
knows which model is actually loaded where. Each slice since 2026-09-13 added another decider.

This replaces all of them with **one service, `orion-gpu-pool`**. Every GPU user asks it for a
**lease** and gets a grant, a place in line, or a durable backlog entry that replays when
capacity returns. It is the only thing that knows what is loaded on each card, and the only
thing allowed to decide who goes first.

- **Rules** live in one YAML file, `config/gpu_pool.yaml`. It speaks in *roles* (chat, agent,
  metacog, …) and *cards*, never in model names.
- **Which model sits in a role** is discovered live: each llama.cpp worker announces its
  `llm_profiles.yaml` profile on the bus, and the pool confirms it against the server's own
  `/props`.
- **Every lease is a checkpointed LangGraph run**, with retry, dead-letter and operator replay
  built into the graph.
- **Everything the pool does is on the bus**, so transport metrics see it.

## Decisions made (Juniper, 2026-09-24)

| Question | Answer |
| --- | --- |
| GPU0 borrowing | **Lent only.** Others use GPU0 only while Juniper's Hub button has it lent. Chat claws it back whether or not it is lent. |
| Placement | **New service** `orion-gpu-pool` on athena. The circe actuator loads and unloads models. Durable-runs becomes a client. |
| Affect | **Out of scope** for the first cut; it becomes a YAML tenant later. |
| Clawback grace | **The borrower finishes its current request**, capped (default 60s). After that it gets no new work. |
| GPU2 | Defaults to world-model and diffusion; it is leased out as a second agent model when needed. |
| Config | YAML-configured so new cards and services need no code. |
| Engine | **LangGraph** for the lease lifecycle. |
| Transport | **As much behind the bus as possible.** |
| Naming | **No model names in the pool config.** Models come and go; `config/llm_profiles.yaml` is the only model catalog. |
| Multi-card | Must handle a model spanning several cards (e.g. the DeepSeek-V4.1-Flash soak on 4×V100, `docker-compose.dsv41.yml`, profile `deepseek-v41-flash-mxfp4-engram-4xv100-32gb-circe-test`). |
| Failure | Everything that hits an unavailable lane degrades gracefully into backlog and replays when capacity returns. |

## Pre-launch: #2317 pollution (found live 2026-09-24)

#2317 was never merged, but it **is running in production**:

- athena's `orion-llm-gateway` container (created 2026-09-24 16:20 UTC) was built from
  `/mnt/scripts/Orion-Sapienform-gpu-lane-swap-burst`, the #2317 worktree. The image contains
  `/app/app/lane_contention.py`, and the container env has `LLM_LANE_CONTENTION_FALLBACK_ENABLED
  =true` plus both JSON keys.
- The primary checkout's `services/orion-llm-gateway/.env` lines 146–148 carry the same three
  keys.
- circe is clean of #2317. However, eleven other prod containers across both hosts also run from
  worktrees rather than main (see the PR report). This is the same "a worktree deploy pins a
  worktree as production" failure recorded before.

Rollback (production action, needs Juniper's explicit go): remove the three keys from the
primary `.env`, then rebuild `orion-llm-gateway` from a clean `origin/main` worktree via
`scripts/safe_docker_build.sh`. Verify: no `lane_contention.py` in the container, no
`LLM_LANE_*` in its env, `/health` ok, and a metacog call served.

## The machine (live, 2026-09-24)

| card | hardware | what runs there today (discovered, not configured) |
| --- | --- | --- |
| gpu0 | V100-PCIE-32GB | chat worker, profile `qwen36-35b-a3b-udq5km-2xv100-32gb-deep-cognition`, :8011 |
| gpu1 | V100-SXM2-32GB | agent worker, profile `qwen3.8-27b-udq4kxl-v100-32gb-circe-agent-flex`, :8015 |
| gpu2 | PG500-216 (V100) | world-model (~1GB, :6613), diffusion-host (:8014, ~24GB when loaded) |
| gpu3 | V100-PCIE-32GB | metacog worker (`qwen3-8b-q5km-…-metacog-16k`, :8012) and fast worker (`qwen3-8b-q4km-…-balanced`, :8013) |

## The rules, stated once

1. **Roles, not models.** A *role* is a named place work can run: `chat`, `agent`, `metacog`,
   `fast`, `world`, `diffusion`, and `experiment` for a multi-card soak. The YAML says which
   cards a role lives on and who may borrow it. Which model fills the role right now is
   discovered live (see "The discovery bridge").
2. **Every lease names a work class.** A class lists the roles able to serve it, in preference
   order, plus requirements checked against the *discovered* model: minimum context per slot,
   vision, and so on. A 30k-token agent prompt is never placed on a role whose loaded model has
   a 4k slot, whatever the preference list says.
3. **Home first.** A request goes to its first role when that role has a free slot.
4. **Borrowing** goes only in the directions the YAML allows:
   - `metacog` ↔ `fast` share freely on gpu3.
   - `metacog` and `fast` may spill up to `agent` (gpu1), to gpu2's agent seat when it is
     loaded, and to `chat` when gpu0 is lent.
   - Nothing from the big roles spills down to gpu3, because those models are too small. This
     is structural: no big-role class lists `metacog` or `fast`.
   - `agent` may use gpu2's agent seat after a swap, and `chat` when gpu0 is lent.
   - `chat` never leaves gpu0.
5. **The owner always wins on its own card.** When an owner-class request is waiting and every
   slot on its role is held by borrowers, the pool **recalls** the newest borrower lease. The
   borrower finishes its current request (grace cap, default 60s) and gets no new work. If it
   overruns it is aborted and retried. A borrower is never granted a slot while an owner waits.
6. **Queue order:** owner before borrower, then `priority` (`interactive > system >
   background`), then oldest first. That is the whole ordering.
7. **Seats that need a swap.** A role may have a *swap seat*: a model the actuator can load onto
   cards that normally host something else. The pool loads a swap seat only when the card's
   VRAM, after evicting the tenants the YAML names, fits the discovered model's footprint.
   - gpu2's agent seat evicts diffusion only, so world-model keeps running.
   - The owner reclaiming a card reverses the swap: recall, grace, unload, restore.
   - A cooldown (default 600s) stops flapping.
8. **Multi-card seats.** A seat may span several cards (the `experiment` seat spans
   gpu0–gpu3). Activating it is an **operator lease on every card it spans**. The pool:
   - stops granting new work on those cards;
   - recalls every lease on them, with grace;
   - has the actuator stop the resident workers and start the experiment;
   - holds it until the operator releases it or its max duration passes;
   - then restores every resident.

   While it is active, work whose roles all live on those cards goes to **backlog** under rule
   10. Nothing fails hard.
9. **GPU0 lend** is a card flag only an operator can set. While it is off, only `chat` may be
   granted gpu0. Turning it off recalls every borrower on gpu0.
10. **Graceful degradation, never silent failure.** When no allowed role can serve a lease, the
    class's YAML `on_unavailable` policy decides what happens:
    - `wait`: stay queued until the caller's deadline, then return `unavailable` with a reason.
      Used for interactive chat, so a person is told, not left hanging.
    - `backlog`: park the lease durably. It replays automatically when an allowed role becomes
      healthy, and its result is delivered on the caller's reply channel or wakes its durable
      run. Used for background thinking, world ticks and curiosity. There is a max age, after
      which it is dead-lettered.
    - `fail`: return `unavailable` immediately. For callers that have their own fallback.

    A caller always gets one of `granted`, `queued`, `backlogged {backlog_id}` or `unavailable
    {reason}`. It never gets a timeout with no explanation.
11. **Health gates grants.** A role whose worker is down, unannounced, or whose announcement
    disagrees with `/props` gets no grants.

## `config/gpu_pool.yaml`

```yaml
version: 1
host: circe
defaults:
  clawback_grace_sec: 60
  request_lease_ttl_sec: 30
  hold_lease_ttl_sec: 90
  swap_cooldown_sec: 600
  retry: {max_attempts: 3, base_sec: 5, max_sec: 300}   # timeouts + upstream failures
  backlog_max_age_sec: 86400

priorities: [interactive, system, background]

cards:
  gpu0: {vram_gb: 32, lendable: true}
  gpu1: {vram_gb: 32}
  gpu2: {vram_gb: 32}
  gpu3: {vram_gb: 32}

roles:                               # where work can run; NO model names
  chat:      {kind: llm, cards: [gpu0], owner: chat, port: 8011}
  agent:     {kind: llm, cards: [gpu1], owner: agent, port: 8015}
  agent-gpu2: {kind: llm, cards: [gpu2], owner: [world, diffusion], port: 8016,
               swap: {evicts: [diffusion], load: gpu2/agent, unload: gpu2/restore}}
  metacog:   {kind: llm, cards: [gpu3], owner: [metacog, fast], port: 8012}
  fast:      {kind: llm, cards: [gpu3], owner: [metacog, fast], port: 8013}
  world:     {kind: service, cards: [gpu2], owner: world, port: 6613, slots: 2, vram_gb: 1}
  diffusion: {kind: service, cards: [gpu2], owner: diffusion, port: 8014, slots: 1, vram_gb: 24}
  experiment: {kind: llm, cards: [gpu0, gpu1, gpu2, gpu3], port: 8099, operator_only: true,
               swap: {evicts: all, load: circe/experiment, unload: circe/restore},
               max_hold_sec: 14400}

classes:                             # what a lease asks for
  chat:      {roles: [chat], on_unavailable: wait}
  agent:     {roles: [agent, agent-gpu2, chat], on_unavailable: backlog}
  metacog:   {roles: [metacog, fast, agent, agent-gpu2, chat], on_unavailable: backlog}
  fast:      {roles: [fast, metacog, agent, agent-gpu2, chat], on_unavailable: wait}
  world:     {roles: [world], on_unavailable: backlog}
  diffusion: {roles: [diffusion], on_unavailable: backlog}
  experiment: {roles: [experiment], on_unavailable: fail}

routes:                              # what gateway callers say -> class (+ priority)
  chat: chat
  harness: chat
  agent: agent
  metacog: {class: metacog, priority: system}
  metacog_background: {class: metacog, priority: background}
  quick: fast
  quick_background: {class: fast, priority: background}
```

Validation runs at boot and in CI. It checks that every role's cards exist, every class resolves
to known roles, and swap evictions name real co-resident roles. It also checks the live side:
every announced profile exists in `llm_profiles.yaml`, and resident VRAM from those profiles
fits the card. Model-dependent checks (VRAM, context per slot, vision) are evaluated against
the **discovered** profile, so swapping a model file never needs a pool-config edit.

The gateway reads `routes:` and the discovered role → URL map. Its `.env` loses
`LLM_GATEWAY_ROUTE_TABLE_JSON`, `LLM_ROUTE_*_SERVED_BY`, `LLM_LANE_*`,
`LLM_GATEWAY_UPSTREAM_MAX_INFLIGHT`, `LLM_GATEWAY_CAPACITY_*`, `LLM_GATEWAY_BACKGROUND_*` and
`LLM_ALLOW_BACKGROUND_TO_CHAT_FALLBACK`.

## The discovery bridge: what is actually loaded in each role

This was never solved before. The facts:
- every worker container already has `LLM_PROFILE_NAME` in its environment (e.g. the agent
  worker has `qwen3.8-27b-udq4kxl-v100-32gb-circe-agent-flex`);
- `orion-llamacpp-host` already connects to the bus and publishes to `orion:system:health`
  (`app/main.py:613-649`);
- llama.cpp's own `GET /props` reports the `model_path` actually loaded, the per-slot `n_ctx`,
  and `total_slots`.

Nobody put these three together.

1. **Announce.** `orion-llamacpp-host` publishes `LlmWorkerAnnounceV1` on
   `orion:llm:worker:announce` at boot and on each heartbeat: `{host, role (from a new
   LLM_ROLE env, defaulting to the compose service name), profile_name, port, physical_gpus
   (resolved from CUDA_VISIBLE_DEVICES_OVERRIDE, UUID-stable), pid, started_at}`.
2. **Confirm.** The pool's discovery poller hits each announced worker's `/props` every 15s. It
   resolves `profile_name` in `llm_profiles.yaml` and checks that the profile's model file
   equals the basename of `/props.model_path`. From `/props` it takes the truth for slots, ctx
   per slot and vision.
3. **Map.** The result is the live role table in `orion:gpu_pool:state`: role → profile →
   model file → cards → slots → ctx/slot → vision → health → `confirmed | mismatch | silent`.
   - `mismatch` (announced profile ≠ loaded file) and `silent` (no announcement but the port
     answers) are shown in red on the panel and get **no grants**.
   - Any port answering on circe that no role claims is listed as an *unclaimed server*: the
     "what the fuck is out there" row.

Known live drift this surfaces: profiles record `device_ids: [0]` for the 8B workers, while they
actually run on gpu3 via `CUDA_VISIBLE_DEVICES_OVERRIDE`. The announcement carries the real
devices, so the profile's `device_ids` is never trusted for placement.

## The lease graph (LangGraph, `thread_id = lease_id`)

Checkpointed by `AsyncPostgresSaver` on a connection pool, the same pattern as durable-runs.

```text
admit ─► place ─┬─ slot free ────────────────────────────────► granted ─► hold
                ├─ allowed role busy ─► queued ──(interrupt)──► granted
                ├─ no allowed role healthy ─► on_unavailable:
                │     wait ─► queued (until deadline) ─► unavailable
                │     backlog ─► backlogged ──(interrupt; woken on role healthy)──► place
                │     fail ─► unavailable
                └─ deadline passed while queued ─► unavailable(reason=deadline)

hold ──(interrupt; heartbeats/release resume it)──┬─► release(outcome=ok) ─► released
                                                  ├─► release(outcome=upstream_error|timeout)
                                                  │        ─► retry_wait ─► place   (attempt<max)
                                                  │        ─► dead_letter           (attempt=max)
                                                  ├─► recalled ─► grace ─┬─► released (finished)
                                                  │                      └─► aborted ─► retry_wait
                                                  └─► heartbeat lost ─► expired ─► retry_wait

dead_letter ──(operator replay)──► place   (new attempt series, same lease_id, audit kept)
backfill: operator selects a set of past leases (by class, holder, time range, outcome)
          ─► each spawns a replay child thread (parent_lease_id) ─► place
```

- **Timeouts and failures retry** with exponential backoff (YAML `retry`). The attempt number,
  each failure reason and each placement are kept in the thread's history.
- **Dead letter** is a terminal-but-replayable state. It is never deleted, and it shows in the
  panel with its whole history.
- **Operator replay and backfill** go over the bus (`orion:gpu_pool:control:request`, verbs
  `replay` and `backfill`). A backfill creates child threads linked to their originals, so a
  replayed day of world ticks is traceable back to what it replays.
- **What a lease replays.** A request lease carries the caller's request envelope reference (the
  gateway's request body is stored in the thread state, size-capped). So a backlogged or
  replayed LLM call can be re-dispatched by the pool through the gateway without its original
  caller still waiting. Its result goes to the reply channel it was issued with, and is
  persisted if nobody is listening. A durable-run lease replays by resuming its run.
- The **scheduler** stays one pure function, `schedule(config, discovered_roles, cards, queue,
  leases, now) -> [grant | recall | abort | swap | wake_backlog]`. A single-writer loop
  (Postgres advisory lock, 1s tick plus a wake on events) runs it and resumes threads with
  `Command(resume=...)`.
- `gpu_pool_leases` is a small **projection** of thread state (one row per lease, fenced with
  `FOR UPDATE`) for fast queries. The checkpoint remains the source of history and replay.
- Cost: about 4–6 checkpoint writes per request lease. Stage 1's eval measures throughput at
  realistic metacog/fast volume before the gateway depends on it.

## Schemas and channels (all registered in `channels.yaml` + `registry.py`)

| channel | schema | direction |
| --- | --- | --- |
| `orion:gpu_pool:lease:request` | `GpuLeaseRequestV1` (verb acquire / heartbeat / release{outcome}) → `GpuLeaseReplyV1` (granted / queued / backlogged / unavailable) | callers → pool (RPC) |
| `orion:gpu_pool:event` | `GpuPoolEventV1` (admitted, queued, granted, backlogged, recalled, aborted, expired, retried, dead_lettered, replayed, released, swapped, lent, discovery_mismatch) | pool → everyone |
| `orion:gpu_pool:state` / `:state:request` | `GpuPoolStateV1` (config digest, cards, discovered roles, leases, queue, backlog, recalls, per-class rolling stats) | pool → panel / field / cortex-exec |
| `orion:gpu_pool:control:request` | `GpuPoolControlV1` (lend, unlend, activate_experiment, release_experiment, replay, backfill, pause_class) | operator (Hub) → pool (RPC, operator token) |
| `orion:gpu_pool:actuate:request` / `:result` | `GpuActuateV1` / `GpuActuateResultV1` | pool → circe actuator |
| `orion:llm:worker:announce` | `LlmWorkerAnnounceV1` | llamacpp-host → pool |

Client library `orion/gpu_pool/client.py`: `async with gpu_lease(work_class, holder, priority,
deadline, replay_payload=None) as lease:`. It handles every reply kind and raises typed
`LeaseUnavailable(reason)` or `LeaseRecalled`. It is the only client.

## How each caller changes

| caller | after |
| --- | --- |
| llm-gateway | maps route → class, takes a request lease per dispatch, and calls the URL of the role granted. It holds the lease for the call and releases it with `outcome`. It honours durable-run hold leases. `backlogged` is returned to the caller as a typed reply (not an error string). |
| durable-runs | `resource_request` takes a hold lease; `resource_wait` interrupts until `granted`; recall or failure goes through the pool's retry. Its own broker and elastic runtime are deleted. |
| world-model, visual chain (diffusion) | `gpu_lease(class="world" / "diffusion")`; ticks that hit `backlogged` are replayed by the pool. |
| Hub | lend button, experiment activate/release, replay and backfill are `control` RPCs; the new operator panel (below); the hold-and-email path is deleted. |
| gpu-lane-controller (circe) | becomes the bus actuator for `gpu2/*` and `circe/experiment|restore`. The GPU1 affect flip is deleted. |
| orion-llamacpp-host | announces its role, profile and devices. |

## Hub operator panel ("GPU pool" tab)

Built on `orion:gpu_pool:state` and `:event` live over Hub's websocket, plus sql-writer
history.

1. **Config.** `gpu_pool.yaml` rendered as a picture, not a text dump:
   - the four cards as columns, with the roles on each and swap seats drawn dashed;
   - arrows for who may borrow what, owners marked;
   - each role annotated with its *discovered* profile, model file, slots and ctx/slot;
   - confirm/mismatch/silent badges, and unclaimed servers listed.

   The raw YAML is one click away.
2. **Traffic, three zoom levels, live or historical** (time-range picker; live is the default):
   - *aggregate*: per card and per role, busy slots over time, queue depth, backlog depth,
     p50/p95 wait, success/failure rate, recalls;
   - *semi-aggregate*: the same broken down by class, holder service and priority;
   - *detail*: a streaming table of individual leases (holder, class, role served, waited,
     ran, outcome, attempt, correlation_id), filterable, with each row linking to the walker.
3. **Graph walker.** Pick any lease, live or historical. It shows the lease graph above with the
   path that lease actually took, each step's timestamps and durations, retries, the recall
   that hit it, and the `correlation_id` link to the turn and trace it belonged to. The data
   comes from LangGraph checkpoint history. Buttons: replay (dead-lettered or failed), cancel
   (queued or backlogged).
4. **Controls.** GPU0 lend on/off, experiment activate/release, pause a class, and backfill
   (choose class, holder, time range and outcome, preview the count, confirm).

## Transport and telemetry

**Today:** cortex-orch → cortex-exec → gateway are bus RPC hops. Each `rpc_request()` records
per-hop latency and timeouts (`orion/core/bus/async_service.py:509-634`), and
`RpcHealthPublisher` drains that to `orion:rpc_health:snapshot`. From there signal-gateway
builds `OrionSignalV1` organs, equilibrium runs its transport metacog gate, and timeouts
become `rpc_transport_timeout` grammar. bus-mirror turns every shared `correlation_id` into
causal-hop edges, which feed `bus_synaptic_prediction_error` → `node:substrate.bus_synaptic`.

**The pool plugs in with the live helpers:**
1. Lease calls are bus RPC, carrying the caller's `correlation_id`. Per-caller rpc_health,
   bus-mirror causal edges and `bus_synaptic_prediction_error` pick up the pool hop
   automatically. The RPC answers immediately (it never waits in line), so an RPC timeout means
   the pool is unreachable, not busy.
2. The pool runs `RpcHealthPublisher` (service `gpu-pool`, hop `gpu_pool:<class>#grant`,
   latency = queue wait, deadline miss = timeout). signal-gateway turns that into an
   `rpc_health_gpu_pool` organ with no registry change.
3. Lease lifecycle facts go out as `GrammarEventV1` (`source_service="orion-gpu-pool"`, trace
   prefix `gpu_pool.lease:`). They are persisted via sql-writer. **A reducer and field wiring
   only come after the metric quality gate** (stage 6).
4. Lease history persists bus → sql-writer (`gpu_pool_events`). The only direct Postgres writer
   is the scheduler's fenced projection plus the checkpoint (a stated exception).
5. **Gateway per-call telemetry is built last** (stage 6, after every cutover, per Juniper). It
   covers rpc_health hops `llm:<role>#call` for model latency and failure, plus refusal grammar.
   Until then, queue wait versus model time is visible only in the pool's own telemetry and
   panel.

## Downstream readers that move with each deletion

Each of these is fail-open, so deleting its source without moving it reads as "calm" or empty,
not as an error.

| reader | reads today | moves to |
| --- | --- | --- |
| field-digester `queue_contention.py` → `FieldStateV1.queue_contention_score` (**locked metric**, curiosity hire decisions read it) | `durable_resource_demands` count + gateway `/admission` waiting | pool queue + backlog depth. **Needs Juniper's decision at stage 2**: re-point like-for-like under the metric gate, or retire it for a pool-native successor. |
| cortex-exec `admission_cue.py` (Orion's "my background thinking was made to wait" cue) | gateway `/admission` ledger | pool events for `priority=background`, keeping its four distinct states |
| Hub `runtime_activity` | gateway `/admission` | `orion:gpu_pool:state` |
| curiosity `run_story`, `hire_progress`, `curiosity_run_store`, atlas | `durable_admission_runs` / `durable_resource_events` | pool events via sql-writer, keyed by holder = run_id, same event names where the meaning is unchanged |
| `turn_orchestrator`, gateway passthroughs | lane gate, lease fencing | pool lease; hold-while-lent is deleted |
| `inner_state_registry`, `field_state` docs | queue_contention sources | updated |
| *(pending: second sweep in progress; results are appended before sign-off)* | | |

## Delete list (in the stage that replaces each piece)

- `orion/durable_admission/*`, durable-runs `admission_runtime.py` / `elastic_runtime.py`, and
  their tables (dropped only after cutover is verified).
- Gateway `capacity.py`, `upstream_admission.py`, `priority_admission.py`, `lane_gate.py`,
  `admission_ledger.py`, `/admission`, plus `BURST_LLM_ROUTES` / `OPERATOR_GATED_LLM_ROUTES` /
  `CHAT_BURST_LENDS_ROUTE`.
- Env keys: `LLM_LANE_*`, `LLM_GATEWAY_{ROUTE_TABLE_JSON,UPSTREAM_MAX_INFLIGHT,CAPACITY_*,
  BACKGROUND_*}`, `DURABLE_RUNS_{ADMISSION,CAPACITY,WIDENING,ELASTIC}_*`, `WM_GPU2_CAPACITY_*`,
  `ORION_VISUAL_CHAIN_GPU2_CAPACITY_*`, `HUB_CURIOSITY_ELASTIC_ACTIVATION_ENABLED`.
- Hub `chat_lane_lend.py` hold-and-email; the #2246 stance fallback chain; the GPU1 affect flip.
- PR #2317 closed unmerged; stray `ATLAS_AGENT_HOST_PORT=8014` fixed.

## Non-goals

- Affect as a tenant (a later YAML entry).
- athena's GPUs. The YAML `host` field leaves room, but this cut is circe only.
- Splitting one request across cards (a multi-card *model* is supported; that is llama.cpp's own
  split).
- Auto re-quantization or context resizing.

## Acceptance checks

1. **Config and discovery:**
   - YAML validation fails on unknown roles, cards or evictions.
   - Discovery marks a worker whose `/props.model_path` ≠ its profile's file as `mismatch` and
     grants it nothing.
   - An unannounced answering port is listed as unclaimed.
2. **Scheduler tests** (fake clock):
   - home-first;
   - metacog↔fast;
   - metacog spill to agent;
   - nothing spills to gpu3;
   - ctx requirement blocks a large prompt from a small-slot role;
   - gpu0 refused unless lent;
   - chat recalls a gpu0 borrower, aborted after grace;
   - owner before borrower, then priority, then FIFO;
   - idempotent `request_id`;
   - lost heartbeat → retry;
   - upstream failure → retry with backoff → dead letter at max;
   - operator replay of a dead letter;
   - backfill spawns linked children;
   - `backlog` wakes when a role turns healthy;
   - `wait` returns `unavailable` at the deadline;
   - gpu2 swap keeps world and evicts diffusion;
   - diffusion reclaims gpu2;
   - cooldown;
   - experiment activation recalls everything on gpu0–3, backlogs background, returns
     `unavailable` to chat with a reason, and restores all residents on release;
   - two schedulers → one grant.
3. **Graph tests:** a pool restart mid-queue, mid-hold and mid-backlog resumes every thread with
   the same `request_id`, priority and position.
4. **Eval:** a synthetic day of mixed traffic, including one worker outage and one experiment
   window. It reports p50/p95 wait per class, recalls, retries, dead letters, backlog replays,
   owner-starvation seconds (target 0 beyond grace), leases lost (target 0), and checkpoint
   writes/sec against capacity.
5. **Live:**
   - the panel shows all four cards with *discovered* profiles confirmed;
   - a metacog burst spills fast → agent;
   - lend on/off with a chat recall visible;
   - stop the metacog worker, and background metacog backlogs, then replays when it returns;
   - the walker shows one real lease's path with the timings of each step.
6. **Enforcement gate:** CI fails on any direct reference to a circe model or service port
   outside the pool, the gateway dispatch and the actuator.

## Build order (each stage stops for Juniper's review)

0. **Pre-launch:** roll back the #2317 pollution (with Juniper's go); close #2317.
1. **Pool core:** YAML + validation, discovery bridge (llamacpp-host announce plus the `/props`
   poller), schemas and channels, migration, lease graph, scheduler, backlog and replay, client
   library, tests, eval. Deployed in observe mode: it discovers, answers and shows, but nobody
   depends on it.
2. **Operator panel** in Hub: config picture, traffic at three zoom levels live and historical,
   graph walker, controls.
3. **Gateway cutover:** all LLM traffic leases; the lend button moves; the gateway deciders are
   deleted; readers move (the `queue_contention` decision happens here).
4. **Durable-runs cutover:** hold leases; `orion/durable_admission` is deleted.
5. **gpu2, world, diffusion and experiment seat:** the actuator becomes the swap executor; the
   elastic code is deleted.
6. **Gateway per-call telemetry and grammar reducers** (after the metric gate), then
   **lockdown:** port gate, old env keys and old tables removed.
