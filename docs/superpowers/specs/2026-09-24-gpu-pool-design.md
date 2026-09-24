# GPU pool — one lease queue for every GPU on circe

Status: DESIGN (awaiting Juniper sign-off before build)
Date: 2026-09-24
Supersedes: `orion/durable_admission/` (broker, policy, capacity, elastic, capacity_client), gateway
`capacity.py` / `upstream_admission.py` / `priority_admission.py` / `lane_gate.py`, PR #2317
(`lane_contention.py`), the #2246 stance fallback chain, and `orion-gpu-lane-controller`'s own
decision-making.

## Arsonist summary

Right now nine different pieces of code decide which GPU a piece of work lands on, and four of
them have their own idea of priority. None of them sees the whole machine. Each slice since
2026-09-13 added another decider instead of consolidating.

This design replaces all of them with **one service, `orion-gpu-pool`**. Every GPU user asks it
for a **lease**: chat, the 27B agent lane, fast, metacog, world-model, diffusion, durable runs,
and Juniper's Hub button. It grants the lease or queues the request. It is the only thing that
knows what is on each card and the only thing allowed to decide who goes first.

Everything it knows (cards, what runs on each card, who may borrow what, priority order, grace
periods) lives in **one YAML file**, `config/gpu_pool.yaml`. Adding a card or a service is a
YAML edit, with no code change and no second table to keep in sync.

## Decisions already made (Juniper, 2026-09-24)

| Question | Answer |
| --- | --- |
| GPU0 borrowing | **Lent only.** Others use GPU0 only while Juniper's Hub button has it lent. Chat claws it back whether or not it is lent. |
| Placement | **New service** `orion-gpu-pool` on athena, next to Postgres and the gateway. A small circe-side actuator loads and unloads models. Durable-runs becomes a client. |
| Affect | **Out of scope** for the first cut. It becomes a YAML tenant later. |
| Clawback grace | **The borrower finishes its current request**, capped (default 60s). After that it gets no new work. |
| GPU2 | Defaults to world-model and diffusion. It is leased out as a second 27B when needed. |
| Config | Cards, tenants, borrow rules and leases are declared in YAML so new cards and services need no code. |

## The machine (live `nvidia-smi` on circe, 2026-09-24)

| card | hardware | home tenants | notes |
| --- | --- | --- | --- |
| gpu0 | V100-PCIE-32GB | chat 35B (:8011, 1 slot, 64k ctx) | reserved for chat; lendable by button |
| gpu1 | V100-SXM2-32GB | agent 27B (:8015, 1 slot) | 27B home |
| gpu2 | PG500-216 (a V100) | world-model (~1GB), diffusion-host (~24GB, :8014) | swappable to agent 27B (:8016, ~25GB) |
| gpu3 | V100-PCIE-32GB | metacog 8B (:8012), fast 8B (:8013) | the two 8B lanes share and can swap |

## The rules, stated once

1. **Every lease names a work class**, meaning what the work needs: `chat`, `agent`, `metacog`,
   `fast`, `world`, `diffusion`. A class lists the models able to serve it, in preference order.
2. **Home first.** A request goes to its home model when that model has a free slot.
3. **Borrowing** goes only in the directions the YAML allows:
   - `metacog` ↔ `fast` swap freely on gpu3.
   - `metacog` and `fast` may spill up to gpu1, to gpu2 when a 27B is loaded there, and to gpu0
     when it is lent. A bigger model can do small-model work.
   - Nothing on gpu0, gpu1 or gpu2 ever spills down to gpu3, because the models are too small.
   - `agent` may use gpu2's 27B (after a swap) and gpu0 when it is lent.
   - `chat` never leaves gpu0.
4. **The owner always wins on its own card.** When an owner-class request is waiting and every
   slot on its home model is held by borrowers, the pool **recalls** the newest borrower lease.
   The borrower finishes its current request (grace cap, default 60s) and gets no new work. If
   it overruns the cap it is aborted and requeued. The pool never grants a borrower a slot while
   an owner is waiting.
5. **Order inside the queue:** owner before borrower, then `priority`
   (`interactive > system > background`), then oldest first. That is the whole ordering. No
   per-lane budgets pretend to be priority.
6. **Memory-aware swaps (gpu2).** A swap-in model is loaded only if the card's VRAM budget
   allows it once the tenants it evicts are gone. The 27B (~25GB) plus world-model (~1GB) fit in
   32GB, so **loading the 27B evicts diffusion only, and world-model keeps running.** A diffusion
   request is an owner request and claws gpu2 back: the 27B leases are recalled, the 27B is
   unloaded, and diffusion is restored. A cooldown (default 600s) stops the card flapping
   straight back to the 27B.
7. **GPU0 lend** is a card flag that only an operator can set. While it is off, no class other
   than `chat` may be granted gpu0. Turning it off recalls every borrower on gpu0.
8. **Health gates grants.** The pool reads each model's `/health` or `/slots`. A model that is
   down gets no grants, and its queued work waits or goes to the next allowed model.

## `config/gpu_pool.yaml` (the whole configuration surface)

```yaml
version: 1
host: circe
defaults:
  clawback_grace_sec: 60        # borrower finishes its current request, capped here
  request_lease_ttl_sec: 30     # heartbeat window for one-inference leases
  hold_lease_ttl_sec: 90        # heartbeat window for held leases (durable runs, button)
  swap_cooldown_sec: 600        # after a clawback, don't re-swap the card for this long

priorities: [interactive, system, background]   # highest first

models:                          # every servable thing, whether resident or swappable
  chat-35b:   {url: http://100.112.254.99:8011, slots: 1, vram_gb: 28, health: /health}
  agent-27b:  {url: http://100.112.254.99:8015, slots: 1, vram_gb: 25, health: /health}
  agent-27b-gpu2: {url: http://100.112.254.99:8016, slots: 1, vram_gb: 25, health: /health,
                   actuator: {load: gpu2/agent, unload: gpu2/restore}}
  metacog-8b: {url: http://100.112.254.99:8012, slots: 4, vram_gb: 8,  health: /health}
  fast-8b:    {url: http://100.112.254.99:8013, slots: 4, vram_gb: 8,  health: /health}
  world-model: {url: http://100.112.254.99:6613, slots: 2, vram_gb: 1, health: /health}
  diffusion:  {url: http://100.112.254.99:8014, slots: 1, vram_gb: 24, health: /health}

cards:
  gpu0: {vram_gb: 32, owner: chat,  resident: [chat-35b], lendable: true}
  gpu1: {vram_gb: 32, owner: agent, resident: [agent-27b]}
  gpu2: {vram_gb: 32, owner: [world, diffusion], resident: [world-model, diffusion],
         swappable: [agent-27b-gpu2]}
  gpu3: {vram_gb: 32, owner: [metacog, fast], resident: [metacog-8b, fast-8b]}

classes:                         # which models may serve a class, in preference order
  chat:      [chat-35b]
  agent:     [agent-27b, agent-27b-gpu2, chat-35b]
  metacog:   [metacog-8b, fast-8b, agent-27b, agent-27b-gpu2, chat-35b]
  fast:      [fast-8b, metacog-8b, agent-27b, agent-27b-gpu2, chat-35b]
  world:     [world-model]
  diffusion: [diffusion]
```

```yaml
routes:                          # what the gateway's callers say -> work class
  chat: chat
  harness: chat
  agent: agent
  metacog: {class: metacog, priority: system}
  metacog_background: {class: metacog, priority: background}
  quick: fast
  quick_background: {class: fast, priority: background}
```

The gateway reads `routes:` and `models:` from this file. Its `.env` loses
`LLM_GATEWAY_ROUTE_TABLE_JSON`, `LLM_ROUTE_*_SERVED_BY`, `LLM_LANE_*`,
`LLM_GATEWAY_UPSTREAM_MAX_INFLIGHT`, `LLM_GATEWAY_CAPACITY_*`, `LLM_GATEWAY_BACKGROUND_*` and
`LLM_ALLOW_BACKGROUND_TO_CHAT_FALLBACK`. Every URL, slot count and fallback lives in exactly one
place.

A class is served on a card only if the card owns that class, or the card allows borrowing
(gpu0 only while lent). Rule 3's "no spill down to gpu3" is structural: no big-model class
lists an 8B model. Validation runs at boot and in CI: every class resolves to known models,
every model sits on exactly one card, and resident VRAM fits the card. A bad YAML fails the
boot instead of degrading.

## Proposed schema / API

**Postgres (one migration, `services/orion-sql-db/manual_migration_gpu_pool_v1.sql`):**
- `gpu_pool_leases`: `lease_id, request_id (unique, idempotency), holder, work_class,
  priority, kind (request|hold), status (queued|granted|recalling|released|expired|aborted),
  model, card, generation, created_at, granted_at, heartbeat_at, expires_at, recall_at,
  deadline_at, correlation_id, detail jsonb`. Queued requests and granted leases share one
  table, so "what is out there" is one query.
- `gpu_pool_cards`: `card, lent bool, swapped_in text null, swap_state, cooldown_until,
  updated_at, updated_by`.
- `gpu_pool_events`: an append-only audit log and outbox for the bus.

**Every lease is a LangGraph run (Juniper, 2026-09-24: "let's use langgraph").** `thread_id =
lease_id`. It is checkpointed by the same `AsyncPostgresSaver` + pool pattern durable-runs
already uses. Graph:

```text
admit ──> enqueue ──> wait_grant ──(interrupt until scheduler resumes)──> granted
                          │                                                 │
                          └── deadline passed ──> expired                   v
                                                                  hold (interrupt;
                                                                  heartbeats resume it)
                                                                    │     │       │
                                                          release <─┘  recalled  lost heartbeat
                                                             │            │          │
                                                             v            v          v
                                                          released   grace ──> requeue ──> enqueue
                                                                           (or aborted)
```

- A wait is an `interrupt`, not an open socket, so nothing times out. The run holds no task or
  connection while it waits. The caller is woken by the bus `granted` event.
- A pool restart resumes every thread from its checkpoint. A failed or recalled lease replays
  through `requeue` with its original `request_id`, `priority` and `created_at`, so it keeps its
  place in line.
- The **scheduler is one node-free deterministic function**, `schedule(yaml, cards, queue,
  leases) -> [grant | recall | abort | swap]`. A single-writer loop (Postgres advisory lock,
  1s tick plus a wake on every admit or release) calls it and then resumes the affected threads
  with `Command(resume=...)`. It is pure, so it is fully unit-testable with a fake clock, and
  the eval replays traffic through it.
- The graph's knobs (grace, TTLs, cooldown, priority order) come from `config/gpu_pool.yaml`.
  The graph shape is code and the policy is YAML.
- `gpu_pool_leases` becomes a small **materialized projection** of thread state, one row per
  lease, updated in the same transaction as the scheduler decision. It exists so "what is out
  there" is one indexed query and so grants are fenced with `FOR UPDATE`. The LangGraph
  checkpoint remains the source of replay.
- Cost, measured in stage 1's eval rather than assumed: a request lease is about 4 checkpoint
  writes (admit, enqueue, granted, released). If metacog/fast volume makes that the bottleneck,
  the eval will show it before the gateway cutover depends on it.

**HTTP (`orion-gpu-pool`, athena):**
- `POST /v1/leases`: `{request_id, holder, work_class, priority, kind, deadline_at?,
  correlation_id}` → `granted {lease_id, generation, model, url, card}` or `queued {position}`.
- `GET /v1/leases/{id}?wait=25`: long-poll until granted or recalled. Interactive callers loop
  on this until their own `deadline_at`. There are no blind HTTP timeouts.
- `POST /v1/leases/{id}/heartbeat` → `{valid, recall: bool, recall_by?}`.
- `POST /v1/leases/{id}/release`.
- `GET /v1/pool`: cards, what is loaded, leases held, queue, recalls in flight. This is the "what
  the fuck is out there" view.
- `PUT /v1/cards/{card}/lend {lent}`: operator only (bearer token). Hub's button calls this.
- `GET /health`.

**Bus:** `orion:gpu:pool:event` → `GpuPoolEventV1 {event: granted|recalled|released|expired|
aborted|swapped|lent, lease_id?, card?, holder?, detail}`. It is registered in `channels.yaml`
and `registry.py`. Durable runs wake on it, and Hub's panel refreshes on it.

**Client library `orion/gpu_pool/client.py`:** one async context manager,
`async with gpu_lease(work_class=..., holder=..., priority=...) as lease:`, which handles
acquire, long-poll, heartbeat and release, and raises `LeaseRecalled` when a recall hits.
Every caller uses this. There is no second client.

## How each caller changes

| caller | today | after |
| --- | --- | --- |
| llm-gateway | route table → port, in-process semaphores, capacity permits, background slot polling, lane gate | maps `route → work_class` and takes a **request lease** per dispatch. The pool returns the URL to call, and the gateway dispatches there. If the caller already holds a **hold lease** (durable run), the gateway dispatches on that lease's model. A recall during dispatch returns `recalled` and the gateway requeues once. |
| durable-runs | `resource_request → resource_wait (interrupt) → ...` against its own broker | same graph shape. `resource_request` takes a **hold lease** from the pool, `resource_wait` interrupts until the bus announces `granted`, and a recall or failure goes to `retry_wait`, which requeues. LangGraph checkpoints keep the replay. |
| world-model, visual chain (diffusion) | `GpuCapacityPermit` on a borrowed key | `gpu_lease(work_class="world" / "diffusion")` |
| Hub | "Lend chat lane" button → gateway Redis gate; hold-and-email logic | the button → `PUT /v1/cards/gpu0/lend`. A new pool panel reads `GET /v1/pool`. Chat traffic is never "held" any more: chat is the owner and claws back. |
| gpu-lane-controller (circe) | GPU1 affect flip, GPU2 transitions with its own lock | becomes the pool's **actuator**: `POST /v1/actuate {card, action}`, taking orders only from the pool (bearer token). The GPU1 flip is deleted (affect is out of scope). |

## Downstream readers of the things being deleted (must move to the pool in the same stage)

These read the gateway's `/admission` snapshot or the durable-admission tables today. Deleting
the source without moving these would not error: each is fail-open, so each would quietly read
"calm" or empty.

| reader | what it reads today | what it means | moves to |
| --- | --- | --- | --- |
| `orion-field-digester/app/digestion/queue_contention.py` → `FieldStateV1.queue_contention_score` | `count_durable_demand_pending()` (SQL on `durable_resource_demands`) and the gateway's `/admission` waiting sum | Orion's "am I backed up" field signal. Curiosity hire decisions read it (#2263/#2281). **It is a locked metric** (`config/metrics/metric_definitions.lock.json`). | pool queue depth per class. Changing a locked metric's producer needs Juniper's approval, so stage 2 either re-points the two sources as a like-for-like count (same meaning: "requests waiting for GPU") under the metric gate, or retires it for a pool-native successor. **Decision needed at stage 2.** |
| `orion-cortex-exec/app/admission_cue.py` | gateway `/admission` ledger (background requests made to wait) | Orion's own metacog cue "was my background thinking made to wait" (scarcity roadmap A5) | pool lease events for `priority=background`: waited / waited-how-long / recalled. Same four states (observed-zero, waited-N, no-requests, unknown). |
| `orion/hub/runtime_activity.py` + `scripts/runtime_activity_routes.py` | gateway `/admission` per-upstream inflight and waiting | Hub runtime activity panel | `GET /v1/pool` |
| `orion/curiosity/run_story.py`, `hire_progress.py`, `hub/scripts/curiosity_run_store.py`, `curiosity_atlas.html` | `durable_admission_runs` and `durable_resource_events` (waiting for lane, granted, lease released or expired) | the curiosity run story and the "reach-out truth" tab | pool lease events via sql-writer, keyed by `run_id` as `holder`, keeping the same event names where the meaning is the same |
| `orion/hub/turn_orchestrator.py`, gateway passthroughs | lease fencing and `lane_gate` refusals | chat hold-while-lent, burst refusals | the lease from the pool; hold-while-lent goes away because chat is the owner |
| `orion/inner_state_registry.py`, `orion/schemas/field_state.py` | documentation of `queue_contention_*` sources | registry of inner-state fields | updated with the new source |

Transport telemetry (bus RPC health, substrate grammar, signals, field nodes, runtime metrics)
is being traced separately; a section follows once the live path is confirmed.

## Delete list (in the same stage that replaces each piece)

- `orion/durable_admission/{broker,policy,capacity,capacity_client,elastic,store}.py`,
  `services/orion-durable-runs/app/{admission_runtime,elastic_runtime}.py`, and their tables
  (dropped by migration only after cutover is verified).
- The gateway's `capacity.py`, `upstream_admission.py`, `priority_admission.py`, `lane_gate.py`
  and `admission_ledger.py`, plus `BURST_LLM_ROUTES`/`OPERATOR_GATED_LLM_ROUTES`/
  `CHAT_BURST_LENDS_ROUTE`.
- The unread `LLM_LANE_CONTENTION_FALLBACK_JSON` / `LLM_LANE_REAL_CAPACITY_JSON`, the
  `DURABLE_RUNS_{ADMISSION,CAPACITY,WIDENING,ELASTIC}_*` keys, `WM_GPU2_CAPACITY_*`,
  `ORION_VISUAL_CHAIN_GPU2_CAPACITY_*`, and `HUB_CURIOSITY_ELASTIC_ACTIVATION_ENABLED`.
- Hub's `chat_lane_lend.py` hold-and-email path.
- The #2246 stance fallback chain in cortex-exec/thought. Spill is now the pool's job.
- PR #2317 is closed unmerged.
- Fix the stray `ATLAS_AGENT_HOST_PORT=8014` in `services/orion-llamacpp-host/.env`.

## Files likely to touch

`config/gpu_pool.yaml` (new); `services/orion-gpu-pool/` (new: app, settings, scheduler, store,
api, Dockerfile, compose, tests, evals); `orion/gpu_pool/{config,client}.py` (new);
`orion/schemas/gpu_pool.py`, `orion/schemas/registry.py`, `orion/bus/channels.yaml`;
`services/orion-sql-db/manual_migration_gpu_pool_v1.sql`; `services/orion-llm-gateway/app/*`;
`services/orion-durable-runs/app/*`; `services/orion-world-model/app/main.py`;
`services/orion-thought/app/visual_chain.py`; `services/orion-hub/{scripts,static,templates}`;
`services/orion-gpu-lane-controller/app/*`; CI gate script.

## Non-goals

- Affect as a tenant (a later YAML entry).
- athena's GPUs (T10, P4, kev). The YAML's `host` field leaves room for them, but this cut is
  circe only.
- Splitting a single llama.cpp request across cards, or live migration of a running request.
- Automatic model re-quantization or context resizing per card.

## Acceptance checks

1. **Config:** `config/gpu_pool.yaml` validates at boot and in CI. A card with too much resident
   VRAM, an unknown model, or a class with no servable model fails the check.
2. **Scheduler unit tests** (deterministic, fake clock):
   - home-first grant;
   - metacog→fast swap;
   - metacog spill to gpu1;
   - agent never to gpu3;
   - gpu0 refused while not lent;
   - a chat request recalls a lent-gpu0 borrower, which is aborted after grace;
   - owner-before-borrower ordering;
   - priority ordering;
   - idempotent `request_id`;
   - expired heartbeat frees the slot;
   - two schedulers → one grant (advisory lock);
   - gpu2 swap evicts diffusion but keeps world;
   - diffusion claws gpu2 back;
   - cooldown blocks re-swap.
3. **Eval:** replay a synthetic day of mixed traffic through the scheduler and report p50/p95 wait
   per class, the number of recalls, and owner-starvation seconds (target: 0 seconds of owner
   wait while a borrower holds its card beyond grace).
4. **Live:**
   - `GET /v1/pool` shows all four cards with the real tenants.
   - A metacog burst beyond 4 slots shows grants on fast-8b, then agent-27b.
   - Toggling the Hub button shows gpu0 lent/unlent, and a chat message during a lend
     shows a `recalled` event followed by a chat grant.
5. **Enforcement gate:** CI fails if any code outside `orion-gpu-pool` and the gateway's
   dispatch references a circe model port (8011–8016, 6613) directly.

## Build order (each stage stops for Juniper's review)

1. **Pool core:** YAML + validation, schema, migration, scheduler, HTTP, bus event, client
   library, tests, eval, Hub read-only panel. Deployed in *observe* mode: it answers, but
   nobody depends on it yet.
2. **Gateway cutover:** all LLM traffic leases from the pool; the Hub lend button moves to the
   pool; the gateway deciders are deleted; #2317 is closed.
3. **Durable-runs cutover:** admission nodes lease from the pool; `orion/durable_admission` is
   deleted.
4. **gpu2 and non-LLM tenants:** world-model and diffusion lease; the actuator becomes the
   swap executor; the elastic code is deleted.
5. **Lockdown:** the port-poacher CI gate, the old env keys removed everywhere, and the old
   tables dropped.

## Recommended next patch

Stage 1, pool core, in `feat/gpu-pool` from this worktree.
