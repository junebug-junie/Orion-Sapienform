# GPU pool stage 5: world and diffusion on pool leases, one generic "load this role on this card" actuator, then delete the old broker

Status: design, for Juniper's review. No service code in this PR.
Date: 2026-09-29
Parent spec: `docs/superpowers/specs/2026-09-24-gpu-pool-design.md` (build order stage 5, downstream readers item 8,
"Cosmetic but misleading", delete list).
Stage 4 spec: `docs/superpowers/specs/2026-09-25-gpu-pool-stage4-durable-runs-and-actuation.md` (the actuation
contract `GpuActuateV1`/`GpuActuateResultV1`, `launch:` blocks, and all its "Corrections" sections).
Standing requirement (Juniper): adding a card, or a new model option on a card, must be config
(`config/gpu_pool.yaml` + `config/llm_profiles.yaml` + compose), plus one generic actuator. No per-card code, and no
gpu2 special cases.

## Arsonist summary

Every LLM call and every durable run already waits in one line: the GPU pool. Two things still don't.

1. **The world model and image generation still ask durable-runs for a permit** before touching gpu2. The durable
   broker behind that permit was deleted in stage 4; the only thing left of it is `capacity.py` guarding this one
   card. Image generation now asks *twice*: its durable run takes a pool hold on `diffusion`, then the generate step
   also takes the old permit (93 permits in 7 days, all released).
2. **Loading a model on a card still goes through two hard-coded gpu2 moves.** The pool decides "load agent-gpu2",
   and circe's controller translates that into one of two fixed gpu2 transitions (`gpu2/agent`, `gpu2/restore`). It
   works (37 loads through the pool in 4 days, about 16 s each), but a gpu4, or a second model option for gpu2, would
   need new controller code.

Stage 5 does four things, in independently deployable PRs:

- **The controller runs any role's `launch:` block.** The card's CUDA device comes from the card's `index`, and the
  model comes from an allowed profile list. Adding gpu4 then touches YAML and compose only (worked examples below).
- **World and diffusion stop taking permits.** The world model takes a short pool lease. Image generation relies on
  the hold it already takes. One new YAML rule, `serialize_with`, keeps today's guarantee that the world model and
  diffusion never compute on gpu2 at the same time.
- **Labels come from the pool.** The Hub biometrics GPU labels (empty in production today, so every card reads
  "unassigned"), the fixed gpu2 slot schema, and diffusion's hard-coded power-intent GPU index are all derived from
  the pool's card and role state instead.
- **Then the old broker is deleted:** `capacity.py`, `capacity_client.py`, `/capacity`, four dead tables, the
  bridge verbs, the GPU1 affect flip (zero calls in 7 days), and their env keys.

The multi-card experiment seat is **recommended deferred**. Nothing exists to put in it (no compose service or
profile uses port 8099 on circe), and making it real means building a four-card evict-everything path with no
workload to test it against.

## Current architecture (verified 2026-09-29)

### Who gets gpu2, and how

| workload | how it gets gpu2 today | live evidence |
| --- | --- | --- |
| 27B agent seat `agent-gpu2` | pool `SwapLoad` → `GpuActuateV1` on `orion:gpu_pool:actuate:request` → circe `orion-gpu-lane-controller` `actuator_bus._dispatch` → `pool_fence.resolve` maps `swap.load: gpu2/agent` via `BRIDGE_TARGETS` → `gpu2.transition("agent-burst")` | `gpu_pool_events` last 4 days: 37 `swap_started`/`swapped reason=demand`, 29 `swapped reason=max_hold`, 5 `swapped reason=owner_reclaim` (a diffusion hold took the card back), 3 `swap_failed reason=burst_upstream_not_idle`. Now: gpu2 `swapped_in=[agent-gpu2]`, generation 77, load 18:33:23 → 18:33:39 UTC |
| image generation (reverie visual) | durable run `reverie-visual-*` takes a pool **hold** on class `diffusion` (`hold_routes.diffusion`). Then `orion-thought` `generate_visual_bytes` (`visual_chain.py:1123`) *also* takes a `GpuCapacityPermit` from durable-runs `/capacity`, backend key `http://100.112.254.99:8014` | 20 diffusion holds (09-28 → 09-29, p50 hold 1 min). `durable_gateway_permits lane=diffusion`: 93 in 7 days, last 2026-09-29 18:11, the same minute as the last diffusion hold. The run-once worker is off (`ORION_VISUAL_CHAIN_ENABLED=false`); the elastic status pre-check is off (`ORION_VISUAL_ELASTIC_STATUS_ENABLED` unset, default false) |
| world model | `orion-world-model` (circe, `cuda:2`, 872 MiB) takes a `GpuCapacityPermit` on the **same** backend key, 2 s budget, and returns `error_code=gpu_contended` if refused (`services/orion-world-model/app/main.py:272-296`) | `lane=world-model` permits: **3 ever**, all 2026-09-24. Container logs for 24 h: `/health` only, no task requests. It is idle, not starved. Its caller is `orion-substrate-runtime` |

**What the permit actually guarantees** (`orion/durable_admission/capacity.py`): for one backend key, at most
`min(max_inflight)` active permits across callers. World and diffusion share a key, so they never run at the same
time. It also refuses any permit while the (now frozen) elastic slot or a durable lease is active on that key.
Both are dead since 4.5. The reason for the mutex (the comment in `world-model/app/settings.py`) is two `CUDA-capable
device(s) is/are busy or unavailable` failures on 2026-09-24. **UNVERIFIED cause:** circe's compute mode is
`Default` on all four cards (`nvidia-smi`), so this is not exclusive-process mode. VRAM exhaustion during a
transition is the likelier cause. Stage 5 keeps the mutex anyway (see Decision 1), because dropping a guarantee
while replacing its mechanism is how a "cleanup" becomes an outage.

### The pool today

- `GPU_POOL_MODE=observe` and `GPU_POOL_ACTUATE_ROLES=agent-gpu2` (athena container env).
- **"observe" no longer means "not in charge".** The pool grants every LLM lease and actuates `agent-gpu2`. In
  `runtime.py` it now switches exactly two behaviours:
  - `_observe_swap_seats` (`:262`) marks a non-actuated swap seat loaded when its worker answers;
  - operator `hold` is refused with `hold_requires_swap_actuation` (`:438`).
- **Guards:**
  - `visual_baseline` fired 5 times (last 2026-09-27);
  - `thermal` fired 3 times;
  - `cooldown` fired 2141 times (edge-triggered observe events; last 2026-09-28).
- **Lent gpu0.** 231 distinct leases were recalled off `chat` (the lent gpu0) in 7 days: agent 116, metacog 89,
  fast 26. 28 of those were durable holds (`gpu_pool_leases.reason LIKE 'recall%'`).

### Controller today

`services/orion-gpu-lane-controller/app/`:

- `actuator_bus.py`: bus consumer, idempotent by `action_id`, fence state on the `orion-gpu-lane-controller-state`
  volume.
- `pool_fence.py`: `resolve()` refuses anything that isn't a bridge verb (`not_a_bridge_role`), and refuses any
  non-null `profile` (`profile_unsupported`).
- `gpu2.py`: fixed `targets()`. Diffusion gets `extra_env={"CUDA_VISIBLE_DEVICES": "2"}`; agent-burst relies on
  compose. It also holds the drain, stop, start, ready and rollback code.
- `lane_control.py` + `/v1/gpu-lanes/flip`: the GPU1 affect flip. **0 POSTs to `/v1/*` in the last 7 days**
  (container logs).
- `GPU2_AUTHORITY=pool`.
- **The circe checkout is at `f720356c1` (2026-09-25).** The digest fence means circe has to pull alongside athena
  for any YAML change to a `launch` role (stage 4 correction 4.2-2).

### The device is hard-coded in one place

`services/orion-llamacpp-host/docker-compose.atlas-workers.yml:200` has `atlas-agent-burst`
`CUDA_VISIBLE_DEVICES_OVERRIDE=2`, a literal (stage 4 correction 4.1-5). The resident workers use
`${ATLAS_*_CUDA_VISIBLE_DEVICES}`, and diffusion-host uses `CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}`.
`orion/gpu_pool/config.py:468-474` already checks that the *resolved* value equals the card index. It does not
require that the actuator can *set* it.

### Tables

| table | rows | status | stage 5 |
| --- | --- | --- | --- |
| `durable_gateway_permits` | 168,979 (127 MB) | live for world + diffusion only | drop after 5.4 |
| `durable_resource_demands` | 158 (all `withdrawn`) | frozen since 4.5 | drop |
| `durable_resource_leases` | 225 (released/expired) | frozen; `store._expire` still updates it | drop |
| `durable_elastic_slot` | 1 | frozen | drop |
| `durable_admission_runs` | 353 | **live run registry** (Hub curiosity views, `urgent_report.py`) | **keep** |
| `durable_resource_events` | 61,432 | **live run event outbox** (`run_story`, Hub; metric lock `bus_channel/orion-durable-runs/orion:durable:resource:event`) | **keep** |

So **"delete `orion/durable_admission/*`" is not literally possible.** `store.py`'s `PostgresAdmissionStore` is the
durable-runs run registry and outbox (`admission_runtime.py:54`). Stage 5 moves it rather than deleting it, and deletes
only its `_expire` (the frozen-lease sweep).

## Decision 1: world and diffusion on pool leases

### How today's outcomes map onto the pool

| today (permit) | after (pool) |
| --- | --- |
| world: permit refused within 2 s → `error_code=gpu_contended` | world takes a **request** lease: class `world`, `priority=system`, deadline = `WM_GPU2_CAPACITY_BUDGET_SEC` (2 s). A `queued` reply past the deadline gives `unavailable reason=deadline` → the same `gpu_contended` code and text. Class `world` changes `on_unavailable: backlog` → **`wait`**: a 30-second prediction is useless when replayed hours later. This corrects the parent spec's "ticks that hit backlogged are replayed" |
| world: permit renew/release | lease heartbeat/release via `orion.gpu_pool.client` (world-model is on the athena bus: `ORION_BUS_URL=redis://100.92.216.81:6379/0`). Lease RPC uses a fresh correlation id, per the stage-1 rule |
| diffusion: permit refused → `DiffusionResourceDeferred` → chain `terminal_reason=resource_deferred` | **no second gate.** The run's diffusion hold *is* the grant. `generate_visual_bytes` drops the permit. `resource_deferred` still fires when `validate_hold_ref` (`visual_steps.py:442`) rejects the hold (expired, recalled, wrong generation). Same outcome label, now with one cause instead of two |
| diffusion vs world mutex (shared backend key) | **new YAML rule `serialize_with`** (below) |
| `durable_waiting` (hold new permits while a durable demand waits) | pool queue order (owner, then priority, then FIFO). No equivalent needed |
| `elastic_requires_owner` (no diffusion permit while the 27B is loaded) | already the pool's behaviour. `diffusion` is `evicted` while `agent-gpu2` is swapped in, so a diffusion hold queues, drains the seat (`scheduler.py:405-410`), and is granted after the unload restores diffusion. Live: 5 `owner_reclaim` swaps in 4 days |
| drain | unchanged. The actuator's `launch.drain` on diffusion (`/v1/lifecycle/drain`) runs before any stop |

### `serialize_with`: one line of YAML, one scheduler check

```yaml
world: {kind: service, cards: [gpu2], owner: world, port: 6613, slots: 2, vram_gb: 1,
        serialize_with: [diffusion]}   # never compute at the same time as these roles (same card)
```

- **Scheduler:** a lease is not placeable on role R while any active lease is on a role in R's `serialize_with`,
  or on a role that lists R. The rule is symmetric. Blocked placement reports `reason=serialized:<role>` on the
  queued event, so it is never silent.
- **Validation:** every role named must exist and share at least one card with R.
- **Precedence:** it follows today's asymmetry without a new priority concept. World's 2 s deadline means it gives
  up; diffusion's hold simply waits its turn in line.
- **Cheap to remove.** If the 09-24 failure turns out to be VRAM during a transition (see the UNVERIFIED note above),
  deleting the one YAML line removes it. PR 5.4's report includes a 20-minute check: run concurrent world and
  diffusion calls with no swap in flight, and record whether any CUDA error occurs. That is evidence for a follow-up,
  not a gate for this PR.

### The `visual_baseline` guard

Stage 4 said to delete it "in the PR that puts the visual chain on diffusion leases". The visual chain is already on
diffusion holds, and owner reclaim is live. What the guard adds is refusing a 27B load when the image baseline is
*overdue but no hold is queued yet*. In that situation a reverie-visual run submits on its own cadence (20 in ~26 h)
and reclaims the card through the queue within one load. **Delete it in 5.4** together with
`GPU_POOL_VISUAL_ACTIVITY_URL`. Thought's `/visual-chain/activity` route stays, because Hub reads it.

## Decision 2: the generic actuator

### What the controller does for any role

For `GpuActuateV1{role, action, cards, profile, generation, launch_digest}`, `pool_fence.resolve()` returns a
**launch plan** built from the controller's own checked-out `gpu_pool.yaml`, instead of a bridge target:

```text
LaunchPlan(role, compose, env_file, service, compose_profile,
           env = {launch.cuda_env: ",".join(str(cards[c].index) for c in role.cards),
                  launch.profile_var: <profile>  # only when the request names a profile
                 },
           drain, ready, port, timeout_sec,
           evicts = [LaunchPlan(...) for each evicted role that has a launch])   # "all" = every role on those cards
```

- **load:** for each evicted role, drain (if it has `drain`) → stop. Then `compose up -d --no-build --no-deps` the
  seat with `env`, and wait for `ready` up to `timeout_sec`. On failure: stop the seat, start the evicted roles in
  reverse order, and wait for each one's `ready`. The result is `failed restored=true|false`.
- **unload:** check the seat's llama.cpp `/slots` are all idle (the existing `burst_upstream_not_idle` refusal,
  made generic for `kind: llm`). Then stop the seat, start the evicted roles, and wait for ready.
- **status / `observed`:** `docker compose ps --format json <service>` for every `launch` role naming this actuator,
  mapped to `running|exited|absent|unknown`. This replaces `TARGET_ROLES`.
- **Fence, idempotency and persisted state are unchanged.** They are already keyed by card set and `action_id`, not
  by gpu2.
- **Safety property kept:** the bus message names a role and a profile, never a container, path or env value. The
  profile must be in the role's `launch.profiles` allow-list, or the request is refused with `profile_not_allowed`.
  Every compose path, service and env var name comes from the actuator's own YAML copy, fenced by `launch_digest`.
  There is no free-form Docker surface.

Code shape:

- `gpu2.py`'s fixed `targets()` and gpu2-specific drain/ready become `launch_exec.py`, which takes a `LaunchPlan`.
  The code moves; the steps it runs stay the same.
- `BRIDGE_TARGETS`, `TARGET_ROLES` and the `swap.load`/`swap.unload` YAML keys are deleted in 5.6.
- In 5.2 the bridge still works, so the cutover (5.3) is a config flip that can be reverted by reverting the YAML.

### Two new optional `launch` keys (a model choice per role)

```yaml
launch:
  profile_var: ATLAS_AGENT_BURST_PROFILE_NAME   # compose interpolation var the actuator sets to the chosen profile
  profiles: [qwen3.8-27b-udq4kxl-v100-32gb-circe-agent-flex, <another-llm_profiles-name>]   # allow-list, first = default
```

- **How the scheduler picks a profile when it emits `SwapLoad`:** the first profile in `profiles` whose
  `llm_profiles.yaml` entry satisfies the queued lease's requirements (`supports_vision` for `needs_vision`,
  per-slot ctx for `min_ctx_tokens`, and fitting the card's VRAM after evictions).
- Discovery then confirms the announced profile against `/props`, as it does today. A role without `profiles` sends
  `profile=None` (compose default), which is today's behaviour.
- **Switching models on a loaded seat is a load of the same seat with a different profile:** unload, then load. The
  pool does it only when a queued lease cannot be placed on the loaded profile and `swap_after_wait_sec` has passed.
  Min residency and cooldown apply as for any swap.

### Meaning change: `cuda_env`, plus a gate

- **Today:** `cuda_env` names the *container* variable, and the gate checks that its resolved value equals the
  card index.
- **Stage 5:** `cuda_env` names the **compose interpolation variable** the actuator sets. The gate
  (`orion/gpu_pool/config.py` launch checks, run by `scripts/check_gpu_pool_config.py` in CI) requires the service's
  device entry to be exactly `${<cuda_env>}` or `${<cuda_env>:-<index>}`, never a literal. `profile_var` gets the
  same rule for `LLM_PROFILE_NAME`.
- **Migration:** only two roles use `cuda_env` today.
  - diffusion: `CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}` already satisfies it, so no change.
  - agent-burst: compose line 200 becomes `CUDA_VISIBLE_DEVICES_OVERRIDE=${ATLAS_AGENT_BURST_CUDA_VISIBLE_DEVICES:-2}`,
    and YAML `cuda_env: ATLAS_AGENT_BURST_CUDA_VISIBLE_DEVICES`. Also `LLM_PROFILE_NAME=${ATLAS_AGENT_BURST_PROFILE_NAME:-${ATLAS_AGENT_PROFILE_NAME}}`.
  - Add both keys to `services/orion-llamacpp-host/.env_example`, then sync.
- **Why not have the actuator set the container variable directly:** compose only accepts outside values through
  interpolation. `lane_control._invoke_env` already passes process env to `docker compose`, and process env wins over
  `--env-file` for interpolation.

### Worked example A: add gpu4, which hosts either an 8B or a vision model

Only these files change:

1. `config/gpu_pool.yaml`:
   ```yaml
   cards:   {gpu4: {vram_gb: 32, index: 4}}
   roles:
     fast2:   {kind: llm, cards: [gpu4], owner: [metacog, fast], port: 8017,
               launch: {actuator: circe, compose: services/orion-llamacpp-host/docker-compose.atlas-workers.yml,
                        env_file: services/orion-llamacpp-host/.env, service: atlas-fast2,
                        cuda_env: ATLAS_FAST2_CUDA_VISIBLE_DEVICES, ready: /health, timeout_sec: 300}}
     vision4: {kind: llm, cards: [gpu4], owner: vision, port: 8018,
               launch: {actuator: circe, compose: services/orion-llamacpp-host/docker-compose.atlas-workers.yml,
                        env_file: services/orion-llamacpp-host/.env, service: atlas-vision4, compose_profile: vision4,
                        cuda_env: ATLAS_VISION4_CUDA_VISIBLE_DEVICES, profile_var: ATLAS_VISION4_PROFILE_NAME,
                        profiles: [<vision profile>], ready: /health, timeout_sec: 600},
               swap: {evicts: [fast2], guards: [thermal]}}
   classes:
     fast:    {roles: [fast, metacog, fast2, agent, agent-gpu2, chat], on_unavailable: wait}
     metacog: {roles: [metacog, fast, fast2, agent, agent-gpu2, chat], on_unavailable: backlog}
     vision:  {roles: [vision4], on_unavailable: backlog}
   ```
2. `services/orion-llamacpp-host/docker-compose.atlas-workers.yml`: two services (`atlas-fast2` always on,
   `atlas-vision4` under profile `vision4`), each with `LLM_ROLE`, `LLM_ANNOUNCE_PORT`,
   `LLM_PROFILE_NAME=${...}`, and `CUDA_VISIBLE_DEVICES_OVERRIDE=${ATLAS_*_CUDA_VISIBLE_DEVICES:-4}`.
3. `services/orion-llamacpp-host/.env_example` (+ synced `.env` on circe): the two profile-name keys.
4. `config/llm_profiles.yaml`: only if the vision model is new.
5. **Only if a caller needs a new route:** a `routes:` line (e.g. `vision: {class: vision}`).

No pool, scheduler, gateway or controller code changes. `check_gpu_pool_config.py` fails the PR if the compose
service, port, `LLM_ROLE`, device interpolation or profile allow-list disagree with the YAML.

**Deploy:** pull on athena and circe (same commit, because of the digest), `up -d atlas-fast2` on circe, then
restart the pool.

### Worked example B: a second model option for an existing card (gpu2's 27B seat)

Only these files change:

1. `config/gpu_pool.yaml`: under `agent-gpu2.launch`, add `profiles: [<27B profile>, <new profile>]` (the
   `profile_var` is already set by 5.1).
2. `config/llm_profiles.yaml`: the new profile, if it doesn't exist yet.

That's it. The seat's compose service already interpolates `LLM_PROFILE_NAME`. The scheduler loads the new profile
only when a queued lease needs something the default can't give (vision, larger per-slot ctx). If both profiles
satisfy every queued lease, the first one wins, so adding an option never changes today's default.

## Decision 3: the experiment seat: recommend deferring

To make it real, all of this would have to exist:

- `launch` blocks for `experiment`, plus launch blocks for **every** resident it evicts. `chat`, `agent`, `metacog`
  and `fast` have none today; they are always-on compose services, not actuated.
- A real `experiment` service. `docker-compose.dsv41.yml` announces `LLM_ROLE=experiment` on 8099, but it is not
  deployed on circe, and there is no profile or plan for it.
- The pool in `enforce` mode, because operator `hold` is refused in observe.
- The Hub operator hold/release flow (the panel control exists in `gpu_pool.test.js`, and would be exercised for the
  first time).
- A restore-all path that brings back four residents, including chat, while reporting partial failure as `fault`.

**Recommendation: defer.**

- Zero operator or experiment leases have ever existed (`gpu_pool_leases`).
- Evicting chat has the highest blast radius of any action in the system.
- Every piece above except the service itself comes free from stage 5's generic actuator. Once 5.2 lands, adding
  `launch` blocks to the residents and one compose service is config.

Stage 5 keeps the `experiment` YAML entry, but moves `swap.load: circe/experiment`/`unload: circe/restore` out
(they are bridge verbs, and 5.6 deletes the bridge). Until launch blocks exist, the validator marks `experiment` as
`not_actuatable`, and the Hub panel greys out its control with that reason.

## Decision 4: labels derived from the pool

| label | today | after |
| --- | --- | --- |
| Hub biometrics GPU lane (`biometrics_preview_routes._parse_lane_map`) | `GPU_LANE_MAP_{ATHENA,CIRCE}_JSON`, **both `{}` in the live container**, so every card reads `unassigned` | circe cards: card index → the roles currently resident (`orion:gpu_pool:state` `cards[].swapped_in` plus non-swap roles on that card, discovery status `confirmed`). Hub already consumes pool state for its panel. athena has no pool (non-goal): its cards read `unassigned`, as they do today. Keys deleted from `services/orion-hub/{.env_example,app/settings.py}` + test |
| `orion/schemas/gpu_slot.py` (`GpuSlotRequestV1`: `circe-gpu1/2`, fixed targets) | used only by the controller's HTTP slot routes and tests | deleted with `/v1/gpu-slots/*`, `/v1/gpu-lanes/flip` and `lane_control.py` (GPU1 flip: 0 calls in 7 days). The pool's `GpuActuateV1` is the only actuation contract |
| `DIFFUSION_POWER_INTENT_GPU_INDEX=2` (diffusion-host `settings.py`, `main.py:602`, `.env_example`, test; world-model `gpu.py` / `.env_example` reference it in comments) | a hard-coded physical index | diffusion-host derives the physical index from its own `CUDA_VISIBLE_DEVICES` (the actuator sets it from the card `index`; `CUDA_DEVICE_ORDER=PCI_BUS_ID` is baked into the diffusion-host image, `services/orion-diffusion-host/Dockerfile:51`, so the CUDA index equals the `nvidia-smi` index). The key is deleted. A boot check refuses a multi-device or non-numeric value with a logged `power_intent_gpu_index_unresolved`, and publishes no intent (it never guesses) |

**Metric gate** (`power_intent_settled` reads `gpu_index`):

1. Provenance: `diffusion-host/app/main.py:602` → `orion:power:intent` → `orion-biometrics/app/power_intent.py`.
2. Independence: the value is unchanged (2), only its source moves.
3. Theory anchor: the settle window must read the card the model runs on.
4. Live check: PR 5.5 compares the published `gpu_index` for the last 20 intents before and after deploy; they must be
   identical.
5. Existing mechanism: none.
6. Reversibility: a one-line revert.

## Decision 5: delete, and kill means kill

After 5.4 is verified live (zero new `durable_gateway_permits` rows for 24 h):

- **Code:** `orion/durable_admission/capacity.py`, `capacity_client.py`, and
  `orion.schemas.resource_admission` `Capacity*V1`.
  - durable-runs `/capacity/*` routes, `capacity_enabled`, `DURABLE_RUNS_CAPACITY_*`.
  - `services/orion-durable-runs/evals/gateway_capacity.py` and its workflow steps.
  - thought `visual_chain_gpu2_capacity_*`, `visual_elastic_*` and the dead elastic pre-check.
  - world-model `WM_GPU2_CAPACITY_*`.
  - `store._expire`.
- **Moved, not deleted:** `orion/durable_admission/store.py` → `orion/durable_runs/registry_store.py` (same class,
  renamed `DurableRunRegistryStore`). `orion/durable_admission/` is then removed entirely.
  - Importers updated: `admission_runtime.py`, `main.py`, tests.
  - Comments that cite the old path: `world_pulse_read/durable.py`, `schema_skew_discovery.py`, `urgent_report.py`.
- **Controller:**
  - `BRIDGE_TARGETS`, `TARGET_ROLES`, `gpu2.py`'s fixed targets;
  - `GPU2_AUTHORITY` and `GPU2_AUTHORITY_URL` (pool is the only authority);
  - `GPU2_POOL_FENCE_STATE_PATH` → `GPU_POOL_FENCE_STATE_PATH` (same file, same volume);
  - the GPU1 flip (`lane_control.py`, `/v1/gpu-lanes/flip`, `/v1/gpu-slots/*`, `GPU_LANE_CONTROLLER_TOKEN`);
  - `swap.load`/`swap.unload` from the YAML schema.
- **Postgres migration** `services/orion-sql-db/manual_migration_drop_durable_capacity_v1.sql`:
  - `DROP TABLE durable_gateway_permits, durable_resource_demands, durable_resource_leases, durable_elastic_slot`;
  - **snapshot first** to `/tmp/gpu-pool-stage5-drop/` (`pg_dump -t ... | gzip`) under the backfill protocol.
    `durable_gateway_permits` is 169k rows / 127 MB, which is **over the 100k rows / 100 MB line**, so Juniper
    decides the snapshot form (Missing question 3);
  - `durable_admission_runs` and `durable_resource_events` stay.
- **Metric lock** (`config/metrics/metric_definitions.lock.json`): re-lock after main merges (standing rule). No
  locked metric reads the dropped tables (grep: `queue_contention` dropped the demands half in 4.6).
- **Unrelated fix carried along** (parent spec item 8): `resource_deferred` added to
  `services/orion-analytics/models/marts/dim_reverie_outcomes.sql` (it is still missing today).

## Decision 6: pool mode end state

- **End state:** `GPU_POOL_MODE=enforce` with **`GPU_POOL_ACTUATE_ROLES` deleted**. A role is actuated if and only if
  it has a `launch` block naming a reachable actuator. That is config, validated in CI, and not a second list that
  can drift from the YAML.
- **What enforce changes, from the code:**
  - `_observe_swap_seats` stops, so a swap seat is loaded only when the pool loaded it (or adopted it via `status`
    at boot);
  - operator `hold` is allowed.
- **Order.** Flip to enforce only after 5.3. Every swap seat must have a launch block before this, or it could never
  load again. The validator refuses `enforce` while any `swap` role lacks `launch` (`experiment` is exempt as
  `not_actuatable`, and operator-only).
- **Rollback:** `GPU_POOL_MODE=observe` + restart. It is harmless, because with every seat actuated, observe only
  re-enables the adoption shortcut.

## Telemetry and readers that move

| reader | change | gate |
| --- | --- | --- |
| world-model `gpu_contended` outcome (its prediction payload) | same code, cause is now `unavailable reason=deadline` or `serialized:diffusion` from the pool | no metric reads it by name (grep: only `orion-world-model/app`) |
| reverie `resource_deferred` terminal reason | one cause instead of two | analytics mart fix above; not a locked metric |
| pool `gpu_pool_events` | new queued reason `serialized:<role>`; `swap_*` gain `detail.profile` | pool-internal; no reducer (stage 6) |
| pool rpc_health `gpu_pool:world#gpu_pool_wait` | new hop label for class world (already in `EQUILIBRIUM_TRANSPORT_EXCLUDE_LABELS` via the `gpu_pool_wait` suffix) | excluded from transport baselines by design |
| world-model / thought rpc_health | `http:…:8124/capacity` / `durable-runs:8121/capacity` hops disappear | no reader keys on them (5.4 PR re-greps) |
| Hub biometrics lane labels | derived from pool state | cosmetic; UI test covers the mapping |
| `power_intent_settled` `gpu_index` | derived source | gate recorded above |
| field-digester `queue_contention` | unchanged (pool tables since 4.6) | none |

## Juniper's answers (2026-09-29)

These override the recommendations above.

1. **gpu2 27B seat limit: raise it to about 2.5 h** (`max_hold_sec: 9000` on `agent-gpu2`). Curiosity and self-sense holds may still land on gpu2. The cost is accepted: image generation can wait up to about 2.5 h for gpu2. Chat reclaiming lent gpu0 stays as is. Ships in 5.1.
2. **Pool events get fresh envelope correlation ids.** `turn_correlation_id` stays in the payload, which is what every join reads. That keeps GPU queue wait out of turn causal chains and out of the bus-synaptic baselines. Ships as its own small PR, alongside 5.1.
3. **Snapshot before the drop:** a gzipped `pg_dump` of the four dead tables to `/tmp/gpu-pool-stage5-drop/`, then the drop. Only on Juniper's go at that step (5.6).
4. **Experiment seat: deferred.** Not built in stage 5.

## Corrections from building 5.1 (PR #2408)

1. **Deploying any `launch` change means rebuilding circe's controller, not just pulling.** The controller reads the YAML from the host checkout but parses it with code baked into its image. A pull without a rebuild makes every action refuse with `config_unloadable:*`.
2. **`serialize_with` needs an explicit reservation.** A lease blocked only by the pair reserves the partner role, so younger work can't take it and starve the older lease. Implemented as scheduler rule Z1 plus a report-only `Serialized` decision (`queued`, `reason=serialized:<role>`).
3. **The `diffusion` launch_digest changes too**, not only agent-gpu2's.
4. **The new `ATLAS_` keys need `sync_local_env_from_example.py orion-llamacpp-host --all-keys`**, because the default sync skips that prefix.
5. **The experiment `not_actuatable` marker has no consumer before 5.7.** 5.1 ships only the validator exemption. Its operator lease still drains everything it evicts; that is unreachable while the pool is in observe mode, and gets fixed in 5.7.

## Corrections from building 5.2 (PR #2409)

1. **Loading a resident role directly (diffusion, fast2) is refused with `not_a_swap_seat`.** Only swap seats load. A direct load could land a resident on a card that a loaded seat still holds.
2. **`bridge_verb_unsupported` stays reachable until 5.6.** Added `bridge_cannot_set_profile`, so a profile sent to a bridged seat is refused, not silently ignored.
3. **`launch.drain` has no timeout key**, so the new path uses `GPU2_DRAIN_TIMEOUT_SEC`. 5.6 must rename it or make it a YAML key, and delete the now-dead `GPU2_MODEL_READY_TIMEOUT_SEC`, `GPU2_DIFFUSION_URL` and `GPU2_AGENT_URL`.
4. **"Ready" means a 2xx carrying `{"ready": true}` or `{"status": "ok"}`.** Every new service's ready endpoint must return one of those; a 503 while not ready is correct.
5. **Unloading a seat with a drain block drains it before the stop**, and un-drains it if the stop fails.
6. **5.3 behaviour changes on agent-gpu2:**
   - The ready wait goes from 600 s to 900 s (`launch.timeout_sec`).
   - Failure reasons gain a `:<role>` suffix; `burst_upstream_not_idle` becomes `upstream_not_idle:agent-gpu2`.
   - `GET /v1/gpu-slots/circe-gpu2/status` stops showing progress. The pool reads the bus results instead.
   - The pool must start sending `profile` for roles that have `launch.profiles`. Choosing a profile by vision, ctx or VRAM is not built.

## Corrections from building 5.7 (enforce)

1. **Emergency stop: the spec was silent, so 5.7 adds the smallest persisted one.** 5.6 removed the controller's
   enable switch and 5.7 removes `GPU_POOL_ACTUATE_ROLES`, which left no fast way to stop every load/unload. Added:
   control verbs `pause_actuation` / `resume_actuation` (Hub button, or `scripts/gpu_pool_pause.py`), persisted on
   every `gpu_pool_cards` row (`actuation_paused_at/_by`, migration `manual_migration_gpu_pool_v3_actuation_pause.sql`,
   required at boot). While paused nothing is sent to the actuator and no seat is drained (scheduler `frozen`);
   swap decisions are still published (`swap_requested reason=actuation_paused`). An action already in flight is not
   stopped (the actuator owns it); stopping the controller container is the documented second step for that.
   Not a mode and not an env key: one verb, visible in state, `/health` and two new events.
2. **`observe` is kept, as the rollback only.** Both modes actuate every seat with a launch block. observe differs in
   exactly three ways: liveness adoption for never-actuated seats, no boot/resume reconcile, operator holds refused
   (`hold_refused_observe_mode`, renamed from `hold_requires_swap_actuation`, which stopped being true). An unknown
   mode fails the boot. Decision 6's "observe only re-enables the adoption shortcut" is weaker than it reads: the
   shortcut skips any seat the pool has already acted on (`_pool_owns`), and agent-gpu2 has (generation 79), so for
   gpu2 observe adopts nothing -- rolling back to it trades the reconcile for no adoption at all.
3. **"Adopted via `status` at boot" did not exist before 5.7**; the boot only reconciled cards left mid-action.
   enforce now sends one read-only `status` per idle actuated seat at boot and on resume and adopts the answer
   (`swapped reason=adopted:<why>`); a half-done card or an action the pool never sent faults the card; no answer in
   90 s (or an actuator that cannot say whether something runs) keeps the persisted state. Until the answer that seat
   neither swaps nor drains (scheduler `frozen`); an answer that agrees leaves the card's last action record alone. The actuator re-publishes its last recorded
   result before answering; while a reconcile is open that stale row is ignored.
4. **Correction 5.1 #5 is fixed in two places:** operator leases on a class whose seat has no launch block are
   refused `not_actuatable:<role>` (every mode, at `acquire` and at the `hold` verb), and the scheduler treats a swap
   seat without a launch as frozen, so it never drains residents for one even if a lease got past the refusal.
5. **Deploy order gains sql-writer first**: `GpuPoolEventV1` (extra=forbid, literal event list) gains
   `actuation_paused` / `actuation_resumed`, and sql-writer validates every pool event. Hub goes last (its new verbs
   are refused by an old pool). circe needs nothing: no launch or config-model change, digests unchanged.
6. **The experiment seat's validator exemption stays**; the "validator refuses enforce while a swap role lacks launch"
   rule was already unconditional (every non-operator swap seat needs a launch since 5.1), so no mode-specific rule
   was added.

## Proposed schema / API changes

- `orion/gpu_pool/config.py`:
  - `RoleSpec.serialize_with: list[str]`;
  - `LaunchSpec.profile_var`, `LaunchSpec.profiles`;
  - `cuda_env` meaning change + interpolation gate;
  - `SwapSpec.load/unload` removed (5.6);
  - `enforce` requires launch on every non-operator swap role.
- `orion/schemas/gpu_pool.py`:
  - `GpuActuateV1.profile` is now produced (the field exists);
  - `GpuActuateResultV1.observed` is unchanged;
  - no new models.
  - `GpuPoolEventV1.reason` is free text (the `serialized:` value needs no schema change).
- **Controller refusals** gain `profile_not_allowed` and `no_launch_block`. `not_a_bridge_role`,
  `bridge_verb_unsupported` and `profile_unsupported` are removed.
- **Deleted:**
  - `orion/schemas/gpu_slot.py`;
  - `resource_admission` `CapacityAcquireV1`, `CapacityPermitV1`, `CapacityTokenV1`, `Capacity*ResultV1` and their
    `registry.py` entries (plus `schema_skew_discovery.py` note);
  - HTTP `POST /capacity/{acquire,renew,release}`, `GET /capacity`, `/v1/gpu-slots/*`, `/v1/gpu-lanes/flip`.
- **Channels:** none added or removed.

## Env/config changes

- **Added:** `ATLAS_AGENT_BURST_CUDA_VISIBLE_DEVICES`, `ATLAS_AGENT_BURST_PROFILE_NAME`
  (`services/orion-llamacpp-host/.env_example`).
- **Renamed:** `GPU2_POOL_FENCE_STATE_PATH` → `GPU_POOL_FENCE_STATE_PATH`.
- **Deleted:**
  - `GPU_POOL_ACTUATE_ROLES`, `GPU_POOL_VISUAL_ACTIVITY_URL`;
  - `GPU2_AUTHORITY`, `GPU2_AUTHORITY_URL`, `GPU2_ENABLED` (→ nothing: a role without a launch block is simply not
    actuated), `GPU_LANE_CONTROLLER_TOKEN`;
  - `WM_GPU2_CAPACITY_*`, `ORION_VISUAL_CHAIN_GPU2_CAPACITY_*`, `ORION_VISUAL_ELASTIC_*`, `DURABLE_RUNS_CAPACITY_*`;
  - `GPU_LANE_MAP_ATHENA_JSON`, `GPU_LANE_MAP_CIRCE_JSON`, `DIFFUSION_POWER_INTENT_GPU_INDEX`.
- **Changed:** `GPU_POOL_MODE=enforce` (5.7).
- Every `.env_example` change is synced with `scripts/sync_local_env_from_example.py` in its PR, **and** on circe
  for circe-side services (the sync script writes the primary checkout on the host it runs on).

## Files likely to touch

- `config/gpu_pool.yaml`, `orion/gpu_pool/{config,scheduler}.py`, `scripts/check_gpu_pool_config.py`,
  `services/orion-gpu-pool/app/{runtime,settings}.py` + tests/evals
- `services/orion-gpu-lane-controller/app/{actuator_bus,pool_fence,main,settings}.py`, new `launch_exec.py`;
  delete `gpu2.py`, `lane_control.py`; README, `.env_example`, tests
- `services/orion-llamacpp-host/{docker-compose.atlas-workers.yml,.env_example}`
- `services/orion-world-model/app/{main,settings}.py`, `.env_example`, tests
- `services/orion-thought/app/{visual_chain,settings}.py`, `.env_example`, tests
- `services/orion-durable-runs/app/{main,settings,admission_runtime}.py`, `evals/gateway_capacity.py`,
  `.github/workflows/orion-durable-runs-tests.yml`
- `orion/durable_admission/*` → `orion/durable_runs/registry_store.py`; `orion/schemas/{resource_admission,gpu_slot}.py`,
  `orion/schemas/registry.py`
- `services/orion-hub/{app/settings.py,scripts/biometrics_preview_routes.py,.env_example}` + test; Hub pool panel
  (experiment `not_actuatable`)
- `services/orion-diffusion-host/app/{settings,main}.py`, `.env_example`, test; `services/orion-world-model/app/gpu.py` comment
- `services/orion-sql-db/manual_migration_drop_durable_capacity_v1.sql`,
  `services/orion-analytics/models/marts/dim_reverie_outcomes.sql`
- `config/metrics/metric_definitions.lock.json` (re-lock)

## Non-goals

- athena's GPUs; affect as a tenant; splitting one request across cards (all unchanged from the parent spec).
- Building the experiment seat (deferred, Decision 3).
- Gateway per-call telemetry, grammar reducers, the `/routes` removal, the port gate (stage 6).
- Choosing *policy* for hold limits and lent-card placement (Missing question 1). Stage 5 builds nothing that
  presumes the answer.
- Root-causing the 09-24 CUDA-busy failures (a check is recorded in 5.4; the fix, if any, follows).

## Acceptance checks (live, observable)

1. **Generic load (5.3):**
   - with a hold queued ≥ 1200 s, the pool emits `swap_started` → `swapped` for `agent-gpu2`;
   - the controller log shows `launch_exec load role=agent-gpu2 service=atlas-agent-burst env=ATLAS_AGENT_BURST_CUDA_VISIBLE_DEVICES=2`;
   - `nvidia-smi` shows the 27B on index 2;
   - no log line contains `gpu2.transition` or `BRIDGE`.
2. **Generic unload + reclaim:**
   - a reverie-visual diffusion hold queued while the 27B is loaded → `swapped reason=owner_reclaim`;
   - diffusion is back on index 2 (`CUDA_VISIBLE_DEVICES` set by the actuator);
   - the hold is granted and the image is stored.
3. **Failed load:**
   - force a bad profile or a readiness timeout → `swap_failed restored=true`, and diffusion is running;
   - with a stopped diffusion image → `fault` in the Hub panel.
4. **Profile gate:**
   - a hand-crafted `GpuActuateV1` with a profile outside `launch.profiles` → `actuate_refused reason=profile_not_allowed`;
   - CI fails on a compose literal device for any launch role (a test with `CUDA_VISIBLE_DEVICES_OVERRIDE=2` restored).
5. **World (5.4):**
   - a world-model task produces a `gpu_pool_leases` row `work_class='world'`;
   - while a diffusion hold is active, a world task returns `gpu_contended` within ~2 s, and the pool event reads
     `serialized:diffusion`;
   - `durable_gateway_permits` gets zero new rows for 24 h.
6. **Diffusion (5.4):**
   - reverie-visual runs complete with images;
   - `durable_gateway_permits lane=diffusion` gets zero new rows;
   - `resource_deferred` count per day is no higher than the prior 7-day average.
7. **Labels (5.5):**
   - Hub biometrics shows circe gpu2 as `agent-gpu2` or `diffusion` matching `/v1/pool`, not `unassigned`;
   - diffusion's last 20 power intents carry `gpu_index=2`, identical before and after deploy.
8. **Delete (5.6):**
   - `grep -r durable_admission` and `capacity_client` return nothing;
   - the four tables are gone and the snapshot exists in `/tmp/gpu-pool-stage5-drop/`;
   - Hub curiosity run views still render from `durable_admission_runs`/`durable_resource_events`.
9. **Enforce (5.7):**
   - `/v1/pool` `mode=enforce`;
   - a pool restart with the 27B loaded adopts it via `status` (one `status` action, no reload);
   - `GPU_POOL_ACTUATE_ROLES` is absent from every env file.

## Recommended next patch: stage 5 as ordered PRs

| PR | what | deploy | behaviour change |
| --- | --- | --- | --- |
| **5.1 config + contracts** | `serialize_with` (config + scheduler + tests); `launch.profile_var`/`profiles` in config (not yet produced); `cuda_env` interpolation gate; agent-burst compose `${ATLAS_AGENT_BURST_*}` + `.env_example` keys; class `world` → `on_unavailable: wait`; experiment bridge verbs removed, marked `not_actuatable` | athena pool + **circe pull** (digest changes); circe `.env` synced | none (world has no leases yet; agent-burst resolves to the same device and profile) |
| **5.2 generic actuator** | `launch_exec.py` from `gpu2.py`'s steps; `resolve()` returns a `LaunchPlan` for roles without bridge verbs; profile allow-list; generic `observed`; bridge still honoured | circe controller | none (agent-gpu2 still has bridge verbs) |
| **5.3 cutover gpu2 to generic** | YAML: remove `swap.load/unload` from `agent-gpu2`. Pool starts sending `profile` for roles with `profiles` | athena + circe pull same commit, then restart the pool | **the controller uses the generic path.** Revert = revert the YAML |
| **5.4 world + diffusion leases** | world-model lease via `orion.gpu_pool.client`; thought drops the permit and the dead elastic pre-check; `visual_baseline` guard deleted; mart fix | pool (guard removal) → thought → world-model (circe) | **yes: permits stop** |
| **5.5 labels** | Hub lane labels from pool state; diffusion-host derives `gpu_index`; delete `GPU_LANE_MAP_*`, `DIFFUSION_POWER_INTENT_GPU_INDEX` | Hub, diffusion-host (via the actuator on next load, or `up -d`) | labels only |
| **5.6 delete** | capacity code, routes, schemas, evals, workflows; `store.py` moved; bridge + GPU1 flip + `gpu_slot.py` + `GPU2_*` keys; migration (after Juniper's go on the snapshot) | durable-runs, controller, then the migration | none (dead paths) |
| **5.7 enforce** | `GPU_POOL_MODE=enforce`; delete `GPU_POOL_ACTUATE_ROLES` (actuation = launch block present); validator rule | pool env flip + restart | observe shortcut off; operator holds possible |

**Cutover runbook outline** (5.3 and 5.4, the two behaviour-changing steps):

1. **Before 5.3:**
   - confirm gpu2 `swap_state=idle`, no `actuation` in flight (`GET :8127/v1/pool`), and no `agent-gpu2` hold
     granted, or accept that it continues on the loaded seat (a restart does not unload it);
   - `git -C /mnt/scripts/Orion-Sapienform pull` on circe and athena **to the same commit**;
   - `scripts/safe_docker_build.sh orion-gpu-lane-controller up -d` on circe;
   - restart the pool on athena;
   - check `status` adoption (one `status` action, `observed.agent-gpu2=running` or `diffusion=running`, no
     transition).
2. **Verify 5.3:** wait for (or force, with a queued hold) one load and one unload through the generic path, per
   acceptance 1–2.
3. **5.4:**
   - deploy the pool (guard deleted) → thought → world-model on circe;
   - check `durable_gateway_permits` for new rows every hour for 24 h (target 0);
   - check reverie-visual images keep landing.
4. **Only then 5.6**, with Juniper's go on the snapshot and the drop.
5. **Rollback points:**
   - 5.3: revert the YAML, pull on both hosts, restart the pool (the bridge is still in the controller until 5.6);
   - 5.4: redeploy the previous thought/world-model images (`/capacity` still exists until 5.6).
