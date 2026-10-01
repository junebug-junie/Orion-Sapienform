# GPU pool stage 6: per-call telemetry, the reducer verdict, and lockdown

Status: design, 2026-09-30. Final stage of `2026-09-24-gpu-pool-design.md` (build order item 6,
"Transport and telemetry" items 3 and 5, "Delete list"). Stages 1-5 are live. Nothing here is
deployed.

## Arsonist summary

- **Orion still can't tell "the GPU is slow" from "the line is long".** The gateway already
  reports its own call outcomes once a minute (#2327, live). But its latency clock starts
  *before* the pool lease, so the numbers it publishes are queue wait plus model time mixed
  together. They are also per machine (`circe`), not per role. And the calls Claude Code makes
  through the gateway's HTTP passthrough are not counted at all.
- **The fix is to extend that existing report, not to build a second one.** The spec asked for
  new bus-health hops named `llm:<role>#call`. This design drops them. Their only consumer would
  be the transport baseline, which would then learn "the model took a long time" as if it were
  "the message was slow to arrive". Refusal grammar also already exists: the pool saying "no" is
  already counted as a refusal (`gpu_pool_unavailable` in `REFUSAL_CLASSES`).
- **No grammar reducer for pool lease events.** Every candidate failed the metric gate. Each one
  was either already covered by an existing signal, too rare to carry signal, or had no consumer
  asking a question (gate record below). The pool's events stay as inspectable history in
  `gpu_pool_events` and `grammar_events`.
- **Lockdown is real work, not a checkbox.** The gateway's `/routes` still gets about 140
  requests an hour, from the Hub and harness-governor. Dead env keys are still in live `.env`
  files on both hosts, and three PR reports have listed them by hand. The "port gate" was never
  defined. Here it becomes a CI check that fails on any direct circe worker port reference
  outside the pool, the gateway and the actuator, plus an optional circe firewall rule that
  Juniper decides on.
- **A new defect found while designing this:** when a gpu2 swap is blocked by cooldown, the pool
  writes a `swap_requested` event **about once a second** for the whole cooldown. Today that was
  628 rows from 08:45 to 08:55 UTC, and there have been similar bursts on 3 of the last 5 days.
  That is spam in the history and in the grammar table. It is the first PR.

## Current architecture (verified live, 2026-09-30 ~09:30 UTC)

- **Pool:** `GET :8127/health` returns `mode: enforce`, `actuation.seats: ["agent-gpu2"]`.
  `/v1/pool` roles carry `url`, `status`, `profile_name`, `model_file`, `ctx_per_slot` and
  `vision`, which is a superset of what `/routes` readers use.
- **Pool history** (`gpu_pool_events`, since 2026-09-24): 52,735 granted, 358 recalled,
  129 aborted (all `recall_grace_exceeded`), 49 unavailable, 6 dead-lettered, 4 swap_failed,
  2,820 swap_requested but only 84 swap_started.
- **The swap_requested storm:** 2026-09-30 08:45:27 to 08:55:26, 628 rows, `reason=cooldown`,
  `detail.blocked=true`, one per scheduler tick (about 1 s). It followed
  `swap_failed upstream_not_idle:agent-gpu2` at 08:45:26. Daily counts: 09-26 1,320,
  09-28 849, 09-30 629. The 1,492 `gpu_pool.lease:agent-gpu2` grammar atoms come from the same
  source.
- **Pool grammar** (`grammar_events` with `trace_id LIKE 'gpu_pool.lease:%'`): 2,045 rows since
  09-27. Nothing reads them except storage.
- **The gateway inference lane (#2327) is on live:** container env `LLM_GATEWAY_GRAMMAR_ENABLED=true`.
  `substrate_llm_inference_projection` updates every minute, for example
  `llm_node:circe calls=3 served=3 latency_p50_ms=9919 latency_p95_ms=23919 inference_failure_pressure=0.0`.
  - The latency clock is `dispatch_started`, taken before `_dispatch_chat`, so it **includes pool
    wait** (`services/orion-llm-gateway/app/main.py:507-527`). The lease is acquired inside
    `_dispatch_chat` (`main.py:384`).
  - Only the bus path `handle_chat` records. The HTTP passthroughs (`/v1/messages`,
    `/v1/chat/completions`, both via `passthrough_proxy.py`) record nothing.
  - `latency_p50_ms`/`latency_p95_ms` have no reader beyond the projection model
    (`orion/schemas/llm_inference_projection.py:93-94`, `orion/substrate/llm_inference_loop/extract.py:163-164`).
- **Why separating wait from model time matters now:** in the last 24 h, 172 of 1,172 chat-class
  grants to cortex-exec (interactive priority) waited more than 60 s, all on role `chat`. The
  p50 wait was 1.9 s, p95 128 s, max 294 s. These were spread across the day, not one incident.
  Whether the chat worker is slow per call or just has a long line is not answerable from
  current data. That is the question item 1 exists to answer. Cause: UNVERIFIED.
- **Transport baseline** (Redis `equilibrium:transport_baseline_state:v1`): `gpu_pool:<class>#gpu_pool_wait`
  keys are learning, for example `gpu_pool:metacog` with fast_count 2226. Every FCC key
  (`fcc:agent`, `fcc:chat`, `fcc:route:*`) still has `last_eval_ts=0`, so it has never
  evaluated. Old `fcc:<model>` keys and `orion-durable-runs|http:llm-gateway:8210/routes`
  (last seen about 4.3 days ago) are frozen leftovers.
- **Who reads `/routes`** (gateway access log, 24 h): 3,359 from `172.18.0.1` (the host network,
  so the Hub: `llm_gateway_client.fetch_routes` for the route picker and vision flag, plus
  `orion/situational/context.py::_fetch_runtime_context` for the unleased "which model am I"
  line) and 68 from `172.18.0.68` (harness-governor: `orion/harness/fcc_motor.py` pre-turn
  probe, about line 544).
  - No calls from cortex-exec in 24 h.
  - context-exec (`llm_profile_resolver.fetch_route_status_map`) has no running container.
  - `services/orion-mind/scripts/verify_mind_llm_e2e.sh` and `scripts/smoke_llm_gateway_routes.py`
    are manual scripts.
- **Old keys still live in the gateway container:** `LLM_GATEWAY_ROUTE_TABLE_JSON` (a full
  pre-pool route table with circe URLs), `LLM_GATEWAY_CAPACITY_*`, `LLM_GATEWAY_LEASE_VALIDATION_*`,
  `LLM_GATEWAY_BACKGROUND_*`, `LLM_GATEWAY_UPSTREAM_MAX_INFLIGHT` and `LLM_ROUTE_*_URL/SERVED_BY`.
  None of them is in `.env_example`, and the code reads none of them except `LLM_ROUTE_DEFAULT`
  and `LLM_ROUTE_HEALTH_TIMEOUT_SEC`. They come from the stale local `.env`.
  - **Not dead:** `LLM_LANE_ROUTING_ENABLED=true` and `LLM_LANE_DEFAULT=chat` are live and read
    by `llm_backend.plan_llm_chat` -> `lane_routes.resolve_llm_lane_route`. They are on the
    spec's delete list, but deleting them changes routing.
- **Dead keys in local `.env` but absent from `.env_example`** (read-only diff, 2026-09-30):
  - athena gateway: 27 keys (above, plus `LLM_ALLOW_BACKGROUND_TO_CHAT_FALLBACK` and
    `ORION_LLM_{LLAMA_COLA,OLLAMA,VLLM}_URL`).
  - durable-runs: 21 keys (`DURABLE_RUNS_{ADMISSION_SHADOW,CAPACITY_ENABLED,ELASTIC_*,LANE_POLICY_JSON,WIDENING_*}`).
  - hub: 3; thought: 10; world-model: 6; cortex-exec: 1 (`CORTEX_EXEC_ADMISSION_CUE_TIMEOUT_SEC`);
    gpu-pool: 1.
  - circe: lane-controller 9 (`GPU2_*`, `GPU_LANE_CONTROLLER_TOKEN`), world-model 6
    (`WM_GPU2_CAPACITY_*`), diffusion-host 1.
  - All inert (`extra="ignore"`). The only circe worker port literals left in live `.env` files
    are inside these dead keys.
- **Tables:** the four legacy admission tables were dropped in stage 5.6. `durable_admission_runs`
  (385 rows, written 09:33 today) and `durable_resource_events` (written 09:29 today, read by
  curiosity run-story, Hub `curiosity_run_store.py` and `urgent_report.py`) are **live**, not
  leftovers. Nothing to drop.
- **Workflows:** `orion-durable-runs-tests.yml` still has a job named "Gateway — shared capacity
  and transport lifecycle". It lists `orion/schemas/resource_admission.py` in its path filters;
  that file still exists and holds the live `ResourceEventV1`. `gpu-lane-controller-tests.yml`
  is the renamed gpu2 workflow and is live. No dead workflow files remain.
- **Stale docs:** `docs/architecture/gpu2-elastic-admission.md`, `docs/architecture/gpu2-elastic-evidence.md`,
  `docs/runbooks/gpu2-elastic-admission.md` and `docs/architecture/durable-resource-admission.md`
  describe deleted mechanisms.
- **Port gate:** nothing enforces the spec's acceptance check 6 today. circe's llama.cpp workers
  bind `0.0.0.0:8011-8016`. The only in-repo circe worker port literal outside the pool config is
  `ORION_DIFFUSION_HOST_BASE_URL=http://100.112.254.99:8014` (thought). That is the diffusion
  host, which runs under a pool lease, not an LLM worker.
- **Pool boot self-heal:** PR #2438 (`fix/gpu-pool-boot-schema-self-heal`) is open. It is
  independent of this design.

## Decisions

### 1. Gateway per-call telemetry: extend the #2327 lane, drop `llm:<role>#call`

What changes, in plain terms: every LLM call gets two clocks instead of one. How long it waited
for a GPU, and how long the model took once it had one. Both are reported per granted role
(chat, agent, metacog, fast, agent-gpu2), and for the HTTP passthroughs too.

- `_run_on_grant` (bus path) and `passthrough_proxy` (HTTP path) stamp `granted_at` when the
  lease context yields and `replied_at` when the upstream reply (or the stream end) returns.
  - `model_ms = replied_at - granted_at`.
  - `wait_ms` is measured by the gateway on its own clock, from `lease_for_route` entry to grant.
    The pool's `waited_ms` (`gpu_pool_events`, and the pool-side `gpu_pool_wait` hop at
    `services/orion-gpu-pool/app/runtime.py:1404`) is the cross-check. It is not the source,
    because the gateway would need a second read to get it.
  - A call that never got a grant records `wait_ms` and outcome `gpu_pool_unavailable`, with no
    `model_ms`.
- If llama.cpp's reply carries `timings.predicted_per_second`, record it as `decode_tps`. This is
  a per-token speed, so it does not depend on how long the answer was. `model_ms` alone mixes a
  0.3 s classifier with a 120 s agent turn on the same role.
- `InferenceWindowRecorder` buckets by **granted role** as well as node. The per-minute trace
  gains `role=<r> model_p50_ms model_p95_ms wait_p50_ms wait_p95_ms decode_tps_p50 calls failed`.
- **Retired, not kept alongside:** `latency_p50_ms`/`latency_p95_ms`. They measure wait plus
  model time, and nothing reads them. Kill means kill (CLAUDE.md 0A).
- **Dropped from the spec:** `llm:<role>#call` rpc_health hops. See gate record G1.
- **Refusal grammar:** already delivered. `classify_outcome` maps `gpu_pool_unavailable`,
  `route_not_in_gpu_pool` and `gpu_pool_recalled` to refusals
  (`orion/schemas/llm_inference_projection.py:43-58`). Nothing new is needed.
- **No field wiring in this stage.** The per-role numbers land in the projection (a debug
  surface) first. Wiring them into `capability:llm_inference` only happens after 48 h of live
  data passes gate step 4 (G2). That check is its own checkpoint, not a PR.

### 2. Grammar reducers for `gpu_pool.lease:` events: none

Every candidate failed. Record below. The spec's "a reducer only after the gate" becomes "no
reducer". The events stay as history (`gpu_pool_events`, Hub panel walker) and as grammar rows.

### 3. Lockdown

- **`/routes` removed.** Readers move first, then the endpoint is deleted.
  - Hub route catalog, vision flag and the situational runtime line read `orion:gpu_pool:state`
    over the bus RPC the Hub already uses for the panel (`#gpu_pool_panel`).
  - `fcc_motor`'s pre-turn probe reads the same state RPC (role -> `model_file`, `ctx_per_slot`).
  - context-exec's `fetch_route_status_map` is deleted. The service is not running, and its
    fallback treats "unreachable" as "all available" anyway.
  - The manual scripts are repointed or deleted.
  - Deleting the endpoint also deletes the compatibility generator in `pool_placement.py:419-520`
    and the `LLM_ROUTE_HEALTH_TIMEOUT_SEC` setting.
- **The port gate, defined:** "only the pool's granted URL reaches a llama.cpp worker". Two
  layers:
  1. **CI (deterministic, required):** `scripts/check_circe_worker_refs.py` fails on any circe
     host:port or worker model-file literal outside an allowlist: `config/gpu_pool.yaml`, the
     llamacpp-host compose files, `orion/gpu_pool/**`, `services/orion-gpu-pool/**`, gateway
     dispatch (`pool_placement.py`, `passthrough_proxy.py`) and the lane-controller actuator.
     Every allowlist entry must match something, so the list can't rot (same rule as
     `check_chat_route_poachers.py`). Comments are skipped. The diffusion-host URL goes on the
     allowlist with its reason, because diffusion is leased.
  2. **Runtime (optional, Juniper's call):** a circe firewall rule allowing 8011-8016 only from
     athena's tailnet IP (100.92.216.81). This stops other hosts. It **cannot** tell the gateway
     from any other athena container, because all of them share one source IP. So it is
     defence in depth, not proof. It needs sudo on circe, so the commands are printed, not run.
- **Env keys:** stop hand-listing them. Add `scripts/report_dead_env_keys.py`: a read-only
  per-service diff of local `.env` keys that are absent from `.env_example` and not read by any
  `settings.py`/compose file. `--apply` writes `.env.bak.<ts>` and then removes them. Run it on
  athena and circe after the gateway `/routes` PR deploys (the rollback image still reads some).
  `LLM_LANE_*` is **excluded** until the lane census in PR 6.4 shows no caller sends
  `options.lane`.
- **Tables:** none to drop (both remaining legacy-named tables are live).
- **Workflows:** rename the stale "Gateway — shared capacity" job. No workflow files to delete.
- **Docs:** add a "superseded by the GPU pool" header on the four stale architecture/runbook docs
  (not deleted: they hold incident evidence), and mark the parent spec's stage 6 done.

### 4. Loose ends

- **swap_requested storm:** emit `swap_requested` only when the blocked state *changes*
  (keyed on `(role, action, reason, guard_state)`), not on every tick. Watch out for the memory
  lesson "debounce keyed on an alternating label fires every tick". The key must not include
  anything that flips per tick.
- **FCC transport baseline never evaluates, and would include queue wait:** exclude the `fcc:`
  prefix from the equilibrium transport baseline, the same way `rpc_delivery.py` already does,
  and delete the stale `fcc:*` state keys. FCC turn time is work time, not transport.
  Per-role model time (item 1) now covers the passthrough calls FCC makes. This retires a
  metric, so it needs Juniper's yes (Q2).
- **`rpc_delivery.py`:** no change. It only counts `orion:` hops, so it already ignores
  `gpu_pool:` and `fcc:`. It would also ignore the dropped `llm:` hops.
- **Stale transport-baseline keys** (`http:llm-gateway:8210/routes`, `fcc:<model>`): cleared in
  the same PR as the `fcc:` exclusion.
- **Pool boot self-heal:** PR #2438. It must deploy before any pool image in this sequence
  (PR 6.1). No PR here adds a pool migration.
- **Merged pool worktrees:** about 20 `gpu-pool-*` worktrees are merged. `make prune-merged-worktrees YES=1`
  runs after this arc closes. That is housekeeping, not a PR.

## Metric quality gate record

| # | Candidate | Verdict | Why |
|---|---|---|---|
| G1 | rpc_health hop `llm:<role>#call` (model latency) | **Fail (3, theory)** | Its consumer is the equilibrium transport baseline, whose anchor is "one latency population per key = message delivery". Model time is work, and it spans two orders of magnitude on one role depending on output length, so a baseline would read ordinary long answers as regime shifts. `rpc_delivery` ignores non-`orion:` hops, so it would have no other consumer. Dropped. |
| G2 | per-role `model_ms` / `decode_tps` / `wait_ms` in the inference lane | **Provisional: record only** | (1) Provenance: `_run_on_grant` / `passthrough_proxy` grant->reply; llama.cpp `timings`. (2) Independence: `wait_ms` is the same fact as `gpu_pool_wait` (kept only as a denominator for the "line vs worker" split, never wired separately); `model_ms` and `wait_ms` are disjoint intervals by construction; `decode_tps` is not derived from `gpu_pressure` (a GPU-util reading), `reasoning_load` (a run weight) or `inference_failure_pressure` (outcomes). (3) Theory: decode throughput per slot is the llama.cpp-reported serving speed, and a drop at fixed model/slot means the worker degraded (thermal, a co-tenant, ctx growth). (4) Live data: **not yet possible**, because the metric does not exist. The 48 h check before any field wiring must show `decode_tps` returning to its per-role rest band at idle and not flat-lining on sparse roles. (5) Existing: #2327's lane is extended, not duplicated. (6) Reversible: projection-only, no field channel, no training default. |
| G3 | pool refusal share (`unavailable`/`dead_lettered` over requests) | **Fail (2, independence)** | For LLM work it is the same fact the gateway lane already counts as `refused` (`gpu_pool_unavailable` in `REFUSAL_CLASSES`). Non-LLM: 49 unavailable in 6 days is too sparse for a rate. |
| G4 | actuator fault rate (`swap_failed`, `aborted`) | **Fail (4, degenerate)** | 4 swap_failed in 6 days, so it reads 0 almost always. The panel already shows each one. An alert on an individual `swap_failed` would be the right shape, not a reducer. Not proposed here. |
| G5 | owner-starvation seconds (`recall_grace_exceeded`) | **Fail (no consumer)** | 129 events, a real operator fact, but no field or cognition consumer asks "was chat starved". It stays a panel/eval number (the spec's eval already reports it). |
| G6 | queue depth / oldest wait | **Fail (2, redundant)** | Already `gpu_pool_waiting` inside `queue_contention_score` (`orion/field/queue_contention.py:70`). |

## Juniper's answers (2026-09-30)

1. **Q1 (telemetry): yes.** Per-role waiting and model-time clocks plus tokens/sec go into the existing inference report (#2327). The `llm:<role>#call` rpc_health hops are dropped.
2. **Q2 (FCC transport-baseline key): deferred, decide with an investigation.** 6.7 moves after 6.2's 48 h checkpoint. The two-clock data is the evidence for whether the FCC key measures anything real. No change before then.
3. **Q3 (circe firewall): yes.** It needs sudo, which agents may not run (CLAUDE.md section 8), so 6.6 prints the exact commands for Juniper to run.

## Missing questions (Juniper only)

1. **Q1.** Is dropping the spec's `llm:<role>#call` rpc_health hops in favour of per-role numbers
   in the existing inference lane OK? (Recommended: yes, see G1.)
2. **Q2.** May the FCC transport-baseline key be retired (`fcc:` excluded from equilibrium's
   baseline)? It has never evaluated once, and it would count queue wait as transport.
   (Recommended: yes.)
3. **Q3.** Do you want the circe firewall rule (port gate layer 2), knowing it only blocks
   non-athena hosts? (Recommended: yes, cheap; commands printed for you to run with sudo.)

## Proposed schema / API changes

- `orion/schemas/llm_inference_projection.py`:
  - `LlmInferenceNodeStateV1` removes `latency_p50_ms`/`latency_p95_ms` and adds
    `by_role: dict[str, LlmInferenceRoleStateV1]`.
  - New `LlmInferenceRoleStateV1` with `calls`, `served`, `failed`, `refused`,
    `model_p50_ms`, `model_p95_ms`, `wait_p50_ms`, `wait_p95_ms`, `decode_tps_p50`
    (all optional), `extra="forbid"`.
  - Because the model is `forbid`, the rollout is consumer first: substrate-runtime (reducer)
    deploys before the gateway.
  - The projection row is overwritten each minute, so no migration is needed.
- Gateway grammar trace `llm_gateway.inference:` gains `role=` kv atoms. The trace prefix is
  unchanged and the channel is already cataloged.
- Gateway HTTP: `GET /routes` removed (PR 6.5). No other API change.
- Pool: `swap_requested` emission becomes edge-triggered. The event shape is unchanged.
- Equilibrium: `fcc:` prefix excluded from the transport baseline (if Q2 = yes).
- No new bus channels, no registry entries, no signal-registry entries, no tables.

## Files likely to touch

- `services/orion-gpu-pool/app/` (scheduler swap emission), plus tests.
- `services/orion-llm-gateway/app/{main.py,passthrough_proxy.py,grammar_emit.py,pool_placement.py,settings.py}`,
  `.env_example`, README, tests.
- `orion/schemas/llm_inference_projection.py`, `orion/substrate/llm_inference_loop/{extract,reduce}.py`
  and tests.
- `services/orion-hub/scripts/{llm_gateway_client.py,api_routes.py}`, `static/js/app.js`,
  `static/js/chat-attachments.js`, tests.
- `orion/situational/context.py`, `orion/harness/fcc_motor.py`, `orion/fcc/context_budget.py`
  (docstring).
- `services/orion-context-exec/app/llm_profile_resolver.py`.
- `services/orion-mind/scripts/verify_mind_llm_e2e.sh`, `scripts/smoke_llm_gateway_routes.py`,
  `scripts/analysis/record_lane_occupancy.py`.
- `orion/metacog/transport_baseline.py` or `services/orion-equilibrium-service/app/settings.py` (`fcc:` prefix).
- New: `scripts/check_circe_worker_refs.py`, `scripts/report_dead_env_keys.py`, and a CI step in
  `orion-gpu-pool-tests.yml`.
- `.github/workflows/orion-durable-runs-tests.yml` (job name), the four stale docs, and the
  parent spec status line.

## Non-goals

- New field channels or cognition wiring from pool or inference telemetry (after G2's live check,
  separately).
- Diagnosing the 60 s+ interactive chat waits. Item 1 produces the data needed to answer that;
  it does not answer it.
- Deleting `LLM_LANE_*` / `lane_routes.py` without the census.
- Renaming `ORION_ADMISSION_TEST_DSN`, `durable_admission_runs` or `resource_admission.py`
  (live names; churn with no behaviour change).
- Legacy backends (`ORION_LLM_DEFAULT_BACKEND=llama-cola`, vllm/ollama URLs). They are outside
  the pool, and the dead-key tool will list them for a separate decision.
- athena GPUs and affect as a tenant (parent spec non-goals).

## Acceptance checks (live, observable)

1. **Swap storm gone:** after a cooldown-blocked swap,
   `select count(*) from gpu_pool_events where event='swap_requested' and reason='cooldown' and created_at > <deploy>`
   grows by at most a few rows per blocked episode, not about 600. Grammar atoms under
   `gpu_pool.lease:agent-gpu2` stop growing between swaps.
2. **Two clocks:** `substrate_llm_inference_projection.projection_json->'nodes'->'llm_node:circe'->'by_role'`
   has `chat`, `metacog` and `fast` entries, each with `wait_p50_ms` and `model_p50_ms`.
   `latency_p50_ms` is absent. For one correlation id, `gpu_pool_events.waited_ms` for the grant
   matches the call's recorded wait to within 1 s.
3. **Passthrough counted:** an FCC turn raises `by_role.<granted role>.calls` in the window it ran.
4. **`/routes` dead:** 24 h of gateway logs with zero `GET /routes` before PR 6.5 merges. After
   it deploys, `curl :8210/routes` returns 404. Hub route picker, attach-image button and the
   "which model am I" line still render. Checked in the browser, not just by the endpoint.
5. **Port gate:** `python scripts/check_circe_worker_refs.py` exits 0 on main and 1 on a planted
   `http://100.112.254.99:8011` in a service `settings.py` (test). If Q3 = yes,
   `curl circe:8011/health` from a non-athena tailnet host times out.
6. **Env:** `scripts/report_dead_env_keys.py` lists zero dead keys for the pool-related services
   on both hosts after `--apply`. `docker exec orion-llm-gateway env | grep ROUTE_TABLE` is empty
   after a restart.
7. **FCC (if Q2):** `equilibrium:transport_baseline_state:v1` has no `fcc:*` keys 1 h after the
   deploy.

## Recommended next patch: stage 6 as independently deployable PRs

Deploy order is top to bottom. Each PR is its own branch off main.

| PR | What | Services / deploy | Migration / circe |
|---|---|---|---|
| **pre** | #2438 pool boot schema self-heal (already open) | orion-gpu-pool | Must be live **before 6.1**. The pool never deploys ahead of its schema. |
| **6.1** | `fix(gpu-pool)`: edge-trigger blocked `swap_requested` + a regression test replaying the 08:45 sequence | orion-gpu-pool | none |
| **6.2** | `feat(llm-inference)`: per-role wait/model/decode_tps in the lane; retire `latency_p*`; passthrough recording. **Consumer first:** substrate-runtime, then gateway, in one PR with a stated deploy order | orion-substrate-runtime, then orion-llm-gateway | none |
| *(checkpoint)* | 48 h of live data, G2 steps 4-6, recorded in a follow-up note. No code. Wiring is proposed only if it passes | none | none |
| **6.3** | `refactor`: move `/routes` readers to pool state (Hub catalog/vision/situational, fcc_motor, context-exec resolver, scripts) | orion-hub, orion-harness-governor, orion-cortex-exec (shared `orion/situational`) | none |
| **6.4** | `chore(llm-gateway)`: lane census (log `options.lane` senders for 24 h, read-only). If none: delete `lane_routes.py` + `LLM_LANE_*`. Otherwise keep and document | orion-llm-gateway | none |
| **6.5** | `chore(llm-gateway)`: delete `/routes` + compatibility generator + `LLM_ROUTE_HEALTH_TIMEOUT_SEC`. Merge only after acceptance 4's 24 h of zero reads | orion-llm-gateway | none |
| **6.6** | `chore`: `check_circe_worker_refs.py` + CI step; `report_dead_env_keys.py`; run `--apply` on athena and **circe**; workflow job rename; superseded headers; parent spec marked done | none (CI + local `.env`) | **circe** `.env` edit (operator, via the script); firewall commands printed if Q3 = yes |
| **6.7** | `fix(metacog)`: exclude `fcc:` from the transport baseline, clear stale keys (only if Q2 = yes) | orion-equilibrium-service | none |

No PR in this sequence adds a database migration.
