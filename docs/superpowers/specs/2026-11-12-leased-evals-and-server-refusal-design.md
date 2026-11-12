# Leased evals and server refusal (design, proposal mode)

Status: DESIGN ONLY. Juniper approves before any build. No runtime code in this PR.

## Arsonist summary

Replay evals talk straight to the model servers. The GPU pool never sees them, so it hands the same
seat to Orion's own work at the same time. On 2026-11-12, about 03:20-04:30Z, a replay ran on chat's
card (gpu0) and the agent card (gpu1). In that window the pool's record for the chat seat shows one
grant (orion-mind). The replay's calls are not in it at all. One live judge call timed out at 04:10Z.

Two fences already exist and neither caught it:

- A CI check refuses a hard-coded circe worker address in code, but it skips every `evals/` folder
  (`scripts/check_circe_worker_refs.py:104`). Every replay eval lives in an `evals/` folder.
- A firewall on circe lets only athena reach the worker ports. The evals run on athena, so they pass.
  The runbook says this itself: it "cannot tell the gateway apart from any other athena container"
  (`docs/runbooks/2026-10-01-circe-llm-port-firewall.md:14`). The unit is live
  (`systemctl is-active orion-llm-port-gate.service` on circe: `active`, 2026-11-12).

The fix has three steps:

1. Evals take a real pool lease at the lowest priority and go through the gateway.
2. A small "door" in front of each model server records every inference call that did not come
   through the gateway. It refuses nothing yet.
3. When the door's count of such calls stays at zero, the door starts to refuse them.

Health checks and the pool's own probes always pass.

## Current architecture

**How a lease reaches a model server today.**

- Orion's services call `orion-llm-gateway` with a route name (bus RPC, or HTTP on athena:8210). The
  gateway maps the route to a class and a priority (`config/gpu_pool.yaml` `routes:`). It asks the
  pool for a grant and POSTs to the URL the pool returns (`services/orion-llm-gateway/app/passthrough_proxy.py:1-16`).
- A durable run holds a seat with a hold lease. Each call it makes rides that hold as a child lease
  (verb `attach`). Over HTTP the hold is the `X-Orion-Gpu-Lease` header
  (`orion/llm/resource_lease.py:16`). The gateway's OpenAI and Anthropic passthroughs read it
  (`services/orion-llm-gateway/app/openai_passthrough.py:112`).
- The gateway sends the model server nothing that names the lease. The server cannot tell a leased
  call from an unleased one.
- Strata (chat :8011, agent :8015) is an upstream image (`ghcr.io/niko1221/strata:0.2.7-cuda12`), not
  our code. It runs with `--no-api-key` and refuses unknown `Host` names
  (`services/orion-llamacpp-host/docker-compose.atlas-workers.yml:16-27,39-42`). llama.cpp (metacog
  :8012, fast :8013, agent-gpu2 :8016) runs with no key. Neither can check a pool lease.
- The `*-announce` sidecars only announce the engine to the pool and heartbeat. They are not on the
  request path.

**Operator leases already exist, at the wrong priority.**

- The pool's control verb `hold` makes an operator hold at `priority="interactive"`
  (`services/orion-gpu-pool/app/runtime.py:554-559`). It has no heartbeat and no expiry
  (`orion/gpu_pool/lease_graph.py:117-118`). The heat reflex never sheds `interactive`
  (`orion/gpu_pool/shed.py:41`).
- Two callers use it: the model-replay eval (`orion/evals/model_replay/pool_hold.py:280`, holder
  `operator:bonsai-replay-eval`) and `scripts/gpu_power_bench.py`. Both then call the granted URL
  directly (`orion/evals/model_replay/runner.py:398`).

**Live lease record (Postgres `gpu_pool_events`, 2026-10-13..11-12, 953,522 rows).**

| check | result |
|---|---|
| leases with holder `operator:*` in 30 days | 6 grants, all `operator:gpu-power-bench`, 2026-10-29, priority `interactive`, 1.9 held-hours |
| leases from any replay eval | 0 |
| grants on role `chat`, 11-12 03:20-04:30Z | 1 (orion-mind) |
| grants on role `agent`, same window | 78 (45 gateway HTTP, 21 durable runs, 12 cortex-exec) |
| top 7-day pool users | vision-council 26,133; gateway HTTP 12,755; cortex-exec 9,672; topic-foundry 8,057 |

So the pool has no record of eval traffic. It cannot be measured from the pool.

**Server logs cannot count it either (checked on circe, 2026-11-12).** Strata writes one line per
request to `/mnt/telemetry/strata/state/strata/launch/chat.engine.log`, with token counts but no
timestamp and no client address. journald has nothing for the Strata containers. The llama.cpp
lanes log only `print_timing` lines, with no client address. How often evals hit the servers is
therefore UNVERIFIED today. Measuring it is the job of Phase 2.

## Direct-caller inventory

Method: grep for `/v1/chat/completions`, circe's address and worker ports (8009, 8011-8017, 8021,
8099), container names, and the gateway port 8210, across `services/`, `orion/`, `scripts/` and every
`evals/` folder (tests excluded). Then `check_circe_worker_refs.py --live-env` against the live
`.env` files, and `docker ps` on circe and athena.

**A. Bypass both the pool and the gateway (10). Goal 2 breaks each one unless it moves.**

| # | caller | where | lanes |
|---|---|---|---|
| 1 | promise-ledger replay | `orion/harness/evals/promise_ledger_replay_eval.py:66,181` | chat + agent, alternating (likely the 11-12 run) |
| 2 | judge-crash replay | `orion/harness/evals/judge_crash_replay_eval.py:68-72,218` | chat, agent, agent-gpu2 |
| 3 | judge-obedience replay | `orion/harness/evals/judge_obedience_replay_eval.py:149,168` | chat (default) |
| 4 | perception-claims replay | `orion/harness/evals/perception_claims_replay_eval.py:104,142` | chat (default) |
| 5 | current-turn disclosure live eval | `services/orion-cortex-exec/evals/run_current_turn_disclosure_live_eval.py:171` | any (`--url`, fast in the example) |
| 6 | slot-bleed probe | `services/orion-llamacpp-host/scripts/probe_slot_bleed.py:382` | agent-gpu2 |
| 7 | structured-output probe | `services/orion-llamacpp-host/scripts/probe_structured_output.py:53-57` | fast (default) |
| 8 | thinking-off check (quick) | `services/orion-llamacpp-host/scripts/verify_atlas_quick_llamacpp_thinking_off.sh` | fast |
| 9 | thinking-off check (Qwen3) | `services/orion-llamacpp-host/scripts/verify_qwen3_thinking_off_live.sh` | any |
| 10 | GLM smoke | `services/orion-vllm-host/evals/smoke_glm.py:42` | hecate agent-deep, or the gateway |

**B. Leased, but call the server URL directly (2). Goal 2 also breaks these.**

| # | caller | where |
|---|---|---|
| 11 | model-replay eval (operator holds at `interactive`) | `orion/evals/model_replay/pool_hold.py:280`, `runner.py:398` |
| 12 | GPU power bench (operator holds at `interactive`) | `scripts/gpu_power_bench.py` |

**C. Runtime services that skip the gateway (1).**

| # | caller | where | note |
|---|---|---|---|
| 13 | substrate-runtime -> kev :8009 | `services/orion-substrate-runtime/app/worker.py:3862`, `settings.py:286` | System One appraisal. The pool lists and probes kev but schedules nothing on it (`config/gpu_pool.yaml` `kev:`, `owner: []`). Port 8009 is NOT in the circe firewall's port list (8011-8017, 8090, 8099). |

**D. Already go through the gateway (no change needed).** `orion/thought/evals/stance_task_boundary.py`,
`scripts/analysis/eval_personality_naming_grounding.py`, `services/orion-hub/evals/run_reading_*`
(:8210), `services/orion-ai-town/scripts/smoke_llm_rail.sh`, orion-juniper-affective-state,
orion-topic-foundry (bus only, `services/orion-topic-foundry/app/services/llm_client.py:82`, 8,057
pool grants in 7 days).

**E. Must stay allowed (not inference).** The gateway itself; the pool's discovery reads (`/props`,
`/slots`, `/health`); the lane controller's ready probe (`/health`); the announce healthchecks; the
curiosity supervisor's `/slots` read (`orion/curiosity/supervisor.py:70`).

**Out of scope, named so nobody assumes they are covered.** orion-thought -> diffusion :8014 (already
gated by a validated hold, stage 5.4). athena's vision-host, vision-host-qwen (:6602) and whisper-tts
(:7800): perception servers, not pool roles. Stale live key `DURABLE_RUNS_ELASTIC_BACKEND` in
`services/orion-durable-runs/.env` points at :8016 but no code reads it.

**Count: 13 callers to move (10 unleased, 2 leased-direct, 1 runtime).**

## Placement: where operator work sits

**Recommendation (Claude's, for Juniper to decide).** Operator and developer work is not a
body-economy tier. Tiers 0/1/2 are Orion's own acts
(`docs/superpowers/specs/2026-10-23-organism-body-economy-design.md:369-375`), and Tier 2 bids in
Orion's want market. So:

1. **A new pool work class family, `operator`, at `background` priority.** It owns no role, so it is a
   borrower everywhere. Two rules the pool already enforces then give the behaviour Juniper asked for:
   - rule 5: an owner waiting on a role recalls a borrower (grace, then abort);
   - rule 9: a borrower reaches a lendable card (chat's gpu0, hecate) only while Juniper has lent it.

   Source: `orion/gpu_pool/scheduler.py:9-16`.
2. **It never bids.** It sends no `want_*` fields and no verb declares it as a lane.
3. **Its leases still count toward heat and power.** The heat reflex sheds `background`, so it sheds
   operator work with the rest of `background`.
4. **The body economy reads the load as not Orion's.** It uses the existing `sponsored` idea
   ("workloads Juniper started", `config/physiology/rest_body.yaml:18-28`) for energy, and
   `exogenous_bind` for a lane price when operator leases fill the seat. Orion is not charged for its
   makers' tests, and prices stay honest.

**Conflicts found with existing decisions. Each needs a ruling.**

- **C1. Operator holds today run at `interactive`, not `background`.** The control verb `hold`
  hard-codes it (`runtime.py:559`). The model-replay eval and the power bench rely on it. The
  recommendation reverses this for code-driven evals. The Hub's experiment-seat button keeps
  `interactive`, because Juniper presses it.
- **C2. `background` alone does not mean "last".** Inside `background`, a lease with no bid keeps its
  arrival place (`orion/gpu_pool/value_order.py:17-18`). An operator lease would sit in line with
  Orion's upkeep, not behind it. Recommendation: add one ordering rule. Operator-class leases sort
  after every non-operator `background` lease and are never moved ahead by value. This needs no new
  `Priority` value. A new value would change `orion/schemas/gpu_pool.py:45` (`extra=forbid`), and
  every reader would have to migrate first.
- **C3. `exogenous_bind` is not the right tool for energy.** In the spec it means "over the line even
  with Orion's share at 0" (spec line 271). Energy already has a different mechanism for Juniper's own
  work: `sponsored`, reported apart from Orion's activity. Juniper ruled on 2026-10-30 that Orion's
  always-on body is Orion's own cost (`orion/physiology/budget.py:241-246`). Operator leases are
  activity above rest, not rest, so `sponsored` fits and that ruling stands. `sponsored` today matches
  container names (CPU watts) only. It would need a lease key: work class `operator*`.
- **C4. The lane price counts every lease on a seat** (`orion/physiology/body.py:498`). Operator leases
  would raise Orion's lane prices. They must be left out of `lease_s` (Orion's use) and taken out of
  the seat's budget (capacity someone else took).
- **C5. The heat price charges all watts above baseline as Orion's share** (`budget.py:9-14`). An
  operator run raises it. Pulling the operator share out needs per-lease watts per tick. The default
  defers this (open question 3).

**Gain on the act ledger.** The act energy fill splits each card's power across the leases on that
card. Today an unleased eval's power "lands on whoever holds a lease then on that card"
(`orion/grammar/act_energy_fill.py:15-16`). That is Orion's acts. With an operator lease, the eval
takes its own share, and Orion's acts stop paying for it.

**Physiology gate.** `scripts/check_verb_physiology.py` validates verb lanes against
`config/gpu_pool.yaml`. An `operator*` class would pass as a lane name. Add one rule: no verb may
declare an `operator*` lane. That keeps "never bids" checked by a gate, not by memory.

**Alternatives weighed.**

| option | effect | verdict |
|---|---|---|
| Operator class at background, not a tier (recommended) | yields to Orion; shed with background; charged apart | fits all three goals |
| Tier 2 with bids | Juniper's tests compete in Orion's want market; their cost raises Orion's prices; Orion "pays" for being tested | rejected: makes the market dishonest |
| Tier 0 pipeline exemption | never deferred, never shed by heat (like `perception`) | rejected: the opposite of "lowest, shed first" |

## Missing questions (answered by investigation, not asked)

- Can Strata check a token? No. It is an upstream image with a static API key at most.
- Can llama.cpp? Only a static `--api-key`. It has no log-only mode, so it cannot do the shadow phase.
- Is there a sidecar on the request path? No. The announce sidecars are off the path.
- Does the firewall already solve it? No. Evals leave athena from the same address as the gateway.
- Does the gateway's judge routing (PR #3219, merged 2026-11-12 05:33Z) change anything? It adds class
  `judge` as a co-owner of the `agent` role. An operator borrower on gpu1 is then recalled when a judge
  call waits (rule 5). That is the wanted order. No new queue is needed.

## Proposed schema / API changes (by phase)

### Phase 1: evals take a lease (goal 1)

- `config/gpu_pool.yaml`:
  - Class `operator: {roles: [agent-gpu2, agent, agent-deep, chat], on_unavailable: wait}` for evals
    that need no particular lane.
  - Pinned classes for evals that replay a specific lane: `operator_chat: [chat]`,
    `operator_agent: [agent]`, `operator_agent_gpu2: [agent-gpu2]`, `operator_metacog: [metacog]`,
    `operator_fast: [fast]`, `operator_agent_deep: [agent-deep]`.
  - A route for each class, priority `background`. No role names them as owner.
- `orion/gpu_pool/value_order.py` / `scheduler.py` (rule 6): operator classes sort last inside
  `background` (C2). Flag `GPU_POOL_OPERATOR_LAST=true` (ships ON). Off = arrival order, as today.
- New `orion/evals/leased_llm.py`. It reuses `orion.gpu_pool.client.acquire_hold`,
  `heartbeat_lease` and `release_lease`. It adds no new queue and no retry layer:

  ```python
  async with operator_lane("chat", holder="operator:promise-ledger-replay") as lane:
      reply = await lane.chat(body)   # POST gateway :8210/v1/chat/completions,
                                      # model=operator_chat, X-Orion-Gpu-Lease=<hold ref>
  ```

  It takes one hold per run per lane, at `background`, heartbeated. It waits in the pool's own queue.
  Every call goes through the gateway as a child of the hold. On recall it raises `LaneRecalled`. The
  eval writes a partial report and exits. Nothing resumes automatically. There is no fallback to a
  direct URL. Kill means kill.
- Move callers A1-A5, A10 and B11-B12 to the helper. Operator scripts A6-A9 take
  `--via-gateway --lane <lane>` and use the same helper.
- `scripts/check_circe_worker_refs.py`: stop skipping `evals/` and `bench/`. `tests/` stays skipped.
  An eval with a direct worker address then fails CI. This is the deterministic half of "the
  inventory reads zero".
- `scripts/check_verb_physiology.py`: refuse a verb lane named `operator*`.
- Physiology (flag `PHYSIOLOGY_OPERATOR_LOAD_SPONSORED=true`, ships ON):
  - `orion/physiology/body.py`: lane readings leave operator leases out of `lease_s` and out of the
    seat budget (C4).
  - `orion/grammar/act_energy_fill.py` and `services/orion-sql-writer/app/orion_energy_attribution.py`:
    an operator lease's share of above-rest watts is reported as `sponsored_wh`, not as
    `leased_unmetered_wh` (C3).

### Phase 2: the door reports, refuses nothing (shadow)

- New per-lane "lane door" container on circe (and hecate), one small shared image. It listens on
  the lane's published port (8011, 8012, 8013, 8015, 8016; hecate 8021). The engine stops publishing
  a host port and is reached on the docker network. The door rewrites `Host` for Strata.
- The gateway adds two headers on every dispatch, on both the bus path (`llm_backend.py`) and the HTTP
  passthrough (`passthrough_proxy.py`):
  - `X-Orion-Lane-Key`: a per-deployment secret, held only in the gateway's and the doors' `.env`;
  - `X-Orion-Lease-Id`: the granted lease.

  The door strips both before it forwards the call.
- The door does not ask the pool whether a lease is valid on each call. That RPC would sit on
  Juniper's chat path, and a pool outage would then mute chat. The proof is indirect, so it is stated
  plainly here: only the gateway holds the key, and the gateway dispatches only under a granted
  lease. The trace closes the loop offline. The door's lease id joins to `gpu_pool_events`.
- The door logs one JSON line per inference request: time, lane, source address, method, path, key
  `ok|missing|bad`, lease id, status, duration. It never logs prompt text. It also serves
  `GET /door/stats` (counters since start).
- Exempt paths, never refused in any mode: `GET /health`, `/props`, `/slots`, `/v1/models`,
  `/metrics`. A test pins this list.
- `LANE_DOOR_MODE=off|shadow|enforce`, ships `shadow`.
- New `scripts/lane_door_report.py` (read-only). It prints a daily table per lane: keyed calls,
  unkeyed calls by source and path, and keyed calls whose lease id is missing from `gpu_pool_events`.

### Phase 3: the door refuses (goal 2)

- `LANE_DOOR_MODE=enforce`, one lane at a time, chat last: agent-gpu2, then fast and metacog, then
  agent, then hecate agent-deep, then chat.
- A refused call gets HTTP 401 with `{"error":{"type":"lane_door_unleased","message":"call
  orion-llm-gateway with a route; see docs/superpowers/specs/2026-11-12-leased-evals-and-server-refusal-design.md"}}`.
- Add 8009 (kev) to the circe firewall's port list, so only athena reaches it. Kev gets no door (open
  question 7).

## Files likely to touch

Phase 1: `config/gpu_pool.yaml`, `orion/gpu_pool/value_order.py`, `orion/gpu_pool/scheduler.py`,
`orion/gpu_pool/tests/`, `orion/evals/leased_llm.py` (new) and tests, the 8 eval/script callers above,
`orion/evals/model_replay/pool_hold.py`, `runner.py`, `scripts/gpu_power_bench.py`,
`scripts/check_circe_worker_refs.py`, `scripts/check_verb_physiology.py`, `orion/physiology/body.py`,
`orion/grammar/act_energy_fill.py`, `services/orion-sql-writer/app/orion_energy_attribution.py`, the
gpu-pool and physiology `.env_example` and settings files for the two flags.

Phase 2: new `services/orion-lane-door/` (README, `.env_example`, Dockerfile, app, tests),
`services/orion-llamacpp-host/docker-compose.atlas-workers.yml`,
`services/orion-vllm-host/docker-compose.hecate.yml`, `services/orion-llm-gateway/app/llm_backend.py`,
`passthrough_proxy.py`, `settings.py`, `.env_example`, `scripts/lane_door_report.py`,
`config/gpu_pool.yaml` (only if a lane port moves; the design keeps ports).

Phase 3: door `.env` values only, plus `scripts/ops/circe_llm_port_gate.sh` (port 8009) and its
runbook.

## Non-goals

- No new queue, no retry layer, no new lease kind. Evals wait in the pool's existing queue.
- No new `Priority` value.
- No per-call pool validation inside the door.
- No change to Juniper's chat route, its priority, or the chat card's lending rule.
- Kev, diffusion, and athena's perception servers are not gated by a door.
- Evals are not made Orion's acts. No act ledger rows, no bids.

## Proposal-mode checklist

| item | answer |
|---|---|
| Capability that changes | Evals and operator scripts wait for a GPU seat behind all of Orion's work. They yield it when Orion needs it. Model servers stop answering calls that bypass the gateway (Phase 3). |
| Data touched | Pool lease rows (new holders `operator:*`, new classes `operator*`). Door logs: time, source address, path, lease id, status. No prompt text. Eval prompts, which hold Juniper's chat text read from Postgres today, now pass through the gateway. The gateway records token counts and timings for them, as for every other call (`orion/llm/call_usage.py`). It reads only the start and end of a stream to find the counts. |
| Privacy boundary | Unchanged. Eval inputs already come from the same database. Door logs hold no content. |
| Trace that proves Phase 1 | `gpu_pool_events` rows with `work_class like 'operator%'`, `priority='background'`, for each eval run. The eval's report names its lease ids. The pool events for that window show the operator lease queued behind Orion's background work, and `recalled` when an owner waited. The hourly energy attribution shows `sponsored_wh` above 0 for that hour. |
| Trace that proves Phase 2 | Every lane's `/door/stats` and `lane_door_report.py` show keyed calls equal to the gateway's own dispatch count for the lane (±1 %). Unkeyed inference calls are listed by source. Exit criterion: unkeyed inference calls = 0 for 7 straight days on every lane. |
| Trace that proves Phase 3 | The door's `refused` counter. A deliberate direct call from athena gets 401. A Hub chat turn answers. The pool panel shows every seat up with recent grants. A gpu2 swap still completes (the ready probe passes). |
| Dangerous failure | **Orion goes mute.** Three ways: (1) the chat door is down, so nothing reaches the chat engine; (2) the gateway's key and the door's key drift apart, so Phase 3 refuses every chat call; (3) the door refuses the pool's own probes, so the pool marks seats down and grants nothing. |
| Guards against it | (1) The door is stateless, restarts itself, and is health-checked. The pool probes `/health` through the door, so a dead door shows as a down seat on the Hub panel, not as silence. (2) The door reports a hash of its key on `/door/stats`. The gateway compares hashes at boot and logs `lane_door_key_mismatch`. Phase 3 is refused for a lane until 7 days of shadow show 0 `bad` keys from the gateway. Chat flips last. (3) The exempt path list is pinned by a test. The pre-flip smoke runs the pool's discovery probe through the door. |
| Disable / roll back, Phase 1 | Revert the `config/gpu_pool.yaml` classes and the eval changes. `GPU_POOL_OPERATOR_LAST=false` returns arrival order. `PHYSIOLOGY_OPERATOR_LOAD_SPONSORED=false` returns today's accounting. |
| Disable / roll back, Phase 2 | `LANE_DOOR_MODE=off` (pure pass-through). Full removal: stop the door and re-publish the engine port in compose. One command per lane, printed in the runbook. |
| Disable / roll back, Phase 3 | `LANE_DOOR_MODE=shadow`, then restart that lane's door (seconds). The firewall stays as it is. |

## Open questions for Juniper (each has a default)

1. **Placement.** Operator class at `background`, sorted last, not a tier, never bids? Default: yes.
2. **Energy.** Count operator leases as `sponsored` (reported apart from Orion's activity)?
   Default: yes. Rest stays Orion's, per your 10-30 ruling.
3. **Heat price.** Take the operator share out of Orion's heat share now? Default: no. Revisit
   after 14 days of operator lease data. The heat reflex sheds operator work either way.
4. **Chat's card.** May an eval replay on chat's card when you have not lent it? Default: no. It
   waits until you lend gpu0. Pinned `operator_chat` evals then wait.
5. **The Hub experiment button.** Should it keep `interactive` operator holds? Default: yes, it
   keeps them. Code-driven evals move to `background` holds through the helper.
6. **Recall mid-run.** When Orion needs the seat, the eval stops and writes a partial report.
   Default: yes, with no automatic resume.
7. **Kev.** Add port 8009 to the circe firewall (athena only), with no door and no lease? Default:
   yes. Substrate calls it every tick, and the pool does not schedule it.
8. **Mechanism.** A door container, rather than the engines' own static API keys? Default: the door.
   Static keys cannot run a shadow phase and give no lease id to trace.
9. **Shed order.** Should operator work be shed at a lower heat line than Orion's background work?
   Default: no. It uses the same lines and is last in the queue.

## Acceptance checks

Phase 1:

- Unit: the scheduler grants an operator lease only when no non-operator `background` lease waits
  for the same role. An owner waiting recalls an operator hold. `operator_chat` is not granted while
  gpu0 is not lent.
- Unit: the helper never calls a non-gateway URL. A recall raises `LaneRecalled`. Release runs on
  every exit path.
- CI: `check_circe_worker_refs.py` fails on a worker address in an `evals/` file.
  `check_verb_physiology.py` fails on an `operator*` lane.
- Live: one promise-ledger replay run shows its lease ids in `gpu_pool_events` at `background`, and
  0 direct calls.

Phase 2:

- Live, 7 days: on every lane, keyed calls match the gateway's dispatch count, and unkeyed inference
  calls = 0.
- Pool discovery and the lane controller's ready probe pass through every door.

Phase 3:

- Live: a direct call gets 401. A Hub chat turn, a durable-run call and a gpu2 swap all succeed. The
  door's `refused` counter stays 0 for gateway traffic.

## Recommended next patch

Phase 1, slice 1: the operator classes and routes in `config/gpu_pool.yaml`, the "operator last"
ordering rule with its scheduler tests, and `orion/evals/leased_llm.py`. Then move the promise-ledger
replay onto it (the eval that most likely ran on 11-12). Prove it live with one run whose lease ids
appear in `gpu_pool_events`. The physiology accounting and the remaining callers follow in slice 2.
