# orion-mind

FastAPI service exposing `run_mind()` (`orion.mind.v1.MindRunRequestV1` -> `MindRunResultV1`). This README does not attempt to cover the whole service -- it documents one specific, verified piece: the `llm_surface_instability` metacog trigger, added here because `services/orion-equilibrium-service/README.md`'s trigger-taxonomy documentation flagged this as the one real `MetacogTriggerV1` producer living outside that service with no README coverage anywhere (2026-07-21).

## `llm_surface_instability` metacog trigger

`app/uncertainty_metacog.py`'s `maybe_publish_llm_surface_instability_trigger()` fires an *advisory* `trigger_kind=llm_surface_instability` metacog trigger when an LLM response's own logprob telemetry looks unstable -- not a factual-confidence signal, a language-surface-instability one. Gated by `MIND_LLM_UNCERTAINTY_METACOG_ENABLED`.

**Flipped on 2026-08-11**, same call as `EQUILIBRIUM_METACOG_INSIGHT_TRIGGER_ENABLE`/`FLOW_TRIGGER_ENABLE` in `services/orion-equilibrium-service/README.md` -- see that file for the full context (an 11-day mesh outage meant no live window existed to judge any of these advisory triggers until now).

Fires (`should_emit_llm_surface_instability()`) when, off the response's `llm_uncertainty` telemetry dict:
- `unstable_span_count >= 1` (reason: `unstable_span`), or
- `mean_top1_margin < 0.75` (reason: `low_mean_margin`), or
- `low_logprob_token_count / token_count_observed > 0.15` (reason: `high_low_logprob_ratio`)

Never fires if `llm_uncertainty.available` is falsy or `token_count_observed <= 0` (reason: `unavailable`/`no_tokens`). `pressure` on the trigger is `min(1.0, low_logprob_token_count / token_count_observed)`; `upstream` carries the full `llm_uncertainty` dict, the phase name, and the specific `instability_detail` reason string.

**Unlike every trigger documented in `orion-equilibrium-service`'s README, this one bypasses equilibrium's gate logic entirely** -- `_publish_metacog_trigger_async()` publishes `MetacogTriggerV1` directly onto `MIND_METACOG_TRIGGER_CHANNEL`, which defaults to `orion:equilibrium:metacog:trigger` (the same channel equilibrium-service's own `_publish_metacog_trigger()` writes to for `orion-cortex-orch` to consume). There is no confidence/level floor here the way `relational`/`telemetry_anomaly` have -- the three boolean/threshold conditions above are the only gate.

| Env | Default | Purpose |
|-----|---------|---------|
| `MIND_LLM_UNCERTAINTY_METACOG_ENABLED` | `true` | Master gate |
| `MIND_METACOG_TRIGGER_CHANNEL` | `orion:equilibrium:metacog:trigger` | Publish channel -- shared with equilibrium-service's own triggers, not a dedicated inbound channel |
| `ORION_BUS_URL` | `redis://100.92.216.81:6379/0` | Bus connection (fresh `OrionBusAsync` per publish, one-shot connect/publish/close). Was stale (`redis://redis:6379/0`) in this table and in `settings.py`'s code-level default before the service-heartbeat rollout fixed it -- the checked-in `.env_example` already had the correct value. |

## Bus-native heartbeat

Publishes a `SystemHealthV1` heartbeat to `orion:system:health` every
`HEARTBEAT_INTERVAL_SEC` (default 10s), on its own independent bus connection via
`app/main.py`'s FastAPI `lifespan` (this service had no `lifespan` at all before this patch).
Separate from the LLM-gateway RPC bus connections used by `MIND_LLM_USE_BUS`. See
docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.

## RPC-health snapshot publish (2026-09-24)

Every LLM call (`MIND_LLM_USE_BUS`) opens its own short-lived bus to
`orion:exec:request:LLMGatewayService` and closes it afterwards, so its timing tally used to
vanish with it. Each call now folds its tally into one process-wide sink
(`app.llm_client.RPC_HEALTH_SINK`, in the `finally` that closes the bus, so timeouts and
errors count), and a separate long-lived bus opened in the FastAPI `lifespan` publishes it
every `RPC_HEALTH_PUBLISH_INTERVAL_SEC` as an `RpcHealthSnapshotV1` on
`orion:rpc_health:snapshot` (`instance=main`, organ `rpc_health_mind`).

| Key | Default | Meaning |
| --- | --- | --- |
| `RPC_HEALTH_PUBLISH_ENABLED` | `true` | start the publisher |
| `RPC_HEALTH_PUBLISH_INTERVAL_SEC` | `30` | window length |
| `RPC_HEALTH_CHANNEL_LATENCY_ENABLED` | `true` | include the per-hop `channel_latency` breakdown |

Rollout is consumer-first: the schema is `extra="forbid"`, so rebuild
`orion-signal-gateway` and `orion-equilibrium-service` on PR #2312's build before deploying
this service with `RPC_HEALTH_CHANNEL_LATENCY_ENABLED=true`.

## Orion Prime 2A: the resident loop (2026-10-30)

orion-mind keeps thinking between conversations. Design: `docs/superpowers/specs/2026-10-29-orion-prime-design.md`
(PR #2980), rollout step 2A. Code: `app/prime_loop.py` (I/O) and `orion/prime/` (pure logic).

**What it does.** Every `PRIME_TICK_SEC` a CPU-only tick reads the body (durable-runs `GET /physiology/state`)
and attention (`GET /temporal-self/frame`) and records what changed: attention moved to another arc, the set of
*binding* body constraints changed, one of Prime's own questions settled, a concern faded. If something changed,
the refractory interval has passed and no Juniper turn is running, Prime bids in the body economy (family `prime`,
Tier 1, lane `chat`) and **obeys the answer**: funded thinks, deferred or no answer waits. A thought is one model
call on route `prime` (circe chat lane 0 at `background` priority, so Juniper's call always starts first). Prime
answers with how it reads things now, edits to its at most 3 concerns, and at most one move.

**The one move (2A).** A question for curiosity's next regular investigate run, queued in Hub's existing focus slot
(`orion/curiosity/focus.py`) with `binding.invocation_context = "prime"`. Hub stores it origin `self` with
`chat_correlation_id = "prime:<thought_id>"`, never replaces a pending focus with it, and tells the run it was
chosen "between conversations", not in a chat. Why not allocator candidates (the spec's first plan): reading,
painting and retro leisure are already admitted on almost every dispatch tick (live 10-30: 3,519 / 2,129 / 3,322 a
day), so a duplicate candidate from Prime could not change anything.

**Narrator control.** `PRIME_HOLDBACK_FRACTION` of questions are opened as acts and never sent. Both arms are graded
the same way: did the curiosity run that took the question (treated) or the next one after it (control) write
something. Prime is told when a question was held back. Readout: `orion/prime/evals/run_prime_report.py`.

**Acts** (`grammar_acts`, `actor = 'prime'`): `infer` per thought (produced = the answer parsed; `preempted` when a
Juniper turn overlapped the call), `ask` per question (`test_before = holdback:<arm>`, `parent_act_id` = the
thought, `goal_ref = prime_concern:<id>`).

**State** persists in Redis `orion:prime:state` (concerns, open questions, last reading), so a restart resumes.

**Juniper first.** `main.mind_run` marks a positively-marked Juniper turn (`engine.carries_live_turn`). Prime does not
bid during one and discards a thought whose model call overlapped one. NOT done: cancelling a model call already
generating on the chat lane (the bus RPC has no cancel); `PRIME_MAX_TOKENS` bounds that delay.

| Key | Default | Meaning |
| --- | --- | --- |
| `PRIME_ENABLED` | `true` | the loop; off = no bids, no acts, run_mind unchanged |
| `PRIME_TICK_SEC` | `15` | CPU tick |
| `PRIME_REFRACTORY_SEC` | `300` | least time between thoughts (at most 12 an hour while E9 is measured) |
| `PRIME_SETTLE_INTERVAL_SEC` | `300` | how often open questions are checked against curiosity's runs |
| `PRIME_LLM_ROUTE` | `prime` | gateway route (config/gpu_pool.yaml) |
| `PRIME_BID_LANE` | `chat` | lane named in the bid and shown in the frame |
| `PRIME_MAX_TOKENS` | `700` | answer budget (live smoke: ~220 tokens used) |
| `PRIME_LLM_TIMEOUT_SEC` | `60` | model call timeout |
| `PRIME_HOLDBACK_FRACTION` | `0.2` | narrator arm share |
| `PRIME_DURABLE_RUNS_URL` | `http://orion-athena-durable-runs:8121` | body + attention |
| `PRIME_HUB_URL` | `http://100.92.216.81:8080` | curiosity runs + pending focus |
