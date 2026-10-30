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
thought, `goal_ref = prime_concern:<id>`), `predict` per concern (below).

**Concerns are graded acts (step 1, 2026-10-30).** Holding a concern is a prediction: "acting on this will
produce something". Each concern gets one `predict` act (`object_ref = prime_concern:<id>`, act id stable on the
concern id). It closes **1 (settled)** when a question Prime sent about it (the move's `concern_id`) gets a
curiosity run that writes something, within 144 h of the act opening (two question horizons: only one question
is open at a time). It closes **0** when Prime drops the concern, lets it fade (not kept for a day), or holds it
past 144 h unsettled ("lapsed"). A held-back (control) question never settles a concern. Settling or lapsing does
not remove the concern; Prime decides whether to keep it. Nothing forbids a drop: the cost is the record. Each
thought's frame shows Prime's last 20 concerns: how many settled, how many were dropped (and how many of those
within an hour), faded, or lapsed. The record lives in `orion:prime:state` (`concern_record`), the same place
the concerns live, so it survives a restart and resets only if the concerns are lost too. The record is also the
outbox: a close is marked sent only after the bus accepts it, and the ledger keeps the first close of an act, so
a restart never closes a concern twice. A concern held from before this change gets its act on the first tick.

These acts are `actor = 'prime'`, the body-economy family `prime` (orion/physiology/appetite.py reads
`grammar_acts.actor` as the family). Once the family has 30 graded closes in 30 days, its want's value is
`intensity x mean outcome` ranked against its own history, so every abandoned concern (a 0) lowers the value of
Prime's next bid. No new price, no veto.

Known side effect, left on purpose for now: appetite also reads every close of the family as "the drive was fed".
So a concern close (a 0 included) resets Prime's hunger clock, and adds a gap to the close rhythm that sets how
fast hunger grows. A fade is closed on a quiet tick, so it can make Prime slightly less hungry after a day with
no thought. A drop is closed inside a thought, seconds from that thought's own close, so it barely moves the
clock, but it adds a near-zero gap that shortens the rhythm. Not measured live yet (no concern act existed before
this PR); `run_prime_report.py` now counts concern ends, so it can be checked after a week. If it matters, the fix is in `orion/physiology/appetite.py`:
count only `infer` closes for the clock and the rhythm, and keep `predict` closes for the value.

**References (step 1).** A question cites ids only through the move's `refs` (`[{kind, id}]`, kinds `arc` and
`act`, `orion/refs.py`). Before sending, Prime opens every ref with the same resolver a curiosity run uses
(`python -P -m orion.dig arc <id>` / `act <id>`, both GET orion-durable-runs). A ref that does not open, a
malformed ref, or an id-looking word in the question or why that is not in `refs` (a 12+ character hex run with
digits and letters, an `act_` id, or a UUID) refuses the move; the refusal is shown in the next frame as an
outcome, not as news, so it cannot loop. Refs travel on the focus row (`curiosity_focus_request.refs`) and the
run's focus section prints the command that opens each one.

**State** persists in Redis `orion:prime:state` (concerns, open questions, last reading, concern record), so a
restart resumes.

**Juniper first.** `main.mind_run` marks a positively-marked Juniper turn (`engine.carries_live_turn`). `mind_run` runs
the synchronous `run_mind` on the event loop, so Prime's tick cannot run during a turn at all (the `active()` check
keeps that true if `run_mind` ever moves to a thread). A thought whose model call overlapped a turn is discarded
(`preempted`), and the `background` route keeps a Prime call from starting while Juniper's call holds the slot.
NOT done: cancelling a model call already generating on the chat lane (the bus RPC has no cancel);
`PRIME_MAX_TOKENS` bounds that delay (live smoke: ~220 tokens, 3-7 s on the 8B metacog model; lane 0 UNVERIFIED).
Prime's model calls do not count in orion-mind's RPC-health window (`MindLLMClient(rpc_health=False)`): they wait
for slack by design.

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
