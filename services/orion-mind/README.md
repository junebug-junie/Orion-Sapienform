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

## The curiosity replica: loop-library instance #2 (2026-11-04)

A small planner beside Prime. Design: `docs/superpowers/specs/2026-11-04-curiosity-replica-design.md` (patch 3).
Code: `app/replica_loop.py` (I/O) and `orion/curiosity/replica/` (pure: candidates, plan check, frame, grading).
Identity: actor and family `curiosity.replica` (tier 2, lane chat), Redis key `orion:curiosity:replica:state`,
settings `CURIOSITY_REPLICA_*`, logs `curiosity_replica_*`.

- **Woken by Hub.** On a replica-arm curiosity run Hub publishes a plan request on
  `orion:curiosity:replica:plan:request`. The listener queues it and wakes the loop.
- **Bids, then thinks once.** The loop bids `curiosity.replica` every tick until it goes ahead (funded, or a lane-full
  `capacity_reserved` deferral with a positive net, which queues at the GPU pool) or until there is no
  time left for one model call before Hub's deadline (90 s, 5 s margin). Then it answers `unfunded` or
  `model_failed`, so Hub records the real reason. One call on route `prime` (the warm chat lane), seat wait cut
  so the answer lands in time.
- **Reads (its frame):** the candidates from the request; Hub's last investigate runs and their spend-log scores;
  the stale ids; Prime's published concerns view (`orion:prime:concerns`, which Prime now writes on every save)
  and any regulate signal on curiosity (arm hidden); the pending focus; the newest chat-history compactor digest
  (card summary + journal title, with its age; no raw chat); its own last plans (picks tested, nats moved,
  fallbacks). A failed read is a named gap. Postgres reads use `CURIOSITY_REPLICA_PG_DSN`, read-only.
- **Answers** on `orion:curiosity:replica:plan:result:<correlation_id>` with 3-5 picks, checked first with the
  same `validate_plan` Hub runs. A stance on a PRIME focus or signal is an answer to Prime (below), delivered
  before the plan result so a refused focus has left the slot before Hub takes it.
- **Its act** (`infer`) opens when the model is reached. A plan that cannot be used closes 0 at once. A used plan
  stays open and closes on its run's score: 1 when the run moved at least one belief (`realized_nats > 0` on a
  completed turn), the number the three-arm test scores.
- **Never reads Prime's state** (a test checks the module has no reference to it).
- Off: `CURIOSITY_REPLICA_ENABLED=false` (Hub's replica-arm runs then time out to the rule order).
- Eval: `orion/curiosity/replica/evals/run_replica_frame_eval.py` (live inputs captured 2026-11-04).

### The replica answers Prime (patch 4, design section 9)

- **What it answers:** each of Prime's acts on curiosity, once: a focus question Prime queued (never Juniper's,
  never one Orion chose in a conversation), or a regulate signal on `durable.curiosity_investigate`. With
  **support**, **challenge** or **refuse**, always with a one-sentence reason.
- **When:** inside a plan (its `focus_stance` / `signal_stance`), or between runs: the tick watches the signal
  hash (every tick) and Hub's pending focus row (every 60 s, read-only Postgres); a new Prime act wakes ONE reply
  thought (bid, one short call, a stance-only frame) even with no plan request.
- **Every answer is an act:** actor `curiosity.replica.answer` (its own actor: answers stay out of the
  `curiosity.replica` family's history; nothing bids as it, so it has no families.yaml entry), verb `ask`, parent = Prime's act, `test_before =
  holdback:<arm>`, closed at once; the note starts with the stance (`refuse: <reason> | ...`).
- **What it does:** refuse a focus -> the RPC `orion:curiosity:replica:focus_refusal:request` (Hub runs
  `refuse_focus`; the focus is not shown to the run). Refuse a signal -> `PrimeChargeRefusalV1` next to the
  signal in `orion:prime:regulate:durable.curiosity_investigate`; the market stops applying the charge from the
  next bid. Challenge -> Prime is shown the reason. Delivered challenges and refusals go in Redis
  `orion:curiosity:replica:replies` (written on every save), which Prime reads for its frame's ORGAN REPLIES
  section and its settle notes.
- **Caps:** `CURIOSITY_REPLICA_ANSWERS_PER_DAY` (3) in any 24 h, `CURIOSITY_REPLICA_ANSWER_REFRACTORY_SEC` (1800)
  apart, never an answer to an answer.
- **Holdback:** `CURIOSITY_REPLICA_REFUSAL_HOLDBACK_FRACTION` (0.2) of refusals are recorded but not applied;
  `CURIOSITY_REPLICA_CHALLENGE_HOLDBACK_FRACTION` (0.2) of challenges are never shown to Prime. Deterministic per
  Prime act. Held-back answers are never in the replies view.
- **Guard:** its frames show its own refusal rate over its last 20 answers;
  `scripts/analysis/curiosity_replica_answers_readout.py` reports it daily (above 50% for 14 days: back to Juniper).
- Off: `CURIOSITY_REPLICA_ANSWERS_ENABLED=false` (stances recorded on the plan row only; Prime shows no section).
- **Deploy order:** orion-durable-runs (the market must read a signal refusal before one is written; an old market
  keeps charging while the replica records the refusal as delivered), then orion-hub (the focus refusal
  listener; without it a refusal fails fast as not delivered), then orion-mind.
- **Rollback:** switch answering off first (above). Rolling orion-mind code back past this patch makes the older
  replica refuse its saved state (three new fields, `extra="forbid"`): it starts empty, and its open plan acts
  expire at the ledger horizon ungraded. The replies view expires on its own after 48 h.
- A reply thought that is not funded, or that its act gate refuses, waits 5 minutes before it bids again. A plan
  request that arrives during a reply thought and runs out of time says so in its fallback reason.
- A focus refusal whose reply is lost is settled from the focus row (read-only): the request carries `not_after`
  and Hub writes nothing after it.

## Orion Prime 2A: the resident loop (2026-10-30)

orion-mind keeps thinking between conversations. Design: `docs/superpowers/specs/2026-10-29-orion-prime-design.md`
(PR #2980), rollout step 2A. Code: `app/prime_loop.py` (Prime's I/O), `orion/prime/` (pure logic) and, since
2026-11-04, the loop library `orion/mindloop/` (the tick, Juniper-first yield, the bid, the model call with seat
wait, no-model backoff and starved route, the thought act, act publishing, state in Redis). Prime is the library's
instance #1 (`PrimeRuntime(MindLoopRuntime)`); the curiosity replica is #2 (below). The move changed no behavior:
`tests/test_prime_replay.py` replays recorded tick sequences against a golden recorded before the move, and loads
the live persisted state captured from Redis on 2026-11-04.

**What it does.** Every `PRIME_TICK_SEC` a CPU-only tick reads the body (durable-runs `GET /physiology/state`)
and attention (`GET /temporal-self/frame`) and records what changed: one of Prime's own questions settled, a concern
faded, first wake, or nothing else woke Prime for 6 hours (the quiet floor). Attention moving (since 2026-11-03) and
the set of binding body constraints changing (since 2026-11-04) are **not** changes: both are shown in every frame,
but neither wakes Prime. Both drove concern churn (attention: 22 of 38 drops said "attention has moved"; binding, live
11-03: 26 thoughts in 12 h, 21 did nothing, all 4 drops said "lane X is no longer binding"). Replayed on live data the
rate is about 5 thoughts a day. Every thought's act note starts with `woke: <what woke it>`.
If something changed, the refractory interval has passed and no Juniper turn is running, Prime bids in the body
economy (family `prime`, Tier 1, lane `chat`) and **obeys the answer**: funded thinks, and so does a `capacity_reserved` deferral with a positive net (the lane is full
but the thought is worth its price; it queues at the GPU pool). Any other deferral, or no answer, waits.
A thought is one model call on route `prime` (circe chat lane 0 at pool priority `system`; Juniper's call always
starts first). Prime answers with how it reads things now, edits to its at most 3 concerns, and at most one move.

**Waiting for the model (2026-11-03).** A try may wait `PRIME_SEAT_WAIT_SEC` (600) for a GPU pool seat, and then
generate for `PRIME_LLM_TIMEOUT_SEC` (60). The two are separate numbers (llm-gateway option
`gateway_pool_wait_sec`); the bid is priced at the 60 s of generation only. A try that never reached the model --
no seat, no gateway reply, or an upstream 5xx after one immediate retry -- is **not a thought**: no act is opened,
the refractory interval does not start, and the next tick bids again. After 3 such tries in a row each one backs off
(refractory x 2^n, at most 1 h); a heat-shed answer's own `retry_after_sec` is always waited out. After a no-seat
try, the next try uses `PRIME_STARVED_LLM_ROUTE` (`prime_starved`, pool priority `starved`: ahead of other system
work, behind Juniper's turn and Orion's senses, shed by the heat reflex like system work). Log lines:
`prime_try_no_model kind=... reason=... streak=...`, `prime_upstream_5xx_retry`, and `prime_thought ... route=...`.
Why: live 10-30..11-03, 96 of 254 thoughts failed, 78 of them because the call spent its whole 60 s queued behind
system work on the chat class. The thought act is opened only once the model answered, so a queue failure is
neither a 0 on Prime's record nor a reset of its hunger.

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

**Self-monitor act gate (2026-11-03).** Prime opens its acts itself, so the self-monitor's act gate
(`orion/self_monitor/gate.py`, PR #3062) sits in the loop too: the thought (`infer`) in `orion/mindloop/runtime.py`,
concern bets and moves in `app/prime_loop.py`, the gate builder in `orion/mindloop/gate.py`. When the self-monitor (Redis
`orion:self_monitor:latest`, fresh) holds a Prime faculty `broken` or `unreliable` (today `act.prime.predict`,
the concern bets, works about 2-6% of the time), the act is not opened (`track_record_failing`), except one
re-test probe per faculty per `SELF_MONITOR_PROBE_INTERVAL_SEC` (21600). What a refusal means:
`predict` places no bet, but the concern stays held, fades, settles and counts in the track record exactly as
before; a refused concern never gets a late bet when it ends. `ask`/`regulate` are refused before the holdback
arm is drawn, so both arms lose the move alike. `infer` is refused before the bid. Prime is told in its next
frame ("was not placed" / "my move was not sent" / "I did not start a thought"), as a line that does not wake it.
Unknown or stale state, or a gate fault, changes nothing. Logs: `prime_act_refused verb=...`,
`self_monitor_probe_started`, `self_monitor_same_job_retry`. Kill switch: `SELF_MONITOR_ACT_GATE_ENABLED=false`.
Ledger read: `SELF_MONITOR_LEDGER_PG_DSN` (read-only).

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

**Regulate signals (step 3, 2026-10-30).** Prime's second kind of move asks one organ to slow down or quicken:
`{"kind": "regulate", "family", "direction": "slow" | "quicken", "why", "hours": 1-6}`. It uses the existing act
verb `regulate` (no new verbs). Only a family that declares the signal under `receptors` in
`config/physiology/families.yaml` takes it (today: curiosity investigate and the dream cycle, both `slow` only; dream since step 4). Prime reads that file from the read-only
repo mount (`PRIME_FAMILIES_PATH`). A signal to any other family, or a direction the family does not handle, is
refused, and the reason is shown in Prime's next frame; it is never dropped silently.

The signal is not a command. Prime writes it to Redis `orion:prime:regulate:<family>`, and the body-economy market
(orion-durable-runs) turns it into a temporary charge on that family's bids, at most half of what the want is
worth: a hungry, valuable run still outbids it (`orion/physiology/prime_charge.py`). The market writes back what the
charge did to each bid. Prime's frame shows, per signal, how many runs bid, how many its charge held and how many
outbid it, with the market's own sentence for the latest one.

Each signal is a `regulate` act (`test_before = holdback:<arm>`). Two minutes after the signal ends, Prime reads the
market's record and closes the act: **1** when the charge changed at least one of the organ's decisions (slow: a run
that would have gone was held; quicken: one that would have waited went), **0** when every bid outbid it or no bid
came. If the market's own switch is off (its `/health` says so), the 0 says that instead of "no bid came".
The write is one Redis transaction: the signal is in force with its expiry, or not at all (then the act closes 0,
"not delivered"). A held-back signal (`PRIME_HOLDBACK_FRACTION`) is written as `control`: the market records what it would have
done and does not apply it, so both arms are graded the same way, and `run_prime_report.py` compares the organ's real
starts per hour under treated vs control signals. If the market's record is gone (Redis lost it), the act is left to
the ledger's horizon expiry, which is not counted as Prime's outcome.

Guards: one signal per organ until it is graded, then the organ rests from Prime's signals for as long as the last
signal lasted (so Prime's charge can hold an organ at most half the time); 1-6 hours per signal.

**State** persists in Redis `orion:prime:state` (concerns, open questions, open regulate signals, last reading,
concern record), so a restart resumes. Rolling the image back to before step 3 needs the two new keys stripped
first (the older code refuses unknown keys and would start empty):
`redis-cli -u redis://100.92.216.81:6379/0 GET orion:prime:state | python3 -c "import json,sys;d=json.load(sys.stdin);[d.pop(k,None) for k in ('open_regulations','regulation_rest_until')];print(json.dumps(d))" | redis-cli -u redis://100.92.216.81:6379/0 -x SET orion:prime:state`.

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
| `PRIME_REGULATE_ENABLED` | `true` | offer the regulate move; off = not offered, answers using it refused (open signals still graded) |
| `PRIME_FAMILIES_PATH` | `/repo/config/physiology/families.yaml` | where Prime reads the receptors (read-only repo mount) |

### Prime's concerns from funded event wants (2026-11-04, patch 2)

`PRIME_CONCERNS_FROM_EVENT_WANTS` (default true): the frame's ATTENTION section (Temporal Self arcs,
`GET /temporal-self/frame`) is replaced by EVENTS WORTH EFFORT: the event wants the body economy funded
and that have not expired (durable-runs `GET /physiology/event-wants?hours=PRIME_EVENT_WANTS_HOURS&limit=
PRIME_EVENT_WANTS_LIMIT`, highest net first). A new concern must name one of those `event_id`s
(`PrimeConcernV1.event_ref`); a concern without one is not added, and Prime sees why. Log:
`prime_concern_added concern=... event_ref=...`. An unread menu is a frame gap, never calm. false = the
arc path, unchanged. Replay: `services/orion-durable-runs/evals/event_wants_replay.py`. Spec: docs/superpowers/specs/2026-11-04-attention-as-want-competition-design.md.

### Every turn ends back in Prime: the turn hand-back (2026-11-12, living Prime Part 2)

`PRIME_TURN_HANDBACK_ENABLED` (default true). Prime listens to `orion:cockpit:hop`, the hops Hub already
publishes for every unified turn (no new bus event). It joins each turn's `ingress` hop (session, prompt) to its
`finalize` hop (reply, verdict, the judge's `promises_made` / `promises_kept`) by correlation id and folds the
turn into its state on the tick (`orion/prime/handback.py`, pure). Only four kinds of turn are handed back:
`chat_juniper` (Hub turns Juniper typed: ingress carries `mode`, session not `orion_*`), `curiosity`
(`orion_curiosity`), `consult` (`consult-*`) and `letter` (`orion_day_letter`, used by Part 3). World-pulse
reads, self-sense evals, PR field notes, reminiscence and outreach are not.

- **Thread opened:** a turn with at least one promise (`promises_made`, offers excluded) opens a concern with
  `event_ref = turn:<correlation_id>` and an id derived from the turn. It links the promises
  (`harness_turn_trace:<corr>#promises_made[i]`, i = the promise's own index), never copies them. It places
  NO bet (`act.prime.predict`): nothing can settle it until the intention ledger's D2 gives kept promises act
  ids, so a bet would only close 0 and feed a false failure into the self-monitor gate. Bounds: total
  `MAX_CONCERNS` (3), at most 1 turn-opened concern held, at most 4 opened in 24 h. A refused thread is
  recorded on the turn. Accepted offers (`offers_accepted`) are out of scope until D2.
- **Concern answered:** a turn whose PROMPT cites `prime_concern:<id>` of an open concern and that completed
  with a reply (verdict `completed`) settles it (produced = 1; `CONCERN_BASIS` names this path). A reply that
  only mentions the id settles nothing.
- Only those two wake a thought; the funded-thought rule still decides. Every handed-back turn shows in the
  frame's TURNS THAT HANDED BACK TO YOU section (last 5).
- Turns Prime started: the existing nomination grading closes the act; the run's
  `harness.turn_correlation_id` is linked onto the turn record (`started_by_act`, also when the turn hands
  back after its run settled) and the act's close note.
- `stop()` folds and saves hops heard since the last tick, so a deploy loses none of them.
- `PrimeStateV1.last_thought_id`: the thought that wrote `last_reading`, for the Prime block to cite.
- Record: Redis `orion:prime:turns` (`PrimeTurnsStateV1`: pending turns, last 20 turns, 300 seen ids for
  dedupe), separate from `orion:prime:state` so a rollback still loads Prime's state.
- Log: `prime_turn_handback corr=... who=... opened=... settled=...`; `prime_handback_subscribed`.
- Pub/sub is not durable: a turn that finishes while orion-mind is down is not handed back.
- Replay eval (read-only, live Postgres): `python -m orion.prime.evals.run_turn_handback_replay --days 3`.

### Prime's choice point and tending lines (2026-11-13, personality rev 4)

Spec: docs/superpowers/specs/2026-10-16-orion-dynamic-personality-graph-design.md (rev 4, move 1 and move 4).

- **Choice point (orion/prime/choice.py):** the thought's answer gains `"pick": "<concern_id>" | "none"` in the
  same model call (no new call). After the thought settles, Prime publishes one `personality.choice.v1` on
  `orion:personality:choice` (lane `prime`, `choice_id` `prime:<thought_id>`, act id = the thought act,
  correlation id None): chosen concern + the other held concerns when 2 or more were shown, or
  `declined_all` over every shown concern on "none". The options are the concerns the frame showed. A pick
  that contradicts the move's concern records nothing. A malformed pick never fails the thought. No `pick`
  key: a move's `concern_id` stands in. `market_rank` = the concern's event's position in the body economy's
  funded-event list (highest net first), among the shown concerns; None when its event is not listed (turn
  promises, day-letter threads, arc mode), and all None when fewer than 2 shown concerns are listed. `situation`: binding constraints and the strain dial from BODY
  (Prime sees no arousal reading). Holdback (PRIME_HOLDBACK_FRACTION) draws after the pick, so a held-back
  question still records its choice. A publish failure logs `prime_choice_publish_failed` and the thought goes
  on. The `prime_thought` log line carries `choice=<outcome>` (`chosen:published`, `declined_all:published`,
  `no_concerns`, `one_concern_picked`, `pick_not_shown`, `pick_conflicts_move`, `no_pick`, `publish_failed`). Not behind a flag: it is
  the graph's evidence and costs nothing.
- **Tending lines:** with `PRIME_PERSONALITY_LINE_ENABLED` (ships false until the live labels read cleanly; a label fix is in progress) the frame gains WHAT YOU
  TEND TO CHOOSE, the same <= 2 `tending: <label> [disposition:<id>]` lines Hub's Prime block shows, read from
  `PRIME_DURABLE_RUNS_URL/personality/frame` (active choice nodes with >= 10 weighted choices). An unreadable
  or stale frame is a gap line; none yet reads "(none yet: ...)".

### Orion's consent to a Claude tool call (2026-11-13, Patch 1b)

`app/prime_consent.py` answers `orion:prime:consent:request` from the Claude tools broker
(orion-harness-governor): may Claude in Hub chat read this part of you? Pure parts:
`orion/prime/consent.py`. Design: `docs/superpowers/specs/2026-11-12-claude-hub-tools-orion-consent-design.md`
section 3.

- One model call on `PRIME_CONSENT_LLM_ROUTE` (`prime_consent`: circe chat lane 0, pool priority
  interactive; never fast or metacog), thinking off, 200 tokens, seat wait 15 s then 20 s generation. The
  frame (Prime block, the ask, the invite text, the last five answers in that category, how to set a
  standing grant) is capped at 4,000 characters in code.
- The answer is one JSON object (yes/no, once/conversation, reason). Anything else is a no, and the answer's
  `path` says which no (`unparsed`, `gate_refused`, `no_model`, `error`).
- Each call that reached the model is an act: actor `prime`, verb `infer`, `object_ref`
  `claude_tools:consent:<call_id>`, so the self-monitor faculty is `act.prime.infer.consent`, apart from
  Prime's thoughts. The E1 gate on that faculty can refuse ("Orion's consent faculty is paused").
- Its own bus connection and task per ask; `PRIME_CONSENT_ENABLED=false` = no listener (every ask is a no).

## Held turns run on their own GPU seat

A `MindRunRequestV1` with `gpu_lease` set (orion-thought copies it from the stance request of a durable/background turn, e.g. a world-pulse read) runs all three LLM phases on route `agent` with the hold on `options.gpu_lease`, whatever `MIND_*_MODEL_ROUTE` / `MIND_TURN_MODEL_ROUTE` say (`app/engine.py`, `orion.llm.resource_lease.held_call_route`). Juniper's rule, 2026-11-12: world-pulse work never goes to the fast or the metacog lane. Unheld runs are unchanged. `MindRunRequestV1` ignores unknown fields, so orion-mind and orion-thought may deploy in either order.

A held run's phases each get `MIND_LLM_HELD_TIMEOUT_SEC` (150 s) instead of `MIND_LLM_TIMEOUT_SEC` (60 s). The held seat may be the agent-gpu2 Bonsai-2-27B, which thinks before it answers: on 11-10..12 its served Mind calls took 10-56 s and 18 of 59 hit the 60 s cap. 150 s covers a full 2048-token semantic reply at Bonsai's slow end (~14 tok/s). The run wall (`policy.wall_time_ms_max`, 180 s from orion-thought) still bounds the whole run, so orion-thought's 210 s HTTP wait and Hub's 400 s stance wait are unchanged; a slow first phase leaves later phases less time and they are skipped with `wall_budget_insufficient`.

Held seats invent claim labels (issue #3274, 2026-11-13). The `agent` seat (Strata flash-next) and the `agent-gpu2` seat (Bonsai 27B) return complete semantic claims with real summaries and evidence refs, but they make up `claim_kind` / `anchor` / `recommended_effect` values (`event_claim`, `technical`, `acknowledge`, ...). Until this fix every held semantic phase failed `semantic_schema_invalid` and the run ended `fallback_contract_only`. Two changes in `app/synthesis.py`: the semantic system prompt now lists the allowed values (read from the contract's `Literal` types), and `try_coerce_off_vocab_claim_fields` maps an invented value onto the contract (anchor -> `unknown`, effect -> alias or `no_effect`; kind -> `current_turn_claim` only when a ref cites `current_turn:`, else `situation_claim`). Each coerced claim keeps the original values in `metadata._orion_off_vocab_fields`, and the run notes `normalized_off_vocab_claim_fields`. Nothing else is rescued: a claim without a non-blank `claim_id`/`label`/`summary`/`claim_kind`, with non-dict `metadata`, or with an out-of-range number, still fails `semantic_schema_invalid`, and claims whose evidence refs are not in the pack are still dropped by the guardrail filter. Fixtures: `tests/fixtures/held_semantic_replies_2026-11-13.json`.

The frontier judge runs with thinking off on a held seat (issue #3294, 2026-11-13). The judge is the only Mind phase that ran with thinking on (`MIND_LLM_THINKING_APPRAISAL=true`). On the held seats the reasoning used the whole `MIND_LLM_MAX_TOKENS_APPRAISAL` (3072) budget, and 12 of 13 held runs ended `invalid_handoff`:
- Bonsai 27B (agent-gpu2) wrote ~9 KB of reasoning; the JSON was cut off (`json_parse_failed`) or never started (`empty_llm_response`).
- Strata flash-next (agent) thought to its 2304-token thinking budget, the answer was cut at 3072, and Strata refused the incomplete structured output with HTTP 502. Mind saw `empty_llm_response` from model `Active-GGUF-Model` (the gateway's default label when no model answered).

A held run now sends the judge `enable_thinking=false`, like the semantic and stance phases (`MIND_LLM_THINKING_APPRAISAL_HELD`, default false; true = rollback). Unheld runs are unchanged. The same captured requests, replayed on the same seats with thinking off, return complete frontiers in 357-846 tokens and 9-23 s. Smaller changes in `app/appraisal.py`:
- The judge prompt lists the allowed `matter_kind` / `recommended_effect` values (from the contract `Literal`s).
- A `matter_kind` that is a claim kind (`task_claim`, `uncertainty_claim`) maps like a claim kind instead of becoming `turn_anchor`. Each replaced value is kept in `metadata._orion_off_vocab_fields`, and the frontier notes `normalized_off_vocab_matter_fields`.
- A `deferred` / `suppressed` entry that does not fit its contract (for example a free-text suppress reason) is dropped and counted (`dropped_invalid_deferred:N`, `dropped_invalid_suppressed:N`) instead of failing the whole reply. Object-shaped hazards use their own text field; an object without text is dropped.
- Telemetry names the cause: an empty reply carries the gateway's reason (`empty_llm_response:upstream_http_5xx:http_502`), and a schema failure carries the field (`frontier_schema_invalid:selected.0.score`).

A reply with no real matter (no label, summary, or claim id) or with an out-of-range number still fails. Fixtures: `tests/fixtures/held_frontier_judge_replies_2026-11-13.json`.

### The day letter is Prime's end-of-day turn (2026-11-12, living Prime Part 3)

`PRIME_DAY_LETTER_ENABLED` (default true; the kill switch for the whole end-of-day move -- Hub's letter timer
is retired, so off means no letters). Pure parts: `orion/prime/day_letter.py`; I/O: `app/prime_loop.py`
`_day_letter_*`. No thought is spent on it: the tick decides.

1. **Day close.** Every 5 min the tick asks whether yesterday (America/Denver) is closed --
   `GET {PRIME_DURABLE_RUNS_URL}/temporal-self/day/<date>` answers 200 once the temporal chronicle drained it --
   and whether Hub has a letter for it (`GET {PRIME_HUB_URL}/api/orion-day/letters/<date>`). Closed with no
   letter: Prime holds it as its own end-of-day concern (`PrimeDayLetterOpenV1`), tracked apart from the
   3-slot concern list so a full list never crowds out the letter.
2. **No bid (tier 0).** Juniper, 2026-11-12: "It needs to happen each day." The letter is not priced, never
   deferred by the market and not asked of the self-monitor gate. (Live 11-11: a tier-2 bid was deferred 10 h,
   then one failed run ended the day.)
3. **The act and the order.** The `make` act opens at once (actor `prime`, object `orion_day_letter:<date>`),
   and Prime POSTs `OrionDayLetterOrderV1` to Hub (`/api/orion-day/letter-run`): run id
   `orion-day-<date>-<attempt>`, a correlation id derived from the date and attempt, the act id, the run deadline
   (`PRIME_DAY_LETTER_RUN_DEADLINE_HOURS`, 12, shared by every attempt) and the missed days not yet told to a
   letter. Hub gathers the day and submits ONE durable unified turn (`orion_day.letter`) at GPU pool priority
   `system` (above background work, below Juniper's chat). An unanswered order is re-sent (Hub answers
   `exists`); Hub's `already_written` / `empty_day` / `refused` close the act at 0.
4. **Settle.** `GET {durable}/runs/<run_id>`: a run that wrote nothing (failed, cancelled, no row) is ordered
   again as the next attempt, up to `PRIME_DAY_LETTER_MAX_ATTEMPTS` (3), while the deadline holds. Otherwise
   terminal -> the act closes (1 = a run wrote the row), and the letter's threads (the run's finish detail) open up to 2 Prime concerns (never past 3 in
   total; the rest are counted as without room). They place the usual concern bet, and the note wakes a
   thought, so curiosity hears the letter only through Prime's own graded `ask`. The letter turn's correlation
   id is linked onto the hand-back record.
5. **Missed days** (every attempt failed, Hub refused) ride on the next order's `missed_letters`, once; Hub
   adds the day before by itself when it has no row and Prime did not list it (Prime was down).

- Record: Redis `orion:prime:day_letter` (`PrimeDayLetterStateV1`: the open day, the last 14 days).
- Logs: `prime_day_letter_due`, `prime_day_letter_opened`, `prime_day_letter_act_open_failed`,
  `prime_day_letter_ordered`, `prime_day_letter_retry`, `prime_day_letter_own_attempt_wrote_it`,
  `prime_day_letter_closed`.
- A run that wrote the row and then ended failed or cancelled (a later step failed) is a written letter. A cancelled
  run with no row is not retried: a cancel is someone's decision.
- Deploy order: orion-hub, then orion-durable-runs, then orion-mind (Prime orders only once both are live).

### Intentions Prime takes on or lets go (2026-11-13, choice-as-EFE Step -1)

`PRIME_INTENTION_TAKE_ON_ENABLED` (default true). Design: `docs/superpowers/specs/2026-11-13-choice-as-expected-free-energy-design.md`
(Step -1). Pure parts: `orion/prime/intention.py` and one adapter per source in `orion/prime/intention_sources/`
(today: `turn_promise.py`); I/O: `app/prime_loop.py` `_take_on`, `_intention_*`.

Today a promise Orion makes in chat becomes a Prime concern, and nothing could carry it out. Now Prime has three
choices about such a concern:

1. **Take it on.** A new move, `{"kind": "take_on", "concern_id", "why", "not_before"?}`, offered only on a concern
   whose source has an adapter (the frame marks it "you can take this on"). Prime opens a `make` act (actor
   `prime`, object `intention:<concern_id>`); that act is the graded bet, since a turn concern places none. Prime
   holds the order until `not_before` itself, then POSTs `IntentionOrderV1` to Hub (`/api/intention/run`). Hub reads
   the promise (the exact quote from `harness_turn_trace`, the one exchange from `chat_history_log`), builds the
   brief and submits ONE `intention.carry` durable run (run id `intention-<concern_id>`; one per concern, no retry
   in v1). The run binds in the body economy as family `durable.intention`, tier 1 (Juniper 2026-11-13: once Orion
   chose, follow-through is keeping its word; the reserved share stops curiosity and consults crowding it out, and
   the economy still picks the time). Its admission deadline is the concern's horizon: promise time + 72 h.
2. **Let it go.** The existing concern `drop` with why. With an open take-on, the act closes 0 (`let go: <why>`)
   and the reply is never delivered.
3. **Not yet.** The existing `keep`, or no edit. A concern with an adapter that was never taken on ages out at its
   horizon.

The carry turn runs in session `orion_intention` with `prime_concern:<id>` first in its prompt, so the turn
hand-back settles the concern when it completes (`classify_who` -> `intention`). Loop guard: a promise made inside
that session opens no concern (and the adapter refuses a promise made in any `orion_*` session). When the run is
terminal Prime POSTs `IntentionDeliverV1` (`/api/intention/deliver`): Hub writes the reply into the originating chat
thread and makes one outreach attempt through the normal gates (quiet hours, the shared daily cap, the economy). A
blocked attempt still leaves the reply in the thread. The act closes 1 when the reply is stored, else 0. A concern
taken on does not fade while its run is open.

- Record: Redis `orion:prime:intentions` (`PrimeIntentionsStateV1`; no quote text is ever stored in Prime's state).
- Logs: `prime_intention_taken`, `prime_intention_ordered`, `prime_intention_order_unanswered`,
  `prime_intention_closed`, `prime_intention_open_failed`, `prime_intention_step_failed`.
- Deploy order: orion-sql-writer, orion-hub, orion-actions, orion-cortex-orch, orion-durable-runs, then orion-mind.
