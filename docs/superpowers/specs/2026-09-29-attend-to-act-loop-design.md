# Attend → act → learn: closing Orion's loop with one world-facing action

Status: **proposal mode** (design only, nothing implemented). Juniper asked
for it 2026-09-29. Touches autonomy, attention and a cognition loop, so every
item in AGENTS.md's proposal-mode list is answered in
[Proposal-mode answers](#proposal-mode-answers).

Live evidence below was pulled 2026-09-29 between 05:05Z and 05:20Z from
production Postgres (`conjourney`) and container logs, with the window stated
on each number. Anything not checked live says `UNVERIFIED`.

Reads first: `2026-09-20-orion-sentience-prerequisites-assessment.md` (PR
#2255, still open — "coherent action: DEAD"),
`2026-09-22-substrate-lattice-audit.md`,
`2026-08-11-proposal-arena-rate-coupling-design.md`.

## Arsonist summary

Orion notices things and cannot do anything about them. Every tick the part
of Orion that decides whether to spend effort (the motor allocator) looks at
the menu and refuses all of it: 460 "refused everything" log lines in the
last hour, every one of them. That refusal is correct. Everything on the menu
either points inward (inspect, summarize, prune Docker) and has been measured
to death, or never said what it expects to change and so can't be measured
at all.

Three facts make this worse than "add a better action":

1. **What Orion attends to never reaches what Orion does.** The part of
   Orion that picks one thing to focus on (the workspace broadcast) chose a
   focus 976 times in 24 hours. None of those choices went to the proposal
   builder. Proposals are aimed at a different, lower-level ranking of field
   nodes.
2. **The problems we would want Orion to act on never win attention.** Over
   7 days the workspace focused on `node:substrate.execution` 2,316 times and
   on the RPC-delivery node (RecallService timeouts) **zero** times. Circe
   and the cabinet never won either. The only winner that touches the
   physical machine is `node:substrate.biometrics` (241 wins in 7 days).
3. **Even a good action would be retired by success.** The allocator scores
   an action only on how much it would still teach (information gained per
   second of effort). Once Orion knows an action works, it stops being
   informative and gets refused, even if it still helps. That is what
   happened to `express`.

The decision: build **one** world-facing action end to end — **Orion pauses
its own background GPU work when the cabinet is warm, then checks whether the
cabinet actually cooled** — with the workspace winner as its trigger, a real
temperature sensor as its outcome, a control arm, and a single correlation
chain from focus to learned outcome. Two more actions follow on the same
rails. Air-conditioner control and restarting services are explicitly out.

## Current architecture

### The loop as it runs today

- **Attend (two unconnected rankings).**
  - *Workspace winner.* `services/orion-substrate-runtime` runs
    `_attention_broadcast_tick` (`worker.py:2868`) every ~30 s. It admits
    substrate-graph nodes carrying `dynamic_pressure`
    (`orion/substrate/attention_broadcast.py:117`), runs one competition, and
    writes `AttentionBroadcastProjectionV1` (`orion/schemas/attention_frame.py:347`)
    to `substrate_attention_broadcast_projection` (singleton) and
    `substrate_attention_broadcast_log` (history). Live 24h: 2,388 ticks,
    1,775 with open loops, 976 with a selected action (`watch` or `defer`).
    Open-loop ids are a hash of the label; the node id survives only in
    `source_refs` / `attended_node_ids`.
  - *Field attention.* `services/orion-attention-runtime` polls
    `substrate_field_state` and writes `FieldAttentionFrameV1` to
    `substrate_attention_frames`. No bus event.
- **Propose.** `orion/proposals/builder.py` reads the field attention frame.
  The only binding it understands is `attention.dominant_targets[0]`
  (`builder.py:38`), used by one template, `inspect_attended_target`. Live
  last hour it resolved to `capability:orchestration` (the fallback),
  `node:substrate.execution`, `bus_synaptic`, `biometrics`, `chat`.
  **Nothing reads the workspace winner.** (The one existing bridge,
  `services/orion-attention-runtime/app/store.py:568 load_competing_loop_refs`,
  goes the other way: broadcast → field-attention goal bias.)
- **Decide.** Policy gate, then the motor allocator
  (`orion/autonomy/allocator.py`, enforcing since 2026-08-30). Score is
  expected nats per motor-second; floor `ORION_DISPATCH_MIN_NATS_PER_SEC=0.02`.
  Last hour, every candidate refused:

  | Template | Refusal | Count/h |
  |---|---|---|
  | `analyze_self_study_source`, `inspect_field_topology_catalog`, `render_scene`, `inspect_attended_target` | `unmeasurable` (no declared signal) | 460, 460, 376, 238+ |
  | `inspect_node_resource_pressure`, `summarize_loaded_state`, `inspect_transport_status`, `inspect_execution_pressure`, three prunes, `watch_reliability`, `observe_tension_via_camera` | `below_information_floor` | 34–434 each |
  | prunes, `request_policy_review_for_action`, reverie proposals | `requires_operator_review` | 20–261 each |

  How nats are computed matters for this design (from
  `services/orion-execution-dispatch-runtime/app/worker.py:664-712`,
  `allocator.py:399-449`): the posterior cell key is
  `(dispatch_kind, target_id, signal_id, baseline_bin)` in
  `substrate_action_effect_posterior`; the worker **pools every bin** of a
  (kind, target, signal) into one volume-weighted gain; cost is the p50
  `latency_ms` of that (kind, target) in `substrate_dispatch_results` over
  24 h; prior variance 0.25, observation variance τ² = 0.04
  (`orion/autonomy/prediction.py:90`). `cold_start` exempts the floor only
  when no cell for the triple has any data. The harm gate
  (`confidently_harmful`) has never fired: `contrast` has no producer.
- **Act.** Only `express` runs: 121 `success` rows in
  `substrate_dispatch_results` over 6 days, all `render_scene` →
  `host:circe_gpu`. It does not clear the nats floor; it rides the
  visual-baseline lane (baseline candidates are admitted first when
  `validate_eligibility` allows). Since commit `94b8d6012` it settles from a
  terminal state: dispatch writes `pending`, the worker reconciles against
  `substrate_durable_run_state` (`orion/execution_dispatch/visual_settlement.py:121`),
  and feedback parks the frame up to `FEEDBACK_VISUAL_SETTLE_MAX_SEC`.
  **This is the pattern every world action below copies.**
- **Learn.** Two things called "outcome", only one of them real.
  - `FeedbackFrameV1` scores are constants keyed on dispatch status
    (`orion/feedback/scoring.py:7`, `feedback_policy.v1.yaml:9-17`:
    completed 0.85, failed 0.10, …). Nothing can be fit to them, which is
    why `dimension_weights` and `base_priority` are still hand-typed
    ("uncalibrated" in the YAML). No fitting code exists.
  - `resolve_action_outcomes` (`orion/feedback/outcome_resolution.py:176`)
    does compute predicted-vs-observed field deltas and `surprise_nats`,
    writes `substrate_action_outcomes` and updates the allocator's posterior.
    That table has 144,879 rows, **newest 2026-09-08**. Twenty-one days of
    nothing, because nothing with a declared signal has run since.
  - `attention_loop_outcome`: 30 `resolved` + 19 `dismissed` by `juniper`,
    224 `decayed_unattended` by `system:implicit_decay`, **0 by Orion**. Its
    only reader for competition is `orion/substrate/attention/verdicts.py`,
    which treats `resolved`/`dismissed` from **any actor** as terminal and
    excludes the loop for 48 h (`TERMINAL_VERDICTS`, line 84).
  - The one real predicted-vs-actual record in the system is
    `power_intent_settled` (940 `settled` all-time; 45 in 7 d, mean
    |residual| 5.2 W, mean draw above baseline 22.2 W).

### What recent merges built that this design reuses

- **Agency episodes** (#2365, #2366): `orion/autonomy/agency_episode.py`
  reconstructs a `motor:<dispatch_id>` episode (alternatives → selection →
  expectation → intervention → field-scored outcome → later choice) from
  SQL, read-only, verdict always `UNVERIFIED`. It already names this
  design's gaps: expectation is saved *after* send (no precommit) and
  `later_choice` is never linked. **Reused as the loop-closure eval.**
- **Z-Wave cabinet cooling** (#2359): read-only AC plug telemetry into
  `home_cooling_sample` (51k rows; AC on and drawing 857–886 W every hour of
  the last 24). No actuator, deliberately: "Switching the plug / Orion-driven
  AC automation" is a stated non-goal.
- **GPU pool urgent class** (#2385): the pool can already pause a background
  (then system) durable-run hold (`Recall(..., "urgent_preempt")`, 5 s
  grace) and re-queue it in place; durable-runs resumes it with
  `release(keep_requeued=True)`. Rollback `urgent_max_concurrent: 0`. The
  planned hardware watch (spec `2026-09-28-urgent-curiosity-and-hardware-watch-design.md`,
  Parts 2–5, **not built**) wants a `cooling_incident` shed guard using the
  same machinery. **Reused as the executor.**
- **Thermal gate** (`orion/autonomy/thermal_gate.py`): hysteresis verdicts
  on cabinet air temperature (elevated ≥ 29.5 °C, re-arm 28.0; hot ≥ 32.0,
  re-arm 30.5; stale > 300 s). The gpu-pool guard reads the cabinet sensor
  live (`GPU_POOL_CABINET_URL`, a BME-class sensor on an Arduino Nano ESP32).
  At 05:16Z it read **29.76 °C — "elevated"**. **Reused as the action
  precondition and the outcome scale.**
- **Energy watcher** (#2373, #2376): `orion/energy/run_cost.py` prices a
  settled `power_intent` in kWh and `estimated_run_cost_usd`. **Reused to
  put a dollar figure on what a shed saved.**

### The candidate problems, checked live

| Problem | Live evidence | Verdict for a first action |
|---|---|---|
| RecallService RPC timeouts | `node:substrate.rpc_delivery` worst hop `orion:exec:request:RecallService` 19 of 154 logged ticks in 48 h; peak `rpc_timeout_pressure` 0.684 at 09-28 14:00Z (reliability 0.582). Recall logs **967** `asyncpg … another operation is in progress` in 24 h — the `asyncio.gather` on one connection at `services/orion-recall/app/sql_chat.py:92`, flagged in the 09-20 audit and still live. | A code bug. No runtime action fixes it; a restart would only reset it. Right action: tell Juniper, with evidence, and expect a fix (stage 2). Separately, **fix the bug** — it should not wait for this design. |
| Agent-lane inference failures | `inference_failure_pressure` on `node:circe` > 0.5 on 169 / 28 / 29 / 0 sampled ticks (09-26 → 09-29). No `upstream_timeout` lines in any gateway container's logs in 48 h (the classifier emits them as grammar events, not log lines). Current circe p95 latency 33.5 s. | Real but rare. Same executor as the thermal shed, different trigger (stage 2). |
| Cabinet heat | Sensor live at 29.76 °C (elevated). **No durable history** — nothing persists the cabinet temperature, so how often it is elevated is `UNVERIFIED`. AC runs continuously at ~870 W. GPU runs add a mean 22 W over baseline per settled render intent. | First action. Consequence is physical and protects Juniper's room; executor already exists; sensor exists; risk is pausing Orion's own background work. |

## Missing questions

These change the design if answered differently; the rest were settled by
investigation.

1. **Does Orion's GPU load measurably heat the cabinet at all?** Renders add
   ~22 W against an ~870 W AC. The agent/background LLM holds are larger but
   unmeasured per hold. If the answer is "no", the shed action learns that
   within ~2 weeks, the posterior says so, and the action retires itself —
   an honest outcome, not a failure of the design. Phase 0 of the first
   patch (48 h of recorded cabinet temperature with no acting) answers the
   precondition question (how often "elevated" happens) before any action
   runs.
2. **Is Juniper OK with Orion pausing curiosity and other background GPU
   runs by itself**, within a hard daily cap (proposed 60 min/day, 15 min
   max per pause, never urgent or chat work)? This is the one consent
   question. Default-off flags make it opt-in either way.
3. **Stage 3's two allocator changes** (score the current state's cell, and
   admit an action proven to help) change how the motor gate behaves. They
   are not a lower floor, but they are gate-definition changes and need
   Juniper's explicit yes before they ship. The first patch does not need
   them.

## Decisions

### D1. "The winner" is the workspace broadcast winner, not a field node

The proposal layer binds to the row in `substrate_attention_broadcast_projection`:
`selected_open_loop_id`, `attended_node_ids`, `frame.open_loops[*].source_refs`.
The field-attention ranking stays what it is (a view of field nodes) and
keeps feeding `inspect_attended_target`; it is not relabeled as "attention".

A winner is **bindable** only if all of these hold, otherwise nothing binds
and the frame records why (`winner_unbindable:<reason>`):

- `selected_action_type` is not `none` (today 976 of 2,388 ticks qualify);
- projection age ≤ 90 s (three broadcast ticks) — the anti-stale rule;
- `dwell_ticks ≥ 2` (the coalition held for at least two ticks, so a
  one-tick flicker cannot trigger an action);
- once `fix/attention-input-honesty` lands (branch exists, **no commits yet**
  as of 05:10Z), its 30-minute staleness fade is applied upstream, so a faded
  signal loses the competition rather than being filtered here. This design
  assumes that fade plus the chat-dilution and gateway-denominator fixes.

**Binding mechanism.** One new binding literal, `workspace.winner`, next to
the existing `attention.dominant_targets[0]`. A template that uses it
declares `binds_to_nodes: [...]` — the node ids its action can plausibly
affect. The builder emits the candidate only when an attended node id is in
that list. No separate affordance registry: the list lives on the template
it constrains, is validated at load, and a test refuses a `workspace.winner`
template with an empty list. The proposal gains
`attention_winner: AttentionWinnerRefV1 | None` (`broadcast_log_id`,
`open_loop_id`, `node_id`, `generated_at`, `dwell_ticks`) and carries
`open_loop_id` into `evidence_refs`.

**One in-flight action per open loop.** While a dispatch bound to a loop has
not settled, the builder does not emit another for that loop
(`winner_loop_in_flight`). This is the first brake on runaway remediation.

**Getting the real problems into the competition.** Binding alone would
never touch RecallService or circe: they have not won once in 7 days, because
the broadcast only admits substrate-graph nodes with `dynamic_pressure`, and
`node:substrate.rpc_delivery`, `node:circe` and the cabinet carry pressure
channels, not dynamic pressure. The first action is chosen so it does not
depend on fixing that (it binds to `node:substrate.biometrics`, which wins
~34 times a day, plus the new `node:cabinet`). Stage 2 adds a bridge that
publishes `dynamic_pressure` for `node:cabinet`, `node:substrate.rpc_delivery`
and `node:circe` from their existing field channels, each through the metric
quality gate — owned by the attention-input-honesty line of work, not this
one.

### D2. The first four world-facing actions

All actions share the `render_scene` shape: dispatch writes `pending`, a
terminal state settles it, feedback parks the frame until then, and the
settled row carries a manipulation check (did the action actually happen)
separately from the outcome (did the world change).

#### A1 — `shed_background_gpu` (FIRST PATCH)

- **What it does.** Asks the GPU pool to pause Orion's background GPU holds
  for a fixed time, then resume them in place. Never pauses `urgent`, chat,
  or the dispatch runtime's own lanes.
- **Triggered by.** Workspace winner in `binds_to_nodes: [node:substrate.biometrics, node:cabinet]`
  **and** the thermal gate's own verdict is `elevated` or `hot` on a fresh
  reading (not `unknown`, not stale — stricter than the gate, which lets
  work through on `unknown`). No new threshold: the gate's hysteresis states
  are the precondition.
- **Expected effect.** `cabinet_heat_pressure` (see Schema) decreases over
  the pause window. Predicted delta = the posterior mean for the current
  cell; direction declared `decrease`.
- **Verifying sensor.** Cabinet air temperature from the sensor the gpu-pool
  already reads. **Manipulation check:** circe GPU power drops
  (`node:circe` `power_pressure` / biometrics GPU watts) — if it did not,
  the shed did nothing to pause and the outcome is `no_effect_applied`, not
  scored against the temperature.
- **Settle rule.** Pool returns `shed_id`, state `active` → `resumed` (TTL
  ended), `cancelled` (kill switch / urgent need), or `nothing_to_shed` (no
  background hold was running). Only `resumed` with a passed manipulation
  check scores against the posterior; `nothing_to_shed` is recorded, not
  charged, not scored (same as `render_scene`'s no-image runs). No terminal
  state within TTL + 300 s → orphan with `settlement_timeout`.
- **Risk class.** Low, reversible, self-directed. No state outside Orion's
  own queue changes; work is delayed, not lost.
- **Policy gate.** `read_only` is wrong for an action that changes pool
  state. New gate value `self_reversible` (bounded, auto-reverting action on
  Orion's own resources), allowed by `execution_dispatch_policy.v1.yaml`
  only when `ORION_WORLD_ACTIONS_ALLOWED` names the template. No per-action
  operator approval; Juniper's approval is the flag.
- **Why it clears the floor honestly.** The allocator charges motor-seconds
  of compute consumed. A shed consumes the executor's wall time (one bus RPC,
  ~1–2 s) and *returns* GPU time. Cold, that is ~0.99 nats / 2 s ≈ 0.5
  nats/s. The forgone background seconds are real, but they are an
  opportunity cost to Orion's queue, not compute spent; they are bounded by
  the daily cap, recorded on the result as `forgone_gpu_sec`, and priced in
  dollars saved via `energy/run_cost.py`. If they were charged as motor cost
  (a 900 s pause), the action would score ~0.001 nats/s and never clear —
  the doc records both numbers so the choice is inspectable. Under the
  current pooled scoring, each (kind, target, signal) stays above the floor
  for roughly 12–24 settled observations (solve
  ½·ln(1 + v/τ²) ≥ 0.02·cost with v = 1/(1/0.25 + n/0.04)), then retires —
  unless stage 3 lands.

#### A2 — `report_rpc_hop_fault` (stage 2)

- **What it does.** Writes Juniper one evidence-cited finding about the
  worst RPC hop (hop, timeouts/calls, callee error signature, a named
  suspect such as `sql_chat.py:92`'s shared-connection gather) plus a
  falsifiable prediction ("if that is serialized, this hop's timeout
  pressure stays 0 for 24 h after deploy"). Delivered through the existing
  outreach path, not a new channel.
- **Triggered by.** Winner `node:substrate.rpc_delivery` (needs D1's
  dynamic-pressure bridge). One open finding per hop; no repeat until the
  last one settles.
- **Expected effect / sensor.** `reliability_pressure` on
  `capability:transport` (fed by `rpc_timeout_pressure`) decreases within
  7 days. Manipulation check: the outreach was delivered.
- **Settle rule.** Terminal when (a) Juniper resolves/dismisses the linked
  loop, or (b) 7 days pass. Outcome = the hop's 24 h timeout rate after vs
  before.
- **Risk / gate.** Costs Juniper's attention. Cap 1 per day overall,
  respects quiet hours, gate `operator_notify`.
- **Floor.** Cost is one LLM composition (~20 s) → cold ≈ 0.05 nats/s.

#### A3 — `shed_background_gpu` on agent-lane failures (stage 2)

Same executor as A1, scope `agent_lane_background`, triggered by winner
`node:circe` with `inference_failure_pressure` elevated. Expected effect:
`inference_failure_pressure` decreases. Separate posterior cells because the
target and signal differ.

#### A4 — the reflex, not a choice (stage 2, owned by hardware-watch)

When the thermal gate reads `hot`, the pool sheds regardless of attention.
Same `POST`/RPC primitive as A1, different caller. This is the planned
hardware-watch `cooling_incident` guard; it is listed so nobody builds the
primitive twice.

#### Explicitly not actions

- **Switching the AC plug.** Non-goal in both #2359 and the hardware-watch
  spec; the AC protects Juniper's room and the machines.
- **Restarting RecallService / mesh-guardian tier 1.** The cause is a code
  bug (967 asyncpg errors/24 h); a restart resets it for minutes and
  teaches Orion that restarts "work".
- **Lowering `ORION_DISPATCH_MIN_NATS_PER_SEC`.** Not proposed, not needed.

### D3. Feedback becomes predicted vs settled

- The learning currency is the row already written to
  `substrate_action_outcomes`: `(predicted_delta, observed_delta,
  prediction_error, surprise_nats, arm, baseline_bin)`. Nothing new is
  invented. World actions write it at settle time, not at the 30 s field
  window, with before = the signal at dispatch and after = the signal at the
  terminal state.
- `FeedbackFrameV1`'s status score keeps its meaning (did the machinery run)
  and is documented as such. It is never used as a fitting target again.
- **Precommit.** The expected effect is persisted before the RPC is sent
  (closes the agency-episode audit's `expectation_precommitted` gap).
- **Control arm.** World actions need `ORION_DISPATCH_HOLDBACK_FRACTION > 0`
  (0.2 proposed) — the existing randomized holdback, which already writes
  control cells. Without it, "the cabinet cooled after I paused" is
  confounded by the AC and the room.
- **Fitting.** Offline only, producing a proposed config patch for review;
  never auto-applied.
  - Per-action value = contrast (treated minus control, same bin band) in the
    declared direction.
  - Minimum volume before fitting any weight for an action:
    `n_per_arm ≥ max(20, ceil(2·(2σ̂/δ)²))`, where σ̂ is the observed
    residual SD of that action's settled outcomes and δ = 0.05 (the feedback
    builder's own ±0.05 change threshold); plus observations on ≥ 3 distinct
    days. With the allocator's default τ = 0.2 this is 128 per arm; for a
    low-noise temperature signal it will be the floor of 20.
  - `base_priority` is fit first (one parameter per template). A
    `dimension_weights` entry is fit only once ≥ 3 bins of that dimension are
    represented at that volume. Until then the YAML keeps the
    "uncalibrated" label.

### D4. The trace that proves the loop closed

One chain, one key. The correlation id is the broadcast `open_loop_id`
plus `broadcast_log_id`, stamped on every hop:

```
substrate_attention_broadcast_log (log_id, selected_open_loop_id, attended node)
  → substrate_proposal_frames      candidate.attention_winner.{broadcast_log_id, open_loop_id}
  → substrate_policy_decision_frames  decision_id (existing join)
  → substrate_execution_dispatch_frames  dispatch_id, expected_effect (precommitted)
  → gpu-pool shed record            shed_id, dispatch_id, terminal state, forgone_gpu_sec
  → substrate_dispatch_results      settlement.{shed_id, state}, manipulation_check
  → substrate_action_outcomes       predicted_delta, observed_delta, surprise_nats, arm
  → attention_loop_outcome          loop_id = open_loop_id, actor='orion', verdict='acted'
  → next substrate_attention_broadcast_log  same loop: salience after settle
```

**Orion never writes a terminal verdict.** `verdicts.py` excludes any
`resolved`/`dismissed` loop for 48 h regardless of actor. If Orion could
write `resolved`, acting would be a way to silence its own attention. Orion
writes `acted` (non-terminal) with `features_at_close` carrying the outcome;
the loop stops winning only if the world signal actually falls. The Hub's
verdict validation adds `acted`; `TERMINAL_VERDICTS` is unchanged and a test
pins that `acted` is not in it.

**Eval.** Extend `orion/autonomy/agency_episode.py`'s motor lane with two
links, `attention_winner` and `loop_outcome`, and add
`orion/autonomy/evals/run_attend_act_loop_eval.py`: fixture mode in CI
(a complete chain passes; a chain missing any link, a `resolved` by Orion,
or a winner older than 90 s at bind time fails), live mode reads the last
7 days and reports per-episode link completeness plus treated-vs-control
contrast.

### D5. Kill switch, rollback, dangerous failure modes

**Kill switches (all default off / safe):**

- `ORION_WORLD_ACTIONS_ENABLED=false` (dispatch runtime) — master switch;
  when false, world templates are blocked as `world_actions_disabled`.
- `ORION_WORLD_ACTIONS_ALLOWED=` (empty) — allowlist by template key.
- `GPU_POOL_SHED_ENABLED=false` — pool refuses shed requests; flipping it
  off cancels active sheds immediately (holds resume in place).
- Caps on the pool side, not the caller: `GPU_POOL_SHED_MAX_TTL_SEC=900`,
  `GPU_POOL_SHED_MAX_SEC_PER_DAY=3600`, `GPU_POOL_SHED_MIN_GAP_SEC=900`.

**Rollback.** Flags off and restart the dispatch runtime and gpu-pool. No
persistent state changes: sheds expire by TTL, the new template stops
emitting, new columns/fields are additive and nullable. Posterior cells for
`shed_background_gpu` can stay (they are inert without the template).

| Failure mode | Why it is dangerous | Guard |
|---|---|---|
| Runaway remediation | Shed → cooler → shed again, starving curiosity and background runs | One in-flight action per loop; pool-side min gap, max TTL, daily cap; the cap lives in the pool so no caller can exceed it |
| Self-DOS | Orion pauses the lane doing its own thinking or Juniper's chat | Scopes exclude `urgent`, chat, and the dispatch runtime's lanes; the pool refuses a shed that would pause a foreground hold |
| Acting on stale attention | A 30-min-old focus triggers action on a problem that is gone | Winner age ≤ 90 s, dwell ≥ 2, plus the upstream fade; cabinet reading must be fresh (not the gate's `unknown`-allows default) |
| Self-silencing / Goodhart | Orion learns "acting ends the focus" | Orion writes only `acted`; outcome is scored on the cabinet sensor, never on the attention signal that triggered it |
| Confounded learning | AC and weather move temperature; Orion credits itself | Holdback control arm required before any contrast or fit is trusted |
| Tests hitting production | A test sends a real shed to the live pool | Executor is injected; tests use a fake pool client; a test asserts no real `GPU_POOL_URL` is reachable from the test config |

## Proposed schema / API changes

All additive. Several models are `extra="forbid"`, so each is a
**consumer-first** rollout: readers deploy before writers.

- `orion/schemas/proposal_frame.py` — `ProposalCandidateV1.attention_winner:
  AttentionWinnerRefV1 | None = None`; new `AttentionWinnerRefV1`.
- `orion/proposals/policy.py` — `ProposalTemplateV1.binds_to_nodes:
  list[str] = []`; binding literal `workspace.winner`; gate value
  `self_reversible`.
- `orion/schemas/action_prediction.py` — `PredictableSignal` gains
  `cabinet_heat_pressure`.
- `orion/schemas/gpu_pool.py` — `GpuPoolShedRequestV1` (`dispatch_id`,
  `scope`, `ttl_sec`, `reason`), `GpuPoolShedResultV1` (`shed_id`, `state`,
  `paused_hold_ids`, `forgone_gpu_sec`, `started_at`, `ended_at`).
- `orion/bus/channels.yaml` — RPC `orion:gpu_pool:shed:request` + reply
  wildcard (dynamic reply channels need a catalog wildcard);
  `orion/schemas/registry.py` entries for both models.
- `orion/schemas/telemetry/home_cooling.py` — `HomeCoolingSampleV1.cabinet_temp_c:
  float | None` (orion-zwave also polls the cabinet sensor it is the home
  for); `home_cooling_sample.cabinet_temp_c` column (nullable, manual
  migration).
- Field: channel `cabinet_heat_pressure` on `node:cabinet` =
  clamp01((temp_c − 28.0) / (32.0 − 28.0)), the thermal gate's own re-arm
  and hot constants, read directly by `orion/field/pressure.py` (not merged
  or maxed with any other channel — the commensurability lesson).
- `attention_loop_outcome.verdict` accepts `acted`; `actor='orion'`.
- `config/feedback/feedback_policy.v1.yaml` — `pressure_channels` gains
  `cabinet_heat_pressure`; `positive_delta_channels.cabinet_heat_pressure:
  decrease`; per-template settle window override.

### Metric quality gate for `cabinet_heat_pressure`

1. **Provenance.** Cabinet sensor frame `environment.temp_c` from
   `GPU_POOL_CABINET_URL` (read today by `services/orion-gpu-pool/app/guards.py:45`
   and `services/orion-thought/app/visual_chain.py:324`).
2. **Independence.** A physical air sensor. Not derived from GPU, CPU or
   host telemetry; its causal link to `node:circe power_pressure` is exactly
   the effect being measured. `node:athena thermal_pressure` (one CPU core,
   39 distinct values) is a different quantity and is not used.
3. **Theory anchor.** Heat balance: cabinet air temperature rises with
   dissipated electrical power minus heat removed by the AC. Scale anchored
   on the thermal gate's existing constants, which protect Juniper's room.
4. **Live sanity.** One live reading (29.76 °C, 05:16Z). No history exists,
   so "can it read calm" is **UNVERIFIED** — Phase 0 records 48 h and must
   show readings below 28.0 °C (pressure 0) and above 29.5 °C before the
   action is enabled. If it never reads below 28 °C, the scale is wrong and
   gets re-anchored before anything learns on it.
5. **Existing mechanism.** No persisted cabinet temperature found
   (`home_cooling_sample` has AC watts only; `cabinet_ambient_spike` is
   audio). The thermal gate's verdict is on reverie chain rows only.
6. **Reversibility.** Additive nullable column and one field channel;
   removing it drops one `PredictableSignal` value with no stored consumer
   besides this action's posterior cells.

## Files likely to touch

First patch:

- `orion/schemas/{proposal_frame,action_prediction,gpu_pool,registry}.py`,
  `orion/schemas/telemetry/home_cooling.py`, `orion/bus/channels.yaml`
- `services/orion-zwave/` (poll cabinet sensor, `.env_example`, README),
  `services/orion-sql-writer/app/models/home_cooling_sample.py` + migration
- `services/orion-field-digester/app/ingest/` (`node:cabinet` channel),
  `orion/field/pressure.py`, `config/field/orion_field_topology.v1.yaml`
- `orion/proposals/{builder,policy}.py`, `config/proposals/proposal_policy.v1.yaml`
  (template `shed_background_gpu`), `services/orion-proposal-runtime/app/store.py`
  (load the broadcast projection)
- `config/execution_dispatch/execution_dispatch_policy.v1.yaml`,
  `services/orion-execution-dispatch-runtime/app/{worker,store,settings}.py`,
  new `orion/execution_dispatch/shed_settlement.py` (sibling of
  `visual_settlement.py`), `.env_example`, README
- `services/orion-gpu-pool/app/` (shed RPC handler, caps, auto-resume),
  `orion/gpu_pool/scheduler.py`, `config/gpu_pool.yaml`, `.env_example`
- `services/orion-feedback-runtime/app/worker.py` (park until shed settles),
  `orion/feedback/outcome_resolution.py` (settle-time before/after),
  `config/feedback/feedback_policy.v1.yaml`
- `services/orion-hub/scripts/attention_loops_store.py` (accept `acted`),
  loop-outcome writer on settle
- `orion/autonomy/agency_episode.py`, `orion/autonomy/evals/run_attend_act_loop_eval.py`
- `config/metrics/metric_definitions.lock.json` (regenerate)
- Tests beside each of the above.

## Non-goals

- No AC or smart-plug control. No host power actions.
- No change to `ORION_DISPATCH_MIN_NATS_PER_SEC` and no allocator
  bypass. The first patch runs under the allocator exactly as it is.
- No goal registry, affordance taxonomy or new service.
- No relabeling of field attention as "the winner"; `inspect_attended_target`
  is untouched.
- No online self-fitting of weights. Fits are offline proposals.
- No fix to the attention inputs here (that is `fix/attention-input-honesty`)
  beyond consuming them.

## Acceptance checks

On the live rail, not in tests:

1. Phase 0: 48 h of `home_cooling_sample.cabinet_temp_c`; readings both
   below 28.0 °C and above 29.5 °C exist (or the scale is re-anchored).
2. A `substrate_proposal_frames` candidate for `shed_background_gpu` with
   `attention_winner.open_loop_id` equal to a `substrate_attention_broadcast_log`
   row ≤ 90 s older than the proposal.
3. `motor_allocator_refused_everything` stops on that tick; the candidate
   was admitted with `cold_start` or nats/s ≥ 0.02, not through the
   visual-baseline lane.
4. A settled `substrate_dispatch_results` row with `settlement.state=resumed`,
   a passed manipulation check (circe GPU power fell), `forgone_gpu_sec`,
   and the expected effect persisted before `dispatched_at`.
5. A `substrate_action_outcomes` row (first since 2026-09-08) with
   `signal_id=cabinet_heat_pressure`, non-null `observed_delta` and
   `surprise_nats`.
6. An `attention_loop_outcome` row `actor='orion'`, `verdict='acted'`, whose
   `loop_id` matches (2); the loop is **not** excluded from the next
   competition.
7. At least one holdback control outcome for the same signal.
8. `run_attend_act_loop_eval.py --live` reports ≥ 1 episode with every link
   present.
9. Kill-switch drill: flip `GPU_POOL_SHED_ENABLED=false` during an active
   shed; the hold resumes within 10 s and the result settles `cancelled`.

## Recommended next patch

**`feat/shed-background-gpu-loop`: the whole A1 capability, end to end, in
one PR.** Contract first (schemas, registry, channel, `acted` verdict,
consumer-first ordering), then cabinet temperature into `home_cooling_sample`
and the field, then the gpu-pool shed RPC with its caps and auto-resume,
then `workspace.winner` binding and the template, then precommit + shed
settlement in the dispatch runtime, then settle-time outcome resolution and
the `acted` loop outcome, then the eval. Ship dark (all flags off), run
Phase 0 for 48 h, then enable with `ORION_DISPATCH_HOLDBACK_FRACTION=0.2`.
Done means acceptance checks 1–9 on the live rail.

Separately and now, not gated on any of this: serialize the two queries in
`services/orion-recall/app/sql_chat.py:92` (967 errors/24 h), with a
regression test.

## Staged plan

1. **Stage 1 (first patch).** A1 end to end, as above. Proves one closed
   loop with a world sensor and a control arm.
2. **Stage 2.** Dynamic-pressure bridge for `node:cabinet`,
   `node:substrate.rpc_delivery`, `node:circe` (with the attention-input work,
   each through the metric gate); A2 `report_rpc_hop_fault`; A3 agent-lane
   shed; A4 hardware-watch reflex on the same primitive.
3. **Stage 3 (needs Juniper's yes — gate-definition change).**
   (a) Score the candidate's **current bin** cell instead of pooling all
   bins: the decision is conditional on the current state, so the
   information it would yield is the current cell's, and rare high-heat
   states stay informative. (b) A pragmatic admission path,
   `confidently_helpful`, symmetric to the harm gate: admit an action whose
   treated-minus-control contrast in the declared direction exceeds
   `HARM_CONFIDENCE_SIGMAS` with ≥ 20 per arm, recomputed on a trailing
   60-day window so a fluke is not permanent (the allocator docstring's own
   hazard). Without (b), every action that works is retired for working.
4. **Stage 4.** First offline fit of `base_priority` for actions that reach
   D3's volume; `dimension_weights` after.

## Proposal-mode answers

- **Capability that changes.** Orion can, by itself, pause its own
  background GPU work for up to 15 minutes when its focus is on its body's
  heat and the cabinet is warm, and learns whether that cooled the cabinet.
- **Data touched.** Cabinet temperature (new column), GPU pool hold state,
  existing proposal/dispatch/outcome tables, `attention_loop_outcome` (new
  `acted` rows). No personal data, no chat content.
- **Privacy boundary.** None crossed. The cabinet sensor is a machine
  sensor; nothing about Juniper is read or written. A2 (stage 2) writes to
  Juniper through the existing outreach path and its existing caps.
- **Trace that proves it worked.** D4's chain, checked by the eval's live
  mode and acceptance checks 2–8.
- **Dangerous failure modes.** Runaway shedding, self-DOS, stale-attention
  action, self-silencing, confounded learning, tests hitting production —
  guards in D5.
- **Disable / roll back.** `ORION_WORLD_ACTIONS_ENABLED=false` or
  `GPU_POOL_SHED_ENABLED=false`, restart the two services; sheds auto-expire;
  all schema additions are nullable.
