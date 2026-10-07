# Attend → act → learn: closing Orion's loop with one world-facing action

Status: **proposal mode** (design only, nothing implemented). Juniper asked
for it 2026-09-29. Touches autonomy, attention and a cognition loop, so every
item in AGENTS.md's proposal-mode list is answered in
[Proposal-mode answers](#proposal-mode-answers).

**Amended 2026-09-29** (same day, after merge in PR #2394): the first action
no longer gets its own GPU pause. It shares the hardware-watch reflex's
load-shedding lever, ships after it, and fires earlier and gentler. Read
[Amendment 2026-09-29](#amendment-2026-09-29--one-shared-shed-lever) first;
sections it changed carry an *Amended* note inline.

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

The decision: build **one** world-facing action end to end — **when the
cabinet is warm and getting warmer while the AC is healthy, Orion stops
starting new background GPU work for a while, then checks whether the
cabinet actually cooled** — with the workspace winner as its trigger, a real
temperature sensor as its outcome, a control arm, and a single correlation
chain from focus to learned outcome. *(Amended 2026-09-29: this was "pauses
its own background GPU work"; it now stops new background grants on the
same shedding guard the hardware-watch reflex owns, and only after that
reflex ships.)* Two more actions follow on the same rails. Air-conditioner
control and restarting services are explicitly out.

## Amendment 2026-09-29 — one shared shed lever

### What was wrong

The first version of this doc had Orion's first action, `shed_background_gpu`,
pause Orion's background GPU work when the cabinet read warm, using the
urgent-class recall machinery from #2385. But the hardware-watch design
(`2026-09-28-urgent-curiosity-and-hardware-watch-design.md`, Part 4 + rule
U4, already approved by Juniper and being built now on
`feat/hardware-watch-and-shedding`) already owns that lever: while an AC
incident is open and the cabinet is rising (≥ 1 °C over 15 min, or at/above
the thermal gate's elevated trip of 29.5 °C), the GPU pool stops granting
*new* background and system work. Running work finishes, nothing is
recalled, interactive and urgent work are untouched.

Building both as written would have meant:

- **Two controllers on one lever**, with different semantics (pause-and-
  recall vs stop-new-grants) and different clear rules (TTL vs incident
  resolved), so neither owner could predict what the pool was doing.
- **Confounded learning.** Orion would credit itself for cooling that the
  reflex (or the AC recovering) caused.
- **The learned action shipping before the safety reflex.** Backwards: the
  thing that protects the room must exist before the thing that experiments
  on it.

Two factual errors are also fixed here:

- The thermal gate's **elevated threshold is 29.5 °C**. 28.0 °C is only its
  *re-arm* point (once elevated, the gate stays elevated until the cabinet
  drops below 28.0). The pressure scale below starts at 28.0 on purpose, and
  now says so; anything here that means "the cabinet is elevated" means the
  gate's hysteretic state, which trips at 29.5.
- "Nothing persists the cabinet temperature" was wrong. `orion-biometrics`
  writes it as `orion_biometrics_summary.measurements->>'cabinet_temp_c'`
  (node athena, ~30 s cadence): **80,395 rows since 2026-08-29**. The
  hardware-watch reflex reads exactly this. The proposed
  `home_cooling_sample.cabinet_temp_c` column and the orion-zwave cabinet
  poll are dropped; Phase 0 is answered from existing history (below).

### Live evidence behind the amendment (pulled 2026-09-29 ~22:45Z)

Cabinet temperature, last 7 days, 19,356 samples:

| | Value |
|---|---|
| min / median / max | 24.8 / 30.0 / 33.3 °C |
| below 28.0 (pressure 0, "calm") | 2,182 samples (11%) |
| at/above 29.5 (elevated trip) | 12,776 (66%) |
| at/above 32.0 (hot) | 1,169 (6%) |
| 15-min change, SD of minute means | 0.42 °C (20-min: 0.48, 30-min: 0.58) |
| minutes rising ≥ 0.5 °C/15 min while 29.5–32 °C | 445 min, **39 episodes (~5.6/day)** on 8 distinct days |
| minutes rising ≥ 1.0 °C/15 min while 29.5–32 °C | 80 min, 11 episodes (~1.6/day) |

Two consequences. First, "elevated" is the cabinet's *normal* state (two
thirds of the week), so a trigger on "elevated" alone would fire most of the
day and teach nothing. The trigger must be "elevated **and rising**".
Second, the temperature is noisy on the scale of the effect: a 15-minute
swing of ±0.4 °C happens with nothing done. That sets the learning volume
(D3) and is why a control arm is not optional.

GPU pool, last 7 days (`gpu_pool_leases`, duration = `updated_at −
granted_at`, a proxy):

| Class | Leases | p50 / p90 duration |
|---|---|---|
| background request (one inference) | 1,137 | 10 s / 16 s |
| background hold (a durable run, e.g. curiosity) | 252 | 620 s / 4,587 s |
| system request | 43,538 | 3 s / 17 s |

Background leases by holder (3 days, requests + holds): `vision-council` 493, `durable-runs` 224,
`orion-memory-consolidation` 27, `cortex-orch` 13. System leases include
`cortex-exec` (10,051) and `orion-mind` (941) — Orion's own execution and
thinking paths. In 140 of 142 elevated-and-rising minutes since 09-26 12:00Z,
at least one background lease was running.

### A1 as amended (supersedes the A1 details in D2)

**Ordering.** The hardware-watch reflex (Part 4 + U4) ships first, on
`feat/hardware-watch-and-shedding`. The learned action is built after it,
on top of it, and its first patch does not start until the reflex's guard
is merged and live.

**One lever, named reasons.** The hardware-watch spec defines one guard,
`cooling_incident`. This design depends on the reflex PR building that guard
as a set of named reasons with a precedence order (an ask to that PR, see
Recommended next patch), and adds exactly one lower-precedence reason and no
second mechanism:

| Reason | Set by | Precedence | Scope | Clears when |
|---|---|---|---|---|
| `cooling_incident` | hardware-watch incident event only | 0 (always wins) | new `background` + `system` grants | the reflex's own rule: re-evaluated each tick (open `cabinet_ac` incident AND rising); fully off when the incident resolves or Juniper resolves it |
| `orion_self_shed` | Orion's `shed_background_gpu` dispatch, via RPC | 1 | new `background` grants only | TTL ends, kill switch, or a `cabinet_ac` incident opens (whether or not `cooling_incident` is asserted that tick) |

Semantics are U4's, unchanged: the pool stops granting **new** leases in
scope; already-granted leases run to completion; nothing is recalled;
`interactive` and `urgent` are never affected. The RPC refuses any attempt
to set or clear `cooling_incident` — Orion cannot assert or cancel the
reflex. Whether a running background hold's *child* call leases count as
"new grants" is the reflex PR's decision; this action inherits it and does
not add a rule of its own.

**Background only, not system.** The reflex sheds system work too because
the AC is down and every watt matters. The learned action should not:

- System work is Orion's own cognition and execution (`cortex-exec`,
  `orion-mind`, topic foundry). Shedding it on a routine warm afternoon is
  the self-DOS failure mode, and it could stall the dispatch/feedback path
  that settles this very action.
- System requests are short (p50 3 s), so shedding them buys little heat
  for a lot of disruption; background holds are the long, heavy GPU users.
- A narrower scope keeps the treatment one clean thing to learn about.

**Mutual exclusion.**

- *Ineligible while the reflex is not idle.* The builder does not propose
  `shed_background_gpu` while hardware-watch reports **any** open incident
  (`cabinet_ac`, CPU heat, GPU heat), or while hardware-watch's health is
  unknown/stale — if we cannot tell the reflex is idle, we assume it is not.
  This is also how "AC healthy" is defined: no open `cabinet_ac` incident,
  per the one rule that owns that judgement.
- *Cleared on overlap.* The trigger is the **incident**, not the guard
  tick: when a `cabinet_ac` incident opens (the `opened` event on
  `orion:hardware:watch:incident`), the pool drops any active
  `orion_self_shed` in the same step and settles it `preempted_by_reflex`,
  even if the reflex's own rising condition has not asserted
  `cooling_incident` yet. The AC is no longer healthy, so the learned
  action's premise is gone.
- *Holdback relative to the reflex being idle.* The control-arm draw happens
  only on decisions that passed the eligibility check above, so treated and
  control rows come from the same population (reflex idle, AC healthy,
  elevated and rising, background work present). A holdback row is not
  "Orion did nothing while the reflex shed"; that row is never created.

**Clean learning — exclude for the AC reflex, keep the rest.** Each
settled row (both arms) carries `overlap` tags. The rule for which to drop:
exclude an overlap only if it cannot have been caused by the treatment.

- `overlap:reflex` (a `cabinet_ac` incident opened during the outcome
  window, in either arm) — **excluded from fitting and from posterior
  updates.** An AC failure is caused by the AC, not by Orion's shed, so
  dropping these rows does not select on the outcome; and when the AC fails
  its ~870 W dominates the heat balance, so the row measures the AC, not
  Orion.
- `overlap:heat_incident` (a CPU or GPU heat incident opened during the
  window) — **kept, reported as a covariate.** Unlike the AC, compute heat
  is plausibly moved by the shed itself, so dropping these rows would select
  on the outcome. (At `t0` no incident of any kind may be open — that is
  eligibility, not a filter on outcomes.)
- `overlap:render_gate` (the reverie render gate in
  `services/orion-thought/app/visual_chain.py`, or the pool's swap-load
  thermal guard in `services/orion-gpu-pool/app/guards.py`, closed during
  the window because the cabinet went `hot`) — **kept, and recorded as a
  covariate for reporting only, not excluded.** These gates close *because
  the temperature rose* — that is the outcome. Dropping those rows would
  throw away exactly the treated runs where shedding failed to stop the
  rise and bias the estimate toward "shedding works". Both arms face the
  same gate, so an intention-to-treat comparison stays fair. The eval
  reports the contrast with and without them as a sensitivity check, and no
  weight is fit on the covariate until D3's per-bin volume exists for it.

**Trigger — earlier and gentler than the reflex.** All of:

1. Workspace winner in `binds_to_nodes: [node:substrate.biometrics, node:cabinet]`
   (D1, unchanged).
2. Thermal gate verdict on a fresh reading is `elevated` — not `hot`
   (hot belongs to the render gate and pool swap guard already), not
   `unknown`/stale.
3. The cabinet is **rising**: 15-minute rise ≥ 0.5 °C, computed by the
   **same function the reflex uses** (see Signal below), with a lower
   threshold argument. The reflex uses 1.0 °C. 0.5 °C is a tunable knob, not
   a finding: it is ~1.2× the 15-minute noise SD, and gives ~5.6 episodes a
   day live vs ~1.6 at 1.0 °C. The shipped value is recorded on every
   proposal so a change is visible.
4. The reflex is idle (above) — so the learned action always fires before
   any AC incident, never during one.
5. At least one `background` lease is granted or queued in the pool.
   Otherwise there is nothing to stop, and the action is not proposed (this
   replaces the old `nothing_to_shed` settle state, and applies to both
   arms, keeping them comparable).

**Settle rule (stop-new-grants drains slowly).** Stopping new grants does
nothing to a background hold that is already running, and holds last 10
minutes at the median and over an hour at p90. So the shed's effect on GPU
load starts when running background work drains, not when the reason is set,
and in many sheds it will not fully drain before the TTL ends. The rule:

- Both arms use the **same clock from the decision time `t0`**: before =
  cabinet minute-mean at `t0`; after = minute-mean at `t0 + TTL + 5 min`
  (TTL 15 min, so 20 min; the extra 5 min lets the air respond). This is an
  intention-to-treat comparison — it measures what the action actually does
  in the world, including when running work keeps going.
- The treated row also records the **manipulation check**: `drained_at`
  (first moment no background lease was granted), `drain_sec`,
  `grants_withheld` (background requests that queued while the reason was
  active), `delayed_grant_sec`, and circe GPU power over the window vs `t0`.
  A treated shed that never drained is still scored (it is what the action
  does) but is labelled `drain:none`, and the eval reports the drained-only
  contrast next to the intention-to-treat one.
- Terminal pool states: `expired` (TTL ended), `cancelled` (kill switch),
  `preempted_by_reflex`, or `refused:<reason>` (cap, gap, disabled,
  reflex active). Only `expired` rows update the posterior. `cancelled` and
  `preempted_by_reflex` rows are recorded, not scored. No terminal state by
  `t0 + TTL + 300 s` → orphan, `settlement_timeout`.

**Caps (unchanged numbers, re-checked).** Max TTL 15 min, at most 60 min of
`orion_self_shed` per day, at least 15 min between the end of one and the
start of the next — all enforced in the pool, all applying to
`orion_self_shed` only (the reflex is safety and is never capped by this
design). They still make sense under stop-new-grants: ~5.6
elevated-and-rising episodes a day is a **ceiling** on eligible decisions,
not an expected rate — it ignores the workspace-winner bind (biometrics wins
~34 times a day with dwell ≥ 2 not yet measured jointly; `node:cabinet`
cannot win before stage 2's dynamic-pressure bridge) and the reflex-idle
check. Even at the ceiling, with half held back (D3 amendment), treated
sheds run ≤ 2–3 a day, under the 4-a-day ceiling the caps imply. The first
patch's eval measures the joint rate live. Stop-new-grants is
milder than recall — running work is never lost — so the harm per minute of
shed is smaller than in the first version, and the caps are, if anything,
conservative. The 15-min gap also keeps one shed's tail out of the next
decision's "before" reading.

**nats/sec honesty (re-derived).** The motor cost the allocator charges is
the executor's wall time: one RPC that sets a reason, ~1–2 s. Cold, the
action is worth ½·ln(1 + 0.25/0.04) ≈ 0.99 nats, so ≈ 0.5–1 nats/s, well
over the 0.02 floor. The real cost to Orion is delayed background work, not
compute spent: it is now *delayed grants*, not paused work, so nothing
already running is lost and the opportunity cost is smaller than before. It
is recorded as `delayed_grant_sec` and `grants_withheld`, bounded by the
daily cap, and priced with `energy/run_cost.py`. If it were charged as
motor cost (a 900 s shed) the action would score ~0.001 nats/s and never
run; both numbers stay in the record. Under pooled scoring the action stays
above the floor for about 12 (2 s cost) to 24 (1 s cost) settled treated
observations, then retires. **New honesty point:** D3's volume rule, with
the live noise above, needs ~47 per arm before any value is fit (next
section). The allocator will retire the action well before that. So without
stage 3's current-bin scoring or `confidently_helpful`, stage 1 can prove
the loop closes and collect ~12–24 treated rows, but it cannot finish
learning whether shedding works. That is stated up front rather than
discovered.

**Signal — reuse, do not duplicate.** Two quantities, one source:

- *Rise (trigger).* The reflex already computes "cabinet up ≥ 1 °C over 15
  min" from `orion_biometrics_summary.measurements->>'cabinet_temp_c'` on
  athena. That computation must live as a pure function in
  `orion/hardware_watch/rules.py` (e.g. `cabinet_rise_c(samples, window_sec)`
  returning the rise, with the threshold applied by the caller). This design
  imports it; it does not write a second one. If the reflex PR buries it
  inside the service, the first patch here lifts it into the shared module
  as a refactor with the reflex's own tests unchanged.
- *Level (outcome).* `cabinet_heat_pressure` on `node:cabinet` =
  clamp01((temp_c − 28.0) / (32.0 − 28.0)): 0 at the elevated **re-arm**
  point 28.0, 0.375 at the elevated **trip** 29.5, 1.0 at hot 32.0. Same
  source column as the rise, fed through the field digester's existing
  biometrics ingest. Live it reads 0 about 11% of the week, so it can
  return to calm.
- The outcome is the level change over the window, not the rise signal
  that triggered the action, so the trigger and the score are not the same
  number read twice.

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
  `release(keep_requeued=True)`. Rollback `urgent_max_concurrent: 0`.
  *(Amended 2026-09-29: this is **not** the executor any more. Recall/pause
  is the urgent class's mechanism. The shed lever is the hardware-watch
  spec's rule U4 — stop new grants, no recall — built with its Part 4 on
  `feat/hardware-watch-and-shedding` (Parts 1, 2, 2b, 3 and 5 of that spec
  are implemented, some live smokes still UNVERIFIED; Part 4 is in
  progress). The shed guard, with named reasons, is
  **reused as the executor**.)*
- **Thermal gate** (`orion/autonomy/thermal_gate.py`): hysteresis verdicts
  on cabinet air temperature (elevated trips at ≥ 29.5 °C and re-arms below
  28.0; hot trips at ≥ 32.0 and re-arms below 30.5; stale > 300 s). The
  gpu-pool guard reads the cabinet sensor live (`GPU_POOL_CABINET_URL`, a
  BME-class sensor on an Arduino Nano ESP32). At 05:16Z it read
  **29.76 °C — "elevated"**. **Reused as the action precondition and the
  outcome scale.** Already live on the same gate: the reverie render gate
  (`services/orion-thought/app/visual_chain.py`, closes when `hot`) and the
  pool's swap-load thermal guard (`services/orion-gpu-pool/app/guards.py`,
  refuses loading an extra model when `hot` or the reading is unusable).
- **Energy watcher** (#2373, #2376): `orion/energy/run_cost.py` prices a
  settled `power_intent` in kWh and `estimated_run_cost_usd`. **Reused to
  put a dollar figure on what a shed saved.**

### The candidate problems, checked live

| Problem | Live evidence | Verdict for a first action |
|---|---|---|
| RecallService RPC timeouts | `node:substrate.rpc_delivery` worst hop `orion:exec:request:RecallService` 19 of 154 logged ticks in 48 h; peak `rpc_timeout_pressure` 0.684 at 09-28 14:00Z (reliability 0.582). Recall logs **967** `asyncpg … another operation is in progress` in 24 h — the `asyncio.gather` on one connection at `services/orion-recall/app/sql_chat.py:92`, flagged in the 09-20 audit and still live. | A code bug. No runtime action fixes it; a restart would only reset it. Right action: tell Juniper, with evidence, and expect a fix (stage 2). Separately, **fix the bug** — it should not wait for this design. |
| Agent-lane inference failures | `inference_failure_pressure` on `node:circe` > 0.5 on 169 / 28 / 29 / 0 sampled ticks (09-26 → 09-29). No `upstream_timeout` lines in any gateway container's logs in 48 h (the classifier emits them as grammar events, not log lines). Current circe p95 latency 33.5 s. | Real but rare. Same executor as the thermal shed, different trigger (stage 2). |
| Cabinet heat | Sensor live at 29.76 °C (elevated). *(Amended: history does exist — `orion_biometrics_summary.measurements->>'cabinet_temp_c'`, 80,395 rows since 2026-08-29; elevated or hotter 66% of the last 7 days, elevated-and-rising ~5.6 episodes/day.)* AC runs continuously at ~870 W. GPU runs add a mean 22 W over baseline per settled render intent. | First action. Consequence is physical and protects Juniper's room; sensor and history exist; executor is the hardware-watch shed guard, which must ship first; risk is delaying Orion's own background work. |

## Missing questions

These change the design if answered differently; the rest were settled by
investigation.

1. **Does Orion's GPU load measurably heat the cabinet at all?** Renders add
   ~22 W against an ~870 W AC. The agent/background LLM holds are larger but
   unmeasured per hold. If the answer is "no", the shed action should
   learn that, the posterior says so, and the action retires itself — an
   honest outcome, not a failure of the design. *(Amended: the old
   "within ~2 weeks" is withdrawn — see below. The old
   Phase 0 — 48 h of recording before acting — is answered from existing
   history; see the Amendment. How much GPU load moves the cabinet is still
   `UNVERIFIED`, and at ~47 per arm stage 1 alone will not settle it.)*
2. **Is Juniper OK with Orion holding back new curiosity and other
   background GPU runs by itself** (running work always finishes), within a
   hard daily cap (60 min/day, 15 min max per shed, never system, urgent,
   interactive or chat work)? This is the one consent question. Default-off
   flags make it opt-in either way. *(Amended: was "pausing"; the lever is
   now stop-new-grants and background only.)*
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

#### A1 — `shed_background_gpu` (FIRST PATCH of this design; ships after the hardware-watch reflex)

*(Amended 2026-09-29. The full rules — ordering, the shared guard,
mutual exclusion, trigger, settle, caps, nats/sec, signal — are in
[A1 as amended](#a1-as-amended-supersedes-the-a1-details-in-d2); this is the
summary.)*

- **What it does.** Sets the named reason `orion_self_shed` on the GPU
  pool's existing shedding guard (the one the hardware-watch reflex
  introduces) for up to 15 min. While set, the pool grants no **new**
  `background` leases; running work finishes; nothing is recalled. Never
  touches `system`, `interactive`, `urgent`, or chat work.
- **Triggered by.** Workspace winner in `binds_to_nodes: [node:substrate.biometrics, node:cabinet]`,
  **and** the thermal gate's verdict on a fresh reading is `elevated`
  (not `hot`, not `unknown`/stale), **and** the cabinet rose ≥ 0.5 °C in
  15 min (the reflex's own rise function), **and** no hardware-watch
  incident is open (AC healthy, reflex idle), **and** background work is
  granted or queued.
- **Expected effect.** `cabinet_heat_pressure` decreases between `t0` and
  `t0 + 20 min`, relative to the control arm. Predicted delta = the
  posterior mean for the current cell; direction declared `decrease`.
- **Verifying sensor.** Cabinet air temperature
  (`orion_biometrics_summary.measurements->>'cabinet_temp_c'`).
  **Manipulation check** (recorded, not a scoring gate): `drained_at`,
  `drain_sec`, `grants_withheld`, `delayed_grant_sec`, circe GPU power
  vs `t0`.
- **Settle rule.** Pool returns `shed_id`; terminal `expired`, `cancelled`,
  `preempted_by_reflex`, or `refused:<reason>`. Only `expired` updates the
  posterior (intention-to-treat on a clock from `t0`). No terminal state by
  `t0 + TTL + 300 s` → orphan with `settlement_timeout`.
- **Risk class.** Low, reversible, self-directed. No state outside Orion's
  own queue changes; new background work waits, running work is never lost.
- **Policy gate.** `read_only` is wrong for an action that changes pool
  state. New gate value `self_reversible` (bounded, auto-reverting action on
  Orion's own resources), allowed by `execution_dispatch_policy.v1.yaml`
  only when `ORION_WORLD_ACTIONS_ALLOWED` names the template. No per-action
  operator approval; Juniper's approval is the flag.
- **Why it clears the floor honestly.** The allocator charges motor-seconds
  of compute consumed. A shed consumes the executor's wall time (one bus RPC,
  ~1–2 s) and *returns* GPU time. Cold, that is ~0.99 nats / 2 s ≈ 0.5
  nats/s. The withheld background work is real, but it is an opportunity
  cost to Orion's queue, not compute spent; it is bounded by the daily cap,
  recorded on the result as `delayed_grant_sec` / `grants_withheld`
  *(amended: was `forgone_gpu_sec`, which assumed paused work)*, and priced
  via `energy/run_cost.py`. If it were charged as motor cost (a 900 s shed),
  the action would score ~0.001 nats/s and never clear — the doc records
  both numbers so the choice is inspectable. Under the current pooled
  scoring, each (kind, target, signal) stays above the floor for roughly
  12–24 settled observations (solve ½·ln(1 + v/τ²) ≥ 0.02·cost with
  v = 1/(1/0.25 + n/0.04)), then retires — unless stage 3 lands. *(Amended:
  that is fewer than the ~47 per arm D3 needs to fit anything, so stage 1
  proves the loop, not the effect.)*

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

Same executor as A1 — the same `orion_self_shed` reason on the same guard,
sharing A1's caps and A1's mutual exclusion with the reflex — scoped to
background work that could run on the agent lane, triggered by winner
`node:circe` with `inference_failure_pressure` elevated. Expected effect:
`inference_failure_pressure` decreases. Separate posterior cells because the
target and signal differ. *(Amended: no separate shed mechanism or reason;
whether the guard can scope by lane is a question for the reflex PR's guard
shape, not something A3 adds.)*

#### A4 — the reflex, not a choice (**ships first**, owned by hardware-watch)

*(Amended 2026-09-29. Was listed as a stage-2 "later action" sharing A1's
primitive; that inverted the order and misdescribed the lever.)* The
hardware-watch reflex is not an Orion action and is not learned. While a
`cabinet_ac` incident is open and the cabinet is rising (≥ 1 °C/15 min, or
at/above the elevated trip 29.5 °C), the pool's guard carries reason
`cooling_incident`: no new `background` or `system` grants, running work
finishes, no recall. It is spec'd in
`2026-09-28-urgent-curiosity-and-hardware-watch-design.md` (Part 4 + U4),
built on `feat/hardware-watch-and-shedding`, and must be merged and live
before A1's first patch starts. It always wins over `orion_self_shed`.

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
  control cells. Without it, "the cabinet cooled after I shed" is
  confounded by the AC and the room. *(Amended 2026-09-29: the draw happens
  only among decisions that passed A1's eligibility — reflex idle, AC
  healthy, elevated and rising, background work present — so both arms come
  from the same population and no control row ever sits inside a reflex
  shed. For `shed_background_gpu`, propose a per-template
  `holdback_fraction: 0.5` on the template (falling back to the global
  value when unset): with a two-arm contrast, equal arms minimise the total
  runs needed, and at 0.2 the control arm would take 6+ weeks to reach ~47
  even at the eligible-episode ceiling (5.6/day × 0.2 ≈ 1.1/day). Rows tagged `overlap:reflex` are excluded
  in both arms; rows tagged `overlap:render_gate` or `overlap:heat_incident`
  are kept — reasons in the Amendment.)*
- **Fitting.** Offline only, producing a proposed config patch for review;
  never auto-applied.
  - Per-action value = contrast (treated minus control, same bin band) in the
    declared direction.
  - Minimum volume before fitting any weight for an action:
    `n_per_arm ≥ max(20, ceil(2·(2σ̂/δ)²))`, where σ̂ is the observed
    residual SD of that action's settled outcomes and δ = 0.05 (the feedback
    builder's own ±0.05 change threshold); plus observations on ≥ 3 distinct
    days. With the allocator's default τ = 0.2 this is 128 per arm; for a
    low-noise temperature signal it will be the floor of 20. *(Amended
    2026-09-29: the cabinet signal is not low-noise. Live 20-min change SD is
    0.48 °C = 0.12 in pressure units, so the rule gives ~47 per arm. At the
    ceiling of ~2–3 per arm per day this is at least 3 weeks — if the
    allocator keeps admitting the action, which it will not past ~12–24
    treated rows without stage 3.)*
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
  → substrate_execution_dispatch_frames  dispatch_id, expected_effect (precommitted), eligibility snapshot
  → gpu-pool shed reason record     shed_id, reason='orion_self_shed', dispatch_id, terminal state,
                                    drained_at, grants_withheld, delayed_grant_sec
  → substrate_dispatch_results      settlement.{shed_id, state}, manipulation_check, overlap tags
  → substrate_action_outcomes       predicted_delta, observed_delta, surprise_nats, arm, overlap
  → attention_loop_outcome          loop_id = open_loop_id, actor='orion', verdict='acted'
  → next substrate_attention_broadcast_log  same loop: salience after settle
```

*(Amended 2026-09-29.)* The eligibility snapshot on the dispatch frame
records what the builder saw at `t0`: thermal verdict and reading age,
15-min rise and the threshold used, hardware-watch health and open incident
ids (must be empty), background leases granted/queued. Control-arm rows
carry the same snapshot, so the eval can check both arms were drawn from the
same population. The shed record shows `reason='orion_self_shed'`, never
`cooling_incident`; a `cabinet_ac` incident `opened` event during a window
is what produces the `overlap:reflex` tag, in either arm (a CPU/GPU heat
incident produces `overlap:heat_incident`).

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
contrast. *(Amended: fixture mode also fails an `orion_self_shed` whose
window overlaps an active `cooling_incident` without being settled
`preempted_by_reflex`, a proposal made while a hardware-watch incident was
open, and an `overlap:reflex` row that reached the posterior. Live mode
reports the contrast three ways: intention-to-treat, drained-only, and
without `overlap:render_gate` rows.)*

### D5. Kill switch, rollback, dangerous failure modes

**Kill switches (all default off / safe):**

- `ORION_WORLD_ACTIONS_ENABLED=false` (dispatch runtime) — master switch;
  when false, world templates are blocked as `world_actions_disabled`.
- `ORION_WORLD_ACTIONS_ALLOWED=` (empty) — allowlist by template key.
- `GPU_POOL_ORION_SHED_ENABLED=false` *(amended: was `GPU_POOL_SHED_ENABLED`,
  which would have read as the switch for the whole guard)* — pool refuses
  `orion_self_shed` requests; flipping it off drops any active
  `orion_self_shed` reason immediately (settles `cancelled`). It never
  touches the reflex's `cooling_incident` reason, whose own switch stays
  `HARDWARE_WATCH_SHED_ENABLED` (hardware-watch spec). Turning Orion's
  action off can never turn the safety reflex off, and vice versa.
- Caps on the pool side, not the caller, applying to `orion_self_shed` only:
  `GPU_POOL_ORION_SHED_MAX_TTL_SEC=900`,
  `GPU_POOL_ORION_SHED_MAX_SEC_PER_DAY=3600`,
  `GPU_POOL_ORION_SHED_MIN_GAP_SEC=900` (gap measured from the end of one
  shed to the start of the next).

**Rollback.** Flags off and restart the dispatch runtime and gpu-pool. No
persistent state changes: sheds expire by TTL, the new template stops
emitting, new columns/fields are additive and nullable. Posterior cells for
`shed_background_gpu` can stay (they are inert without the template). The
reflex is unaffected by any of this.

| Failure mode | Why it is dangerous | Guard |
|---|---|---|
| Runaway remediation | Shed → cooler → shed again, starving curiosity and background runs | One in-flight action per loop; pool-side min gap, max TTL, daily cap; the cap lives in the pool so no caller can exceed it |
| Self-DOS | Orion holds back the lane doing its own thinking or Juniper's chat | *(Amended)* `orion_self_shed` scope is new `background` grants only — never `system`, `interactive`, `urgent`, or chat; running work is never recalled |
| Two controllers on one lever *(added)* | Orion's shed and the reflex fight over the pool, or Orion's TTL ends a shed the room still needs | One guard, named reasons, `cooling_incident` always wins; the RPC refuses to set or clear `cooling_incident`; an active reflex clears `orion_self_shed` (`preempted_by_reflex`); separate kill switches |
| Learned action before safety reflex *(added)* | Orion experiments on the room with no incident response behind it | A1's first patch starts only after the reflex's guard is merged and live; A1 is ineligible when hardware-watch health is unknown |
| Acting on stale attention | A 30-min-old focus triggers action on a problem that is gone | Winner age ≤ 90 s, dwell ≥ 2, plus the upstream fade; cabinet reading must be fresh (not the gate's `unknown`-allows default) |
| Self-silencing / Goodhart | Orion learns "acting ends the focus" | Orion writes only `acted`; outcome is scored on the cabinet sensor, never on the attention signal that triggered it |
| Confounded learning | AC, weather, the reflex, or the render gate move temperature; Orion credits itself | Holdback control arm drawn only while the reflex is idle; `overlap:reflex` rows excluded in both arms; `overlap:render_gate` and `overlap:heat_incident` kept and reported (excluding them would select on the outcome) |
| Mistaking no-drain for no-effect *(added)* | Stop-new-grants leaves long holds running, so many sheds change little; a naive reading says "shedding does not work" | Intention-to-treat and drained-only contrasts reported side by side; `drain_sec` recorded on every treated row |
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
- `orion/schemas/gpu_pool.py` *(amended)* — the shed-reason models come
  from the reflex PR; this design adds only: the reason value
  `orion_self_shed` (precedence 1, scope `background`) next to
  `cooling_incident`; `GpuPoolShedReasonRequestV1` (`dispatch_id`,
  `reason` — must be `orion_self_shed`, `ttl_sec`, `action: set|clear`);
  `GpuPoolShedResultV1` (`shed_id`, `reason`, `state:
  active|expired|cancelled|preempted_by_reflex|refused`, `refusal`,
  `started_at`, `ended_at`, `drained_at`, `grants_withheld`,
  `delayed_grant_sec`). No `paused_hold_ids` / `forgone_gpu_sec`: nothing is
  paused. If the reflex PR ships a different guard shape, these adapt to it;
  they do not introduce a second guard.
- `orion/bus/channels.yaml` — RPC `orion:gpu_pool:shed:request` + reply
  wildcard (dynamic reply channels need a catalog wildcard);
  `orion/schemas/registry.py` entries for both models.
- ~~`HomeCoolingSampleV1.cabinet_temp_c` + `home_cooling_sample.cabinet_temp_c`
  column~~ *(dropped 2026-09-29: the cabinet temperature is already
  persisted as `orion_biometrics_summary.measurements->>'cabinet_temp_c'`,
  which the reflex reads too.)*
- Rise function *(added)*: `cabinet_rise_c(...)` as a pure function in
  `orion/hardware_watch/rules.py` — owned by the reflex, imported here, not
  re-implemented.
- Proposal template field *(added)*: `ProposalTemplateV1.holdback_fraction:
  float | None = None` (per-template override of
  `ORION_DISPATCH_HOLDBACK_FRACTION`).
- Field: channel `cabinet_heat_pressure` on `node:cabinet` =
  clamp01((temp_c − 28.0) / (32.0 − 28.0)) — zero at the thermal gate's
  elevated **re-arm** point (28.0), 0.375 at the elevated **trip** (29.5),
  one at hot (32.0) — read directly by `orion/field/pressure.py` (not merged
  or maxed with any other channel — the commensurability lesson).
- `attention_loop_outcome.verdict` accepts `acted`; `actor='orion'`.
- `config/feedback/feedback_policy.v1.yaml` — `pressure_channels` gains
  `cabinet_heat_pressure`; `positive_delta_channels.cabinet_heat_pressure:
  decrease`; per-template settle window override.

### Metric quality gate for `cabinet_heat_pressure`

1. **Provenance.** Cabinet sensor frame `environment.temp_c` from
   `GPU_POOL_CABINET_URL` (read today by `services/orion-gpu-pool/app/guards.py:45`
   and `services/orion-thought/app/visual_chain.py:324`). *(Amended: the
   persisted copy used here is `orion/telemetry/cabinet_sensors.py:109`
   `put("cabinet_temp_c", env["temp_c"])` → `orion_biometrics_summary.measurements`,
   node athena — the same column the hardware-watch reflex reads.)*
2. **Independence.** A physical air sensor. Not derived from GPU, CPU or
   host telemetry; its causal link to `node:circe power_pressure` is exactly
   the effect being measured. `node:athena thermal_pressure` (one CPU core,
   39 distinct values) is a different quantity and is not used.
3. **Theory anchor.** Heat balance: cabinet air temperature rises with
   dissipated electrical power minus heat removed by the AC. Scale anchored
   on the thermal gate's existing constants, which protect Juniper's room.
4. **Live sanity.** *(Amended 2026-09-29 — the first version said no
   history existed; it does.)* 7 days, 19,356 samples, 751 distinct values:
   min 24.8, median 30.0, max 33.3 °C; below 28.0 (pressure 0) 11% of
   samples; at/above 29.5 66%; at/above 32.0 6%. Not flat, not saturated,
   and it returns to a genuine calm reading regularly. The 15-min change has
   SD 0.42 °C, which is noise the outcome must be measured against (hence
   the control arm and D3's volume). The old Phase 0 is satisfied by this
   history and dropped.
5. **Existing mechanism.** *(Amended.)* The temperature is already
   persisted in `orion_biometrics_summary`, and the reflex computes a 15-min
   rise from it; both are reused. No existing field channel carries it as a
   pressure (`cabinet_ambient_spike` is audio), so `cabinet_heat_pressure`
   is the one new thing.
6. **Reversibility.** One field channel *(amended: no new column)*;
   removing it drops one `PredictableSignal` value with no stored consumer
   besides this action's posterior cells.

## Files likely to touch

Prerequisite, not this design's patch: `feat/hardware-watch-and-shedding`
(reflex + pool shed guard with named reasons) merged and live.

First patch *(amended 2026-09-29)*:

- `orion/schemas/{proposal_frame,action_prediction,gpu_pool,registry}.py`,
  `orion/bus/channels.yaml`
- ~~`services/orion-zwave/`, `home_cooling_sample` column + migration~~
  (dropped: temperature already persisted)
- `orion/hardware_watch/rules.py` — only if the reflex left the rise
  computation inside its service: lift it to a pure shared function, reflex
  tests unchanged
- `services/orion-field-digester/app/ingest/` (`node:cabinet` channel from
  biometrics `cabinet_temp_c`), `orion/field/pressure.py`,
  `config/field/orion_field_topology.v1.yaml`
- `orion/proposals/{builder,policy}.py`, `config/proposals/proposal_policy.v1.yaml`
  (template `shed_background_gpu`, `holdback_fraction: 0.5`),
  `services/orion-proposal-runtime/app/store.py` (load the broadcast
  projection, hardware-watch open incidents + health, pool background
  occupancy)
- `config/execution_dispatch/execution_dispatch_policy.v1.yaml`,
  `services/orion-execution-dispatch-runtime/app/{worker,store,settings}.py`,
  new `orion/execution_dispatch/shed_settlement.py` (sibling of
  `visual_settlement.py`), `.env_example`, README
- `services/orion-gpu-pool/app/` (shed-reason RPC handler for
  `orion_self_shed` only, caps, TTL expiry, clear-on-reflex, drain
  bookkeeping), `orion/gpu_pool/scheduler.py` (add the reason to the
  reflex's guard — no new rule), `config/gpu_pool.yaml`, `.env_example`
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

On the live rail, not in tests. *(Amended 2026-09-29.)*

0. **Prerequisite:** the hardware-watch reflex is live — during its own
   simulated-AC-incident smoke (hardware-watch spec, check 7: "pool shed
   guard visible while open") the guard additionally shows the reflex under
   the named reason `cooling_incident` (an added expectation of this design,
   not something that spec states) — before A1's first patch is deployed.
1. ~~Phase 0, 48 h of `home_cooling_sample.cabinet_temp_c`~~ Replaced: the
   `node:cabinet` `cabinet_heat_pressure` channel reads 0 below 28.0 °C and
   0.375 at 29.5 °C against the same `orion_biometrics_summary` rows (already
   shown to span both, see the Amendment).
2. A `substrate_proposal_frames` candidate for `shed_background_gpu` with
   `attention_winner.open_loop_id` equal to a `substrate_attention_broadcast_log`
   row ≤ 90 s older than the proposal, and an eligibility snapshot showing
   `elevated`, rise ≥ 0.5 °C/15 min, zero open hardware-watch incidents,
   and ≥ 1 background lease.
3. `motor_allocator_refused_everything` stops on that tick; the candidate
   was admitted with `cold_start` or nats/s ≥ 0.02, not through the
   visual-baseline lane.
4. A settled `substrate_dispatch_results` row with `settlement.state=expired`,
   a shed record with `reason='orion_self_shed'`, a manipulation check
   (`drained_at` or `drain:none`, `grants_withheld`, circe GPU power vs
   `t0`), and the expected effect persisted before `dispatched_at`.
5. A `substrate_action_outcomes` row (first since 2026-09-08) with
   `signal_id=cabinet_heat_pressure`, non-null `observed_delta` and
   `surprise_nats`, measured `t0` → `t0 + 20 min`.
6. An `attention_loop_outcome` row `actor='orion'`, `verdict='acted'`, whose
   `loop_id` matches (2); the loop is **not** excluded from the next
   competition.
7. At least one holdback control outcome for the same signal, with an
   eligibility snapshot showing the reflex idle.
8. `run_attend_act_loop_eval.py --live` reports ≥ 1 episode with every link
   present, and zero proposals made while a hardware-watch incident was open.
9. Kill-switch drill: flip `GPU_POOL_ORION_SHED_ENABLED=false` during an
   active `orion_self_shed`; the pool grants the next queued background
   lease within 10 s and the result settles `cancelled`. The reflex's
   reason is unaffected.
10. Precedence drill (simulated AC incident via the hardware-watch test
    hook, with an `orion_self_shed` active): the pool shows only
    `cooling_incident`, system grants also stop, the Orion shed settles
    `preempted_by_reflex`, no new `shed_background_gpu` proposal appears
    until the incident resolves, and the overlapping outcome row is tagged
    `overlap:reflex` and absent from the posterior.

## Recommended next patch

*(Amended 2026-09-29.)*

**First: finish and ship `feat/hardware-watch-and-shedding`** (hardware-watch
spec Part 4 + U4). Nothing in this design starts until its shed guard, with
named reasons and `cooling_incident` at top precedence, is merged and live.
One ask to that PR so this design needs no second mechanism: keep the
cabinet-rise computation a pure function in `orion/hardware_watch/rules.py`
with the threshold as an argument, and keep the guard's reason set open to
one more, lower-precedence reason.

**Then `feat/shed-background-gpu-loop`: the whole A1 capability, end to
end, in one PR.** Contract first (schemas, registry, channel, `acted`
verdict, `holdback_fraction`, consumer-first ordering), then
`cabinet_heat_pressure` into the field from the existing biometrics rows,
then the `orion_self_shed` reason on the reflex's guard (RPC, caps, TTL
expiry, clear-on-reflex, drain bookkeeping), then `workspace.winner`
binding, the template and its eligibility (reflex idle, elevated, rising
via the shared function, background present), then precommit + shed
settlement in the dispatch runtime, then settle-time outcome resolution with
overlap tags and the `acted` loop outcome, then the eval. Ship dark (all
flags off); enable with the template's `holdback_fraction: 0.5`. Done means
acceptance checks 0 and 2–10 on the live rail.

Separately and now, not gated on any of this: serialize the two queries in
`services/orion-recall/app/sql_chat.py:92` (967 errors/24 h), with a
regression test.

## Staged plan

0. **Stage 0 (amended 2026-09-29; owned by hardware-watch).** A4, the
   reflex: hardware-watch Part 4 + U4 and the pool shed guard with named
   reasons. Safety first.
1. **Stage 1 (first patch of this design).** A1 end to end, as above, on
   the stage-0 guard. Proves one closed loop with a world sensor and a
   control arm; does not by itself collect enough to fit the effect.
2. **Stage 2.** Dynamic-pressure bridge for `node:cabinet`,
   `node:substrate.rpc_delivery`, `node:circe` (with the attention-input work,
   each through the metric gate); A2 `report_rpc_hop_fault`; A3 agent-lane
   shed on the same `orion_self_shed` reason.
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

- **Capability that changes.** Orion can, by itself, stop new background
  GPU work from starting for up to 15 minutes when its focus is on its
  body's heat and the cabinet is warm and rising while the AC is healthy,
  and learns whether that cooled the cabinet. Running work always
  finishes. It cannot override, extend or cancel the hardware-watch
  reflex. *(Amended 2026-09-29: was "pause".)*
- **Data touched.** Cabinet temperature (existing `orion_biometrics_summary`
  rows, read only — amended, no new column), GPU pool shed-reason state,
  existing proposal/dispatch/outcome tables, `attention_loop_outcome` (new
  `acted` rows). No personal data, no chat content.
- **Privacy boundary.** None crossed. The cabinet sensor is a machine
  sensor; nothing about Juniper is read or written. A2 (stage 2) writes to
  Juniper through the existing outreach path and its existing caps.
- **Trace that proves it worked.** D4's chain, checked by the eval's live
  mode and acceptance checks 2–8.
- **Dangerous failure modes.** Runaway shedding, self-DOS, two controllers
  on one lever, learned action before the safety reflex, stale-attention
  action, self-silencing, confounded learning, mistaking no-drain for
  no-effect, tests hitting production — guards in D5.
- **Disable / roll back.** `ORION_WORLD_ACTIONS_ENABLED=false` or
  `GPU_POOL_ORION_SHED_ENABLED=false`, restart the two services; sheds
  auto-expire; all schema additions are nullable. The reflex keeps running.
