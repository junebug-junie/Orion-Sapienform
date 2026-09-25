# Attention with stakes: gaps and proposals

- **Date:** 2026-09-25
- **Status:** DESIGN, proposal mode (CLAUDE.md 0A: changes cognition loops). Nothing here is built.
  Two small defects (D1, D2) are worth fixing regardless of the rest.
- **Evidence basis:** read-only survey of `main` at `e5ef19e`. This session had no live database or
  host access. Every live number is quoted from a dated spec or PR report in this repo and says so.
  Findings from reading code that were not reproduced live are marked `UNVERIFIED`.
- **Builds on, does not duplicate:** open PR #2255 (sentience prerequisites assessment and "Orion asks
  and waits"), open PR #2109 (concept attention supply).

## Arsonist summary

Orion's attention is two halves that were built separately and never connected.

One half notices things. There are three different "what stands out right now" winners, a per-turn
chat frame, and a curiosity loop that knows which of Orion's beliefs are shakiest. None of them pays
for anything, and none of them ever learns whether noticing something was worth it.

The other half spends things. There is a motor allocator that buys actions by how much each is
expected to teach Orion per second, plus a GPU pool, reading wallets, outreach caps, a heat gate and
a Claude quota meter. Only the motor allocator changes what Orion does, and it sits at the end of a
pipeline that no attention winner reaches. #2255 measured it live on 2026-09-20: it had executed
nothing since 2026-09-08 and logged "refused everything" on every tick, because nothing worth doing
ever arrives.

In short: noticing is free, spending is blind, and nothing learns. The allocator is the one place
where all three are done correctly. It scores in expected belief change, charges in the real unit of
the scarce resource, refuses what doesn't clear the bar, and learns from the outcome. That is the
design the rest of attention needs.

**Proposal:** make every place Orion spends something scarce work like the allocator. Feed those
places from what attention notices, and score attention by what it actually bought. Start with
curiosity. It is Orion's most expensive self-directed act (a roughly 20-minute turn on the shared
pipeline, 7 a day). It is started by a clock, and its study material comes from `ORDER BY random()`.
It already records how far each investigation moved Orion's beliefs
(`:PriorRevision {from_confidence, to_confidence}`), and nothing reads that record. So the outcome
data already exists and nobody uses it.

## Current architecture

### The loop a working attention mechanism needs, and where Orion stands

Attention that has stakes does five things. Orion has each piece somewhere, but the pieces don't
connect.

1. **Notice.** Find what might be worth attending to.
   *Exists, in fragments.* Field attention ranks hosts, capabilities and 5 reducer domains every 2s
   (`orion/attention/field_attention/`). The workspace competition picks 1 winner from 5 loops every
   30s (`orion/substrate/attention_broadcast.py`). Field tension runs a vote across deviating channels
   (`orion/attention/tension/`). There is a chat-turn frame (`orion/substrate/attention_frame.py`),
   and curiosity keeps a prior list ordered by uncertainty (`orion/curiosity/worldview.py:987`).
   The three winners are separate, and each is read by different consumers.
2. **Value.** Estimate what attending would buy.
   *Only two places do this.* The motor allocator uses expected information gain in nats, in closed
   form (`orion/autonomy/allocator.py`). Curiosity puts its most uncertain priors first, which is
   the ordering binary entropy would give. Everything else ranks by how *unusual* something is. That
   is not the same as what attending to it would buy.
3. **Price.** Know what attending costs right now.
   *Only motor-seconds is a binding price.* GPU slots queue by a fixed route map. Watts, room heat,
   Claude quota, Cursor quota and Juniper's interruptions are measured, or capped by config. None of
   them is a budget any attention mechanism reads.
4. **Spend or wait.** Commit the scarce thing, or say "not worth it now".
   The allocator can do this, and it says no to everything. The workspace winner never reaches it.
   Curiosity starts on a timer.
5. **Check and update.** Measure what attending bought, then update the estimate.
   *This works for dispatched actions only.* `orion/feedback/outcome_resolution.py` writes
   `substrate_action_effect_posterior`. As an action's outcome becomes well known, the action falls
   below the allocator's floor. One local loop also works: the diffusion host predicts its own watts
   (`power_intent_settled`). Nothing checks attention itself.

### What each spend point reads today

| Spend point | What is scarce | How it chooses | Reads attention? | Outcome recorded? |
|---|---|---|---|---|
| Motor allocator (`orion/autonomy/allocator.py:278`) | motor-seconds, 129,600 s/day | nats/sec against a 0.02 floor, then greedy | no; its candidates come from proposals bound to the field node | yes, as an effect posterior |
| Curiosity investigation (`services/orion-hub/scripts/curiosity_investigation.py`) | a roughly 20-min turn on the shared pipeline; cap 7/day | clock (30-min cooldown, window off since 09-20); material by `ORDER BY random()` (`orion/curiosity/study_material.py:258,325`); 8 priors shown most-uncertain-first; the LLM picks | no | `:PriorRevision` is written and never scored; what was offered is not recorded |
| Durable-run admission (`orion/durable_admission/broker.py:57`) | agent lanes | first come, first served | no | run rows |
| World-pulse reading (`orion/world_pulse_read/queue.py:142`) | 6 reads/day (wallet A) and 6/day (wallet B) | fixed class, then attempts, then age; 104 digest items pending on 09-25 | no | done/failed; nothing reaches the worldview (#2199) |
| GPU pool (`orion/gpu_pool/scheduler.py:190`) | llama.cpp slots | fixed route-to-class map, then FIFO | no | `waited_ms` |
| Outreach, Door B (`services/orion-hub/scripts/endogenous_outreach.py`) | Juniper's interruptions: 4/day, 45-min cooldown, quiet hours | tension winner held for 6 ticks | yes, the tension winner | decision rows; replies are stamped `in_reply_to` but only a display reads them |
| Outreach, Door A (after a curiosity run) | same | the run's own verdict; **skips every schedule gate** (`endogenous_outreach.py:452`) | no | skips are recorded |
| Contractor peer (`services/orion-curiosity-peer/app/worker.py`) | Cursor and Claude quota | budget gates that fail closed | no | `PeerBriefV1` |
| Mutation queue (`orion/substrate/mutation_worker.py:154`) | trial capacity | every item at priority 60, so first come, first served | no | decisions; all inconclusive since 09-03 per #2255 |

## Gaps

**G1. What wins attention never reaches anything that spends.**
- The proposal arena binds to field attention's top target, not to the workspace winner
  (`orion/proposals/builder.py`, `target_binding: "attention.dominant_targets[0]"`).
- The one attention-bound template, `inspect_attended_target`, declares no `expected_signal`
  (`config/proposals/proposal_policy.v1.yaml:286-301`). Since the allocator began enforcing on
  2026-08-30, it has therefore been refused as `unmeasurable` (`allocator.py:321`). This is inferred
  from code; confirm it against the allocator's refusal log.
- The workspace competition runs with `max_asks=0` (`orion/substrate/attention_broadcast.py:217`).
  It decides to ask a question, then drops it (see #2255's pathway map).
- Curiosity never sees any winner. The path from workspace loops into endogenous-curiosity seeds
  needs novelty ≥ 0.5 (`orion/substrate/endogenous_curiosity.py:167`). Workspace loops score novelty
  as `max(0.35, salience)` (`orion/substrate/attention/scoring.py:135`), and live salience tops out
  near 0.14 (`services/orion-substrate-runtime/.env_example:158-168`). So that path cannot fire.
  Even when seeds exist, the evaluator's plan is never executed
  (`services/orion-substrate-runtime/app/worker.py:3584-3590`).

**G2. Only one budget has any effect.** Motor-seconds is enforced. Everything else is display,
advice, or fixed config:
- GPU priority is a fixed route map (`config/gpu_pool.yaml`).
- No power budget exists in code. The roughly 454 W of UPS headroom appears only in specs.
- `orion/autonomy/quota_budget.py` is advisory, and only a report script calls it.
- The Claude-limit publisher is off (`COCREATION_SIGNALS_CLAUDE_LIMIT_ENABLED=false`).
- The heat gate is hysteresis only (`orion/autonomy/thermal_gate.py`).
- Curiosity outreach skips Juniper's caps (G6).

**G3. Spending never reports back.**
- The allocator's refusals are stamped on dispatch rows, and nothing upstream reads them. No
  reference exists in proposals, policy, or either attention runtime.
- #2255 found zero hits in any self surface: `felt_state_reader`, `self_panel`, `chat_stance`.
- GPU queue waits reach two prompts, as text only.

Orion can't notice it is being refused. Attention can't tell "unaffordable right now" apart from
"already learned".

**G4. Nothing checks attention against what it bought.**
- `attention_loop_outcome` is resolved only by `actor=juniper`. Workspace and reverie cards return
  409 (`services/orion-hub/scripts/attention_loops_routes.py:75-82`). #2255 counted 48 of 48 loops
  `decayed_unattended` over 14 days.
- Curiosity writes `:PriorRevision {from_confidence, to_confidence}`, read at
  `orion/curiosity/atlas.py:117` and `orion/curiosity/attention_schema.py:81`. That is the direct
  measure of what an investigation changed, and nothing turns it into a score. The hop supervisor
  runs only from the command line, and its own header says it changes nothing Orion does next
  (`orion/curiosity/supervisor.py:10`).
- Outreach replies are attributed through `in_reply_to`
  (`services/orion-hub/scripts/outreach_provenance.py:300`). Only the run-story display reads them.
- Recall records the ids it selected (`recall_telemetry`) but no label for whether they were useful.

**G5. Nothing stops Orion re-attending to what it has already learned.**
- Habituation was removed on 2026-07-31 with nothing put in its place. The sentience-program README
  (§13) calls this an accepted loss.
- Human verdicts no longer apply to workspace loops (G4), so a loop can win indefinitely.
- The allocator shows the right shape. An action's expected information falls as its posterior
  tightens, so "don't repeat what you've learned" happens without any rule for it. Nothing upstream
  has an outcome that could tighten.

**G6. The biggest spends are chosen by clock, queue position, or chance.**
- Curiosity's material is random.
- Its trigger is a cooldown. With the waking window off since 09-20, `paced_cooldown_sec` falls back
  to the 30-minute floor (`curiosity_investigation.py:373-417`). All 7 daily runs then land in the
  few hours after local midnight, and `daily_cap` blocks for the rest of the day. That is the exact
  "3am cluster" the function was written to remove. `UNVERIFIED` live: this reads
  `.env_example`, not the host `.env`.
- Durable admission is first come, first served. Reading uses a fixed class plus age. The mutation
  queue gives every item the same priority.

**G7. Salience that ignores cost has already hurt Juniper.** On 2026-09-05, lowering the workspace
gate from 0.2 to 0.05 (PR #2110) made about 45× as many ticks contested. Within three minutes,
background LLM timeouts doubled, and Juniper's chat turns failed with `stance_react_failed`
(`docs/superpowers/pr-reports/2026-09-05-stance-react-attention-salience-gateway-starvation-incident.md`).
More noticing meant more narration, which used GPU time the chat lane needed. Nothing in the
attention path could see that cost.

**G8. Attention's inputs are less trustworthy than they look.**
- *Freshness.* Per-domain prediction errors keep their last value between producer ticks. Staleness
  guards check the age of the whole field row, which is written about every 2s, not the age of each
  channel (`orion/substrate/bus_synaptic_surprise.py:59-80`,
  `orion/substrate/metacog_trend_signals.py:95-114`). The self-model reads attention frames with no
  age check at all, and the frame lane was dead 09-20 to 09-23 with nobody noticing
  (`docs/superpowers/specs/2026-09-22-substrate-lattice-audit.md`).
- *Scale.* Curiosity's single 0.55 threshold compares four different kinds of number: z-scored
  domains, biometrics' raw ÷0.30 value, bus_synaptic's anomalous-edge fraction (which rests near
  0.027), and a constant (`orion/substrate/endogenous_curiosity.py:190-227`).
- *Constants posing as measurements.* An unresolved harness turn writes a fixed 0.65 prediction
  error (`services/orion-substrate-runtime/app/worker.py:135`). That clears curiosity's 0.55 bar for
  several minutes after each write.
- *Mislabels.* Vision's "prediction_error" is really camera staleness
  (`orion/substrate/prediction_error.py:920-958`), so generic consumers read an outage as surprise.
- *Unreachable cutoffs.* Min-max normalisation always gives the top target 1.0, so `overall_salience`
  sits near 1.0. The workspace action cutoffs (0.65 / 0.48 / 0.35,
  `orion/substrate/attention/policy.py:48-62`) were tuned for the deleted formula. A lone candidate
  probably scores below 0.35 (`UNVERIFIED`).

**G9. Attention sees very little.** Only 6 of 1,796 graph nodes carry any salience (#2109). Supply
is its own design, parked in #2109. Pricing without supply buys nothing. Supply without pricing
floods the GPU (G7).

**G10. No control arm, so no causal claim holds up.** `ORION_DISPATCH_HOLDBACK_FRACTION=0.0`
(`services/orion-execution-dispatch-runtime/.env_example:161`). It ran for about 1h on 2026-08-25.
No other spend point has a control arm.

## Defects found while mapping

These are worth fixing regardless of the rest of this design. Each is small and needs a
regression test.

**D1. Host and capability "novelty" switches on and off for a perfectly steady input.** Reproduced.
The "how new is this" score compares this tick's pressure against last tick's *novelty*, when it
should compare against last tick's *pressure*. Anything steady and non-zero scores full novelty every
other tick, forever.
- **Where.** `novelty_for_target()` (`orion/attention/field_attention/scoring.py:66-74`) diffs against
  `prior_salience_for_target()`, which returns `salience_score`. `_novelty_targets()` sets
  `salience_score` to the novelty (`orion/attention/field_attention/selectors.py:214-220`). The prior
  proxy is already stored as `pressure_score`, which is the value to compare against.
- **Repro.** Real functions, the same capability vector every tick, proxy 0.8. The output is
  `0.000, 0.800, 0.000, 0.800, 0.000, 0.800`.
- **Impact.** Roughly half of host and capability novelty is manufactured, and it feeds dominant
  targets, the proposal binding and the self-model. The docstring's "self-resolving after exactly one
  tick" (`selectors.py:186-202`) is wrong. The existing test
  (`tests/test_attention_field_selectors.py:378-409`) checks one transition, so it can't tell the
  difference. It may also be part of #2255's "vision wins the field by an artifact" finding
  (`UNVERIFIED`).
- **Fix.** Diff against the prior entry's `pressure_score`. Add a regression test that runs at least
  3 steady ticks and expects 0 after the first.

**D2. The contractor peer's Claude quota check reads the Cursor meter.** Verified in code.
- **Where.** In `handle_help_request`, `observe = observe_limit or _observe_default`
  (`services/orion-curiosity-peer/app/worker.py:462`) makes `observe` a local name for the whole
  function. That hides the Claude reader imported from `rate_limit_events` at `:26`. The fallback
  gate `claude_observe = observe_claude_limit or observe` (`:507`) therefore calls the Cursor meter
  in production, because production passes neither argument.
- **Why tests miss it.** Every test injects `observe_claude_limit`
  (`services/orion-curiosity-peer/tests/test_fallback_once.py:102,131,160`, `test_acceptance.py:203`,
  `evals/run_contractor_peer_eval.py:135`), so the default path has never run under test.
- **Impact.** Claude spend, which Orion shares with Juniper, is gated on the wrong resource. The real
  Claude limit reading never takes part in the decision.
- **Fix.** Rename the local to `cursor_observe`. Add a test that the default path calls
  `rate_limit_events.observe`.

**Dead edges to decide, not fixed here:**
- `inspect_attended_target` is unmeasurable (G1).
- The workspace-loop curiosity seed path can't be reached (G1).
- The 0.65 harness-closure constant feeds curiosity as if it were a measurement (G8).

## The design principle

Every place Orion spends something scarce should work the way the motor allocator already does.
Attention's job is to feed those places and to learn from what they bought.

There are four rules. Each already exists somewhere in Orion.

1. **Value is expected belief change, measured in nats.** The question is not how unusual something
   is, but how much attending to it is expected to change what Orion believes. The allocator already
   uses this unit: Bayesian surprise (Itti & Baldi 2009) and expected information gain (Lindley 1956),
   in `orion/autonomy/prediction.py:140`. Suppose Orion holds a belief at confidence *p*. A single
   investigation can teach at most the belief's entropy, H(p). That is 0.69 nats at p = 0.5, 0.33 at
   0.9, and 0.20 at 0.95. Curiosity already orders priors by |p − 0.5|, which gives the same order as
   H(p). The ordering is right. What is missing is any record of what the ordering bought.
2. **Cost stays in the scarce thing's own unit, read from its live meter.** The units are
   motor-seconds, curiosity turns, reading slots, Juniper's interruptions and GPU queue waits. No
   hand-typed exchange rate between them, for the reason the allocator's docstring gives
   ("NO INVENTED EXCHANGE RATE"). Each budget asks one question: is this candidate worth one of *my*
   units?
3. **The budget sets each budget's bar, not a constant.** When a budget is spent faster than it
   refills, its bar rises. When the budget sits idle, the bar falls to zero and only the absolute
   "worth anything at all?" floor applies. This is budget pacing with a dual price (Balseiro & Gur
   2019). The version with many budgets is "bandits with knapsacks" (Badanidiyuru, Kleinberg &
   Slivkins 2013). The price rests at exactly 0 when nothing is scarce, so it passes the rest-state
   check by construction.
4. **Realized belief change comes back and updates the estimate.** What a spend actually changed,
   in the same nats, updates the expected value of that kind of candidate. Two things follow. First,
   whatever Orion keeps attending to without learning from loses value automatically. That is earned
   habituation, not a penalty constant. Second, noisy things that can't be learned stop pulling
   attention, because what counts is learning progress (error going down), not error being high
   (Oudeyer, Kaplan & Hafner 2007). The "noisy TV" trap (Burda et al. 2018) is the failure this
   avoids.

With these rules Orion could say four things it can't say today, and each would be checkable:
- "Nothing is worth investigating right now."
- "That question is worth more than this GPU time."
- "I've tested this three times and learned nothing, so someone else has to answer it."
- "Juniper's chat is waiting, so my own thinking can wait."

## Proposals

Each proposal states what changes, which stake it uses, what trace would prove it worked, and how to
turn it off. They are ordered by recommended sequence.

### P1. Curiosity pays for what it learns (recommended next patch)

**What changes**
- **a. Score every run by what it changed.** For each of a run's `:PriorRevision` rows, compute
  KL(Bern(to) ‖ Bern(from)) in nats, with confidences clamped to [0.01, 0.99]; sum per run. The live
  example in `orion/curiosity/attention_schema.py`, run `3b2d038cf18e` (0.95 → 0.92), scores 0.0081
  nats. A move from 0.5 to 0.8 scores 0.19; a reversal from 0.95 to 0.05 scores 2.65. Orion is asked
  for nothing new, since it already writes these rows. A run that formed priors but revised none
  reads 0 nats and reports the formations beside it. A run whose graph could not be read reads
  `null`, never 0.
- **b. Record the offer.** Record which priors and which study-material ids were shown. For each
  prior, record its confidence, entropy and expected value when shown, and which ones the run then
  touched (`last_run_id` / `PriorRevision.run_id` already mark this). This is the choice set the hop
  supervisor has never been able to see.
- **c. Estimate expected value per prior.** Each prior's value is its entropy times its learning
  yield. Yield is the share of available uncertainty past tests actually resolved, pulled toward the
  pool average while evidence is thin:

  ```text
  expected_nats(prior) = H(confidence) × yield(prior)
  per_test_yield       = min(1, realized_nats / H(confidence_before))     # expectation of KL <= H
  yield(prior)         = (Σ per_test_yield + k × pool_yield) / (n_tests + k)
  ```

  A prior tested three times without moving ends up with yield near 0. It stops being worth a turn
  without any `stale_after` count, and it becomes a hand-off candidate (P4).
- **d. Trigger on value, not the clock.** The existing enabled, cap and cooldown gates stay as
  ceilings. A run starts only when both conditions hold:
  - the best offered expected value clears the curiosity budget's paced price; and
  - no interactive-priority lease is waiting in the GPU pool (`gpu_pool_leases`, the pool's own
    queue, `services/orion-gpu-pool/app/runtime.py:487`).

  The GPU condition is a plain count with a true zero, not a new metric. It was chosen over
  `queue_contention_score` on purpose. That score is the max over three sources, and its seed-queue
  source was frozen with 144 items on 09-25, so it would hold curiosity forever for a reason that
  has nothing to do with the GPU.

  ```text
  run if  max(expected_nats) >= max(price, floor)  and  interactive_waiting == 0
  price <- max(0, price + η × (ran_this_tick − cap × tick_sec / 86400))
  floor  = 10th percentile of realized nats per run over the trailing 30 days,
           recomputed nightly and persisted
  ```

  The floor says: don't spend a turn expected to be worse than your worst typical turn.
- **e. Control arm.** On a fraction ε of eligible ticks, use the old clock trigger regardless. Record
  the arm and its propensity on every decision.
- **f. Say why.** Publish every run or hold decision as an `AttentionSchemaV1` row with
  `process="curiosity"`. That uses an existing contract, the existing channel
  `orion:attention:schema`, and the existing chat cue (`orion/substrate/recent_attention_cue.py`).
  An example is `attention_reason="held_off:interactive_waiting"` with a computed narrative. Orion
  can then answer "why didn't you investigate anything this morning?" from a record, not a guess.

**Hand-picked constants, disclosed:**
- `k`, the number of pseudo-tests, which sets how much evidence it takes to move a prior's yield off
  the pool average;
- `η`, the price step, which sets how fast the price reacts, not what anything is worth;
- the p10 / 30-day definition of the floor;
- the clamp at 0.01 and 0.99;
- `ε`.

Each is persisted with the decision it produced, so it can be re-derived rather than inherited.

**Stake used.** Time on the shared pipeline, the daily cap, and interference with Juniper's chat.

**Trace that proves it**
- a hold decision whose recorded `interactive_waiting > 0` at that timestamp;
- a run decision whose top-offered prior later shows a non-zero-nats `PriorRevision` from that run;
- realized-nats rows for at least 90% of completed runs;
- an arm label on every decision.

**Off switch.** One flag restores the clock trigger. The offer records and the reducer are read-only
and harmless to leave in place.

**Dangerous failure mode: Goodhart.** If Orion were shown its own yield, it could learn, within the
prompt, to swing its confidence to look productive. In v1, realized nats and yield never enter the
kickoff prompt. Only the scheduler reads them. P7 is where that line gets reconsidered, with a
calibration check in place.

### P2. Refusals and waits flow back upstream

**What changes.** Each spend point records its verdict per candidate, with a reason. Attention
producers read the latest verdict for their candidate. There are two kinds of verdict, and they are
handled differently:
- **Already learned** (the allocator's `below_information_floor`, or P1 yield near 0): the loop loses
  standing.
- **Unaffordable right now** (`allowance_exhausted`, interactive work waiting): the loop keeps
  standing and waits.

The allocator already records this distinction in its refusal reasons, and nobody reads it.

**Two concrete changes**
- Reverie's narration tick, the path in the 09-05 incident, holds while any interactive lease is
  waiting in the GPU pool.
- A workspace loop whose candidate was refused as already learned gets an `attention_loop_outcome`
  row with `actor="orion"`. It is then excluded through the existing verdict path
  (`orion/substrate/attention/verdicts.py:125-180`, same TTL). This is the first verdict Orion writes
  for itself. It complements #2255's `actor=orion` "resolved" verdict; it does not replace it.

**Stake used.** GPU time (reverie) and motor-seconds (allocator verdicts).

**Trace.** A reverie hold log line carrying the pool reading; an excluded workspace loop whose
verdict row has `actor=orion` and reason `below_information_floor`.

**Off switch.** One flag per consumer.

### P3. Connect attention to spending

- **Adopt #2255's camera loop:** workspace winner → proposal → allocator → `verify_expectation` →
  `actor=orion` resolution. It is the motor-side twin of P1, and it touches different services, so
  the two can proceed in parallel.
- **Decide `inspect_attended_target`.** Either declare an `expected_signal` it can actually move, or
  retire it. As written, it can't pass the allocator.
- **Let the workspace winner into P1's offer.** It enters as a question about the attended node,
  valued by the same rule: the entropy of a related prior where one exists, otherwise a one-time
  cold-start value like the allocator's cold-start exemption. This replaces the unreachable seed
  path (G1) instead of lowering its threshold to make it fire.

### P4. Spend Juniper's attention as the scarcest thing Orion has

- **Record the expectation.** Every outreach records which prior her answer would move, the entropy
  on the line, and whether a reply is expected.
- **Score the reply.** Her reply is already attributed through `in_reply_to`. Score it the same way
  as P1: did a prior move in the next run that cites the reply? That gives realized nats per
  interruption.
- **Give Door A the gates.** Door A stops skipping the schedule gates. Both doors share one paced
  price over the daily cap of 4.
- **Hand off when self-study stops paying.** When a prior's own yield (P1c) is near 0 and its
  entropy is still high, the value of asking someone else is that remaining entropy. This is the
  measured form of ask-Claude's "tested three or more times and still unsettled" trigger
  (`orion/autonomy/ask_claude_trigger.py`). Who gets asked is decided by which budget clears (Juniper's
  paced price, or the Cursor or Claude quota gates after D2), so no exchange rate is needed.
- **Ride the existing contract.** This uses #2255's "asks and waits" contract
  (`HelpRequestV1 → PeerBriefV1`, with Juniper as a peer), not a new ledger.

### P5. Turn the control arms on

- **Dispatch.** Set the dispatch holdback above 0 (#2255 stage 4).
- **Curiosity.** P1's ε arm.
- **Outreach.** Hold back a random fraction of eligible outreach. That means sometimes deliberately
  not messaging Juniper, so it is her call (see Missing questions).
- **Propensities.** Record the propensity on every spend decision, so later work can estimate what
  another policy would have bought (inverse-propensity weighting).

Without control arms, "the value trigger learned more per GPU-hour" is a story, not a finding.

### P6. Prices for the rest of the stakes, after P1 and P2 prove the pattern

- **Reading wallet.** Rank seeds by expected nats against open priors once reading has an outcome
  path. That path is blocked today: reading never writes the graph (#2199), and #2255 missing
  question 2 asks for the adoption step.
- **Power.** No budget exists, so don't invent one. When the power-budget design's Stage 1
  (`power_draw_log`) lands, watts become a native cost unit like the others. The diffusion host's
  own-watts prior (`services/orion-diffusion-host/app/power_prior.py`) is already the template for
  predicted versus actual.
- **Claude quota.** After D2, turn the Claude-limit publisher on before any spend decision reads it.
- **Durable-run admission.** Order the agent-lane queue by expected nats per lane-second instead of
  first come, first served. This waits until P1's realized nats exist to estimate it.
  (The 2026-09-22 backlog, 7 runs with the oldest at 14h, was a lane-eligibility bug that has since
  been fixed. It is not evidence against first come, first served:
  `services/orion-durable-runs/README.md:319`.)

### P7. Attention that predicts its own spending (later)

Once P1 through P5 are producing outcomes, the attention self-model predicts its next spend and that
spend's expected nats. Realized against predicted then becomes a calibration score Orion can report.

On Graziano's attention-schema theory, a model of your own attention earns its keep by controlling
attention. Today's schema only narrates: `bottom_up_salience` accounts for 19,774 of 19,774 rows
(`orion/sentience_striving_program/README.md` §16a). P7 comes only after there is data to calibrate
against.

### Prerequisite, running alongside: inputs attention can trust

Fix these before any value estimate reads them:
- **Per-channel freshness.** Check `node_vector_updated_at`, not the age of the whole row (G8).
- **Stop the harness-closure constant from feeding curiosity.** Either stop feeding it, or replace
  it with a measured surprise level. The code already names the missing field:
  `surprise_level_at_draft`, at `worker.py:134`.
- **Rename vision's staleness channel,** so it stops posing as prediction error.
- **One threshold per domain,** not one for all of them.

P1 does not depend on any of these, because it reads Orion's own belief revisions, not field
signals. P2 and P3 do depend on them.

## Missing questions

1. **Is Orion's own stated confidence the right thing to score?** P1 scores what Orion writes in
   `:PriorRevision`, which is Orion's own belief, not ground truth. That is the right measure of what
   Orion learned. If you want an outside reference as well, the camera expectation checks in #2255
   are one.
2. **May curiosity skip turns?** With a value trigger, a quiet day might run 2 investigations instead
   of 7. That is the point, but you'll notice it.
3. **Does background thinking always yield to your chat?** P1d and P2 make curiosity and reverie
   wait while an interactive lease is waiting. That is a policy, stated so you can overrule it.
4. **Is an outreach control arm OK?** It means sometimes deliberately not messaging you.
5. **P1 first, or #2255's camera loop first?** They touch different services and can run in
   parallel. If only one goes first, I'd pick P1, because its outcome data already exists and it
   needs no new dispatch kind.
6. **`inspect_attended_target`:** give it an expected signal, or retire it?

## Proposed schema / API changes

Scope is P1 only; later proposals get their own specs.

- **Added table `curiosity_spend_decisions`.** One row per scheduler decision that passed the
  existing gates. It follows the pattern of `endogenous_outreach_decisions`: a manual migration in
  `services/orion-sql-db/` and a small Hub writer like
  `services/orion-hub/scripts/endogenous_outreach_decisions.py`. Columns:
  - identity and result: `decision_id`, `decided_at`, `action` (run | hold), `reason`,
    `arm` (value | clock_control), `propensity`;
  - the numbers the decision used: `price`, `floor`, `best_expected_nats`, `interactive_waiting`;
  - `offered` (jsonb, one entry per prior: `prior_id`, `confidence`, `entropy_nats`, `yield`,
    `expected_nats`; plus the material ids);
  - `constants` (jsonb: k, η, ε, clamp, floor definition);
  - `run_id` (nullable).
- **Added table `curiosity_run_outcomes`.** One row per completed run: `run_id`, `realized_nats`
  (nullable, since an unreadable graph is not zero), `n_revisions`, `n_formed`, and `per_prior` as
  jsonb (from, to, nats).
- **Bus / channels.** No new channel. Decisions reach the self surfaces as `AttentionSchemaV1` on
  `orion:attention:schema`. `attention_reason` is free-text vocabulary owned by each process
  (`orion/schemas/attention_schema.py`), so new values need no schema change.
- **P2 later.** `attention_loop_outcome.actor` is already a free string (`orion/schemas/attention_salience.py:71`).
  An `actor="orion"` row needs no migration. A new *verdict* value would need the
  `AttentionOutcomeVerdictV1` Literal extended (`:22`) and the sql-writer rebuilt first
  (consumer-first).
- **Metric registration.** Register `curiosity_realized_nats` and `curiosity_price` in the existing
  metric semantic layer (the inner-state registry that `orion/metrics/lineage.py` resolves), not in
  a new registry.
- **Env (services/orion-hub).** Four keys, added to `.env_example`, synced into the local `.env`
  with `scripts/sync_local_env_from_example.py`, and wired through settings and compose in the same
  changeset:
  - `HUB_CURIOSITY_VALUE_TRIGGER_ENABLED`, default `false` until the realized-nats gate passes live;
  - `HUB_CURIOSITY_VALUE_TRIGGER_CONTROL_FRACTION` (ε, suggested 0.2);
  - `HUB_CURIOSITY_PRICE_STEP` (η);
  - `HUB_CURIOSITY_YIELD_PSEUDO_TESTS` (k).

## Files likely to touch

For P1, plus the two defect fixes:

- **Core logic.** `orion/curiosity/` gets a new `value.py`: entropy, Bernoulli KL, yield, price and
  floor. These are pure functions with no I/O.
- **Graph reads.** `orion/curiosity/atlas.py` and `orion/curiosity/worldview.py` supply the
  `PriorRevision` and prior reads. Reuse the existing readers; add no new Cypher.
- **Scheduler.** `services/orion-hub/scripts/curiosity_investigation.py`: the value/price gate goes
  after `scheduling_block_reason`, plus the offer record and the control arm.
- **Hub writers.** New `services/orion-hub/scripts/curiosity_spend_decisions.py` (the writer) and
  the outcome reducer, which runs after `read_turn_outcome`.
- **GPU pool read.** A read of `gpu_pool_leases` for interactive leases that are waiting. Use the
  pool's existing Hub read path (`services/orion-hub/scripts/gpu_pool_routes.py`).
- **Migrations and self surface.** Migrations in `services/orion-sql-db/`. `AttentionSchemaV1`
  publishing goes through `orion/curiosity/attention_schema.py`, the existing write-only adapter,
  whose lane currently writes 0 rows.
- **Config.** `services/orion-hub/app/settings.py`, `.env_example`, the local `.env`,
  `docker-compose.yml`, and `README.md`.
- **Metrics.** Registry entries in the existing metric semantic layer.
- **Tests.** Unit tests for `value.py`, including a KL rest point of exactly 0 when from equals to,
  and yield shrinkage at n = 0. Add scheduler gate tests, plus a replay eval over historical
  `PriorRevision` rows.
- **D1.** `orion/attention/field_attention/scoring.py` and `tests/test_attention_field_selectors.py`.
- **D2.** `services/orion-curiosity-peer/app/worker.py` and its tests.

## Non-goals

- No new service, no attention registry, no taxonomy of attention kinds, no drives.
- No hand-typed exchange rate between budgets. Each budget judges in its own unit.
- Do not lower the allocator's floor. It is correct; the fix is better candidates (#2255).
- Do not merge or replace the three salience producers in this pass.
- Do not show realized nats or yield to the LLM in v1 (Goodhart).
- Do not change the GPU pool's route priorities. Interactive-first is right. The fix belongs on the
  attention side, where background attention holds back.
- Do not bring back habituation as a penalty constant. It should emerge from yield (rule 4).
- Do not solve attention supply here; that is #2109.

## Acceptance checks

**For P1**, on the live rail, not only in tests:

1. **The realized-nats metric passes the gate on history before any decision reads it.** A read-only
   replay over every existing `PriorRevision` row must report:
   - count, distinct values, fraction of zeros, and p50/p90;
   - how coarse the confidence values are (LLM-written numbers may snap to 0.05 steps);
   - the fraction of runs with at least one revision.

   It must also show the rest state can happen: runs with no revision read exactly 0, and unreadable
   runs read `null`. If the numbers are degenerate (almost always 0, or almost never), P1c and P1d
   don't ship. P1a and P1b ship anyway, as measurement.
2. At least 95% of curiosity scheduler decisions that passed the existing gates have a
   `curiosity_spend_decisions` row with at least 2 offered priors, an arm, and a propensity.
3. At least 90% of completed runs have a `curiosity_run_outcomes` row.
4. **Holds happen, and for real reasons.** At least one `hold` row has `interactive_waiting > 0` at
   that timestamp, and at least one has `best_expected_nats < max(price, floor)`.
5. **Price returns to rest.** On a day with fewer runs than the cap, `price` reaches exactly 0 at some
   point. On a clustered day, it rises above 0.
6. **The midnight cluster is gone.** Under the value arm, runs are not all within a few hours of
   local midnight.
7. **The comparison is stated in advance.** After 14 days, compare mean realized nats per run between
   the value arm and the clock arm, with the effect size and its interval. A null result is reported
   as null.
8. **No harm to chat.** `stance_react` deferrals during curiosity turns do not rise compared with the
   14 days before, measured on the same log lines.
9. **Orion can explain itself.** At least one chat turn cites a curiosity `held_off` or `invested`
   `AttentionSchemaV1` row through the recent-attention cue. Since that cue's live rendering is
   `UNVERIFIED` (#2255 missing question 3), check this with a stored trace.

**For D1:** the steady-input regression test passes. Also, on live frames, the fraction of
host/capability targets whose novelty alternates between 0 and a steady value drops to about 0.

**For D2:** a default-path test shows the Claude fallback calling `rate_limit_events.observe`.

## Metric quality gate records

Required by CLAUDE.md 0A before wiring.

**`curiosity_realized_nats`, per run**
1. **Provenance.** Orion writes `:PriorRevision {from_confidence, to_confidence}` in-turn, as
   instructed at `orion/curiosity/kickoff_prompt.py:680`. It is read by
   `orion/curiosity/atlas.py:117` and `orion/curiosity/attention_schema.py:81`. The KL is new code.
2. **Independence.** A new sensor: Orion's own stated belief. It is not a transform of any field,
   biometric or prediction-error channel. It is related to the hop supervisor's `HopReadingV1`
   (0 rows, command-line only), which grades hops and does not measure belief change. Not redundant.
3. **Theory anchor.** Bayesian surprise, KL(posterior ‖ prior) (Itti & Baldi 2009). It is the same
   unit as the allocator's `surprise_nats` (`orion/autonomy/prediction.py:140`), which is what makes
   curiosity and motor outcomes comparable without an exchange rate.
4. **Live-data sanity.** `UNVERIFIED`. This is Acceptance check 1, and it has to pass first.
   - *Known hazard:* LLM-written confidences may be coarse or stick to round numbers.
   - *Rest state:* exactly 0 when from equals to, by construction. Check that it really occurs,
     and that unreadable runs are `null`, not 0.
5. **Existing mechanism.** `atlas.py:366-368` already computes the raw difference (to − from), and
   `attention_schema.py` narrates from and to. No KL, and no per-run aggregate, exists. Reuse both
   readers.
6. **Reversibility.** A derived table and a flag. Nothing enters a training default or a wire
   schema. Cheap to remove.

**`curiosity_price`, the dual price per tick**
1. **Provenance.** Computed from the curiosity daily counter, which already exists (Redis, local-date
   keys), and `HUB_CURIOSITY_INVESTIGATION_DAILY_CAP`.
2. **Independence.** It is the state of a controller, not a sensor. It depends only on this budget's
   own spending.
3. **Theory anchor.** Adaptive pacing by dual descent (Balseiro & Gur 2019), and bandits with
   knapsacks (Badanidiyuru, Kleinberg & Slivkins 2013).
4. **Live-data sanity.** `UNVERIFIED`. Replay the last 30 days of curiosity start times. The price
   must reach 0 on under-spent days and rise during clustered stretches (Acceptance check 5).
5. **Existing mechanism.** `paced_cooldown_sec` paces by clock. This keeps it as a ceiling and
   replaces it as the trigger.
6. **Reversibility.** One flag.

**`interactive_waiting`.** A count of waiting interactive-priority leases, from the pool's own
table. It is a raw count with a true zero, not a derived metric, so steps 2 to 4 do not apply. Step 1
is `services/orion-gpu-pool/app/runtime.py:487`.

## Recommended next patch

This is two pull requests. They are independent, so they can land in either order.

1. **`fix/attention-novelty-and-peer-claude-meter`: D1 and D2.** One-line fixes with regression
   tests. D1 changes live attention behaviour, by removing manufactured novelty. So it needs the
   usual review, but not a design gate: it restores the documented intent.
2. **`feat/curiosity-pays-for-what-it-learns`: P1, in two commits.**
   - *Commit 1: measurement.* `value.py`, the outcome reducer, the offer record, the migrations, and
     the replay script that runs the realized-nats gate over history.
   - *Commit 2: behaviour, behind `HUB_CURIOSITY_VALUE_TRIGGER_ENABLED=false`.* The value/price
     trigger, the control arm, and the `AttentionSchemaV1` "why" rows.

   The flag flips only after the replay shows the metric isn't degenerate (Acceptance check 1).

**After those:** P2 (refusals flow upstream; reverie holds while interactive work waits), then P3 and
#2255's camera loop, then P4 (Juniper's attention and hand-off), and P5 alongside each of them. P6
and P7 come once there is outcome data.

## Sources

- **This repo, the attention history.**
  - `docs/architecture/attention-salience/README.md`
  - `orion/sentience_striving_program/README.md` §12, §13 and §16
  - `docs/notes/2026-07-30-chat-attention-ground-truth-gap-finding.md`
  - `docs/notes/2026-07-24-attention-reason-branch-starvation-finding.md`
- **This repo, stakes and incidents.**
  - `docs/superpowers/pr-reports/2026-09-05-stance-react-attention-salience-gateway-starvation-incident.md`
  - `docs/superpowers/pr-reports/2026-09-25-world-pulse-wallet-refund-pr.md`
  - `docs/superpowers/specs/2026-09-20-queue-contention-metric-gate.md`
  - `docs/superpowers/pr-reports/2026-08-30-motor-budget-enforcement-pr.md`
- **Open PRs.** #2255 (prerequisites assessment and "asks and waits"); #2109 (concept attention
  supply).
- **Theory.**
  - Lindley 1956, "On a measure of the information provided by an experiment"
  - Itti & Baldi 2009, "Bayesian surprise attracts human attention"
  - Feldman & Friston 2010, "Attention, uncertainty, and free-energy"
  - Oudeyer, Kaplan & Hafner 2007, "Intrinsic motivation systems for autonomous mental development"
  - Badanidiyuru, Kleinberg & Slivkins 2013, "Bandits with knapsacks"
  - Balseiro & Gur 2019, "Learning in repeated auctions with budgets"
  - Burda et al. 2018, "Large-scale study of curiosity-driven learning"
  - Graziano 2013, *Consciousness and the Social Brain*
