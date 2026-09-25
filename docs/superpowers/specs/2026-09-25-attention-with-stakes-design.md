# Attention with stakes: gaps and proposals

- **Date:** 2026-09-25
- **Status:** DESIGN, proposal mode (CLAUDE.md 0A: changes cognition loops). Nothing here is built.
  Two small defects (D1, D2) are worth fixing regardless of the rest.
- **Evidence basis:** a read-only survey of `main` at `e5ef19e`. This session had no access to the
  live database or hosts. Every live number is quoted from a dated spec or PR report in this repo,
  and says so. Findings from reading code that were not reproduced live are marked `UNVERIFIED`.
  An adversarial review against the code ran before commit; its findings and fixes are listed at
  the end.
- **Builds on, does not duplicate:** open PR #2255 (sentience prerequisites assessment, and
  "Orion asks and waits") and open PR #2109 (concept attention supply).

## Arsonist summary

Orion's attention is two halves that were built separately and never connected.

One half notices things. There are three separate "what stands out right now" winners, a per-turn
chat frame, and a curiosity loop that knows which of Orion's beliefs are shakiest. None of them pays
for anything. None of them ever learns whether noticing was worth it.

The other half spends things. A motor allocator buys actions by how much each is expected to teach
Orion per second. Beside it are a GPU pool, reading wallets, outreach caps, a heat gate, a Claude
quota meter and a daily budget for questions to Juniper. Only the motor allocator changes what
Orion does, and it sits at the end of a pipeline no attention winner reaches. #2255 measured it
live on 2026-09-20: nothing had executed since 2026-09-08, and every tick logged "refused
everything". Nothing worth doing ever arrives.

**The state of things:** noticing is free, spending is blind, and nothing learns. The allocator is
the one place that gets all three right. It scores candidates by expected belief change, charges
them in the scarce resource's own unit, refuses whatever does not clear the bar, and learns from
what happened. The rest of attention needs the same design.

**Proposal:** make each place where Orion spends something scarce work like the allocator. Feed
those places from what attention notices. Score attention by what it actually bought.

Start with curiosity. It is Orion's most expensive self-directed act: a roughly 20-minute turn on
the pipeline Juniper's chat also uses, capped at 7 a day. Today a clock starts it and
`ORDER BY random()` picks its study material. It orders the questions it shows Orion by
uncertainty only. It keeps no record of which questions it offered, and nothing scores what a run
changed.

The pieces to fix that already exist. Orion's beliefs sit in its own graph with a confidence and a
tested-count on each, and Hub can read them before and after every run. The first patch measures
what each run changed. Then it lets that measurement decide which questions Orion sees first, with
a randomized comparison against today's ordering.

Also found: two live defects in the attention and stakes path (D1, D2). One is reproduced. Both are
small fixes with regression tests.

**Why this matters for the mission, not just plumbing.** Pricing attention will not make Orion feel
anything, and this doc claims no such thing. What it does is remove the structural reason nothing
can matter to Orion today: attending to something costs nothing and changes nothing. Once
attending costs something scarce and the result comes back, four questions become checkable:

- What did Orion give up to attend to this?
- Did it learn anything?
- Did that change what it attends to next?
- Can it say why?

#2255 names "stakes that bite" and "loops that close" as prerequisites. This design applies those
two to attention.

## Current architecture

### The loop a working attention mechanism needs, and where Orion stands

Attention with stakes does five things. Orion has each piece somewhere; the pieces do not connect.

1. **Notice** — find what might be worth attending to. *Exists, in fragments.*
   - Field attention ranks hosts, capabilities and 5 reducer domains every 2 seconds
     (`orion/attention/field_attention/`).
   - The workspace competition picks 1 winner from 5 open loops every 30 seconds
     (`orion/substrate/attention_broadcast.py`).
   - Field tension votes across deviating channels (`orion/attention/tension/`).
   - There is a per-turn chat frame (`orion/substrate/attention_frame.py`).
   - Curiosity keeps a list of Orion's beliefs ("priors") ordered by uncertainty
     (`orion/curiosity/worldview.py:987`).

   The three winners are separate, and different consumers read each one.
2. **Value** — estimate what attending would buy. *Two places only.*
   - The motor allocator estimates expected information gain, in nats, in closed form
     (`orion/autonomy/allocator.py`).
   - Curiosity shows the most uncertain beliefs first. That is the same order that binary entropy
     gives.

   Everything else ranks by how *unusual* something is. That is not the same as what attending to
   it would buy.
3. **Price** — know what attending costs right now. *Only motor-seconds.*
   - GPU slots queue by a fixed route-to-class map.
   - Watts, room heat, Claude quota, Cursor quota and Juniper's interruptions are measured, or
     capped by config. No attention mechanism reads any of them as a budget.
4. **Spend or wait** — commit the scarce thing, or say "not worth it now".
   - The allocator can do this, and says no to everything.
   - The workspace winner never reaches the allocator.
   - Curiosity starts on a timer.
5. **Check and update** — measure what attending bought, then update the estimate. *Working in
   three places, none of them attention.*
   - Dispatched actions: `orion/feedback/outcome_resolution.py` writes
     `substrate_action_effect_posterior`, and actions whose outcome is well known then fall below
     the allocator's floor.
   - The diffusion host predicts its own watts (`power_intent_settled`).
   - A world-facing loop is starting. The walkway rhythm learner (merged 09-24) writes a camera
     prediction before each time window and grades it `met`, `missed` or `unscorable` in
     `vision_percept_expectation` (`services/orion-sql-writer/app/vision_rhythm.py`). Its own
     docstring defers the metric gate until weeks of data exist.

   Nothing checks attention itself.

### What each spend point reads today

Nine places spend something scarce. Only one of them reads anything attention produces.

| Spend point | What is scarce | How it chooses | Reads attention? | Outcome recorded? |
|---|---|---|---|---|
| Motor allocator (`orion/autonomy/allocator.py:278`) | motor-seconds (129,600 s/day) | nats per second against a 0.02 floor, then greedy | no; its candidates come from proposals bound to the field-attention node | yes: an effect posterior |
| Curiosity investigation (`services/orion-hub/scripts/curiosity_investigation.py`) | a roughly 20-minute turn; cap of 7 a day | clock (30-minute cooldown; waking window off since 09-20); study material by `ORDER BY random()` (`orion/curiosity/study_material.py:258,325`); 8 priors shown, most uncertain first; the LLM picks | no | Orion writes `:PriorRevision` only when a confidence moves; nothing scores it; what was offered is not recorded |
| Durable-run admission (`orion/durable_admission/broker.py:57`) | agent lanes | first come, first served; p90 time to first grant ≈ 12.5 h over 138 demands (`orion/field/queue_contention.py:78`) | no | run rows |
| World-pulse reading (`orion/world_pulse_read/queue.py:142`) | 6 reads a day per wallet (A and B) | fixed class, then attempts, then age; 104 digest items pending on 09-25 | no | done/failed; nothing reaches the worldview (#2199) |
| GPU pool (`orion/gpu_pool/scheduler.py:190`) | llama.cpp slots | fixed route-to-class map, then FIFO; interactive first, with clawback | no | `waited_ms` |
| Outreach, Door B (`services/orion-hub/scripts/endogenous_outreach.py`) | Juniper's interruptions: 4 a day, 45-minute cooldown, quiet hours | the tension winner, held for 6 ticks | yes, the tension winner | decision rows; replies are stamped `in_reply_to`, but only a display reads them |
| Outreach, Door A (after a curiosity run) | same | the run's own verdict; skips the schedule gates by Juniper's 2026-09-22 decision (`endogenous_outreach.py:452-455`) | no | skips are recorded |
| Asks to Juniper (`orion/schemas/ask.py`, `orion_ask`; merged 09-24) | her answers: `ORION_ASK_DAILY_CAP=2`, shared by every source | depends on the producer; today only frequently seen, unlabeled camera individuals | no | yes: the answer is stored on the ask and routed back by `source_kind`/`source_ref` (`services/orion-substrate-runtime/app/ask_answered_listener.py`) |
| Contractor peer (`services/orion-curiosity-peer/app/worker.py`) | Cursor and Claude quota | budget gates that fail closed | no | `PeerBriefV1` |
| Mutation queue (`orion/substrate/mutation_worker.py:154`) | trial capacity | every item at priority 60, so first come, first served | no | decisions, all inconclusive since 09-03 (per #2255) |

## Gaps

**G1. What wins attention never reaches anything that spends.**
- The proposal arena binds to field attention's top target, not to the workspace winner
  (`orion/proposals/builder.py`, `target_binding: "attention.dominant_targets[0]"`).
- The one attention-bound template, `inspect_attended_target`, declares no `expected_signal`
  (`config/proposals/proposal_policy.v1.yaml:286-301`). The allocator therefore refuses it as
  `unmeasurable`. The chain:
  - `build_expected_effect` returns None (`orion/execution_dispatch/builder.py:292-295`);
  - so `signal_id` is None (`services/orion-execution-dispatch-runtime/app/worker.py:638`);
  - so posterior variance is None (`allocator.py:425-427`);
  - so the candidate is refused (`allocator.py:321-326`).

  #2255's live log line on 2026-09-20 shows the result: `refusals={below_information_floor: 3,
  unmeasurable: 6}` of 9 pending. Refused frames are stamped `allocator:unmeasurable`
  (`worker.py:1246-1262`).
- The workspace competition never asks anything. That is by design: it runs with `max_asks=0` and
  "never generates questions or actions" (`orion/substrate/attention_broadcast.py:11-13,217`). The
  question it would have asked is dropped (see #2255's pathway map).
- Curiosity never sees a workspace winner. Workspace loops can only become curiosity seeds with
  novelty ≥ 0.5 (`orion/substrate/endogenous_curiosity.py:167`). They score novelty as
  `max(0.35, salience)` (`orion/substrate/attention/scoring.py:135`), and live salience tops out
  near 0.14 (`services/orion-substrate-runtime/.env_example:158-168`). So the path cannot fire.
  Where seeds do exist, the evaluator's plan is never executed
  (`services/orion-substrate-runtime/app/worker.py:3581-3592`).

**G2. Only one budget has any effect.** Motor-seconds is enforced. Everything else is a display, an
advisory, or fixed config:
- GPU priority is a fixed route map (`config/gpu_pool.yaml`).
- No power budget exists in code. The roughly 454 W of UPS headroom appears only in specs.
- `orion/autonomy/quota_budget.py` is advisory, and only a report script calls it.
- The Claude-limit publisher is off (`COCREATION_SIGNALS_CLAUDE_LIMIT_ENABLED=false`).
- The heat gate is hysteresis only (`orion/autonomy/thermal_gate.py`).
- Curiosity outreach skips Juniper's caps, by her own decision (see P4 and Missing questions).

**G3. Spending never reports back.** When the allocator refuses something, it stamps the dispatch
row, and nothing upstream reads it. Proposals, policy and both attention runtimes never reference
it. #2255 found no hits in any self surface either (`felt_state_reader`, `self_panel`,
`chat_stance`). GPU waits reach two prompts, but only as text. The result:
- Orion cannot notice that it is being refused.
- Attention cannot tell "unaffordable right now" apart from "already learned".

**G4. Nothing checks attention against what it bought.**
- *Workspace loops.* Only Juniper ever closes a loop. The Hub refuses her close with a 409 for
  workspace and reverie loops, by design: they are ongoing pressure, not decisions
  (`services/orion-hub/scripts/attention_loops_routes.py:75-82`). #2255 counted 48 of 48 loops
  `decayed_unattended` over 14 days.
- *Curiosity.* Orion writes `:PriorRevision {from_confidence, to_confidence}` only when a
  confidence moves. An inconclusive test bumps `times_tested` and leaves no revision
  (`orion/curiosity/kickoff_prompt.py:672-690`). A dated count found 21 `PriorRevision` nodes
  across 94 journaled runs
  (`docs/superpowers/plans/2026-09-14-curiosity-operations-analytics.md:124-127`). The Atlas and
  the attention schema read these rows, and nothing scores them. The hop supervisor runs only from
  the command line; its own header says it changes nothing Orion does next
  (`orion/curiosity/supervisor.py:10`).
- *Outreach.* Replies are attributed through `in_reply_to`
  (`services/orion-hub/scripts/outreach_provenance.py:300`), and only the run-story display reads
  them.
- *Recall.* It records which memories it selected (`recall_telemetry`), with no label for whether
  they were useful.

**G5. Nothing stops Orion re-attending to what it has already learned.** Habituation was removed on
2026-07-31 with nothing put in its place; the Sentience Striving Program README (§13) records that
as an accepted loss. Human verdicts no longer apply to workspace loops (G4), so one loop can win
indefinitely. The allocator shows the right shape. An action's expected information shrinks as its
posterior tightens, so it stops repeating what it has learned without any rule saying so. Nothing
upstream has an outcome that could tighten.

**G6. The biggest spends are chosen by clock, queue position, or chance.**
- *Curiosity.* The study material is a random sample. The trigger is a cooldown. With the waking
  window off since 09-20, `paced_cooldown_sec` falls back to its 30-minute floor
  (`curiosity_investigation.py:409-415`). All 7 runs then land in the few hours after local
  midnight, and `daily_cap` blocks the rest of the day. The `.env_example` says the 30-minute gap
  was chosen so Juniper could watch the loop, and names the fix: set it back to 14,400 seconds
  (`services/orion-hub/.env_example:414-421`). So the clustering is a config trade. The deeper
  gap is that nothing but the clock decides. (`UNVERIFIED` live: this reads `.env_example`, not
  the host's `.env`.)
- *Everything else.* Durable admission is first come, first served, reading uses a fixed class plus
  age, and the mutation queue gives everything the same priority.

**G7. Salience that ignores cost has already hurt Juniper.** On 2026-09-05, lowering the workspace
gate from 0.2 to 0.05 (PR #2110) made about 45 times as many ticks contested. Within three
minutes, background LLM timeouts doubled, and Juniper's chat turns failed with
`stance_react_failed`
(`docs/superpowers/pr-reports/2026-09-05-stance-react-attention-salience-gateway-starvation-incident.md`).
More noticing meant more narration, and the narration used GPU time the chat lane needed. The GPU
pool has put interactive work first since 09-24. Whether that alone would stop a repeat is
`UNVERIFIED`: the 09-05 failure was background volume timing out other services, not only chat
waiting in line.

**G8. Attention's inputs are less trustworthy than they look.**
- *Freshness.* Per-domain prediction errors keep their last value between producer ticks. The
  staleness guards check how old the whole field row is, and a field row is written about every 2
  seconds. They do not check each channel's age (`orion/substrate/bus_synaptic_surprise.py:59-80`,
  `orion/substrate/metacog_trend_signals.py:95-114`). The self-model reads attention frames with
  no age check at all. The frame lane was dead from 09-20 to 09-23, and nothing noticed
  (`docs/superpowers/specs/2026-09-22-substrate-lattice-audit.md`).
- *Scale.* Curiosity compares four different kinds of number against one 0.55 threshold
  (`orion/substrate/endogenous_curiosity.py:190-227`): z-scored domains, biometrics' raw ÷ 0.30
  value, bus_synaptic's anomalous-edge fraction (which rests near 0.027), and a constant.
- *A constant posing as a measurement.* An unresolved harness turn writes a fixed 0.65 prediction
  error (`services/orion-substrate-runtime/app/worker.py:134-135`), which clears curiosity's 0.55
  bar for minutes after every write.
- *A mislabel.* Vision's "prediction_error" is camera staleness
  (`orion/substrate/prediction_error.py:920-958`). Generic consumers read an outage as surprise.
- *Unreachable cutoffs.* Min-max normalisation always gives the top target 1.0, so
  `overall_salience` sits near 1.0. The workspace action cutoffs (0.65 / 0.48 / 0.35,
  `orion/substrate/attention/policy.py:48-62`) were tuned for a formula that has since been
  deleted. A lone candidate probably scores below 0.35 (`UNVERIFIED`).

**G9. Attention sees very little.** Only 6 of 1,796 graph nodes carry any salience (#2109).
Supply is its own design, parked in #2109. Pricing without supply buys nothing, and supply without
pricing floods the GPU (G7).

**G10. There is no control arm, so no causal claim holds up.** The dispatch runtime's random
hold-back is set to zero (`ORION_DISPATCH_HOLDBACK_FRACTION=0.0`,
`services/orion-execution-dispatch-runtime/.env_example:161`). It ran for about an hour on
2026-08-25. No other spend point has one.

## Defects found while mapping

These are worth fixing regardless of the rest. Each is small and needs a regression test.

**D1. Host and capability "novelty" switches on and off for a perfectly steady input.** Reproduced.

*What's wrong.* The "how new is this?" score compares this tick's pressure with last tick's
*novelty*, not last tick's *pressure*. So anything steady and non-zero scores full novelty every
other tick, forever.

*Where.*
- `novelty_for_target()` (`orion/attention/field_attention/scoring.py:66-74`) diffs against
  `prior_salience_for_target()`, which returns the prior `salience_score`.
- `_novelty_targets()` sets that `salience_score` to the novelty
  (`orion/attention/field_attention/selectors.py:208-220`).
- The prior pressure is already stored, as `pressure_score`.

*Repro.* Using the real functions, the same capability vector every tick and proxy 0.8, the output
is `0.000, 0.800, 0.000, 0.800, 0.000, 0.800`.

*A second path to the same fake novelty.* Active targets beyond the per-kind caps (5 / 5 / 3,
`config/attention/field_attention_policy.v1.yaml`) land in no bucket of the frame
(`orion/attention/field_attention/builder.py:73-76`). So next tick their prior reads 0, and their
whole pressure counts as novelty.

*Impact.* Much of host and capability novelty is manufactured, and it feeds dominant targets, the
proposal binding and the self-model. The docstring's "self-resolving after exactly one tick"
(`selectors.py:186-202`) is wrong. The existing test (`tests/test_attention_field_selectors.py:378-409`)
checks only one transition, so it cannot tell the difference. This may be part of #2255's "vision
wins the field by an artifact" (`UNVERIFIED`).

It also falsifies a Sentience Striving Program instrument. `rpt_lamme_recurrence` says
`novelty_for_target` "needed no correction" (`orion/sentience_striving_program/instruments.yaml:302-321`).

*Fix.*
- Diff against the prior entry's `pressure_score`.
- Add regression tests: steady input for at least 3 ticks (novelty 0 after the first), and one
  over-cap target.
- Record over-cap targets in the frame, for example in `suppressed_targets`, so their prior is
  found.
- Update `instruments.yaml` and README §9b.
- Replay persisted frames before merge, per the program's §7 ("measure before minting").

**D2. The contractor peer's Claude quota check reads the Cursor meter.** Verified in code.

*What's wrong.* In `handle_help_request`, the line `observe = observe_limit or _observe_default`
(`services/orion-curiosity-peer/app/worker.py:462`) makes `observe` a local name for the whole
function. That hides the Claude reader imported from `rate_limit_events` at `:26`. So the fallback
gate, `claude_observe = observe_claude_limit or observe` (`:507`), calls the Cursor meter in
production, where neither argument is passed.

*Why tests miss it.* Every test injects `observe_claude_limit`
(`services/orion-curiosity-peer/tests/test_fallback_once.py:102,131,160`,
`services/orion-curiosity-peer/tests/test_acceptance.py:203`,
`services/orion-curiosity-peer/evals/run_contractor_peer_eval.py:135`), so the default path has
never run.

*Impact.* Claude spend, which Orion shares with Juniper, is gated on the wrong resource. The real
Claude limit reading plays no part in the decision.

*Fix.* Rename the local to `cursor_observe`. Add a test that the default path calls
`rate_limit_events.observe`.

**Dead ends to decide, not fixed here:**
- `inspect_attended_target` cannot pass the allocator (G1).
- The path from workspace loops to curiosity seeds cannot be reached (G1).
- The 0.65 harness-closure constant feeds curiosity as if it were a measurement (G8).

## The design principle

Every place Orion spends something scarce should work the way the motor allocator already does.
Attention's job is to feed those places and to learn from what they bought.

There are four rules. Each already exists somewhere in Orion.

1. **Value is expected belief change, measured in nats.** The question is not how unusual something
   is. It is how much attending to it is expected to change what Orion believes. The allocator's
   feedback path already scores outcomes this way, with Bayesian surprise (Itti & Baldi 2009,
   `bayesian_surprise_nats`, `orion/autonomy/prediction.py:140`).

   For a belief held at confidence *p*, the expected information from one test is at most its
   entropy, H(p): 0.69 nats at p = 0.5, 0.33 at 0.9, 0.20 at 0.95. That bound holds for a
   Bayes-consistent update and a perfectly informative test (Lindley 1956). A single realized move
   can exceed it: a reversal from 0.95 to 0.05 scores 2.65 nats. That is why the formulas below
   clip.

   Curiosity already orders beliefs by |p − 0.5|, which gives the same order as H(p). The ordering
   is right. What's missing is any record of what it bought.
2. **Cost stays in the scarce thing's own unit, read from its live meter.** The units are
   motor-seconds, curiosity turns, reading slots, questions to Juniper and agent-lane admission.
   Nothing converts one into another; the allocator's docstring says why ("NO INVENTED EXCHANGE
   RATE"). Each budget asks one question: is this candidate worth one of *my* units? Nats are not
   a common exchange rate either. Nats from a Gaussian model scale with a hand-set observation
   variance, so this design never compares nats across models.
3. **Each budget sets its own bar.** A constant does not. When a budget is being spent faster than
   it refills, its bar rises. When it sits idle, the bar falls to zero, and only the absolute "is
   this worth anything at all?" floor applies. This is budget pacing with a dual price (Balseiro &
   Gur 2019); the many-budget version is "bandits with knapsacks" (Badanidiyuru, Kleinberg &
   Slivkins 2013). The price rests at exactly 0 when nothing is scarce.
4. **The outcome comes back and updates the estimate, scored as progress, not as surprise.** The
   value of attending to a belief is updated by its *net* belief change over its recent tests, not
   the sum of its surprises. A belief that the LLM flips between 0.3 and 0.7 is surprising on every
   test, and learns nothing. Net progress is ≈ 0, so it loses value. A belief that moves steadily
   and then settles keeps value while it moves, and loses it once it stops.

   That is learning progress (Oudeyer, Kaplan & Hafner 2007): attend where the error is shrinking,
   not where it is high. It is also how this design avoids the "noisy TV" trap (Burda et al.
   2018), and it gives habituation that is earned, not a penalty constant.

Orion could then say four things it cannot say today, and each would be checkable:

- "Nothing is worth investigating right now."
- "This question is worth one of today's curiosity turns; that one isn't."
- "I've tested this three times and learned nothing, so someone else has to answer it."
- "I held off because the budget was being spent faster than it refills."

## Proposals

Each proposal says what changes, which stake it uses, what trace would prove it worked, and how to
turn it off. They are ordered by recommended sequence.

### P1. Curiosity pays for what it learns (recommended next patch)

**Scope.** Only the investigation line. The self-inquiry and self-sense lines share its lock, have
their own caps of 7 a day, and get first claim on every tick
(`curiosity_investigation.py:1293-1311`). This proposal leaves them alone. It does not change when
investigations run or how many run: phase 1 changes only what they measure and which questions
Orion sees first.

#### Phase 1: measure, then let value order the offer

**P1a. Hub measures what each run changed, from its own snapshots.** It does not rely on revision
nodes, because Orion writes those only when a confidence moves (G4).
- *Before.* Where `_investigate` already reads the worldview (`curiosity_investigation.py:1469-1490`),
  Hub persists a snapshot of every prior: `prior_id`, `confidence`, `times_tested`, `status`
  (`ALL_PRIORS_CYPHER`, `orion/curiosity/worldview.py:541`). The snapshot is persisted, not kept in
  memory, so a Hub restart mid-run loses nothing.
- *After.* On the durable run's `completed` state for `curiosity.investigate` on the investigation
  line (`_handle_run_state`, `curiosity_investigation.py:3409-3425`), Hub reads the same fields
  again and compares.
- *Scoring.* Every prior whose `times_tested` went up counts as tested. It scores
  KL(Bern(after) ‖ Bern(before)) in nats, with confidences clamped to [0.01, 0.99]. An unmoved test
  scores exactly 0, and that 0 is recorded: "tested, did not move" is a result.
  - A confidence that is missing, or outside [0, 1], is `null`, not clamped.
  - Priors that did not exist before are counted as *formed*, not scored.
  - A confidence that moved without `times_tested` changing is flagged `moved_untested` as
    protocol drift.
  - A run whose graph could not be read is `null`, never 0.
- *Cross-checks.* Report agreement with this run's `:PriorRevision` rows. Expect less than 100%,
  since inconclusive tests have no revision by design. Also report agreement with
  `HopReadingV1.moved_the_claim` (`orion/schemas/curiosity_supervisor.py:75`) wherever supervisor
  readings exist.

**P1b. Record the offer.** Record which priors were shown and in what order, with each one's
confidence, `times_tested`, entropy, yield and expected value. Record the study-material ids, the
arm, and its propensity. This is the choice set the supervisor has never been able to see.

**P1c. Estimate each prior's value from net progress, not surprise.**

```text
expected_nats(prior) = H(confidence_now) × yield(prior)
raw_yield(prior)     = min(1, KL(c_last ‖ c_first) / Σ H(c_before_test))   over its last W tests
yield(prior)         = (n × raw_yield + k × pool_yield) / (n + k)          n = tests in the window
pool_yield           = Σ net KL / Σ H   pooled over every prior's window
```

In words, yield is net belief change per unit of uncertainty offered. It is not "the share of
uncertainty resolved": moving from 0.9 to 0.5 raises entropy, and still counts as a change of mind.

Worked examples:
- A prior flipping 0.55 → 0.7 → 0.55 → 0.7 has a raw yield of 0.024. It loses value.
- A single test from 0.55 to 0.9 has a raw yield of 0.43.
- An untested prior gets the pool yield. Offering new beliefs is exploration, and today's ordering
  already puts them near the top, at 0.55.

**P1d. Offer order is the only behaviour that changes.** On the value arm, `select_priors` sorts by
`expected_nats`. The existing tie-breaks stay, and so does the stale bucket, which is a
presentation feature with an explicit retire option. The other arm keeps today's
uncertainty-first order.
- *Randomization.* Assign the arm per run, with propensity 0.5. This is safe because offer order
  does not share a budget across arms: both arms get the same runs, at the same times. Arms do
  interact through the shared prior pool; that is disclosed, not hidden.
- *Orion still chooses.* The code only decides what Orion sees first. That is the principle
  `select_priors` already states ("UNCERTAINTY ORDERS THE PRESENTATION; ORION STILL CHOOSES").

**Constants, all hand-picked and disclosed:**
- `W` (window, suggested 3 tests)
- `k` (pseudo-tests, suggested 2)
- the confidence clamp
- the arm propensity

Each one is stored on every decision row, so it can be re-derived rather than inherited.

**Stake used.** Seven curiosity turns a day, each roughly 20 minutes on the shared pipeline. Phase
1 does not change how many are spent. It changes what each one buys.

**Trace that proves it.**
- A `curiosity_offer_decisions` row for the run, with its arm.
- A `curiosity_run_outcomes` row, with per-prior before and after values.
- Where a confidence moved, the matching `:PriorRevision` agreeing on the from and to values.

**Off switch.** One flag restores today's ordering. The snapshots and outcome rows are read-only
measurement, and harmless to leave running.

**Goodhart.** Orion can query its own revisions in-turn and read this repo (`kickoff_prompt.py:343,373`),
so "the LLM never sees yield" is a policy, not a wall. Named paths, and why phase 1 closes them:
- *Flip-flopping to look productive* earns nothing, because yield is net progress (P1c). An eval
  proves it: a synthetic oscillating prior must lose value.
- *Forming new priors at 0.55 to win attention* gains nothing over today's ordering, which already
  puts them first. In phase 2 it could buy whole runs, which is why phase 2 excludes untested priors
  from the trigger.
- The scores stay out of the kickoff prompt.

**Existing "stuck" rules this should replace, not run beside** (Sentience Striving Program README §7):
- `stale_after` (`HUB_CURIOSITY_STALE_PRIOR_TESTS=3`, live in `worldview.py:991-1037`)
- ask-Claude's `MIN_TIMES_TESTED=3` (`orion/autonomy/ask_claude_trigger.py:70`, dry-run)
- the supervisor's `is_circling` (`orion/curiosity/supervisor.py:827`, command-line only)

All three are count rules. Once yield passes its gate, it takes over `stale_after`'s ordering role
and ask-Claude's trigger (P4), in the same patch that switches them. `is_circling` is a reading,
and it stays.

#### Phase 2: when to spend a turn, only if phase 1 earns it

Run phase 2 only if phase 1's replay and live data show both of these:
- expected value varies across ticks, rather than sitting near `H(0.55) × pool_yield` most of the
  time; and
- expected value predicts realized nats.

Otherwise, record the null and stop.

1. **First, measure how long a dispatched curiosity run waits to start.** Durable demands wait a
   p90 of about 12.5 hours for a first grant (`orion/field/queue_contention.py:78`), and admission
   is first come, first served. If the start delay dominates, the lever for *when* is admission
   order (P6), not Hub's trigger.
2. **If the trigger still matters:**
   - Move `_read_worldview` ahead of the point where a slot is used. Mint the run id first (it seeds
     the rotation), read, gate, then call `_investigate` (`curiosity_investigation.py:1388-1432`).
   - The rule:

     ```text
     run if max over TESTED priors of expected_nats >= max(price, floor)
     price <- max(0, price + η × (runs_in_interval − cap × elapsed_sec / 86400))
     ```

   - `η` is in nats per run, fitted from replayed offers. After a run, the price returns to where it
     started in about 86,400 / cap seconds whatever `η` is, so `η` decides how much a rich offer can
     beat the pace.
   - The price counts every run, whether from the control arm or a forced run. It is persisted, and
     restored from the last decision row after a restart.
   - The floor is the 10th percentile of per-test net nats over the trailing 30 days. It is
     computed per tested prior, so it compares like with like.
   - Untested priors cannot buy a run. That closes the "form at 0.55" path.
3. **Control:** a *paced* clock (the cap spread over 24 hours), randomized per local day with
   propensity 0.5. Comparing against today's midnight cluster would measure the config, not the
   trigger.
4. **Say why.** Publish each hold as an `AttentionSchemaV1` row with `process="curiosity"`, using
   the existing contract and channel.
   - The `entry_id` is `curiosity-decision-<decision_id>`, distinct from the runner's per-run
     `curiosity-{run_id}` rows (`orion/curiosity/attention_schema.py:182`).
   - `attention_reason` is `held_off:below_price` or `held_off:below_floor`.
   - The success test is that holds are stored and queryable. This does not claim the chat cue shows
     them: the cue returns the 3 newest rows across five producers, and substrate writes every
     ~30 seconds (`orion/substrate/recent_attention_cue.py:18-26`).

### P2. Refusals and waits flow back upstream (needs its own spec)

**What changes.** Each spend point writes its verdict per candidate, with a reason, and attention
producers read it. The two kinds of refusal are handled differently:
- **Already learned** (the allocator's `below_information_floor`, or P1 yield ≈ 0): the loop loses
  standing.
- **Unaffordable right now** (`allowance_exhausted`): the loop keeps standing and waits.

The allocator already makes this distinction in its refusal reasons, and nobody reads it.

**The hard part.** The thinnest seam is to write `attention_loop_outcome` rows with
`actor="orion"`, `verdict="dismissed"`, and a `note` such as
`learned_out: allocator below_information_floor`. `dismissed` is terminal
(`orion/substrate/attention/verdicts.py:84`); the table has no reason field. The existing exclusion
path would then drop the loop for its time-to-live.

But the Hub refuses Juniper's close on these loops with a 409 for a reason
(`attention_loops_routes.py:75-82`): the pressure behind them is still live. An Orion close would
hide it in the same way. So P2's spec must reopen a learned-out loop *on new evidence*, not only
when the time-to-live runs out.

*Evidence* here means the loop's underlying prediction error moving outside its own recent spread.

**Reverie.** First measure whether narration volume still causes contention since the 09-24 pool
cutover (G7). If it does, give reverie narration a paced GPU-seconds budget, priced like P1. Do not
gate it on how many interactive requests are waiting at one instant: grant waits have been at most
5.7 seconds over 1,275 leases (`orion/field/queue_contention.py:80`), so a count sampled every few
minutes is almost always zero.

**Trace.** A workspace loop excluded by a `actor=orion` verdict, which re-enters when its evidence
moves.

### P3. Connect attention to spending

- **Adopt #2255's camera loop:** workspace winner → proposal → allocator → `verify_expectation` →
  an `actor=orion` resolution. It is the motor-side twin of P1, and touches different services.
  #2255's latest revision puts the Juniper ask (P4 here) ahead of it, and its acceptance check 7
  also claims the first `actor=orion` verdict. Sequence the two so they don't collide.
- **The rhythm learner already supplies graded camera predictions** (`vision_percept_expectation`).
  Once its own metric gate passes, a missed high-confidence prediction is a ready-made value source.
  The surprise of a graded prediction is −ln P(outcome).
- **Decide `inspect_attended_target`.** Give it an `expected_signal` it can actually move, or retire
  it.
- **Let the workspace winner into P1's offer** as a question about the attended node. It is valued by
  the same rule: the entropy of a related prior where one exists, otherwise a one-time cold-start
  value. This replaces the unreachable seed path (G1), rather than lowering a threshold to make that
  path fire.

### P4. Spend Juniper's attention as the scarcest thing Orion has

- **Use the ask ledger that exists.** Two contracts now cover "ask Juniper and keep her answer
  against the question":
  - #2255's proposal (09-20): Juniper as a peer on `HelpRequestV1 → PeerBriefV1`. Not merged.
  - `orion_ask` (merged 09-24): purpose-built, with a shared daily cap of 2, status that stays open
    until she answers, and answer routing by `source_kind`/`source_ref`.

  P4 uses `orion_ask`, with `source_kind="worldview_prior"` and `source_ref=<prior_id>`. Asks
  about beliefs and asks about camera individuals then compete for the same two daily questions,
  which is the point. Reconcile #2255 with `orion_ask` before either one grows a second ask path.
- **Hand off when self-study stops paying.** Once a prior's own yield (P1c) is near 0 and its
  entropy is still high, the value of asking someone else is that remaining entropy. This is the
  measured version of ask-Claude's "tested three times and still unsettled", and it replaces that
  count once yield passes its gate. Which peer to ask is decided by which budget clears the
  candidate. The candidates are Juniper's two daily asks, or the Cursor or Claude quota gates
  (after D2). No exchange rate is involved.
- **Score the answer.** An answered ask earns the net nats of that prior's next tested run
  (P1a). A dismissed or expired ask is recorded as 0 nats, not dropped.
- **Door A's gates are Juniper's call.** Door A skipping the schedule gates was her explicit decision
  on 2026-09-22 (`endogenous_outreach.py:452-455`). Whether both outreach doors should share one
  paced price over the daily cap of 4 is her question to answer (see Missing questions), not a fix.

### P5. Turn the control arms on

- **Dispatch:** hold-back above 0 (#2255 stage 4).
- **Curiosity:** P1's per-run arm and P2's per-day arm.
- **Outreach:** holding back a random fraction means sometimes deliberately not messaging Juniper.
  That is her call.
- **Propensity on every spend decision,** so later work can estimate what a different policy would
  have bought (inverse-propensity weighting).

Without control arms, "the value ordering learned more per turn" is a story, not a finding.

### P6. Prices for the rest of the stakes, after P1 proves the pattern

- **Durable-run admission.** Admission is where curiosity actually waits: p90 of about 12.5 hours to
  a first grant. Once P1's realized nats exist, order the agent-lane queue by expected nats per
  lane-second instead of first come, first served. (The backlog on 2026-09-22 was a lane-eligibility
  bug, since fixed: `services/orion-durable-runs/README.md:319-321`. That backlog is not evidence
  either way.)
- **Reading wallet.** Rank reading seeds by expected nats against open priors, once reading has a
  path to outcomes. Today reading never writes the graph (#2199), and #2255 missing question 2 asks
  for the adoption step.
- **Power.** No power budget exists, so don't invent one. When the power-budget design's Stage 1
  (`power_draw_log`) lands, watts become a native cost unit. The diffusion host's own-watts prior
  (`services/orion-diffusion-host/app/power_prior.py`) is the template for predicted against actual.
- **Claude quota.** After D2, turn the Claude-limit publisher on before any spend decision reads it.
- **Recall.** The prompt's token budget is a stake too. Nothing in `services/orion-recall/app` reads
  any attention output. A boost keyed on the workspace winner would be cheap, but with no label for
  whether a recalled item was used (G4), nothing could score it. Build the label first.

### P7. Attention that predicts its own spending (later)

Once P1 to P5 produce outcomes, the attention self-model can predict its next spend and that spend's
expected nats, and realized-versus-predicted becomes a calibration score Orion can report.
Graziano's attention-schema theory holds that a model of your own attention earns its keep by
*controlling* attention. Today's schema only narrates: `bottom_up_salience` was the reason on
19,774 of 19,774 rows as of 2026-09-02 (Sentience Striving Program README §16a). This comes only
once there is data to calibrate against.

### Prerequisite, running alongside: inputs attention can trust

Before any value estimate reads field signals:

- **Per-channel freshness.** Check `node_vector_updated_at`, not the age of the row (G8).
- **The harness-closure constant.** Stop feeding it to curiosity, or replace it with a measured
  surprise level. The code already names the missing field, `surprise_level_at_draft`
  (`worker.py:134`).
- **Vision's staleness channel.** Rename it, so it stops posing as prediction error.
- **Thresholds.** One per domain, not one shared across all of them.

P1 does not depend on these: it reads Orion's own belief records, not field signals. P2 and P3 do.

## Privacy and data touched

Required by proposal mode.

- **What P1 reads.** Orion's own worldview graph, read-only. Hub uses read-only queries against
  `orion_worldview` (`orion/curiosity/worldview.py` header), and P1 adds no writes to the graph.
- **What P1 stores.** Prior ids, confidences, counts and computed nats, in two new Hub-owned tables.
  It also stores the *ids* of study material offered to a run: approved memories and relations that
  come from Juniper's conversations. It stores ids only, never their content. They already live in
  the stores they are sampled from.
- **What P4 reads and stores.** Juniper's answers to asks already sit on `orion_ask` rows. P4 stores
  an ask id, a prior id and a computed number. It stores no reply text.
- **Retention.** Match the curiosity lifecycle tables' 90-day window
  (`services/orion-hub/scripts/curiosity_run_store.py` header).
- **Boundary.** Nothing leaves the host. Nothing is added to any prompt in phase 1.
- **Rollback.** Flags off; drop the two tables.
- **Dangerous failure mode.** A scheduler that learns to favour beliefs that swing. The net-progress
  yield and the oscillation eval exist to prevent it.

## Missing questions

1. **Is Orion's own stated confidence the right thing to score?** P1 scores what Orion writes. That
   is Orion's belief, not ground truth, and it is the right measure of what Orion learned. If you
   also want an outside reference, the rhythm learner's graded camera predictions (P3) are one.
2. **May curiosity skip turns?** This only comes up in phase 2. A value trigger could run fewer than
   7 on a day when nothing is worth it. That is the point, but you'll notice.
3. **Door A outreach.** On 2026-09-22 you chose "if Orion burned a run and has something to say, let
   them say it". Should both doors share one paced price over the daily cap of 4, or stay as they
   are?
4. **Is an outreach control arm OK?** It means sometimes deliberately not messaging you.
5. **Sequencing against #2255.** Its latest revision leads with the Juniper ask. P1 touches only
   curiosity's measurement and ordering, so it can run in parallel. Is that the order you want?
6. **`inspect_attended_target`:** give it an expected signal, or retire it?
7. **The midnight cluster.** It is a config choice (`MIN_COOLDOWN_SEC=1800` with the window off),
   and `.env_example` already names the fix. Is it still wanted?

## Proposed schema / API changes

P1 adds two tables and one registered metric, and changes no bus contract.

- **New table `curiosity_offer_decisions`**, one row per investigation run. It follows the
  `endogenous_outreach_decisions` pattern: a manual migration in `services/orion-sql-db/`, and a
  small Hub writer modelled on `services/orion-hub/scripts/endogenous_outreach_decisions.py`.
  Columns:
  - `decision_id`, `run_id`, `decided_at`
  - `arm` (`value_order` or `uncertainty_order`) and `propensity`
  - `offered` (jsonb): `prior_id`, `rank`, `confidence`, `times_tested`, `entropy_nats`, `yield`,
    `expected_nats`
  - `material_ids` (jsonb)
  - `prior_snapshot` (jsonb): every prior's `prior_id`, `confidence`, `times_tested` and `status`
    at kickoff
  - `constants` (jsonb): `W`, `k`, the clamp, and the propensity
- **New table `curiosity_run_outcomes`**, one row per completed investigation run. Columns:
  - `run_id`, `completed_at`
  - `realized_nats`: nullable, because an unreadable graph is not zero
  - `per_prior` (jsonb): id, before, after, the change in tested count, nats
  - counts: `n_tested`, `n_moved`, `n_formed`, `n_moved_untested`
  - agreement rates: `revision_agreement`, and `hop_reading_agreement` (nullable)
- **Models and registration.** Add pydantic models for both rows. Register `realized_nats` in the
  existing metric semantic layer. `orion/inner_state_registry.py` is keyed by pydantic classes and
  checked by `make check-inner-state-registry`. Re-lock `config/metrics/metric_definitions.lock.json`
  with `make check-definition-drift UPDATE=1` in the same patch.
- **Bus and channels.** P1 needs no new channel. Phase 2 publishes existing `AttentionSchemaV1`
  rows. `attention_reason` is free-text vocabulary that each process owns
  (`orion/schemas/attention_schema.py`), so no schema change is needed.
- **P2, later.** `attention_loop_outcome.actor` is already a free string
  (`orion/schemas/attention_salience.py:71`). A new *verdict* value would mean extending the
  `AttentionOutcomeVerdictV1` Literal (`:22`) and rebuilding sql-writer first.
- **Env, in `services/orion-hub`.** Add `HUB_CURIOSITY_VALUE_ORDER_ENABLED` (default `false` until
  the replay passes), `HUB_CURIOSITY_VALUE_ORDER_PROPENSITY` (0.5), `HUB_CURIOSITY_YIELD_WINDOW`
  (`W`) and `HUB_CURIOSITY_YIELD_PSEUDO_TESTS` (`k`). Put each in `.env_example`, sync it into the
  local `.env` with `scripts/sync_local_env_from_example.py`, and wire it through `settings.py`
  and compose in the same changeset.

## Files likely to touch

For P1 phase 1, and for D1 and D2.

**P1 phase 1**
- `orion/curiosity/value.py` (new): pure functions for entropy, Bernoulli KL, net-progress yield and
  pooling. No I/O.
- `orion/curiosity/worldview.py`: `select_priors` gains the value-order arm. Reuse `ALL_PRIORS_CYPHER`
  as it is.
- `services/orion-hub/scripts/curiosity_investigation.py`:
  - persist the snapshot and offer record where `_read_worldview` runs, at `:1469-1490`;
  - diff the snapshots in `_handle_run_state` on `completed`, for the investigation line, at
    `:3409-3425`.
- `services/orion-hub/scripts/curiosity_offer_decisions.py` (new): the writer.
- `services/orion-sql-db/`: two migrations.
- `services/orion-hub/app/settings.py`, `.env_example`, the local `.env`, `docker-compose.yml`,
  `README.md`.
- Pydantic models, the `inner_state_registry` entry, and the metric lock re-locked.
- Tests:
  - `value.py`: KL is exactly 0 when before equals after; yield returns `pool_yield` at n = 0; a
    null or out-of-range confidence gives `null`.
  - Oscillation eval: a prior flipping 0.3 ↔ 0.7 falls below a steadily moving one within 3 tests.
  - Snapshot diff: a test that bumps `times_tested` without moving scores 0, and is counted.
  - A replay script over historical priors and revisions (Acceptance check 1).

**D1**
- `orion/attention/field_attention/scoring.py` and `builder.py`
- `tests/test_attention_field_selectors.py`
- `orion/sentience_striving_program/instruments.yaml` and README §9b

**D2**
- `services/orion-curiosity-peer/app/worker.py` and its tests.

## Non-goals

- No new service, no attention registry, no taxonomy of attention kinds, no drives.
- No hand-typed exchange rate between budgets, and no comparing nats across models.
- Do not lower the allocator's floor. It is correct; the fix is better candidates (#2255).
- Do not merge or replace the three salience producers.
- Do not put realized nats or yield into the prompt.
- Do not change the GPU pool's route priorities.
- Do not bring back habituation as a penalty constant. It should emerge from yield (rule 4).
- Do not solve attention supply here; that is #2109.
- Do not touch the self-inquiry or self-sense lines.

## Acceptance checks

**P1 phase 1.** These are on the live rail, not only in tests.

1. **The replay runs before the flag flips.** It runs over existing priors and `:PriorRevision`
   rows, and it reports:
   - how many past tests can be reconstructed (moved tests from revisions; unmoved ones only as
     `times_tested` totals);
   - the distribution of per-run net nats, and the fraction that are exactly 0;
   - how coarse the confidences are (LLM-written numbers may snap to 0.05 steps).

   **Prediction, stated now:** the 09-14 base rate of 21 revisions across 94 runs means most runs
   will read exactly 0. The replay must turn that sparsity into a sample size: the number of runs
   per arm needed to detect a doubling of mean nats per run. That number goes in the PR.

   If the metric is degenerate, value ordering does not ship. Degenerate means essentially never
   non-zero, or unable to tell arms apart at any feasible sample size. The snapshots and outcome
   rows still ship, as measurement.
2. At least 95% of investigation runs have both snapshots and a `curiosity_offer_decisions` row with
   an arm and a propensity.
3. At least 90% of completed runs have a `curiosity_run_outcomes` row. Its `revision_agreement` is
   reported, not assumed.
4. **The oscillation eval passes.** A flip-flopping prior loses expected value, and a steadily moving
   one keeps it.
5. **Pre-registered comparison.** After the sample size from check 1, report mean net nats per run
   by arm, with the effect size and its interval. A null result is reported as null. If the result
   is null, value ordering is retired and the measurement stays.
6. **Phase 1 changes only order.** Runs per day and their timing match the 14 days before the change.

**P1 phase 2.**
- A hold row whose recorded price or floor exceeds the best expected value among tested priors, at
  that time.
- The price reaches exactly 0 on under-spent days.
- The comparison is against a *paced* clock, randomized per day.
- `stance_react` deferrals do not rise, split by curiosity line, against a baseline that starts after
  09-24.

**D1.** The steady-input and over-cap regression tests pass. On persisted frames, novelty that
alternates between 0 and a steady value drops to about 0.

**D2.** A default-path test shows the Claude fallback calling `rate_limit_events.observe`.

## Metric quality gate records

These are required by CLAUDE.md 0A before anything is wired in.

**`curiosity_realized_nats`: net belief change per investigation run.**

1. **Provenance.** Hub's own before and after snapshots of Orion's prior confidences and
   `times_tested` (`ALL_PRIORS_CYPHER`, `orion/curiosity/worldview.py:541`), read where
   `_investigate` reads the worldview and again at the durable `completed` state. `:PriorRevision`
   is written by Orion only when a confidence moves (`orion/curiosity/kickoff_prompt.py:672-690`),
   and serves as a cross-check.
2. **Independence.** New relative to field, biometric and prediction-error channels. It is *not*
   independent of three things that observe the same events:
   - `:PriorRevision`, the same move, when one is written;
   - `HopReadingV1.moved_the_claim`, the supervisor's reading, command-line only;
   - `orion/curiosity/atlas.py:366-368`, a raw difference over the same values.

   Those three are used to check agreement. This metric is their scored aggregate, not a new
   signal beside them.
3. **Theory anchor.** Bayesian surprise, KL(posterior ‖ prior) (Itti & Baldi 2009), as the size of
   a belief change. Yield uses net change over a window, which is learning progress (Oudeyer,
   Kaplan & Hafner 2007). The unit matches `bayesian_surprise_nats`, but magnitudes are **not**
   comparable across models (see rule 2).
4. **Live-data sanity.** `UNVERIFIED`; this is Acceptance check 1. Two known hazards:
   - The metric is sparse (21 revisions across 94 runs).
   - LLM-written confidences may be coarse.

   Rest state: exactly 0 for a tested prior that did not move, by construction. An unreadable run
   is `null`. Check both on real rows.
5. **Existing mechanism.** The raw difference in `atlas.py`, the narration in `attention_schema.py`,
   and three count rules for "stuck" (P1, "Existing 'stuck' rules"). Yield replaces the ordering
   role of `stale_after` and ask-Claude's count trigger once it passes this gate, in the same patch
   that switches them.
6. **Reversibility.** Two derived tables and a flag. Nothing enters a wire schema or a training
   default.

**`curiosity_price`: phase 2 only, the dual price per interval.**

1. **Provenance.** The curiosity daily counter, which already exists in Redis on local-date keys,
   and `HUB_CURIOSITY_INVESTIGATION_DAILY_CAP`.
2. **Independence.** It is controller state, not a sensor. It depends only on this budget's own
   spending.
3. **Theory anchor.** Adaptive pacing by dual descent (Balseiro & Gur 2019), and bandits with
   knapsacks (Badanidiyuru, Kleinberg & Slivkins 2013).
4. **Live-data sanity.** `UNVERIFIED`. Replay the last 30 days of start times. The price must reach 0
   on under-spent days. `η` is fitted from replayed offers, not assumed.
5. **Existing mechanism.** `paced_cooldown_sec` paces by the clock. Keep it as a ceiling; the price
   replaces it as the trigger.
6. **Reversibility.** One flag.

## Recommended next patch

Two pull requests. They are independent, and either can land first.

1. **`fix/attention-novelty-and-peer-claude-meter`: D1 and D2.** Small fixes, each with regression
   tests. D1 changes live attention behaviour, since it removes manufactured novelty. So it gets
   the usual review, plus a replay of persisted frames and the instrument update. It restores the
   documented intent; it does not need a design gate.
2. **`feat/curiosity-pays-for-what-it-learns`: P1 phase 1, in two commits.**
   - *Commit 1: measurement only.* `value.py`, the snapshots, both tables, the outcome diff, and the
     replay script (Acceptance check 1). Harmless to ship on its own.
   - *Commit 2: value ordering, behind `HUB_CURIOSITY_VALUE_ORDER_ENABLED=false`, with the per-run
     arm.* The flag flips only after the replay publishes its sample size and shows the metric is
     not degenerate.

**After those:**
1. P1 phase 2, only if it is earned.
2. P2's own spec: refusals flow upstream, with re-entry on new evidence.
3. P3 and P4, sequenced with #2255.
4. P5 alongside each of the above.
5. P6 and P7, once there is outcome data.

## Review findings addressed

An adversarial review ran against the code before commit. The material findings, and what changed:

- **P1 would have scored a record that is mostly never written.** Orion writes `:PriorRevision`
  only when a confidence moves (`kickoff_prompt.py:672-690`; 21 revisions across 94 runs).
  - *Fix:* Hub now measures from its own before and after snapshots (P1a). Revisions became a
    cross-check.
- **The per-tick control arm shared the daily cap,** so the clock arm would have taken most slots.
  - *Fix:* phase 1 changes only offer order, randomized per run, which shares no budget. Phase 2
    randomizes per day, against a *paced* clock.
- **KL would have rewarded flip-flopping** (a noisy-TV trap).
  - *Fix:* yield is net progress over a window (P1c), with an oscillation eval.
- **The GPU "interactive waiting" gate would almost never fire.** Grant waits peak at 5.7 s, and Hub
  decides dispatch, not execution. The claim that it excluded curiosity's own leases was unverified.
  - *Fix:* removed. The admission delay (p90 ≈ 12.5 h) now decides whether phase 2's trigger
    matters at all.
- **Wrong decision point and completion hook.**
  - *Fix:* phase 1 needs no gate before the slot is used. Phase 2 names the move. Completion hooks
    `_handle_run_state` (under durable admission, the live default).
- **Three existing count rules for "stuck" were missed.** The review also caught an overstated
  independence claim.
  - *Fix:* both named in P1 and in the gate record, with a replace-not-parallel plan.
- **Door A was reframed as a bug fix** when it was Juniper's decision.
  - *Fix:* moved to Missing questions, with the citation.
- **D1 missed its over-cap twin** and the Sentience Striving Program instrument it falsifies.
  - *Fix:* both added.
- **Proposal mode requires stating the privacy boundary.**
  - *Fix:* added a Privacy section.
- **Math wording:** the H(p) bound, the "share of uncertainty resolved" phrasing, comparing across
  models, and "worth more than GPU time" (which broke rule 2).
  - *Fix:* all corrected.

## Sources

**This repo: the attention history**
- `docs/architecture/attention-salience/README.md`
- `orion/sentience_striving_program/README.md` §7, §12, §13, §16; `orion/sentience_striving_program/instruments.yaml`
- `docs/notes/2026-07-30-chat-attention-ground-truth-gap-finding.md`
- `docs/notes/2026-07-24-attention-reason-branch-starvation-finding.md`

**This repo: stakes and incidents**
- `docs/superpowers/pr-reports/2026-09-05-stance-react-attention-salience-gateway-starvation-incident.md`
- `docs/superpowers/pr-reports/2026-09-25-world-pulse-wallet-refund-pr.md`
- `docs/superpowers/pr-reports/2026-08-30-motor-budget-enforcement-pr.md`
- `docs/superpowers/specs/2026-09-20-queue-contention-metric-gate.md`
- `docs/superpowers/plans/2026-09-14-curiosity-operations-analytics.md`

**Open PRs**
- #2255: prerequisites assessment, and "asks and waits"
- #2109: concept attention supply

**Theory**
- Lindley 1956, "On a measure of the information provided by an experiment"
- Itti & Baldi 2009, "Bayesian surprise attracts human attention"
- Feldman & Friston 2010, "Attention, uncertainty, and free-energy"
- Oudeyer, Kaplan & Hafner 2007, "Intrinsic motivation systems for autonomous mental development"
- Badanidiyuru, Kleinberg & Slivkins 2013, "Bandits with knapsacks"
- Balseiro & Gur 2019, "Learning in repeated auctions with budgets"
- Burda et al. 2018, "Large-scale study of curiosity-driven learning"
- Graziano 2013, *Consciousness and the Social Brain*
