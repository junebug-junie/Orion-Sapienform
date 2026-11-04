# Curiosity replica: a small mind in curiosity's arm (Orion Prime step 2C')

**Design specification · 2026-11-04 · Proposal mode, design only, no code**
**Approved to write:** Juniper, 2026-11-04. She reviews this before anything is built.
**Builds on:** Orion Prime (`2026-10-29-orion-prime-design.md`, section 8c and rollout step 2C'), the curiosity offer
experiment (P1 of `2026-09-25-attention-with-stakes-design.md`), the body economy
(`2026-10-23-organism-body-economy-design.md`), and the curiosity focus slot (`orion/curiosity/focus.py`).

All live numbers below were read on 2026-11-04 (UTC) from Postgres `conjourney` (read-only), the running
`orion-mind` and `orion-athena-hub` containers, and durable-runs `localhost:8124`. Anything not checked live is
marked UNVERIFIED.

## Arsonist summary

Before each regular curiosity run, the code puts Orion's open beliefs (priors) in an order. Orion mostly tests the
first one it sees. Today a fixed rule makes that order: most uncertain first. This design adds a small model-based
planner, the **curiosity replica**. It reads curiosity's world and writes the order for some runs. Code checks the
plan and falls back to the rule if the plan is bad or missing. The planner has to beat both existing rules in a fair
random test, or it is removed.

Four things found while writing this change the plan:

1. **The experiment the replica must win has not started.** All 234 recorded runs since 2026-09-26 used the
   uncertainty rule. The second rule (the "value arm") is built but switched off
   (`HUB_CURIOSITY_VALUE_ORDER_ENABLED=false`, live and in `.env_example`). Its own gate said "stay off until the
   replay shows the measurement is usable". The replay now says **USABLE** (run today, see below). Nobody turned it
   on. **Rec: turn the value arm on now, as its own one-line patch, before any replica code.**
2. **Order matters, but only when Orion tests something.** 92 of 202 completed runs (46%) tested no prior at all.
   When Orion does test, it picks the first prior shown a third of the time (46 of 141 tests). So the replica can
   only help on about half the runs. This limits how big a win it can show.
3. **The test is slow.** About 5.3 runs a day finish. With three arms, each arm gets under 2 runs a day. To detect
   a replica that **doubles** belief change takes about 2.5 to 6 months. A smaller effect cannot be detected at
   this cadence. The replica must roughly double the result to survive.
4. **Prime has never sent a regulate signal.** Regulation has been on since 2026-10-30
   (`PRIME_REGULATE_ENABLED=true`), but `grammar_acts` holds 0 `regulate` acts from `actor=prime`. Prime has sent
   10 curiosity focus questions. So the "push back on Prime" test (E4) can start on focus questions only.

Decisions this design makes (each explained below):

- The replica runs **in orion-mind**, as the second instance of a loop library extracted from Prime.
- Hub **asks for the plan at kickoff**, waits at most 90 seconds, then falls back. Only runs on the replica arm ask.
- The replica's reasons are **never shown to Orion**. The kickoff shows only the order. This keeps the three arms
  identical except for the order, and it keeps the replica from speaking as Orion.
- Fallbacks **count against the replica** (intention-to-treat). A replica that is often unfunded or slow loses.
- It bids as a new family **`curiosity.replica`, tier 2**, on the chat lane, like Prime.
- It can **challenge or refuse** Prime's focus questions and regulate signals (Juniper, 11-04). A refusal always
  carries a reason. A random share of challenges and of refusals is held back, so we can tell whether they change
  Prime and whether refusing was right.

## Current architecture

- **Service:** the curiosity organ lives in Hub (`services/orion-hub/scripts/curiosity_investigation.py`). Prime
  lives in orion-mind (`services/orion-mind/app/prime_loop.py`, pure parts in `orion/prime/`).
- **Entry point (curiosity):** `CuriosityInvestigationLoop.tick` passes every gate (cooldown, daily cap 7, window,
  rest drive), then holds `_run_lock` for the whole turn and calls `_investigate`
  (`curiosity_investigation.py:2443`).
- **Order of work inside `_investigate` today (the timing that matters):**
  1. draw the offer arm from the run id (`offer_arm`, `orion/curiosity/value.py:494`);
  2. load the learning-yield model from scored history (`_load_yield_model`);
  3. read the worldview and sort the priors (`read_snapshot` -> `select_priors`, `orion/curiosity/worldview.py:1005`);
  4. write the offer row (`_record_offer` -> `curiosity_offer_decisions`);
  5. take the pending focus, dream hypotheses, carry-forward, corrections, lived questions, today fact;
  6. build the kickoff prompt (`build_kickoff_prompt`) and dispatch the durable run.
  The prompt is built at dispatch, before the durable run bids. **The plan must exist between steps 1 and 3.**
- **How the order is made:** `select_priors` takes up to `HUB_CURIOSITY_PRIOR_SAMPLE=8` live priors. It splits off
  stale priors (tested `HUB_CURIOSITY_STALE_PRIOR_TESTS=3` times) into their own bucket. It sorts the rest by
  uncertainty (`|p - 0.5|`, then times tested, then a per-run rotation), or by expected nats on the value arm. The
  top 6 are "ordered". The last 2 are **coverage slots**, drawn at random from the rest so confident priors still get
  re-checked (`HUB_CURIOSITY_PRIOR_COVERAGE_SLOTS=2`).
- **How a run is scored:** Hub snapshots every prior when the turn starts and again when it ends. Each prior this
  run tested (stamped with its `last_run_id`) scores KL(after || before) in nats. The sum is `realized_nats` in
  `curiosity_run_outcomes`. Zero means "tested, nothing moved" or "tested nothing". A failed turn is flagged
  `turn_ok=false`. (`orion/curiosity/value.py:diff_snapshots`.)
- **How the value arm works:** expected nats = entropy(p) x learning yield. Yield is net, straight-line belief change
  per unit of uncertainty offered, over the prior's last 3 tests, shrunk toward the pool average
  (`build_yield_model`). The live pool yield is 0.150 (latest offer row), so the value order would now differ from
  the uncertainty order.
- **Arm assignment:** `offer_arm(run_id, enabled, propensity)` hashes the run id. Two arms only:
  `value_order` and `uncertainty_order`. Off means every run is `uncertainty_order` with propensity 0.
- **The prompt sentence about the order** (`orion/curiosity/kickoff_prompt.py:214-219`): "first come the ones where
  the code estimates a test could change your mind the most". This is true for both rule arms. It would be false
  for a replica arm.
- **Prime today (step 2A + step 3, live since 10-30):** a 15 s tick in orion-mind. It reads body state and the
  temporal-self frame, decides whether to think, bids `prime` (tier 1, lane chat), and calls route `prime`. 297
  thought acts so far, about 37 a day; median thought act 9.3 s, p90 61 s (last 3 days). Its moves: one
  curiosity focus question (`ask`, 10 sent) or one regulate signal (`regulate`, 0 sent). Holdback share 0.2.
- **Body economy:** `config/physiology/families.yaml`. One family per act-ledger actor. Tier 1 has a reserved lane
  share; tier 2 is the open market. Over the last 3 days `prime` bids were funded 165 times and deferred 267 times
  (`capacity_reserved`, lane chat). Prime re-bids every tick, so a deferral only delays it.
- **Bus:** Prime queues focus through `orion:introspect:queue_curiosity_focus:request` (Hub replies on
  `orion:introspect:result:<id>`). Prime's regulate signal is a Redis hash `orion:prime:regulate:<family>`
  (`orion/physiology/prime_charge.py`); the market applies it.
- **Schema registry:** spend log tables in `services/orion-sql-db/manual_migration_curiosity_spend_v1.sql`. Focus
  table in `manual_migration_curiosity_focus_request_v1.sql`. Prime state models in `orion/prime/state.py`.
- **Tests:** `services/orion-hub/tests/` (curiosity), `services/orion-mind/tests/`, `orion/prime/tests/`,
  `orion/curiosity/tests/` (directories exist; individual test files not read for this design).
- **Evals:** `scripts/analysis/replay_curiosity_realized_nats.py` (two arms only).
- **Current gap:** curiosity's offer order is a fixed rule. No mind in the arm can read what curiosity has been
  finding and decide what to look at next. No organ can answer Prime.

### Live experiment numbers (2026-11-04)

| Arm | Runs offered | Completed (`turn_ok`) | Moved at least one prior | Mean nats per completed run |
|---|---|---|---|---|
| `uncertainty_order` | 234 | 202 | 94 (46.5%) | 0.064 (sd 0.210, median 0, p90 0.119, max 1.88) |
| `value_order` | 0 | 0 | — | — (switch off) |
| replica | — | — | — | — (does not exist) |

- 31 of 233 outcome rows are failed turns (13%). 0 runs are unknown.
- Tests per completed run: 0 tested in 92 runs, 1 in 81, 2 in 27, 3 in 2. 114 of 141 tests moved the belief.
- Where tested priors sat in the offer: rank 1: 46, rank 2: 7, rank 3: 11, rank 4: 12, ranks 5-8: 18, stale bucket:
  24, not offered at all: 23. Rank 1 is the biggest single lever the order has.
- Completed runs over the last 30 days: 158, which is **5.3 a day** (the cap is 7; heat lockdowns and failures take
  the rest).
- The replay gate (`replay_curiosity_realized_nats.py --pg`, run today): "USABLE: ~171 runs per arm (~49 days at
  7/day)" for the existing two-arm test.

**How long a three-arm test takes.** The replica must beat both rule arms. That is two one-sided tests, both at
p < 0.05 (an intersection-union test, so no multiplicity correction is needed). At 5.3 completed runs a day split
three ways, each arm gets about 1.76 runs a day.

| Effect the replica must show | Method | Runs per arm | Days |
|---|---|---|---|
| doubles mean nats | normal approximation, one-sided, 80% power | 135 | ~77 |
| doubles mean nats | the replay's own formula (two-sided) | 171 | ~97 |
| doubles mean nats | bootstrap from the 202 real runs, permutation test, must win both | 300 gives 68% power | ~170 |
| +15 points in "moved at least one prior", same size when it moves | bootstrap, permutation on mean nats | 300 gives 15% power | not reachable |

The bootstrap is the honest row: the data is mostly zeros with a long tail, so the normal approximation is too
hopeful. Simulation script: run on the 202 real outcomes, 200 resamples per cell, 400 permutations per test.

## Decisions (Juniper, 2026-11-04)

1. **Turn the value arm on now (patch 0).** The two rule arms build up their own comparison while the replica is
   built, so the three-arm test starts with both rule arms live.
2. **A 3 to 6 month test is acceptable,** with the fixed stop in section 6.
3. **The replica may challenge and refuse Prime** (section 9).
4. **`curiosity.replica` is tier 2.**
5. **It plans only on its own arm** for now. No shadow plans.
6. **What Juniper has been talking about comes from the chat compactor's daily digest**, plus any pending focus she
   asked for (section 2). No raw chat text.

## Proposed design

### 1. What the replica does, in one run

1. Hub draws the arm for the run (now three arms). On a rule arm, nothing changes.
2. On the replica arm, Hub builds a **candidate list** from the worldview it already read and sends a plan request.
3. orion-mind's replica instance wakes on the request. It bids `curiosity.replica` on lane chat. If funded, it makes
   **one short model call** and answers with a plan: 3 to 5 prior ids in order, one sentence of why for each, and an
   optional stance on Prime's pending focus.
4. Hub checks the plan with deterministic code. A good plan fills the top of the ordered list. A bad, late, or
   missing plan means the rule order is used, and the reason is recorded.
5. The kickoff prompt shows the order. It does not show the reasons.
6. When the run ends, the normal spend log scores it. The replica's tick reads that score and closes its plan act.
   The next frame shows the replica its own track record.

### 2. What the replica reads (its frame)

Each field carries its age. A missing field is shown as a gap, never as calm (same rule as Prime's frame).

- **Candidates** (from Hub's request, so the replica sees exactly what can be offered): fresh live priors only, never
  the stale bucket. At most 60: the 30 most uncertain, the 20 with the highest expected nats under the current yield
  model, every prior tested in the last 7 days, and 10 drawn by the run-id rotation from the rest (de-duplicated, cut
  to 60). Each line: id, claim (120 characters), confidence, times tested, age of last test, last test's move. About
  2,400 tokens. Live pool today: 246 live priors of 277, so the full pool does not fit a short frame.
- **What recent runs found and failed:** the last 6 investigate runs from Hub's `/curiosity/api/runs` (the same read
  Prime uses): status, finding text (240 characters), focus taken, `realized_nats`, which priors it tested.
- **Stale priors:** count and ids only, so the replica knows what the stale bucket will show anyway.
- **Prime's current concerns and signals:** Prime's open concerns, the pending focus if Prime queued it, and any
  live regulate signal on `durable.curiosity_investigate` (Redis `orion:prime:regulate:durable.curiosity_investigate`).
  The concern text is **not** on the act rows today (checked live: `predict` acts with `actor=prime` carry only
  `prime_concern:<id>` and the outcome note). So Prime publishes a read-only view: on every save it writes
  `PrimeConcernsViewV1` (id, text, age, arc id) to Redis `orion:prime:concerns`. That is Prime's first published
  call. The replica reads that key, never Prime's private state.
- **What Juniper has been talking about** (Juniper, 11-04: use the chat compactor): the newest chat-history
  compactor digest (`orion/cognition/chat_history_compactor`, one a day around 13:00 UTC: the card summary and the
  journal title), shown with its age, plus any pending focus with origin `juniper` for today. No raw chat text.
  The digest covers the previous day, so it can be up to about 36 hours old; the pending focus covers today.
- **Its own track record:** its last 10 plans. For each pick: was it tested, did it move, by how many nats. Plus its
  pick uptake (share of its picks Orion tested) and its fallback count. It does **not** see the arm comparison or any
  p-value. Showing it the running score invites chasing noise.

### 3. What the replica writes (its answer)

One JSON object, parsed into `CuriosityOfferPlanAnswerV1`:

```json
{"reading": "one or two sentences: how it reads curiosity's situation now",
 "picks": [{"prior_id": "...", "why": "one sentence"}],
 "focus_stance": null | {"stance": "support" | "challenge" | "refuse", "reason": "one sentence"},
 "signal_stance": null | {"signal_id": "...", "stance": "support" | "challenge" | "refuse", "reason": "one sentence"},
 "abstain": false}
```

"abstain: true" is a valid answer ("nothing here is worth reordering"). It counts as a fallback for the arm.

### 4. What code checks (validation), and the fallback

Pure function `validate_plan(answer, candidates, view)` in `orion/curiosity/replica/plan.py`. The whole plan is
rejected on any failure. A partial plan is never repaired, because a repaired plan is a blend of two arms.

- 3 to 5 picks; no duplicate ids;
- every id is in the candidate list and still live and fresh in the worldview Hub just read;
- every `why` is 1 to 200 characters;
- `signal_stance.signal_id` names a live signal, if present.

On rejection, timeout (90 s), unfunded bid, model failure, or abstain: Hub uses the uncertainty order, writes the
reason on the plan row, and logs `curiosity_replica_fallback run=<id> reason=<reason>`.

**How a good plan becomes the offer** (`select_priors` gains a `head_ids` argument):

- ordered slots 1..k = the replica's picks, in its order (k = 3 to 5);
- ordered slots k+1..6 = the uncertainty order, skipping the picks;
- the 2 coverage slots = the same random draw as on the rule arms, skipping the picks;
- the stale bucket = unchanged.

So every arm shows the same number of priors in the same layout. Only the order of the top differs.

### 5. Keeping the three arms fair

- **Same prompt sentence on every arm.** Replace the current sentence with one that is true of all three arms, for
  example: "These are ORDERED, and the order is not neutral: it is a presentation choice made before this run, not a
  recommendation. Which one, if any, is worth your time is your call." The self-inquiry line shares this section and
  gets the same sentence. Ship the new sentence on the same day the three-arm test starts. Compare only runs
  randomized at the same time (the weekly mean moved from 0.027 to 0.273 nats over the log, so old runs are not a
  fair control).
- **Reasons never reach Orion.** They stay on the plan row. The plan table gets no grant to `orion_readonly`, the
  same rule as the spend log, so an in-turn sandbox cannot read them either.
- **Intention-to-treat.** The arm is drawn before the plan exists. Every replica-arm run counts as replica, fallback
  or not. A secondary "as treated" number is reported, never used for the decision.
- **Propensities recorded.** `offer_arm` becomes a three-way hash draw, default 1/3 each. All three propensities go
  on the offer row.

### 6. Scoring and the retirement rule (pre-registered)

- **Primary outcome:** `realized_nats` per completed run (`turn_ok=true`), the same number P1 already records.
- **Test:** one-sided permutation test of the replica mean against each rule arm, 10,000 permutations. The replica
  wins only if **both** p-values are below 0.05.
- **Stop:** when every arm has 250 completed runs, or 180 days after the start, whichever comes first.
- **Interim harm look, every 30 days:** if the replica arm's failed-turn rate or "tested nothing" rate is worse than
  the pooled rule arms at p < 0.01, pause the arm and report it.
- **If it does not win: retire it fully.** Set the replica propensity to 0, stop the instance, remove the
  `curiosity.replica` family and its code in the next patch. Keep the plan rows as history. No "left running, just
  not used" state (kill means kill).
- **Report the result whatever it is.** A loss is a real finding about whether a small model can steer curiosity.

### 7. Where the replica runs, and why

**In orion-mind, as instance #2 of the loop library.** Three options were considered:

| Option | For | Against |
|---|---|---|
| A. Hub runs the replica in its own process | it lives next to its organ, the literal octopus arm | Hub already carries the curiosity loop (5,871 lines in one file); it has no bid client for model calls, no act gate, no Juniper-first preemption, so all of Prime's machinery would be copied a second time |
| B. orion-mind pre-computes a plan after each run and Hub picks it up | Hub never waits | the arm is only known at kickoff, so it plans for every run (3x the calls) or needs a second arm draw; the plan can be stale against the pool Hub reads |
| **C. orion-mind answers a plan request at kickoff (recommended)** | reuses Prime's bid, gate, lane and preemption code; plans only on its own arm (~1.8 calls a day); validated against the exact pool Hub is about to show | Hub waits up to 90 s, but Hub already holds `_run_lock` for the whole 20-minute turn, so the wait blocks nothing else |

Timing: Hub sends the request right after the arm draw and the worldview read (step 3 of `_investigate`), and before
`_record_offer` (step 4). The offer row then records the order that was really shown. 90 s covers Prime's p90 thought
act (61 s, which includes seat wait). The replica re-bids every tick (15 s) until the deadline, so one deferral does
not kill the plan.

Being in the same process as Prime is not a side channel. The two instances keep separate Redis state keys, and each
reads the other only through act rows and published calls (Prime design 8c, rule 5). A unit test checks that the
replica instance has no reference to Prime's state object.

### 8. Body economy: family, tier, lane

New family in `config/physiology/families.yaml`:

```yaml
  curiosity.replica:
    # Orion Prime 2C' (docs/superpowers/specs/2026-11-04-curiosity-replica-design.md): one short plan per
    # replica-arm investigate run, plus at most 3 answers to Prime a day. Tier 2: when it is unfunded, the rule
    # order runs and curiosity loses nothing. 90 s: the kickoff will not wait longer.
    tier: 2
    deferrable_window_s: 90
    lanes: [chat]
    placement: one_of
```

- **Why tier 2.** Tier 1 means "critical upkeep" with a reserved lane share. The replica is not upkeep: when it
  cannot think, the rule runs and nothing breaks. The organ it serves (`durable.curiosity_investigate`) is itself
  tier 2. A tier-1 brain for a tier-2 organ would let the planner reserve lane time its own organ cannot.
- **The cost of tier 2.** Prime is tier 1 and is still deferred on lane chat 62% of the time. A tier-2 replica will
  be deferred at least as often, and each deferral past 90 s is a fallback that counts against it. Rec: measure the
  fallback rate in the first 14 days. If more than 30% of replica-arm runs fall back for `unfunded`, bring it back to
  Juniper with the number (raise to tier 1, or move to another lane) rather than letting the arm lose by starvation.
- **Why lane chat.** It is the warm lane Prime already uses (route `prime`), so no new allocation and no cold start.
  Juniper's turn still goes first (pool priority, `JUNIPER_TURNS`). Cost: about 1.8 plans a day at about 10 s each,
  small next to Prime's ~37 thoughts a day. E9 (Juniper first) is checked with the same latency measure as Prime.
- **No receptors in v1.** Prime regulating a planner that thinks under 2 times a day changes nothing measurable. A
  family gets a receptor only when it has a handler, a test, and binding bids (families.yaml rule).

### 9. Second receiver of Prime's signals: challenge and refuse

Prime has two moves today. The replica may answer each one with **support**, **challenge** or **refuse**, always
with a one-sentence reason. Every answer is an act (`actor=curiosity.replica`, verb `ask`, `parent_act_id` = Prime's
act) that closes at once with the reason as its note.

- **Prime's focus question.**
  - *Challenge:* the focus is still shown to the run. Prime hears the reason.
  - *Refuse:* the focus is not shown to the run. Hub marks the focus row `refused_by_organ` with the reason, and
    Prime's question act closes 0 with note `refused by curiosity.replica: <reason>`. It still counts in Prime's
    treated arm (intention to treat), so Prime's own 2A comparison stays honest: it then measures Prime's effect
    including the organ's right to say no.
  - Never for a focus with origin `juniper`: refusal applies to Prime's acts only.
- **Prime's regulate signal on curiosity.**
  - *Challenge:* the charge stays. Prime hears the reason.
  - *Refuse:* the market stops applying that signal's charge from the next bid (the replica writes the refusal next
    to the signal in `orion:prime:regulate:durable.curiosity_investigate`; `orion/physiology/choice.py` reads it). The
    decision's `why` says "Prime's slow charge refused by curiosity.replica: <reason>". Prime's regulate act grades on
    what actually happened.
- **Holdback for refusals.** 20% of refusals are recorded but not applied (arm `control`): the focus is still shown,
  the charge still applies. Comparing refused-and-applied against refused-but-held-back runs tells us whether the
  replica's refusals were right (did the run do better without Prime's push?). Same 20% holdback for challenges
  (whether they reach Prime), for the E4 test below.
- **Guard against an organ that always says no.** The replica's frame shows its own refusal rate over the last 20
  answers. The readout reports it; a rate above 50% for 14 days goes back to Juniper with the numbers. No hard cap:
  the holdback tells us whether the refusals were worth it.
- **Between runs.** A new Prime focus or regulate signal on curiosity is a reason for the replica to think once, even
  with no plan request: one answer per Prime act, at most 3 a day, 30 minutes refractory.
- **Prime hears it.** Prime's frame gains an "ORGAN REPLIES" section: challenges and refusals to its acts in the last
  48 hours, with the organ's reason (`orion/prime/thought.py`, new `_organ_reply_lines`).
- **The E4 test.** E4 counts when Prime's next move about the same concern differs after a delivered challenge or
  refusal more often than after a held-back one. Prime sent 10 focus questions in 5 days, so expect about 2
  opportunities a day at most, and only on focus questions until Prime first regulates curiosity. E4 on regulate is
  UNVERIFIED until Prime sends its first `regulate` act.

### 10. Groundwork: one loop library, two instances

Today `prime_loop.py` (998 lines) mixes two kinds of code:

- **Generic loop machinery:** start, bus connect with backoff, stop; load and save state in Redis; the tick timer;
  the act gate (self-monitor) check; build a `WantV1`, bid, release; cost ceiling; route choice and starved route;
  the no-model backoff (`note_no_model`); opening the thought act only once the model is reached; Juniper-turn
  preemption; publish `act_opened` / `act_closed`.
- **Prime-specific code:** its snapshot and `observe`; concerns and their bets; the frame; the system prompt; the
  answer schema; the two moves (nominate, regulate) and how they settle.

**Proposed split:**

- `orion/mindloop/` (new package, pure plus an async harness with all I/O injected):
  - `instance.py`: a `LoopInstance` protocol with exactly the four things Prime design 8c names, plus identity:
    `name`, `family`, `actor`, `state_key`; `read_frame_inputs()` (its frame); `receptors()`; `calls()` (requests it
    answers, each with a typed input and output); `effectors()` (move kind -> handler); `wants_to_think(state, now)`;
    `build_frame()`; `parse_answer()`; `settle(now)`.
  - `runtime.py`: `MindLoopRuntime(instance, bus, llm, bid, release, gate, clock, turns)`. The tick, bid, think,
    and close logic moved from `PrimeRuntime` without change.
  - `backoff.py`: `classify_try`, `no_model_backoff`, `note_no_model` moved from `orion/prime/state.py`.
- `orion/prime/instance.py`: Prime as instance #1. Everything Prime-specific from `prime_loop.py` moves here.
- `orion/curiosity/replica/`: `frame.py`, `plan.py` (answer schema, validation, offer assembly), `track_record.py`,
  `instance.py`. Pure, no I/O.
- `services/orion-mind/app/mind_loops.py`: starts the instances that are enabled, each with its own bus connection
  and its own task. `prime_loop.py` shrinks to Prime's I/O adapters.

**Rule for the extraction patch:** zero behavior change. Every existing Prime test passes unchanged, and a replay of
one recorded tick sequence produces the same act rows. Existing-mechanism check: `orion/mind/` already exists (mind
run schemas, unrelated), so the library is named `orion/mindloop/` to avoid a collision.

## Proposed schema / API changes

- **Added: `orion/schemas/curiosity_replica.py`**
  - `CuriosityOfferCandidateV1`: `prior_id`, `claim` (≤120), `confidence`, `times_tested`, `last_tested_age_h`,
    `last_move`.
  - `CuriosityOfferPlanRequestV1`: `run_id`, `correlation_id`, `deadline_at`, `candidates` (≤60), `stale_ids`,
    `pending_focus` (`focus_id`, `origin`, `question`) or null.
  - `CuriosityOfferPlanAnswerV1`: the model's JSON (section 3).
  - `CuriosityOfferPlanResultV1`: `run_id`, `plan_id`, `status` (`ok` | `abstain` | `unfunded` | `model_failed` |
    `unparseable`), `picks`, `focus_stance`, `signal_stance`, `act_id`, `latency_ms`. Hub adds `invalid:<reason>`
    and `timeout` itself.
  - `PrimeConcernsViewV1` (in `orion/prime/state.py`): `as_of`, `concerns` [{`concern_id`, `text`, `age_h`,
    `arc_id`}]. Written by Prime to Redis `orion:prime:concerns`, read by the replica.
  - Registered in `orion/schemas/registry.py`.
- **Added bus channels** (`orion/bus/channels.yaml`): `orion:curiosity:replica:plan:request` (producer orion-hub,
  consumer orion-mind) and `orion:curiosity:replica:plan:result:<correlation_id>` (producer orion-mind, consumer
  orion-hub). Same request/reply shape as the introspect focus RPC.
- **Added SQL** (`services/orion-sql-db/manual_migration_curiosity_replica_plan_v1.sql`, header
  `-- ORION-MIGRATION-REQUIRED-BY: orion-hub`, additive DDL only so the post-pull applier accepts it):
  - table `curiosity_offer_plans`: `run_id` PK, `plan_id`, `requested_at`, `answered_at`, `status`,
    `fallback_reason`, `picks` jsonb, `focus_stance` jsonb, `signal_stance` jsonb, `act_id`, `latency_ms`,
    `frame_tokens`. No grant to `orion_readonly`.
  - `ALTER TABLE curiosity_offer_decisions ADD COLUMN IF NOT EXISTS arm_propensities jsonb`.
- **Behavior changed:** `offer_arm` returns one of three arms with a propensity map. New constant
  `ARM_REPLICA_ORDER = "replica_order"`. `value_arm_propensity` stays written for old readers.
  `select_priors` gains `head_ids` (default empty, so every current caller is unchanged).
- **Act ledger:** new actor `curiosity.replica`. Acts: `infer` (a plan, or a thought answering Prime),
  `ask` (a support, challenge or refusal, parented to Prime's act). No new verbs, no new `ActSignal`.
- **Family:** `curiosity.replica` (section 8).
- **Compatibility:** with the replica flag off, `offer_arm` draws only the two rule arms, exactly as today.

## Env / config changes (proposed)

- Hub (`services/orion-hub/.env_example`, `settings.py`, `docker-compose.yml`):
  - `HUB_CURIOSITY_VALUE_ORDER_ENABLED=true` (patch 0, existing key, flipped);
  - `HUB_CURIOSITY_REPLICA_ARM_ENABLED=true`, `HUB_CURIOSITY_REPLICA_ARM_PROPENSITY=0.3333`,
    `HUB_CURIOSITY_REPLICA_PLAN_DEADLINE_SEC=90`.
  - With both on, the value propensity becomes 0.3333 too, so the three arms are equal.
- orion-mind (`services/orion-mind/.env_example`, `settings.py`):
  - `CURIOSITY_REPLICA_ENABLED=true`, `CURIOSITY_REPLICA_LLM_ROUTE=prime`, `CURIOSITY_REPLICA_MAX_TOKENS=500`,
    `CURIOSITY_REPLICA_CHALLENGE_HOLDBACK_FRACTION=0.2`, `CURIOSITY_REPLICA_CHALLENGES_PER_DAY=3`.
- Flags ship ON (standing rule). Local `.env` synced with `python scripts/sync_local_env_from_example.py` in each
  patch that changes a template.

## Files likely to touch

- `orion/mindloop/__init__.py`, `instance.py`, `runtime.py`, `backoff.py`: new loop library.
- `orion/prime/instance.py`, `orion/prime/state.py`, `orion/prime/thought.py`: Prime as instance #1; organ replies in
  its frame.
- `services/orion-mind/app/prime_loop.py`, `services/orion-mind/app/mind_loops.py`, `services/orion-mind/app/main.py`,
  `services/orion-mind/app/settings.py`, `services/orion-mind/.env_example`, `services/orion-mind/README.md`.
- `orion/curiosity/replica/` (new): frame, plan, track record, instance.
- `orion/curiosity/value.py`: three-way `offer_arm`.
- `orion/curiosity/worldview.py`: `select_priors(head_ids=...)`.
- `orion/curiosity/kickoff_prompt.py`, `orion/curiosity/self_inquiry_prompt.py`: the neutral order sentence.
- `services/orion-hub/scripts/curiosity_investigation.py`: plan request between worldview read and offer record.
- `services/orion-hub/scripts/curiosity_offer_decisions.py`: write `arm_propensities` and the plan row.
- `services/orion-hub/app/settings.py`, `services/orion-hub/.env_example`, `services/orion-hub/docker-compose.yml`.
- `orion/schemas/curiosity_replica.py`, `orion/schemas/registry.py`, `orion/bus/channels.yaml`.
- `orion/prime/state.py` / `prime_loop.py`: publish `PrimeConcernsViewV1` to `orion:prime:concerns`.
- `config/physiology/families.yaml`.
- `services/orion-sql-db/manual_migration_curiosity_replica_plan_v1.sql`.
- `scripts/analysis/replay_curiosity_realized_nats.py`: three arms, permutation tests, the stop rule.
- Tests: `orion/curiosity/tests/test_replica_plan.py`, `orion/mindloop/tests/`, existing Prime tests unchanged,
  Hub kickoff tests for the fallback paths.

## Non-goals

- The replica never writes to Orion's graph, never queues a focus, never sends outreach, never writes chat text.
- The replica never speaks as Orion. Its text reaches only its own plan row and Prime's frame, labeled as
  curiosity's planner.
- No change to when curiosity runs, how often, or its cap. Only the order of the top of the offer.
- No change to the stale bucket, the coverage slots, or the self-inquiry and self-sense lines.
- The replica cannot refuse Juniper's own focus (origin `juniper`): refusal applies to Prime's acts only.
- No new verbs, no new `ActSignal`, no new scheduler or queue.
- No second replica (dream) until this one has a verdict.

## Acceptance checks

For the design:
- Every new contract above has a producer, a consumer and a test in the patch that adds it.
- The extraction patch changes no Prime behavior: existing tests green, recorded tick replay gives identical acts.

For the build (the trace that proves it worked):
- A replica-arm run has a `curiosity_offer_decisions` row with arm `replica_order` and its propensities, a
  `curiosity_offer_plans` row with status `ok`, and an offered list whose top k ids equal the plan's picks.
- The same run has an `act_opened`/`act_closed` pair with `actor=curiosity.replica`, verb `infer`, and a
  `want_ledger` row for family `curiosity.replica`.
- A forced fallback (stop orion-mind) gives a plan row with `fallback_reason=timeout` and a run that still
  dispatches with the uncertainty order, within 95 s of the arm draw.
- The kickoff prompt text is byte-identical across the three arms except for the prior lines (a test renders one
  view three ways).
- No plan `why` text appears in any kickoff prompt (test, plus a live grep of one replica-arm prompt from
  `cockpit_turn_sighting`).
- `make substrate-ladder-check`, `check_body_economy_contract.py` and `check-verb-physiology` pass with the new
  family.
- First 14 days: fallback rate reported by reason; pick uptake reported; Juniper chat latency with the replica on
  compared to the 14 days before (E9).
- At the stop: the three-arm report is published whatever it says.

## Proposal mode (§0A)

- **Capability change:** curiosity gains its own small planner that chooses what Orion sees first in about one third
  of regular investigate runs, and that can answer Prime. This is the first organ with a mind of its own.
- **Data touched:** reads Orion's live priors (claims, confidences, test history), recent run findings and scores,
  Prime's open concerns, focus questions and regulate signals, and Juniper's last 5 prompt openings. Writes plan
  rows, act rows, want rows, and its own state in Redis.
- **Privacy boundary:** none between Juniper and Orion (standing decision). One boundary inside Orion on purpose:
  the replica's reasons never reach Orion's curiosity turn (no prompt text, no `orion_readonly` grant), so the
  experiment measures the order and not the replica's words.
- **Trace that proves it worked:** the rows and acts in the acceptance checks, then the three-arm comparison from
  `replay_curiosity_realized_nats.py`.
- **Dangerous failures and guards:**
  - *A favorite prior steered to the top again and again.* Guard: candidates exclude stale priors (tested 3 times),
    so a prior can be pushed at most until it goes stale; the track record shows the replica its own repeats.
  - *A replica that loses by starvation, not by judgment.* Guard: fallback rate by reason, reviewed at 14 days with a
    30% threshold (section 8).
  - *A Prime and replica argument loop.* Guard: one answer per Prime act, 3 a day, 30 minutes refractory, and
    Prime's existing refractory per receiver. Loop check: no answer to an act whose parent is itself an answer.
  - *Kickoff stuck waiting.* Guard: 90 s hard deadline in Hub, then the rule.
  - *Fallback that looks like success.* Guard: every plan row has a status; fallbacks are counted against the arm.
  - *Juniper's chat slowed.* Guard: same lane rules as Prime (Juniper first, preemption), measured in E9.
- **Disable / roll back:** `HUB_CURIOSITY_REPLICA_ARM_ENABLED=false` stops new replica-arm draws at once (rule arms
  only, as today). `CURIOSITY_REPLICA_ENABLED=false` stops the instance; Hub then times out to the rule on any arm
  already drawn. Neither touches Prime. Retirement after a lost test removes the code and family (section 6).

## Recommended next patch

**Patch 0 (tiny, independent, now):** set `HUB_CURIOSITY_VALUE_ORDER_ENABLED=true` in `services/orion-hub/.env_example`
and the local `.env`, with the replay output in the PR. The value arm is built, its gate passes today, and the
replica cannot be judged against an arm that has never run. This is a behavior change, so it needs Juniper's yes.

Then, in order:

1. **Loop library extraction** (`orion/mindloop/`, Prime as instance #1). No behavior change.
2. **Contracts:** schema models, registry, channels, migration, families entry, three-way `offer_arm`,
   `select_priors(head_ids)`, the neutral prompt sentence. Replica flag off at the Hub side until patch 3 lands.
3. **Replica instance + Hub consumer:** frame, plan, validation, fallback, plan rows, acts. Flags on. The three-arm
   test starts the day this deploys.
4. **Prime hears the organ:** challenges and refusals, holdback, "ORGAN REPLIES" in Prime's frame.
5. **Analysis:** three-arm replay with permutation tests and the stop rule; 14-day fallback and E9 report.
