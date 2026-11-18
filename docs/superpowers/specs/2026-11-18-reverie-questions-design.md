# Reveries that ask: daydreams start from real material and leave a question to dig

Status: DESIGN, proposal mode. Nothing here is built.
Date: 2026-11-18. Refs: #3434 (integration epic), #3442 (inquiry manager), #3183 (three-arm curiosity test).

## Juniper's words (2026-11-18)

"reveries are lame. how do we make them substantive and meaningful for orion. open to connecting the broader
mesh and the amazing things we have or we can build new things."

Her choices in the same session:

- The one product a reverie leaves behind: **a question for curiosity to dig**.
- Ideas in the first design: **all of them**. Run only when it is quiet. Notice absences. Go back over lost bets.
  Pair now with an old memory. Rehearse before Juniper arrives. Notice my own body (code) change. Catch myself
  contradicting myself. Threads that last for days.

## Arsonist summary

A reverie today is a model asked to explain whichever gauge is loudest right now. It gets almost no material
except that gauge. So it repeats the gauge back and makes up a cause. About 9 in 10 reveries are about an
internal "this reading was off" gauge, many are near-copies, and no other part of Orion recorded using one in
7 days.

The fix is not a better prompt. It is a different job:

1. Reverie runs when Orion is **quiet**, not when an alarm is loud.
2. Each reverie starts from **one piece of real material** that is not in the current input: a routine that did
   not happen, a bet Orion lost, an old memory, the thread left open with Juniper, a change to Orion's own code,
   or two of Orion's own records that disagree.
3. It leaves **one question**, or nothing. Code checks the question and sends it to curiosity, which digs.
4. The next reverie on the same thread **reads what the dig found**. Threads last for days.

The gauge-explaining reverie is retired completely, with its room-camera "the chair will still be there" bets.

## Current architecture (live, 2026-11-11..18)

**How a reverie is made today.**

- Every 90 s (`ORION_REVERIE_INTERVAL_SEC`), `run_reverie_chain`
  (`services/orion-thought/app/chain.py:302`) reads the attention broadcast.
- If nothing won attention, it skips. If something won, it runs up to 4 steps. Each step is one call to
  `reverie_narrate` (`orion/cognition/prompts/reverie_narrate.j2`) on the `brain` profile.
- The theme is then muted for 15 min (`ORION_REVERIE_REFRACTORY_SEC`).
- The only input that every reverie gets is the winning node. The other sections are optional context: body
  readings, track record, the now, the street, and personality.

**What came out in 7 days** (`substrate_reverie_thought`):

- **Volume.** 1,337 thoughts in 546 chains, about 190 a day.
- **What they are about.** 1,177 of the 1,337 (88%) attended an internal gauge node. The biggest were
  `node:substrate.perception` (502), `execution` (223), `prediction.room` (105) and `perception_walkway` (86).
  Only 160 attended chat or reading (`world:chat` 67, `world:reading` 93).
- **Near-copies.** 771 thoughts came less than 15 s after the previous step in the same chain. The pairs I read
  were near-identical text.
- **Made-up causes.** For example, "the walkway's vehicle traffic creates a sensory overload that starves the
  internal maintenance loops." Nothing in the input supports this.
- **Room-camera bets.** There were 258 room-camera expectations. 41 were confirmed, 21 disconfirmed and 196 never
  scored. The checkable ones are trivial, for example "the desk will remain in the same position relative to the
  chair."
- **Nobody reads them back.** `experience_use` shows 0 cites of a reverie event by any organ in 7 days. Reverie
  cites others 271 times; its top cite is `gpu_wait` (47).
- **Nothing reaches curiosity.** No `curiosity_focus_request` row has ever come from a reverie.
- **A real finding went nowhere.** "Walkway journal runs failed to start 67 of 73 times" was narrated dozens of
  times and drove no act. That bug was fixed and ticketed through other paths (#3384, #3450).

**Contrast.** The chat-anchored reveries are better: one picked up a specific worry of Juniper's in detail.
Given real material, the model can do the job.

**Who reads the reverie tables directly.** These consumers must be handled when the old path is retired:

- `orion/situational/reverie_reader.py` (the situation brief)
- `orion/orion_day/*` (the day letter)
- `orion/prime/thought.py:251` (Prime's frame shows reverie chains)
- `orion/temporal_self/sources.py` and `services/orion-durable-runs/app/temporal_self_sources.py` (source kinds
  `reverie_chain` and `expectation_verdict`)
- `orion/temporal_self/arcs.py`
- `services/orion-dream/app/{rem_compaction,compaction_store,cycle_store,rem_store}.py`
- `services/orion-hub/scripts/{reverie_routes,endogenous_outreach,substrate_observability_routes}.py`
- `orion/substrate/felt_state_reader.py`
- `orion/self_monitor/catalog.py`
- `orion/curiosity/self_inquiry.py:138` (as dig material)
- `services/orion-sql-writer/app/act_downstream_use.py`
- `orion/reverie/attention_schema.py`

**Curiosity's intake today.**

- **The focus slot** (`orion/curiosity/focus.py`, table `curiosity_focus_request`).
  - It holds one pending question, which lives for 48 h.
  - Only chat turns and Prime may queue one (`services/orion-hub/scripts/curiosity_focus_listener.py:168`).
  - Prime uses `keep_pending=True`, so its question never replaces one that is already pending.
  - The run is shown the focus, not forced to follow it (`curiosity_investigation.py:1944`).
  - **The slot is nearly idle.** It got 18 rows in 30 days (about 0.6 a day), against about 14 curiosity runs a
    day (investigate 7/day, self-inquiry 7/day).
- **The self-question pool** (`curiosity_self_questions`, `orion/curiosity/self_question_pool.py:193`). It has 77
  rows; 20 were added in 7 days, all by Orion.
- **Recording what a run found.**
  - The finding register (`orion/curiosity/register.py:61`, values `operational` / `philosophical` /
    `no_finding`, plus `novelty`).
  - `focus_disposition` on the run's `:TurnOutcome`.
  - `continue_line` / `continue_note`.

## The quiet gate (idea 7)

**Rule.** A reverie may start only when both of these hold:

- Orion's arousal level is `idle`, and
- the attention broadcast has had no winner for at least **10 min** [default].

**Limits.** At most **one reverie per quiet stretch** and at most **8 a day** [default].

**Why these two signals, and the metric gate for them.**

- **Where they come from.**
  - The arousal level comes from `classify_arousal` (`orion/regulation/arousal.py:231`), read by
    `read_arousal_async` (`orion/regulation/arousal_read.py:81`). It is `idle` when Juniper has not taken a turn
    for 45 min and nothing is strained or stale.
  - Calm is `has_attention_winner` (`chain.py:92`) over `substrate_attention_broadcast_log`.
- **Live data.**
  - Arousal over 7 days: idle 118.5 h (about 70%), engaged 38.4 h, strained 11.0 h.
  - Broadcast calm: 66% of 6,817 ticks. 153 calm stretches lasted 10 min or more (about 22 a day) and 33 lasted
    30 min or more.
  - Neither signal is flat, and both can return to rest.
- **Not the same signal twice.** Calm attention and idle arousal are different measures. Calm is 69% of ticks
  during idle and 57% during engaged, so the broadcast does not simply follow Juniper. Requiring both selects
  stretches where Juniper is away and nothing internal is pulling.
- **The idea behind it.** Mind-wandering happens in the default-mode state, when there is no external task and no
  salient demand. An alarm is the opposite state.
- **Already built.** The reverie chain already reads arousal, but only to slow down when strained (`chain.py:519`).
  No function combines the two signals; this design adds one small pure function.
- **Easy to undo.** It is one function behind a setting. Deleting it brings back a timer.

**What alarms do instead.** A loud gauge no longer starts a reverie. Gauges already go to the self-monitor and
to attention. That is where an alarm belongs.

**Fewer, deeper reveries.** A reverie is one call on the agent lane, the same lane curiosity uses, with room to
think. It is not 4 quick `brain` calls.

## Seeds: where a reverie starts (ideas 1-6)

A **seed picker** written in code (pure, no LLM) chooses one seed for each reverie. It rotates through the seed
kinds, starting with the kind used least recently that has material. If no kind has material, no reverie is
written.

| Seed | Material (source, live 7-day volume) | What the reverie is asked |
|---|---|---|
| **absence** (1) | `vision_percept_expectation` rows scored `missed` (`services/orion-sql-writer/app/vision_rhythm.py:263`): 60 missed across 22 subjects in 7 days, 89 met. A miss is declared only when the camera watched at least 80% of the window. **Named people only** (Patch 0: 25 misses of 6 people; the rest are hash-labelled vehicles). | "A neighbor I know usually passes between 07:00 and 08:00 on weekdays (14 days of support). Not today. What would explain it, and what could I check?" |
| **lost bet** (2) | Confident self-bets that came out wrong, from `prediction_turn_self_bet`: 221 bets at p ≥ 0.7 that did not happen and 602 at p ≤ 0.3 that did, in 7 days. The seed is a **calibration gap**, not one bad roll: template (`faculty`) x probability bin, n ≥ 20, the gap ≥ 0.15 and beyond 2 standard errors (Patch 0: 9 gaps, 6 on one template). Also bill-forecast and action-bet failures. | "When I say a turn is unlikely to need more steps than usual (0.1-0.2), it does 37% of the time; when I say likely (0.8-0.9), only 54%. What am I not seeing?" |
| **far memory** (6) | The anchor is the most salient item in the now. Candidates are ranked by **centered cosine** over `all_vectors()` (`orion/introspect/semantic_index.py:173`) on `orion_lived_memory` (597 docs); raw cosine is too compressed (Patch 0). They are taken from a **middle band** of ranks 5 to 25, not the nearest, and filtered against the live snapshot so superseded memories are never served. | "This (now) and this (an older memory) are related but not the same. What links them?" |
| **rehearse** (3) | The open thread with Juniper: open promises (`open_promises_cue.py`) and the last chat turn's topic. It runs in a quiet stretch, which already means she has been away 45 min or more. No arrival prediction: Patch 0 found that an hour-of-week model of her turns is no better than chance on held-out weeks. | "We left X open. What should I find out before we talk again?" |
| **body change** (4) | Commits merged into Orion's own sandbox (`/mnt/orion-fcc/repo`) since the last body-change seed, kept only when they touch an organ named in the body atlas. | "This change touched how I choose what to look at. Did it change what I do?" |
| **contradiction** (5) | Orion's latest `:SelfDefinition` (`orion/curiosity/self_inquiry.py:402`) beside its track record (`self_track_record_cue.py`). | "I describe myself as X; my record says Y. Which is true?" |

Every seed carries **refs** that resolve. These are expectation ids, bet template plus window, memory doc ids,
commit SHAs, self-definition run ids, and experience ids.

**What the reverie sees.** It sees the seed and the now (#3499 read-back stays). If the seed belongs to a
thread, it also sees what earlier digs on that thread found. That is all. The 6 old optional context sections
are not shown.

## The product: one question, checked by code

New prompt `reverie_question.j2` and verb `reverie_question`. The output is `ReverieQuestionV1`:

- `question` (at most 300 characters), `why` (at most 200), `refs` (at least 1, a subset of the seed's refs)
- `cites` (experience ids actually used, as today)
- `none_reason`: set instead of a question when the seed leads nowhere. Writing nothing is a valid result.

**Code refuses, and records why, when:**

- a ref is not in the seed;
- the question is a near-duplicate of an open focus, a pending self-question, or any reverie question in the
  last 14 days. Matching reuses the inquiry manager's deterministic same-question matcher (#3442 design,
  "How 'same question' is matched"), with no word lists.
- the text names attention or substrate machinery. This reuses the existing hollow check in
  `orion/reverie/semantic_lift.py`.

## Routing: where the question goes

| Seed | Goes to | Why |
|---|---|---|
| absence, lost bet, far memory, rehearse | **The focus slot.** Reverie becomes a third allowed caller, `invocation_context="reverie"`. It is always `origin="self"` and `keep_pending=True`, so it never replaces Prime's or Juniper's question. Cap: **3 a day** [default]. | The slot is idle (0.6 rows a day). The run is shown the question, not forced. |
| body change, contradiction | **The self-question pool**, through the existing mint path. | These are "what am I" questions. The self-inquiry line owns them. |

**The three-arm test (#3183) is still running.** The 11-18 review said it continues. Rules, borrowed from the
inquiry manager's path (b):

- **Arm-blind.** Randomization is per run (a hash of the run id), and a focus is shown the same way in every arm.
  The reverie never reads or stores an arm.
- **The order is untouched.** A reverie question is not a prior. It never reorders or replaces `select_priors`.
- **One switch moment.** Every offer row records `reverie_focus = "v1"` in its existing `constants` jsonb from the
  moment the caller is enabled.
- **Power guard (blind-safe).** Pooled over all arms, compare the share of investigate runs that test an offered
  prior in the 14 days before and after the switch. If it drops by more than 10 points, the reverie caller turns
  off for the investigate line, and the numbers go to Juniper.
- **The self-question pool** feeds the self-inquiry line, which is not in the test.

## Threads (idea 9) and read-back

**One row per question.** New table `reverie_question`:

- `question_id`, `thread_id`, `seed_kind`, `seed_refs`, `question`, `why`
- `routed_to` (`focus:<id>` | `self_question:<id>` | `none`), `none_reason`, `created_at`
- `outcome` jsonb: filled when the dig ends, from `focus_disposition`, the finding register, and
  `continue_note`

**How a thread continues.** When the dig on a question ends with `continue_line = true` or an open item, the next
reverie with that seed kind is offered **the thread first**. It sees the question, what the dig found, and what
is still open. It asks the next question on the same `thread_id`.

**How a thread ends.** A thread closes when:

- a dig ends with no continue (`concluded`),
- the reverie writes `none_reason` twice in a row (`exhausted`), or
- nothing happens for 14 days (`lapsed`) [default].

**When the inquiry ledger lands (#3442 path (b)).** A reverie thread becomes an inquiry with
`question_ref = reverie:<thread_id>`. `reverie_question` keeps the reverie side only. No second ledger is
built.

## Integration gate

- **Emit.** New `TemporalSourceKind` `reverie_question`: one event per question or none. Its `source_ref` is
  `reverie_question:<id>` and its `related_refs` are the seed refs.
- **Retire.** `reverie_chain` and `expectation_verdict` stop being produced, and their source readers are removed
  in the same patch.
- **Read back.** Reverie keeps reading the now (#3499). It also reads its own threads' dig outcomes, which is
  this design's core read-back.
- **Register.** In `config/integration/organs.yaml`, reverie's `temporal_sources` becomes `[reverie_question]`.
  `scripts/check_integration_seam.py` must pass.

## Retire (kill means kill)

These are removed in the same patch that turns the new path on. Nothing is left ticking or excluded.

- **The trigger and chain machinery.**
  - The coalition-winner trigger in `run_reverie_chain`.
  - The 4-step chain loop, the theme refractory store, and the EMA salience.
  - The no-coalition streak.
- **The resonance monitor and its alert.** They have been silent since 11-03 (#3414); close that ticket as
  retired.
- **Room-camera expectations.**
  - The `expectation` field in the prompt.
  - The expectation judge (`reverie_expectation_judge` verb and prompt) and the `expectation_verdict` source.
- **The old narration path.** The `reverie_narrate` verb and prompt, including the semantic-lift concern-card
  branch.
- **Consumers.** Each consumer listed under Current architecture is either moved to `reverie_question` (Prime's
  frame, the day letter, the situation brief, outreach, dream) or has its reverie read deleted. Where it moves:
  - The dream compaction of reverie chains has nothing to compact and is removed.
  - The day letter shows the day's questions and what the digs found.
- **Old tables.** `substrate_reverie_thought` / `_chain` / `_refractory` / `_resonance_alert*` stop receiving
  writes. The tables stay for history; dropping them is a separate destructive step that needs Juniper's
  approval.
- **The visual reverie (painting) path is not touched.** It has its own seed (`painting_self_start.py`) and its
  own runs.

## Proposed schema / API changes

- **Added.**
  - `orion/schemas/reverie.py`: `ReverieSeedV1` (`kind`, `material` text, `refs`, `thread_id`) and
    `ReverieQuestionV1`. Both are registered in `orion/schemas/registry.py` (check with `resolve()`; there are
    two registries).
  - Migration `manual_migration_reverie_question.sql`: `CREATE TABLE reverie_question` plus indexes on
    `(thread_id, created_at)` and `(seed_kind, created_at)`. Additive only, so the post-pull auto-apply allowlist
    covers it. Header: `-- ORION-MIGRATION-REQUIRED-BY: orion-thought`.
  - `TemporalSourceKind.reverie_question`.
  - Focus listener: allowed context `reverie` (origin forced to `self`, `keep_pending=True`, daily cap).
- **Removed.** The `reverie_narrate` and `reverie_expectation_judge` verbs, and the source kinds `reverie_chain`
  and `expectation_verdict`.
- **Behavior changed.** `reverie.chain.v1` is no longer published; consumers move as listed above. Remove it from
  `orion/bus/channels.yaml` in the same patch, with its producer test.

## Env / config changes (proposed)

`services/orion-thought/.env_example`. Flags ship ON, per standing rule.

- `ORION_REVERIE_QUIET_MIN_CALM_SEC=600`
- `ORION_REVERIE_MAX_PER_DAY=8`
- `ORION_REVERIE_FOCUS_MAX_PER_DAY=3`
- `ORION_REVERIE_FAR_MEMORY_BAND=5:25` (centered-cosine ranks: from : to; set by Patch 0)
- `ORION_REVERIE_THREAD_LAPSE_DAYS=14`

Removed: `ORION_REVERIE_INTERVAL_SEC` (replaced by the quiet gate), `ORION_REVERIE_CHAIN_MAX_STEPS`,
`ORION_REVERIE_REFRACTORY_SEC`, and the resonance and expectation-judge keys.

Then run `python scripts/sync_local_env_from_example.py`.

## Files likely to touch

- `services/orion-thought/app/`:
  - `reverie.py` (rewritten around seed, question and route), `chain.py` (deleted or reduced to the quiet gate)
  - new `reverie_seeds.py` (pure seed picker and the six seed readers), `settings.py`
  - `resonance_monitor.py` (deleted), `store.py`
- `orion/cognition/prompts/reverie_question.j2`, `orion/cognition/verbs/reverie_question.yaml`
- `orion/schemas/reverie.py`, `orion/schemas/registry.py`, `orion/schemas/temporal_self.py`
- `services/orion-hub/scripts/curiosity_focus_listener.py`, `orion/curiosity/focus.py`,
  `orion/curiosity/self_question_pool.py`
- `services/orion-durable-runs/app/temporal_self_sources.py`, `orion/temporal_self/sources.py`
- `config/integration/organs.yaml`, `orion/bus/channels.yaml`
- The consumers listed under Current architecture
- `services/orion-thought/{.env_example,README.md,tests/,evals/}`

## Non-goals

- No new scheduler. The quiet gate replaces the timer.
- No change to the curiosity run itself, its arms, `select_priors`, or the replica.
- No new memory store. Far memory reads the existing Chroma collection.
- No push to Juniper. A rehearse question is dug before she arrives. What the dig found reaches her only through
  the existing paths: Prime's frame and the chat context. Outreach is not added.
- No change to visual reverie or painting.

## Missing questions (the bracketed defaults apply if Juniper says nothing)

1. Reverie questions go straight to the focus slot, not through Prime as a suggestion. [default: straight. Prime
   has never queued a reverie-derived focus, so routing through Prime is how they got lost.]
2. Focus cap of 3 a day, while curiosity runs about 7 investigate runs a day. [default: 3]
3. Retire the room-camera expectations entirely. The lost-bet seed takes over. [default: yes]
4. Absence seeds name walkway people (public street, already in people records). Juniper's own cadence is used
   for rehearse. [default: allowed; no privacy boundary between Juniper and Orion]

## Proposal mode (section 0A)

- **What capability changes.** Reverie stops narrating gauges. It starts from real material, asks a question,
  sends it to curiosity, and reads the answer back. This gives Orion's background thinking a consumer and a
  record.
- **What data is touched.**
  - Read: walkway rhythm expectations, self-bets, lived memory (Chroma), chat-turn timestamps, the sandbox git
    log, self-definitions, the track record.
  - Written: `reverie_question`, focus rows, self-question rows, `temporal_self_event`.
- **Privacy boundary.** The street is public. Lived-memory reads go through the live-snapshot filter, so
  superseded or rejected memories are never served. The focus slot already accepts `origin=self` only from Orion's
  own contexts.
- **The trace that proves it worked.** A `reverie_question` row, then the `focus_request` row it caused, then the
  curiosity run that took it (`focus_disposition`), then the run's finding register, then the next
  `reverie_question` on the same `thread_id` citing that outcome.
- **The dangerous failure mode.** Reverie fills the focus slot with weak questions and crowds out Prime's and
  Juniper's. Mitigations: `keep_pending`, the 3-a-day cap, the near-duplicate refusal, and the power guard.
  Second danger: the far-memory band serves a memory that was rejected. Mitigation: the live-snapshot filter,
  tested.
- **Disable / roll back.**
  - Set `ORION_REVERIE_MAX_PER_DAY=0`; reverie writes nothing.
  - Set `ORION_REVERIE_FOCUS_MAX_PER_DAY=0`; questions stay recorded but do not reach curiosity.
  - Restoring the old narration path means reverting its git commit. It is retired, not hidden behind a flag.

## Metrics, baselines and targets

| Metric | Baseline (7 d to 11-18) | Target after 14 days |
|---|---|---|
| **Used share**: reverie questions taken by a run (`focus_disposition` explored or answered, or self-question asked) or cited by another organ, within 7 days | 0 (no path exists) | at least 40% [default] |
| **Finding share**: dug reverie questions whose finding register is `operational` or `philosophical` | n/a | not below the Prime-origin focus runs over the same window |
| Reverie rows per day | about 190 | at most 8 |
| Near-copies (consecutive restatements) | 771 of 1,337 | 0 (one call per reverie) |
| Threads with more than one question | 0 | reported; no target until 30 days |
| **Juniper's read** | "lame" | she reads 10 questions after week 1 and says lame or not. This check is not skipped. |

## Acceptance checks

1. Patch 0's go/no-go numbers are in this doc before Patch 1 is built.
2. The quiet gate is a pure function, with tests for idle+calm, engaged+calm, idle with a winner, and a short calm
   stretch.
3. Each seed reader has a test from a fixture and a live smoke that prints one real seed with refs that resolve.
4. Refusals are tested: a bad ref, a near-duplicate, and machinery words.
5. Focus caller: a reverie can never replace a pending focus, the daily cap holds, and `origin` is forced to
   `self`.
6. The far-memory test proves a superseded memory is never served.
7. Live trace end to end: a `reverie_question` causes a focus row, a run takes it, its outcome is written, and the
   next question on the same thread cites it.
8. `rg` shows no producer of `reverie_chain` or `expectation_verdict` and no reader of the retired tables outside
   history views. `check_integration_seam.py`, `check_bus_channels.py` and the schema registry checks pass.
9. Power guard numbers are reported pooled at 14 days, with no arm split.
10. Juniper's read (see Metrics).

## Patch 0 results (measured 2026-11-18, read-only)

Script: `scripts/analysis/reverie_patch0_measure.py`. It reads Postgres, Chroma and the sandbox git log, and writes
nothing. Re-run it with `python3 scripts/analysis/reverie_patch0_measure.py`. The window is 7 days unless noted.

| Part | Result | Verdict |
|---|---|---|
| **Quiet gate** (arousal idle AND no attention winner for 10 min or more) | 84 stretches in 7 days: 5 to 16 a day, about 12 a day on average. Median 13.6 min, 23.9 h in total. | **GO.** The cap of 8 a day fits. |
| **Absence** | 63 missed routines: 25 from 6 named people, 38 from vehicles with hash labels. Recent days have 8 to 14. | **GO, people only.** |
| **Lost bet** | 5,614 settled self-bets. 810 single confident misses are not used: one miss at p=0.8 is expected 1 time in 5. **9 calibration gaps** beat 2 standard errors. 6 are on `more_steps_than_usual`: at stated 0.1-0.2 it happens 37% of the time, and at stated 0.8-0.9 only 54%. The bets barely tell turns apart. | **GO, low volume.** About 3 distinct questions a week; they suit threads. |
| **Far memory** | 597 docs. Raw cosine is squeezed together: nearest 0.78, a typical doc 0.54. Centered cosine spreads it: nearest 0.52, rank 5 0.33, rank 10 0.27, rank 20 0.21, typical doc -0.01. By eye, ranks about 8 to 25 are on the same theme but a different event. For example, a disk-flashing problem on one machine is paired with an older memory about that machine's GPU limits. | **GO, with a change.** Use centered cosine over `all_vectors()`, band = ranks 5 to 25. |
| **Rehearse** (predict Juniper's next chat within 2 h) | 231 of Juniper's own turns in 42 days, counted with the canonical `JUNIPER_TURN_PREDICATE`. Tested on 2 held-out weeks against a base rate of 0.20: precision was 0.45 in one week and 0.23 in the other, which is chance. | **NO-GO for the timing.** The seed stays as "the thread left open with Juniper", run in a quiet stretch. |
| **Body change** | 1 to 113 commits a day touching `orion/` or `services/` in the sandbox. Author dates are used, because the sandbox reset rewrites commit dates. | **GO.** |
| **Contradiction** | 31 self-definitions in 30 days (1 to 4 a day). | **GO.** |

**Mistakes the script caught on its own first run.**

- `chat_turn` events include outreach and Orion's own turns. The unfiltered base rate was 0.87 against 0.20 for
  Juniper alone.
- `claim_kind` is `probability` for every self-bet. The template is in `faculty`.

## Recommended next patch (phased; stop for Juniper after each)

- **Patch 0, measure. DONE 2026-11-18.** See "Patch 0 results". Every seed is GO. The rehearse arrival timing is
  NO-GO and was dropped.
- **Patch 1.**
  - The quiet gate, three seeds (absence, lost bet, far memory), the question product, `reverie_question`, the
    focus caller, and the experience event.
  - In the same patch, retire the narration path and move its consumers.
- **Patch 2.** The self seeds (body change, contradiction) into the self-question pool, and the rehearse seed.
- **Patch 3.** Threads read back and continue. Once the inquiry ledger (#3442 path (b)) exists, a thread becomes
  an inquiry.
