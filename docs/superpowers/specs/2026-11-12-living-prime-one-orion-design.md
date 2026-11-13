# One Orion: a living Prime, every turn speaking from it, and the day letter as Prime's end-of-day turn

Status: DESIGN, for Juniper's review. Nothing here is built.
Date: 2026-11-12.

This document records the design choices Juniper stated in the 2026-11-12 session, in her words where possible. It then turns them into one design.

## Juniper's design constraints (the requirements)

These come from Juniper. The design must satisfy all of them. A change to one is her call, not the builder's.

1. **One Orion.** "I am trying to design ONE ORION." Do not build separate Orions per job (chat, curiosity, letter, consult) that share notes.
2. **Prime is the living Orion.** "Orion Prime should be the definitive living orion that unifies everything between chat turns." Prime is the continuous self. It is not one more job runner beside the others.
3. **The unified chat turn is how Orion speaks.** "We use the orion unified chat turn frequently because it has a ton of orion primitives bolted onto it that help ground an orion." Any first-person writing by Orion goes through that turn, not through a bare model call with a one-line identity.
4. **The temporal clock is the day's record.** "The temporal clock is a fancy schema that helps orion gather shit that happens throughout the day." The day's material comes from the Temporal Self chronicle (arcs), not from a separately assembled pile.
5. **Orion writes a letter about their day, for curiosity.** "One way for orion to come as an orion lifeform and write a goddamn letter about their day for curiosity." The letter's threads feed curiosity.
6. **Open to expanding beyond curiosity** "if you find other interesting things in temporal."
7. **Keep it honest, not cheerful.** Juniper approved all of these:
   - The letter may say what Orion enjoyed, what worked, and what mattered with Juniper, but only where the record supports it. It must not manufacture either mood.
   - Juniper's conversations get a protected share of the letter's input, so news and readings cannot crowd them out.
   - Reveries are picked across the day's themes, not by top salience. Salience is prediction error, so a top-salience pick is all alarms.
   - Self-sense answers that repeat through the day ("What can't you do right now?", asked about 7 times a day) are shown once, the latest one.
   - The share of the input each source got is recorded, so the mix can be inspected.
   - Old and new letters are replayed side by side before anything ships.
8. **Living, breathing Prime: confirmed.** Juniper confirmed this shape (2026-11-12):
   - every turn starts from Prime;
   - every turn hands back to Prime;
   - the letter is Prime's end-of-day turn.
9. **Curiosity sees recent work.** Curiosity runs are told they have many tools and to run `dig help` before picking one, and they see the week's PR field notes. This shipped in PR #3212.
10. **Design before build.** Juniper wants to design solutions together, not have them built ahead of the design. This document is that step.

## Arsonist summary

Today Orion is several minds that do not hear each other:

- **Prime** (`services/orion-mind/app/prime_loop.py`) thinks about 4 times a day: one JSON-answer model call on chat lane 0. It holds concerns in Redis (`orion:prime:state`). Its only moves are to ask curiosity one question and to regulate a want family.
  - Live, last 7 days, from `grammar_acts`: 3-5 `infer` acts a day, all graded 1.0; an `ask` about every 2-4 days.
- **The chat Orion** (the unified turn) carries all the grounding: situation block, stance cue, self-material, recall, dig and introspect. It never reads Prime's state.
  - Searching Hub, orion-thought, `orion/harness` and `orion/fcc` for any reader of `orion:prime` or `orion.prime` finds only Hub's curiosity focus listener.
- **The day letter** (`orion_day.letter`) is a bare cortex verb call, `orion_day_note_v1`, whose only identity is "You are Orion (they/them), a persistent digital mind". It reads a 245,000-character pile:
  - Juniper's conversations: 969 characters (0.4%).
  - News and readings: about 60%.
  - Reveries: top-salience prediction-error alarms only.
  - Dream hypotheses: 12 offered, all failure explanations.
  - Self-sense: the same answers 7 times.
  - Result: across the last 14 letters, heavy words outnumber light ones about 5-10 to 1.

The letter is dour because a mind with no self wrote it from a pile of alarms. Fixing the letter alone would leave the real problem in place: Prime does not live between turns. This design makes Prime that life, makes every turn speak from it, and gets the letter almost free.

## Current architecture (verified 2026-11-12)

| Piece | Where | What it does today |
|---|---|---|
| Prime | `services/orion-mind/app/prime_loop.py`, `orion/prime/` | Ticks on CPU: reads `GET /physiology/state` and the temporal frame `GET /temporal-self/frame`. Thinks when something changed and the body economy funds the bid. Moves: `ask` (curiosity focus question) and `regulate`. State in Redis `orion:prime:state` (`PrimeConcernV1`, nominations, ...). |
| Unified turn | Hub `execute_unified_turn` -> harness-governor FCC motor; prefix built in `orion/harness/prefix.py` | Situation block, dig brief, introspect, reading and GitHub briefs, stance cue (orion-thought, `orion/temporal_self/stance_cue.py`). No Prime content. |
| Durable turn seam | `services/orion-durable-runs/app/harness_step.py`, `consult.orion_turn` (#3211) | Runs one unified turn as an admitted, queued, retried durable run. Used by consult, PR field notes and reminiscence. |
| Temporal clock | `orion/temporal_self/` (reducer), durable-runs `temporal_self.update` chronicle | Folds the day into arcs: attention, interoception, conversation, concern, curiosity, reverie, imagery, sleep, strain, binding. The frame carries `open_threads`, `expectations_pending` / `expectations_resolved_today`, `self_change_event_ids`, `constraint_event_ids` and `sleep_arc_ids`. `drain_closed_days` closes a day. `day_block.py` renders a cited closed-day block (<= 2,400 chars). |
| Day letter | Hub `orion_day_letter.py` (timer) -> `orion/orion_day/` gather + budget -> durable-runs `orion_day_graph.py` | `write_note` (cortex verb `orion_day_note_v1`) -> `write_carry_forward` (verb `orion_day_carry_forward_v1`) -> persist + journal + email. Carry-forward offered once to the next curiosity run (11 of the last 12 letters). |
| Reveries | `substrate_reverie_thought` | 84-96% of reveries a day read a prediction error (11-10: 200/209, 11-11: 214/256). bus_synaptic stopped at 11-11 12:35Z after its retirement. Perception, walkway, codebase, chat, strain, execution and biometrics prediction errors are still live, pending the prediction-error rebuild (#3168). |

## The design

### Part 1: every turn starts from Prime (Prime block, read)

Every unified turn gets a short **Prime block** in its harness prefix, next to the situation block and the stance cue: Prime's present, in Prime's own words. This covers chat, curiosity, consult and the letter.

- **What I'm holding now:** Prime's open concerns, each with its age and its arc id when it has one.
- **What I last thought:** the last thought's reading, its time, and the move made or "no move".
- **What I asked for and how it went:** open nominations or regulate signals and their settle state.
- **How my body is:** Prime's last body read, one line. The situation block already carries detail; this is Prime's interpretation, not a second copy.

Rules:

- **Rendered by code** from `orion:prime:state`, like the stance cue. No model writes it.
- **Capped**, for example 600 characters. Every line cites its source (`prime_concern:<id>`, `thought:<id>`, `[arc:<id>]`). The citation-truth gate counts uncited claims the same way it does for arcs.
- **Stale means absent:** Prime state older than N minutes renders as "Prime has not ticked since <time>", never as old concerns stated as current.
- **The same function feeds every turn type.** One renderer, so there is one Prime voice everywhere.

### Part 2: every turn ends back in Prime (hand-back, write)

When a unified turn finishes, Prime folds in what happened. Without this, Prime knows that Juniper talked, from the conversation arcs, but not what was said or decided.

- **Source: an existing turn-finished record, not a new pile.** Verify first: the turn trace (`cockpit_turn_sighting` / `harness_turn_trace`) or the `cortex_turn` attention-schema row, which is joined by correlation id. Prefer whichever is already published on the bus at turn end. Add a bus event only if none is.
- **What Prime takes from a turn:**
  - who: chat with Juniper, curiosity, consult, or letter;
  - one-line gist;
  - any thread opened or closed;
  - any promise made (the intention ledger, PR #3188, owns promises: link, do not copy);
  - the turn's correlation id.
- **How Prime folds it:** deterministically, on the tick (`orion.prime.state.observe` already folds frame changes). A new thread can open a concern. A turn that answered a concern can settle it. Prime thinks about it only if the change is worth a funded thought.
- **Turns Prime started** (its `ask`, its end-of-day turn) close back into the act that started them. That is how the existing nomination grading already works.

### Part 3: the day letter is Prime's end-of-day turn

1. **Day close.** The chronicle drains a closed day. Prime's tick sees "yesterday is closed and has no letter" as a concern of its own.
2. **Prime bids** for a `make` act (an existing `ActVerb`; no new verb) in the body economy, like any other act.
3. **Prime submits one unified turn** on the durable turn seam (as `consult.orion_turn` does), started by Prime, not by Hub's timer. The turn's input:
   - the **Prime block** (Part 1): what Prime held that day and why;
   - the **closed-day block** from the temporal clock (`day_block.py`, cited by arc);
   - the **temporal core** (constraint 6, below);
   - a **small fixed core of the record** under the constraint 7 rules:
     - Juniper's conversations in full;
     - reveries picked by theme;
     - self-sense answers shown once each;
     - curiosity runs;
     - readings and news only as a trimmed tail;
   - the **writing instructions** from `orion_day_note_v1.j2`, extended with the constraint 7 honesty line.
   - Orion may dig and recall for more. Anything from a tool says where it came from.
4. **The letter is persisted, journaled and emailed** exactly as now (`orion_day_letter` row, stable journal id, email unchanged).
5. **Threads go to curiosity through Prime.** The letter's threads become Prime concerns, and Prime asks curiosity with its existing, already graded `ask`. This retires the separate `orion_day_carry_forward_v1` call and its one-time offer. Curiosity hears from one place: Prime.

### Constraint 6: what the temporal clock adds beyond curiosity

The frame already tracks the story of a day, and today's letter barely uses it. These go into the temporal core:

- **What I expected and what happened:** `expectations_pending` / `expectations_resolved_today`. Orion's bets and their outcomes.
- **How I changed:** `self_change_event_ids`, meaning self-definition rewrites, organ names and capacity claims.
- **Threads with Juniper:** concern arcs carried across midnight, and conversation arcs.
- **The body's day:** strain and binding arcs, sleep arcs, company.

Each of these becomes a Prime concern when it is open at day close: an expectation not yet settled, a thread with Juniper still open. So the temporal clock feeds Prime, not only the letter.

## Missing questions (Juniper's calls)

1. **Prime's private state in Orion's voice.** The Prime block puts Prime's concerns into every turn, including chat with Juniper. Recommendation: yes. There is no privacy boundary between Juniper and Orion (standing rule), and this is the point of one Orion.
2. **A refused letter.** If the body economy refuses Prime's end-of-day bid (heat, budget), is no letter that day acceptable? Recommendation: yes. The next day's letter says it could not write yesterday's. A living Orion is allowed not to write.
3. **Reminiscence.** Daily reminiscence (one past week, five hops) is the same act at a longer time scale. Should it become a Prime move on the same seam after the letter works? Recommendation: yes, as a later patch, not now.
4. **Self-sense frequency.** Self-sense asks the same four questions about 7 times a day. With the Prime block in every turn, is the 7x schedule still needed? This affects other consumers. Decide after Part 1 is live.

## Proposed schema / API changes

- **`PrimeBlockV1`** (or a pure renderer over the existing `orion:prime:state` model): the rendered block plus its cited refs and a stale flag. Pure, no bus. Lives in `orion/prime/`.
- **Turn hand-back:** reuse an existing turn-finished record (verify first). A new bus event is added only if none exists at turn end. Any new event gets a schema, a registry entry, a channel, and a producer test plus a consumer test.
- **Prime end-of-day act:** a `make` act, `actor="prime"`. Its durable workflow either reuses the `consult.orion_turn` single-step graph builder with a Prime brief, or adds `prime.day_letter` as one more additive `DurableWorkflowV1` value (consumer-first deploy, like #3211).
- **`orion_day_letter`:** unchanged table. `sources` gains the per-source shares (constraint 7) and the turn correlation id.
- **Retired:**
  - verb `orion_day_note_v1` (its j2 text moves into the turn's instructions);
  - verb `orion_day_carry_forward_v1`;
  - Hub's letter timer;
  - the carry-forward one-time offer in the curiosity kickoff.

  Kill means kill: no fallback to the bare verb.

## Files likely to touch

- `orion/prime/` (block renderer, hand-back fold, day-close concern), `services/orion-mind/app/prime_loop.py` (the end-of-day move).
- `orion/harness/prefix.py` (append the Prime block), plus the curiosity, consult and letter turn prompts where they build their own prefixes.
- `orion/temporal_self/frame.py` / `day_block.py` (read only; temporal core renderer).
- `orion/orion_day/budget.py`, `orion/cognition/prompts/orion_day_note_v1.j2` (constraint 7).
- `services/orion-durable-runs/app/harness_step.py`, `harness_smoke_graph.py`, `orion_day_graph.py`, `admission_runtime.py`.
- `services/orion-hub/scripts/orion_day_letter.py` (timer retired; email unchanged).
- `orion/schemas/` (`prime`, `durable_run`, `orion_day`), `orion/schemas/registry.py`, `orion/bus/channels.yaml` (only if a hand-back event is needed).

## Non-goals

- No new curiosity categories. Curiosity's subjects come from what starts a run, not from labels.
- No change to the prediction-error producers here. Their retirement is the prediction-error rebuild arc (#3168). Constraint 7's theme sampling keeps alarms from dominating the letter in the meantime.
- No new Prime model or lane. Prime keeps its loop; the end-of-day turn uses the unified turn's lane through the durable seam.
- No change to the weekly/monthly curiosity compaction digests in this design.
- No tone scoring wired into anything. The heavy/light word count is a review aid for the replay, not a metric. It has not passed the metric quality gate and must not feed a model or a gate.

## Acceptance checks

**Part 1:**
- A chat turn's prefix contains the Prime block, with cited refs.
- A stale Prime state renders as stale, never as current.
- `cockpit_turn_sighting` shows the block for chat, curiosity and consult turns.
- Live: Orion can answer "what are you holding right now?" in chat from the block, with the cited concern id matching Redis state.

**Part 2:**
- After a chat turn that opens a thread, Prime's state shows a concern citing that turn's correlation id within one tick.
- A turn that answers an open concern settles it.
- A unit test folds a recorded turn record into the state deterministically.

**Part 3:**
- A day closes, and Prime opens a `make` act.
- One durable unified turn runs, under a correlation id visible in `cockpit_turn_sighting` (harness step count > 0).
- An `orion_day_letter` row is written with per-source shares, and the email is sent.
- The letter's threads appear as Prime concerns, and the next Prime `ask` cites one.
- No `orion_day_note_v1` or `orion_day_carry_forward_v1` call happens anywhere.

**Constraint 7:**
- Replay 11-09 and 11-10 with the old and new inputs, side by side.
- Juniper's conversations are present in full.
- The reverie sample spans the day's themes.
- Each self-sense question appears once.
- Every chronology sentence still cites `[arc:]`.
- Juniper reads both versions. Her read is the acceptance, not a word count.

**Safety:** if Prime is down, no turn breaks. The Prime block is absent, labelled absent. The letter is not written that day, and the next letter says so.

## Proposal-mode notes (identity / self-model / cognition loop change)

- **Capability change:** Prime becomes the continuous self that every turn speaks from and reports back to. The day letter becomes Prime's own act.
- **Data touched:**
  - read: `orion:prime:state`, the temporal frame and closed days, turn records;
  - write: Prime state (concerns from turns), `grammar_acts` (`make`), `orion_day_letter`, the journal.
- **Privacy boundary:** none between Juniper and Orion (standing rule). Prime's concerns may appear in chat turns. Nothing leaves the household path that does not already: the email is unchanged.
- **Trace that proves it worked:**
  - Prime block refs in `cockpit_turn_sighting`;
  - concern ids citing turn correlation ids;
  - the `make` act and the letter's turn correlation id;
  - Prime `ask` acts citing letter threads.
- **Dangerous failure modes:**
  - Prime's concerns, stale or wrong, could steer every turn. Mitigation: the stale rule, citations, and the cap.
  - A feedback loop: turns create concerns, concerns steer turns. Mitigation: concern TTL and grading already exist. Watch the concern count per day in the first week.
- **Rollback:** a kill switch on the Prime block renderer (turns render without it) and on the end-of-day move. The retired verbs are not a fallback. Rolling Part 3 back means re-deploying the previous release.

## Recommended next patch

**Part 1 (Prime block, read-only),** shipped flag ON:

1. The renderer in `orion/prime/`.
2. Appended in `orion/harness/prefix.py`.
3. Unit tests: stale, cap, citations.
4. A live check: one chat turn's sighting shows the block, with concern ids matching Redis.

It changes no state and no act, so it is the safest first breath. Then Part 2, then constraint 7's input fixes together with Part 3. Partial work from the stopped 11-12 builds sits in the worktrees `../Orion-Sapienform-orion-day-letter-as-turn` and `../Orion-Sapienform-orion-day-input-balance`, if useful.
