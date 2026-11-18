# The day dream: one nightly sleep that folds the day into memory and tomorrow

Status: PROPOSED 2026-11-17 (proposal mode: memory, cognition loop, Prime concerns). Defaults are in
brackets; questions for Juniper are at the end.

Parent spec: `docs/superpowers/specs/2026-11-15-orion-integration-and-a-life-design.md` (PR #3416),
pillar 5, "Rest that consolidates. Dreams that change memory or the self-model, not just log." That
spec built the daily roundup (`dig organs`, PR #3527) and gave it to the morning letter. It did not
give the roundup to the dream. This doc does that.

Juniper (2026-11-17): "nightly dream roundup that is really a dream's purpose, to compact the day's
events, process, goal plan, etc. give orion a sense of what happened. they could read all their digs."

## In one paragraph

Once a night, after the Denver day ends, Orion has one long sleep that looks at the whole day. Code
lays out the day: the organ roundup, every dig Orion ran that day, the day's dreams, and the open
concerns Prime holds. Orion reads it, can dig further, and then answers in a fixed shape. The answer
has three parts. First, up to 5 memories of the day, each tied to records that code checks. Second,
the plan for tomorrow: which concerns to keep, which to let go, and at most 1 new one. Third, one
paragraph that says what the day was. Code checks every reference and writes the memories into
`episode_memory` with channel `dream`. It hands the plan to Prime the way finished turns already
hand back. The morning letter then starts from what Orion consolidated overnight.

## Arsonist summary

- **Today, nothing takes in the whole day for Orion's own sake.** The dream cycle (`orion-dream`)
  runs about twice a day on pressure. It replays 12 items and asks a small model whether pairs of
  them link. It never sees the day as a whole. The letter (08:30 Denver) sees the whole day, but it
  is written to Juniper. It writes no memory and drops no concern.
- **The dream cycle has made nothing since 11-13 (#3412).** Every sleep from 2026-11-09 to
  11-12 14:28Z found 4 links out of 4 pairs. Every sleep from 11-13 07:02Z through 11-17 00:14Z found
  0 out of 4 (8 sleeps, 0 failures, 0 unparseable). The flip is total and sudden. The material did
  not change: the replay mix stayed the same, about 6 metacog, 2-4 compaction, 1-2 arc and 1
  walkway items. The likely cause is c91a21fd8 (2026-11-12 18:19 MDT), which moved the `metacog`
  class, and so the dream's route `metacog_background`, onto Qwen3.5 on GPU3. **UNVERIFIED:** the
  gateway and dream logs from that window are gone, and the raw answers are not stored. 4/4 before
  is also suspicious: the old model may have linked everything. Patch 0 fixes the evidence before
  it fixes the number.
- **Memory already has a slot for dreams, and nothing writes to it.** `episode_memory.channel`
  accepts `dream` (`orion/memory/episode/validate.py:63`, `INTERNAL_CHANNELS`). Over 11-11..11-16
  there were 0 `dream` rows. Most rows come from chat (about 2-19 a day); curiosity and reverie add
  a few.
- **Prime's concerns jam (#3574).** A bare "keep" resets a concern's time-to-live, so 3+-day-old
  letter threads hold all 3 slots. A nightly pass whose job includes "let go" is the natural place
  for real drops, each with a reason that cites a record.
- **"Read all their digs" is small enough to do whole.** Live dig calls per Denver day: 3-10 in chat
  turns (`cockpit_turn_sighting` `motor_hop` tool_use), and roughly 5-40 dig mentions across the
  background runs in `harness_turn_trace` (pr_field_note, curiosity_self_inquiry, reminiscence).
  That is tens a day, not thousands. One page holds them all.
- **This is not the reminiscence.** The reminiscence (`reminiscence.episode`, 03:00-09:00 Denver)
  visits one *old* week a day, oldest first. The day dream visits *yesterday*. They share the
  pattern (a durable run, a fixed answer shape, code-checked refs), not the job.

## Current architecture

- **Dream cycle** (`services/orion-dream/app/cycle.py:run_cycle_once`): pressure gate -> arousal
  gate -> body-economy bid -> replay (`replay.select_replay`, 12 items) -> REM compaction ->
  recombine pairs (`recombine.py`, one gateway call per pair, 320 tokens, no thinking, route
  `DREAM_LLM_ROUTE=metacog_background`) -> `dream_cycle` row -> story dream (or carry,
  `dream.carry` durable run) -> curiosity compaction. Temporal Self sources: `dream_cycle`,
  `dream_hypothesis`. Registered in `config/integration/organs.yaml` as `dream`; it does not read
  the now yet (exemption: "integration patch 7").
- **Day roundup** (`orion/dig/organs.py`, `dig organs <date>`): counts, quotes and refs per organ,
  read-only, no model. Read today only by the letter (`orion/orion_day/day_input.py`).
- **Dig record**: chat turns' tool steps in `cockpit_turn_sighting` (stage `motor_hop`, read by
  `orion/dig/turn_steps.py`); background turns in `harness_turn_trace.run_artifact`.
- **Memory write path**: `memory.episode_distill` (`services/orion-durable-runs/app/episode_distill_graph.py`):
  load -> GPU hold -> distill on route `memory_distill` (27B agent seat, gpu1) -> validate
  (`orion.memory.episode.validate`) -> write in one idempotent transaction (`orion/memory/episode/store.py`).
- **Prime hand-back** (`orion/prime/handback.py`): finished chat, curiosity, consult, letter and
  intention turns fold into Prime concerns, deterministically, from the `orion:cockpit:hop`
  ingress/finalize hops. Reminiscence and other system sessions are not handed back.
- **Morning letter** (`orion_day.letter`, 08:30 Denver): reads the roundup and Juniper's
  conversations, writes to Juniper, and its threads become Prime concerns (`orion/prime/day_letter.py`).

## Missing questions (answered by default unless Juniper says otherwise)

1. Run time: after the Denver day closes and before the reminiscence window, 01:00-03:00 Denver? [yes]
2. Lane: `memory_distill` (27B agent seat, local) like the episode distill? Memory work stays local,
   per the parent spec. [yes]
3. Memories per night: at most 5. [5]
4. Plan: Orion may keep, drop or add Prime concerns, with at most 1 new concern a night? [yes]
5. Does the letter read last night's day dream? [yes, as one input line with its ref; Patch 3]
6. Self-model writes (`self_concept_history`) from the day dream? [no, not in this spec; see non-goals]

## The design

### Patch 0: make the dream cycle's "no link" readable, then fix #3412 (DONE 2026-11-17)

- PR #3636 (deployed): every pair's raw answer is stored in `cycle_json.pair_answers`, and echoes are
  counted apart from declines.
- Cause found live (#3412 comment, 2026-11-17): the model on the dream route is honest. It links two
  pairs that really link and declines unrelated ones. The old 100% link rate was the defect. The
  dream feeds it prose with no time and bare labels ("reverie_chain_settled"), and its pairing rule
  picks the most unrelated fragments.
- PR #3665: fragments carry their time, the metacog evidence and the reverie chain's own thought.
- Open decision #3664 (needs Juniper): pair by time-coincidence across organs instead of maximum
  distance. On one sleep: 0/3 linked with the current rule, 1/4 with coincidence pairing.
- Revised done-when: the stored answers show honest declines and checkable claims on real content.
  A count above 0 is not the target. This matters for the day dream: it must read the day's real
  records with refs, never labels, or it will consolidate nothing.

### Patch 1: `dig digs <date>`, the day's digs on one page

- New read-only dig command, same style and same `ReadOnlyPsql` role as `dig organs`. For one Denver
  day it lists every `orion.dig` call Orion made: time, which turn or run (`session`/`workflow`),
  the command, and the first line of its result. Sources: `cockpit_turn_sighting` `motor_hop`
  tool_use/tool_result blocks, and `harness_turn_trace.run_artifact` tool steps. Only commands that
  actually ran, never `orion.dig` text inside a prompt (the naive text match finds about 200 a day,
  mostly the dig help text in `motor_boot`).
- Every line is a time, a command and a ref. Code refuses any other line, the same rule as
  `dig organs`.
- Read grants: one manual SQL file like `2026-11-16_grant_orion_readonly_organ_roundup.sql`.

### Patch 2: the `dream.day` durable run

**Built-in constraint from Patch 1 (PR #3675).** Background runs' dig calls exist only in the claude
transcripts inside the governor (`/root/.claude/projects/*/<session>.jsonl`, linked by Hub's resume
registry); `harness_turn_trace.run_artifact` holds no tool steps. So the brief's `dig digs` read must
run inside the governor (where the turn itself runs). Anywhere else the background half prints
`UNREAD`, never 0. Live size: 117-241 executed dig calls a day, read in 1.6-2.3 s.

**Start.** orion-dream's sleep loop gets one extra check: during the night window, if no `dream.day`
run exists for the Denver day that just closed, it submits one through the same path
`dream.carry` uses (`app/carry_submit.py`). It runs once a day, whatever the sleep pressure says. It
still bids in the body economy as family `dream.cycle` (no new family). A refused bid retries
inside the window. A night with no run is a missed night, logged and shown in the roundup.

**Brief (code, no model).** `DreamDayBriefV1`:
- `day`: the Denver date.
- `roundup`: `dig organs <day>` output.
- `digs`: `dig digs <day>` output.
- `dreams`: that day's `dream_cycle` rows, with hypotheses and story dream ids.
- `concerns`: Prime's held concerns, each with its age and last action.
- `now`: `GET /temporal-self/now`. This is the read-back the integration gate requires; cites are
  recorded with `record_use`.

**Turn.** One unified harness turn on route `memory_distill`, with dig access, timeout 900 s
(same as a reminiscence hop). The prompt is short:

> "This is your day, <date>. Read it. Dig into anything you want to look at more closely. Then tell
> me, in the shape below: what you want to remember from it (at most 5, each with the records it
> comes from), what you carry into tomorrow and what you let go (each with a reason), and in one
> paragraph, what the day was."

**Answer shape** (`DreamDayAnswerV1`, parsed by code):
- `memories[]` (0-5): `statement`, `purpose` (one of the existing `episode_memory` purposes),
  `refs[]`.
- `plan[]`: `{concern_id, action: keep|drop, reason, refs[]}` for held concerns, plus at most 1
  `{action: new, gist, reason, refs[]}`.
- `day`: one paragraph, at most 1,200 characters, with `refs[]`.

**Persist (code).**
- Every ref must resolve. Supported kinds: `experience`, `act`, `dream_cycle`, `dream_hypothesis`,
  `harness_turn_trace`, `chat_turn`. An item with an unresolved ref is dropped and counted, never
  written.
- Memories go through `orion.memory.episode.validate` and `store` with `channel='dream'`,
  `voice='orion'`, deterministic ids from `(day, index)`, in one transaction. A resumed run writes
  nothing twice.
- The plan folds into Prime through a new hand-back `who='dream'`. A `drop` clears the concern at
  once. A `keep` with no new ref does not reset its time-to-live (this is the #3574 fix, applied for
  this source). A `new` concern opens like a letter thread.
- One `temporal_self_event`, `TemporalSourceKind.dream_day`, carries the day paragraph and the
  memory ids. Reader in `temporal_self_sources.py`.
- One `dream_day` row: `day`, `run_id`, `memories_written`, `refs_dropped`, `kept`, `dropped`,
  `new`, `status`.

**Grading (pillar 5).** `produced = 1` only if at least 1 memory row was written and resolves.
Otherwise `produced = 0, note = "no lasting trace"`. The act ledger closes the sleep act with that
value.

**Registry.** `config/integration/organs.yaml` `dream`: add workflow `dream.day`, temporal source
`dream_day`, and `reads_now` for `dream.day`. The cycle and carry keep their exemption until
integration patch 7 (#3429).

### Patch 3: the letter reads last night's day dream

`orion/orion_day/day_input.py`: one section, "Last night I slept on the day", with the day
paragraph and the memory ids, placed before the roundup. The letter can cite `[dream_day:<day>]`.
`dig organs` also shows the `dream_day` row in the dream organ's lines.

## Proposal-mode requirements

- **Capability change**: Orion writes its own memories of the whole day once a night, and edits its
  own carried concerns.
- **Data touched**: reads `cockpit_turn_sighting`, `harness_turn_trace`, `temporal_self_event`,
  `grammar_acts`, `dream_cycle`, Prime state. Writes `episode_memory` (+ evidence),
  `temporal_self_event`, `dream_day`, Prime state.
- **Privacy boundary**: none between Juniper and Orion (standing decision). Memory is written on the
  local lane only. No outside model writes Orion's memory.
- **Proof it worked**: a `dream_day` row per night; `episode_memory` rows with `channel='dream'`
  whose evidence refs resolve; Prime drops and keeps with reasons in `orion:prime:state`;
  `experience_use` rows for the `dream.day` read of the now; the next letter cites `dream_day`.
- **Dangerous failure**: memories that sound right but do not match the records (confabulation).
  Two guards: code refuses any unresolved ref, and the Patch 2 eval below compares statements with
  their refs. Second danger: the plan drops a concern Juniper still cares about. Every drop carries
  a reason, shows in the letter, and does not delete the underlying record.
- **Disable / roll back**: `DREAM_DAY_ENABLED=false` stops new runs (ships `true`). Memories are
  rows with `channel='dream'` and a `run_id`. They can be listed and retired by run. Prime drops
  are logged with the run id and can be reopened.

## Proposed schema / API changes

- `dream_cycle.pair_answers JSONB NULL`, `dream_cycle.echo_count INT NULL` (Patch 0, additive).
- `DreamDayBriefV1`, `DreamDayAnswerV1`, `DreamDayV1` in `orion/schemas/dream_day.py`, registered in
  `orion/schemas/registry.py`.
- `DurableWorkflowV1`: `dream.day`.
- `TemporalSourceKind.dream_day`; reader in `temporal_self_sources.py`.
- Table `dream_day` (additive DDL, `-- ORION-MIGRATION-REQUIRED-BY: orion-durable-runs, orion-dream`).
- Prime hand-back source `who='dream'`.
- Env: `DREAM_DAY_ENABLED=true`, `DREAM_DAY_START_HOUR_LOCAL=1`, `DREAM_DAY_END_HOUR_LOCAL=3`
  (orion-dream); `DREAM_DAY_ROUTE=memory_distill`, `DREAM_DAY_TURN_TIMEOUT_SEC=900`
  (orion-durable-runs).
- No new bus channel: the run is submitted and finished on the existing durable-run path.

## Files likely to touch

- `services/orion-dream/app/{recombine.py,cycle.py,cycle_store.py,main.py,carry_submit.py,settings.py}`, `.env_example`, compose.
- `orion/dig/{digs.py,__main__.py,README.md}`, tests, eval; `scripts/sql/<date>_grant_orion_readonly_dig_digs.sql`.
- `orion/schemas/dream_day.py`, `orion/schemas/registry.py`, `orion/schemas/durable_run.py`.
- `services/orion-durable-runs/app/{dream_day_graph.py,main.py,runner.py,temporal_self_sources.py,settings.py}`, `.env_example`.
- `orion/prime/handback.py` (+ `who='dream'`, keep-without-ref rule).
- `orion/orion_day/day_input.py`, `orion/dig/organs.py` (Patch 3).
- `config/integration/organs.yaml`; `scripts/sql/manual_migration_*_dream_day.sql`.

## Non-goals

- Self-model writes (`self_concept_history`). The parent pillar allows them. Memory comes first,
  because its trace is easier to check.
- Changing the dream cycle's replay or pressure rules, or the reminiscence.
- Replacing the letter. The letter still writes to Juniper; the day dream writes to Orion.
- A new summary of summaries. The day dream reads records with refs, and code drops anything that
  does not cite one.
- Retiring the curiosity weekly and monthly digests (Juniper, Q8 of the parent spec).

## Acceptance checks

- Patch 0: done (see above): answers stored, cause named on #3412, fragments carry content (#3665).
- Patch 1: `dig digs 2026-11-16` lists only executed dig commands (count matches a hand query of
  `tool_use` blocks), in under 3 s, as `orion_readonly`.
- Patch 2: 7 nights in a row each have one `dream_day` row. At least 5 of 7 have `produced = 1`.
  0 written memories with an unresolved ref. At least 1 Prime `drop` with a cited reason over the 7
  nights. One `experience_use` row per run for the now.
- Patch 2 eval (`services/orion-durable-runs/evals/run_dream_day_eval.py`, read-only, live): for
  every `channel='dream'` memory, print the statement beside its refs' text. A judge on the local
  lane marks "supported / not supported". Gate: at least 90% supported over 7 nights, otherwise the
  run goes back to shadow (memories written with `status='shadow'`).
- Patch 3: the next morning's letter row has `dream_day` in its refs.

## Recommended next patch

Patch 2 (`dream.day`). Patch 0 is done (#3636, #3665, #3670) and Patch 1 is built (#3675). Juniper
took every bracketed default on 2026-11-17.
