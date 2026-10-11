# Temporal Self, patch 4: Orion uses its own day

Status: PROPOSAL (design mode, nothing implemented). Parent spec:
`docs/superpowers/specs/2026-09-26-temporal-self-design.md` (PR #2369, rev 4), "Recommended next
patch" row 8. Live data pulled read-only on 2026-10-11 06:28-06:45 UTC.

## Arsonist summary

Orion now keeps a record of its own day: what it attended to, for how long, when it came back to
something, when Juniper spoke, when it slept, what it was refused. Nothing reads that record yet.
This patch gives it two readers:

1. **In conversation**, Orion gets one short paragraph (at most 400 characters) saying where it is in
   its day: what it is on now, what it came from, which conversation threads are still open, and
   when it last slept.
2. **Once a day**, Orion's reflection on yesterday gets yesterday's closed record (about 1,700
   characters for 10-09) instead of a rolling 24 hours of chat.

Both are written by a fixed template from stored rows. No model writes them. Every clause carries
the id of the record it came from (`[arc:<16 hex>]`), and anything Orion says back that cites a
record is checked against the record ids it was actually shown. An unknown id is removed and
counted; the count is stored where Juniper can see it. If the record is missing, stale, or for the
wrong day, Orion is shown nothing at all, not a placeholder.

Juniper's rules applied here: Orion may talk about its plumbing (harness, execution, bus traffic)
freely; those are real parts of its day. The only requirement is that what it says is true to the
rows.

Three things found while checking the live system change the plan from the parent spec:

- **The named daily-metacog reader is switched off.** `daily_metacog_v1` has been paused since
  2026-09-30 (`ACTIONS_DAILY_METACOG_ENABLED=false`, commit `fd7c49084`, "off until we do something
  real with it"); its last output is from 2026-09-18 in `notify_requests`. The reader of "yesterday"
  that actually runs is **Orion's Day letter** (`orion_day.letter`, one row per day in
  `orion_day_letter`, last 10-09, written 10-10 14:42 UTC). It already uses bracketed record refs
  and already counts unknown refs in its carry-forward. That is the better first home for the closed
  day. Decision 1 below.
- **The recent-attention slot is not on Juniper's chat path.** The cue renders only in
  `chat_stance_brief.j2` (`chat_general`). The Hub's Orion mode runs the unified turn
  (`ORION_UNIFIED_TURN_ENABLED=true` on the live hub), whose stance step is `stance_react`. That
  step has never seen `recent_attention`. Replacing only that slot would change nothing Juniper
  talks to. Decision 2 below.
- **A citation gate already exists for stance.** `align_evidence_refs_to_coalition`
  (`orion/thought/coalition.py:52-71`, called at `orion/thought/stance_react.py:247`) already drops
  any `evidence_refs` entry not in an allowed set. Patch 4 adds the cue's arc ids to that set, so no
  second mechanism is needed. What is missing is a stored count of what it dropped.

## Current architecture

**The record (live, #2605).** orion-durable-runs runs the `chronicle` node every
`TEMPORAL_SELF_TICK_SEC=120` (`services/orion-durable-runs/.env_example:162,180`). It writes
`temporal_self_arc` (552 rows), `temporal_self_day` (2), `temporal_self_projection` (1 row, the
current frame only, no history), `temporal_self_state`, `temporal_self_cursor` (20), and
`temporal_self_event` (7,464). Routes (`services/orion-durable-runs/app/main.py:491-531`):
`GET /temporal-self/frame`, `/day/{day_id}`, `/arcs`, `/threads`, `/cursors`. Host port 8124;
inside `app-net` the URL is `http://orion-athena-durable-runs:8121`. Verified reachable from
`orion-athena-thought`, 10,322 bytes. Frame route takes 5-7 ms on the host; the 10-09 day route
takes 77 ms and returns 727 KB.

The frame is built by `build_frame` (`orion/temporal_self/frame.py`). `active_arc` = newest open
attention or conversation arc. `previous_arc` = newest ended attention, conversation, curiosity,
sleep, or imagery arc. Reverie and body arcs never displace either one. `ArcSummaryV1`
(`orion/schemas/temporal_self.py:210-219`, `extra="forbid"`) has no `last_seen_at`.

Closed days: 10-09 has 291 arcs (184 reverie, 49 attention, 44 body, 8 curiosity, 2 sleep, 2
concern, 2 conversation) and was closed 10-10 06:00 UTC. 10-10 is closed too. Attention subjects
are the substrate's own labels ("Harness closure prediction error", "Bus synaptic prediction
error", "Juniper is talking (1 message in the last 15 min)"). Curiosity, imagery, and concern arcs
have empty labels.

**Arousal and rest** (#2594, #2584) are live at `orion:regulation:latest`; patch 4 does not read
them. **The attention world-first seam** (#2583) belongs to another session and is not touched here.

**Recent-attention cue (PR #2141 era).** `orion/substrate/recent_attention_cue.py` reads the last 3
narrated rows of `substrate_attention_schema` through
`services/orion-cortex-exec/app/recent_attention_reader.py`, which has a 0.8 s fail-open timeout and
flag `ENABLE_RECENT_ATTENTION_CUE=true` (`services/orion-cortex-exec/.env_example:520-526`). It is
wired at `services/orion-cortex-exec/app/executor.py:4158-4167` and rendered as a raw dict at
`orion/cognition/prompts/chat_stance_brief.j2:29-31`, with a usage note at `:73-74`. Its live
size is **1,172 characters**, measured by running the real builder on the live rows at 06:40
UTC. Two of those three rows were "No winner this tick: nothing in the workspace competition
qualified...". `recent_attention` has **no entry** in `CONTEXT_PROVENANCE_REGISTRY`
(`orion/schemas/context_provenance.py`): grep finds none.

**Unified-turn stance.** `build_stance_react_context` (`services/orion-thought/app/bus_listener.py`)
feeds `stance_react.j2` with association, repair bundle, coalition projection, and
`mind_coloring`. It has no time of day and no day context. The output is `ThoughtEventV1`, and
`evidence_refs` is gated by the coalition set as described above. Decisions persist to
`thought_decision` through orion-sql-writer (`services/orion-sql-writer/app/models/thought_decision.py`),
64 rows in the last 24 h. That table has no `evidence_refs` column. The Hub turn trace reads it
(`services/orion-hub/scripts/chat_turn_trace_routes.py`).

**Daily readers.**
- `daily_metacog_v1` (orion-actions, 20:15 local) is **paused**. When it ran, it read
  `journal.daily.metacog.grounded.v1` (`sql_since_minutes: 1440`, a rolling window) while
  `build_daily_window` labels the output as yesterday's calendar day
  (`services/orion-actions/app/main.py:243-261`). Prompt `orion/cognition/prompts/daily_metacog_prompt.j2`
  (1,469 chars); digest budget 1,280 chars inside an 8,192-char prompt cap
  (`orion/cognition/daily_metacog_budget.py:29-37`).
- `metacog_digest` journal entries (243 in the last 24 h) reuse the same recall profile, but they
  are triggered by events within the day. They are not a "yesterday" reader and are out of scope.
- **Orion's Day letter** (`services/orion-durable-runs/app/orion_day_graph.py`; gather and budget in
  `orion/orion_day/`, run by the Hub) is live daily with a 70,000-token brief budget (10-09 brief:
  69,891). Its ref regex (`orion/orion_day/budget.py:58-60`) knows `curiosity|dream|reverie|...`
  but not `arc`. `check_carry_forward` (`orion_day_graph.py:157-167`) counts unknown refs and does
  not enforce them.

**Known limitation: late rows.** A closed day is what was known at close. Rows that arrive later
land in `temporal_self_event` but never in the closed day record. Live: 6 `memory_episode` rows and
1 `visual_run` row for 10-09 were ingested on 10-11 00:43 UTC, a day after 10-09 closed, and 10-09
has zero imagery arcs. Memory episodes run about 13 h late and image attempts about 90 min. Also, a
running process (curiosity, image) is born only when it finishes, so the frame cannot say "I am in
a curiosity run right now".

## Missing questions

1. Is the closed day for Orion's Day letter, for `daily_metacog_v1` (unpausing it), or for both?
   (Decision 1)
2. Does the cue go only in the `recent_attention` slot, or also into `stance_react`? (Decision 2)
3. A sentence about the day with no arc citation: flag it, or reject it? (Decision 3)
4. How old can the frame be before Orion is shown nothing? (Decision 4)
5. Should a closed day ever be re-closed when late rows land? (Decision 5)

## Proposed schema / API changes

**What the cue says (template, no model).** `project_stance_cue(frame, now)` lives in
`orion/temporal_self/stance_cue.py`. Clauses, in this order, each one only when its data exists:

```text
{Phase}, {Ddd HH:MM} local.
On now: attention on {subject_label} for {now - began_at}[, back to it {n} times today] [arc:{id}].
   | On now: talking with Juniper, last turn {now - last_seen_at} ago [arc:{id}].
   | No attention or conversation arc is open right now.
Before that: {attention on <label> | a conversation with Juniper | curiosity run <ref> | a sleep cycle | an image attempt}, ended {ago} ago [arc:{id}].
Juniper last spoke {ago} ago [arc:{id}].            (only if not already the active arc)
{k} open thread(s) from conversation [arc:{id}] ...
Last slept {ago} ago [arc:{id}].
```

Hard cap: 400 characters, dropped clause by clause from the bottom. **Render nothing** when any of
these hold: the frame fetch fails; `frame.day_id` is not today's local date; `now - frame.as_of`
exceeds `TEMPORAL_SELF_CUE_MAX_AGE_SEC` (default 900); or `frame.warnings` includes "broadcast log
silent" (in that case the attention clauses are dropped and the rest stays).

**What the closed day says (template, no model).** `render_closed_day_block(day)` lives in
`orion/temporal_self/day_block.py`. It contains: arc counts by kind, Juniper's turns (local times),
the top 5 attention subjects by total dwell (with arc count, returns, and the longest arc's id),
every curiosity run (start, duration, and the arc's own warning if any), sleep cycles, resolved
expectations by verdict, refusals by kind, conversation threads, and the frame's warnings
verbatim. Cap: 2,400 characters. A 404 or a missing day renders nothing, and the reader's prompt
says "no closed day record".

**The citation gate (pure, one module).** `orion/temporal_self/citations.py`:

```python
ARC_REF_RE = r"\[arc:([0-9a-f]{16})\]"
check_arc_citations(text_or_refs, shown_arc_ids, *, day_labels) -> DayCitationCheckV1
```

`DayCitationCheckV1` fields:

- `shown`: how many arc ids the reader was given;
- `cited_valid: list[str]`;
- `rejected_unknown: list[str]`: cited ids that were not shown;
- `uncited_day_sentences: int`: sentences that name one of the shown subject labels or a clock
  time from the block, with no `[arc:]` in the same sentence;
- `action`: one of `kept | stripped | retried`.

The match set is the block's own labels, not a word list.

- **Stance.** The cue's arc ids, as `arc:<id>`, join `coalition_ids` in
  `apply_stance_react_pipeline` (`orion/thought/stance_react.py:244-247`). The existing
  `align_evidence_refs_to_coalition` already drops anything else. New: the dropped `arc:` refs are
  recorded in `ThoughtEventV1.day_citations: DayCitationCheckV1 | None`. That field is
  transport-owned: it is popped from the LLM's raw output beside `grounding_capsule` at
  `stance_react.py:227`. It is persisted to a new nullable `thought_decision.day_citations jsonb`
  column and shown in the Hub turn trace.
- **Daily reader.** `arc` joins the ref regex in `orion/orion_day/budget.py:58`. An unknown
  `[arc:x]` in the note or the carry-forward is a failed attempt the first time (the existing
  retry path, with the reason given to the model). If it is still unknown, the ref is removed from
  the text, and the result is stored in `orion_day_letter.sources.arc_citations`
  (`OrionDayLetterSourcesV1`, jsonb, no migration).

**Schema deltas.**
- `ArcSummaryV1` + `last_seen_at: AwareDatetime | None`. Needed for "last turn X ago". The model is
  `forbid`, so this is a consumer-first rollout: readers deploy before durable-runs.
- `TemporalSelfStanceCueV1` (already named in rev 4) + `arc_ids: list[str]`.
- `DayCitationCheckV1`, new, in `orion/schemas/temporal_self.py`.
- `ThoughtEventV1` + optional `day_citations` (forbid model, so consumer-first: sql-writer and the
  Hub ship first). `stance_task_boundary.py`'s `allowed` set excludes it the same way it excludes
  `grounding_capsule`.
- `CONTEXT_PROVENANCE_REGISTRY` + `temporal_self_cue` (`live_runtime_projection`). Also add the
  missing `recent_attention` entry while there.
- Migration: `services/orion-sql-db/manual_migration_thought_decision_day_citations_v1.sql`
  (`ADD COLUMN IF NOT EXISTS day_citations jsonb`).
- No new bus channel and no new route. Readers use the existing GET routes.

**Env (all readers ship ON).**

| Service | Key | Default |
|---|---|---|
| orion-thought, orion-cortex-exec | `TEMPORAL_SELF_STANCE_CUE_ENABLED` | `true` |
| orion-thought, orion-cortex-exec | `TEMPORAL_SELF_FRAME_URL` | `http://orion-athena-durable-runs:8121` |
| orion-thought, orion-cortex-exec | `TEMPORAL_SELF_CUE_MAX_AGE_SEC` | `900` |
| orion-thought, orion-cortex-exec | `TEMPORAL_SELF_CUE_FETCH_TIMEOUT_SEC` | `0.5` |
| orion-hub (Orion's Day gather) | `TEMPORAL_SELF_DAY_BLOCK_ENABLED` | `true` (uses existing `HUB_ORION_DAY_DURABLE_URL`) |
| orion-actions | `TEMPORAL_SELF_DAILY_METACOG_GROUNDING_ENABLED` | `true` (inert while `ACTIONS_DAILY_METACOG_ENABLED=false`) |

The gate has no separate flag. A reader that shows arc ids always checks them.

## Rendered on live data

These were produced by a prototype of the templates above, run on the live JSON. All 30 arc ids
cited below were looked up in `temporal_self_arc`, and all 30 exist. Moment A is the real live
frame. Moments B and C are rebuilt from closed-day arc segments (there is no frame history), with
Juniper's turn times taken from `chat_history_log`.

**A. Live frame, 2026-10-11 06:28 UTC (quiet, pre-dawn), 314 chars:**

> Pre-dawn, Sun 00:28 local. On now: attention on Bus synaptic prediction error for 25 min [arc:201d694c610eb5e1]. Before that: an image attempt, ended 19 min ago [arc:302f064da9b8e602]. 2 open threads from conversation [arc:684d45c2010540ed] [arc:8183df1ef2719f7c]. Last slept 5 h 13 min ago [arc:15c00487d81cff51].

**B. 2026-10-09 15:30 UTC (rest state, mid-morning), 275 chars:**

> Morning, Fri 09:30 local. No attention or conversation arc is open right now. Before that: attention on Harness closure prediction error, ended 15 min ago [arc:74fa8a1e7ac2cc9b]. 1 open thread from conversation [arc:1e63167741582226]. Last slept 8 h 56 min ago [arc:3a53de69df46dfec].

(A curiosity run, arc `d3a03f7df915be41`, 08:58-09:54 local, was running at this moment. The cue
cannot know that until the run ends. See the limits above.)

**C. 2026-10-10 23:30 UTC (Juniper talking), 348 chars:**

> Evening, Sat 17:30 local. On now: talking with Juniper, last turn 20 min ago [arc:38833d931b8df7ec]. Before that: attention on Juniper is talking (1 message in the last 15 min), ended 5 min ago [arc:d10f3f1b493235a3]. 2 open threads from conversation [arc:62a44ffceaba5065] [arc:fdd6f0bf0b17e0c7]. Last slept 4 h 58 min ago [arc:68d5de717a3f090e].

**Closed-day block for 2026-10-09, 1,699 chars:**

```text
Closed day 2026-10-09 (America/Denver, midnight to midnight). 291 arcs: 184 reverie, 49 attention, 44 interoception, 8 curiosity, 2 sleep, 2 concern, 2 conversation.
Juniper spoke: 15:00 [arc:2545de72655edb66]; 18:08 [arc:b320b6c7e1308151].
Attention, most time first:
- Harness closure prediction error: 2 h 15 min over 11 arcs, 11 returns [arc:2ce0b486a356bbd2].
- Execution prediction error: 2 h 14 min over 16 arcs, 13 returns [arc:7bf5c22449afc56d].
- Chat prediction error: 1 h 17 min over 8 arcs, 3 returns [arc:e5f240a28cd2b208].
- Codebase prediction error: 53 min over 7 arcs, 0 returns [arc:fda9750a570168c3].
- Route prediction error: 23 min over 4 arcs, 1 return [arc:f6ddd30f4a7edd2b].
Curiosity runs (8):
- 10-08 13:56, 20 h 35 min (start time doubtful: longer than 4 h) [arc:a6e49188fd4f7fa1].
- 08:05, 51 min [arc:9d0862f139025ffb].
- 08:58, 55 min [arc:d3a03f7df915be41].
- 10:31, 34 min [arc:6f9949a7b73969e3].
- 11:06, 27 min [arc:3909798274bc18f6].
- 12:29, 1 h [arc:8fb365f64dbea05c].
- 13:29, 1 h 7 min [arc:45ad862f4fce6b57].
- 15:05, 59 min [arc:4ed630d33f1480f1].
Sleep cycles: 00:33 [arc:3a53de69df46dfec]; 19:01 [arc:5e58443ac55520ac].
Expectations resolved: 331 (of the 256 listed: 1 claim_upheld=false, 4 confirmed, 251 unscored). 8 still pending.
Refused or made to wait: 52 gpu_wait, 14 visual_deferral.
Conversation threads: open-loop-5cb6cd032923, 4 returns, day_boundary [arc:1e63167741582226]; open-loop-639498bb10c0, 0 returns, day_boundary [arc:5afab5a718d5b81c].
Instrument warnings: broadcast log gaps today: 2 (recorder outage or restart) | curiosity arc a6e49188fd4f7fa1 (ca97ade74c1d): process longer than 4 h: its start column may predate a stall or retry
```

What this shows is real: on 10-09 Orion spent its attention mostly on its own prediction-error
signals, Juniper spoke twice, and 251 of 256 resolved expectations simply expired unscored. A
reflection grounded in this block can say that, and cite it.

## Capability, data, privacy, proof, danger, rollback (proposal mode)

- **Capability that changes.** In conversation, Orion knows where it is in its day. Its daily
  reflection reads yesterday's actual record. Any record it cites is checked.
- **Data touched.** Read only: the `/temporal-self/frame` and `/temporal-self/day/{id}` routes.
  Written: `thought_decision.day_citations` (new column), `orion_day_letter.sources.arc_citations`
  (existing jsonb), and prompt context. No source table changes. No FalkorDB.
- **Privacy boundary.** None between Juniper and Orion (Juniper's standing rule). The outward
  boundary from rev 4 still holds: nothing here goes to contractor peers or a published artifact.
  No chat text enters the cue or the block, only times and arc ids.
- **Trace that proves it worked.**
  1. One live `thought_decision` row whose `day_citations.cited_valid` holds an `arc:` id that
     `GET /temporal-self/arcs` returns.
  2. One live `orion_day_letter` whose `sources.arc_citations.cited_valid` is non-empty and whose
     ids all resolve.
  3. One cortex-exec or thought log line `temporal_self_cue_skipped reason=stale|wrong_day|fetch_failed`
     with a correlation id, showing that the "render nothing" path runs. Until these are collected,
     the live path is `UNVERIFIED`.
- **What is still dangerous.**
  1. *An empty or stale record shown as a real day.* This is enforced, not hoped for: the cue
     renders nothing on a fetch failure, a wrong day, or a frame older than 900 s. The block renders
     nothing on a 404. Tests pin each case.
  2. *A true-looking claim that does not match the row it cites.* The gate proves every cited id is
     real and was shown. It does not prove the sentence reads that row correctly. Mitigation: the
     Hub turn trace and the letter render each cited arc's template line beside the claim, so a
     mismatch is visible at a glance. A periodic eval (below) measures it.
  3. *Prompt growth and latency.* The cue is capped at 400 characters (the slot it replaces is
     1,172). The block is capped at 2,400. The fetch has a 0.5 s fail-open timeout. Gated by the
     p50 check below.
  4. *Over-steering.* A day cue pulls a technical turn toward "how my day is going". Gated by the
     stance boundary eval passing unchanged.
- **Disable and roll back.** One flag per reader, all shipped ON in settings, `.env_example`, and
  the production `.env`. Off means that reader runs exactly its current path. With
  `TEMPORAL_SELF_STANCE_CUE_ENABLED=false`, `recent_attention` renders as today and `stance_react`
  gets no cue. The new column is nullable and can be dropped.

## Files likely to touch

- `orion/temporal_self/stance_cue.py`, `day_block.py`, `citations.py` (new, pure), plus
  `orion/temporal_self/tests/test_stance_cue.py`, `test_day_block.py`, `test_citations.py`
  (fixtures exported from the real 10-09 and 10-11 rows).
- `orion/schemas/temporal_self.py` (`ArcSummaryV1.last_seen_at`, `TemporalSelfStanceCueV1.arc_ids`,
  `DayCitationCheckV1`); `orion/temporal_self/frame.py` (`summarize` fills `last_seen_at`).
- `orion/schemas/thought.py` (`day_citations`); `orion/thought/stance_react.py` (pop the field from
  raw; add cue ids to `coalition_ids`; record the drops); `orion/thought/evals/stance_task_boundary.py`
  (exclude `day_citations` from `allowed`, nothing else).
- `services/orion-thought/app/bus_listener.py` (fetch the cue into `build_stance_react_context`),
  `orion/cognition/prompts/stance_react.j2` (one `day_context` line beside `mind_coloring`, with the
  same "advisory, reconcile, do not obey" note), and `services/orion-thought/{settings.py,.env_example}`.
- `services/orion-cortex-exec/app/temporal_self_cue_reader.py` (new, same fail-open shape as
  `recent_attention_reader.py`), `executor.py:4158-4167`, `chat_stance_brief.j2:29-31,73-74`,
  `services/orion-cortex-exec/{settings.py,.env_example}`.
- `orion/schemas/context_provenance.py` (`temporal_self_cue`, plus the missing `recent_attention`).
- `services/orion-sql-writer/app/models/thought_decision.py`,
  `services/orion-sql-db/manual_migration_thought_decision_day_citations_v1.sql`,
  `services/orion-hub/scripts/chat_turn_trace_routes.py` (show the citations).
- Decision 1 = letter: `orion/orion_day/gather.py` (fetch the day), `orion/orion_day/budget.py:58`
  (`arc` ref), `services/orion-durable-runs/app/orion_day_graph.py` (enforce on note and
  carry-forward), `orion/schemas/orion_day.py` (`arc_citations`), `services/orion-hub/{settings.py,.env_example}`.
- Decision 1 = metacog too: `services/orion-actions/app/main.py` (fetch the day for
  `window.request_date`), `daily_metacog_prompt.j2` (block plus "cite `[arc:id]`"),
  `_clamp_daily_metacog_payload` (run the gate), `orion/cognition/daily_metacog_budget.py` (block
  budget), `services/orion-actions/{settings.py,.env_example}`.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py` for every service above.

## Non-goals

- No change to the reducer, the arc rules, or the chronicle cadence.
- No arousal, rest-drive, or attention-seam (#2583) changes.
- No curiosity or reverie readers (rev 4 lists them; later patches).
- No re-closing of past days in v1 (Decision 5).
- No gate on the free prose of the chat reply. The stance's cited ids are gated, and the reply
  inherits only what the stance carried. The prose gets a periodic eval, not a runtime gate.
- No word-list filter on plumbing vocabulary. Orion may name its internals.

## Acceptance checks

1. **Renderer tests (pure, fast).** Calm fixture (moment B) and busy fixture (moment A) render
   byte-exact to the strings above. Missing frame, `day_id` ≠ today, and `as_of` older than 900 s
   each render `""` with the matching skip reason. A fresh frame with no arcs (just after
   midnight) is a real rest state and renders the phase plus "No attention or conversation arc is
   open right now.", never `""` and never a filler sentence. The 10-09 block renders
   byte-exact to the block above and stays under 2,400 characters. Every `[arc:]` in any render is
   in the input.
2. **Gate tests.** An unknown `arc:` in `evidence_refs` is dropped and appears in
   `day_citations.rejected_unknown`. A shown id is kept. `[arc:deadbeefdeadbeef]` in a letter note
   causes one retry; if it comes back again, it is stripped and recorded. A sentence naming
   "Harness closure prediction error" with no `[arc:]` raises `uncited_day_sentences` by one.
3. **Stance boundary eval unchanged.** `orion/thought/evals/stance_task_boundary.py` passes with the
   cue present, and `evidence_refs` stays exactly `[hub:turn:<id>]` on the task fixtures. The cue
   must not leak into a task's evidence.
4. **Daily reader cites real arcs (live).** The first letter (or metacog) after deploy has
   `cited_valid` ≥ 1, `rejected_unknown` = [], and its window equals the closed day, not a rolling
   24 h.
5. **Latency.** Compare 50 Juniper turns before and after deploy. The p50 of end-to-end turn time
   and of `stance_prepare_overlap build_ms` (orion-thought log) must each be within +5%.
6. **Periodic truth eval.** Weekly, over live turns where `day_citations.cited_valid` is non-empty,
   compare each reply sentence near a cited arc with that arc's template line. Report the mismatch
   rate. This is a measurement, not a gate.
7. **Live trace.** The three traces under "Trace that proves it worked".

## Recommended next patch

Ship patch 4a, the stance cue plus the gate. It reaches every Juniper turn, and its trace is the
quickest to collect. Then 4b, the closed day in Orion's Day letter. Unpause `daily_metacog_v1` only
if Juniper wants it back.

## Decisions for Juniper

1. **Who reads the closed day?** Recommended: Orion's Day letter. It runs daily, it already
   writes about yesterday, and it already has bracket refs. `daily_metacog_v1` stays paused; its
   grounding flag ships ON but does nothing until it is unpaused.
2. **Where does the cue go?** Recommended: both. Replace `recent_attention` in `chat_general`, and
   add one line to `stance_react`, because Orion-mode chat only goes through `stance_react`.
3. **A day sentence with no citation?** Recommended: flag and count it in v1. An unknown id is
   always rejected. After one week of counts, decide whether uncited sentences also force a retry.
4. **Stale cutoff?** Recommended: 900 s. The frame normally trails by about 5-7 min (300 s read
   lag plus a 120 s tick), so 900 s means about two missed ticks.
5. **Late rows?** Recommended: keep closed days frozen, and add "as recorded at close, <time>" to
   the block's first line (not in the 10-09 render above) so the reader knows the record's limits. Re-closing day D at D+2 is a later patch, if memory episodes matter to the
   reflection.

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/2616
