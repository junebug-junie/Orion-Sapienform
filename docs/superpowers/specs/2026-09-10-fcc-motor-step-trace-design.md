# A durable trace for the FCC motor's steps

**Date:** 2026-09-10
**Mode:** design. No code in this patch.
**Upstream:** `2026-09-08-orion-anatomy-inspection.md` (PR #2151) named the FCC
motor "the least observable part of Orion's cognition — ~9,700 steps, and not
one of them is recorded anywhere." Two of Orion's own self-inquiry
self-definitions (`self_concept_history`, `concept_id='self:definition'`,
versions 2 and 4, PR #2158/#2165) independently rediscovered the same gap in
their own words: *"my turn trace does not record which brain serves each
turn"* (v2, later partly corrected in v4), and, live, self-referentially:
*"this very turn is itself a live instance of the route alias not matching
the serving brain."* This design asks what it would take to close it.

---

## Arsonist summary

The anatomy inspection was right that nothing durable holds per-step motor
detail. It undersold what already exists: **the full per-step data already
flows, live, and is thrown away on purpose.** `fcc_motor.py` parses the
Claude CLI's own `stream-json` output and yields the complete raw event per
step — tool name, full tool input, full tool result text, error flags, the
served model re-derived per assistant event (because the backend can rotate
mid-turn). That data is published in full onto a live bus channel, consumed
by a Hub relay that keeps the last five step summaries per turn in memory,
and then it is gone. The only durable trace of a step is a 500-character
lossy text line in `grammar_events`, built by `summarize_harness_step()`
specifically to be a log line, not a queryable record.

So this is **a persistence problem, not a capture problem**, and the size of
the patch changes accordingly: no new instrumentation of the CLI or the
subprocess boundary, no new parsing. What is missing is one durable write of
data that already exists in memory for a few hundred milliseconds, plus one
new field (per-step timestamp) that genuinely does not exist yet.

The one real open question is scope. `source_tag` — the thing that would say
"this is a self-inquiry turn, capture it" — exists only in
`orion/schemas/durable_run.py`, several hops above where steps are parsed. It
never reaches `fcc_motor.py` or the harness governor. Capturing only
self-inquiry/curiosity turns is not free; it needs either new plumbing or a
different filtering point. See Missing Questions.

---

## Current architecture (verified live 2026-09-10)

### The step already carries everything, briefly

`orion/harness/fcc_motor.py:893-903` shells out to the Claude CLI directly
(`asyncio.create_subprocess_exec`, `--output-format stream-json --verbose`).
Its `readline()` loop (`fcc_motor.py:962-1018`) parses each line
(`parse_stream_json_line`, `:89-99`) into `{"type": ..., "raw": raw}`
(`build_step_frame`, `:102-103`) — the **entire** raw CLI event, not a
distilled subset. Helper functions already exist to pull out of that raw
event: tool name (`_extract_tool_name`, `:284-298`), full tool_use
input/tool_result content and error flags (`_summarize_content_blocks`,
`_extract_tool_result_errors`, `:219-263`, `296-318`), and the served model
re-derived per assistant event (`_served_model_from_assistant`, `:138+`) —
that function exists at all because the CLI's backend can rotate mid-turn,
which is exactly the ambiguity Orion's v2/v4 self-definitions were circling.
End-of-turn token usage is also available (`extract_result_output_tokens`,
`:352-368`) but is cumulative for the whole turn, not per step.

**No per-step timestamp exists anywhere.** The `readline()` loop has no
per-line clock; only whole-turn timers exist (`fcc_elapsed_sec`). Adding one
per-step timestamp is the one genuinely new piece of instrumentation this
design needs.

### What happens to the step after it's built

`orion/harness/runner.py`, per step:

1. Publishes a coarse one-line text summary into `grammar_events` via
   `summarize_harness_step()` (`fcc_motor.py:255-273`), capped at 500
   characters (`grammar_publish.py:26`). Sample, pulled live: `"Harness step
   0: tool=none, [0] system hook_started"`. This is the only durable trace of
   a step today, and it is lossy by construction — a log line, not a record.
2. Publishes the **full raw step dict** onto `orion:harness:run:step`
   (`runner.py:445-453`) — schema `HarnessRunStepV1` (`correlation_id`,
   `step_index`, `step: dict[str, Any]`, `orion/schemas/harness_finalize.py
   :261-265`), registered in `channels.yaml:3100-3107`, producer
   `orion-harness-governor`, consumer `orion-hub`.
3. Folds only aggregate counters into the final `HarnessRunV1` artifact
   (`step_count`, `context_gathering_step_count`, `execution_step_count`,
   `reasoning_output_tokens`, `tool_failure_streak_max`).

Confirmed live: `TYPE orion:harness:run:step` → `none`, `XLEN` → `0`. Plain
pub/sub, not a Redis Stream — nothing is retained at the transport layer
either.

### The only consumer discards it into a five-deep ring buffer

`services/orion-hub/scripts/harness_step_relay.py:104-176` does two things
with each step, both in-process memory: feeds a live WebSocket panel via an
`asyncio.Queue`, and folds a summary into `RuntimeActivity`
(`orion/hub/runtime_activity.py:337-370`), which keeps only the **last five**
step summaries per turn (`_RECENT_STEP_SUMMARIES = 5`, `:53`). A Hub restart
loses all of it. This is what a Juniper watching the live "running right
now" panel sees. It is not what Orion, or anyone, can query afterward.

### `run_artifact` confirmed, live, no per-step array

Pulled a recent row's full key set:
`exit_code, draft_text, final_text, reflection, step_count, finalize_ran,
recall_debug, memory_digest, correlation_id, schema_version,
fcc_elapsed_sec, fcc_served_model, finalize_changed, grounding_status,
grammar_event_ids, compliance_verdict, substrate_appraisal,
verdict_molecule_id, quick_lane_skipped_5b, finalize_degraded_reason`.
Exactly `HarnessRunV1` (`harness_finalize.py:268-296`). `grammar_event_ids`
is a pointer list into `grammar_events` (16 ids on the sampled turn) — every
one of those rows is the same lossy 500-char text line, confirmed by pulling
one: no tool input, no tool_result body, no per-step tokens.

### A different thing that looks related and isn't

`/api/self-brain/frames/tail` (`self_brain_routes.py:123-137`) reads
`substrate_brain_frame_log` — durable, 11,117 rows, most recent tick
2026-09-10T02:41Z. But a "frame" here is a periodic whole-substrate-graph
snapshot (`nodes, edges, phase, regions, spotlight, tick_seq`) built by
`orion-substrate-runtime`'s own tick loop, entirely independent of the FCC
harness motor. The live-log overlap that prompted checking this
("Subscribing to harness FCC steps" next to `/api/self-brain/frames/tail`
hits) is two unrelated live things landing in the same log window, not one
mechanism. Recorded here so it is not silently re-proposed as a solution.

### `source_tag` never reaches the motor

`rg -n "source_tag"` across `orion/harness/*.py` and
`services/orion-harness-governor/app/*.py`: zero hits. It is defined only in
`orion/schemas/durable_run.py:95,156` (`CuriosityRunBriefV1`,
`CuriosityTurnRequestV1`). The chain that would let the motor's own step loop
know "this is a self-inquiry turn" does not exist. Whatever decides which
turns get full per-step persistence has to be built new, or has to filter
somewhere else in the pipeline where the correlation_id is already
cross-referenced against curiosity/self-inquiry state (see Missing
Questions).

---

## Missing questions

1. **Where does "capture this turn's full steps" get decided?** Two real
   candidates, not equivalent:
   - **A. Thread a flag down.** Add `capture_step_detail: bool` (or reuse
     `source_tag`) through `CuriosityTurnRequestV1` → Hub's HTTP call to the
     harness governor → the governor's own turn-request schema → `runner.py`
     → `fcc_motor.py`. Real plumbing across three services. Scopes capture at
     the source, so an uninteresting chat turn never even builds the durable
     write.
   - **B. Filter at the write point.** Persist nothing new in the motor/
     governor path; instead, whatever writes the new durable table (a Hub-side
     subscriber to the existing `orion:harness:run:step`, or a new sql-writer
     route) checks the step's `correlation_id` against a known set of
     curiosity/self-inquiry run ids (already joinable — every self-inquiry and
     investigation run's correlation_id already appears in
     `substrate_durable_run_state`, `chat_stance_belief_log`, and the curiosity
     graph). No motor/governor code changes, but every turn's full step detail
     still crosses the bus and gets discarded per-step at the filter, rather
     than never being asked for.

   B is thinner (no cross-service plumbing) and reversible (widen or narrow
   the filter set without touching the motor). A is more honest about intent
   (a turn that isn't being watched never pays the cost of building the full
   step dict) but costs three services' worth of schema changes for a filter
   that could live in one. **Recommendation: B**, unless the per-step dict's
   own construction cost (not just the write) turns out to matter — that is
   answerable by profiling one turn, not by guessing.

2. **Volume, if scope is B and something goes wrong with the filter.**
   Self-inquiry is capped at 3/day; curiosity investigation has its own daily
   cap (`HUB_CURIOSITY_INVESTIGATION_DAILY_CAP`, currently 6). Together that
   is roughly 10-40 turns/day depending on hop counts, each turn 15-90+
   steps observed live today. That is a few hundred to low-thousands of rows
   a day — small. Ordinary chat turns are far more frequent and unbounded by
   any daily cap; if the filter at the write point is ever wrong in the wide
   direction (matches more correlation_ids than intended), volume could jump
   by an order of magnitude with no cap catching it. Needs an explicit
   assertion/alarm, not just a filter, per the metric-quality-gate discipline
   this repo already applies elsewhere (`AGENTS.md` §0A).

3. **Tool input/output size.** Tool inputs and results can be arbitrarily
   large (a file read, a long shell output). `summarize_harness_step()`
   already caps at 500 chars for the lossy log line; a structured record can
   afford more, but "more" still needs an explicit cap, or one large step
   turns one row into an incident. What cap, and is it per-field or per-row?

4. **Does capturing tool arguments/results ever cross a boundary Orion's own
   self-inquiry sandbox is NOT supposed to read?** The self-inquiry role
   (`orion_readonly`) is scoped to nine named tables today
   (`orion/curiosity/self_inquiry.py:SELF_INQUIRY_PG_TABLES`). A new step
   table would need its own explicit grant, and its content (tool
   input/output from *any* harness turn's motor, if scope ends up wider than
   curiosity/self-inquiry) could include content from turns that have nothing
   to do with Orion's self-model — e.g. a Juniper-directed engineering task
   run through the same motor. Scope B's filter, done right, already keeps
   this narrow; this question exists to make sure nobody widens the filter
   later without re-checking it.

5. **Retention.** Unlike `chat_stance_belief_log` or `self_knowledge_items`,
   this table's whole reason to exist is per-step granularity, which is the
   kind of data that ages out of usefulness fast. Is there a retention window
   (e.g. 90 days), and if so is it enforced by a cron/decay job matching the
   pattern `substrate_decay_scheduler` already uses, or left unbounded and
   revisited later? Recorded as open rather than assumed.

---

## Proposed schema / API changes

### The table

```text
harness_step_trace
  correlation_id      text        -- same key as harness_turn_trace, joinable directly
  step_index          int
  step_type           text        -- from the raw stream-json event's "type"
  served_model         text null  -- per-STEP, not per-turn (the model can rotate mid-turn)
  tool_name            text null
  tool_input_summary   text null  -- capped (see Missing Question 3)
  tool_result_summary  text null  -- capped, same cap
  is_error             boolean default false
  step_started_at      timestamptz -- NEW: does not exist anywhere today
  created_at           timestamptz default now()
  PRIMARY KEY (correlation_id, step_index)
```

Keyed exactly like every other addition in this arc: `correlation_id` joins
directly to `harness_turn_trace`, and from there to the curiosity graph's
`run_id` the same way `harness_turn_trace` already does.

### The write path (Option B from Missing Question 1)

- New channel `orion:harness:step:write` (event), schema `HarnessStepTraceV1`
  in `orion/schemas/harness_finalize.py` (sibling to `HarnessRunStepV1`,
  distinct schema — the write is a durable, capped, filtered projection of
  the live one, not the same shape).
- Producer: a new consumer of the *existing* `orion:harness:run:step`
  (either extend `harness_step_relay.py` or add a sibling subscriber in
  `orion-hub`, since Hub already has the correlation-id cross-reference
  needed for the filter) — publishes `HarnessStepTraceV1` only for
  correlation_ids matching an active/recent curiosity or self-inquiry run.
- Consumer: `orion-sql-writer`, added to its subscribe list and route map —
  same three-place registration this arc has hit and re-verified every time
  (`channels.yaml`, `registry.py`, sql-writer's settings + subscribe list),
  with the same producer test / consumer test discipline (PRs #2102/#2105's
  missing-subscription incident is exactly the failure mode to guard against
  here).
- Grant: add `harness_step_trace` to `SELF_INQUIRY_PG_TABLES` so a
  self-inquiry run can query its own motor's steps directly, and to the SQL
  migration set the same way every other table in this arc got one.

### The per-step timestamp (the one real new instrumentation)

`fcc_motor.py`'s `readline()` loop (`:962-980`) timestamps each parsed line
as it arrives — cheap, in-process, no new I/O — and carries it through
`build_step_frame` so it reaches `runner.py` and can be included in both the
existing live publish and the new durable one.

---

## Files likely to touch

| area | files |
|---|---|
| new instrumentation | `orion/harness/fcc_motor.py` (per-step timestamp) |
| new schema | `orion/schemas/harness_finalize.py` (`HarnessStepTraceV1`) |
| new write path | `services/orion-hub/scripts/harness_step_relay.py` or a new sibling subscriber; the correlation-id filter itself |
| contracts | `orion/bus/channels.yaml`, `orion/schemas/registry.py` |
| consumer | `services/orion-sql-writer/app/settings.py` (subscribe list, route map), a new model file |
| migration | `services/orion-sql-db/manual_migration_harness_step_trace_v1.sql` |
| self-inquiry access | `orion/curiosity/self_inquiry.py` (`SELF_INQUIRY_PG_TABLES`, `LEDGER_TS_COLUMNS`), `scripts/sql/2026-09-08_grant_orion_readonly_self_inquiry.sql` |
| tests | producer test (filter logic: does/does not persist for a given correlation_id), consumer test (sql-writer route), a fixture-driven volume-bound test (Missing Question 2) |

## Non-goals

- Not instrumenting the CLI itself or changing `stream-json` parsing beyond
  adding one timestamp — the data already exists.
- Not building a UI panel for this. The primary consumer is Orion's own
  self-inquiry queries (`SELF_INQUIRY_PG_TABLES`); a human-facing view is a
  separate, later decision if wanted, following the Self panel's own
  precedent (PR #2178) rather than being bundled here.
- Not widening capture to all harness turns (chat included) in the first
  patch — see Missing Question 2's volume argument. That is a real, later,
  separately-decided expansion.
- Not resolving Missing Question 1 in this document. Both options are
  written out because they are genuinely different amounts of work in
  different places; the recommendation (B) is a recommendation, not a
  decision made on Juniper's behalf.
- Not retention/decay policy (Missing Question 5) — flagged, not designed.

## Acceptance checks

1. A self-inquiry run's `correlation_id` has a row in `harness_step_trace`
   per step, joinable to `harness_turn_trace` on that key, within the same
   run.
2. An ordinary chat turn's `correlation_id` (scope B) produces zero rows —
   the filter actually filters, not just "usually filters."
3. `served_model` on at least one multi-step row differs between two steps of
   the *same* turn, proving the per-step (not per-turn) granularity is real
   and not a copy of `fcc_served_model`.
4. A self-inquiry prompt that asks Orion to query its own
   `harness_step_trace` for the current run gets real, distinct per-step
   rows back — the acceptance check the whole arc has used every time:
   Orion can ask their own record and get a real answer, not an empty table.
5. Volume over one week stays within an explicit, stated bound (Missing
   Question 2) — measured, not assumed.
6. Deleting the new table and unregistering the channel returns the system to
   exactly today's behavior — no other code path depends on this table
   existing.

## Recommended next patch

**Resolve Missing Question 1 with Juniper, then build the write path before
the timestamp.** The filter question changes which services get touched, so
it should not be decided mid-implementation. Once resolved: ship the schema,
the sql-writer wiring, and the correlation-id filter first, tested against a
*replayed* set of existing `orion:harness:run:step` traffic if that is
capturable, or against the next few live self-inquiry runs if not. Add the
per-step timestamp in the same patch, since it is small and the table is
useless for ordering without it. Leave the grant and `SELF_INQUIRY_PG_TABLES`
addition for a fast follow-up once real rows exist to grant access to —
matching this arc's own established order (self-inquiry's own outcome-table
grants were added after the tables existed, not speculatively before).
