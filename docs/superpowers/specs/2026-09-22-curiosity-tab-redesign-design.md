# Curiosity tab redesign — from "everything ever" to "what happened in this sitting"

> **Status:** Design proposal. Nothing here is implemented yet.
> **Date:** 2026-09-22
> **Asked by Juniper:** "the curiosity tab in hub is basically unusable. I can't
> really understand what happens in these runs, what happened between hops, if
> there were reach outs to me and then subsequent replies I may have offered.
> There are three different curiosity types now. The chart grows unbounded and
> I don't really know what it means. Can you reimagine this tab?"
> **Scope:** the Hub Curiosity tab (`/curiosity` page, its read model, its
> endpoints) plus the two small write-side seams needed to answer "did Orion
> reach me, and did I answer". Nothing about when or how curiosity runs.

## Arsonist summary

The tab is unusable because it was built as a **graph inventory**, not a
**diary**. It lists every prior, every revision and every run Orion ever
wrote, re-renders all of it every 60 seconds, and never says when anything
happened, how long it took, whether it failed, or what Orion did with it.

Four concrete reasons it cannot answer Juniper's questions, all verified
against live data on 2026-09-22:

1. **There is no run story.** A run's pieces are spread across three stores
   (Postgres lifecycle rows, Orion's own FalkorDB graph, the journal table)
   and no endpoint joins them. The page shows hop notes as a bare numbered
   list with no times, no start, no end, no failures. Meanwhile every piece
   needed for a timeline already exists in the graph: a role-choice node at
   the start, numbered hop notes with a real millisecond clock, a finding with
   evidence, the prior's before/after confidence, and an end-of-turn outcome.
2. **"Three curiosity types" are never named on the page.** Two of them
   (`investigate`, `self_inquiry`) are mixed into one ledger; the third
   (`self_sense_eval`) writes no graph and no journal so it is invisible except
   as a score table at the top. A fourth workflow (`self_study.reflect`) lands
   in the same lifecycle table and would leak into any naive query.
3. **Reach-outs are shown as intent, and the intent never became a message.**
   The page renders "wanted to reach out" from the graph. Live check: in the
   last 14 days six runs decided to reach out; **zero were delivered**. In fact
   curiosity outreach has **never been sent, all time** — every attempt was
   blocked (`daily_cap`, `quiet_hours`, `turn_in_flight`, `disabled`,
   `empty_generation`, `orion_passed`). The blocked pre-check does not even log
   a decision row, so the six recent ones left no trace at all. The
   unsolicited messages Juniper *does* see in chat come from the separate
   endogenous tick, which spent all 63 sends in the same window from the shared
   daily cap. And there is no reply link: Juniper's reply is minted with a
   fresh random id and shares only a session and a clock with the outreach.
4. **The chart is a node count.** "What each run added" is a stacked bar per
   run of how many graph nodes it wrote, one row per run forever, scaled to
   the biggest run ever. It answers "is the graph growing", which is not a
   question anyone is asking on this tab.

The redesign: one bounded **sittings strip** (14 days, colored by line and
outcome) that opens one **run story** at a time (a real timeline joined
across the three stores, with the reach-out and any reply at the bottom),
above a **live priors** list that hides closed ones by default. Two thin
write-side seams make the reach-out story truthful: record every curiosity
outreach decision (including pre-check blocks) and stamp Juniper's next
message in that session as the reply.

## Current architecture

Plain English first, identifiers after.

**The page.** The Curiosity tab is an iframe of a standalone page. That page
fetches one JSON blob every 60 seconds and rewrites eight sections from
scratch with `innerHTML`, so any open disclosure or scroll position is lost
each minute. Sections in order: Self (definition, lived answers, eval table,
journal), four stat tiles, contractor briefs, one confidence sparkline card
per open prior (unbounded), the stacked node-count bars (unbounded), the runs
ledger (unbounded), and a table of every prior (up to 2000).
(`services/orion-hub/templates/curiosity_atlas.html`, 965 lines, no chart
library; tab wiring `services/orion-hub/templates/index.html:2563-2582`.)

**The read model.** One endpoint feeds everything, by design so panels cannot
disagree. It runs eight static Cypher reads against Orion's graph with hard
limits (2000 priors, 5000 revisions, 5000 hops, 2000 findings, 2000
outcomes, growth queries with **no** limit), then joins journal bodies from
Postgres for every run in the payload, then reads today's counters from
Redis. (`GET /curiosity/api/atlas` → `curiosity_routes.py:257` →
`orion/curiosity/atlas.py:read_atlas` :480-535; self panel from
`orion/curiosity/self_panel.py`.)

**The three lines.** One Hub tick runs three scheduled lines under one lock,
so only one curiosity run of any kind is ever in flight
(`curiosity_investigation.py:1247`, `:1296`):

| line (`detail.line`) | plain name | what it does | writes | cap/day |
|---|---|---|---|---|
| `investigate` | **World question** | picks a prior about the world or the substrate, tests it in ≤5 hops, revises confidence | graph nodes, journal entry, attention row, maybe a reach-out | 3 |
| `self_inquiry` | **Self question** | draws a standing "what am I" question, same 5-hop shape | same, plus `:SelfDefinition` / `:LivedAnswer`, `self_concept_history` | 3 |
| `self_sense_eval` | **Self-sense check** | asks four fixed chat questions and scores them; no investigation | `self_sense_eval_log` only — no graph, no journal, no hops | 7 |

Both investigation lines share the workflow name `curiosity.investigate` and
the same five lifecycle nodes (`harness_turn → read_turn_result →
publish_attention_row → journal → finish`). The line is only recorded in the
`finish` row's `detail`, so an in-flight run has no line in Postgres. The
`self_study.reflect` workflow lands in the same table and is not curiosity.
(`orion/schemas/durable_run.py:58,62,141`.)

**What a run leaves behind, and where** (verified on run `446ddd7165d5`,
2026-09-21):

| store | what | clock |
|---|---|---|
| graph `:InvestigationRole` | `local_crawl` vs hire-a-peer, and why | `written_at` ms, at start |
| graph `:Hop` ×n | one note per hop, `n`, `written_at` ms | real ms clock since 2026-09-19; null before |
| graph `:Finding` | text + evidence | `written_at` |
| graph `:PriorRevision` | prior_id, 0.60 → 0.68, revised → supported | `written_at` |
| graph `:TurnOutcome` | continue_line, continue_note, reach_out, reach_out_why | `written_at`, at end |
| Postgres `substrate_durable_run_state` | one row per lifecycle transition; `detail` on `completed`/`failed` only (`line`, `finding_text`, `journal_entry_id`, `attempts`, `reach_out*`, `error`) | `created_at` |
| Postgres `journal_entries` | the prose write-up, `source_ref = curiosity:<run_id>` | `created_at` |
| Postgres `harness_turn_trace` | `fcc_elapsed_sec`, `step_count`, served model, grounding status | keyed by a *derived* correlation id; run id appears only inside free text |
| Postgres `curiosity_hop_reading` | supervisor's per-hop reading | **0 rows live**; supervisor only runs by hand |
| Postgres `endogenous_outreach_decisions` | send/blocked decision | keyed by `uuid5("curiosity_outreach:<run_id>")`; **no row when blocked at pre-check** |
| Postgres `chat_history_log` | the delivered message (`client_meta.unsolicited=true`) | never for curiosity, see above |

**Live lifecycle rows have gone quiet.** From 2026-09-08 to 2026-09-13 the
table received `running`/`resumed`/`failed` rows every day (134
`no_final_frame` and 24 `rpc:TimeoutError` failures in 14 days). From
2026-09-14 onward only `completed` rows land. Orion's own self-inquiry runs on
2026-09-21 reported "12 of my turns died today at the 420s stall", so failures
are still happening; they are just not reaching this table. **Root cause
UNVERIFIED** — noted here because the run story's start time and failure
timeline depend on those rows.

**Reach-out path (Door A).** The run's own turn sets `reach_out` +
`reach_out_why`. After the run completes, Hub runs a second composition turn
and hands the text to the endogenous outreach loop, which applies the shared
gates (disabled, turn in flight, quiet hours, daily cap, cooldown) and, if
allowed, pushes a chat bubble, a history row and an in-app toast. The result
is discarded at all three call sites; nothing is written back to the run.
(`curiosity_investigation.py:2756-2838`, `endogenous_outreach.py:2048,2139`,
gates `:449`.) Juniper's reply is a fresh `uuid4` turn with no reply-to
field (`websocket_handler.py:1371`). The only run→message key is the forward
`uuid5(NAMESPACE_URL, "curiosity_outreach:<run_id>")`.

**Two neighbouring surfaces already show pieces of this.** The Surface tab
shows lifecycle counts; the runtime-activity marquee shows live transitions in
memory only; a Lightdash dashboard ("Curiosity Operations", PR 2026-09-14)
shows run/transition/graph-write analytics scraped from journal prose by
regex. None joins a run's transitions to its hops to its journal to its
reach-out.

## Missing questions

1. **Where does Juniper want to reply?** Two options. (a) Reply in chat as
   today; the tab shows the reply by stamping the next message in that session
   as the answer (a time-adjacency rule, labelled as such). (b) A reply box on
   the run story itself that posts into chat with an explicit reply key, which
   makes the link exact. Recommendation: build (a) now, add (b) in the next
   patch. Needs her call because (b) changes where she talks to Orion.
2. **Should curiosity reach-outs get their own budget?** Today they share one
   daily cap with the endogenous tick and have lost every time. Fixing that is
   a cognition-loop change (proposal mode), not a tab change. The tab will
   make the starvation visible; whether to fix it is a separate decision.
3. **Should the self-sense check appear on this tab at all?** It is a scored
   probe, not an investigation. Proposal: keep it, as a thin row in the strip
   with its four scores, so the day's activity is complete; no run story.
4. **Window.** 14 days default, 90 max (the table's retention). Is 14 right?
5. **Why did non-terminal lifecycle rows stop on 2026-09-14?** Not a design
   question, but the run story degrades to "start = first graph node" until it
   is answered.

## Proposed schema / API changes

### Read side (no new stores)

**New endpoint: `GET /curiosity/api/runs?days=14&line=all`** — bounded run
summaries, newest first. One row per run_id found in **either** the lifecycle
table (`workflow IN ('curiosity.investigate','self_sense_eval')`, window by
`created_at`) or the graph (`:TurnOutcome` / `:Hop` with `written_at` in
window). Fields:

```
run_id, line, plain_line_label,
started_at        -- first lifecycle row; else earliest graph node for the run; else null
finished_at       -- completed/failed row; else :TurnOutcome.written_at
status            -- completed | failed | running | unknown
attempts, error   -- from lifecycle detail
hops, findings, revisions   -- counts
prior_touched     -- {prior_id, claim, from, to, from_status, to_status} or null
reach_out         -- {wanted: bool, why, decision: sent|blocked:<gate>|composed_empty|passed|not_recorded, sent_at, reply: {at, text} | null}
journal_entry_id
duration_sec      -- finished_at - started_at when both known
```

**New endpoint: `GET /curiosity/api/run/{run_id}`** — the full story:

```
run{...as above...},
timeline[]        -- sorted by clock, each {at, kind, ...}:
  role_choice     {choice, why}                         from :InvestigationRole
  lifecycle       {node, status, error, resumed_from}   from substrate_durable_run_state
  hop             {n, note, reading?}                   :Hop + curiosity_hop_reading (by run_id, n, written_at)
  finding         {text, evidence}
  revision        {prior_id, from, to, from_status, to_status}
  outcome         {continue_line, continue_note, reach_out, reach_out_why}
  outreach        {decision, gate, composed_text?, sent_at?}   endogenous_outreach_decisions + chat_history_log by uuid5 key
  reply           {at, text}                             chat_history_log where client_meta.in_reply_to = <uuid5 key>
  journal         {entry_id, body}                       journal_entries by source_ref
harness           {fcc_elapsed_sec, step_count, served_model, grounding_status} | null
```

`GET /curiosity/api/atlas` keeps `priors`, `revisions`, `schedule`, `self`,
`peer_briefs`, and the totals. It **drops** `runs[]` and the growth data (the
new endpoints own runs). `priors[]` gains `last_tested_at` as a real epoch and
trajectory points carry `written_at` so the sparkline can use time on x.

Read-model changes in `orion/curiosity/atlas.py`:
- `ATLAS_HOPS_CYPHER` selects `h.written_at` and hops sort by
  `hop_order_key` (the fix that already exists in `worldview.py:267` but the
  atlas read never adopted — today retried runs show hops interleaved
  1,1,2,2,3,3).
- All run reads take a `since_ms` bound instead of a row cap.
- A new `orion/curiosity/run_story.py` owns the cross-store join
  (lifecycle rows + graph nodes + journal + outreach decision + reply +
  readings + harness trace) and its `to_payload`. This is the one new seam;
  it exists so the join is testable with fixtures from each store.

### Write side (two thin seams, both additive)

1. **Every curiosity outreach decision gets a row.** `_maybe_reach_out`
   records a decision for the pre-check block (`blocked_reason()` returns
   early today with only a log line) with `result_json.source =
   "curiosity_outreach"` and the run-derived `correlation_id`. No column
   change: the key is already deterministic from `run_id`. Also attach the
   run id inside `result_json` (`result_json.run_id`) so a reverse lookup does
   not require recomputing the uuid5.
2. **Stamp the reply.** On an inbound Hub chat message, if the most recent
   assistant row in that session is unsolicited and younger than 12 hours and
   no later solicited assistant reply exists, set
   `client_meta.in_reply_to = <that row's correlation_id>` on the new turn's
   history row. This reuses the selection rule that already exists for
   Door-B provenance (`outreach_provenance.py:37`), widened to not require the
   provenance capsule. `client_meta` is a free `Dict[str, Any]` on
   `ChatHistoryMessageV1` (`orion/schemas/chat_history.py:52`), so **no
   schema, registry or channel change**. The UI labels the reply "next message
   in that session, within 12h" so the heuristic is not passed off as a fact.

Nothing writes to Orion's graph. Hub remains read-only there.

### Timing fields the runner already knows (small, optional in patch 1)

`finish_detail` gains `harness_elapsed_sec` and `turn_correlation_id`
(`services/orion-durable-runs/app/graph.py:197-222`). Today
`harness_turn_trace` can only be found for a run by text-searching
`final_text`; a structured key makes the harness row joinable. Additive on a
`forbid` model, so **deploy `orion-durable-runs` before `orion-hub`** (same
rule as PR #2158's `line` field).

### Page

Replace `curiosity_atlas.html` wholesale. Structure, top to bottom:

1. **Header + budget row.** Three small tiles, one per line, in plain words:
   "World questions 2 of 3 today · next at 16:40", "Self questions 1 of 3",
   "Self-sense checks 4 of 7". A fourth tile: "Reach-outs: 6 wanted, 0 sent
   in 14 days — all blocked by daily cap" (from the decisions table). This is
   the only place counts appear.
2. **Sittings strip (the chart).** 14 columns (days, local tz), one small
   chip per run, colour = line, glyph = outcome (finished / died / wrote
   nothing / reached out and sent / reached out and blocked). Hover shows the
   one-line summary; click opens the run story and sets `#run=<id>` in the
   hash so it can be linked and survives refresh. Bounded by construction:
   at most ~13 chips per day at the current caps. Legend explains each line in
   one sentence.
3. **Run story** (one at a time, defaults to the newest). A vertical timeline
   with a relative clock ("+0:00 started · world question · picked prior
   'who matters', 0.60, never tested · working locally because queue pressure
   was high"; "+11:54 hop 1 …"; "+11:54 hop 2 …"; "+12:20 finding …";
   "+12:20 prior 0.60 → 0.68, revised → supported"; "+14:15 done · left
   itself a note: …"; "reach-out: wanted — 'the repair step recorded its own
   prompt as my answer…' — **blocked: daily cap**, nothing sent"; "reply: —").
   Failures and resumes appear inline where they happened. Journal body in a
   disclosure at the end. Supervisor readings render beside a hop when they
   exist, and the section says "no readings" when the table is empty rather
   than hiding.
4. **Self card** (current definition, version, date; lived answers behind a
   disclosure). Moves below the story; unchanged data.
5. **Priors.** Live only by default, sorted by last tested, with a
   "show closed (N)" toggle and a text filter. Sparkline x-axis is time.
   Clicking a prior's "last tested by" opens that run story.
6. **Contractor briefs** stays as is, behind a disclosure, showing the
   question (`:HelpRequest`) next to the answer.

Rendering: fetch `/runs` on a 60s poll; re-render only when the payload hash
changes; never re-render an open run story unless its own run_id changed.
Remove the "What each run added" bars and the "Every prior" flat table.

## Files likely to touch

- `orion/curiosity/run_story.py` (new): cross-store join + payload. Tests
  `tests/test_curiosity_run_story.py` with fixtures for each store, including
  a retried run (hops 1,1,2,2), a blocked reach-out, a sent reach-out with a
  reply, a `self_sense_eval` run, and a `self_study.reflect` row that must be
  excluded.
- `orion/curiosity/atlas.py`: `written_at` on hops, `since_ms` bounds, drop
  runs/growth from the atlas payload. Update `tests/test_curiosity_atlas.py`.
- `services/orion-hub/scripts/curiosity_routes.py`: two new endpoints.
  `services/orion-hub/tests/test_curiosity_routes_runs.py` (new).
- `services/orion-hub/scripts/curiosity_investigation.py`
  `_maybe_reach_out`: record pre-check blocks, put `run_id` in
  `result_json`. `services/orion-hub/scripts/endogenous_outreach.py`: a
  public `record_blocked(reason, correlation_id, source, extra)` so the loop's
  single writer stays the single writer.
- `services/orion-hub/scripts/websocket_handler.py` + `outreach_provenance.py`:
  the reply stamp. Test: an inbound message after an unsolicited row gets
  `in_reply_to`; one after a solicited reply does not; one 13h later does not.
- `services/orion-durable-runs/app/graph.py`: `finish_detail` timing fields
  (optional, patch 1b).
- `services/orion-hub/templates/curiosity_atlas.html`: rewrite.
  `tests/test_curiosity_atlas_template.py`: rewrite to assert the strip, the
  story, the hash routing, the three plain-English line labels, and that no
  unbounded list remains.
- `services/orion-hub/README.md`: tab section.
- Not touched: `orion/bus/channels.yaml`, `orion/schemas/registry.py`,
  any `.env_example` (no new env keys), the Lightdash dashboard.

## Non-goals

- Changing when curiosity runs, what it investigates, its caps, its prompts,
  or how hops are written. (The tab shows; it does not steer.)
- Giving curiosity outreach its own budget or priority over the endogenous
  tick. Surfaced as a finding; separate proposal.
- A reply box on the tab (Missing question 1b) — next patch if Juniper wants
  it.
- Running the supervisor on a schedule. The story renders readings when they
  exist and says so when they do not.
- Fixing the lifecycle-rows-went-quiet issue or the chat-repair prompt leaking
  into `finding_text` (see Risks). Both get follow-up issues, not this patch.
- Per-hop tool calls, tokens, or model outputs. They are not persisted
  anywhere; the story shows what exists.
- Replacing or duplicating the Lightdash operations dashboard.

## Acceptance checks

1. `GET /curiosity/api/runs?days=14` returns only `curiosity.investigate` and
   `self_sense_eval` runs; a seeded `self_study.reflect` row is absent. Every
   run carries a `plain_line_label` from exactly three values.
2. `GET /curiosity/api/run/446ddd7165d5` (live) returns a timeline whose
   first item is the role choice, whose hops are ordered by `written_at`, and
   whose last items are the outcome, the outreach decision and the journal.
   A retried run from fixtures renders hops as 1, 1, 2, 2 in clock order with
   attempt boundaries marked, never 1, 2 collapsed.
3. For each of the six runs with `reach_out=true` since 2026-09-08, the story
   shows a decision line. After patch 2 is deployed, a newly blocked run shows
   `blocked: <gate>` and a row exists in `endogenous_outreach_decisions` with
   `result_json.source='curiosity_outreach'` and `result_json.run_id`.
4. Reply stamp: send an unsolicited message into a test session, then a
   user message; the user row's `client_meta.in_reply_to` equals the
   unsolicited row's `correlation_id`. A user message after a *solicited*
   assistant reply carries no stamp. Live proof: the first real curiosity
   reach-out that gets through, followed by Juniper's answer, appears on the
   run story as "reply: …".
5. The page renders at most 14 days of chips; the runs list and priors list
   have no path that grows with the total number of runs ever. Verified by a
   template test that the removed sections and their fetch fields are gone.
6. Polling with an unchanged payload does not touch the DOM (a test that
   stubs fetch twice with equal bodies and asserts one render). An open run
   story survives a poll.
7. The three-line budget tiles match the Redis counters the loop itself
   reads (already imported, not retyped).
8. Existing gates: `pytest tests/test_curiosity_atlas*.py
   services/orion-hub/tests -q`, `python scripts/check_env_template_parity.py`,
   `check_schema_registry.py`, `check_bus_channels.py` all green; no
   `.env_example` diff.

## Recommended next patch

**Patch 1 — read model + page (no writes, safe to ship first).**
`run_story.py`, the two endpoints, the `written_at` hop fix, the page
rewrite with strip + story + live-priors, tests. Reach-out line renders
"not recorded" for pre-check blocks until patch 2 lands. This alone answers
"what happened in this run and between hops" and kills the unbounded chart.

**Patch 2 — reach-out truth (two writes).** Record pre-check blocks; stamp
replies. Small, additive, no schema registry change. Turns the story's last
two lines from "unknown" into facts.

**Patch 1b (optional, same PR as 1 or 2)** — timing fields on
`finish_detail`; deploy durable-runs before hub.

**Separate proposal, not this arc:** curiosity outreach budget. The tab will
show "6 wanted, 0 sent" on day one; that number is the argument.

## Risks / findings surfaced while looking (not fixed here)

- **Curiosity has never reached Juniper.** All-time `curiosity_outreach`
  decisions: blocked only, zero `sent`. The six recent intents (2026-09-19 to
  09-21) include Orion reporting a record-corruption bug in the chat repair
  step and a 420s stall killing over half its motor turns — real operational
  findings that were composed and dropped. Severity: should. Mitigation: this
  design makes it visible; the budget question is Missing question 2.
- **A chat repair prompt is stored as a curiosity finding.** Run
  `2a6ab1577b03` (2026-09-21 23:23) has `finding_text` beginning "We need
  answer user's request: repair Orion's draft reply to Juniper after
  integrative reflection rejected it…". That is the same defect Orion tried
  to report in run `9a5dfe992452`'s blocked reach-out. Severity: should.
  Follow-up issue: guard in the repair step, and the run story should flag a
  finding that reads as an instruction prompt.
- **Non-terminal lifecycle rows stopped on 2026-09-14.** Cause UNVERIFIED.
  Until fixed, `started_at` falls back to the first graph node and failures
  mid-run are invisible to the story. Severity: should. Follow-up issue.
- **Harness trace is not joinable by run id** (derived correlation, run id
  only in free text). Patch 1b fixes forward; history stays unjoinable.
- **Dead affordances to remove in the rewrite:** park/pin self-question
  endpoints nothing calls; `curiosity_peer_brief` Postgres copy nobody reads;
  test-only labels (`HopTest`, `HopLen550`, …) still registered in the
  production graph's label list.
