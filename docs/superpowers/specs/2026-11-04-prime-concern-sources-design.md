# Prime's concerns come from loops that close, not from attention winners

Date: 2026-11-04. Status: proposal (docs only). It changes what Orion chooses to care about, so
proposal mode comes first (CLAUDE.md 0A).

## In one paragraph

For seven months Orion's attention work has mostly re-scored the same internal surprise gauges.
That work covered normalization, percentiles, sustain and fade. Each fix shortened the gauges' wins
without replacing them.

Prime inherits the result. Over 7 days it took 42 concerns. About half were about internal gauges,
and half were about body-economy "which lane binds" lines. **None** came from a world event, from
Juniper, from reading or from curiosity. 40 were dropped, and 2 produced something.

Meanwhile the loops that really close all start from an event or a schedule that comes with an
action and a checkable result. The camera → person ask → Juniper's answer loop closed 495 times in
14 days. Not one of them feeds Prime.

This proposal stops trying to win "attention". Prime takes concerns from a short list of **concern
sources**. Each source is a loop that already closes. The self-monitor grades each source by
whether its concerns produced something, and Prime gives more room to the sources that earn it.

## Arsonist summary

- **What fizzled, and why.** The attempts below changed how gauges are scored:
  - #1247 normalization, which made the top candidate always 1.0;
  - #2583 world-first: 67% of frames now have no winner, but internal gauges still win 57% of the
    frames that do;
  - #2591 fade;
  - #2705 sustain;
  - #2555 focus runs: 100% internal over 14 days.

  The two attempts to add *event* sources died: `node:substrate.street` has read 0.0000 in all
  122,342 frames over 3 days, and the curiosity-findings candidate in #2722 stayed docs only. #2906
  (known-person arrival) was declined.
- **The design flaw.** A single global winner ranks everything against everything. The always-on
  gauges are the only candidates that fire every tick, so they win by being present, not by being
  useful.
- **What already works.** These loops tie an event to an action to an outcome:
  - camera asks: 495 answered and applied, 33 dismissed, in 14 days;
  - reading pulls: 442 a week, 0.90 ran;
  - reading finding → outreach: 47 a week, 0.31 replied;
  - curiosity runs: 0.98.

  The self-monitor (rung 6, live 11-03) now grades loops like these.

## Current architecture (verified 2026-11-03/04)

- **What Prime reads.** It reads Temporal Self arcs (`GET /temporal-self/frame`,
  `orion/prime/thought.py:219 _attention_lines`), not substrate frames. Attention arcs over 7 days
  by dwell: internal `node:*` 56%, `world:chat` 28%, `world:reading` 16%. Street, doorbell and
  perception produce no attention arcs.
- **Where concerns live.** Redis `orion:prime:state` holds them (last 20 plus the held ones).
  `grammar_acts` holds the bet: `actor=prime`, `verb=predict`, `signal=artifact_produced`. The
  concern's source is not stored anywhere queryable: `goal_ref` and `workspace_ref` are empty.
- **What a concern does.** Prime's effector is one question into Hub's curiosity focus slot
  (#3001). Prime's other acts close well: `prime.ask` treated 0.63 vs control 0; `prime.infer` 297
  thoughts at 0.67.
- **The self-monitor today.** It marks `act.prime.predict` unreliable (5% vs a stated ~11%), so the
  act gate (#3081) now refuses new concern bets, with a re-test every 6 h.

## Proposal

### 1. A concern source contract

A concern source is a loop that meets all three tests:
- (a) it emits an **event** about the world, Juniper or Orion's own work;
- (b) it comes with an **action** Prime's effector can take;
- (c) it has an **outcome** the world settles.

Each source publishes `ConcernCandidateV1` records:

| field | meaning |
|---|---|
| `candidate_id`, `source` | stable id; source name, e.g. `camera.person_ask`, `reading.finding`, `reading.pull`, `curiosity.finding`, `vision.expectation_miss`, `self_monitor.failing_faculty` |
| `subject`, `summary` | what it is about, in plain words, from the record itself |
| `affordance` | the action Prime could take, as a curiosity focus question template |
| `evidence_refs` | the source rows |
| `at`, `expires_at` | when it happened, and when it stops being worth a concern |

v1 sources, from the live audit:

| source | event → action → outcome | volume |
|---|---|---|
| `camera.person_ask` | arrival → look and ask "who is this?" → Juniper answers | ~250/week answered |
| `reading.finding` | finding → tell Juniper / investigate → reply or result | ~47/week |
| `curiosity.finding` | run finding → follow-up question → investigate disposition | ~50/week |
| `vision.expectation_miss` | an expected visitor didn't show → look again / ask → met or missed | ~30/week |
| `self_monitor.failing_faculty` | a skill went broken or unreliable → "why?" → recovered or explained | rare |

**Not concern sources:** internal surprise gauges and body-economy binding lines. They have no
action in the world, and their only "outcome" is the gauge moving away. They stay where they
belong: in regulation, prices and the attention frame. This proposal does not touch the attention
frame.

### 2. Prime picks from candidates, not from attention arcs

Each tick, Prime gets the open candidates. It keeps its current "is this worth holding?" judgment,
and still asks the LLM. But the menu is now candidates, not gauge arcs. The concern stores its
`source` and `candidate_id` in `goal_ref`, so its origin can be queried.

### 3. Each source earns its share from its track record

- Each concern bet is graded per source by the self-monitor, as the faculty
  `act.prime.predict.<source>`. This needs a per-source key in the act ledger.
- Prime's share of tries per source follows that record: a Beta posterior per source, picked by
  Thompson sampling.
- No source ever gets a zero share. Each keeps a small floor so it can recover. This is the same
  rule as the gate's re-test.
- The result is visible: the self-monitor and Orion's chat line can say "my concerns from camera
  arrivals produce something 60% of the time; my concerns from reading findings, 25%."

### 4. Two prerequisite fixes, each its own small PR

- **The street node is pinned at 0.** The doorbell/walkway street node reads 0.0000 in every frame,
  so the street never competes. This is a field pinned at zero, and nobody has filed it. Fix it
  whatever happens to this proposal.
- **The camera-ask loop isn't graded.** Add a self-monitor adapter for `orion_ask` answers. That
  makes the system's most reliable loop visible, and it gives the camera source a track record from
  day one.

## Missing questions (defaults in brackets)

1. **Do gauge and binding concerns stop entirely, or keep a small floor?** [stop. They fail the
   three tests. The gauges still reach Orion through regulation and the chat-line self-readings]
2. **Should `self_monitor.failing_faculty` feed Prime, or stay with E2's direct self-question?**
   [both: E2 asks once; Prime may hold it as a concern if it earns its share]
3. **Floor share per source.** [10% of Prime's concern tries, split across sources]
4. **Camera asks already go to Juniper directly.** What does a Prime concern add? [a reason to look
   closer or follow up, e.g. "this person came twice today". If the record shows it adds nothing,
   the source earns no share, and that is the right answer]

## Proposed schema / API changes

- `orion/schemas/concern_candidate.py` with `ConcernCandidateV1`, registered.
- One writer per source. They share one place: a Redis sorted set `orion:prime:concern_candidates`
  with a TTL per candidate, plus a bus channel only if a bus consumer needs one.
- Prime: the candidate menu replaces `_attention_lines` as the concern intake. A concern stores
  `source`/`candidate_id` in `goal_ref`.
- Act ledger: Prime's `predict` act carries `object_ref = concern_source:<source>`, so the
  self-monitor faculty becomes `act.prime.predict.<source>`. `act_faculty` keeps the name segment.
- Self-monitor: an `orion_ask` adapter (camera asks).

## Files likely to touch

- `orion/prime/{thought,state,acts}.py`
- `services/orion-mind/app/prime_loop.py`
- one small writer per source:
  - Hub `reading_finding_outreach` (finding)
  - Hub curiosity investigation (finding)
  - sql-writer `vision_individuals` / `vision_rhythm` (asks, expectation misses)
  - Hub `self_monitor_questions`
- `orion/self_monitor/adapters.py` and `collect.py` (the `orion_ask` adapter)
- the street-node fix: `node:substrate.street`, from the street-attention work (#2677/#2826)

## Non-goals

- No new global attention scorer. The attention frame is left alone.
- No banning gauges anywhere except as Prime's concern sources.
- No new Prime effector. It stays one curiosity focus question.
- No LLM in source writers. Candidates are built from the records' own fields.

## Proposal-mode checklist

- **Capability that changes:** what Prime chooses to care about. The menu moves from internal gauge
  arcs to closed-loop events, and each source's share is earned.
- **Data touched:** reads existing source tables. Writes the candidate set, Prime state, and the
  act-ledger object_ref.
- **Privacy boundary:** none (Juniper's standing rule).
- **Trace that proves it worked:**
  - `act.prime.predict.*` faculties appear in the self-monitor;
  - over 14 days, Prime's concern produce-rate beats the 5% baseline;
  - per-source records differ, and shares follow them.
- **Dangerous failure modes:**
  1. **One source monopolises Prime.** Mitigation: the floor, plus Thompson sampling, which keeps
     exploring.
  2. **Camera concerns turn into nagging Juniper.** Mitigation: Prime's effector is a focus
     question, not outreach; outreach keeps its own gates.
  3. **The produce-rate rises only because candidates are easier.** Mitigation: report
     produce-rate per source, not only in total.
- **Disable or roll back:** `PRIME_CONCERN_SOURCES_ENABLED=false` returns to today's arc intake. Each
  source writer also has its own flag.

## Acceptance checks

1. Each v1 source writes candidates from real rows, and each candidate carries evidence refs and an
   expiry.
2. Prime holds concerns only from candidates. `goal_ref` names the source, and it is non-empty on
   100% of new predict acts.
3. The self-monitor shows `act.prime.predict.<source>` faculties within 7 days.
4. Over 14 days, the total produce-rate is above the 5% baseline, with confidence (the Wilson lower
   bound above 0.05), and it is reported per source.
5. The street node is non-zero on street events, and street candidates appear.
6. With the flag off, Prime's behaviour is identical to today.

## Recommended next patch

1. **Street-node zero-pin fix**, a small PR that stands on its own.
2. **The `orion_ask` self-monitor adapter**, so camera asks get a track record.
3. **The contract, plus two sources** (`reading.finding`, `camera.person_ask`), plus Prime intake,
   behind the flag.
4. **The remaining sources**, plus per-source shares from the track record.
