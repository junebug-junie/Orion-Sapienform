# Orion Situation Graph: design

Status: **APPROVED by Juniper 2026-10-07.** Build in the order given in section 9.
Board (diagrams): https://claude.ai/artifact/4k6YbuEXPTDqQMg2tJSrrg
Evidence: read-only queries on `conjourney`, code on main @ bbc51703a.

## 1. The problem

Orion keeps no picture of the current situation between chat turns. Each turn rebuilds context from scratch: 8 turns of raw text, plus three memory searches that each use Juniper's raw message as the search query. When the message is casual ("just chilling"), nothing relevant comes back.

**Incident, 2026-10-05 → 10-06.**
- **What happened:** Juniper said "I'm in Chicago till Wednesday." 44 hours later, Orion asked whether she'd make it back to the basement before bed (`3af6a31d…`).
- **The fact was saved three times:**
  - in the chat log;
  - as a crystallization (raw text, salience 0.662, the same as "wowza so many prs");
  - correctly, as an episode memory ("Juniper is in Chicago for a team meeting until Wednesday").
- **None of the copies reached the turn:**
  - The chat log copy was outside the 8-turn window.
  - The crystallization was in the belief search's candidate pool, but the open-loop bucket filter dropped it (`services/orion-recall/app/collectors/active_packet.py:19,185-186`). Over 7 days, belief recall picked that open-loop profile on 634 of 634 turns.
  - Episode memory has no chat reader, and `validate.py:406-407` drops `expires_at` unless `purpose=follow_up`.
- **The distiller also stored a confabulated row:** "Juniper corrected me that she lives in Ogden, Utah, not Chicago" (`c0dc86c8…`). Its only evidence quote grounds "lives in Ogden". The "correction" came from Orion's own reply.

**Per-turn search cost today** (`recall_telemetry`, 7-day averages):

| Search | Average | Note |
|---|---|---|
| continuity search | 531 ms | |
| belief search | 1,107 ms | open loops only |
| self-check search (finalize-reflect) | 327 ms | 77 candidates fetched |

On top of these, the stance build reloads concept induction and the felt-state self-definition lane.

## 2. The model: how a mind holds "now"

A person doesn't search memory to know where their partner is. They carry a picture of the situation: who, where, doing what, waiting on what, until when. That picture updates when something happens (Zwaan and Radvansky's event-indexing situation model). The rest of memory works off it:
- **recall** is driven by cues from the situation, not the last sentence;
- **checking** happens on mismatch;
- **consolidation** happens offline;
- **ended situations** become history.

This design gives Orion that loop: write offline, carry the situation, recall by cue, check on mismatch.

## 3. Architecture

A single long-lived durable LangGraph thread, `situation:juniper`, is the **only writer** of `SituationStateV1`.

**Event sources.** All of them already exist and already emit:

| Event | Source |
|---|---|
| chat turn completed | hub, after the reply; carries mind's turn claims, referents, draft |
| episode distilled | `memory.episode_distill` persisted |
| concept profile delta / topic run finished | spark concept induction; topic foundry, daily |
| self-definition / lived answer written | curiosity self-inquiry |
| reading turn finished | `reading.turn` |
| presence / camera stream change | vision window |
| clock tick, every 15 min | end dates come due |
| Juniper correction | Hub: not true / ended / edit until |

**Readers:** chat turn (one Redis read), curiosity (what to investigate given now), outreach/endogenous, and the Hub "Right now" panel.

**Chat turn after the change:**
1. read the situation;
2. stance, using the primed memories in the state;
3. one bounded referent search, **only** if the turn names a person or place not in the cues;
4. draft;
5. mismatch check of the draft against the state, with a targeted lookup **only** on conflict;
6. reply;
7. emit the turn-completed event, asynchronously.

**Removed from the turn:**
- continuity search
- belief search
- finalize-reflect search
- per-stance concept reload
- self-definition fetch

**Crystallization** leaves chat recall (the active-packet collector). It keeps feeding dream, reverie, curiosity study material, the Hub workbench and graphiti until Stage 4 (#2440) moves them to episode memory.

## 4. The durable run: `situation.update`

Reuses `services/orion-durable-runs` patterns:

| Block | Source |
|---|---|
| TypedDict state, `StateGraph`, `compile(checkpointer)` | `episode_distill_graph.py:52-70,208-226` |
| pooled `AsyncPostgresSaver` | `main.py:139-166` |
| fixed thread id | as `base_run_id` |
| `retry_wait` interrupt | `episode_distill_graph.py:179-182` |
| event-triggered submit | `main.py:110-136` |
| `DurableRunStateV1` publishing | `runner.py:789` |

No GPU lease in v1: every node is deterministic, because the distiller already did the thinking.

**Nodes:**
1. **ingest**: validate the event.
2. **order**: revision +1. Duplicate or stale events skip to done.
3. **reduce**: upsert or supersede Facts.
4. **expire**: lapse facts whose `valid_until` has passed, and emit `memory.situation.lapsed.v1` so a past-tense episode memory row gets written.
5. **cue**: referent and concept ids from the current facts plus recent turn referents.
6. **prime_recall**: ≤6 memories through the Stage-2 referent recall engine, using bounded neighborhood reads (#2497/#2519), with a ≤400 ms budget. If the cues are unchanged, keep the last primed set. If the engine is down, `retry_wait` and still project.
7. **project**: write Redis `orion:situation:latest` and publish on bus `orion:situation:state`, only when the revision changed.
8. **done**.

Every node checkpoints, so a crash resumes from the last node. At boot, the Redis projection is rebuilt from the last checkpoint. Until then, a turn renders "situation unavailable", never an empty situation.

## 5. State payload: `SituationStateV1`

Every field names its producer and reader. Facts point to episode memory; the full text lives there.

```text
SituationStateV1
  schema_version   "situation.state.v1"
  thread_id        "situation:juniper"
  revision         int            # turns record the revision they served
  updated_at       datetime
  last_event       {event_id, kind, correlation_id}
  juniper
    whereabouts    Fact | null    # distiller / same-turn claim → chat, outreach, Hub
    doing          Fact[] ≤3
    waiting_on     Fact[] ≤3      # follow_up memories
    affect_ref     {key, as_of}   # pointer to orion:juniper_affect:latest
  orion
    place          {home, body}   # settings; replaces the fixed Place line
    working_on     RunRef[] ≤3    # durable run state (curiosity, reading)
    self_view      SelfRef[] ≤2   # curiosity self:definition, self:lived:*
    recently_read  ReadRef[] ≤3   # reading.turn, voice=orion_read
  shared
    open_threads   Thread[] ≤5    # open loops with age; stale ones age out
    themes         ThemeRef[] ≤3  # spark concept profile + topic foundry, with trend
  present
    who            [display_name]
    camera         {stream_id, travels_with_juniper: bool}   # laptop webcam ≠ "Room"
    audience_mode  str
  recall
    cues           {referent_ids[], concept_ids[], from_revision}
    primed         Primed[] ≤6
    primed_at      datetime
    primed_revision int           # < revision means priming is behind
  lapsed           LapsedRef[] ≤3 # "forgot" vs "never told"

Fact     {fact_id, memory_id|null, gist ≤120, valid_from, valid_until|null,
          until_source: juniper_words|default_ttl, voice, confirmation,
          evidence[], supersedes|null}
Primed   {memory_id, gist ≤160, voice, confirmation, why: referent_id, score}
```

**Render budget** in the chat prompt: about 900 characters. Order: whereabouts → doing → waiting_on → primed → threads → themes. The cap trims from the end. A `default_ttl` end date renders as "as of <date>".

## 6. Intersections

- **Concept induction and the topic model:** the slow layer. Profile and topic deltas update `shared.themes`, and a referent's concept neighbors (bounded read, 16 nodes max) widen the cues. This replaces the per-stance concept reload.
- **Curiosity:** self-definitions, lived answers and in-flight runs become `orion.self_view` and `orion.working_on`, so Orion's concepts about themself sit in the same "now" as Juniper's day. Curiosity also reads the situation before choosing what to investigate.
- **Reading:** finished reading turns become `orion.recently_read` with the `orion_read` voice. Their landed concepts become cues.
- **Episode memory:** the one offline writer of facts. Lapsed facts come back to it as history ("was in Chicago, 10-04 to 10-08").

## 7. Writer fixes (prerequisite)

The state amplifies distiller errors, so these land first:

1. **End dates on every purpose.** Stop discarding `expires_at` (`validate.py:406-407`). The prompt asks "until when is this true?" for all purposes and returns Juniper's own words as `until_text`. Code resolves the date deterministically against `occurred_at` in Juniper's timezone. Only an `until_text` that literally appears in the evidence is accepted.
2. **Entailment check.** A corrective or contrastive statement ("corrected me", "not X", "no longer") must have the contrasted term in Juniper's own (`chat_prompt`) evidence. Otherwise it becomes `pending_confirmation` with high stakes.
3. **Repair** `c0dc86c8…` to `corrected`, with an audit event. Dry run, snapshot, then apply.

## 8. Risks

- **A wrong "now" is worse than none.** Section 7 lands first, and the Hub panel lets Juniper end or correct a fact.
- **Priming lags a fast second message.** The turn uses the previous primed set. `primed_revision` records this, and the eval measures how often it happens.
- **One writer, eight sources.** Order by revision, dedupe by event id, coalesce when backed up, never run in parallel.
- **Open loops are present on 100% of turns.** Measure their age distribution before setting the `open_threads` aging rule.
- **Overlap with the Stage-2 referent spec (#2496).** Its `referent_region` collector should be the priming engine. This design adds the cue source and the state, not a second search engine.

## 9. Order of work

1. **Writer fixes** (section 7). Proof: tests for date resolution, the entailment refusal on the Ogden case, and end dates kept on `happened` rows. The repair snapshot is in `/tmp/`.
2. **Situation graph in shadow.** Workflow, state, checkpoint, projection; chat doesn't read it. Proof: a week of revisions in the run views, plus the turns it would have changed.
3. **Chat reads the state.** It replaces the Place line, the three searches and the self-check search, flag on. Proof: the replay eval of 10-05 → 10-06 passes, and per-turn recall time in `recall_telemetry` drops.
4. **Curiosity, reading and concept wiring.**

## 10. Proposal-mode checklist

- **Capability change:** Orion carries a durable picture of the current situation and recalls by cue from it.
- **Data touched:** `episode_memory` (validity, one repaired row), a new LangGraph thread and checkpoints, Redis key `orion:situation:latest`, bus `orion:situation:state`, the chat prompt.
- **Privacy boundary:** none. Single user, shared life (Juniper, 2026-10-07).
- **Proof:** the turn's trace records the served `revision` and primed memory ids; the replay eval passes.
- **Dangerous failure:** a stale or confabulated fact stated as current. Mitigations: section 7, `default_ttl` rendered as "as of", the Hub correction control.
- **Rollback:** a flag per stage. `ORION_SITUATION_GRAPH_ENABLED=false` stops the run, and `ORION_SITUATION_CHAT_READ_ENABLED=false` restores the current chat path. Writer fixes revert by prompt-version pin.
