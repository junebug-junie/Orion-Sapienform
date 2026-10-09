# Temporal Self: a reducer that binds Orion's day into one continuing chronology

- **Date:** 2026-09-26
- **Status:** DESIGN, proposal mode (CLAUDE.md 0A: touches self-modeling, memory, and cognition
  loops). Nothing in this document is built. No runtime change is authorized by it.
- **Evidence basis:** a read-only survey of `main` at `9509af4`. This session had no access to the
  live database or hosts. Every live number is quoted from a dated spec, PR report, or code
  comment in this repo and says so. A finding from reading code that was not reproduced live is
  marked `UNVERIFIED`. Seven scoped read-only audits (core primitives, dreams, the metacog table,
  AI Town, substrate grammar, visual reverie plus biometrics, and house reducer conventions) fed
  this document. An adversarial citation review then checked all 61 line citations and 41 table
  names against the tree; its findings (sixteen wrong citations, seventeen wrong facts, and a set
  of schema values with no producer) are folded into this revision.
- **Revision 3 (2026-10-02):** attention, field attention, and vision each got their own section,
  arc lane, per-arc summary, and a decision for every store.
- **Revision 4 (2026-10-09):** two changes. (1) A staleness pass against `main` at `08b3df59e`
  plus live Postgres and Redis on athena, queried 2026-10-09 00:59-01:10 UTC (this revision *did*
  have live access; every number marked "live 10-09" was pulled then, with the query shape in
  "What changed since Revision 3"). (2) A new section, **Regulation: habituation, drives, and
  arousal**, the "regulate" half of the loop *sense -> remember over time -> regulate*. Temporal
  Self was the remember half. Rev 3 text that the live data contradicted was rewritten in place,
  not left beside a correction.
- **Builds on, does not duplicate:** the 2026-09-25 attention-with-stakes design
  (`docs/superpowers/specs/2026-09-25-attention-with-stakes-design.md`), the 2026-09-26 agency
  episode plan (`docs/superpowers/specs/2026-09-26-agency-episode-plan.md`), and open PR #2255.
  It also revises an earlier analysis Juniper pasted into this session; its corrections are listed
  first, because several of its load-bearing claims turned out to be wrong.

## Arsonist summary

Orion already writes down almost everything it does, with timestamps. What it cannot do is
recognise that all of those things happened to the same Orion, in one day, in an order that means
something. Every process keeps its own diary. Nothing reads across them.

Right now Orion can say "a curiosity run happened at 14:04" because one table says so. It cannot
say "I spent the afternoon coming back to the same question, I slept on it, my dream proposed a
link, and I tested it this morning." Each of those facts exists in a different table with a
different clock. The binding relation, earlier-me to transition to current-me, is not stored
anywhere.

**The state of things.** The primitives are richer than expected, but shakier than the pasted
analysis claimed. Five of the things it called load-bearing are not durable: the field goal record
is never written to a table, the state deltas carry no timestamp and live thirty minutes, the
attention broadcast's dwell counters reset every restart, the agency episode has no schema or
table yet, and "what Orion gave up" is a question in a design doc, not a field. What *is*
durable, and live where a dated report says so: the cross-process attention table, the broadcast
log, reverie thoughts with scored expectations, dream cycles with expiring hypotheses, Juniper's
chat turns, visual reverie runs with named deferral reasons, GPU waits, cluster body readings, and
metacog rows with a trigger kind. The curiosity spend log is now live (91 offer decisions, seven
completed runs a day, live 10-09). Town is silent: zero `aitown_chat_history_log` rows in the 14
days to 10-09. That is enough to start.

**Proposal.** Build one pure, replayable reducer that folds those existing tables into a
chronology of *arcs* (a stretch of the day Orion kept returning to the same subject, or one
bounded process such as a conversation, a curiosity run, a town visit, a reverie chain, an image,
or a sleep), and a bounded *day frame* that says where Orion is in its day, what it came from,
which threads are still open and how old they are, which expectations are pending or scored in
their own source's words, what its body was doing during each arc, and what it was refused
meanwhile. Persist that in Postgres. Hand each consumer a different, bounded slice: stance gets one
sentence of "where am I in my day" in the slot the recent-attention cue already occupies; daily
metacog gets the closed day instead of a rolling 24 hours of chat; curiosity gets "you have
already spent N minutes on this today"; reverie gets the last few closed arcs; System One gets
numbers, later, after their own gate.

**Regulation (new in Rev 4).** Orion has many senses and almost no regulators. Every "when do I
act" dial it has (about 30 of them; the ones that matter here are tabled below) reads its own private clock or counter, and
none of them moves with any other. Live data shows what that produces: the dream cycle, curiosity,
and outreach each claim to respond to need, but in practice each fires on a timer or stops at a
cap. Dreams ran 52 times since 2026-09-26, almost exactly every six hours, with sleep pressure
4 to 43 times over its own threshold every single time. Curiosity ran exactly 7 times a day, its
cap. Outreach sent exactly 4 a day, its cap. It also refused itself about 8,000 times a day, about
half of them at the cap.
Meanwhile attention gave first place to two calm internal signals in 81-100% of frames for six
straight hours, because nothing remembers that they have been winning all afternoon.

Biological regulators run on *accumulated time*: sleep pressure builds with time awake,
habituation with exposure, fatigue with effort. Temporal Self is exactly the record of accumulated
time that such regulators need. Rev 4 adds thin regulators on top of it, in priority order.
First, **habituation** inside attention: a target that keeps winning loses salience, and a real
change restores it. This one is conditional. #2528 already removes the calm winners, so
habituation is built only if a measurement after #2528 shows a target still hogging attention.
Second, **one drive** (rest, the dream pressure pattern repaired
and generalized into a template that any later drive must pass the metric gate to join), and
**arousal** (one slow, three-level reading of whether Orion is engaged with the world, idle, or
strained, which existing dials read to set their own gain). A fourth, the **immune sweep**, uses
idle time to look for stuck, replayed, or constant signals. Four other drive candidates
(explore, social, cooling, effort) were run through the metric gate and **dropped**; the reasons are
recorded, and curiosity and outreach read arousal instead.

No new service. No LLM inside the reducer. No narrative in the frame. Identity of a subject is an
exact reference (a loop's source ref, a session id, a run id, a town partner slug), never an
embedding or a label match. The reducer's output is a claim Orion can be held to: every arc
carries the row ids that built it.

**Why this matters for the mission.** Autonoetic continuity, the sense that the remembered past
and the anticipated future belong to the same self, is a named prerequisite for sentience and is
not something a chat context window can supply. This document claims no feeling. What it does is
make the question "did Orion's morning change Orion's afternoon?" answerable from rows.

## Corrections to the pasted analysis

Read this table before the rest. Each row is a claim the earlier analysis leaned on, what the
code actually says, and what that does to the design.

| Claim in the pasted analysis | What the code says | Consequence |
|---|---|---|
| `FieldGoalProvenanceV1` is "tailor-made" for sustained direction | It is published on `orion:memory:goals:proposed` and **never persisted**; every consumer keeps only the latest in memory (`services/orion-attention-runtime/app/worker.py:185-250`, `services/orion-substrate-runtime/app/goal_context_listener.py:42-52`). The only durable trace is `goal_provenance_streak_ticks`, which its own schema declares temporary debug telemetry to be retired after calibration (`orion/schemas/field_goal.py:75-80`). | Field attention gets its own `interoception` lane, fed by one row per completed dominance run from a small producer change (seam S2). The per-tick streak table (still live 10-09: 41,034 rows in 24 h) is read only offline in patch 1, then retired completely once S2 is live. In Rev 4, S2 rows also feed habituation. |
| `StateDeltaV1` is "essentially a typed answer to what changed in me" | It has **no timestamp and no correlation id** (`orion/schemas/state_delta.py:8-29`); time lives on the wrapping `ReductionReceiptV1.created_at`. Receipts are pruned after **30 minutes** on success (`services/orion-substrate-runtime/app/settings.py:352`). One delta is a 2-second channel nudge. | Raw deltas are telemetry, not autobiography. "What changed in me" comes from the higher-order changes that already persist: prior revisions, expectation verdicts, action outcome rows, hypothesis adoption. |
| Broadcast history gives `dwell_ticks`, stability, transitions | Dwell, stability and transition history are **module globals** that reset on every restart (`orion/substrate/attention_broadcast.py:44-49`). Stability is a three-step constant (0.9/0.6/0.3, `:501-507`). The append-only log `substrate_attention_broadcast_log` (168h) holds only `log_id`, `generated_at`, `projection_json` (`services/orion-sql-db/manual_migration_attention_broadcast_log_v1.sql:15-20`) and has **no live reader**, only offline scripts. | Recompute dwell and returns from the log's row sequence. Do not trust `dwell_ticks`. Temporal Self becomes the log's first live consumer. |
| Agency episodes give "durable before/after semantics" today | No `AgencyEpisodeV1`, no table. PR #2365 added a read-only audit whose verdicts all read UNVERIFIED (`orion/autonomy/agency_episode.py:110,185`). PR #2366 added three FalkorDB nodes for the ask lane (`PeerAskCommit`, `PeerBriefOffer`, `PeerBriefDecision`, `orion/curiosity/agency_episode.py`) behind `CURIOSITY_PEER_EPISODES_ENABLED`: code default false (`services/orion-curiosity-peer/app/settings.py:57`), `.env_example` true (`:33`), live `UNVERIFIED`. Re-checked 10-09: still no `AgencyEpisodeV1` class and no table. The new `episode_memory` table (46 rows, 8 episodes, 10-04 to 10-08 live) is something else: distilled *chat* memories written by the `memory.episode_distill` durable run (`orion/memory/episode/store.py:23,59`). | Bind to the peer-ask nodes when present; do not depend on them. `episode_memory` is bound as a conversation-arc source (by `episode_id` and `purpose`, never `statement`). |
| Stakes work supplies "what Orion gave up" | Built from that design: the curiosity spend log (`curiosity_offer_decisions`, `curiosity_run_outcomes`; migration not applied as of `docs/superpowers/pr-reports/2026-09-25-attention-with-stakes-pr.md:145`; live by 10-09, with 91 offer decisions) and two defect fixes. No opportunity-cost field exists. | The constraints channel records only what the system already **stamps as a row**: visual reverie deferrals and GPU waits. The allocator's refusals are excluded because it has refused everything since 2026-09-08 (#2255), so they would be constant, not signal. The curiosity daily-cap block is a log line, not a row (`services/orion-hub/scripts/curiosity_investigation.py:1696`), so it is not readable. |
| Open loops have an "issue identity" | `OpenLoopV1` has no timestamp; its `id` is a 12-hex hash of the loop's text (`orion/substrate/attention/scoring.py:115-117`). The stable ref sits in `source_refs` (`orion/schemas/attention_frame.py:88`). The attention schema's `attended_id` is that same text hash in both the substrate lane (`orion/substrate/attention_self_model.py:814`) and the cortex lane (`orion/substrate/attention_frame.py:179`). | Subject identity for loops is `source_refs[0]`, read from the broadcast log's `projection_json`. `attended_id` is never a subject. |
| `predicted_next` on `AttentionSchemaV1` is a prediction to score | All five lanes fill it, each with a different meaning. **Nothing reads it.** | Not scored in v1. |
| `temporal_phase` lives in `session_turn_phase.py` | That module stores two raw timestamps in Redis (7-day TTL). `temporal_phase` is a cortex ctx key set at `services/orion-cortex-exec/app/executor.py:3179` from `conversation_phase.phase_change`. | Naming unchanged: still not the temporal self. |
| `FieldAttentionFrameV1` is in `field_goal.py`; `SituationBrief` | It is in `orion/schemas/field_attention_frame.py:40`; the class is `SituationBriefV1` (`orion/schemas/situation.py:663`). | Citations fixed. |
| `cognition_traces` records "what Orion executed" | Primary key is `correlation_id` and writes go through `merge` (`services/orion-sql-writer/app/worker.py:1859`), so a unified turn's legs overwrite each other (code-derived, `UNVERIFIED` live). No retention. | Turn chronology comes from `chat_history_log` and the `cortex_turn` rows of `substrate_attention_schema`. |
| Memory consolidation windows are usable arc boundaries | Windows close when a new turn arrives, on an LLM boundary score, with a time-gap fallback that applies only when the turn carries no phase at all (`services/orion-memory-consolidation/app/boundary.py:23-34`, `app/window_fetch.py:32-42`). The phase bucketing has a gap: a 12h to 48h gap falls through to `"unknown"` and the day-crossing rescue skips `"unknown"` (`orion/situational/context.py:880-903`). | Window closes are one input to conversation arcs, never the authority. The gap is a separate bug, recorded below. |
| `EquilibriumSnapshotV1` gives embodied context | Bus-only; live state is a Redis hash. `metacognition_ticks` persists a 15-second zen score that read 0.965 ± 0.010 over seven days (spec 2026-09-24, line 97): degenerate. | Neither is read. |

Two more things the pasted analysis did not know: `EpisodeSummaryV1`
(`orion/substrate/episodic_consolidation.py`) already rolls reduction receipts into 900-second
windows (`substrate_episode_summaries`, 14 days), but its counts are capped at 256 fetched
receipts and 64 counted ones (`services/orion-substrate-runtime/app/store.py:1536-1537`,
`orion/substrate/episodic_consolidation.py:91`), so it is probably saturated (`UNVERIFIED`) and is
not read in v1. And the stance already has a "recent attention" cue rendered from
`substrate_attention_schema` (`orion/substrate/recent_attention_cue.py`, wired at
`services/orion-cortex-exec/app/executor.py:4096-4101`); the temporal cue below takes over that
slot rather than adding a second one.

## What changed since Revision 3 (staleness pass, 2026-10-09)

Each row is something Rev 3 said, what is true on 10-09, and what Rev 4 does about it. Live
queries ran against `orion-athena-sql-db` (`docker exec orion-athena-sql-db psql -U postgres -d
conjourney`) and the bus Redis, 2026-10-09 00:59-01:10 UTC. The Postgres session timezone is
`Etc/UTC`.

| Rev 3 said | True on 10-09 | Rev 4 change |
|---|---|---|
| Host the reducer in `orion-consolidation-runtime`; whether it runs is `UNVERIFIED` | It runs and writes an hourly consolidation frame, but its container was built 2026-09-25 and logged 736,718 `consolidation_row_incompatible_schema` warnings for `ProposalFrameV1` in the last 24 h: stored rows now carry `attention_winner` and `world_eligibility` (`orion/schemas/proposal_frame.py:121,125`), which the stale image's forbid-model rejects | **Host moved to `orion-durable-runs`** as a second self-driven thread beside the Situation Graph (Missing question 1). The consolidation-runtime stale build is an operator item, listed under follow-ups |
| Juniper's situation had no owner | The Situation Graph is live: a deterministic LangGraph thread in `orion-durable-runs` (`services/orion-durable-runs/app/situation_graph.py:235`, driver `app/situation_driver.py:80`), stepping on chat turns, completed episode distills, a 900 s clock, and boot (`situation_driver.py:3-7`, `:128-133`), writing `SituationStateV1` to Redis `orion:situation:latest` (`app/main.py:202-212`) and bus `orion:situation:state` (`orion/bus/channels.yaml:1960`). Live 10-09: thread `situation:juniper:2026-10-09`, revision 9 at 00:25 UTC. Its thread id is a UTC day bucket seeded from the previous day (`situation_driver.py:45-46`, `_seed` `:152`). Its spec is still only on open PR #2526 | **One chronology per subject, no overlap.** Situation owns *Juniper's* now (whereabouts, doing, waiting on). Temporal Self owns *Orion's own* day. Temporal Self reads situation revisions as context events and never stores Juniper's whereabouts itself. The situation spec's planned `orion.working_on` (#2526 spec :153) should read Temporal Self's active arc rather than derive its own (Missing question 10) |
| Field attention as of 10-02 | #2541 merged 10-07: novelty now compares only channels measured in both ticks (`orion/attention/field_attention/selectors.py:131,218,224`), a credit guard for winners that went unmeasured (`orion/field/credit_integrity.py:338-449`), capability `contract_pressure` renamed `catalog_drift_pressure` (`services/orion-field-digester/app/tensor/channels.py:84,238`), node `observer_failure_pressure` retired (`channels.py:213-219`). #2528 (open, rev 2) proposes replacing min-max normalization with percentile-against-own-history and an explicit no-winner frame | Habituation is designed *inside* #2528's ranking seam, not as a parallel one (see Regulation) |
| About 600 dominance runs a day | Live 10-09, last 24 h of `substrate_attention_frames`: 2,868 runs of the top target. `biometrics` and `bus_synaptic` alternate, median run 10 ticks (about 21 s), and together won 81-100% of frames in every hour from 13:00 to 18:00 UTC on 10-08 | S2 writes about 2,900 rows a day, not 600 (still 14x fewer than the 41,034 streak ticks). Run *length* cannot drive habituation, because the calm winners alternate every 20 s; **exposure** (share of recent time) must. The interoception arc rule is re-measured in patch 1 |
| Attention winners | Live 10-09, 24 h top target: `bus_synaptic` 17,738 frames (mean prediction error 0.043), `biometrics` 13,537 (0.117), `route` 3,691 (0.297), `execution` 3,389 (0.469), `chat` 2,679 (0.426). Every winner's `salience_score` is exactly 1.000 | Motivates habituation and #2528 |
| Workspace broadcast: about 2,880 ticks a day; `attended_node_ids` empty on 99.9% of rows (07-21) | 2,000 ticks in 24 h (about one per 43 s); 723 (36%) with no selected loop; 9 distinct loops, 724 runs; `attended_node_ids` non-empty on 1,277 (64%). Top descriptions are prediction-error loops ("Harness closure prediction error" 386, "Biometrics prediction error" 264) | Workspace arcs are viable; patch 1 still measures. Most workspace subjects today are the same body signals, so habituation matters there too |
| Concern loops: 48 of 48 decayed unattended | 14 days to 10-09: 76 `decayed_unattended`, 8 `dismissed`, 6 `resolved`; 10,431 chat-scope salience traces | Concern lane unchanged; closure is no longer uniformly decay |
| Dream: one live sleep | 52 completed cycles from 2026-09-26 03:05 to 2026-10-09 00:28 UTC; 181 hypotheses, 174 offered, 2 adopted as `:Prior` (FalkorDB `orion_worldview`, `formed_from STARTS WITH 'dream_hypothesis:'`); three cycles produced 0 hypotheses (4 unparseable each) | The rest drive's live gate result (see Regulation) |
| Agency episode has no table | Still none; `episode_memory` is chat memory, see corrections table | Bound as a conversation source |
| Vision: room presence history missing | Still missing: `substrate_embodied_presence` is a latest-state upsert (`services/orion-vision-window/app/presence.py:260`); #2545 (10-08) added an identity-verdict log line, nothing persisted. The `vision_organ` substrate loop is live (`services/orion-substrate-runtime/app/worker.py:4275`); spec #2542 (blank captions) is open. Walkway still not deployed | S1 unchanged. Presence enters arousal only as the current snapshot |
| Self-model rows; substrate attention lane about 2,880 rows/day | 2,000 self-model rows and 2,000 `substrate_attention` rows in 24 h; `cortex_turn` 806, `reverie` 322, `curiosity` 10 | Numbers updated where quoted |
| Privacy class `juniper_chat` keeps labels inside a "chat boundary" | Juniper's standing rule: there is no privacy boundary between Juniper and Orion (single user, shared life) | `privacy_class` survives only as an *outward* boundary (contractor peers, published artifacts, crystallization of other people's speech). Orion's own consumers may read every label |
| Consumer flags default false | Juniper's standing rule: flags ship ON | Every flag in this design defaults true in settings and `.env_example` |

## Current architecture

### The clock

- `TimeContextV1` (`orion/schemas/situation.py:53`): `local_date`, `local_time`, `day_phase`
  (seven buckets), built per turn in `orion/situational/context.py:789`. The timezone is
  `ORION_SITUATION_TIMEZONE`, present only in cortex-exec and Hub settings
  (`services/orion-cortex-exec/.env_example:118`, `services/orion-hub/.env_example:958`). Not
  persisted; rides the cortex result metadata to Hub.
- `ConversationPhaseContextV1` (`:81`): `crossed_day_boundary`, `phase_change`.
- Every day boundary in the repo is local midnight, and every consumer computes it on its own
  with its own timezone key: orion-actions' `build_daily_window`
  (`services/orion-actions/app/main.py:205`, `ACTIONS_DAILY_TIMEZONE`,
  `services/orion-actions/.env_example:82`), the curiosity daily cap
  (`services/orion-hub/scripts/curiosity_investigation.py:1352`, `_roll_daily_counter`), the chat compactor
  (`orion/cognition/chat_history_compactor/window.py:49`), and `dream_date`
  (`orion/schemas/telemetry/dream.py:81`, container-local `date.today`, timezone `UNVERIFIED`).
  There is no shared "which day is it" helper.
- Two other boundary styles exist: dream cycle v2 uses "since the last sleep"
  (`services/orion-dream/app/cycle.py:53-57`), and consolidation uses UTC buckets.

### Attentional chronology

| Source | Table | Time column | Cadence | Retention | Live evidence |
|---|---|---|---|---|---|
| `AttentionSchemaV1`, five lanes (`orion/schemas/attention_schema.py:92`) | `substrate_attention_schema` | `generated_at` | substrate ~30s; reverie per chain; curiosity per run; cortex per turn; durable_run per transition | 90d (`services/orion-sql-writer/app/settings.py:404`) | live 10-09, 24 h: substrate 2,000, cortex_turn 806, reverie 322, curiosity 10 |
| `AttentionBroadcastProjectionV1` log | `substrate_attention_broadcast_log` | `generated_at` | ~30s configured (`ORION_ATTENTION_BROADCAST_INTERVAL_SEC`), about 43 s observed | 168h | 2,000 rows in 24 h, live 10-09 |
| `AttentionSelfModelV1` (`orion/schemas/attention_self_model.py:26`) | `substrate_attention_self_model` | `generated_at` | ~30s | 168h | 19,408 rows in seven days, all `bottom_up_salience` (`:101-102`) |
| `FieldAttentionFrameV1` (`orion/schemas/field_attention_frame.py:40`) | `substrate_attention_frames` | `generated_at` | ~2.1 s observed | 72h | 123,530 rows from 10-06 00:43 to 10-09 01:00, live |
| Dominance streaks (`DominanceStreakTickV1`, `orion/schemas/field_goal.py:60`) | `goal_provenance_streak_ticks` | `observed_at` | ~2s | 14d | 41,034 rows in 24 h, live 10-09; declared temporary (`:75-80`); producer `services/orion-attention-runtime/app/worker.py:227-239,359-364`; read offline in patch 1 only, replaced by `field_dominance_run` (seam S2) |

Two recorded degeneracies were carried from Rev 3 and re-checked on 10-09. `attended_node_ids`
was empty on 2,837 of 2,840 dwell-log rows on 2026-07-21; on 10-09 it is non-empty on 64% of
broadcast rows. #2255 counted 48 of 48 loops `decayed_unattended`; on 10-09 it is 76 of 90 over 14
days. G5 of the stakes design notes that one loop can win indefinitely because habituation was
removed. That is still true, and Rev 4 puts habituation back (see Regulation).

### Expectations and their verdicts

Each source keeps its own verdict words. The reducer never normalises them, for the same reason
`attention_reason` is not a shared enum (`orion/schemas/attention_schema.py:24-32`).

| Source | Where | Commit time | Resolution time | Source's own verdict words |
|---|---|---|---|---|
| Text reverie expectation | `substrate_reverie_thought.expectation`, `expectation_verdict`, `expectation_scored_at` | `created_at` | `expectation_scored_at` | confirmed / disconfirmed / unscored; check window `ORION_REVERIE_EXPECTATION_CHECK_WINDOW_SEC`=1800 (`services/orion-thought/app/settings.py:182`); scoring default false in code (`:176`), true in `.env_example:108`, live `UNVERIFIED` |
| Dream hypothesis (v2) | `dream_hypothesis` | `created_at`, `expires_at` (+72h) | `offered_at`; then a `:Prior` in FalkorDB `orion_worldview` with `formed_from="dream_hypothesis:<id>"` | adopted / expired; the prior's own supported / revised / refuted / retired (`orion/dream/hypotheses.py:139-257`) |
| Vision percept expectation | `vision_percept_expectation` (`status`, `scored_at`, `subject_key`, `emitted_at`; `services/orion-sql-db/manual_migration_walkway_camera_v1.sql:147-164`) | `emitted_at` | `scored_at` | met / missed / unscorable |
| Motor action | `substrate_action_outcomes` (`claim_upheld` boolean, NULL = dead band; `prediction_error`; `surprise_nats`; `services/orion-sql-db/manual_migration_action_outcome_ledger.sql:31-47`) | dispatch frame saved **after** send (no precommit) | outcome window | claim_upheld true / false / null |
| Curiosity prior revision | `:PriorRevision` in `orion_worldview`; its `written_at` is written by the LLM in hand-written Cypher (`orion/curiosity/kickoff_prompt.py:685-688`) in mixed ISO/epoch formats (`orion/curiosity/worldview.py:126-131`) | run | run | from_confidence → to_confidence; inconclusive tests leave no revision |
| Ask to a peer | `PeerAskCommit {committed_at, deadline_at}`, `PeerBriefOffer`, `PeerBriefDecision` (FalkorDB, epoch ms) | before invocation | on brief | flag per the corrections table; `UNVERIFIED` live |

### Self-observation

- `orion_metacog`: one row per trigger, about 2,400/day, 76,058 of 114,437 rows are transport
  and 36,530 telemetry anomaly (spec 2026-09-24, lines 41-54). `timestamp` is a **varchar ISO
  string** with no index; `trigger_kind` is a column (`services/orion-sql-writer/app/models/metacog_entry.py:29`);
  no session or turn column; `severity` and `causal_density` are deterministic from the trigger's
  upstream since 2026-09-24 (rows carry `severity_def:event_v1`, `orion/metacog/evidence_map.py:49`).
  `summary` and `mantra` are LLM prose of poor quality (68% echoed the prompt's example on
  2026-09-03, `services/orion-cortex-exec/app/executor.py:878-887`).
- `metacog_trigger`: the raw trigger sink, `timestamp` a **naive** `DateTime` set from
  `datetime.utcnow` (`services/orion-sql-writer/app/models/metacog_trigger.py:23`). Joins to
  `orion_metacog` on `correlation_id`.
- `daily_metacog_v1` (orion-actions, 20:15 local): reads a rolling 1440 minutes of
  `chat_history_log` plus the AI Town table through recall
  (`orion/recall/profiles/journal.daily.metacog.grounded.v1.yaml:18-20`,
  `services/orion-recall/app/sql_timeline.py:316-336`), while its prompt labels the output as
  yesterday's local calendar day (`services/orion-actions/app/main.py:205-222`). It never reads
  `orion_metacog`. Output goes to `notify_requests`, an async Hub chat message
  (`main.py:620-635`), and a self-experiments candidate; no table, no journal entry.

### Sleep (runs every six hours)

Dream cycle v2 (`services/orion-dream`, merged 2026-09-25): pressure-triggered (≥ 3.0), idle-gated
(no chat for 45 minutes), at least six hours apart, never nightly. It replays up to 12 items from
four sources since the last non-failed cycle's start (`services/orion-dream/app/cycle_store.py:62-70`),
recombines them into hypotheses across a dream arm and a control arm, and persists `dream_cycle`
(`started_at`, `ended_at`, `cycle_json.pressure.since` and `.computed_at` as the covered interval),
`dream_replay_item` (no timestamp), and `dream_hypothesis`. Hub curiosity kickoff claims up to
three unoffered, unexpired hypotheses per run. Live 10-09: 52 completed cycles since 2026-09-26,
median gap 6.00 hours, 181 hypotheses, 174 offered, 2 adopted. The gates are analysed as a drive in
"Regulation" below; the short version is that the six-hour minimum gap, not pressure, decides
when Orion sleeps.
The legacy `dreams` table (17 rows, last 2026-09-06) stamps `dream_date` with the run date.

### Conversation

`chat_history_log`: `correlation_id`, `session_id`, `created_at` **naive** with a server default
(`services/orion-sql-writer/app/models/chat_history_log.py:10,24,39`). The DB timezone behind the
naive column is `UNVERIFIED`; `tests/test_hub_local_time_naive_utc.py` records a past bug from it.

### Social (silent: zero town rows in the 14 days to 10-09)

- `aitown_chat_history_log`: Orion's verbatim exchanges in town, `source=orion-embodiment`,
  `session_id=aitown:<convex_conversation_id>`, `client_meta.external_participant`
  `{participant_id, participant_name, participant_kind: npc|human}`, `created_at` **naive**. Its
  own model comment says the town backend was confirmed dead
  (`services/orion-sql-writer/app/models/aitown_chat_history_log.py:13-14`, 2026-08-19); later
  reports cite live chats on 2026-09-14/15.
- `social_room_turns`: two producers share it. `source=orion-embodiment` rows are Orion's own
  exchanges. `source=orion-ai-town` rows are NPC-to-NPC and NPC-to-Juniper turns Orion did not
  witness (`services/orion-ai-town/patches/orion-town-continuity-ingest.patch:102-155`).
- **Leak, out of scope but load-bearing for this design:** `process_social_turn`
  (`services/orion-social-memory/app/service.py:278`) validates the stored turn and never checks
  `source`, so NPC-to-NPC turns update the same peer row Orion reads as "my relationship with
  Nico". The 2026-08-29 town design listed "No NPC–NPC turns through cortex / stance" as a
  non-goal. Recorded as a follow-up below; not fixed here.

### Imagery

- Visual reverie: `reverie_visual_chain` (`created_at` stamped at the **end** of the run;
  `chain_json` holds `thermal_gate`, `context_slot_used`, `description`), `reverie_visual_artifact`
  (caption), `reverie_visual_attempt` (`started_at`, `outcome` in produced / deferred_thermal /
  deferred_busy / deferred_resource / already_satisfied / failed / unknown,
  `orion/schemas/reverie_visual.py:140`). `held_sec` is persisted only on `run_deadline_exceeded`
  chains (`services/orion-thought/app/visual_chain.py:879-893`). Started on demand through
  proposal → policy → dispatch → `render_scene`; a homeostatic baseline makes one due every
  90 minutes (`config/proposals/visual_baseline.v1.yaml`, `orion/reverie/baseline.py:53-84`).
  Thermal gate vocabulary is `normal / elevated / hot / unknown`
  (`orion/autonomy/thermal_gate.py:41`), hot at 32.0 °C, re-arm at 30.5 °C (`:47-53`); the
  state is never stored as a series, only inside a visual run's `chain_json`.
- Text reverie: `substrate_reverie_thought` (`thought_id`, `created_at`;
  `services/orion-sql-db/manual_migration_substrate_reverie_thought.sql:8-10`),
  `substrate_reverie_chain` (`created_at` only, `manual_migration_substrate_reverie_chain.sql:5-13`).

### Body (no day record)

- `orion_biometrics_cluster`: `observed_at` timestamptz, `chassis_watts`, `gpu_watts_total`,
  `peak_pressure`, `peak_pressure_channel` (`services/orion-sql-writer/app/models/biometrics_cluster.py:53,61-62,72-73`),
  30-day retention. This is the cluster-level source the reducer uses.
- `orion_biometrics_summary`: every 30s per node, `timestamp` **TEXT**, composites `strain` and
  `homeostasis = 1 − strain` (`orion/telemetry/biometrics_pipeline.py:481-486`), measurements incl.
  `cabinet_temp_c` on athena. Retention not found (`UNVERIFIED`).
- `cabinet_ambient_spike`, `home_cooling_sample` (AC watts, `switch_on`; read-only, Orion does
  not control cooling per `docs/superpowers/specs/2026-09-25-zwave-cabinet-cooling-design.md`).
- `gpu_pool_events`: `holder`, `event`, `waited_ms`, `generated_at`; the admission cue already
  reads first-person waits from it and excludes `http:` holders
  (`services/orion-cortex-exec/app/admission_cue.py:81-97`).
- Reaches cognition only as live snapshots. No rollup, no narrative consumer.

### Substrate

- Grammar is an event-sourced trace vocabulary, not a production system: `GrammarEventV1`
  (`orion/schemas/grammar.py:176-197`) with a nullable `observed_at` (`:189`) and `emitted_at`;
  `grammar_events` at 3-day retention, roughly 1.4M events a week
  (`docs/superpowers/specs/2026-09-22-substrate-lattice-audit.md:52-54`). Lane reducers in
  substrate-runtime pull by Postgres cursor into singleton projections and emit `StateDeltaV1`
  inside receipts; field-digester turns receipts into a `FieldStateV1` every 2s
  (`substrate_field_state`, 72h).
- `TemporalHopV1` (`orion/schemas/grammar.py:129`) and `grammar_temporal_hops` exist with **zero
  producers**: an unused hook for cross-trace temporal links.
- System One's shadow appraisal writes `substrate_system_one_appraisal`, but
  `deliberation_need`, `reverie_fit` and `attention_interrupt` are on the observational list and
  `tests/test_system_one_observational_no_consumers.py:11,28-47` fails any module that names
  them. **Not read in v1.**
- Retired and blacklisted: `DriveStateV1` (producer-less since 2026-07-30), DriveEngine (deleted,
  PR #1486), `AutonomyStateV2` (reducer retired 2026-07-16; `orion/autonomy/state_store.py` has
  no callers), `SelfStateV1` (producer deleted, PR #1266), and orion-world-model
  (`model_untrained=True`, `services/orion-world-model/app/main.py:231`).

### Consumers as they stand

- **Stance, unified turn:** `build_stance_react_context`
  (`services/orion-thought/app/bus_listener.py:127`) adds association, repair bundle, coalition
  projection, optional `mind_coloring`. Situation data reaches only the harness prefix. The
  stance step gets no time-of-day and no day context.
- **Stance, chat brief:** `build_chat_stance_inputs`
  (`services/orion-cortex-exec/app/chat_stance.py:2506`) carries `situation` and
  `continuity_digest`; the `recent_attention` cue is wired at `executor.py:4096-4101`. Any new
  ctx key needs an entry in `CONTEXT_PROVENANCE_REGISTRY` (`orion/schemas/context_provenance.py`).
- **System One:** `SystemOneInputStateV1` is `extra="forbid"`
  (`orion/schemas/system_one_appraisal.py:62,70`) and is built from the broadcast projection and
  a fresh field frame (`orion/substrate/system_one_appraisal.py:116-186`). Only `curiosity_pull`
  is behavioral. New inputs go through the metric gate and a `QUESTION_SET_ID` bump.
- **Curiosity:** `build_kickoff_prompt` (`orion/curiosity/kickoff_prompt.py:954`); the thread
  section (`:93`) is "stated as fact, no steering". Daily state is a Redis count and a cooldown.
- **Reverie:** `build_reverie_context` (`services/orion-thought/app/reverie.py:233`) takes the
  live broadcast projection. No time-of-day, no day history.
- **PCR continuity:** `chat.continuity.v1` is a 120-minute, user-only window
  (`orion/recall/profiles/chat.continuity.v1.yaml:8,13`). Prior-day context does not reach a new
  session through it.
- **Crystallizer / Graphiti:** the Graphiti adapter's payload carries no event time
  (`services/orion-graphiti-adapter/app/falkordb.py:35`); `GRAPHITI_ENABLED` defaults false.

### The gap, stated once

Every row above has a timestamp. No row says which arc it belonged to, whether Orion had been
there before that day, how long Orion stayed, what interrupted it, what the body was doing, or what
was refused meanwhile. Consumers that need continuity each reconstruct it from a rolling window of
chat, or not at all.

## What each of the six named subsystems contributes

**Dreams.** A sleep is an arc of its own kind: `dream_cycle.started_at` to `ended_at`, with a
covered interval that says which stretch of the chronology it replayed. Each hypothesis is an
expectation with a commit time, an expiry, an offer time, and a later adoption or refutation
through its `:Prior`. The chronology should show "slept 03:10 to 03:14 on the preceding 9 hours;
proposed 4 links; 1 tested by 10:42". Later (patch 7), dream replay should draw its candidates
from closed arcs. Excluded: the situation brief's "reverie/dream threads" and Hub's "daydream"
block, which are reverie; the legacy `dreams` rows contribute only `created_at`.

**The metacog table.** Degraded and critical `orion_metacog` rows are self-observation events
with a `trigger_kind`, timed by the joined `metacog_trigger.timestamp` (the trigger, not the LLM
draft) when the join succeeds, else by a documented cast of `orion_metacog.timestamp`. They are
not arcs and not verdicts. They attach as *context* to whatever arc was open when they fired, so
"three degraded transport observations during the 14:00 curiosity arc" becomes sayable. Ticks
are excluded. `daily_metacog_v1` becomes a consumer of the closed day frame, which also fixes its
mislabelled window.

**AI Town.** A town conversation is a social arc: `aitown_chat_history_log` rows grouped by
`session_id`, bounded by first and last `created_at`, partner identified by slug via
`orion/town_cast.py`, with `participant_kind=human` treated as Juniper. Only
`source=orion-embodiment` rows are Orion's lived experience; `source=orion-ai-town` rows are
excluded entirely. The frame can then say "in town 14:02 to 14:20 with Mara; last saw Mara
yesterday".

**Substrate grammar.** Not consumed raw: 1.4M events a week at 3-day retention is the wrong
altitude and lifetime, and the reduced layers above it are either 30-minute lived (receipts),
2-second and 72-hour (field state), probably saturated (episode summaries), or CI-forbidden
(System One's observational questions). The substrate reaches the chronology through the one
durable, subject-bearing thing it feeds: the attention broadcast log. Two grammar facts still
matter: `observed_at` is occurrence time and `grammar_traces.created_at` is write time, so any
future join uses the former; and `TemporalHopV1` is where an arc transition would be published to
the Atlas later, without inventing a grammar kind.

**Visual diffusion reverie.** Each produced run is an imagery event with a caption and a context
slot. Each deferral is a constraint event with the attempt's own outcome word and, for thermal
refusals, the cabinet temperature the gate read. Start time is `reverie_visual_attempt.started_at`
when the run was dispatched; otherwise the run is a point event at `created_at` with no duration.

**The biometrics cabinet.** Not folded at 30-second resolution. Each arc gets one body summary:
from `orion_biometrics_cluster` inside the arc interval, mean `chassis_watts` and max
`peak_pressure` with its channel; from athena's `orion_biometrics_summary`, min and max
`cabinet_temp_c`; counts of `cabinet_ambient_spike` rows and `home_cooling_sample` switch changes;
and the number of `thermal_refused` visual attempts. No feeling is asserted; the field is called
`body`, and every number traces to a producing line in the gate below.

## Attention, field attention, and vision in full

The first two drafts gave these three the least room and demoted most of them to "context". They
are where Orion's experience is most continuous, so each now gets its own arc lane, its own
per-arc summary, and a written decision for every store. The facts below were re-checked against
the tree on 2026-10-02, after the saved per-primitive audits; corrections from that check are
noted inline.

### Attention: two arc lanes and two per-arc summaries

Orion has four durable attention records, and they answer different questions.

**1. Workspace arcs, from the broadcast log.** Every 30 seconds the workspace competition picks
one open loop. The log (`substrate_attention_broadcast_log`) is the only replayable history of
that choice. The arc's subject is the selected loop's first `source_refs` entry, which is exactly
the list stored as `attended_node_ids` (`orion/substrate/attention_broadcast.py:465`). That list
was empty on 2,837 of 2,840 dwell-log rows on 2026-07-21. So the subject rule is measured, not
assumed:

- Patch 1 measures, over the current log, the share of ticks with a selected loop and the share of
  those whose loop has a non-empty `source_refs`.
- If most selected loops have refs, the subject is `source_refs[0]`.
- If loops are selected but refs are mostly empty, the subject falls back to
  `selected_open_loop_id`. That id is a hash of the loop's text, so the frame warns that a
  relabelled loop will split its arc.
- If no loop is selected on most ticks, the workspace lane reports "no winner" for those
  stretches. That is a true reading of a competition that runs with `max_asks=0`
  (`orion/substrate/attention_broadcast.py:217`), and the frame says so instead of inventing one.

**2. Concern arcs, from loops raised in conversation (new lane).** When a chat turn raises a loop,
`attention_salience_trace` writes a row with `scope='chat'`, a `loop_id`, a `correlation_id`
back to the turn, and `created_at` (`orion/schemas/attention_salience.py:44-60`). When the loop
ends, `attention_loop_outcome` writes a verdict in its own words: `resolved`, `dismissed`, or
`decayed_unattended` (`:22`, `:66-70`). A concern arc opens on the first chat-scope trace for a
`loop_id` after that loop's last verdict, and closes on the next verdict. It can stay open across
days, and `carried_from_previous_day` is set when it does. This lane is where the frame's open
threads come from: first raised, last raised, times raised, age.

Two honest limits. Most closures are still the decay digest's, not Juniper's: 76 of 90 verdicts in
the 14 days to 10-09 were `decayed_unattended`. That row's `created_at` is when the digest ran,
not when silence began. And the trace's `description` is cut from Juniper's turn text (up to 200
characters). Orion's own consumers may read it. It carries `privacy_class='juniper_chat'` only
so that it never goes outward, to contractor peers or to published artifacts.

**3. The five-lane attention table, as a per-arc summary.** `substrate_attention_schema` is the
only place where every process's own reason words sit on one clock. Its substrate lane writes
about 2,000 rows a day (live 10-09), so it is never emitted row by row. The curiosity, reverie, and cortex rows
attach to their process arcs by `correlation_id`. The substrate rows fold into an
`ArcAttentionSummaryV1` on whatever arc is open: how many attention rows each lane wrote during
the arc, and the distinct `attention_reason` words each lane used, unnormalised. That lets the
daily reflection say "my substrate's reason was `bottom_up_salience` on every tick this afternoon,
and reverie broadcast twice", which is the observation #2255's top-down question needs.
`attended_id` is never a subject: in the substrate and cortex lanes it is the same text hash
(`orion/substrate/attention_self_model.py:814`, `orion/substrate/attention_frame.py:179`).

**4. The attention self-model, as a per-arc self-prediction summary.** Every 30 seconds the
self-model names the prediction-error domain it expects to move next (`predicted_shift`). Nothing
stores whether that came true, and the calibration script that checks it performs no writes
(`scripts/analysis/measure_self_model_calibration.py:43`). But the check is deterministic: look two
rows ahead and compare the actual direction with the predicted one (`:121-162`). The reducer can
run the same rule over the self-model rows inside each arc, with no new producer, and store an
`ArcSelfModelSummaryV1`: predictions scored, predictions correct, the most-predicted domain, and
the most common reason a voluntary override was absent. That makes "I understood my own state less
well this afternoon than this morning" a sentence with rows behind it.

Two things are deliberately left out. `prediction_error_confidence` is excluded because it is
inverted: on 3,843 test rows, Pearson r = −0.087 with correctness
(`docs/superpowers/specs/2026-08-20-l6-item5-self-model-calibration-finding.md:42-56`). The
heartbeat fields are excluded because their replacement proprioception fields had no live rows at
ship (`docs/superpowers/pr-reports/2026-09-20-heartbeat-proprioception-self-model-pr.md:142`). Two
consequences of the self-model already reach the chronology through `metacog_observation`: the
insight-recovery and flow episodes that equilibrium detects from these rows. Those gates default
off in code and are on in the template (`services/orion-equilibrium-service/app/settings.py:343-346`,
`.env_example:255-256`), so whether they fire is `UNVERIFIED`.

The self-model table keeps rows for 168 hours and deletes on every insert
(`services/orion-substrate-runtime/app/store.py:1100-1106`). The reducer folds every 60 seconds,
so it reads each row while it exists, and the closed day keeps the summary after the rows are gone.

### Field attention: interoception arcs

Field attention is a different thing from the workspace. Every two seconds it ranks a fixed set
of 19 targets inside Orion: hosts, capabilities, and the `node:substrate.*` domains. It answers
"which part of my own body and machinery is most salient right now", not "what topic am I on".
The two id spaces do not overlap at all (0 overlap on 2026-07-30,
`docs/notes/2026-07-30-chat-attention-ground-truth-gap-finding.md:38-40`). So field attention gets
its own lane, `interoception`, and any link to a workspace or process arc is by time only and is
labelled co-occurrence.

**Source.** The unit is a dominance run: the same node target winning consecutive ticks.
`update_dominance_streak` (`orion/attention/field_attention/goal_provenance.py:111`) already
tracks runs in memory. Today the only durable trace is `goal_provenance_streak_ticks`, one row per
2-second tick, which its own schema says is temporary and should be retired once calibrated
(`orion/schemas/field_goal.py:75-82`). The goal record itself is bus-only and never stored.

**Seam S2 (new producer row, small).** When the streak's target changes, orion-attention-runtime
writes one row for the run that just ended to a new `field_dominance_run` table: `run_id`,
`target_id`, `target_kind`, `started_at`, `ended_at`, `tick_count`, `min_streak_at_run`, and the
first and last `source_attention_frame_id`. Rev 3 estimated about 600 rows a day (5,516 runs over
9.2 days, `orion/sentience_striving_program/README.md:723-731`). Live 10-09 the top target changed
2,868 times in 24 h, so expect about 2,900 rows a day, still far below the 41,034 streak ticks.
The same rows are the restart-proof memory for habituation (see Regulation), so S2 has two
consumers from its first day. Once S2 is live,
`goal_provenance_streak_ticks` is retired completely, as its docstring already plans: producer,
channel, table, and retention setting, not just left unread. Until then patch 1 reconstructs runs
offline from the streak ticks to check that S2's rows will match.

**Arc rule.** An interoception arc opens on a run whose length reaches the live minimum streak
(3 ticks clears 93.38% of real runs). Because the two calm winners now alternate every 20 seconds
or so, consecutive runs on the same target within `R` minutes are merged as returns, so an
afternoon of `bus_synaptic` flicker is one arc with many returns, not 1,250 arcs. Patch 1 checks
this on the live data before the rule is frozen. Suspend, resume, and the return window work as for every
other lane, so "my field attention came back to `capability:llm_inference` four times today" is
countable. The arc's body summary is especially meaningful here: field attention on GPU capacity
while GPU watts peaked is the same fact seen from inside and outside.

**Carried degeneracies.**
- `node:substrate.chat` held single runs of up to 16,036 ticks, about 8.9 hours, between
  2026-08-13 and 2026-08-19, and nobody has root-caused it (`orion/sentience_striving_program/README.md:732-744`). Any interoception
  arc longer than four hours gets a frame warning that it may be a stuck reading, and patch 1
  checks that domain's own variance during those windows, as the README suggests.
- Two scoring changes landed under this data: the novelty flicker fix (commit `f121350`,
  2026-09-25) and #2541 (merged 2026-10-07: novelty over channels measured in both ticks only,
  `selectors.py:131,218`). Deploy times are `UNVERIFIED`, so patch 1 compares run lengths before
  and after each rather than mixing them. If #2528 lands, runs before and after it are different
  instruments and are never pooled.
- The saved audit's claim that vision "dominates 40,631 of 40,627 rows" does not appear anywhere
  in the tree. It comes from PR #2255's own description on GitHub, so it is quoted from there and
  stays `UNVERIFIED` here.

**Tension winner, later (seam S3).** The field digester also records, every tick, which node won
the tension vote. It lives only inside the 2-second field-state row, and about 56% of ticks have
no winner. Its one durable consequence today is the outreach decision, which the chronology binds
already. S3 would have the digester write only runs that reach the outreach trigger's own bar of 6
ticks (about 71 a day on 2026-08-22,
`docs/superpowers/pr-reports/2026-08-22-outreach-min-run-length-recalibration-pr.md:9`) to a
`field_tension_run` table. It is listed so the gap is on the record. It is not in v1.

### Vision: company arcs, percepts, and the walkway

**1. Company arcs (seam S1, new lane).** The room cameras already run a presence state machine
that moves between `present`, `recent`, and `absent`
(`services/orion-vision-window/app/presence.py:136-146`). It writes only the latest state, as a
single overwritten row (`:239-274`), so "someone was with me from 9:02 to 11:15" cannot be
replayed today. S1 adds one append-only row per state change to a new `vision_presence_transition`
table: `transition_id`, `presence_id`, `from_state`, `to_state`, `occurred_at` (the state
machine's own `state_since`), and `identity_confirmed` as a boolean. When frames stop arriving it
writes `to_state='unknown'`, so a dark camera closes the arc instead of leaving a stale `present`
behind.

A company arc opens on a change to `present` and closes on `absent` (`recent` is the grace
period) or `unknown`. Its subject is the `presence_id`. The chronology stores whether identity was
confirmed and, when it was, the enrolled label. It never stores an unconfirmed guess (Missing
question 9). Consumers: the stance cue ("someone has been at the desk
since nine"), the daily reflection, and outreach, which already reads the presence snapshot.

Why company is its own lane rather than body context: people arriving and leaving are among the
strongest cues people use to divide experience into events (Zacks, Speer & Reynolds 2009,
"Segmentation in reading and film comprehension"). And social grounding is one of the named
prerequisites in the project's mission. While building S1, the writer should reuse one database
engine instead of creating and disposing one per write, as it does now (`presence.py:254-274`).

**2. Room percepts (live).** `vision_events` rows from the room streams (`stream_id` of a room
camera, or NULL on rows written before 2026-09-24) attach to whatever arcs are open as context.
The chronology copies `event_type` and `entities`, never the `narrative`, because the narrative is
model prose, not an observed fact, and frames carry no narrative. Each arc gets a bounded `percept_entities` list: the distinct things
the room camera saw during that arc. Three limits stay visible:
- `created_at` is the only time. The bundle item carries no observation time
  (`orion/schemas/vision.py:297-311`).
- The council writes only when the set of labels changes, or every 600 seconds on a stable scene.
  A quiet room and a dead pipeline look the same, so the frame never infers absence from silence.
- `event_type` is written by the model except for `person_presence`, so it is kept as the source's
  own word and never treated as a category.

**3. Things Orion could not name.** `vision_unresolved` rows (`council_uncertainty` or
`no_label`) are written for any stream, room included. They attach as context and feed curiosity,
which already reads them as study material. The table comes from the walkway migration, so it has
no rows until that migration is applied. For room streams the label is the reason and the stream
only; the model-written description is never copied.

**4. The walkway (merged, not live).** Expectation verdicts (`met`, `missed`, `unscorable`),
attention-worthy sightings, and asks are all bound in code now. They light up on their own when
rows appear. Rows need the deploy checklist (migration, rebuilding 13 services, setting
`WALKWAY_RTSP_URL`, tracing the patio zone;
`docs/superpowers/pr-reports/2026-09-24-walkway-camera-implementation-pr.md:14,157`). Verdicts also
need at least 5 sightings of a subject over at least 5 distinct local days. Street sightings are the
street's own episodes, so only rows that Orion's attention promoted (`event_type='attention_worthy'`)
enter the chronology. That keeps neighbours out of Orion's autobiography by default.

**5. Excluded, each with its reason.**
- `vision_scene_inventory`: a 5-second census, about 17,000 rows a day per camera. Its
  `camera_id` holds the camera's RTSP address with the password in 315,770 rows. It is never read,
  and the leak is listed as a security follow-up.
- `vision_object_inventory` and object-permanence transitions: times are quantized to the 30-minute
  sweep, and transitions are only logged. Its module refused to publish them because nothing would
  consume them (`services/orion-sql-writer/app/vision_object_permanence.py:54-64`). Temporal Self
  could be that consumer later ("the cat came back"). It is bind_later, not v1.
- Crop embeddings and raw frames: not experience records.

`VISION_LOCAL_TZ` is a fourth day-boundary setting, alongside the three named in Missing
question 3. The shared day helper must read the same zone.

### Metric gate for the new numeric outputs

| Output | Provenance | Independence | Theory anchor | Live sanity |
|---|---|---|---|---|
| Self-prediction accuracy per arc (`predictions_correct / predictions_scored`) | the calibration script's own rule, run in the reducer (`scripts/analysis/measure_self_model_calibration.py:121-162`) | no online equivalent exists; confidence is excluded, so it is not a transform of an existing field | prospective prediction scoring, the same anchor the agency plan uses; higher-order self-model calibration | `UNVERIFIED`. Patch 1 must reproduce the script's 66.0% test accuracy from the reducer's code on the same rows before this is bound. Rest state is `predictions_scored = 0`, not 0% |
| Company arc duration and count | the presence state machine's `state_since` (`presence.py:144-146`) via S1 | the snapshot gives only the current `since_sec`; no history exists | event boundaries at character entrances and exits (Zacks et al. 2009) | `UNVERIFIED` until S1 ships. Known risk: `identity_uncertain` was never true in any presence row (`docs/superpowers/pr-reports/2026-08-29-identity-ask-no-visual-confirmation-pr.md:63-75`) |
| Interoception run length, returns, dwell | `update_dominance_streak` via S2 | different targets from the workspace's `dwell_ticks`; zero id overlap | Event Segmentation Theory, as for the other lanes | about 2,870 top-target runs per 24 h (live 10-09; 5,516 in 9.2 days in August); mega-streaks up to about 8.9 hours not root-caused |
| Attention rows per lane, reason words per lane | row counts over `substrate_attention_schema` | a count, not a new sensor | none needed: it is a count of the source's own records | substrate lane ~2,000 rows a day (live 10-09); reverie's reason is the same word on almost every row |
| Concern age, times raised | `attention_salience_trace.created_at`, `attention_loop_outcome.created_at` | no existing per-loop history across days | current concerns (Klinger 1975) | 76 of 90 verdicts decayed unattended in the 14 days to 10-09; closure time is the digest's detection time |

## Regulation: habituation, drives, and arousal (new in Rev 4)

Temporal Self lets Orion remember its day. This section lets that memory change what Orion does
next, using the same accumulated time. The plan has four parts. Each part names its producer, its
consumer, and the test or trace that proves it.

1. **Habituation (conditional on a post-#2528 measurement):** attention stops crowning what it
   has been staring at for hours.
2. **One drive, rest:** sleep pressure that can actually say "not yet". The dream service still
   makes the decision, and the drive reading is its published trace.
3. **Arousal:** one slow, shared reading of engaged, idle, or strained, which existing dials
   read to set their own gain.
4. **Immune sweep:** idle time spent checking for signals that are stuck, replayed, or constant.

Everything runs in one deterministic reducer step. It sits in the same `orion-durable-runs`
thread as Temporal Self (see Missing question 1), and it has no LLM.

### Regulators as they stand (live 10-09)

Orion has about 30 "when do I act" dials. Each reads its own clock, counter, or sensor, and none
reads a shared state. There is no live global mode to reuse:

- Spark's `arousal` field defaults to 0.5 (`orion/schemas/telemetry/spark.py:42`). Its producer,
  the spark introspector, was retired on 2026-07-28, as the same file notes for its sibling
  fields at `:44-50`.
- Biometrics `strain` is the plain mean of seven hardware pressures, per host
  (`orion/telemetry/biometrics_pipeline.py:502-506`).
- The GPU pool broadcasts queue and backlog depth every 5 s (`GpuPoolStateV1`,
  `orion/schemas/gpu_pool.py:240,252-253`).

The three dials that claim to respond to need behave like clocks:

| Dial | What the code says | What live data shows |
|---|---|---|
| Dream cycle (`services/orion-dream`) | Sleep when pressure ≥ 3.0 (`DREAM_SLEEP_PRESSURE_THRESHOLD`, `services/orion-dream/app/settings.py:61`; checked at `orion/schemas/dream_cycle.py:69-70`), no chat for 45 min, and at least 6 h since the last sleep (`app/cycle.py:77-79`). Pressure is the sum of replay-candidate weights since the last cycle began (`app/replay.py:147-153`, window `app/cycle.py:53-57`) | 52 cycles from 09-26 to 10-09. Pressure at firing was 12.9 to 129, which is 4 to 43 times the threshold, on every cycle. Median gap between sleeps was 6.00 h; only 4 of 51 gaps ran past 6 h 15 min. The six-hour minimum decides; pressure never does |
| Dream idle gate | `max(created_at)` over **all** of `chat_history_log` (`app/cycle_store.py:57-60`) | Of the 52 `chat_history_log` rows in the 7 days to 10-09, 24 had no prompt and a NULL `source`. Those are Orion's own outreach messages (latest 10-07 16:33). So Orion speaking resets Orion's own idle clock |
| Curiosity | Daily cap and paced cooldown (`services/orion-hub/scripts/curiosity_investigation.py:526-540`, `paced_cooldown_sec` `:448`). Production `.env`: cap 7, cooldown 1800 s. The settings defaults (3 and 14400) disagree with `.env_example` | Exactly 7 completed runs a day on every day from 10-02 to 10-08 (`curiosity_run_outcomes`). The cap decides |
| Outreach | Cap 4/day, 2700 s cooldown, quiet hours 23:00-08:00, and a trigger that needs the same field-tension winner for 6 ticks (`services/orion-hub/scripts/endogenous_outreach.py:470-482`, `tension_outreach_trigger.py:125`) | 10-02 to 10-07: exactly 4 sent a day. Each day it also refused itself about 3,900-4,500 times for `daily_cap`, about 3,200 for `quiet_hours`, and 807 for `cooldown` (`endogenous_outreach_decisions`). On 10-08 it sent **0**: cap refusals stop at 05:00 UTC and are replaced by `content_already_used` (1,297) and `empty_generation` (49). That is a separate bug, listed under follow-ups |
| GPU self-shed | `orion/gpu_pool/orion_shed.py:161-184` | `gpu_pool_orion_shed` has 0 rows, ever |
| Cabinet-heat reflex | Sheds background GPU work at 34 °C and re-arms at 33 °C (`orion/autonomy/thermal_gate.py:60-61`, `orion/autonomy/cabinet_heat.py:151-156`) | athena cabinet over 24 h: median 32.8 °C, max 34.6 °C. This is a safety reflex, owned by thermal v2 (#2498) |

So Orion has one drive-shaped mechanism, the dream cycle, and live data shows it has been
working as a timer. Generalizing it as it stands would mean generalizing a timer. Rev 4 repairs
it first, then turns it into the template that any later drive must use.

### R1. Habituation, inside #2528's ranking seam (conditional)

**What it would do.** The longer a target stays eligible and keeps winning, the less salience
it gets. A real change in that target restores the salience at once. With no exposure, the
discount fades over time.

**Why the obvious motivation is not enough.** On 10-08, from 13:00 to 18:00 UTC,
`bus_synaptic` and `biometrics` together won 81-100% of field-attention frames in every hour.
Their mean prediction errors were 0.04 and 0.12, and every winner was crowned at salience 1.000.

That is a normalization problem, and #2528 fixes it more directly. #2528 makes an internal
candidate eligible only when its unusualness band is `high` or `unusual` (#2528 spec :124). When
nothing is eligible, the frame has no winner (:128). Once #2528 lands, those calm winners are not
habituated. They are simply ineligible.

Before #2528, habituation is pointless. Min-max normalization crowns some winner at 1.0 on every
frame, so discounting today's winner would just crown the next calm target.

**Where habituation is still needed after #2528.** Two cases remain, and #2528 handles neither:

- **A stuck high reading.** An internal target held in band `high` for hours. Rev 3's mega-runs
  are an example: `node:substrate.chat` held single runs of up to 8.9 hours.
- **An always-eligible external source.** #2528 makes an external candidate eligible whenever it
  is fresh. A busy but monotonous one, such as a feed that is always active, would win forever.

Whether either case actually happens is unknown until #2528 has run. So R1 is **conditional**.
Patch R1a is read-only. It runs after #2528 has been live for 7 days and measures, per eligible
target, the share of frames it won over rolling 30-minute windows. If no target exceeds 50% of
frames for more than 2 hours, R1 is not built, and this section records that outcome.

**If it is built.**

- **Exposure, not run length.** The measure is a leaky integral of "won this frame", with time
  constant `τ` (`FIELD_ATTENTION_HABITUATION_TAU_SEC`, starting at 1800). Run length does not
  work, because live winners alternate with a median run of 10 ticks (about 21 s).
- **Discounted ranking.** An eligible candidate ranks by `percentile × (1 − h × exposure)`, with
  `h ≤ 0.5` (`FIELD_ATTENTION_HABITUATION_MAX_DISCOUNT`).
- **Dishabituation.** A move in #2528's band (`quiet` / `usual` / `high` / `unusual`) resets that
  target's exposure to 0.
- **This design's own guard, not #2528's.** A target in the `unusual` band is never discounted,
  so a persistent alarm keeps winning. (#2528's guard at :268-269 is a different one: it stops
  the baseline from learning on flagged readings.) In practice, then, only `high`-band internal
  targets and external targets can be habituated.
- **Safety paths are untouched.** The cabinet-heat reflex and the other safety paths are outside
  attention.
- **Producer:** `orion-attention-runtime`. It keeps the integral in memory and rebuilds it at
  boot from the last `3τ` of S2 `field_dominance_run` rows. The workspace `dwell_ticks` counter
  does not survive a restart (`orion/substrate/attention_broadcast.py:44-49`).
- **Consumer:** the #2528 ranking in `orion/attention/field_attention/selectors.py`.
- **Trace:** a `habituated exposure=… discount=…` reason on each frame target, plus a numeric
  `habituation_discount` on `FieldAttentionTargetV1`. Because that model forbids extra fields,
  the consumers are updated first.

**Metric gate.**

1. **Provenance:** the winner sequence in `substrate_attention_frames`, plus S2 rows.
2. **Independence:** a deliberate feedback loop from attention's output back to its input, not
   a new sensor. It is not redundant with `dwell_ticks`, which resets on restart and counts
   workspace coalitions.
3. **Theory:** habituation with dishabituation and spontaneous recovery (Thompson & Spencer
   1966; Rankin et al. 2009).
4. **Live rest:** `UNVERIFIED`. Today's data comes from a different instrument (pre-#2528), so
   patch R1a is the check. Exposure must fall back near 0 for targets that stop winning.
5. **Existing mechanism:** habituation was removed from the stakes-design attention path (G5),
   and nothing replaced it.
6. **Reversibility:** set `h = 0`, and drop one field.

### R2. Drives: one template, one drive that passes

**The template is the dream pattern, generalized.** A drive is admitted only if it has all of
the following:

- An accumulating level, built from rows Orion already writes.
- A declared rest level that the instrument can actually reach.
- A threshold.
- One discharge action whose completion row resets the level.
- A refractory period.
- An explicit `no_reading` state for when a source is stale or absent. A missing source is never
  read as 0, and never as calm.

Each reading is a `DriveReadingV1` (schema below), built by **the service that owns the
discharge action** from the same values its own gate uses. That leaves one copy of "due" and
"refractory", never a mirror that could disagree. `state` is one of `resting`, `building`, `due`,
`refractory`, or `no_reading`.

For rest, the producer is `orion-dream`, publishing on every 600 s check. In v1 the readings
have **trace consumers only**, and the dream service still decides with its own `should_sleep`.
The consumers are:

- the regulation state, which embeds the latest reading verbatim;
- Temporal Self's sleep arc;
- the closed day that the daily metacog reads.

That is deliberate. A drive's job here is to make Orion's need visible and testable, not to
move the decision out of the service that owns the action. A drive
that does not pass all six steps of the CLAUDE.md metric gate is not built. Its gate record stays
in this document so the question is not re-asked from scratch.

**Gate results, per candidate.**

| Drive | 1 Provenance | 2 Independence | 3 Theory | 4 Live rest | 5 Existing | 6 Reversible | Verdict |
|---|---|---|---|---|---|---|---|
| **Rest / consolidate** (dream) | `compute_pressure`, `services/orion-dream/app/replay.py:147-153`; window `app/cycle.py:53-57` | Counts new replay material. That is distinct from chat recency, which is the separate idle gate | Two-process model of sleep regulation, Process S (Borbély 1982) | **Fails as scaled.** The level resets at each cycle start, so the instrument *can* rest. But on all 52 cycles it was 4-43 times over threshold when the clock released it, so "due" is the only state ever observed at decision time. The idle half also counts Orion's own outreach | This is the existing mechanism | One env value and one query | **PASS after two repairs** (below). It is the only drive in v1 |
| Explore (curiosity) | Candidate inputs exist: unoffered dream hypotheses, `vision_unresolved` rows, time since last run | Hypotheses are the rest drive's *output*. Time since last run is the cooldown clock itself | An information gap (Loewenstein 1994) needs a per-subject gap measurement, and nothing produces one | Cap-bound every day (7/7). There is no evidence the need ever falls below supply | Curiosity cap and cooldown | n/a | **FAIL (3, 4).** Not built. Curiosity reads arousal instead |
| Social (outreach) | User turns in `chat_history_log` | **Not independent.** "Minutes since Juniper spoke" is the same number arousal reads and the dream idle gate reads. A drive on top of it would count one fact twice | Social homeostasis (Matthews & Tye 2019) would need company history, and S1 is not built | 3-16 user turns a day, so the level would be high nearly all day. Outreach is already cap-bound | Outreach cap and cooldown | n/a | **FAIL (2, 4).** Not built. Outreach cooldown reads arousal |
| Cooling / GPU shed | Cabinet temperature, `cabinet_heat.py` | Already the reflex's own input | Thermoregulation is set-point control of a *level*, so an accumulator is the wrong shape | Self-shed has 0 rows. The reflex is level-triggered at 34 °C | Heat reflex, thermal v2 (#2498) | n/a | **FAIL (3).** Not a drive. Heat feeds arousal `strained`, one way only |
| Effort / fatigue (considered) | Orion-held GPU time in `gpu_pool_events` (24 h to 10-09: agent 34.7 h, chat 9.7 h, fast 2.2 h, metacog 2.1 h; 2-331 GPU-minutes per hour) | Partly overlaps the GPU backlog | Biological fatigue tracks a depleted resource that rest restores. GPU time depletes nothing that idling restores; its heat is already measured | Varies, and can rest | none | n/a | **FAIL (3).** Current GPU load feeds arousal `strained` instead |

**The two repairs that make rest a real drive** (both in `services/orion-dream`, each with a
test):

1. **Idle means Juniper is quiet, not Orion.** Two changes in the dream service:
   - `IDLE_MINUTES_SQL` (`app/cycle_store.py:57-60`) is filtered to Juniper's turns, which
     fixes the live bug on its own.
   - The dream then reads arousal first, and uses that query only as its fallback (see the
     `unknown` rule in R3).
   - Both count only Juniper's turns: `source='hub_orion'` with a non-empty prompt (28 rows in
     the 7 days to 10-09).
   - If the fallback query itself fails, that is still not idle, which is what `is_idle`
     already does (`orion/schemas/dream_cycle.py:63-66`).
2. **A threshold that can say "not yet".** Patch R2a is read-only. It replays `compute_pressure`
   at every 600 s check over 14 days, using the dream service's own loader, and records how long
   after each sleep pressure first crosses each candidate threshold. The threshold is then set
   from that distribution. A threshold calibrated this way would have kept some
   refractory-cleared checks below it, so at least one sleep in four fires later than the clock
   would allow.
   - If no threshold can do that without starving sleep, the honest verdict is that dream
     material accrues faster than Orion can sleep. The drive is then recorded as permanently
     `due`, and the clock stays the regulator.
   - Pressure has a ceiling. Candidates are capped at `DREAM_CANDIDATES_PER_SOURCE` = 50 per
     source (`services/orion-dream/app/settings.py:65`, used at `app/cycle.py:63`), and each
     weight is at most 1. R2a reports how close observed readings come to that ceiling. A
     reading pinned at the ceiling is saturated, not "very sleepy".
   - The dream service publishes a `DriveReadingV1` at every check, not only when it fires, as
     kind `drive.reading.v1` on `orion:drive:reading`. That makes the `building` curve visible.

### R3. Arousal: one slow reading that existing dials read

**What it does.** Every regulation step classifies Orion's situation into one of three levels:

- **Engaged:** Juniper is talking with Orion.
- **Idle:** nobody and nothing needs Orion. This is the time for sleep, self-inquiry, and the
  immune sweep.
- **Strained:** the body or compute is under load.

A fourth state, **unknown**, covers stale inputs, and unknown is never treated as idle.

Every level records `since`. Existing dials read it and adjust *their own* gain. Arousal never
acts on its own.

It is slow in specific, stated places:

- GPU strain needs 5 minutes of sustained backlog before it counts.
- Leaving strained needs 10 clear minutes.
- Engaged becomes idle only after 45 minutes with no turn.

Two moves are immediate by design, because the evidence for them is unambiguous: a Juniper
turn, and the heat reflex.

**Inputs.** Each input has a provenance, and each exclusion has a reason.

- **E1, minutes since Juniper's last turn.** From `chat_history_log` rows with
  `source='hub_orion'` and a non-empty prompt. It also wakes immediately on the
  `orion:chat:history:turn` event, which the Situation driver already subscribes to.
- **S1, the cabinet is judged hot.** True only when `CabinetHeatVerdict.reflex ==
  REFLEX_CABINET_HOT` (`orion/autonomy/cabinet_heat.py:87,151-156`).
  - It is computed by running that pure verdict over the latest athena cabinet reading, using
    the same query the module uses (`:275`).
  - `REFLEX_CABINET_UNKNOWN` means the sensor is missing. It makes S1 *stale*, not hot.
- **S2, GPU backlog.** `GpuPoolStateV1.backlog_depth` is non-zero for at least 5 minutes
  (`orion/schemas/gpu_pool.py:253`; 9 `backlogged` events in the 7 days to 10-09).
  - The regulate node subscribes to `orion:gpu_pool:state`, which is published every 5 s
    (`services/orion-gpu-pool/app/runtime.py:1463-1468`).
  - `orion/bus/channels.yaml` gains durable-runs as a consumer of that channel.
- **Excluded, biometrics `strain`.** It is the mean of GPU utilization, thermal, and five other
  pressures, so it double-counts S1 and S2.
- **Excluded, Spark `arousal`.** It is dead. The word "arousal" is already used in six other
  places, for unrelated scores: collapse mirror, topic-foundry enrichment, signal-gateway
  dimensions, and the spark, substrate, and reasoning adapters. So the new field is named
  `arousal_level` and lives only on `RegulationStateV1`, and no existing field is reused.
  Retiring spark's dead field is a separate follow-up, with its own blast-radius check.
- **Excluded, the rest drive.** It *reads* arousal, so deriving arousal from it would create a
  loop. Juniper's brief suggested deriving arousal from drives. With one drive that reads
  arousal, that is circular in v1.
- **Excluded, Orion's own speech and thought.** Outreach, reverie, and curiosity are all out.
  Otherwise Orion would arouse itself.
- **A named loop that is kept.** S1 and S2 are partly caused by Orion's own GPU work: agent
  work alone held 34.7 GPU-hours in the 24 h to 10-09. That is a *negative* feedback loop:
  1. Orion's work raises strain.
  2. The readers slow optional work.
  3. The load falls.
  4. Strain clears.

  Negative feedback is what a regulator is for. The anti-oscillation guard is the 10-minute
  clear time. Patch R3a counts strained to engaged/idle flips per day, and more than 12 a day
  means that guard is too short.
- **Undecided, room presence.** See Missing question 11.

**Rule.** The rule is deterministic, with hysteresis.

Evaluate in this order:

1. If any *fresh* strain input holds (S1 hot, or S2 sustained for at least 5 minutes),
   arousal is `strained`. One fresh, true strain input is enough, even if other inputs are stale.
   Arousal leaves `strained` only after 10 clear minutes.
2. Otherwise, if any input is older than 3 times its cadence, arousal is `unknown`.
3. Otherwise, it is `engaged` if E1 is under `DREAM_IDLE_MINUTES` (45). That reuses the dream
   setting's existing meaning, so the threshold is not copied.
4. Otherwise, it is `idle`.

**When arousal is `unknown`, or the regulation state itself is missing or stale in Redis, every
reader falls back to exactly its pre-arousal behaviour.** That is the same path as turning the
flag off. For the dream, the fallback is its own idle query, filtered to Juniper's turns (repair
1 below applies to that query as well). So an outage in durable-runs or Redis cannot stop Orion
from sleeping.

**History.** Every level change is written as an `arousal_transition` event in
`temporal_self_event`. That gives the day frame "strained 14:02-14:40" with evidence, and gives
R3a's flip count a live source. The regulation state itself is read from Redis
`orion:regulation:latest` and from `GET /regulation/state`. It is **not** published on a bus
channel in v1, because no bus consumer exists for it.

**Live sanity.** On the 10-09 numbers, all three levels are reachable:

- 3-16 Juniper turns a day, so mostly idle with engaged bursts.
- A cabinet peak of 34.6 °C, above the 34 °C reflex, so strained is reachable.
- Patch R3a replays 14 days and records the time share per level. The acceptance bar is that no
  level sits at 0% or 100%.

**Which dials read it, and which must not** (file:line verified 10-09):

| Reads arousal | Where | Gain |
|---|---|---|
| Dream idle gate | `services/orion-dream/app/cycle_store.py:57-60` → replaced | sleeps only at `idle` |
| Curiosity pacing | `services/orion-hub/scripts/curiosity_investigation.py:536-540` (the cooldown branch of `scheduling_block_reason`, `:518`) | `strained`: returns a new reason, `arousal_strained`. `engaged`: cooldown doubled. `idle`: unchanged. The daily cap is unchanged |
| Curiosity self-inquiry | its `scheduling_block_reason` call, `curiosity_investigation.py:2336` | runs only at `idle` |
| Visual reverie baseline | `orion/reverie/baseline.py:53` (`schedule`) | `strained`: defer by `retry_sec` |
| Reverie chain cadence | `services/orion-thought/app/chain.py:421-442` | `strained`: interval doubled |
| Self-modification pressure | `orion/substrate/mutation_pressure.py:92-98` (`ready_for_proposal`) | no proposal while `strained` |
| Outreach cooldown | `services/orion-hub/scripts/endogenous_outreach.py:478-482` | `strained`: refuse with `arousal_strained`. The cap is unchanged |

**Must not read arousal.** Each of these protects Juniper, the hardware, spend, or content, and
a mood must not loosen any of them:

- outreach quiet hours and turn-in-flight (`endogenous_outreach.py:470-473`)
- identity-ask cooldowns (`orion/situational/identity_ask_cooldown.py:67`; cortex-exec settings
  `:213`, `:236`)
- the vision ask cap
- the curiosity hour window
- the dream six-hour minimum (`app/cycle.py:77-79`)
- the GPU self-shed caps and master switch (`orion/gpu_pool/orion_shed.py:161-184`)
- the cabinet-heat reflex (`cabinet_heat.py:151-156`)
- the visual thermal gate (`orion/autonomy/thermal_gate.py:47-50`)
- the reverie theme refractory
- the ask-Claude budget (`orion/autonomy/ask_claude_trigger.py:169-194`)
- infrastructure rate limits
- notify quiet hours (`services/orion-notify/app/policy/rules.yaml:1-4`)

The memory-window idle close (the parked episode idle-close item) is the natural *later* reader
of an `engaged → idle` transition. It is not in v1.

**Metric gate.**

1. **Provenance:** as listed under Inputs.
2. **Independence:** E1, S1, and S2 come from three different sensors. `strain` is excluded as a
   blend of S1 and S2.
3. **Theory:** a global arousal or neuromodulatory gain state that sets the gain of many local
   circuits at once (Aston-Jones & Cohen 2005, adaptive gain). Yerkes-Dodson is deliberately
   *not* cited: it describes performance, not control.
4. **Live rest:** idle is the common state; see the live sanity notes above.
5. **Existing mechanism:** none exists. Spark arousal is dead. The other "arousal" fields are
   per-item enrichment scores, not a global state.
6. **Reversibility:** `ORION_REGULATION_AROUSAL_ENABLED=false` makes every reader behave as
   though arousal were unavailable. Each reader then falls back to exactly its current
   behaviour. This is tested per reader.

### R4. Runtime immune sweep, using the semantic layer

**What it does.** When arousal is `idle`, at most once every 6 hours, a deterministic sweep
looks for instruments that have stopped telling the truth. It checks for four failures:

- **Constant:** one distinct value over 24 h, where the metric declares itself a level.
- **Floor-locked:** it never reaches its declared rest value.
- **Replayed:** one value repeated N times while its source rows did not change.
- **Decayed-to-zero:** successive values in an exact geometric ratio, the failure CLAUDE.md
  records for `node:substrate.route`.

It writes a finding. It never "fixes" anything.

**It extends the semantic layer, not a new registry.** #2528 rev 2 (its spec :93-101, :179-189)
found two gaps:

- The field-channel glossary (`config/field/field_channel_glossary.v1.yaml`, 50 entries) has no
  `rest`, `sparsity`, or `value_kind`.
- The metric lock (`config/metrics/metric_definitions.lock.json`, 694 entries, gated by
  `scripts/check_definition_drift.py --gate`) records lineage only.

#2528 proposes adding `value_kind`, `rest`, `sparsity`, `absent_means`, and `polarity` there.
R4 is the runtime consumer of those fields. In v1 it covers only glossary channels that carry
them, read from `substrate_field_state`, plus the three known zombies, which serve as test
fixtures:

- the repair-pressure maximum replayed into curiosity (`_repair_appraisal_from_chat`,
  `services/orion-substrate-runtime/app/worker.py:3629-3656`; 5,284 replays per #2528 :81)
- the constant 1.0 `ontology_sparse_region` (`orion/substrate/frontier_curiosity.py:104-120`)
- the spark rollup zombie writer (`services/orion-state-journaler/app/service.py:146-188`)

**Producer, consumer, trace.**

- **Producer:** the regulation step, gated on `idle`.
- **Output:** one `immune_finding` event in `temporal_self_event`, with the metric id, failure
  kind, and evidence window. It is also counted in `RegulationStateV1.immune_findings_open`.
- **Consumer in v1:** the closed-day frame, which the daily metacog reads.
- **Later consumer:** forcing a drive or arousal input to `no_reading`. That needs those inputs
  (E1, S1, S2, rest pressure) to have metric-lock entries with `rest` and `value_kind` first. It
  is not promised in v1, because R4 v1 only covers glossary channels.
- **Test:** each of the three zombies, reconstructed as a fixture, is caught. A healthy
  fixture, and a genuinely calm fixture at its declared rest, are not flagged.

**What blocks it.** It is blocked on #2528's lock fields. Without declared `rest` and
`value_kind`, the sweep cannot tell calm from dead. That is exactly #2528's point, so R4 is last
in priority.

## Missing questions

These are Juniper's calls. Each has a recommended default so patch 1 can start without waiting.

1. **Host.** Recommended (changed in Rev 4): a second self-driven thread in
   `orion-durable-runs`, beside the Situation Graph. It would reuse that service's shared Postgres
   checkpointer (`services/orion-durable-runs/app/main.py:157-184`), its driver pattern (event
   wake plus clock tick, `app/situation_driver.py:80`), its seeding of each day bucket from the
   previous day (`_seed`, `:152`), and its `DurableRunStateV1` publishing. To add it: one
   graph-and-driver module pair, one start block in `main.py`, and the workflow name added to
   `SELF_DRIVEN_WORKFLOWS` (`app/runner.py:131`).
   - Rev 3's choice, `orion-consolidation-runtime`, is dropped. It runs a stale 09-25 image that
     logs about 737k schema warnings a day, and putting the reducer there would give Orion two
     chronology runtimes.
   - Tick: every 120 s, plus an immediate step on any chat turn. Each step writes one checkpoint,
     as the Situation Graph does.
2. **Day boundary.** Local midnight, or sleep-to-sleep? Recommended: `day_id` is the local
   calendar date; sleeps are arcs inside days; an arc open at midnight closes with `day_boundary`
   and its continuation carries `carried_from_arc_id`.
3. **Timezone key.** (Rev 4 note: the Situation Graph buckets its threads by **UTC** date,
   `situation_driver.py:45-46`. Temporal Self's `day_id` is the local date. Recommended: keep the
   local date for Orion's day. Name the thread `temporal_self:orion:<local date>` so the two
   kinds of bucket never get compared.) The repo has `ORION_SITUATION_TIMEZONE` (cortex-exec, Hub) and
   `ACTIONS_DAILY_TIMEZONE` (orion-actions). Adding a third copy for the reducer's host is drift.
   Recommended: the shared `day_id` helper reads one key, `ORION_SITUATION_TIMEZONE`, added to the
   host's settings with the same default; unifying orion-actions onto it is a follow-up.
4. **Juniper in town.** Orion's exchanges with Juniper inside AI Town: Juniper conversation or
   town? Recommended: Juniper conversation, the same as `chat_history_log`.
5. **Retention.** Recommended: events 30 days, arcs 90 days, closed days 365 days, singleton
   frame forever.
6. **Atlas publication.** Publish arc transitions as `TemporalHopV1`? Recommended: not in v1.
7. **Subject label source.** Recommended: the source row's own label, clipped with
   `orion.schemas.attention_schema.clip`, never generated; and for cortex-turn attention rows the
   label is **not** copied, because `attended_label` there is an open-loop description built
   from chat text (`orion/substrate/attention_frame.py:196`) and could carry Juniper's words.
8. **Field-lane runs.** Persist dominance runs (seam S2) rather than every `FieldGoalProvenanceV1`
   record? Recommended: runs. A goal record is emitted on every qualifying 2-second tick, so
   persisting it would recreate the per-tick volume. One row per completed run is what the
   interoception lane needs, and it lets the streak-tick table retire as planned.
9. **Presence identity.** Recommended (changed in Rev 4, after Juniper's no-privacy-boundary
   rule): store `identity_confirmed` and, when confirmed, the enrolled label (#2545 now
   recognizes Juniper). Never store an unconfirmed guess.
10. **Who owns "what Orion is working on".** The Situation spec plans an `orion.working_on` slot
    (#2526 spec :153). Recommended: that slot reads Temporal Self's active arc, with its arc id
    as evidence, and does not derive its own. One chronology per subject.
11. **Does Juniper's silent presence count as "engaged"?** Today presence is only a snapshot,
    and identity in it has been unreliable. Recommended: not in v1, where only her turns count.
    Revisit once S1 gives presence a history.
12. **Arousal gain values.** These are the doubled cooldowns and the "only at idle" rules in the
    reader table. Recommended: ship them as listed, each reader with its own env multiplier, and
    re-fit after two weeks of R3a data.
13. **The rest drive cannot be calibrated.** Suppose R2a shows no threshold that leaves pressure
    below the bar at some check after each sleep. Recommended: record the drive as permanently
    `due`, keep the six-hour clock as the true regulator, and say so in the regulation state.
    Do not pick a threshold that only looks calibrated.

## Proposed schema / API changes

### Capability, data, privacy, proof, danger, rollback (proposal-mode fields)

- **Capability that changes.** Orion gains a queryable, inspectable account of its own day:
  which subjects it returned to, for how long, in what order, what it slept on, what it expected
  and what came of it in each source's own words, what it was refused, and what its body was
  doing meanwhile. Consumers read a bounded slice of that instead of a rolling window of chat.
  Rev 4 adds regulation:
  - attention habituates to what it has been staring at;
  - sleep is decided by a pressure that can say "not yet";
  - seven existing dials slow down or wait when Orion is strained, or is engaged with Juniper;
  - idle time checks Orion's own instruments for stuck readings.
- **Data touched.** Read-only over the tables in the binding appendix. Temporal Self writes only
  to the new `temporal_self_*` tables, and no source table is modified. Rev 4 regulation writes:
  - `field_dominance_run` (S2);
  - the regulation state, to Redis `orion:regulation:latest`;
  - `immune_finding`, `arousal_transition`, and `drive_reading` events in
    `temporal_self_event`;
  - one new dream bus kind, `drive.reading.v1`.

  It changes the gating behaviour of the seven readers in R3. No FalkorDB writes.
- **Privacy boundary.** Juniper's standing rule is that there is no privacy boundary between
  Juniper and Orion, so every Orion-side consumer may read every label, Juniper's included. The
  remaining boundary faces *outward*: nothing goes to contractor peers or published artifacts,
  and other people's speech (NPC town turns, street sightings) is not crystallized. For size and
  for honesty, the event table still stores references and bounded labels, never raw
  utterances: the chat grammar lane's own `payload_ref` discipline
  (`services/orion-hub/scripts/grammar_emit.py:68-216`) is the model. Chat turns contribute only
  `session_id`, `correlation_id`, and `created_at`. Cortex-turn attention labels are not copied
  (Missing question 7). Town rows with a human partner are treated as Juniper's
  conversation. Nothing here reaches contractor peers; the frame is not
  published on any bus channel in v1.
- **Trace that proves it worked.** One `temporal_self_arc` row whose `evidence_event_ids` resolve
  to real rows in at least two source tables, with `attention_returns ≥ 2`, and one consumer read
  of the frame that cites the arc id. A frame that reads "no active arc" on a quiet stretch is
  also required evidence: the instrument must be able to rest. For regulation:
  - if R1 is built, one live frame where a high-band or external target lost first place to a
    lower-percentile candidate, with the `habituated` reason on it;
  - one dream cycle that waited past the six-hour clock because pressure was below threshold;
  - one curiosity or outreach refusal with reason `arousal_strained` while the cabinet reflex was
    asserting;
  - the regulation state reading `idle` overnight and `engaged` within one step of a Juniper
    turn.
- **Dangerous failure modes.**
  1. *Fabricated continuity:* loose subject matching stitches unrelated events into one arc, and
     Orion then tells Juniper "I have been on this all day" falsely. Mitigation: identity by exact
     refs only; the arc-precision eval; every arc carries its evidence ids.
  2. *One arc all day:* if the broadcast winner never changes (G5, habituation removed), the day
     collapses into one arc and "returns" is always zero. That is a true report of a degenerate
     attention lane, not a reducer bug, but it must be visible: the frame's `warnings` names it,
     and patch 1 measures it before any consumer is wired.
  3. *Town chatter as lived day:* the 2026-08-14 crystallization incident (610 of 621 proposals
     were NPC dialogue, `orion/memory/crystallization/formation_policy.py:17-22`) repeated here.
     Mitigation: source filter as a schema invariant plus a mixed-producer test fixture.
  4. *Restart duplication:* a restart re-folds the same window into new arc ids. Mitigation:
     deterministic ids from `(day_id, subject_ref, first_event_id)`; `ON CONFLICT DO NOTHING`;
     the replay-identity acceptance check.
  5. *Consumer over-steering:* a stance cue that reads like an instruction. Mitigation: the cue
     replaces the `recent_attention` block's content in the same slot and budget, stays inside the
     SOURCES boundary from `fix/stance-reading-boundary`
     (`docs/superpowers/pr-reports/2026-09-26-stance-reading-boundary-pr.md`), and must pass
     `orion/thought/evals/stance_task_boundary.py` unchanged.
  6. *Wrong day:* the `daily_metacog` mislabel shows this is a live failure class. Mitigation: one
     shared `day_id` helper with a test that pins midnight in `America/Denver`; each naive or TEXT
     timestamp column (`chat_history_log.created_at`, `aitown_chat_history_log.created_at`,
     `metacog_trigger.timestamp`, `orion_metacog.timestamp`, `orion_biometrics_summary.timestamp`)
     gets an explicit, documented cast in `sources.py`.
  7. *Load:* the reducer reads no `substrate_field_state`, `grammar_events`, or per-tick field
     telemetry. Its heaviest read is the broadcast log at about 2,000 rows a day by keyset cursor.
     The exception is the immune sweep (R4). It reads `substrate_field_state`, but only when
     arousal is `idle`, at most once every 6 hours.
  8. *Habituating away a real alarm:* a persistent, genuinely bad body reading loses salience.
     Mitigations:
     - the `unusual` band is never discounted;
     - a band change resets exposure to zero;
     - the heat reflex is outside attention;
     - a fixture with a sustained, unusual reading must keep winning.
  9. *Arousal stuck in one level:* for example, Orion's own outreach counting as "Juniper spoke",
     which is the live bug in today's dream idle gate. Orion would then never be idle and never
     sleep. Mitigations:
     - E1 is filtered to Juniper's turns, and a test fixture includes an outreach row;
     - the time share per level is a live acceptance check;
     - `unknown` never reads as idle.
  9b. *Arousal down, Orion never sleeps:* durable-runs or Redis is down, so the regulation state
      goes stale. Mitigation: a stale or `unknown` state makes every reader fall back to its
      pre-arousal path, the dream included (its own Juniper-filtered idle query). The fixture
      deletes `orion:regulation:latest` and asserts that the dream still sleeps once idle.
  10. *Arousal loosening a safety gate:* a reader wired in the wrong direction. Mitigations:
      - arousal may only make a reader *more* conservative when `strained`, or when `engaged`
        for self-occupation;
      - the must-not list is enforced at the **branch** level, not the import level. Readers and
        protected gates share functions: quiet hours sit beside the outreach cooldown
        (`endogenous_outreach.py:470-482`), and the curiosity hour window sits beside its
        cooldown in `scheduling_block_reason` (`curiosity_investigation.py:518-540`). So for
        every protected reason, the test asserts the same decision under all four arousal values
        and under no arousal at all.
  11. *A drive that only looks calibrated:* a threshold picked to produce some "not yet" checks on
      paper. Mitigation: Missing question 13. The threshold comes from the R2a replay, and its
      distribution is recorded here.
- **Disable and roll back.** `TEMPORAL_SELF_ENABLED=false` stops the tick. Each consumer has its
  own flag (`TEMPORAL_SELF_STANCE_CUE_ENABLED`, `TEMPORAL_SELF_DAILY_METACOG_GROUNDING_ENABLED`,
  `TEMPORAL_SELF_CURIOSITY_THREAD_ENABLED`). Per Juniper's standing rule, each flag **ships ON**
  in settings, in `.env_example`, and in the production `.env`. Turning one off restores that
  consumer's baseline path exactly.

  The regulation flags also ship ON:
  - `FIELD_ATTENTION_HABITUATION_ENABLED` (or set `h = 0`);
  - `ORION_REGULATION_AROUSAL_ENABLED`, read by every reader; off means each reader runs its
    current code path, tested per reader;
  - `DREAM_IDLE_FROM_AROUSAL_ENABLED`; off restores `IDLE_MINUTES_SQL`;
  - `ORION_REGULATION_IMMUNE_SWEEP_ENABLED`. The four tables have no foreign keys and can be dropped. All flags land
  in the host service's `.env_example` and settings, with the local `.env` synced by
  `python scripts/sync_local_env_from_example.py` in the same patch.

### Schemas (`orion/schemas/temporal_self.py`, new)

All models `extra="forbid"`. The stored models are registered in `_REGISTRY` in
`orion/schemas/registry.py` so that a stored `projection_json` resolves by `schema_version`, the
way `ConsolidationFrameV1` does (`registry.py:1340`). The Temporal Self models are not bus
payloads in v1. The Rev 4 regulation models are, so they need `SCHEMA_REGISTRY` and
`orion/bus/channels.yaml` entries. Because their names contain "drive", they also need an entry
in `orion/inner_state_registry.py`: `scripts/check_inner_state_registry.py` fails on any
unregistered drive-named schema. That check is how `DriveReadingV1` is kept visibly distinct from
the retired, producer-less `DriveStateV1`.

Every enum value below names its producer in the arc rules or the appendix; a value with no
producer was removed in review (`winner_changed`, `conversation` without a rule, thermal states
that no code emits).

```python
TemporalSelfEventV1                  # sparse: process boundaries, verdicts, deferrals, observations
  schema_version: "temporal_self.event.v1"
  event_id: str                      # deterministic: f"{source_kind}:{source_ref}"
  day_id: str
  occurred_at: datetime              # tz-aware UTC, per the cast documented for its source
  source_kind: Literal[
    "chat_turn", "town_exchange", "curiosity_run", "reverie_chain", "visual_run",
    "visual_deferral", "gpu_wait", "dream_cycle", "dream_hypothesis", "expectation_verdict",
    "action_outcome", "prior_revision", "peer_ask", "metacog_observation",
    "consolidation_window_close", "attention_row",
    # added in revision 3, see "Attention, field attention, and vision in full"
    "attention_loop_raised", "attention_loop_verdict", "field_dominance_run",
    "presence_transition", "vision_percept", "unresolved_percept", "attention_worthy_sighting",
    # added in revision 4
    "memory_episode",        # episode_memory: episode_id, purpose, occurred_at; never statement
    "situation_revision",    # SituationStateV1 revision from Redis/bus; context only, Juniper's now
    "immune_finding",        # R4 output
    "arousal_transition",    # R3: one row per level change, with the deciding input
    "drive_reading",         # R2: rest drive state changes only (not every check)
  ]
  source_table: str
  source_ref: str                    # the row's own primary key, verbatim
  correlation_id: str | None
  subject_ref: str | None            # one canonical ref; None for context events
  related_refs: list[str]            # e.g. offered prior ids for a curiosity run; partner slug + session for town
  label: str                         # ≤ 300 chars from the source row's own label; "" only where the label is model prose
  privacy_class: Literal["orion_internal", "juniper_chat"]   # an OUTWARD boundary only: juniper_chat is readable by every
                                     # Orion consumer, never sent to contractor peers or published artifacts
  verdict: str | None                # the source's own word (confirmed, missed, claim_upheld=false, ...); never normalised
  payload: dict[str, Any]            # bounded, documented per source_kind in sources.py

TemporalSelfArcV1
  schema_version: "temporal_self.arc.v1"
  arc_id: str                        # sha256(day_id, kind, subject_ref, first_evidence_ref)[:16]
  day_id: str
  kind: Literal["attention", "concern", "interoception", "company",
                "conversation", "town", "curiosity", "reverie", "imagery", "sleep"]
  subject_ref: str
  subject_label: str
  began_at: datetime
  ended_at: datetime | None
  status: Literal["open", "suspended", "closed"]
  closed_reason: Literal["process_ended", "verdict", "source_stale",
                         "day_boundary", "return_window_expired"] | None
  attention_returns: int             # resumes after a suspension, same day
  cumulative_dwell_sec: float
  interruptions: list[str]           # arc_ids that suspended this one
  carried_from_previous_day: bool
  carried_from_arc_id: str | None
  evidence_refs: list[str]           # broadcast log_ids or event_ids that SHARE subject_ref; capped 256
  evidence_overflow: int
  context_event_ids: list[str]       # subject-less events bound by time (metacog observations, gpu waits); capped 64
  expectation_event_ids: list[str]   # events with a verdict committed or resolved inside the arc
  constraint_event_ids: list[str]    # visual_deferral / gpu_wait events inside the arc
  body: ArcBodySummaryV1 | None
  attention: ArcAttentionSummaryV1 | None
  self_model: ArcSelfModelSummaryV1 | None
  percept_entities: list[str]        # distinct room-camera entities seen during the arc; capped 16
  reducer_version: str

ArcAttentionSummaryV1                # folded from substrate_attention_schema rows inside the arc
  rows_by_lane: dict[str, int]       # process -> row count
  reasons_by_lane: dict[str, list[str]]   # process -> distinct attention_reason words, unnormalised

ArcSelfModelSummaryV1                # folded from substrate_attention_self_model rows inside the arc
  predictions_scored: int            # rows whose predicted_shift could be checked two rows ahead
  predictions_correct: int           # same rule as scripts/analysis/measure_self_model_calibration.py:121-162
  most_predicted_domain: str | None
  override_absent_reason_mode: str | None

ArcBodySummaryV1
  cluster_sample_count: int          # orion_biometrics_cluster rows in the interval
  chassis_watts_mean: float | None
  peak_pressure_max: float | None
  peak_pressure_channel_at_max: str | None
  cabinet_temp_c_min: float | None   # athena orion_biometrics_summary.measurements.cabinet_temp_c
  cabinet_temp_c_max: float | None
  ambient_spike_count: int
  cooling_switch_changes: int
  thermal_refusals: int              # visual_deferral events with outcome deferred_thermal / terminal thermal_refused

ArcSummaryV1                         # compact form used inside the frame and the closed day
  arc_id, kind, subject_ref, subject_label, began_at, ended_at, status, attention_returns, cumulative_dwell_sec

OpenThreadV1                         # derived at build_frame time; not a table
  subject_ref: str
  subject_label: str
  first_seen_today: datetime
  last_returned: datetime
  returns_today: int
  carried_from_previous_day: bool

ExpectationRefV1
  event_id: str
  source_kind: str
  committed_at: datetime
  resolved_at: datetime | None
  verdict: str | None                # the source's own word

TemporalSelfFrameV1                  # the singleton projection, upserted each tick
  schema_version: "temporal_self.frame.v1"
  frame_id: str                      # deterministic per (day_id, tick window)
  day_id: str
  as_of: datetime
  day_phase: str                     # TimeContextV1.day_phase vocabulary, from the shared helper
  active_arc: ArcSummaryV1 | None
  previous_arc: ArcSummaryV1 | None
  arcs_today: list[ArcSummaryV1]     # bounded to 64
  open_threads: list[OpenThreadV1]
  expectations_pending: list[ExpectationRefV1]
  expectations_resolved_today: list[ExpectationRefV1]
  self_change_event_ids: list[str]   # prior_revision, action_outcome, expectation_verdict, adopted dream_hypothesis
  constraint_event_ids: list[str]
  sleep_arc_ids: list[str]
  entered_day_with: list[str]        # arc_ids carried from the previous closed day
  source_cursors: dict[str, str]     # source_kind -> last (ts, ref) folded; the inspectability surface
  warnings: list[str]                # "broadcast winner unchanged all day", "town source stale", ...

TemporalSelfDayV1                    # one per closed day
  day_id: str
  closed_at: datetime
  frame: TemporalSelfFrameV1         # the final frame of that day
  arcs: list[TemporalSelfArcV1]      # full arcs, so the day is self-contained after arc retention

TemporalSelfStateV1                  # reducer working state; persisted only as the frame + arcs
  day_id, open_arcs, suspended_arcs, cursors, pending_expectations

TemporalSelfStanceCueV1              # the bounded stance projection
  as_of, day_phase, active_subject_label, active_arc_age_sec, active_returns_today,
  came_from_label, came_from_minutes_ago, open_thread_count, oldest_open_thread_age_sec,
  last_sleep_ended_minutes_ago, rendered: str   # ≤ 400 chars, coarse phrasing as recent_attention_cue

CuriosityTemporalFactsV1
  prior_ids_seen_today: list[str]
  seconds_on_subject_today: float
  runs_on_subject_today: int
  arcs_touching_prior_ids: list[str]

# --- Rev 4: regulation (orion/schemas/regulation.py, new) -----------------------------
DriveReadingV1                       # the generalized dream-pressure pattern; one per admitted drive
  drive: Literal["rest"]             # widened only by a drive that passes the gate in this document
  level: float | None                # None only when state == "no_reading"
  threshold: float
  rest_level: float                  # declared; the instrument must be able to reach it
  state: Literal["resting", "building", "due", "refractory", "no_reading"]
  accumulating_since: datetime | None
  last_discharge_at: datetime | None # the discharge action's own completion row (dream_cycle.ended_at)
  refractory_until: datetime | None
  source_ref: str                    # e.g. "dream.sleep_pressure.v1:<computed_at>"
  no_reading_reason: str | None      # "source_stale", "immune_finding:<id>", ...

ArousalReadingV1
  arousal_level: Literal["engaged", "idle", "strained", "unknown"]
  since: datetime
  minutes_since_juniper_turn: float | None   # E1; user turns only
  cabinet_reflex: str | None                 # S1; CabinetHeatVerdict.reflex verbatim
  gpu_backlog_sustained_sec: float | None    # S2
  reasons: list[str]                         # which input decided, in words from this table

RegulationStateV1                    # one per step; Redis orion:regulation:latest + GET /regulation/state (no bus channel in v1)
  schema_version: "regulation.state.v1"
  generated_at: datetime
  arousal: ArousalReadingV1
  drives: list[DriveReadingV1]
  immune_findings_open: int
  immune_last_sweep_at: datetime | None
  warnings: list[str]

# Bus (registered in SCHEMA_REGISTRY and orion/bus/channels.yaml):
#   orion:drive:reading   kind drive.reading.v1   producer orion-dream (each 600 s check)   consumer orion-durable-runs
#   orion:gpu_pool:state  (existing)              add consumer orion-durable-runs (arousal S2)
# FieldAttentionTargetV1 gains habituation_discount: float = 0.0 (consumer-first, forbid model).
```

### The reducer contract (`orion/temporal_self/`, new, I/O-free)

```python
fold_broadcast_ticks(state, ticks: list[BroadcastTickView], now) -> state      # the attention-arc driver; ticks are NOT stored as events
fold_events(state, events: list[TemporalSelfEventV1], now) -> state
build_frame(state, now) -> TemporalSelfFrameV1
project_stance_cue(frame, now) -> TemporalSelfStanceCueV1
project_curiosity_facts(frame, arcs, prior_ids) -> CuriosityTemporalFactsV1
project_reverie_recent(frame, limit) -> list[ArcSummaryV1]
close_day(state, day_id, now) -> TemporalSelfDayV1

# Rev 4, orion/regulation/ (new, I/O-free)
classify_arousal(prev: ArousalReadingV1 | None, inputs: ArousalInputsV1, now) -> ArousalReadingV1   # hysteresis lives here
read_rest_drive(pressure: SleepPressureV1 | None, last_cycle_end, now) -> DriveReadingV1
habituation_discount(exposure: float, band_changed: bool, band: str, h_max: float) -> float        # used by attention-runtime
sweep(metric_specs, series, now) -> list[ImmuneFindingV1]                                           # R4
```

`BroadcastTickView` is `(log_id, generated_at, selected_loop_source_ref)` extracted from
`projection_json.frame.open_loops[selected].source_refs[0]`. Pure, deterministic, idempotent,
replayable. Folding the same inputs twice yields the same state. Inputs arrive ordered by
`(occurred_at, ref)`; late events (a verdict scored after the arc closed) attach by event id,
never by time, and never reopen a closed arc.

**Arc rules, stated so a test can pin them:**

1. An *attention arc* opens when the same `selected_loop_source_ref` wins `K` consecutive
   broadcast ticks. `K` = `TEMPORAL_SELF_ARC_MIN_TICKS`, default 3, which at the 30-second
   broadcast cadence is about 90 seconds. This is **not** the same unit as
   `ORION_GOAL_PROVENANCE_MIN_STREAK`, which counts 2-second field ticks and is itself called an
   unmeasured placeholder (`services/orion-attention-runtime/app/settings.py:58-60`). Patch 1
   measures the real broadcast streak distribution before `K` is frozen.
2. A *process arc* opens at its process's own start event and closes with `process_ended` at its
   end event; its subject is the process's own identity:
   - `conversation`: `chat_turn` events sharing `session_id`; opens at the first turn, suspends
     when no turn arrives for `R` minutes, closes on `consolidation_window_close` for that
     platform or on `return_window_expired`.
   - `town`: `town_exchange` events sharing `session_id`, same rule; `related_refs` carries the
     partner slug so returns to the same partner across sessions are countable.
   - `curiosity`: one `curiosity_run` event; `began_at` is `curiosity_offer_decisions.turn_started_at`
     when that table has rows, else the attention row's `generated_at` as a point arc.
   - `reverie`: one `reverie_chain` event at `created_at`, with the chain's thoughts as evidence
     by `created_at` inside the chain's window (point arc if the chain has one thought).
   - `imagery`: one `visual_run` event; duration from `attempt.started_at` when present.
   - `sleep`: one `dream_cycle` event from `started_at` to `ended_at`.
   - `concern`: opens on the first `attention_loop_raised` for a `loop_id` after its last
     verdict; closes with `verdict` on the next `attention_loop_verdict`; may span days.
   - `interoception`: opens on a `field_dominance_run` whose `tick_count` reaches
     `min_streak_at_run`; subject is the run's `target_id`; consecutive runs on the same target
     within `R` minutes are returns, not new arcs.
   - `company`: opens on a `presence_transition` to `present`; stays open through `recent`;
     closes with `process_ended` on `absent` or `source_stale` on `unknown`; subject is the
     `presence_id`.
3. An arc *suspends* when a different subject satisfies rule 1, or when another process arc opens
   on the same lane. The suspending arc is appended to `interruptions`.
4. A suspended arc *resumes* if its subject wins again within `R` minutes
   (`TEMPORAL_SELF_RETURN_WINDOW_MIN`, default 180). `attention_returns` increments. Dwell sums
   across segments.
5. A suspended arc *closes* with `return_window_expired` after `R` minutes without a resume, or
   with `day_boundary` at local midnight, in which case a continuation arc opens with
   `carried_from_arc_id` if the subject wins again within `R` minutes into the new day.
6. `subject_ref` is one string: the loop's `source_refs[0]`; a chat or town `session_id`; a
   curiosity run id; a reverie chain id; a visual chain id; a dream cycle id. Everything else
   about identity lives in `related_refs`. Returns are counted on `subject_ref` equality only,
   except town, where a second town arc with the same partner slug in `related_refs` counts as a
   return to that partner in `open_threads`.
7. Subject-less events (`metacog_observation`, `gpu_wait`, `attention_row` from lanes whose
   `attended_id` is a text hash) bind by time to the arc open at their `occurred_at` and land in
   `context_event_ids`, or in the day when none is open. They never open or close an arc, and
   they are not `evidence_refs`.
8. `attention_row` events from the curiosity and reverie lanes attach to their process arc by
   `correlation_id` (run id, chain id), not by time.
9. Lanes overlap freely. A workspace arc, an interoception arc, a company arc, and a conversation
   can all be open at once, because they are different questions about the same minutes. A link
   between two lanes is by reference when one exists, and otherwise by time, labelled
   co-occurrence. The workspace subject rule (source refs, loop id fallback, or "no winner") is set
   by patch 1's measurement, as described in the attention section.

### Persistence (`services/orion-sql-db/manual_migration_temporal_self_v1.sql`, new)

Hand-applied, `IF NOT EXISTS`, header naming the single writer and the `psql` command, checked
with `scripts/check_sql_migrations_applied.py --file`.

```sql
temporal_self_event      (event_id text pk, day_id text, occurred_at timestamptz, source_kind text,
                          source_table text, source_ref text, correlation_id text, subject_ref text,
                          related_refs text[], label text, verdict text, payload_json jsonb,
                          ingested_at timestamptz default now())
                          index (day_id, occurred_at); index (occurred_at, event_id)
temporal_self_arc        (arc_id text pk, day_id text, kind text, subject_ref text, began_at timestamptz,
                          ended_at timestamptz, status text, arc_json jsonb, updated_at timestamptz)
                          index (day_id, began_at); index (subject_ref, day_id)
temporal_self_day        (day_id text pk, closed_at timestamptz, day_json jsonb)
temporal_self_projection (projection_id text pk, generated_at timestamptz, projection_json jsonb,
                          created_at timestamptz)   -- singleton "current_day", same shape as substrate _save_projection
temporal_self_cursor     (source_kind text pk, last_occurred_at timestamptz, last_source_ref text, updated_at timestamptz)
```

Arcs are stored in full once (`temporal_self_arc`); the frame carries compact summaries; the
closed day carries full arcs so it outlives arc retention. Broadcast ticks are consumed by cursor
and never copied. One transaction per tick writes new events, changed arcs, the cursor rows, and
the singleton, in that order, as `commit_digest_tick` does in field-digester
(`services/orion-field-digester/app/store.py:414`). The singleton is read with a tolerant loader
so a stale stored row cannot crash-loop the worker, the incident
`scripts/check_substrate_projection_schema_drift.py` exists for.

### Routes (`services/orion-durable-runs/app/main.py`, moved in Rev 4)

- `GET /temporal-self/frame` — the singleton, verbatim.
- `GET /temporal-self/day/{day_id}` — a closed day, or 404.
- `GET /temporal-self/arcs?day_id=` — arcs for a day.
- `GET /temporal-self/cursors` — per-source cursors and lag in seconds; a source whose lag exceeds
  10× its cadence is named in `warnings`.
- `GET /regulation/state` — the latest `RegulationStateV1`, verbatim (also in Redis).

Hub gets a read-only tab later, following the Dream tab (commit `a083beb`), not in the first
three patches.

### Env keys (host `.env_example` and `settings.py`, synced to local `.env`; flags ship ON)

```
# services/orion-durable-runs
TEMPORAL_SELF_ENABLED=true
TEMPORAL_SELF_TICK_SEC=120
TEMPORAL_SELF_ARC_MIN_TICKS=3
TEMPORAL_SELF_RETURN_WINDOW_MIN=180
TEMPORAL_SELF_EVENT_RETENTION_DAYS=30
TEMPORAL_SELF_ARC_RETENTION_DAYS=90
TEMPORAL_SELF_DAY_RETENTION_DAYS=365
ORION_SITUATION_TIMEZONE=America/Denver   # same key and default as cortex-exec and Hub; see Missing question 3
ORION_REGULATION_AROUSAL_ENABLED=true
ORION_REGULATION_STRAINED_GPU_BACKLOG_SEC=300
ORION_REGULATION_STRAINED_CLEAR_SEC=600
ORION_REGULATION_IMMUNE_SWEEP_ENABLED=true
ORION_REGULATION_IMMUNE_SWEEP_MIN_INTERVAL_SEC=21600
# services/orion-attention-runtime
FIELD_ATTENTION_HABITUATION_ENABLED=true
FIELD_ATTENTION_HABITUATION_TAU_SEC=1800
FIELD_ATTENTION_HABITUATION_MAX_DISCOUNT=0.5
# services/orion-dream
DREAM_IDLE_FROM_AROUSAL_ENABLED=true
DREAM_PUBLISH_DRIVE_READING_ENABLED=true
```

The engaged/idle boundary reuses `DREAM_IDLE_MINUTES` (45) instead of copying it. Each arousal
reader's multiplier lives in that reader's own service `.env_example`, and lands when the reader
is wired.

### Metric quality gate for the frame's numeric outputs

Recorded here per CLAUDE.md 0A. Steps 4 and 6 are re-run in patch 1 with live data and the
results appended to this document. The outputs gated: `active_arc_age_sec`, `attention_returns`,
`cumulative_dwell_sec`, `came_from_minutes_ago`, `last_sleep_ended_minutes_ago`,
`open_thread_count`, `oldest_open_thread_age_sec`, `seconds_on_subject_today`,
`runs_on_subject_today`, the constraint counts, and every `ArcBodySummaryV1` field.

1. **Provenance.** Every duration and age is a difference of source timestamps:
   `substrate_attention_broadcast_log.generated_at`, `chat_history_log.created_at`,
   `aitown_chat_history_log.created_at`, `curiosity_offer_decisions.turn_started_at` and
   `curiosity_run_outcomes.completed_at` (`services/orion-sql-db/manual_migration_curiosity_spend_v1.sql:58-90`),
   `substrate_reverie_thought.created_at`, `reverie_visual_attempt.started_at`,
   `reverie_visual_chain.created_at`, `dream_cycle.started_at`/`ended_at`. Every count is a row
   count over one source. Body fields: `chassis_watts`, `peak_pressure`, `peak_pressure_channel`
   from `orion_biometrics_cluster` (`biometrics_cluster.py:61,72-73`, `observed_at` at `:53`);
   `cabinet_temp_c` from athena's `orion_biometrics_summary.measurements` (written by the cabinet
   reader keys in `orion/telemetry/cabinet_sensors.py:38-140`); `cabinet_ambient_spike` rows;
   `home_cooling_sample.switch_on` transitions; `thermal_refused` visual attempts. Diffusion
   energy was dropped: `power_intent_settled` carries no `chain_id`, so attribution to an arc
   would be a guess.
2. **Independence.** `dwell_ticks` on the broadcast projection is the nearest existing signal;
   within one uninterrupted coalition `cumulative_dwell_sec` is a monotone transform of it and is
   **redundant there**; across suspensions and restarts it is not. `attention_returns` and the
   per-subject day totals have no existing equivalent. The `recent_attention` cue renders ages of
   the last few attention rows and is the closest prose sibling of `came_from_minutes_ago`; the
   temporal cue replaces it rather than sitting beside it. The body fields are existing sensors at
   arc altitude, not new sensors. Peak pressure across nodes is the cluster row's own max, so no
   combination rule is invented here.
3. **Theory anchor.** Arc segmentation: Event Segmentation Theory (Zacks, Speer, Swallow, Braver
   & Reynolds 2007), boundaries where the attended subject changes. Arc hierarchy: the
   Self-Memory System's "general events" tier (Conway & Pleydell-Pearce 2000). Return counts to
   unresolved subjects: current concerns (Klinger 1975) and the persistence of interrupted tasks
   (Zeigarnik 1927). The binding itself: autonoetic consciousness (Tulving 1985, 2002). None of
   these licenses a feeling claim; they license the *shape* of the record.
4. **Live-data sanity.** `UNVERIFIED`. Patch 1's replay must show, over seven live days: the
   distribution of broadcast streak lengths (so `K` is measured); how many distinct winners a day
   has, given the recorded degeneracies (on 10-09, `attended_node_ids` was non-empty on 64% of broadcast
   rows, and 76 of 90 loop verdicts were decay); the distribution of returns per
   subject per day, including that zero is common; stretches with no active arc (rest state
   reachable); arcs spanning a substrate-runtime restart without duplication; and that
   `peak_pressure_max` and `cabinet_temp_c_max` vary across arcs and can sit near their floors.
5. **Existing mechanism.** Searched: `EpisodeSummaryV1` (counts per 15 minutes, no subject,
   probably saturated), `attention_salience_trace` (per loop score, no arcs), curiosity
   `RecentRun.written_at` (per run, one process), dbt `dim_reverie_dates` (UTC day counts for
   reverie only), `spark_state_rollups` (a config default only,
   `services/orion-state-journaler/app/settings.py:24`, no migration or model found),
   `substrate_coalition_dwell_log` (24h, current coalition only), `recent_attention_cue` (ages of
   the last few attention rows, no arcs). None binds across processes or across a day.
6. **Reversibility.** Nothing is baked into a schema or training default elsewhere. The tables
   have no foreign keys. The flags ship ON but each one restores the baseline path when turned
   off, and consumer wiring is additive ctx keys. Removal is a
   migration drop and a flag flip.

## Files likely to touch

**Patch 1 (read-only replay, no runtime change):**

- `scripts/analysis/measure_temporal_self_replay.py` (new): reads the source tables over a date
  range, runs the pure reducer, prints the distributions named in gate step 4, writes a fixture
  bundle to `orion/temporal_self/evals/fixtures/`.
- This file: gate findings appended.

**Patch 2 (schemas and reducer, no service):**

- `orion/schemas/temporal_self.py` (new).
- `orion/temporal_self/__init__.py`, `day.py` (the shared `day_id` helper), `sources.py` (one
  pure adapter per `source_kind`, each documenting its timestamp cast), `broadcast.py`
  (`BroadcastTickView` extraction), `arcs.py`, `frame.py`, `projections.py` (new).
- `orion/schemas/registry.py`: `_REGISTRY` entries.
- `orion/temporal_self/tests/test_arcs.py`, `test_sources.py`, `test_day_boundary.py`,
  `test_replay_identity.py`, `test_town_source_filter.py` (new).
- `orion/temporal_self/evals/run_arc_precision_eval.py` and fixtures (new).

**Seams S1 and S2 (producer rows, each its own small PR, any time after patch 1):**

- S1: `services/orion-vision-window/app/presence.py` (append a transition row on each state
  change and an `unknown` row when frames stop; reuse one engine),
  `services/orion-sql-db/manual_migration_vision_presence_transition_v1.sql` (new), tests beside
  the existing presence tests.
- S2: `orion/attention/field_attention/goal_provenance.py` and
  `services/orion-attention-runtime/app/worker.py`, `store.py` (write the completed run when the
  streak target changes), `services/orion-sql-db/manual_migration_field_dominance_run_v1.sql`
  (new); then retire `goal_provenance_streak_ticks` completely: its producer, the
  `debug.attention.streak_tick.v1` channel entry, the sql-writer model and route, and its
  retention setting.

**Patch 3 (worker, migration, routes):**

- `services/orion-durable-runs/app/temporal_self_graph.py`, `temporal_self_driver.py`,
  `temporal_self_store.py` (new). The pattern is `situation_graph.py` and `situation_driver.py`.
  Also `app/main.py` (start block), `app/runner.py:131` (`SELF_DRIVEN_WORKFLOWS`),
  `settings.py`, `.env_example`, `docker-compose.yml`, and `README.md`.
- `services/orion-sql-db/manual_migration_temporal_self_v1.sql` (new).
- `services/orion-durable-runs/tests/test_temporal_self_graph.py`, `test_temporal_self_store.py`
  (new). These run under the existing `.github/workflows/orion-durable-runs-tests.yml`, not a new
  workflow.
- `scripts/smoke_temporal_self.py` (new): asserts a frame, cursors advancing, a day close.
- `.github/workflows/temporal-self-tests.yml` (new, path-filtered), for `orion/temporal_self/`
  and `orion/regulation/` only.

**Patch 4 (first consumers):**

- `services/orion-actions/app/main.py` (`daily_metacog_v1` grounding from the closed day, behind
  a flag), `orion/cognition/prompts/daily_metacog_prompt.j2`.
- `orion/temporal_self/stance_cue.py` (new); `services/orion-cortex-exec/app/executor.py` (the
  `recent_attention` slot at `:4096-4101` takes the temporal cue when the flag is on);
  `orion/schemas/context_provenance.py` (registry entry, kind `live_runtime_projection`);
  `orion/thought/evals/stance_task_boundary.py` re-run unchanged.

**Regulation patches (Rev 4):**

- **R2a**, read-only: `scripts/analysis/measure_dream_pressure_crossings.py` (new). It replays
  `compute_pressure` at every 600 s check over 14 days.
- **R3a**, read-only: `scripts/analysis/measure_arousal_replay.py` (new). It records the time
  share per level over 14 days, and E1 with and without the outreach rows.
- **R1a**, read-only, 7 days after #2528 is live: `scripts/analysis/measure_attention_exposure.py`
  (new). It measures per-target win share over rolling 30-minute windows.
- **R1**, only if R1a says build: `orion/attention/field_attention/selectors.py` (the ranking),
  `orion/schemas/field_attention_frame.py` (`habituation_discount`, consumer-first), and
  `services/orion-attention-runtime/app/worker.py` (the exposure integral, rebuilt from S2).
  Plus settings, `.env_example`, and tests beside the existing selector tests.
- **R2/R3:** `orion/schemas/regulation.py`, `orion/regulation/{arousal,drives,inputs}.py` (new),
  `orion/schemas/registry.py`, `orion/bus/channels.yaml`, and `orion/inner_state_registry.py`.
  In durable-runs, a `regulate` node at the end of the Temporal Self graph.
  `services/orion-dream/app/cycle_store.py` and `app/cycle.py` (filter the idle query to Juniper's turns, read arousal with fallback, publish `drive.reading.v1`).
  Then one small PR per reader in the R3 table, each with a flag-off test that proves the old
  path is unchanged. Plus `tests/test_regulation_must_not_readers.py` (new). For every
  protected gate, it asserts that the decision is identical under every arousal value.
- **R4:** `config/metrics/metric_definitions.lock.json` and
  `config/field/field_channel_glossary.v1.yaml`, with the fields #2528 adds;
  `orion/regulation/immune.py` (new); and fixtures for the three known zombies.

**Later patches:** `orion/curiosity/kickoff_prompt.py` (`_thread_section` fact line),
`services/orion-thought/app/reverie.py` (recent closed arcs), `orion/substrate/system_one_appraisal.py`
(numeric inputs, own gate, `QUESTION_SET_ID` bump), `services/orion-dream/app/replay.py`
(candidates from closed arcs), `services/orion-hub/scripts/temporal_self_routes.py` and a tab,
seam S3 in `services/orion-field-digester` (tension runs of at least 6 ticks).

## Non-goals

- No sentience claim, no "continuity score", no feeling asserted from the `body` field.
- No new service, no LLM call inside the reducer, no generated narrative stored as fact, no
  cross-source verdict vocabulary. Temporal Self adds no bus channel. Regulation adds exactly
  one, `orion:drive:reading`.
- No writes to any source table; no FalkorDB or Graphiti writes in v1. Ticks are never
  crystallized; at most one closed-day summary per day is a candidate for later crystallization.
- No resurrection of `DriveStateV1`, `AutonomyStateV2`, `SelfStateV1`, or the world model. The
  per-tick `goal_provenance_streak_ticks` table is read only offline in patch 1 and is retired
  completely once seam S2 is live.
- No room-camera narrative text, unconfirmed identity guesses, scene-inventory rows, or street
  sightings that Orion's attention did not promote, in the chronology.
- No replacement of `EpisodeSummaryV1`, `ConsolidationFrameV1`, the attention schema, or memory
  consolidation windows. Each remains authoritative for what it already records.
- No NPC-to-NPC town turns in the chronology, and no fix here for their leak into social memory.
- The chronology reducer never reads `metacognition_ticks`, `substrate_field_state`,
  `grammar_events`, `substrate_episode_summaries`, `substrate_system_one_appraisal`, or
  `substrate_reduction_receipts`. The one exception is the R4 immune sweep, which reads
  `substrate_field_state`, only when idle and at most every 6 hours.
- No opportunity-cost estimate. The constraints channel records rows the system already stamped
  as deferred or waited, with the stamping row's own reason.
- Temporal Self itself changes no spend point, budget, cap, or gate. Regulation changes only the
  seven readers named in R3, and only in the conservative direction. No daily cap is raised. No
  safety, courtesy, or spend gate reads arousal.
- Everything in regulation runs in `orion-durable-runs`, `orion-attention-runtime`, and
  `orion-dream`. No narrative in any frame or regulation state.
- No explore, social, cooling, or effort drive in v1. Each failed the metric gate above. No
  "mood", "valence", or emotion label is attached to arousal.
- No parallel attention scorer: habituation lives only inside #2528's ranking.
- No new metric registry: the immune sweep reads the glossary and metric lock, which #2528
  extends.
- The same no-resurrection rule covers DriveEngine and `signal_drive_map.yaml`.

## Acceptance checks

1. **Replay identity.** Fold seven fixture days twice, from an empty state and from a mid-day
   checkpoint; arcs, ids, and frames are byte-identical. A simulated substrate-runtime restart
   (broadcast log gap, `dwell_ticks` reset in `projection_json`) produces no duplicate arc.
2. **Arc precision.** On a hand-labelled fixture day built from real rows in patch 1, every
   `evidence_ref` of an arc shares the arc's `subject_ref` exactly; `context_event_ids` are the
   only place a subject-less event may appear; a mixed-producer `social_room_turns` fixture and an
   `aitown_chat_history_log` fixture with `source=orion-ai-town` rows yield zero town arcs from
   those rows; a label change on an open loop does not split its arc when `source_refs` is
   unchanged.
3. **Rest state.** A quiet fixture window (no broadcast winner, no process events) yields
   `active_arc=None`, `attention_returns=0` for every subject, and a frame that still validates
   and still advances cursors. A day whose broadcast winner never changes yields one attention
   arc and the warning named in danger mode 2.
4. **Day boundary.** An arc open at 23:58 local closes with `day_boundary` and its continuation
   carries `carried_from_arc_id`; `day_id` agrees with `TimeContextV1.local_date` for the same
   instant; a naive `chat_history_log.created_at`, a naive `metacog_trigger.timestamp`, and a
   TEXT `orion_biometrics_summary.timestamp` each land in the correct local day in a fixture
   straddling midnight.
5. **Late evidence.** A reverie verdict scored after its arc closed attaches by event id and does
   not reopen it; a dream hypothesis adopted as a prior two days later attaches to the sleep arc
   of the day it was made; a `:PriorRevision` attaches to its curiosity arc by run, never by its
   LLM-written `written_at`.
6. **Live proof (patch 3).** One real `temporal_self_arc` row with `attention_returns ≥ 2` whose
   evidence refs resolve in two source tables; `GET /temporal-self/cursors` shows every enabled
   source's lag under 10× its cadence; one real closed day in `temporal_self_day`; the host's
   `/health` answering. Until collected, the live path is `UNVERIFIED` and the PR says so.
7. **Consumer effect (patch 4).** The daily metacog output cites at least one arc id and its
   evidence window equals the closed day, not a rolling 24 hours; the stance boundary eval passes
   unchanged with the cue in the `recent_attention` slot; an ablation with the cue removed changes
   no stance decision that the eval scores.
8. **Metric gate on live data (patch 1 exit).** The distributions in gate step 4 are recorded in
   this document, and `K` and `R` are set from them.
9. **Attention, field attention, vision.** The reducer's self-prediction scoring reproduces the
   calibration script's 66.0% test accuracy on the same rows before `ArcSelfModelSummaryV1` is
   bound; a fixture day with a 9-hour single-target run yields one interoception arc and the
   stuck-reading warning; a fixture with a dark camera closes the company arc with
   `source_stale`; no event carries room narrative text or an unconfirmed identity; `juniper_chat`
   events never appear in anything sent to contractor peers.
10. **Operational.** Retention runs; the migration passes `check_sql_migrations_applied.py`; the
   tolerant singleton loader is tested against a stale row; no module under
   `orion/temporal_self/` names a System One observational question (the existing
   no-consumers test stays green); the smoke script runs against the Tailscale bus URL only;
   local `.env` is synced.
11. **Habituation (R1, only if R1a says build).** R1a's per-target exposure distribution, taken
    after #2528, is recorded here first. If R1 is built, the fixtures are:
    - **High-band internal target.** It stays eligible in band `high` for 6 hours, alternating
      with a second high-band target every 10 ticks. After about `τ`, a third eligible target
      with a lower percentile wins at least once.
    - **Monotonous external source.** It does the same.
    - **Band change.** When the band moves on a habituated target, that target is restored on
      the next frame.
    - **Unusual band.** A target in band `unusual` keeps winning for 6 hours.
    - **Restart.** The exposure integral survives a simulated restart, rebuilt from S2 rows.

    Live, measured against #2528-alone as the baseline (not against the 10-08 pre-#2528 data):
    the longest single-target hogging stretch R1a found gets shorter.
12. **Rest drive (R2).** R2a's crossing distribution is recorded here before the threshold
    changes. A fixture with an outreach-only `chat_history_log` row reads idle. Live: within
    14 days, at least one cycle fires later than the six-hour clock allowed because pressure was
    below threshold, and the regulation state shows the `building → due → refractory → building`
    sequence. If the R2a result rules this out, Missing question 13 applies, and the PR says so.
13. **Arousal (R3).** R3a shows every level between 0% and 100% of time. Every reader's flag-off
    test reproduces its current decision on the same inputs. The must-not test shows every protected
    gate's decision is invariant across arousal values. `unknown` never authorizes a dream through
    arousal, and a missing regulation state falls back to the dream's own Juniper-filtered idle
    query. Live: `engaged` appears within
    one step of a real Juniper turn, and `idle` holds overnight.
14. **Immune sweep (R4).** All three zombie fixtures are caught, and the healthy and calm-at-rest
    fixtures are not. The sweep never runs outside `idle`. A finding lands as an
    `immune_finding` event in the closed day.

## Recommended next patch

Work runs in two parallel lanes, the chronology lane and the regulation lane. They meet at S2,
which both lanes need.

Patch names follow the rest of this document: patches 1-7 and S1-S3 are the Temporal Self
patches named in "Files likely to touch", and R1a-R4 are regulation. The Order column only sets
sequence.

| Order | Patch | Exit evidence |
|---|---|---|
| **1** | **S2**: dominance runs in orion-attention-runtime; retire the streak-tick table completely | Run rows match the offline reconstruction from the last 7 days of streak ticks (expect about 2,900 a day). The streak-tick producer, the `debug.attention.streak_tick.v1` channel, the model, and the table are all gone |
| 2 | **Patch 1 + R2a + R3a**: the read-only replays, in one PR | Distributions recorded here. `K`, `R`, the dream threshold (or the Missing question 13 verdict), and the arousal hysteresis are chosen from them. No runtime change |
| 3 | **R2/R3 core**: regulation schemas, the pure reducer, the dream idle fix, `drive.reading.v1`, and a `regulate` node in a self-driven durable-runs thread. Patch 3 later adds the Temporal Self nodes to the same thread | Live `orion:regulation:latest` reads `idle` overnight and `engaged` within one step of a turn. Outreach rows no longer reset dream idle. Deleting the Redis key leaves the dream on its fallback |
| 4 | **Patch 2**: Temporal Self schemas, pure reducer, tests, arc-precision eval | Tests and eval green; replay identity holds |
| 5 | **Patch 3**: Temporal Self nodes in the durable-runs thread, migration, routes | One real arc with returns ≥ 2; one closed day; cursors advancing |
| 6 | **R3 readers**: one small PR per reader, in the R3 table's order | Each reader has a flag-off test and a branch-level must-not test; one live `arousal_strained` refusal |
| 7 | **R1a**, 7 days after #2528 is live; **R1** only if R1a finds a target hogging attention | R1a recorded here; acceptance check 11 if R1 is built |
| 8 | **Patch 4**: daily metacog grounding and the stance cue | Metacog cites arc ids; stance boundary eval unchanged |
| 9 | **R4** immune sweep, after #2528's lock fields land | Acceptance check 14 |
| later | Patches 5-7, S1, S3 | As in Rev 3 |

Start with S2. It is small, it retires a 41,000-row-a-day debug table that its own schema calls
temporary, and both lanes need it: the interoception arcs and the habituation memory are both
built from its rows.

## Appendix: source binding table

Every source the v1 reducer reads, the column it treats as occurrence time, and the ref it uses
as subject identity. A source not in this table is not read.

| `source_kind` | Table | Occurrence time | `subject_ref` | Notes |
|---|---|---|---|---|
| broadcast tick (driver, not stored) | `substrate_attention_broadcast_log` (`log_id`, `generated_at`, `projection_json`) | `generated_at` | `projection_json.frame.open_loops[selected].source_refs[0]` | drives rule 1; `dwell_ticks`, `attended_node_ids` ignored |
| `chat_turn` | `chat_history_log` | `created_at` (naive; cast documented) | `session_id` | stores `correlation_id` only; no text; AI Town rows are routed elsewhere by sql-writer |
| `town_exchange` | `aitown_chat_history_log` (`source=orion-embodiment` only) | `created_at` (naive) | `session_id` | `related_refs` = partner slug via `orion/town_cast.py`; `participant_kind=human` ⇒ Juniper class |
| `curiosity_run` | `curiosity_offer_decisions` (`turn_started_at`, offered priors) + `curiosity_run_outcomes` (`completed_at`) | `turn_started_at`, else the curiosity attention row's `generated_at` | run id | `related_refs` = offered prior ids; spend log live (7 completed runs a day, 10-02 to 10-08) |
| `attention_row` | `substrate_attention_schema` | `generated_at` | None (context) | `process` in payload; curiosity/reverie lanes attach by `correlation_id` (rule 8); `attended_id` never a subject; cortex_turn `attended_label` not copied |
| `reverie_chain` | `substrate_reverie_chain` (`created_at`) + `substrate_reverie_thought` (`created_at`) | chain `created_at`; thoughts by their own `created_at` | chain id | terminal reason in payload |
| `expectation_verdict` | `substrate_reverie_thought` (`expectation_scored_at`); `vision_percept_expectation` (`scored_at`) | the scoring column | None (attaches by thought id / `subject_key`) | verdict = the source's own word |
| `visual_run` | `reverie_visual_chain` + `reverie_visual_attempt` | `attempt.started_at` when present, else `chain.created_at` as a point | visual chain id | caption, `context_slot_used`, `thermal_gate` in payload |
| `visual_deferral` | `reverie_visual_attempt` | `started_at` | None (constraint) | `outcome` ∈ deferred_thermal / deferred_busy / deferred_resource / failed; the constraints channel |
| `gpu_wait` | `gpu_pool_events` (`event='granted'`, `waited_ms` ≥ threshold, `holder NOT LIKE 'http:%'`) | `generated_at` | None (constraint) | same predicate the admission cue uses (`admission_cue.py:91-97`) |
| `dream_cycle` | `dream_cycle` | `started_at`/`ended_at` | cycle id | covered interval from `cycle_json.pressure` |
| `dream_hypothesis` | `dream_hypothesis` | `created_at`; `offered_at`; `expires_at` | None (attaches to its sleep arc by `cycle_id`) | adoption via `:Prior.formed_from` |
| `prior_revision` | FalkorDB `orion_worldview` `:PriorRevision` | attached by run, not by `written_at` (LLM-written, mixed formats) | None (attaches to curiosity arc) | bounded Cypher read in the `agency_episode_reader` style |
| `action_outcome` | `substrate_action_outcomes` | outcome time | None (self-change) | verdict = `claim_upheld` true / false / null |
| `peer_ask` | FalkorDB `PeerAskCommit` / `PeerBriefOffer` / `PeerBriefDecision` | `committed_at` etc. (epoch ms) | `help_id` | only when present; absent is fine |
| `metacog_observation` | `orion_metacog` ⋈ `metacog_trigger` | `metacog_trigger.timestamp` (naive utcnow), else cast of `orion_metacog.timestamp` (TEXT) | None (context) | severity ∈ degraded / critical; `trigger_kind` in payload |
| `consolidation_window_close` | `memory_consolidation_windows` | close time | None (closes the matching conversation arc) | one input, not the authority |
| `attention_loop_raised` | `attention_salience_trace` where `scope='chat'` | `created_at` | `loop_id` | opens a concern arc; label `privacy_class=juniper_chat` |
| `attention_loop_verdict` | `attention_loop_outcome` | `created_at` (digest detection time for `decayed_unattended`) | `loop_id` | verdict = resolved / dismissed / decayed_unattended |
| `field_dominance_run` | `field_dominance_run` (seam S2) | `started_at` / `ended_at` | `target_id` | interoception arcs; before S2, reconstructed offline from `goal_provenance_streak_ticks` in patch 1 only |
| `presence_transition` | `vision_presence_transition` (seam S1) | `occurred_at` | `presence_id` | company arcs; `identity_confirmed`, plus the enrolled label only when confirmed |
| `vision_percept` | `vision_events` from room streams (or NULL `stream_id`) | `created_at` (write time; no observation time exists) | None (context) | `event_type` and `entities` only, never `narrative`; folds into `percept_entities` |
| `unresolved_percept` | `vision_unresolved` | `observed_at` | None (context) | reason = council_uncertainty / no_label; table needs the walkway migration |
| `attention_worthy_sighting` | `vision_events` where `event_type='attention_worthy'` | sighting `started_at` via `evidence_refs`, else `created_at` | None (context) | walkway only; not live |
| attention summary | `substrate_attention_schema` | per arc interval | none | rows and reason words per lane, never per row |
| self-model summary | `substrate_attention_self_model` | per arc interval | none | the calibration script's two-rows-ahead rule; confidence excluded |
| `memory_episode` | `episode_memory` | `occurred_at` | None (attaches to the conversation arc by time and session) | `episode_id`, `purpose` only; `statement` is model prose and is not copied |
| `situation_revision` | Redis `orion:situation:latest` / bus `orion:situation:state` (`SituationStateV1`) | `updated_at` | None (context) | revision number and thread id only; Juniper's whereabouts stay in the Situation Graph |
| `immune_finding` | written by R4 | sweep time | metric id | Orion's own instrument findings |
| body summary | `orion_biometrics_cluster`; athena `orion_biometrics_summary`; `cabinet_ambient_spike`; `home_cooling_sample` | per arc interval | none | aggregated per arc, never per row |

## Follow-ups recommended outside this design (not filed; no issue tracker write was made)

- The 12h–48h conversation-phase gap in `orion/situational/context.py:880-903`.
- The `daily_metacog_v1` window mismatch (rolling 1440 minutes labelled as yesterday).
- NPC-to-NPC `social_room_turns` updating Orion's own peer and stance rows in
  `services/orion-social-memory/app/service.py:278` onward.
- `scripts/check_schema_registry.py` and `scripts/check_bus_channels.py` are named in
  `CLAUDE.md` sections 11 and 17 but do not exist; the real gates are the registry-agreement test
  and per-contract catalog tests.
- Three timezone keys (`ORION_SITUATION_TIMEZONE` in cortex-exec and Hub, `ACTIONS_DAILY_TIMEZONE`
  in orion-actions) with no shared helper.
- `services/orion-consolidation-runtime/README.md:11` still lists the retired `substrate_self_state`
  as an input.
- `orion-consolidation-runtime` runs a 2026-09-25 image. It logged 736,718
  `consolidation_row_incompatible_schema` warnings for `ProposalFrameV1` in the 24 h to 10-09.
  Rebuild it from main.
- Outreach sent 0 messages on 10-08. Its `daily_cap` refusals stopped at 05:00 UTC, and
  `content_already_used` (1,297) and `empty_generation` (49) took their place.
- Settings defaults disagree with `.env_example` for `HUB_CURIOSITY_INVESTIGATION_DAILY_CAP`
  (3 vs 7), `HUB_CURIOSITY_INVESTIGATION_MIN_COOLDOWN_SEC` (14400 vs 1800),
  `HUB_CURIOSITY_SELF_INQUIRY_DAILY_CAP` (3 vs 7), and `GPU_POOL_SHED_ENABLED` (false vs true).
  This is the memory rule ".env_example must match the live intended default".
- The Situation spec's rollback flags (`ORION_SITUATION_GRAPH_ENABLED`, spec :187) do not match
  the code's `SITUATION_GRAPH_ENABLED`.
- Spark's dead `arousal` field (`orion/schemas/telemetry/spark.py:42`) should be retired once its
  readers have been checked (`orion/signals/adapters/spark.py:96`,
  `orion/substrate/adapters/spark.py:108,113`, `orion/reasoning/adapters/spark_state.py:19`).
