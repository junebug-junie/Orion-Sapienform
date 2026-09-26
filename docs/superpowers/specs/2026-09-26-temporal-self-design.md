# Temporal Self: a reducer that binds Orion's day into one continuing chronology

- **Date:** 2026-09-26
- **Status:** DESIGN, proposal mode (CLAUDE.md 0A: touches self-modeling, memory, and cognition
  loops). Nothing in this document is built. No runtime change is authorized by it.
- **Evidence basis:** a read-only survey of `main` at `9509af4`. This session had no access to the
  live database or hosts. Every live number is quoted from a dated spec or PR report in this repo
  and says so. A finding from reading code that was not reproduced live is marked `UNVERIFIED`.
  Seven scoped read-only audits (core primitives, dreams, the metacog table, AI Town, substrate
  grammar, visual reverie plus biometrics, and house reducer conventions) fed this document; their
  file:line citations were spot-checked against the tree before being used here.
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
is never written to a table, the state deltas live 30 minutes, the attention broadcast's dwell
counters reset every restart, the agency episode has no schema or table yet, and "what Orion gave
up" is a question in a design doc, not a field. What *is* durable and live: the cross-process
attention table, the broadcast log, the goal streak ticks, reverie thoughts with scored
expectations, dream cycles with expiring hypotheses, curiosity runs and prior revisions, town
conversations with named partners, visual reverie chains with deferral reasons, 30-second body
readings, and metacog rows with a trigger kind. That is enough.

**Proposal.** Build one pure, replayable reducer that folds those existing tables into a
chronology of *arcs* (a stretch of the day Orion kept returning to the same subject), and a
bounded *day frame* that says where Orion is in its day, what it came from, which threads are
still open and how old they are, which expectations are pending or scored, what its body was
doing during each arc, and what it was prevented from doing. Persist that in Postgres. Hand each
consumer a different, bounded slice of it: stance gets one sentence of "where am I in my day";
daily metacog gets the closed day instead of a rolling 24 hours of chat; curiosity gets "you have
already spent N minutes on this today"; reverie gets the last few closed arcs; System One gets
numbers, later, after their own gate.

No new service. No LLM inside the reducer. No narrative in the frame. Identity of a subject is an
exact reference (a node id, a loop source ref, a prior id, a town partner slug), never an
embedding or a label match. The reducer's output is a claim Orion can be held to: every arc
carries the row ids that built it.

**Why this matters for the mission.** Autonoetic continuity, the sense that the remembered past
and the anticipated future belong to the same self, is a named prerequisite for sentience and
is not something a chat context window can supply. This document claims no feeling. What it
does is make the question "did Orion's morning change Orion's afternoon?" answerable from rows.

## Corrections to the pasted analysis

Read this table before the rest. Each row is a claim the earlier analysis leaned on, what the
code actually says, and what that does to the design.

| Claim in the pasted analysis | What the code says | Consequence |
|---|---|---|
| `FieldGoalProvenanceV1` is "tailor-made" for sustained direction | It is published on `orion:memory:goals:proposed` and **never persisted**; every consumer keeps only the latest in memory (`services/orion-attention-runtime/app/worker.py:185-250`, `orion/substrate/goal_context_listener.py:42`). The durable trace is the per-tick telemetry `goal_provenance_streak_ticks` (14-day retention) and the single upserted streak row `substrate_goal_provenance_streak`. | Read dominance from `goal_provenance_streak_ticks`, not from goal records. |
| `StateDeltaV1` is "essentially a typed answer to what changed in me" | It has **no timestamp and no correlation id** (`orion/schemas/state_delta.py:8-29`); time lives on the wrapping `ReductionReceiptV1.created_at`. Receipts are pruned after **30 minutes** on success (`services/orion-substrate-runtime/.env_example:302-303`, `app/receipt_pruner.py:16-39`). One delta is a 2-second channel nudge. | Raw deltas are telemetry, not autobiography. Take "what changed in me" from the higher-order changes that already persist (prior revisions, expectation verdicts, action outcome posteriors, hypothesis adoption) and take substrate activity level from `substrate_episode_summaries`, which already rolls receipts into 15-minute windows. |
| Broadcast history gives `dwell_ticks`, stability, transitions | Dwell, stability and transition history are **module globals** that reset on every restart (`orion/substrate/attention_broadcast.py:44-49`). Stability is a three-step constant (0.9/0.6/0.3, `:501-507`). Transitions carry no coalition ids. The append-only log `substrate_attention_broadcast_log` (168h) has **no live reader**, only offline scripts. | Recompute dwell and returns from the log's row sequence (`selected_open_loop_id`, `attended_node_ids`, `generated_at`). Do not trust `dwell_ticks`. Temporal Self becomes the log's first live consumer. |
| Agency episodes give "durable before/after semantics" today | No `AgencyEpisodeV1`, no table. What merged (PR #2366) is a read-only audit dict whose verdicts all read UNVERIFIED, plus three FalkorDB nodes for the ask lane (`PeerAskCommit`, `PeerBriefOffer`, `PeerBriefDecision`, `orion/curiosity/agency_episode.py`) behind `CURIOSITY_PEER_EPISODES_ENABLED`, code default false, not deployed. | Bind to those nodes when present. Do not depend on them. The expectation channel in v1 comes from reverie verdicts, dream hypotheses, vision percept expectations, and action outcomes. |
| Stakes work supplies "what Orion gave up" | Built from that design: the curiosity spend log (`curiosity_offer_decisions`, `curiosity_run_outcomes`, migration not yet applied per its PR report) and two defect fixes. No opportunity-cost field exists anywhere. | The constraints channel in v1 records what the system *already* stamps as refused or deferred: visual reverie `deferred_thermal`/`deferred_busy`/`deferred_resource`, `thermal_refused`, allocator `unmeasurable`/`below_information_floor` stamps on dispatch frames, GPU pool `waited_ms` on `gpu_pool_events`, and curiosity's daily-cap block. It does not invent a counterfactual. |
| Open loops have an "issue identity" | `OpenLoopV1` has no timestamp; its `id` is a 12-hex hash of the loop's text (`orion/substrate/attention/scoring.py:115-117`), so it changes when a label changes. The stable node id sits in `source_refs` (`orion/schemas/attention_frame.py:80-87`). | Subject identity for loops is `source_refs`, with the loop id secondary. |
| `predicted_next` on `AttentionSchemaV1` is a prediction to score | All five lanes fill it, each with a different meaning (a domain trend, a reverie `next_focus`, a curiosity `continue_note`, a deferred item, a state-machine `next_node`). **Nothing reads it.** | Not scored in v1. Curiosity's `continue_note` versus the next run's question is the one lane worth scoring later. |
| `temporal_phase` lives in `session_turn_phase.py` | That module stores two raw timestamps in Redis (7-day TTL). `temporal_phase` is a cortex ctx key set at `services/orion-cortex-exec/app/executor.py:3179` from `conversation_phase.phase_change`. | Naming unchanged: still not the temporal self. |
| `FieldAttentionFrameV1` is in `field_goal.py`; `SituationBrief` | It is in `orion/schemas/field_attention_frame.py:40`; the class is `SituationBriefV1` (`orion/schemas/situation.py:663`). | Citations fixed below. |
| `cognition_traces` records "what Orion executed" | Primary key is `correlation_id` and writes go through `merge`, so a unified turn's several legs overwrite each other; last writer wins (`services/orion-sql-writer/app/worker.py:1857`, code-derived, `UNVERIFIED` live). No retention. | Take turn chronology from `substrate_attention_schema` (`cortex_turn` rows) and `chat_history_log`, not from `cognition_traces`. |
| Memory consolidation windows are usable arc boundaries | Windows close only when a new turn arrives, on an LLM boundary score (`services/orion-memory-consolidation/app/boundary.py:23-34`). The phase bucketing they read has a gap: a 12h to 48h gap falls through to `"unknown"`, and the day-crossing rescue is skipped for `"unknown"` (`orion/situational/context.py:880-903`). | Window-close events are *one input* to conversation arcs, never the authority. The bucketing gap is a separate bug, recorded as a follow-up below. |
| `EquilibriumSnapshotV1` gives embodied context | It is bus-only; live state is a Redis hash. Only `equilibrium_service_transitions` persists. `metacognition_ticks` persists a 15-second zen score that read 0.965 ± 0.010 over seven days (spec 2026-09-24, line 97): degenerate. | Organs participating comes from `equilibrium_service_transitions`. Zen is excluded. |

One more thing the pasted analysis did not know about: `EpisodeSummaryV1`
(`orion/substrate/episodic_consolidation.py`) already rolls reduction receipts into 900-second
windows every 300 seconds, stored in `substrate_episode_summaries` with 14-day retention. It
holds counts only, no channel deltas, no narrative. Temporal Self consumes it as the substrate
activity level per quarter hour; it does not replace it.

## Current architecture

### The clock

- `TimeContextV1` (`orion/schemas/situation.py:53`): `local_date`, `local_time`, `day_phase`
  (seven buckets), built per turn in `orion/situational/context.py:789` from
  `ORION_SITUATION_TIMEZONE` (default `America/Denver`). Not persisted; rides the cortex result
  metadata to Hub.
- `ConversationPhaseContextV1` (`:81`): `crossed_day_boundary`, `phase_change`.
- Every day boundary in the repo is local midnight, and every consumer computes it on its own:
  orion-actions' `build_daily_window` (`services/orion-actions/app/main.py:205`), the curiosity
  daily cap (`services/orion-hub/scripts/curiosity_investigation.py:1212`), the chat compactor
  (`orion/cognition/chat_history_compactor/window.py:49`), and `dream_date`
  (`orion/schemas/telemetry/dream.py:81`, container-local `date.today`, timezone `UNVERIFIED`).
  There is no shared "which day is it" helper.
- Two other boundary styles exist: dream cycle v2 uses "since the last sleep"
  (`services/orion-dream/app/cycle.py:53-57`), and consolidation uses UTC buckets.

### Attentional chronology (live, durable)

| Source | Table | Time column | Cadence | Retention | Subject ref |
|---|---|---|---|---|---|
| `AttentionSchemaV1`, five lanes (`orion/schemas/attention_schema.py:92`) | `substrate_attention_schema` | `generated_at` | substrate ~30s; reverie per chain; curiosity per run; cortex per turn; durable_run per transition | 90d (`services/orion-sql-writer/app/settings.py:404`) | `attended_id`, `correlation_id` |
| `AttentionBroadcastProjectionV1` log | `substrate_attention_broadcast_log` | `generated_at` | ~30s | 168h | `selected_open_loop_id`, `attended_node_ids`, open loops' `source_refs` |
| `AttentionSelfModelV1` (`orion/schemas/attention_self_model.py:26`) | `substrate_attention_self_model` | `generated_at` | ~30s | 168h | `predicted_shift`, `prediction_error_by_domain`, heartbeat fields |
| `FieldAttentionFrameV1` (`orion/schemas/field_attention_frame.py:40`) | `substrate_attention_frames` | `generated_at` | ~2s | 72h | `dominant_targets[].target_id` |
| Dominance streaks (`DominanceStreakTickV1`, `orion/schemas/field_goal.py:63`) | `goal_provenance_streak_ticks` | `observed_at` | ~2s | 14d | `target_id`, `streak_count`, `qualified` |
| Per-loop salience history | `attention_salience_trace` | `created_at` | per reverie tick / chat turn | (per table) | `loop_id`, `theme_key` |
| System One shadow appraisal (`SystemOneAppraisalFrameV1`) | `substrate_system_one_appraisal` | `generated_at` | ~30s | 168h | `curiosity_pull`, `deliberation_need`, source ids |

Live evidence: 9,049 attention-schema rows on 2026-09-08 (`docs/superpowers/specs/2026-09-08-orion-anatomy-inspection.md:51`); 6,359 broadcast-log rows on 2026-08-13 (`orion/schemas/attention_self_model.py:47`); 19,408 self-model rows in seven days (same docstring). The curiosity lane of the attention schema is claimed live but has no per-lane count on record (`UNVERIFIED`).

### Expectations and their verdicts (live, durable)

| Source | Where | Commit time | Resolution time | Verdict vocabulary |
|---|---|---|---|---|
| Text reverie expectation | `substrate_reverie_thought.expectation`, `expectation_checkable_by`, `expectation_verdict`, `expectation_scored_at` | `created_at` | `expectation_scored_at` | confirmed / disconfirmed / unscored (judge window 1800s; falls back to unscored on any failure) |
| Dream hypothesis (v2) | `dream_hypothesis` | `created_at`, `expires_at` (+72h) | `offered_at` then a `:Prior` in FalkorDB `orion_worldview` with `formed_from="dream_hypothesis:<id>"`; tested/supported/refuted via the prior's own fields | scorecard verdict "too early" until 20 dream-arm and 5 control-arm offers (`orion/dream/hypotheses.py:177-188`) |
| Vision percept expectation | `vision_percept_expectation` | before each window | after the window | met / missed / unscorable |
| Motor action | `ExpectedEffectV1` inside `substrate_execution_dispatch_frames`; `ActionOutcomeRecordV1` in `substrate_action_outcomes`; posteriors in `substrate_action_effect_posterior` | frame saved **after** send (no precommit, per the agency audit) | outcome window | resolved / clamped |
| Curiosity prior revision | `:PriorRevision {from_confidence, to_confidence}` in `orion_worldview`; `HopRecord.written_at` (`orion/curiosity/worldview.py:260-277`, None before 2026-09-19) | run | run | confidence moved; inconclusive tests leave no revision |
| Ask to a peer | `PeerAskCommit {committed_at, deadline_at}`, `PeerBriefOffer`, `PeerBriefDecision` (FalkorDB, epoch ms) | before invocation | on brief | flag default off, `UNVERIFIED` live |

### Self-observation (live)

- `orion_metacog`: one row per trigger, about 2,400/day, 76,058 of 114,437 rows are transport
  and 36,530 telemetry anomaly (spec 2026-09-24, lines 41-54). `timestamp` is a **varchar ISO
  string** with no index; there is no session or turn column; `severity` and `causal_density` are
  deterministic from the trigger's own upstream since 2026-09-24 (rows carry
  `severity_def:event_v1`, `orion/metacog/evidence_map.py:49`). `summary` and `mantra` are LLM
  prose of poor quality (68% echoed the prompt's example on 2026-09-03,
  `services/orion-cortex-exec/app/executor.py:878-887`).
- `metacog_trigger`: the raw trigger sink with `trigger_kind`, `reason`, `pressure`,
  `signal_refs`, and a naive `timestamp`. Joins to `orion_metacog` on `correlation_id`.
- `orion_metacognitive_trace`: per-chat-turn reasoning text, 14-day retention.
- `daily_metacog_v1` (orion-actions, 20:15 local): reads a rolling 1440 minutes of
  `chat_history_log` plus the AI Town table through recall
  (`orion/recall/profiles/journal.daily.metacog.grounded.v1.yaml:18-20`,
  `services/orion-recall/app/sql_timeline.py:320-333`), while its prompt labels the output as
  yesterday's local calendar day (`services/orion-actions/app/main.py:205-222`). It never reads
  `orion_metacog`. Output goes to `notify_requests` and a self-experiments candidate; there is no
  table and no journal entry.

### Sleep (live, ran once)

Dream cycle v2 (`services/orion-dream`, merged 2026-09-25): pressure-triggered (≥ 3.0), idle-gated
(no chat for 45 minutes), at least six hours apart, never nightly. It replays up to 12 items from
four sources since the last non-failed cycle's start (`cycle_store.py:62-70`), recombines them into
hypotheses across a dream arm and a control arm, and persists `dream_cycle`
(`started_at`, `ended_at`, `cycle_json.pressure.since` and `.computed_at` as the covered interval),
`dream_replay_item` (no timestamp), and `dream_hypothesis`. Hub curiosity kickoff claims up to
three unoffered, unexpired hypotheses per run. One live sleep as of 2026-09-26: 12 replay items,
4 hypotheses, all offered, none adopted (`docs/superpowers/pr-reports/2026-09-26-hub-dream-pr.md:101-107`).
The legacy narrative `dreams` table (17 rows, last 2026-09-06) stamps `dream_date` with the run
date, not the day dreamed about.

### Social (live status `UNVERIFIED` since 2026-09-15)

- `aitown_chat_history_log`: Orion's verbatim exchanges in town, `source=orion-embodiment`,
  `session_id=aitown:<convex_conversation_id>`, `client_meta.external_participant`
  `{participant_id, participant_name, participant_kind: npc|human}`, `created_at` **naive**.
  Juniper's in-town lines land here too, because routing looks at platform, not partner.
- `social_room_turns`: two producers share it. `source=orion-embodiment` rows are Orion's own
  exchanges. `source=orion-ai-town` rows are NPC-to-NPC and NPC-to-Juniper turns Orion did not
  witness, posted by Convex (`services/orion-ai-town/patches/orion-town-continuity-ingest.patch:102-155`).
- `journal_entries` with `source_kind="embodiment"` and `trigger_kind="town_episode"` (in
  `journal_entry_index`): LLM-drafted episode notes with a timestamp.
- `social_participant_continuity.last_seen_at` per partner slug (stored as a string).
- Nothing renders relative time ("yesterday") or a partner's claim into Orion's town prompt.
- **Leak, out of scope but load-bearing for this design:** `process_social_turn` does not filter
  on `source`, so NPC-to-NPC turns update the same peer row Orion reads as "my relationship with
  Nico" and blend into the single global stance row (`services/orion-social-memory/app/service.py:260-350`,
  `:815-821`). Recorded as a follow-up below.

### Imagery (live)

- Visual reverie: `reverie_visual_chain` (`created_at` stamped at the **end** of the run;
  `chain_json` holds `thermal_gate`, `context_slot_used`, `description`, `production_receipt`),
  `reverie_visual_artifact` (caption), `reverie_visual_attempt` (`started_at`, `outcome` in
  produced / deferred_thermal / deferred_busy / deferred_resource / already_satisfied / failed).
  Started on demand through proposal → policy → dispatch → `render_scene` since the 600s cron was
  retired 2026-08-30; a homeostatic baseline makes one due every 90 minutes
  (`config/proposals/visual_baseline.v1.yaml`, `orion/reverie/baseline.py:53-84`). Thermal gate
  hot at 32.0 °C, re-arm at 30.5 °C (`orion/autonomy/thermal_gate.py:47-53`).
- Text reverie: `substrate_reverie_thought`, `substrate_reverie_chain` (terminal reasons
  `pressure_discharged` / `max_steps` / `no_coalition` / `refractory` / `low_salience`), tick
  every 90s.
- Day-level facts already exist for both in dbt (`services/orion-analytics`:
  `fct_visual_reverie_chains`, `dim_reverie_dates`, a **UTC** day spine).

### Body (live, no day record)

- `orion_biometrics_summary`: every 30s per node, `timestamp` **TEXT**, composites `strain` and
  `homeostasis = 1 − strain` (`biometrics_pipeline.py:481-486`), measurements incl.
  `cabinet_temp_c`, `cabinet_humidity_pct`, `cabinet_ambient_rms`, `chassis_watts`. Retention
  not found (`UNVERIFIED`).
- `orion_biometrics_cluster`: `observed_at` timestamptz, `chassis_watts`, `gpu_watts_total`,
  `peak_pressure`, `peak_pressure_channel`, 30-day retention.
- `power_intent_settled`: per-generation energy (`workload_kind="reverie_diffusion"` carries no
  `chain_id`; join by time window).
- `cabinet_ambient_spike`, `home_cooling_sample` (AC watts, `switch_on`; read-only, Orion does
  not control cooling per `docs/superpowers/specs/2026-09-25-zwave-cabinet-cooling-design.md`).
- Reaches cognition only as live snapshots: `CabinetContextV1` in the situation brief, a metacog
  biometrics cue, the field channels. No rollup, no narrative consumer.

### Substrate

- Grammar is an event-sourced trace vocabulary, not a production system: `GrammarEventV1`
  (`orion/schemas/grammar.py:176-197`) with `observed_at` (occurrence) and `emitted_at`;
  `grammar_events` and `grammar_traces` at 3-day retention, roughly 1.4M events a week
  (`docs/superpowers/specs/2026-09-22-substrate-lattice-audit.md:52-54`). Lane reducers in
  substrate-runtime pull by Postgres cursor (`substrate_reduction_cursor`) into singleton
  projections and emit `StateDeltaV1` inside receipts; field-digester turns receipts into a
  `FieldStateV1` every 2s (`substrate_field_state`, 72h). `EpisodeSummaryV1` rolls receipts into
  15-minute windows (`substrate_episode_summaries`, 14d).
- `TemporalHopV1` (`orion/schemas/grammar.py:129`) and `grammar_temporal_hops` exist with **zero
  producers**. This is an unused hook for cross-trace temporal links.
- Retired and blacklisted from this design: `DriveStateV1` (producer-less since 2026-07-30),
  DriveEngine (deleted, PR #1486), `AutonomyStateV2` (reducer retired 2026-07-16;
  `orion/autonomy/state_store.py` has no callers), `SelfStateV1` (producer deleted, PR #1266), and
  orion-world-model (`model_untrained=True` hard-coded, `services/orion-world-model/app/main.py:231`).

### Consumers as they stand

- **Stance, unified turn:** `build_stance_react_context`
  (`services/orion-thought/app/bus_listener.py:127`) adds association, repair bundle, coalition
  projection, optional `mind_coloring`. Situation data is built later and reaches only the harness
  prefix. The stance step gets no time-of-day and no day context at all.
- **Stance, chat brief:** `build_chat_stance_inputs`
  (`services/orion-cortex-exec/app/chat_stance.py:2506`) already carries `situation`
  (conversation phase and time) and `continuity_digest`. Any new ctx key needs an entry in
  `CONTEXT_PROVENANCE_REGISTRY` (`orion/schemas/context_provenance.py`).
- **System One:** `SystemOneInputStateV1` is `extra="forbid"` and reads only the broadcast
  projection and a fresh field frame (`orion/substrate/system_one_appraisal.py:116-186`). Only
  `curiosity_pull` is behavioral. New inputs go through the metric gate and a
  `QUESTION_SET_ID` bump.
- **Curiosity:** `build_kickoff_prompt` (`orion/curiosity/kickoff_prompt.py:954`); the thread
  section (`:93`) is documented as "stated as fact, no steering". Daily state is a Redis count
  and a cooldown; nothing records time spent per subject.
- **Reverie:** `build_reverie_context` (`services/orion-thought/app/reverie.py:250`) takes the
  live broadcast projection. No time-of-day, no day history. The 2026-07-14 narration-continuity
  design records that ticks have no memory of prior ticks; not implemented.
- **PCR continuity:** `chat.continuity.v1` is a 120-minute, user-only window
  (`services/orion-cortex-exec/app/pcr_chat_memory.py:177`). Prior-day context does not reach a
  new session through it.
- **Crystallizer / Graphiti:** consolidation windows crystallize with `chat_turn` and
  `grammar_event` evidence; the Graphiti adapter's payload carries no event time
  (`services/orion-graphiti-adapter/app/falkordb.py:35`); `GRAPHITI_ENABLED` defaults false.

### The gap, stated once

Every row above has a timestamp. No row says which arc it belonged to, whether Orion had been
there before that day, how long Orion stayed, what interrupted it, what the body was doing, or what
was refused meanwhile. Consumers that need continuity (stance, curiosity, reverie, daily metacog,
next-day recall) each reconstruct it from a rolling window of chat, or not at all.

## What each of the six named subsystems contributes

**Dreams.** A sleep is an arc of its own kind: `dream_cycle.started_at` to `ended_at`, with a
covered interval `[cycle_json.pressure.since, cycle_json.pressure.computed_at]` that says which
stretch of the chronology it replayed. Each hypothesis is an expectation with a commit time, an
expiry, an offer time, and a later adoption or refutation through its `:Prior`. The chronology
should show "slept 03:10 to 03:14 on the preceding 9 hours; proposed 4 links; 1 tested by 10:42".
Later (patch 7), dream replay should draw its candidates from closed arcs rather than four raw
tables. Excluded: the situation brief's "reverie/dream threads" and Hub's "daydream" block, which
are reverie; the legacy `dreams` rows contribute only `created_at`.

**The metacog table.** Degraded and critical `orion_metacog` rows, joined to `metacog_trigger`
on `correlation_id`, are self-observation events with a `trigger_kind`. They are not arcs and
they are not verdicts. The chronology binds them to whatever arc was open when they fired, so
"three degraded transport observations during the 14:00 curiosity arc" becomes sayable. Ticks
are excluded (degenerate). `daily_metacog_v1` becomes a consumer: its grounding is the closed
day frame plus the rows it cites, which also fixes its mislabelled window.

**AI Town.** A town conversation is a social arc: `aitown_chat_history_log` rows grouped by
`session_id`, bounded by first and last `created_at`, partner identified by slug via
`orion/town_cast.py`, with `participant_kind=human` treated as Juniper. Only rows with
`source=orion-embodiment` are Orion's lived experience. `source=orion-ai-town` rows are town
events Orion did not witness and are excluded from the self chronology entirely. The frame can
then say "in town 14:02 to 14:20 with Mara; last saw Mara yesterday". The NPC-to-NPC leak into
social state is recorded as a follow-up below and not fixed here.

**Substrate grammar.** Not consumed raw: 1.4M events a week at 3-day retention is the wrong
altitude and the wrong lifetime. The reducer reads the layers grammar already feeds:
`substrate_episode_summaries` for activity level per quarter hour, `substrate_system_one_appraisal`
for the 30-second `curiosity_pull` and `deliberation_need` readings, and the broadcast log. Two
grammar facts do matter: `observed_at` is occurrence time and `grammar_traces.created_at` is write
time, so any join uses the former; and `TemporalHopV1` is the natural place to publish an arc
transition to the Atlas later, without inventing a new grammar kind.

**Visual diffusion reverie.** Each run is an imagery event with a caption and a context slot
(`context_slot_used` says whether it drew on text reverie, self-study, or a crystallization).
Each deferral is a constraint event with a named reason and a body cause (`thermal_gate` state,
cabinet temperature at refusal). The chronology can therefore say "wanted to render at 15:31,
refused, cabinet at 32.4 °C, rendered at 16:10 after re-arm". Start time comes from
`reverie_visual_attempt.started_at` when dispatched, else `created_at − held_sec`.

**The biometrics cabinet.** Not folded at 30-second resolution. Each arc gets one embodied
summary computed from `orion_biometrics_summary` rows inside its interval: mean and max
`strain`, cabinet temperature range, `chassis_watts` mean, thermal gate state at start and end,
and any `cabinet_ambient_spike` or cooling switch change. `power_intent_settled` rows for
`reverie_diffusion` inside the arc give the energy spent. No feeling is asserted; the field is
called `body`, and every number in it traces to a producing line named in the metric gate below.

## Missing questions

These are Juniper's calls. Each has a recommended default so patch 1 can start without waiting.

1. **Host.** Pure reducer in `orion/temporal_self/`, ticked by (a) `orion-consolidation-runtime`,
   (b) a new lane in `orion-substrate-runtime`, or (c) `orion-actions`. Recommended: (a). It is
   small (about 1,500 lines across service and `orion/consolidation/`), already windowed and
   deterministic (`stable_consolidation_frame_id`, skip-if-exists, `ON CONFLICT DO NOTHING`),
   has `/health` and `/latest`, and its README already calls itself "pattern observation, not
   learning". Substrate-runtime's worker is already past 3,900 lines. Note that the consolidation
   runtime still lists `substrate_self_state` as an input, whose producer was deleted; that is a
   separate cleanup.
2. **Day boundary.** Local midnight in `ORION_SITUATION_TIMEZONE`, or sleep-to-sleep? Recommended:
   `day_id` is the local calendar date, computed by one shared helper that every consumer can
   import; sleeps are events inside days; an arc open at midnight is split, and the continuation
   carries `carried_from_previous_day=true`.
3. **Juniper in town.** Are Orion's exchanges with Juniper inside AI Town Juniper conversation or
   town? Recommended: Juniper conversation, same privacy class as `chat_history_log`, never
   fanned out to peers.
4. **Raw deltas.** Fold `StateDeltaV1` receipts on a sub-30-minute tick, or only the 15-minute
   episode rollups? Recommended: rollups only in v1. If a consumer later needs channel-level
   before/after inside an arc, that is a new source with its own gate.
5. **Retention.** Recommended: events 30 days, arcs 90 days, closed days 365 days, singleton
   frame forever. Closed days are small (one row per day).
6. **Atlas publication.** Should arc transitions be published as `TemporalHopV1` grammar events so
   the Hub Atlas can draw them? Recommended: not in v1; revisit after patch 4 when the arc
   precision eval has a number.
7. **Subject label source.** Arcs need a human-readable label for the stance cue. Recommended:
   reuse the source row's own label (`attended_label`, loop `target_text`, prior text, partner
   name), clipped with `orion.schemas.attention_schema.clip`, never generated.

## Proposed schema / API changes

### Capability, data, privacy, proof, danger, rollback (proposal-mode fields)

- **Capability that changes.** Orion gains a queryable, inspectable account of its own day:
  which subjects it returned to, for how long, in what order, what it slept on, what it expected
  and what came of it, what it was refused, and what its body was doing meanwhile. Consumers can
  read a bounded slice of that instead of a rolling window of chat.
- **Data touched.** Read-only over the tables listed in the binding appendix. Writes go only to
  four new `temporal_self_*` tables. No source table is modified. No FalkorDB writes in v1.
- **Privacy boundary.** The event table stores references and bounded labels, never raw
  utterances: the chat grammar lane's own `payload_ref` discipline
  (`services/orion-hub/scripts/grammar_emit.py:68-216`) is the model. Town rows with a human
  partner are Juniper's and inherit `chat_history_log`'s access class. Nothing here reaches
  contractor peers; the frame is not published on any bus channel in v1.
- **Trace that proves it worked.** One `temporal_self_arc` row whose `evidence_event_ids` resolve
  to real rows in at least two different source tables, with `attention_returns ≥ 2`, and one
  consumer read of the frame that cites the arc id (the daily metacog output, or a stance cue
  eval). A frame that reads "no active arc" on a quiet stretch is also required evidence: the
  instrument must be able to rest.
- **Dangerous failure modes.**
  1. *Fabricated continuity:* loose subject matching stitches unrelated events into one arc, and
     Orion then tells Juniper "I have been on this all day" falsely. Mitigation: identity by exact
     refs only; the arc-precision eval below; every arc carries its evidence ids.
  2. *Town chatter as lived day:* the 2026-08-14 crystallization incident (610 of 621 proposals
     were NPC dialogue, `orion/memory/crystallization/formation_policy.py:17-22`) repeated in the
     chronology. Mitigation: source filter as a schema invariant plus a test fixture with mixed
     `social_room_turns` producers.
  3. *Restart duplication:* a restart re-folds the same window into new arc ids. Mitigation:
     deterministic ids from `(day_id, subject_ref, first_event_id)`; `ON CONFLICT DO NOTHING`;
     the replay-identity acceptance check.
  4. *Consumer over-steering:* a stance cue that reads like an instruction. Mitigation: the cue
     stays inside the SOURCES boundary that `fix/stance-reading-boundary` introduced
     (`docs/superpowers/pr-reports/2026-09-26-stance-reading-boundary-pr.md`) and must pass
     `orion/thought/evals/stance_task_boundary.py` unchanged.
  5. *Wrong day:* the `daily_metacog` mislabel shows this is a live failure class. Mitigation: one
     shared `day_id` helper with a test that pins midnight in `America/Denver`, and the
     TEXT/naive timestamp columns (`orion_metacog.timestamp`, `orion_biometrics_summary.timestamp`,
     `aitown_chat_history_log.created_at`) each get an explicit cast documented in `sources.py`.
  6. *Load:* reading `substrate_field_state` at 0.4 rows a second or `grammar_events` at all.
     Mitigation: neither is a source. The heaviest source is `goal_provenance_streak_ticks` at
     about one row per two seconds, read by keyset cursor in bounded batches.
- **Disable and roll back.** `TEMPORAL_SELF_ENABLED=false` stops the tick. Each consumer has its
  own flag (`TEMPORAL_SELF_STANCE_CUE_ENABLED`, `TEMPORAL_SELF_DAILY_METACOG_GROUNDING_ENABLED`,
  `TEMPORAL_SELF_CURIOSITY_THREAD_ENABLED`), default false, and disabling one restores that
  consumer's baseline path exactly. The four tables have no foreign keys and can be dropped.
  All flags land in the service `.env_example` and settings, with the local `.env` synced by
  `python scripts/sync_local_env_from_example.py` in the same patch.

### Schemas (`orion/schemas/temporal_self.py`, new)

All models `extra="forbid"`, registered in `_REGISTRY` in `orion/schemas/registry.py`. None is a
bus payload in v1, so `SCHEMA_REGISTRY` and `orion/bus/channels.yaml` are untouched; the CI test
`tests/test_agent_trace_schema_registry.py::test_registry_and_schema_registry_agree` therefore
still passes.

```python
TemporalSelfEventV1
  schema_version: "temporal_self.event.v1"
  event_id: str            # deterministic: f"{source_kind}:{source_ref}"
  day_id: str              # local calendar date from the shared helper
  occurred_at: datetime    # tz-aware UTC; the source's occurrence column, cast as documented
  source_kind: Literal[
    "attention_schema", "broadcast_tick", "dominance_tick", "self_model_tick",
    "reverie_thought", "reverie_chain", "visual_run", "visual_deferral",
    "curiosity_run", "prior_revision", "dream_cycle", "dream_hypothesis",
    "town_exchange", "metacog_observation", "episode_summary", "organ_transition",
    "expectation_verdict", "action_outcome", "peer_ask", "consolidation_window_close",
  ]
  source_table: str
  source_ref: str          # the row's own primary key, verbatim
  correlation_id: str | None
  subject_refs: list[str]  # exact refs only: node ids, loop source_refs, prior ids, partner slugs
  label: str               # ≤ 300 chars, clipped from the source row's own label; never generated
  duration_sec: float | None
  payload: dict[str, Any]  # bounded, source-specific; documented per source_kind in sources.py

TemporalSelfArcV1
  schema_version: "temporal_self.arc.v1"
  arc_id: str              # deterministic: sha256(day_id, subject_ref, first_event_id)[:16]
  day_id: str
  kind: Literal["attention", "curiosity", "conversation", "town", "reverie", "imagery", "sleep"]
  subject_ref: str
  subject_label: str
  began_at: datetime
  ended_at: datetime | None
  status: Literal["open", "suspended", "closed"]
  closed_reason: Literal["winner_changed", "process_ended", "day_boundary", "return_window_expired"] | None
  attention_returns: int   # number of resumes after a suspension, same day
  cumulative_dwell_sec: float
  interruptions: list[str] # arc_ids of arcs that suspended this one
  carried_from_previous_day: bool
  carried_from_arc_id: str | None
  evidence_event_ids: list[str]  # capped at 256; overflow counted in evidence_overflow
  evidence_overflow: int
  expectations: list[ArcExpectationRefV1]   # event ids committed during the arc + their verdict event ids
  self_changes: list[str]                   # prior_revision / action_outcome / expectation_verdict event ids
  constraints: list[str]                    # visual_deferral / thermal / allocator-refusal event ids
  body: ArcBodySummaryV1 | None
  reducer_version: str

ArcBodySummaryV1
  sample_count: int
  strain_mean: float | None
  strain_max: float | None
  cabinet_temp_c_min: float | None
  cabinet_temp_c_max: float | None
  chassis_watts_mean: float | None
  thermal_state_start: Literal["cool", "hot", "degraded", "unknown"]
  thermal_state_end: Literal["cool", "hot", "degraded", "unknown"]
  diffusion_energy_joules: float | None
  ambient_spike_count: int
  cooling_switch_changes: int

TemporalSelfFrameV1            # the singleton projection, upserted each tick
  schema_version: "temporal_self.frame.v1"
  frame_id: str                # deterministic per (day_id, tick window)
  day_id: str
  as_of: datetime
  day_phase: str               # TimeContextV1.day_phase vocabulary, computed by the same helper
  active_arc: TemporalSelfArcV1| None
  previous_arc: TemporalSelfArcV1 | None
  arcs_today: list[TemporalSelfArcV1]      # closed + suspended, bounded to 64
  open_threads: list[OpenThreadV1]         # subject_ref, first_seen_today, last_returned, returns_today, carried_from_previous_day
  expectations_pending: list[str]          # event ids
  expectations_resolved_today: dict[Literal["confirmed","disconfirmed","unscored","met","missed","unscorable"], list[str]]
  self_changes_today: list[str]
  constraints_today: list[str]
  sleeps_today: list[str]                  # dream_cycle event ids
  entered_day_with: list[str]              # arc ids carried from the previous closed day
  source_cursors: dict[str, str]           # source_kind -> last (ts, id) folded; the inspectability surface
  warnings: list[str]                      # e.g. "biometrics source stale 900s", "town source unreadable"

TemporalSelfStanceCueV1        # the bounded stance projection
  as_of: datetime
  day_phase: str
  active_subject_label: str | None
  active_arc_age_sec: float | None
  active_returns_today: int
  came_from_label: str | None
  came_from_minutes_ago: float | None
  open_thread_count: int
  oldest_open_thread_age_sec: float | None
  last_sleep_ended_minutes_ago: float | None
  rendered: str                # ≤ 400 chars, coarse phrasing in the recent_attention_cue style
```

`OpenThreadV1` is derived, not stored: it is the per-subject view over today's arcs plus the
previous closed day's open subjects.

### The reducer contract (`orion/temporal_self/`, new, I/O-free)

```python
fold_events(state: TemporalSelfStateV1, events: list[TemporalSelfEventV1], now: datetime) -> TemporalSelfStateV1
build_frame(state: TemporalSelfStateV1, now: datetime) -> TemporalSelfFrameV1
project_stance_cue(frame: TemporalSelfFrameV1, now: datetime) -> TemporalSelfStanceCueV1
project_curiosity_thread(frame, prior_ids: list[str]) -> CuriosityTemporalFactsV1
project_reverie_recent(frame, limit: int) -> list[TemporalSelfArcV1]
close_day(state, day_id) -> TemporalSelfDayV1
```

Pure, deterministic, idempotent, replayable. Folding the same events twice yields the same state.
Events arrive ordered by `(occurred_at, event_id)`; late events (a verdict scored after the arc
closed) attach by event id, never by time, and never reopen a closed arc.

**Arc rules, stated so a test can pin them:**

1. An *attention arc* opens when the same `subject_ref` wins `K` consecutive `broadcast_tick`
   events (`K` = `TEMPORAL_SELF_ARC_MIN_TICKS`, default 3, chosen to mirror
   `ORION_GOAL_PROVENANCE_MIN_STREAK`; patch 1 measures the real streak distribution before this
   is frozen).
2. A *process arc* (curiosity, town, reverie, imagery, sleep) opens at its process's own start
   event and closes at its end event. The process's own identity is its subject ref. It does not
   depend on rule 1.
3. An arc *suspends* when a different subject wins rule 1, or when a process arc of another kind
   opens on the same lane. The suspending arc is appended to `interruptions`.
4. A suspended arc *resumes* if its subject wins again within `R` minutes
   (`TEMPORAL_SELF_RETURN_WINDOW_MIN`, default 180). `attention_returns` increments. Dwell is
   summed across segments.
5. A suspended arc *closes* with `return_window_expired` after `R` minutes without a resume, or
   with `day_boundary` at local midnight, in which case a continuation arc opens with
   `carried_from_arc_id` set if the subject wins again before `R` minutes elapse into the new day.
6. Loop identity is `source_refs[0]` of the open loop, not the loop id. Field target identity is
   `target_id`. Curiosity identity is the run's prior ids. Town identity is the partner slug.
   Reverie identity is the coalition's node ids. Sleep identity is the cycle id.
7. Events with no subject (a metacog observation, an organ transition, an episode summary) bind to
   whichever arc is open at their `occurred_at`, or to the day if none is. They never open or
   close an arc.

### Persistence (`services/orion-sql-db/manual_migration_temporal_self_v1.sql`, new)

Follows the house pattern: hand-applied, `IF NOT EXISTS`, header naming the single writer and the
`psql` command, checked with `scripts/check_sql_migrations_applied.py --file`.

```sql
temporal_self_event   (event_id text pk, day_id text, occurred_at timestamptz, source_kind text,
                       source_table text, source_ref text, correlation_id text, subject_refs text[],
                       label text, duration_sec double precision, payload_json jsonb,
                       ingested_at timestamptz default now())
                       index (day_id, occurred_at); index (occurred_at, event_id)
temporal_self_arc     (arc_id text pk, day_id text, kind text, subject_ref text, began_at timestamptz,
                       ended_at timestamptz, status text, arc_json jsonb, updated_at timestamptz)
                       index (day_id, began_at); index (subject_ref, day_id)
temporal_self_day     (day_id text pk, closed_at timestamptz, frame_json jsonb)     -- one row per closed day
temporal_self_projection (projection_id text pk, generated_at timestamptz, projection_json jsonb,
                       created_at timestamptz)  -- singleton "current_day", upsert, same shape as substrate _save_projection
temporal_self_cursor  (source_kind text pk, last_occurred_at timestamptz, last_source_ref text, updated_at timestamptz)
```

One transaction per tick writes new events, changed arcs, the cursor rows, and the singleton, in
that order, the way `commit_digest_tick` does in field-digester
(`services/orion-field-digester/app/store.py:414`). Retention per the recommended defaults in
Missing question 5, applied by the same worker at boot and daily.

### Routes (`services/orion-consolidation-runtime/app/main.py`)

- `GET /temporal-self/frame` — the singleton, verbatim.
- `GET /temporal-self/day/{day_id}` — a closed day, or 404.
- `GET /temporal-self/arcs?day_id=` — arcs for a day.
- `GET /temporal-self/cursors` — the per-source cursors and their lag in seconds (the reducer
  health surface; a source whose lag exceeds its own cadence by 10× is reported in `warnings`).

Hub gets a read-only tab later (`services/orion-hub/scripts/temporal_self_routes.py`, following
the Dream tab, commit `a083beb`), not in the first three patches.

### Env keys (service `.env_example` and `settings.py`, synced to local `.env`)

```
TEMPORAL_SELF_ENABLED=false
TEMPORAL_SELF_POLL_INTERVAL_SEC=60
TEMPORAL_SELF_ARC_MIN_TICKS=3
TEMPORAL_SELF_RETURN_WINDOW_MIN=180
TEMPORAL_SELF_EVENT_RETENTION_DAYS=30
TEMPORAL_SELF_ARC_RETENTION_DAYS=90
TEMPORAL_SELF_DAY_RETENTION_DAYS=365
ORION_SITUATION_TIMEZONE=America/Denver   # already exists for TimeContextV1; reused, not duplicated
```

Consumer flags live in the consuming service's `.env_example` when each consumer patch lands.

### Metric quality gate for the frame's numeric outputs

Run for `active_arc_age_sec`, `attention_returns`, `cumulative_dwell_sec`, `open_thread` ages,
`constraints_today` count, and the `body` summary. Recorded here per CLAUDE.md 0A; steps 4 and 6
are re-run in patch 1 with live data and the results appended to this document.

1. **Provenance.** Every number is a count or a difference over the source rows' own timestamps:
   `substrate_attention_broadcast_log.generated_at`, `substrate_attention_schema.generated_at`,
   `goal_provenance_streak_ticks.observed_at`, `substrate_reverie_thought.created_at` and
   `expectation_scored_at`, `dream_cycle.started_at`/`ended_at`, `dream_hypothesis.created_at`,
   `reverie_visual_attempt.started_at`, `aitown_chat_history_log.created_at`,
   `orion_biometrics_summary.timestamp` (cast from TEXT), `orion_biometrics_summary.composites.strain`
   (`biometrics_pipeline.py:481-486`). The reducer adds nothing that is not a timestamp arithmetic
   over those.
2. **Independence.** `dwell_ticks` on the broadcast projection is the nearest existing signal;
   within one uninterrupted coalition, `cumulative_dwell_sec` is a monotone transform of it and is
   therefore **redundant there**. Across suspensions and restarts it is not, because `dwell_ticks`
   resets and does not sum. `attention_returns` has no existing equivalent (`coalition_history`
   carries no ids and is process memory). The `body` summary is a windowed mean of an existing
   composite, so it is not a new sensor; it is the same sensor at arc altitude.
3. **Theory anchor.** Arc segmentation: Event Segmentation Theory (Zacks, Speer, Swallow, Braver
   & Reynolds 2007), boundaries where the attended subject changes. Arc hierarchy: the
   Self-Memory System's "general events" tier between lifetime periods and event-specific
   knowledge (Conway & Pleydell-Pearce 2000). Return counts to unresolved subjects: current
   concerns (Klinger 1975) and the persistence of interrupted tasks (Zeigarnik 1927). The
   binding itself: autonoetic consciousness (Tulving 1985, 2002), the capacity to locate a
   remembered event in one's own past. None of these licenses a feeling claim; they license the
   *shape* of the record.
4. **Live-data sanity.** `UNVERIFIED`. Patch 1's replay script must show, over seven days of the
   log tables: the distribution of attention streak lengths (so `K` is measured, not guessed); the
   distribution of returns per subject per day, including that zero is common; stretches with no
   active arc (rest state reachable); arcs that span restarts of substrate-runtime without
   duplication; and that `strain_mean` per arc varies and can sit near its floor.
5. **Existing mechanism.** Searched: `EpisodeSummaryV1` (counts per 15 minutes, no subject),
   `attention_salience_trace` (per loop score, no arcs), curiosity `RecentRun.written_at` (per run,
   one process), dbt `dim_reverie_dates` (UTC day counts for reverie only), `spark_state_rollups`
   (affect buckets, live status unknown), `substrate_coalition_dwell_log` (24h, current coalition
   only). None binds across processes or across a day.
6. **Reversibility.** Nothing is baked into a schema or training default. Four tables with no
   foreign keys; flags default off; consumer wiring is additive ctx keys. Removal is a migration
   drop and a flag flip.

## Files likely to touch

**Patch 1 (read-only replay, no runtime change):**

- `scripts/analysis/measure_temporal_self_replay.py` (new): reads the source tables over a date
  range, runs the pure reducer, prints the distributions named in gate step 4, and writes a
  fixture bundle to `orion/temporal_self/evals/fixtures/`.
- `docs/superpowers/specs/2026-09-26-temporal-self-design.md` (this file): gate findings appended.

**Patch 2 (schemas and reducer, still no service):**

- `orion/schemas/temporal_self.py` (new).
- `orion/temporal_self/__init__.py`, `day.py` (the shared `day_id` helper), `sources.py` (one
  pure adapter per `source_kind`, each documenting its timestamp cast), `arcs.py`, `frame.py`,
  `projections.py` (new).
- `orion/schemas/registry.py`: `_REGISTRY` entries.
- `orion/temporal_self/tests/test_arcs.py`, `test_sources.py`, `test_day_boundary.py`,
  `test_replay_identity.py` (new).
- `orion/temporal_self/evals/run_arc_precision_eval.py` and fixtures (new).

**Patch 3 (worker, migration, routes):**

- `services/orion-consolidation-runtime/app/temporal_self_worker.py`, `temporal_self_store.py`
  (new); `settings.py`, `main.py`, `.env_example`, `docker-compose.yml`, `README.md`.
- `services/orion-sql-db/manual_migration_temporal_self_v1.sql` (new).
- `services/orion-consolidation-runtime/tests/test_temporal_self_worker.py`,
  `test_temporal_self_store.py` (new).
- `scripts/smoke_temporal_self.py` (new): asserts a frame, cursors advancing, and a day close.
- `.github/workflows/temporal-self-tests.yml` (new, path-filtered, after the System One workflow).

**Patch 4 (first consumers):**

- `services/orion-actions/app/main.py` (`daily_metacog_v1` grounding from the closed day, behind
  a flag), `orion/cognition/prompts/daily_metacog_prompt.j2`.
- `orion/temporal_self/stance_cue.py` (new, in the `recent_attention_cue` style);
  `services/orion-cortex-exec/app/chat_stance.py` (`inputs["temporal_self"]`);
  `orion/schemas/context_provenance.py` (registry entry, kind `live_runtime_projection`);
  `orion/cognition/prompts/chat_stance_brief.j2`; `orion/thought/evals/stance_task_boundary.py`
  re-run unchanged.

**Later patches:** `orion/curiosity/kickoff_prompt.py` (`_thread_section` fact line),
`services/orion-thought/app/reverie.py` (recent closed arcs), `orion/substrate/system_one_appraisal.py`
(numeric inputs, own gate, `QUESTION_SET_ID` bump), `services/orion-dream/app/replay.py`
(candidates from closed arcs), `services/orion-hub/scripts/temporal_self_routes.py` and a tab.

## Non-goals

- No sentience claim, no "continuity score", no feeling asserted from the `body` field.
- No new service, no new bus channel, no LLM call inside the reducer, no generated narrative
  stored as fact.
- No writes to any source table; no FalkorDB or Graphiti writes in v1. Ticks are never
  crystallized; at most one closed-day summary per day is a candidate for later crystallization.
- No resurrection of `DriveStateV1`, `AutonomyStateV2`, `SelfStateV1`, or the world model.
- No replacement of `EpisodeSummaryV1`, `ConsolidationFrameV1`, the attention schema, or memory
  consolidation windows. Each remains authoritative for what it already records.
- No NPC-to-NPC town turns in the chronology, and no fix here for their leak into social memory.
- No 30-second body rows in the frame; no `metacognition_ticks`; no `substrate_field_state`
  reads; no `grammar_events` reads.
- No opportunity-cost estimate. The constraints channel records what the system already stamped
  as refused or deferred, with the stamping row's own reason.
- No change to any spend point, budget, cap, or gate. This design informs; it does not allocate.

## Acceptance checks

1. **Replay identity.** Fold seven fixture days twice from an empty state and from a mid-day
   checkpoint; arcs, ids, and frames are byte-identical. A simulated substrate-runtime restart
   (broadcast log gap, `dwell_ticks` reset) produces no duplicate arc.
2. **Arc precision.** On a hand-labelled fixture day (built from real rows in patch 1), every arc's
   evidence events share the arc's `subject_ref` exactly; a mixed-producer `social_room_turns`
   fixture yields zero arcs from `source=orion-ai-town` rows; a label change on an open loop
   does not split its arc when `source_refs` is unchanged.
3. **Rest state.** A quiet fixture window (attention schema `attended_id=None`, no process events)
   yields `active_arc=None`, `attention_returns=0` for every subject, and a frame that still
   validates and still advances cursors.
4. **Day boundary.** An arc open at 23:58 local closes with `day_boundary` and its continuation
   carries `carried_from_arc_id`; `day_id` agrees with `TimeContextV1.local_date` for the same
   instant; a naive `aitown_chat_history_log.created_at` and a TEXT `orion_metacog.timestamp` land
   in the correct local day in a fixture straddling midnight.
5. **Late evidence.** A reverie verdict scored after its arc closed attaches to that arc by event
   id and does not reopen it; a dream hypothesis adopted as a prior two days later attaches to the
   sleep event of the day it was made.
6. **Live proof (patch 3).** One real `temporal_self_arc` row with `attention_returns ≥ 2` whose
   evidence ids resolve in two source tables; `GET /temporal-self/cursors` shows every enabled
   source's lag under 10× its cadence; one real closed day in `temporal_self_day`. Until
   collected, the live path is `UNVERIFIED` and the PR says so.
7. **Consumer effect (patch 4).** The daily metacog output cites at least one arc id and its
   evidence window equals the closed day, not a rolling 24 hours; the stance boundary eval passes
   unchanged with the cue present; an ablation with the cue removed changes no stance decision
   that the eval scores (the cue is advisory).
8. **Metric gate on live data (patch 1 exit).** The distributions in gate step 4 are recorded in
   this document, and `K` and `R` are set from them.
9. **Operational.** Retention runs; the migration passes `check_sql_migrations_applied.py`;
   `scripts/check_substrate_projection_schema_drift.py`'s lesson is honoured (the singleton row is
   read with a tolerant loader so a stale stored row cannot crash-loop the worker); the smoke
   script runs against the Tailscale bus URL only; local `.env` is synced.

## Recommended next patch

| Order | Patch | Exit evidence |
|---|---|---|
| 1 | Read-only replay over seven live days; gate steps 4 and 6; fixture bundle | Distributions recorded here; `K`, `R` chosen; no runtime change |
| 2 | Schemas, pure reducer, tests, arc-precision eval | Tests and eval green on fixtures; replay identity holds |
| 3 | Worker in consolidation-runtime, migration, routes, flag default off, smoke | One real arc with returns ≥ 2; one closed day; cursors advancing |
| 4 | Daily metacog grounded on the closed day; stance cue behind a flag | Metacog output cites arc ids; stance boundary eval unchanged |
| 5 | Curiosity thread fact ("N minutes on this today"); reverie recent arcs | Kickoff prompt shows the fact from real rows; no steering language |
| 6 | System One numeric inputs, own metric gate, `QUESTION_SET_ID` bump | Gate record; no-consumers test still guards the other questions |
| 7 | Dream replay from closed arcs; Hub tab; optional `TemporalHopV1` publication | Dream cycle cites arc ids; tab renders a real day |

Start with patch 1. Its job is to find out whether the arcs this document describes exist in
Orion's real rows, and how long they are, before a single table is created.

## Appendix: source binding table

Every source the v1 reducer reads, the column it treats as occurrence time, and the ref it uses
as subject identity. A source not in this table is not read.

| `source_kind` | Table | Occurrence time | Subject ref | Notes |
|---|---|---|---|---|
| `attention_schema` | `substrate_attention_schema` | `generated_at` | `attended_id` | five lanes; `process` kept in payload; `predicted_next` kept, not scored |
| `broadcast_tick` | `substrate_attention_broadcast_log` | `generated_at` | selected loop's `source_refs[0]`, else `attended_node_ids` | drives rule 1; `dwell_ticks` ignored |
| `dominance_tick` | `goal_provenance_streak_ticks` | `observed_at` | `target_id` | `streak_count`, `qualified` in payload; 14d retention bounds backfill |
| `self_model_tick` | `substrate_attention_self_model` | `generated_at` | none | `predicted_shift`, `prediction_error_by_domain`, heartbeat fields |
| `reverie_thought` | `substrate_reverie_thought` | `created_at` | coalition node ids from `thought_json` | expectation fields carried |
| `expectation_verdict` | `substrate_reverie_thought` | `expectation_scored_at` | same thought | one event per scored thought; also `vision_percept_expectation` |
| `reverie_chain` | `substrate_reverie_chain` | chain start/end | chain id | terminal reason in payload |
| `visual_run` | `reverie_visual_chain` + `reverie_visual_attempt` | `attempt.started_at`, else `created_at − held_sec` | chain id | caption, `context_slot_used`, `thermal_gate` |
| `visual_deferral` | `reverie_visual_attempt` | `started_at` | need id | outcome ∈ deferred_* / failed; the constraints channel |
| `curiosity_run` | `curiosity_run_outcomes` + `substrate_attention_schema` (`process=curiosity`) | `turn_started_at` | offered prior ids | spend-log migration live status `UNVERIFIED` |
| `prior_revision` | FalkorDB `orion_worldview` `:PriorRevision`, `HopRecord.written_at` | `written_at` (None before 2026-09-19 → attach by run) | prior id | read through `agency_episode_reader`-style bounded Cypher |
| `dream_cycle` | `dream_cycle` | `started_at`/`ended_at` | cycle id | covered interval from `cycle_json.pressure` |
| `dream_hypothesis` | `dream_hypothesis` | `created_at`; `offered_at`; `expires_at` | hypothesis id | adoption via `:Prior.formed_from` |
| `town_exchange` | `aitown_chat_history_log` (`source=orion-embodiment` only) | `created_at` (naive → UTC, `UNVERIFIED` DB tz) | partner slug via `orion/town_cast.py` | `participant_kind=human` ⇒ Juniper class |
| `metacog_observation` | `orion_metacog` ⋈ `metacog_trigger` | `metacog_trigger.timestamp` (trigger time, not draft time) | none | severity ∈ degraded/critical only; `trigger_kind` in payload |
| `episode_summary` | `substrate_episode_summaries` | window start | none | counts only |
| `organ_transition` | `equilibrium_service_transitions` | transition time | none | organs participating |
| `action_outcome` | `substrate_action_outcomes`, `substrate_action_effect_posterior` | outcome time | action/proposal id | posteriors as self-change |
| `peer_ask` | FalkorDB `PeerAskCommit`/`PeerBriefOffer`/`PeerBriefDecision` | `committed_at` etc. (epoch ms) | `help_id` | only when `CURIOSITY_PEER_EPISODES_ENABLED`; absent is fine |
| `consolidation_window_close` | `memory_consolidation_windows` | close time | `source_platform` | one input to conversation arcs, not the authority |
| body summary | `orion_biometrics_summary`, `orion_biometrics_cluster`, `power_intent_settled`, `cabinet_ambient_spike`, `home_cooling_sample` | per arc interval | none | aggregated per arc, never per row |

## Follow-ups recommended outside this design (not filed; no issue tracker write was made)

- The 12h–48h conversation-phase gap in `orion/situational/context.py:880-903`.
- The `daily_metacog_v1` window mismatch (rolling 1440 minutes labelled as yesterday).
- NPC-to-NPC `social_room_turns` updating Orion's own peer and stance rows in
  `services/orion-social-memory`.
- `scripts/check_schema_registry.py` and `scripts/check_bus_channels.py` are named in
  `CLAUDE.md` sections 11 and 17 but do not exist; the real gates are the registry-agreement test
  and per-contract catalog tests.
