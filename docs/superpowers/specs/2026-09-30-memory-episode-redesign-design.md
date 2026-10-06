# How Orion forms memories: episodes, Orion's own words, knowing whose words they are, and closing the loop

Status: **APPROVED (Juniper, 2026-10-01). Stage 0 in progress.** Revision 3 records her decisions on the revision-2 questions; see "Decisions". One question is still open: the reflection rows.
Date: 2026-09-30 (rev 1-2), 2026-10-01 (rev 3)
Evidence base: read-only live queries on the `conjourney` Postgres, FalkorDB `GRAPH.RO_QUERY`, `ssh circe@circe nvidia-smi`, and docker logs/inspect/env, all taken 2026-09-30 between 09:30 and 10:30 UTC. Code read against main @ a005658db.

Related specs:
- `2026-07-07-purpose-conditioned-recall-design.md`: PCR, the retrieval spine this design keeps.
- `2026-07-07-consolidation-crystallization-gate-design.md`: the write-side gate this design replaces.
- `2026-07-06-graphiti-rail-activation-design.md`
- `2026-08-14-crystallization-queue-auto-gate-analysis.md`
- `2026-09-29-recall-semantic-retrieval-pipeline-design.md` (#2413, "recall by referent, not resemblance")
- `2026-09-29-recall-retrieval-query-architecture-design.md`

Anything not checked live is marked **UNVERIFIED**.

---

## What changed in revision 2

1. **A false claim is withdrawn.** Revision 1 said the chat compactor made up "as an introvert". It did not. Juniper's 09-28 09:06 message ends "super draining for me--I'm an introvert :)". My query had cut the prompt at 160 characters (`left(prompt,160)`), and the message is 188 characters long, so I never saw the end of it. The crystallization row, the digest, and `bus_fallback_log` all hold the full text. The compactor was faithful, and the claim is removed as the voice-rule regression case. The pronoun defect is real and stays: the digest says "He then pivoted…" about Orion (journal `68bb8201`). The lesson is written into the evals: **never judge fidelity against truncated text.**
2. **The episode boundary reuses what already exists.** No new idle timer. Orion already has a conversation wall clock and an LLM boundary run. Both are traced below, and each has a specific defect that is fixed rather than replaced. The Austin example is re-run against the real boundaries.
3. **PCR, Graphiti and pageindex are kept and fixed, not retired.** Each section now covers the original intent, what is actually broken, and a concrete role. Also, PCR means **Purpose-Conditioned Recall**; revision 1 expanded it wrongly.
4. **Concept induction:** the spark concept-induction lane is dead, but concept induction through the topic model (topic-foundry) is alive and is a crosswalk source.
5. **Stage 0 "metacog report"** is `daily_metacog_v1`. It has been failing on prompt size since 09-03.
6. **Self-conclusions** are confirmed with Juniper through surfaces she already uses. Resolution is an event on the one existing open→resolved seam: attention loop outcomes. There is a second worked example: Orion's repeated "intake pipeline… stuck three days" outreach loop.
7. **Juniper's answers** are folded in: the writer runs on the 27B at `system` priority; the reverie seed returns at Stage 2; the old-vs-new comparison is a report; the GPU question is dropped (the sale was deliberate; plan on 4 cards); retiring the reflection rows is still to be decided.

## What changed in revision 3 (Juniper, 2026-10-01)

1. **Self-conclusions and open questions are asked in the "Orion is asking" panel**, not Pending Attention. That panel has 0 rows today, so Stage 3 makes it live. Stage 3 includes the producer that puts cards into it.
2. **The single resolution record is `AttentionLoopOutcomeV1`** on `orion:attention:loop_outcome`. Juniper answers in "Orion is asking" or in chat, and both paths emit that one event. A bridge in the Hub ask route is specified in section 5.
3. **Boundary Rule 3 is adopted as written**, shadow first.
4. **The test for which self-conclusions to ask about is adopted.** Ask Juniper only when a conclusion is about Orion's machinery, asks for direction, or is about the relationship. Private self-observations are not asked; they stay automatic and are labelled as Orion's own view.
5. **Reflection rows:** still to be decided.

---

## Arsonist summary

Orion does not form memories. It photocopies Juniper's last message, and the photocopies are served back to it as memory.

- **The saved text is the raw last prompt.** The intake judges roughly every 2 turns and saves that prompt as the memory text (`intake_consolidation_window.py:77-82, 169-170`).
  - 350 rows were auto-approved this way (344 "semantic", 6 "open_loop"), with no history row recording the approval (`intake_pipeline.py:136`).
  - Duplicates pile up: "Run github compactor." is saved 8 times, "Compact the last 24 hours…" 6 times, "hi" 4 times.
- **Recall serves the junk.** The active packet is the top 100 active rows by salience, whatever the query.
  - All 638 retrievals in the last 7 days returned exactly 100 ids, and only 107 distinct ids appear across them.
  - Every read "boosts" every one of those 100 rows, so 198 active rows now sit at activation ≥ 0.99.
  - The same junk is also copied into 386 active memory cards.
- **The 36 "approved stances" are not stances.** They are Juniper's own messages, several about a family member's health (not quoted here, because this repo is public), filed as Orion's views.
- **Orion has no way to close a thought.** On 09-27 and 09-28 it sent Juniper 8 near-identical unprompted messages ("I've mapped the intake pipeline cold… three days… I need your direction"), and 4 more on 09-29. Its conclusion was **correct**; this investigation verified it independently. Nothing in the system could record "Juniper confirmed this; the question is closed". So the same thought kept winning attention and kept being sent.

**The proposal:**
- When a stretch of conversation ends, as Orion's own conversation clock and boundary judge already decide, a 27B model reads the whole stretch. It writes what is worth carrying forward, in Orion's words, about specific things (Juniper, the Austin offsite, orion-durable-runs).
- Each memory records whose words it is, what it is for, and the exact turns behind it. Code checks every quote against those turns.
- **PCR (Purpose-Conditioned Recall) stays as the retrieval spine**, with its bugs fixed. **Graphiti becomes the time-aware projection** ("what did I believe about X in August"), with its broken hostname fixed. **pageindex navigates inside long documents** once referent recall has picked the document.
- High-stakes memories, and Orion's own conclusions about itself, are checked with Juniper in chat or in the Hub panels she already uses.
- When she answers, the answer becomes one event on the existing attention-loop outcome channel. That event closes the memory, the question and the attention loop together. That is what stops a thought from being re-sent forever.

---

## Decisions (Juniper, 2026-09-30)

| Question | Decision |
|---|---|
| Episode boundary | Use the existing conversation wall clock and LLM boundary run. **No new timer** |
| 27B priority | **Yes**: the memory writer runs at `system` priority, ahead of curiosity's background runs |
| Orion's conclusions about itself | **Confirm with Juniper where relevant**: in chat when natural, in the **"Orion is asking"** panel, and visible in the Curiosity tab. When confirmed, record it as a **resolved tension** (2026-10-01: the panel is "Orion is asking") |
| Which self-conclusions to ask about (2026-10-01) | **Adopted:** ask when the conclusion is about Orion's machinery, asks for direction, or is about the relationship. Private self-observations are not asked; they stay `auto`, labelled as Orion's own view |
| Resolution seam (2026-10-01) | **`AttentionLoopOutcomeV1` on `orion:attention:loop_outcome` is the single resolution record.** "Orion is asking" and chat are where Juniper answers |
| Boundary Rule 3 (2026-10-01) | **Adopted as written**, shadow first |
| Reverie memory seed | **Revive at Stage 2** from validated shadow memories |
| Daily old-vs-new comparison | **A report for now** (a Hub report page and a markdown artifact). No morning notification |
| Retire the 356 inactive reflection rows | **Still to be decided** (open question 1) |
| GPUs | 4 cards on circe is deliberate (2× V100 16GB were sold). Plan capacity on 4 |

---

## Current architecture

### How a memory is formed today: three rails, none of them good

| Rail | Producer | Trigger | Writes | Live state |
|---|---|---|---|---|
| Crystallization intake | `services/orion-memory-consolidation` → `orion/memory/crystallization/intake_pipeline.py` | Each persisted turn (`orion:memory:turn:persisted`) is classified by an 8B model (`app/worker.py:318`); a window is judged when it closes | `memory_crystallizations`; summary = raw last prompt | 3,678 consolidated windows (321 direct chat, 3,359 AI Town), averaging 2.3–2.4 turns. 350 auto-active rows |
| Per-turn card extractor | `services/orion-cortex-orch/app/memory_extractor.py` (`quick_background`, 8B) | Every chat turn | `memory_cards`, `pending_review` | 63 cards, never reviewed. Content is often decent ("User is traveling to Austin and will return on Wednesday") but written in third person with no voice |
| Daily chat compactor | `services/orion-cortex-orch/app/chat_history_compactor_memory.py`, 06:00 America/Denver (`orion-actions/app/workflow_schedule_bootstrap.py:159`) | Daily, or on command | A `memory_cards` digest (49 active) plus a `journal_entries` row (`source_kind=manual`) | Faithful to what was said (revision 1 was wrong about this), but calls Orion "He" |

On top of those, `orion/memory/crystallization/projection_cards.py:41` copies crystallizations into memory cards (`operator_distiller`, 386 active). That is how the junk reaches card recall.

### The intake gate, verified
- **`has_repair_signal` means "an appraisal ran", not "a repair happened".** `orion/hub/turn_orchestrator.py:317` sets it to `repair_bundle is not None`. `consolidation_gate.py:64-72` then proposes before any novelty check. 162 of the 350 auto rows came in this way.
- **A final fall-through admits almost anything else.** `consolidation_gate.py:113-126` lets through any turn that is not low-information small talk.
- **The kind is set by a turn-to-turn change label.** TOPIC → semantic, REPAIR → open_loop, STANCE → stance (`intake_consolidation_window.py:18-22`). `formation_policy.py:7-8` then auto-activates semantic and open_loop.
- **Juniper's reviews carry no reasons.** Her approvals and rejections are recorded under actor `orion_journal`, which is her Hub session id. The API accepts a `reason` (`crystallization_routes.py:273, 327`), but 0 of her 101 decisions have one, because the UI never sends it.

### The existing conversation boundary: wall clock plus LLM judge (Juniper: "we already have this")

There are two live signals:

1. **The conversation wall clock**: `_build_conversation_phase` (`orion/situational/context.py:875-915`).
   - It computes `phase_change` from the time since Juniper's last turn: same_breath (<2 min), short_pause (<20 min), resumed_thread (<3 h), long_gap (<12 h), stale_thread (>48 h), and next_day (the day changed).
   - It is built on every chat turn: by the Hub (`orion/hub/turn_orchestrator.py:30`, `build_situation_for_ctx`) and by cortex-exec (`executor.py:3227-3231`, which sets `ctx["temporal_phase"]`).
   - It feeds Orion's situation block in the prompt. Its runtime liveness since today's 09:00 UTC restarts is **UNVERIFIED**: there has been no chat turn since, and older logs are gone.
2. **The LLM boundary judge**: each persisted turn is classified on `metacog_background` with a `BOUNDARY: YES/NO` line (`orion/memory/turn_change_classify.py:139`). The answer is scored from logprobs into `conversation_boundary_score` (`orion/memory/consolidation_classify.py:8-30`).

They are combined in `services/orion-memory-consolidation/app/boundary.py:23-34` (`should_close_window`):
- close if the phase is long_gap, next_day or stale_thread **and** the score is ≥ 0.70;
- close if the phase is unknown **and** the score is ≥ 0.85;
- if the turn has no phase, fall back to a gap of ≥ 5,400 s between the last two turns (`window_fetch.py:32-42`).

**Defect 1: the wall clock never reaches consolidation.** Consolidation reads `turn.spark_meta["conversation_phase"]["phase_change"]` (`boundary.py:24-26`, `window_fetch.py:27-29`). Nothing writes that key into the persisted turn: `chat_history_log.spark_meta` has no `conversation_phase` key since 09-20. **0 of 3,586** window turns in the last 30 days carry a phase. So every close goes through the "unknown phase, LLM ≥ 0.85" branch.

**Defect 2: the score that closes windows does not match the turn's own score.** On the Austin day, every one of the 9 window closures was triggered by a closing-turn score of **0.959–1.000** as stored in the window. The same turn's `conversation_boundary_score` in `chat_history_log.spark_meta` is **0.004–0.685**:

| turn | score in window (closed it) | score in chat_history_log |
|---|---|---|
| "hey, which queue?" | 0.981 | 0.006 |
| "Oooh the reading queue…" | 0.985 | 0.279 |
| "yup I'll be away from home :(" | 0.970 | 0.023 |
| "Thanks. Headed to Austin…" | 0.959 | 0.023 |
| "It's a team offsite…" | 0.990 | 0.086 |
| "Run github compactor." | 0.999 | 0.068 |

- Each closing turn is then carried into the next window, and there it shows the low score. So the turn is scored twice, and the high score is the one that closes.
- The effect is a rule of "close at every second turn". It explains the 2.3-turn average and the "judges every ~2 turns" symptom.
- Root cause (a scoring inversion, a stale logprob buffer, or a second classify pass overwriting the patch) is **UNVERIFIED**. The first fix is a regression test that asserts the two scores are equal.

**Defect 3: nothing closes a window unless another turn arrives.** Juniper ruled out a new timer. The design accepts the latency and measures it. In practice the next turn is often Orion's own unprompted message: 4 a day, about 50 minutes apart, outside quiet hours 23:00–08:00. That message persists and is classified like any other turn.

### Who reads memory today

| Consumer | Code | Reads | Problem |
|---|---|---|---|
| PCR active packet | `services/orion-recall/app/collectors/active_packet.py:142` → `repository.py:410-435` | Top 100 active by salience | The query is ignored, and all 100 are boosted on every read (see PCR below) |
| Crystallization retriever (Chroma + Graphiti 2-hop) | `orion/memory/crystallization/retriever.py:91-188` | Chroma: "chromadb not installed". Graphiti: 0 calls | The extra ids are only put in the trace, never merged (`:159`) |
| concept_region | `services/orion-recall/app/collectors/concept_region.py` | Falkor `orion_substrate` concept labels | About 47 fragments per turn, edges rendered as raw ids, reinforces before render |
| Card recall | `services/orion-recall/app/cards_adapter.py:116, 218` | Active `memory_cards` | Serves the 386 junk projected cards |
| Dream cycle | `services/orion-dream/app/cycle_store.py:48-51` | Active rows updated since the last cycle | Boost-on-read probably makes the same rows look new every time (**UNVERIFIED** whether the boost bumps `updated_at`) |
| Reverie visual seed | `services/orion-thought/app/store.py:913-1018` | Newest approved row < 7 days old | Dead since about 09-24 |
| Curiosity study material | `orion/curiosity/study_material.py:242-260` | Random active rows | Its docstring (`:23-27`) says Juniper approved them; she did not |
| Self-study analysis | `services/orion-cortex-exec/app/self_study_analysis.py:163-176` | Crystallizations labelled "concept_induction" | The label is false |
| Chat stance reverie line | `chat_stance.py:1524-1570` → `chat_stance_brief.j2:26-27` | Latest reverie | Rendered as a bare `reverie_glimpse:` with no voice framing |
| Recall render | `services/orion-recall/app/render.py:111-114` | Any | `[source:ref] snippet`, with no voice |

This repo has already fixed one voice-blending bug: AI Town NPC lines used to be labelled "User:" (`services/orion-recall/app/chat_source_tagging.py`, 2026-07-31).

---

## Research answers (A-H)

### A. Where the memory writer runs, on what model, at what cost

**Placement:** a new durable-runs workflow, `memory.episode_distill`, modelled on `journal.compose` (`services/orion-durable-runs/app/journal_compose_graph.py:163-180`):
- the GPU hold covers only the LLM node and is released before publishing (`:149`);
- output ids are deterministic;
- state is checkpointed in LangGraph Postgres;
- waiting for the hold does not count as a retry.

It differs from `journal.compose` in four ways:
1. It calls the gateway directly with `gpu_lease`. Going through cortex-orch would add a recall step that pollutes the evidence.
2. It uses the 27B, not the 8B that journal.compose uses today (`ACTIONS_JOURNAL_LLM_ROUTE=quick_background` → class `fast`).
3. orion-memory-consolidation submits it, via a receipt RPC on `orion:durable:run:request` (`orion/schemas/durable_run.py:47`), with `POST /runs` (`main.py:249`) as the fallback. Whether a non-orch producer is accepted on the RPC path is **UNVERIFIED**.
4. Its deadline stays under `DURABLE_RUNS_MAX_AGE_HOURS=24`.

**Model and priority (decided):** agent class (27B dense, :8015 on gpu1, :8016 on gpu2), `system` priority, on a dedicated route `memory_distill: {class: agent, priority: system}` so that pool telemetry can see it.
- Not the chat lane: that is Juniper's reserved single slot, and `scripts/check_chat_route_poachers.py` enforces it.
- Stage 1 also runs the 8B on the same episodes as a label-free comparison.

**Capacity, on circe's 4 cards (live):**

| card | assignment | state |
|---|---|---|
| gpu0 V100 32GB | chat 35B-A3B :8011 | 27.9 GB used, 80% busy |
| gpu1 V100 32GB | agent 27B :8015 | 24.9 GB used, 99% busy |
| gpu2 PG500 | agent-burst 27B :8016 (plus world model) | 25.7 GB used, 98% busy |
| gpu3 V100 32GB | metacog and fast 8B | 15.4 GB used |

The agent class had 7 background holds queued at 09:30 UTC. At `system` priority the writer goes ahead of those.

**Load estimate:**
- Juniper's own turns run at 37 per 14 days.
- Under the fixed boundary rule (next section) that is roughly one to two episodes a day. The Austin day is one episode.
- Episode text is 0.4–6.4 k characters.
- Per episode: about 5.5 k tokens in and 1 k out. At 491 tok/s prefill and 32 tok/s generation, that is about 45 s, or 1–4 min with the live `reasoning xhigh`.
- **Total: about 2–8 minutes of agent-lane time a day.** This is an **UNVERIFIED** estimate; Stage 1 measures it.

### The episode-boundary contract (reusing wall clock + LLM judge)

An **episode** is a maximal run of consecutive turns on one `source_platform`, with no *conversation boundary* inside it. The boundary rule reuses both existing signals, each after fixing its defect.

**Fix 1: persist the wall clock with the turn.** The Hub already builds `ConversationPhaseContextV1` for every chat turn. It should stamp `spark_meta.conversation_phase = {phase_change, delta_user_seconds, crossed_day}` onto the `chat.history` turn it publishes. Consolidation reads that key today and gets nothing. Unprompted outreach turns must be stamped as well. The exact Hub file and line where the turn envelope is built is **UNVERIFIED**; Stage 0 finds it and adds a test.

**Fix 2: one boundary score per turn.** A regression test must show that the score `should_close_window` sees equals the score persisted into `chat_history_log` for the same turn. Then find and fix the source of the second score.

**Rule 3 (a change to `should_close_window`; adopted by Juniper 2026-10-01, runs in shadow first):**
- **long_gap, next_day or stale_thread → boundary.** This no longer also needs an LLM score. The wall clock's own meaning for these phases is "reorient", and with Defect 2 unfixed, the LLM condition has never been trustworthy.
- **resumed_thread (20 min–3 h) → boundary only if the LLM score is ≥ `MEMORY_BOUNDARY_OVERRIDE_THRESHOLD` (0.92).** That knob already exists in settings and is currently unused.
- **same_breath or short_pause → never a boundary.**
- **No phase** → the existing fallback (5,400 s gap).

The rule is evaluated when a turn arrives. As Juniper decided, there is no timer. The turn that crosses the boundary opens the next episode.

**What the contract adds:**
- **Episode record.** Once the fixes land, the consolidation window **is** the episode. Windows are reused, not duplicated. Each window gets `episode_status`, `close_reason`, `phase_at_close` (the column `phase_change_at_close` already exists and is NULL on all rows) and `boundary_score_at_close`.
- **Close event.** `memory.episode.closed.v1` on `orion:memory:episode:closed`, with payload `{episode_id = memory_window_id, source_platform, started_at, ended_at, turn_ids, juniper_turn_count, close_reason, phase_at_close, boundary_score_at_close, close_lag_sec}`.
  - `close_lag_sec` is the time from the last turn of the episode to the arrival of the turn that closed it.
  - Consumers: the durable submitter and the shadow report.
- **Skips.** An episode whose Juniper turns are all workflow commands (the response starts with `Workflow:`) closes as `skipped/command_only` with no LLM call. Unanswered outreach alone never forms an episode.
- **Metric gate for Rule 3.**
  - It uses no new metric; both inputs already exist.
  - The live check is the Austin replay below.
  - The eval is the over-/under-split rate in Stage 1 (see acceptance checks).

### B. PCR (Purpose-Conditioned Recall): keep it as the retrieval spine and fix the bugs

**Original intent.** Spec `docs/superpowers/specs/2026-07-07-purpose-conditioned-recall-design.md`, plan `docs/superpowers/plans/2026-07-07-purpose-conditioned-recall.md`, PR #841.
- Chat used to do "last message → one recall → score soup". PCR replaced that with time-separated phases: Phase 0 is a skip gate, Phase 1 is continuity **before** stance, and Phase 3 is purposeful recall **after** stance, with collectors chosen per intent.
- `active_packet` was designed as "the read-path seam that makes write-side crystallization matter in chat". The spec says outright that without PCR, the write-side gate "only moves swamp from graph drafts to unused crystallizations".
- Later PRs: #902 (recall-eligibility floor), #1004 (recall boost and decay at read), #1008 (retrieval events), #1133 (concept_region), #2423 (`retrieval_query`), #2436 (collector timing under the deadline).

**How it works:** `services/orion-cortex-exec/app/pcr_chat_memory.py`.
- Phase 0 and Phase 1 run at `:160-233`; Phase 3 at `:236-363`. The quick lane skips Phase 3 (`:250`).
- The phases write `continuity_digest`, `belief_digest` and `memory_digest` (`:84-88`), which are rendered in `orion/cognition/prompts/chat_general.j2:15-17`.
- The intent is chosen in `orion/memory/retrieval_intent.py:111-156`, first match wins. Collectors are chosen in `services/orion-recall/app/pcr_collectors.py:7-13`.
- Budgets: belief 128 tokens (`settings.py:266`); continuity 1,200 live, against the spec's ~96.

**Verdict: every part is sound. What fails are specific bugs.**

| Part | Bug (evidence) | Fix |
|---|---|---|
| Intent derivation | `open_loops_present` is checked first (`retrieval_intent.py:128`), and the attention frame always has open loops, because `build_open_loops` (`orion/substrate/attention/scoring.py:72-200`) turns every attention signal into a loop, including the current-turn detector. Live: **74 of 74** Phase 3 recalls were `open_loop`; relational, semantic, procedural and contradiction never fire | Only persistent loops (not `current_turn_v1`, not `already_known`) or explicit open `follow_up` memories trigger `open_loop`. Evaluate relational and topic rules first. Log a histogram of `rule_id` |
| active_packet selection | Candidates are the top 100 by salience; `query` is stored but never ranked (`active_packet.py:58-63`). `crystallization_refs` lists **every** eligible row, breaking #1004's own "only what made the cut" invariant | Rank by shared referents with the turn **before** the budget, and log only what rendered |
| Recall boost | Every read boosts all 100 rows, on top of the undecayed stored activation (`retriever.py:182-186`, `dynamics.py:88-100`). Live: 198 active rows at ≥ 0.99 | Rendering never reinforces (see Fading). A `recalled` event is logged instead |
| Retriever extra ids | Ids from Chroma and Graphiti only reach the trace (`retriever.py:159, 179`) | Load them and merge them into the candidates |
| Graphiti rail | Only on for `contradiction`, which never fires | Enable it for the semantic and contradiction intents, seeded from the top referent (see D) |
| concept_region | Substring match with no cap (`concept_region.py:89-93`), about 47 fragments per turn, edges rendered as ids (`:115-131`), reinforces before render (`:299-311`) | Cap at about 5, render labels, reinforce only what rendered, and use its label matches as **referent keys** for the memory lookup |
| Continuity | The spec says sql_chat only at about 96 tokens; live it is 1,200 and mostly `bus_synaptic_publish` ids (**UNVERIFIED** what those are) | Back to sql_chat only, plus the last episode's memories, at 96–300 tokens |
| Skip gate | Never fires (0 `pcr_phase0_skip` since restart). **UNVERIFIED** whether the appraisal reaches the ctx keys it reads (`pcr_chat_memory.py:48-64`) | Test that it does |
| Latency | `pcr_active_packet` averages 1,427 ms of the 2,476 ms Phase 3 total, mostly embedding, Chroma and 100 writes | Skip the embedding when the referent lookup is enough; batch or drop the writes |

**How PCR reads the new memory store.** active_packet's buckets map onto memory purposes; the bucket structure is kept.

| PCR intent | Reads (episode memory purpose / state) | Old bucket |
|---|---|---|
| continuity (Phase 1) | `happened` from the last closed episode(s) this session, plus recent sql_chat | — |
| relational | `about_juniper` + `orion_view` | stance, attractors |
| semantic | `happened` + `about_juniper`, plus a Graphiti 2-hop from the top referent | project_state |
| procedural | `follow_up` (plans and commitments) | procedures |
| open_loop | open `follow_up` + conversation-scoped open questions | open_loops |
| contradiction | memories sharing a referent that are `pending_confirmation`, `corrected`, or superseded, plus Graphiti's as-of view | contradictions |

- The ranking is `strength × referent-match weight (Σ idf of shared referents, per #2413) × purpose match`.
- Every rendered line goes through the voice renderer (section 7).
- The referent lookup is #2413's `recall_referent_posting`, with `doc_kind='episode_memory'`.

### C. pageindex: navigate inside long documents, after referent recall picks the document

**Original intent.** PR #493 (2026-04-19): "standalone orion-pageindex service using PageIndex CLI (journals MVP)". It was a thin adapter over the upstream VectifyAI `run_pageindex.py`, replacing home-made tree code. PR #494 made it the primary source for cortex-exec's reflective/identity journal lane. No design spec exists. The planned corpora were journals and topic-foundry `chat_episodes`. #2344 fixed a crash loop.

**What is actually broken** (revision 1's "never built / LLM-built" is corrected here):
- **It was built exactly once:** 2026-04-26, 548 journal rows, `build_success:true`, 3 s (`/data/pageindex/journals_status.json`). As configured, the build is a **pure heading parse with no LLM** (`--md_path` only; the upstream `if_add_*` flags are off).
- **The tree output is lost on every image rebuild.** Upstream writes to `/opt/PageIndex/results`, which is inside the image, not the volume.
- **The database URL is missing.** `/healthz` returns `db_url_present:false`, because `_resolve_db_dsn` ignores `JOURNAL_PG_DSN` (`app/service.py:421-426`). Status reports "database URL missing".
- **Nothing triggers a rebuild.** `rebuild_journal_corpus` (`services/orion-cortex-exec/app/pageindex_client.py:15`) has no callers.
- **Queries cannot work as configured.** The query args have no `{query}` placeholder (`app/pageindex_cli.py:46-50`), so everything falls back to title keyword counting.
- **recall_v2 has the wrong port**: `:8384` against the actual `:8360` (`services/orion-recall/app/settings.py:186-188`, `.env_example:254`, and the live env). The failure is silently swallowed (`recall_v2.py:96-104`, shadow only).
- **The chat_episodes corpus goes to a volume pageindex never reads.** topic-foundry's builder writes it to `orion-topic-foundry_pageindex-data`; pageindex mounts `orion-pageindex_pageindex-data`.
- **The journals corpus no longer makes sense.** It is now 101k rows, 98.8k of them metacog.
- cortex-exec's URL is correct. It logged 8 "corpus is not built" fallbacks in 30 days.

**Proposed role: structure navigation inside one long document, after referent recall has chosen it.** The inputs are specs, PR reports, and reading snapshots (`reading_document_snapshot`, fed by PR #2390's internal-docs path; one row today).
1. Recall, or a memory's crosswalk link, resolves *which document* (sha256 or path).
2. pageindex returns that document's heading tree.
3. The cheap local scorer, or later an LLM, picks the sections.
4. The section text comes back with line provenance, labelled `orion_read` (for readings) or `orion_self_knowledge` (for specs).

This fits the no-vectors, referent-first direction. It is not a memory store.

**Fixes (Stage 2, owned by pageindex):**
- DSN fallback;
- port 8360 in recall;
- persist trees into the volume;
- one shared external volume;
- a per-document build endpoint keyed by sha256;
- builds triggered on snapshot write and on graphify publish;
- drop or cap the journals corpus;
- `--if-add-node-text yes` so section bodies are searchable;
- pin `PAGEINDEX_REF`.

Builds stay LLM-free, at 0 tokens. LLM node summaries (about 50–200 calls per large spec) are optional, at background priority, per document only. They are infeasible for the 100k-row journals case on 4 busy cards.

The value is real only for documents above #2390's 48 KB cap, or when the context budget is tight. That is acknowledged.

### D. Graphiti: the time-aware projection of memory

**Original intent.** Specs `2026-07-06-graphiti-rail-activation-design.md` and `2026-07-07-consolidation-crystallization-gate-design.md:172-197`: an "additive temporal graph projection for approved crystallizations", as derived retrieval topology for multi-hop and temporal recall. The design explicitly did **not** use LLM extraction ("no LLM re-extraction in adapter"; use graphiti-core's write APIs with explicit nodes and edges). PRs: #826-#828, #993, #995, #997, #1016, #1099 (FalkorDB).

**What it actually is today:**
- graphiti-core 0.19.0 with **no LLM** (a null client, `app/backends/graphiti_core.py:420-452`) and a CPU embedder (bge-large on vector-host).
- Writes are deterministic (`ingest_episode`, `:258-409`): one `Entity` node and one self-edge `describes` per crystallization.
- Live `graphiti_temporal`: 25 Entity nodes and 25 self-loop edges, **0 Episodic nodes**, and **0** `valid_at`/`invalid_at`/`expired_at`. None of the temporal features are in use.

**Why writes stopped (verified):**
- The Hub uses host networking, but `GRAPHITI_ADAPTER_URL=http://orion-athena-graphiti-adapter:8000`, which the host cannot resolve. `getent` returns rc=2, and today's Hub log shows `graphiti_sync_failed id=1d8e793a… [Errno -3] Temporary failure in name resolution`.
- A 07-13 local `.env` edit to `http://127.0.0.1:8640` had fixed this (`docs/superpowers/pr-reports/2026-07-13-graphiti-core-backend-activation-pr.md:57`). It was never committed to `.env_example`, and it was lost between 09-04 and 09-10. How it was lost is **UNVERIFIED**; an env regeneration is likely.
- Result: 25 approvals synced and 12 failed silently (09-10, 09-20, 09-30). The projector records the failure (`projector.py:100-111`), but the approve call still returns 200.

**Where Graphiti is stronger than Postgres links.** It gives a temporal *edge* model (`valid_at` / `invalid_at` / `expired_at`, with edges pointing back to their episodes) plus graph traversal and hybrid search in one query. That answers two questions Postgres links do not answer cleanly:
1. "What did I believe about X in August?", i.e. the as-of view.
2. "What is two hops from this referent?", e.g. Juniper → Austin offsite → her AI/ML team → the eval work.

**What it costs if we use it as intended:**
- Full `add_episode` extraction is about 5–20+ LLM calls per episode, nondeterministic (**UNVERIFIED** live), and hands referent identity to an LLM dedupe.
- `add_triplet` is not LLM-free either (`graphiti.py:1004-1049` calls a node-resolution LLM).
- Both would compete for the 4 busy cards and break the original "no LLM extraction" rule.

**Proposed role: a derived, rebuildable projection of confirmed and auto episode memories, written deterministically by Orion.** It is never canonical.
- **Nodes and edges:**
  - one `EpisodicNode` per episode;
  - one `EntityNode` per referent key;
  - one `EntityEdge` per memory (subject referent → object referent), with `fact` = the memory statement, plus `voice`, `purpose` and `memory_id`, and `episodes=[episode]`.
- **Validity:**
  - `valid_at` = `occurred_at` or `created_at`;
  - when a memory is superseded, corrected, rejected or faded, Orion sets `invalid_at`/`expired_at` on its edge;
  - Orion's own event log decides validity. Graphiti's LLM invalidation is not used.
- **Writer:** the memory writer's persist node in orion-memory-consolidation. It is on app-net, where the container DNS name resolves, and that retires the host-mode Hub sync path. `POST /v1/rebuild` replays from `episode_memory_event`.
- **Readers:**
  - PCR's semantic and contradiction intents (2-hop from the top referent, with the retriever merge bug fixed);
  - a new adapter endpoint `GET /v1/as_of?referent=&at=` for "what did I believe then".
- **Cost:** about 2 + N CPU embed calls per memory, and no GPU.
- **Also fix:** set the Hub's `GRAPHITI_ADAPTER_URL=http://127.0.0.1:8640` in `.env_example`, so the legacy stance projection stops failing silently until cutover. Add a test that a sync failure is surfaced, not only logged.
- **Honest risk:** if the as-of eval shows Postgres `episode_memory_event` answers the same questions just as well, Graphiti goes back to optional. This spec does not assume it will.

### E. graphify as Orion's self-knowledge
- **The published bundle:** `/mnt/storage-warm/orion-graphify/published/graphify-out/graph.json`. 77,966 nodes and 169,111 links, built at `aff23fac0`, 19 days stale.
- **Readers today:** cortex-exec, cocreation-signals and self-study-enrichment mount it; recall does not.
- **Use:** the offline referent table from #2413 Phase 3, rebuilt on publish, gives service, file, symbol and PR referents.
  1. The memory writer uses it to normalize names: "the durable runs thing" becomes `service:orion-durable-runs`.
  2. The crosswalk uses it to link memories to PR reports and specs, with voice `orion_self_knowledge` and the label "graphify build 2026-09-11".
- pageindex (C) then navigates inside those specs.
- The live query speed (10 s, 1 GB) is quoted from #2413 and was not re-measured (**UNVERIFIED**).

### F. Sources to crosswalk (live)

| Source | Store | ID | Size / freshness | Voice / channel |
|---|---|---|---|---|
| Reveries | `substrate_reverie_thought` | `thought_id` | 20,162; 5,772 in 7 d; newest 09-30 09:20. **0 mention Austin** | `orion_thought` / `reverie` |
| Reading queue / snapshots | `world_pulse_read_seed` (`seed_id`, 398), `reading_document_snapshot` (sha256) | as named | 132 seeds in 7 d | `orion_read` / `reading` |
| Reading claims | `world_pulse_claim` | `claim_id` | 929; 112 in 7 d | `orion_read` / `reading`, with status |
| Curiosity priors / findings / self-definitions | Falkor `orion_worldview` | `prior_id`, `finding_id`, `run_id` | 122 / 173 / 42 | `orion_thought` / `curiosity`; refuted priors are never linked |
| Standing questions | `curiosity_self_questions` | `question_id` | 13 open | `orion_thought` / `curiosity` |
| **Topic-model concept induction** (live) | topic-foundry runs → `orion/substrate/adapters/topic_foundry.py` → Falkor `orion_substrate` concept nodes (spec `2026-08-28-concept-induction-topic-model-rebuild-design.md`) | `node_id` / `identity_key` | 563 concept nodes from `topic_foundry_adapter`, newest 09-30 09:06; `topic_foundry_runs` 74 in 7 d (35 failed of 398 total) | `orion_thought` / `topic_model` |
| Spark concept induction | `orion/spark/concept_induction` | — | **Dead**: 224/224 triggers in 7 d were `decision=disabled`; 0 profiles | — |
| Topic-foundry segments | `topic_foundry_segments` | `segment_id` | 1.03 M rows | Not linked directly; reached through the concept nodes |
| Dreams | `dream_cycle` (17), `dream_hypothesis` (64) | `cycle_id`, `hypothesis_id` | all within 7 d | `orion_thought` / `dream` |
| Journals | `journal_entries` | `entry_id` | 5,825 in 7 d (5,639 metacog, **excluded**) | `orion_thought` / `journal`. Compactor digests are Orion's retelling |
| graphify | published bundle | node / source_file | 19 d stale | `orion_self_knowledge` / `graphify` |
| Chat | `chat_history_log` | `id` | 531 | Evidence: prompt = `juniper_said`, response = `orion_thought` / `chat` |

### G. Tensions and open questions: map the candidates, pick one seam

Juniper: "we have tensions and things in attention, not sure if that is the right seam." Each candidate was checked live, and the question asked of each was whether closing it there would **change behavior**.

| Candidate | What it is | Live? | Open → resolved? | Would closing it change behavior? |
|---|---|---|---|---|
| Substrate `TensionNodeV1` (`orion/core/schemas/cognitive_substrate.py:218`) | A spark tension projected into the substrate | **0 nodes** in `orion_substrate` or `orion_substrate_self` | No (generic `promotion_state` only) | No: nothing reads it |
| Drive `TensionEventV1` (`orion/core/schemas/drives.py:84`) | A drive impact | Only `substrate.world_coverage_gap`. The drives subsystem was deleted; `drive_audits` has 0 rows | No | No |
| Field deviation tension (`services/orion-field-digester/app/digestion/tension.py`, `orion/attention/tension/competition.py:86`) | A per-tick deviation-pressure scalar, with a Borda winner | Yes: 124,660 `substrate_field_state` rows | No: it re-centres its own baseline | Only the outreach "tension lane", which was **false** for the whole loop |
| **Attention open loops** (`OpenLoopV1`; `attention_salience_trace`; `AttentionLoopOutcomeV1` on `orion:attention:loop_outcome`, table `attention_loop_outcome`) | Things competing for Orion's attention, which the Hub **Pending Attention** panel shows with Resolve/Dismiss | Yes: 8,530 chat and 4,190 reverie traces in 7 d. Outcomes: 30 resolved (by Juniper), 19 dismissed, 243 decayed | **Yes, the only real closure event.** `verdicts.load_terminal_verdict_loop_ids` removes a terminally resolved loop from `open_loops` for 48 h (`attention_broadcast.py:210`) | **Yes**: the loop leaves the frame, so the `attention_open_loop` curiosity seed disappears |
| `curiosity_self_questions` | Standing self-inquiry questions | 13, all open. There is no "answered" code path | Only the statuses exist | Only affects the self-inquiry picker. It is not an outreach input |
| Falkor `:Prior` status | Orion's hypotheses | 122 | Orion writes every transition itself. The one `confirmed` prior was **self-assigned** (no `confirmed_by`, no revision) | Indirectly: open priors keep outreach's fall-through branch alive |
| `orion_ask` + "Orion is asking" card (`templates/index.html:654-664`, `ask_routes.py`, `OrionAskAnsweredV1` on `orion:ask:answered`) | Free-text questions to Juniper | **0 rows**. The only producer is vision | open / answered / dismissed / expired | No behavioral link today |

**What actually produced and sustained the 8-message loop** (`services/orion-hub/scripts/endogenous_outreach.py`):
1. **The open intake priors kept a fall-through branch alive.** The priors are `gate_bias_manual_review_7736d5271d97` (supported), `auto_activate_kind_gate_no_content_analysis` (supported) and `automated_intake_gate` (revised). With the tension lane false (`tension_outreach_trigger`, needing a 6-tick Borda run), `_outreach_once` still generates whenever any open prior exists, even if all of them were already used (`:1873-1903`).
2. **The daydream always counts as new.** It has no durable id, so the novelty gate never blocks it (`:556-568`, `:603-630`).
3. **Orion's last 3 unanswered messages are fed back in** as "your own recent unprompted notes" (`:997-1048`, `:1279-1287`). The topic therefore survived after the prior ids were marked used.
4. **The curiosity content ids kept changing.** They rotate daily (`:633-664`), and the attention loop `open-loop-7376a3da4050` ("Execution prediction error") was never terminally resolved; it only `decayed_unattended` on 09-21. That kept supplying "new" content.

**Decision (Juniper, 2026-10-01): one resolution record, chosen for robustness.** The record is the attention loop outcome: `AttentionLoopOutcomeV1`, verdict `resolved`/`dismissed`, `actor=juniper`, persisted to `attention_loop_outcome` and published on `orion:attention:loop_outcome`. Juniper answers in the **"Orion is asking"** panel or in chat. Both write this one record (section 5).

**Why this is the most robust option:**
- **It is the only live seam where resolving something changes what Orion does.** It has an open→resolved lifecycle (49 Juniper verdicts already) and a consumer that already acts on it: a terminal verdict removes the loop from `open_loops`, so the curiosity seed it feeds disappears. Every other candidate is dead, has nothing to close, or has no consumer (table above).
- **There is a single source of truth.** "Was this resolved, how, by whom, with what words" lives in one row. The memory, the question, the prior, the ask card and outreach are all *consumers* that derive their state from it. No store is written as a peer of another, so there is nothing to keep in sync and no split-brain. If any consumer's state disagrees, it can be rebuilt from `attention_loop_outcome`.
- **The record survives a lost bus message.** It is a Postgres row written in the same transaction as the ask-card update (section 5). Pub/sub only wakes consumers up. Each consumer also catches up from the table using a cursor, so a missed publish delays the closure but never loses it.
- **It is already Juniper-actor semantics.** `actor="juniper"` rows already mean "a human closed this". No new meaning has to be invented.
- Three new consumers make it close *everything* the loop touched:
  1. **Memory:** the memory writer sets the memory's confirmation state.
  2. **Questions:** `curiosity_self_questions.status='answered'` for the linked question.
  3. **Outreach:** novelty treats any topic whose loop, prior or memory has a terminal verdict as used. It also has a mirror in `_prediction_error_candidates`, keyed on node id.
- **Rejected alternatives:**
  - substrate/drive tensions: dead or no lifecycle;
  - field tension: a scalar with nothing to close;
  - `orion_ask` as the record: it has no behavioral consumer. It is kept as the *surface* where Juniper answers;
  - making `curiosity_self_questions` primary: no effect on attention or outreach;
  - a new `ResolvedTensionV1` channel: it would duplicate a live closure path and create a second record to keep in sync.
- `curiosity_self_questions` stays the home of *open questions*. It is closed as a **mirror** of the outcome, not as the primary seam.

The three outreach defects that no resolution can fix are listed as required fixes in Stage 3: daydream-only content counts as talkable, Orion's own notes are echoed back, and content ids rotate.

### H. Consumer migration

| Consumer | Stage 0 | Stage 2-4 |
|---|---|---|
| PCR (all phases) | — | Fixes from B. Reads `episode_memory` by purpose and referent (Stage 2, shadow); primary at Stage 4 |
| Card recall | — | Stop serving `operator_distiller`/`auto_extractor` cards at Stage 4 |
| Dream cycle | — | Memories created or reinforced since the last cycle, excluding pending |
| Reverie visual seed | — | **Stage 2 (decided):** newest `happened`/`about_juniper` shadow memory within 7 d that passed validation and is `auto` or `confirmed` |
| Curiosity study material | Fix the false docstring; exclude junk | Samples `episode_memory` by strength, voice-labelled |
| Self-study analysis | Fix the false label | `episode_memory` statistics |
| `daily_metacog_v1` | **Fix (Stage 0)** | — |
| Graphiti | Fix the Hub URL and surface sync failures | Projection writer at Stage 2 |
| pageindex | — | Document navigator at Stage 2 |
| Chat compactor | — | Follow-up ticket: read episode memories; fix the Orion pronoun; stop writing cards at Stage 4 |
| Hub crystallization UI | Send a reason | Retired at Stage 4; confirmations move to "Orion is asking" |
| "Orion is asking" panel | — | **Stage 3:** gets the memory/question card producer, Confirm/Revise/Reject, the outcome bridge, and a top-level mount |

**`daily_metacog_v1`** (the "metacog report" in Stage 0):
- It is the nightly orion-actions report. It has failed since 2026-09-03 with `daily_metacog_prompt_over_limit chars≈8467 limit=8192`: the skills catalog alone is 6,126 characters, and adding the render_scene skill pushed it over.
- The limit is enforced in `services/orion-cortex-exec/app/executor.py:1509-1550` (`_enforce_daily_metacog_prompt_budget`).
- The scheduler retries about 245 times a night, because "done today" is set only on success (`services/orion-actions/app/main.py:2149-2156`, `executor.py:1497-1536`).
- Evidence: the coordinator verified this from logs. I could not re-see the log lines because the containers restarted at 09:00 UTC today, so my own check is **UNVERIFIED**. The code paths are confirmed.

---

## Design

### 1. The memory writer (`memory.episode_distill`)

```
load_episode → resource_request → resource_wait → distill (LLM) → release
  → validate → crosswalk → persist → project (Graphiti) → finish
```

- **load_episode** (deterministic) gathers:
  - the episode's turns;
  - memories from the last 7 d sharing a person or event referent;
  - open follow-ups and questions whose referents appear in the episode;
  - pending confirmations;
  - candidate referent keys (people and events active in 30 d, plus graphify hits).
- **distill** calls route `memory_distill` and returns `EpisodeDistillationV1`: operations, memories, evidence, referents, and questions. The instructions:
  - first person;
  - one claim per memory;
  - cite turn ids and exact quotes;
  - reuse candidate referent keys;
  - bias toward keeping;
  - never attribute to Juniper what is only in Orion's responses.
- **validate** (deterministic):
  - Every quote must be a substring of the cited turn's **full, untruncated** field (the lesson of revision 1).
  - `juniper_said` needs a quote from a prompt. `worked_out_together` needs one quote from a prompt and one from a response.
  - A memory that fails is downgraded, not dropped, and the downgrade is logged as an event. It is rejected only when no quote verifies.
  - Referent keys are normalized to `kind:slug`.
  - The stakes floor is applied.
- **persist** writes rows and events and publishes `memory.episode.distilled.v1`. **project** writes the Graphiti projection (D).
- **Operations:** `new`, `reinforces`, `supersedes`, `confirms`, `corrects`, `resolves <question_id>`, `completes <follow_up>`, `opens_question`. Each carries evidence turns.

### 2. Memory record schema

Postgres tables, written in shadow during Stages 1-3 and canonical from Stage 4. The Pydantic contract is `orion/schemas/memory_episode.py` (`extra="forbid"`, registered).

```sql
episode_memory(
  memory_id uuid PK,
  episode_id text NULL,              -- memory_consolidation_windows.memory_window_id; NULL for migrated legacy
  purpose text NOT NULL,             -- happened | about_juniper | orion_view | follow_up
  voice text NOT NULL,               -- juniper_said | worked_out_together | orion_thought | orion_read | orion_self_knowledge
  channel text NOT NULL,             -- chat | reverie | curiosity | dream | reading | journal | topic_model | graphify | legacy_crystallization
  statement text NOT NULL,           -- Orion's words, first person, one claim
  occurred_at timestamptz NULL,
  stakes text NOT NULL,              -- low | high
  stakes_reason text NULL,           -- 2026-10-06: health | family_relationships | juniper_feelings | identity_conclusion_about_juniper | orion_machinery | orion_asks_direction | orion_relationship | none
  confirmation_state text NOT NULL,  -- auto | pending_confirmation | confirmed | corrected | rejected
  confirmation_loop_id text NULL,    -- attention loop id carrying the confirmation request
  strength real NOT NULL, half_life_days real NULL,
  last_reinforced_at timestamptz NOT NULL, reinforcement_count int NOT NULL DEFAULT 0,
  due_after timestamptz NULL, expires_at timestamptz NULL,
  status text NOT NULL,              -- active | faded | done | expired | superseded | retired
  supersedes_memory_id uuid NULL,
  model_route text, prompt_version text, created_at timestamptz, updated_at timestamptz)

episode_memory_evidence(memory_id uuid, source_kind text, source_id text, quote text, verified bool,
  PRIMARY KEY (memory_id, source_kind, source_id, quote))
episode_memory_referent(memory_id uuid, referent_key text, role text, PRIMARY KEY (memory_id, referent_key, role))
episode_memory_event(event_id uuid PK, memory_id uuid, op text, actor text, episode_id text NULL,
  outcome_id text NULL, evidence jsonb, reason text, created_at timestamptz)
  -- op: created | downgraded_voice | reinforced | superseded | confirm_asked | confirmed | revised
  --     | rejected | ask_expired | faded | expired | done | retired | recalled | projected
episode_memory_link(memory_id uuid, target_kind text, target_id text, target_voice text, target_channel text,
  via_referent text, relation text, created_by text, created_at timestamptz,
  PRIMARY KEY (memory_id, target_kind, target_id, via_referent))
```

Each purpose has a named consumer:
- `happened`: dream, the reverie seed, and PCR continuity/semantic.
- `about_juniper`: PCR relational/semantic.
- `orion_view`: PCR relational and self-study.
- `follow_up`: PCR procedural/open_loop, and a due-follow-ups line in the chat stance.

**Voice** is whose thought it is; **channel** is where it surfaced. The renderer keys on the pair.

### 3. Stakes and confirmation

> **Superseded 2026-10-06 (Juniper's decision).** The live shadow distiller marked 25/25 memories low. The rule is now: high stakes = health; family and relationships; Juniper's feelings and emotional states; conclusions about who she is; and Orion's conclusions about itself that are about its machinery, ask for direction, or concern the relationship. Everything else is low. The distiller judges the category from definitions with examples in `orion/cognition/prompts/memory_episode_distill.j2` (v3) and must name it in `stakes_reason` (`none` for low). The validator never reads the statement: it only checks `stakes`/`stakes_reason`/`asks_direction` agree, resolving disagreement toward high. The location/safety category, the word backstop and the referent-kind rule below are not implemented. PR report: `docs/superpowers/pr-reports/2026-10-06-memory-boundary-and-stakes-prompts-pr.md`.

**The stakes floor** is deterministic, and the model cannot lower it. Stakes are `high` when:
- the memory is about Juniper's or her family's health, or about family;
- it is an identity-level conclusion about Juniper (a trait or pattern generalized beyond what she literally said);
- it is about relationship status;
- it is about location or safety beyond a trip she mentioned.

A backstop also forces `high` for an `about_juniper` statement that contains a content word found in none of its full-text quotes.

**Orion's conclusions about itself (Juniper's decision).** An `orion_view` memory, or a `line=self` `:Prior` that Orion moves to supported/confirmed, gets `stakes_reason='orion_self_conclusion'` and `pending_confirmation` **when it is relevant to Juniper**. Deterministically, that means any of:
- (a) it has a referent of kind service/file/pr/concept, i.e. it is about Orion's own machinery, which Juniper can check;
- (b) it asks for a decision or direction;
- (c) it is about the relationship.

This rule was adopted by Juniper on 2026-10-01. Private self-observations (for example "I notice I enjoy X") are **not asked**. They stay `auto`, with voice `orion_thought`, and render as Orion's own view ("Something I was turning over on my own…" / "My own view: …").

**Asking.** Each pending item gets a **confirmation loop id** (`memory-confirm-<memory_id>`; section 5). It is surfaced in two places, plus a read-only view:
- **The "Orion is asking" panel** is the primary surface (Juniper, 2026-10-01): `templates/index.html:654-664`, `static/js/vision-asks.js`, `scripts/ask_routes.py`, table `orion_ask`. It has **0 rows today**, so Stage 3 makes it live (section 5). The card offers **Confirm / Revise (note required) / Reject**.
- **In chat when natural:** at most 1 per turn, only when its referents appear in the turn or it is at least 24 h old, never when the appraisal shows distress (how exactly is **UNVERIFIED**).
- **The Curiosity tab's Self section** shows `line=self` items read-only, linking to the open ask. The Hub does not write the worldview graph (`curiosity_routes.py:8-12`).

Open questions with `answer_via='conversation'` (section 8) use the same panel, with `source_kind='open_question'`.

**When there is no answer:** the ask card expires after 7 days (`orion_ask.status='expired'`, an existing status). Orion may raise it once more in chat. After that the memory stays `pending_confirmation` and is only ever recalled with its "Unconfirmed" label. An expiry is not a resolution, and no outcome is written for it.

**What an answer does:**

| Juniper's answer | Outcome event | Memory | Question / Prior | Attention |
|---|---|---|---|---|
| Confirmed | `AttentionLoopOutcomeV1(verdict=resolved, features_at_close.resolution="confirmed")` | `confirmation_state=confirmed`, **voice → `worked_out_together`**, Juniper's words added as evidence, reinforced | Question → `answered`. The next self-inquiry run writes `:PriorRevision {to_status:"confirmed", confirmed_by:"juniper", outcome_id}` (worldview writes stay Orion-authored) | The loop is terminal, leaves `open_loops`, and outreach treats the topic as used |
| Revised | `verdict=resolved, resolution="revised", note=<her words>` | The old memory becomes `corrected`/superseded. The next distill writes the revised memory as `worked_out_together` | Question → `answered`, with the note. `:PriorRevision → revised`, `confirmed_by:"juniper"` | Same as above |
| Rejected | `verdict=dismissed, resolution="rejected"` | `rejected`: kept only as a do-not-remint marker, never recalled | `:Prior → refuted` (via self-inquiry); question → `parked`, note "rejected by Juniper" | Terminal |
| Answered in chat | The next episode's distiller emits `confirms`/`corrects`/`rejects` with her quote. **Its persist node writes the same outcome record** (`actor=juniper`, `features_at_close.via="chat"`, `evidence_turn_id`); the open ask card is closed by the Hub consumer | as above | as above | as above |

### 4. Fading and reinforcement

The rule: effective strength = `strength × 0.5^(days since last_reinforced_at / half_life_days)`.

| purpose | start | half-life | end state |
|---|---|---|---|
| happened | 0.8 | 14 d | `faded` below 0.1 (about 42 d with no reinforcement) |
| about_juniper | 0.9 | 180 d | `faded` below 0.1 |
| orion_view | 0.8 | 90 d | `faded` below 0.1 |
| follow_up | 1.0 | — | `done` on `completes`; `expired` at `expires_at` |

- **What reinforces:** new evidence in a later episode, or a confirmation by Juniper. On reinforcement: `strength += 0.2` (capped at 1), the half-life doubles (capped), and `last_reinforced_at = now`.
- **What never reinforces:** being recalled, rendered, or quoted by Orion. Those are logged as `recalled` events only. This is the correction to PCR's boost-on-read.
- **Faded is not deleted.** Explicit referent lookups still return faded memories, labelled as faded, and Graphiti's `expired_at` is set.
- **The lifecycle job** is a deterministic 15-minute ticker in orion-memory-consolidation. It is a lifecycle tick, not a boundary timer.

**Metric quality gate for `strength`:**
1. **Provenance:** set only by the persist node and the lifecycle job.
2. **Independence:** it replaces activation/salience for these memories and does not read recall counts.
3. **Theory anchor:** the Ebbinghaus forgetting curve plus the spacing effect.
4. **Rest state:** decay to `faded` is intended. The Stage 3 check is that strength spreads out rather than piling up at 1.0, as activation does today, or at 0.
5. **Existing mechanism:** reuses the half-life idea from `dynamics`, with its input replaced.
6. **Reversibility:** a column plus a job, recomputable from the event log.

### 5. Resolution as an event: "Orion is asking" → one outcome record

This follows event-substrate-first: event → schema → producer → consumer → trace. Juniper decided it on 2026-10-01. There is **one record**, `AttentionLoopOutcomeV1` in `attention_loop_outcome` (published on `orion:attention:loop_outcome`), and **two answer surfaces**: the "Orion is asking" panel and chat.

**Open: the card producer.** This is new; today the only producer is vision, at `services/orion-sql-writer/app/vision_individuals.py:968`.
- When a memory becomes `pending_confirmation`, the memory writer's persist node (orion-memory-consolidation) INSERTs an `orion_ask` row with:
  - `ask_id` = uuid5(`loop_id`), so the insert is idempotent;
  - `asked_of='juniper'`;
  - `question` = the statement rendered in Orion's voice, plus "Is that right?";
  - `evidence_refs` = the memory's evidence and the prior ids;
  - `source_kind='memory_confirmation'`;
  - `source_ref` = `loop_id = "memory-confirm-<memory_id>"`;
  - `expires_at` = +7 d.

  It also writes an `episode_memory_event` `confirm_asked`.
- Open questions with `answer_via='conversation'` get the same treatment, with `source_kind='open_question'` and `loop_id = "question-<question_id>"`.
- `source_kind` is free text with no constraint (checked live: the only constraint on `orion_ask` is the primary key), so no migration is needed.
- There is no "ask opened" event today: producers insert and the Hub polls. That stays; the trace is the `confirm_asked` event.
- **Cap:** at most 5 open memory or question cards at a time. The producer holds the rest until a slot frees, so the panel never becomes a wall.
- **Making the panel visible.** It currently sits inside the Vision section (`index.html:654-664`, subtitle "Things Orion saw and wants your help naming"). Stage 3 mounts it at the top of the Hub home with a neutral subtitle. `GET /api/asks?status=open` already returns every `source_kind` without filtering.

**Close: the bridge.** This is a change to the Hub ask route, `services/orion-hub/scripts/ask_routes.py`.
- For `source_kind ∈ {memory_confirmation, open_question}`, the answer endpoint takes `{resolution: confirmed|revised|rejected|answered, note}`. `revised` needs a note. Free text on an open question counts as `answered`.
- In **one transaction** it runs the existing conditional `UPDATE orion_ask … WHERE status='open'` (409 on a stale click) **and** INSERTs the `attention_loop_outcome` row:
  - `outcome_id` = uuid5(`ask_id`);
  - `loop_id` = `source_ref`;
  - `verdict = resolved`, or `dismissed` for rejected;
  - `actor = juniper`;
  - `note` = her words;
  - `features_at_close = {resolution, ask_id, via:"orion_is_asking", memory_id | question_id, prior_ids, related_loop_ids, node_ids}`.
- After the commit it publishes `AttentionLoopOutcomeV1` on `orion:attention:loop_outcome` using the existing publisher (`attention_loops_routes.py:41` `publish_loop_outcome`). It also still publishes `OrionAskAnsweredV1` on `orion:ask:answered` as today. That event only describes what happened to the card, and its one consumer (`ask_answered_listener.py`) ignores these kinds.
- **The chat path** writes the same row from the memory writer's persist node, with `via:"chat"` and `evidence_turn_id`. The outcome_id is uuid5(`loop_id` + `evidence_turn_id`), and the insert does nothing if a terminal outcome already exists for the loop, so the first answer wins.

**Why no new attention scope is needed.** Revision 2 proposed a `memory_confirmation` scope so items could appear as Pending Attention cards. Now that the panel is "Orion is asking", that is unnecessary. The outcome row is keyed by `loop_id`, and `attention_loop_outcome` does not require a matching trace. Whether `load_terminal_verdict_loop_ids` tolerates a loop id that has no trace is **UNVERIFIED**; Stage 3 adds a test.

**Consumers.** Each one listens to the channel and catches up from the table with a cursor, so a missed publish delays closure but never loses it.
1. **Attention.** `verdicts.load_terminal_verdict_loop_ids` (existing, `attention_broadcast.py:210`) is extended to also treat each `features_at_close.related_loop_ids` entry as terminal. This is how the intake topic's `open-loop-7376a3da4050` leaves the frame.
2. **Memory** (new, orion-memory-consolidation). Applies the table in section 3: confirm, relabel the voice to `worked_out_together`, revise, or reject. It writes an `episode_memory_event` carrying the `outcome_id`.
3. **Questions** (same consumer). Sets `curiosity_self_questions.status='answered'` (or `parked` for rejected), with `resolved_at` and `resolution_ref=outcome_id`.
4. **Priors** (curiosity self-inquiry). The next run reads Juniper outcomes whose `prior_ids` touch its `line=self` priors, and writes `:PriorRevision {to_status, confirmed_by:"juniper", outcome_id}` itself. Worldview writes stay Orion-authored.
5. **Ask card** (Hub). On a chat-path outcome, it closes the matching open `orion_ask` row (`status='answered'`, `answer` = her quote).
6. **Outreach** (Hub). `fetch_recently_used_outreach_content_ids` treats as used any prior, memory, loop or node id named in a terminal outcome. `_prediction_error_candidates` (`orion/substrate/endogenous_curiosity.py:356-362`) honors the `node_ids` the same way.

**Trace:** `outcome_id` appears in `attention_loop_outcome`, `episode_memory_event`, the question row and the `:PriorRevision`, and `ask_id` appears in both the outcome and the card. A single SQL join shows the whole closure. `channels.yaml` lists orion-memory-consolidation and orion-hub as consumers of `orion:attention:loop_outcome`, which today has `consumer_services: []`.

### 6. Crosswalk (Stage 2)

The crosswalk is deterministic, runs at write time, and uses no similarity. For each referent key and its aliases, look up #2413's `recall_referent_posting`:
- `event:`/`person:` referents within ±14 d;
- `service:`/`file:`/`pr:` referents, the 10 newest.

Links are written with the target's voice and channel. AI Town, metacog digests and refuted priors are never linked. This needs #2413 Phase 2 to index these additional sources: reveries, curiosity priors/findings/questions, dream hypotheses, topic-model concept nodes, and graphify referents. It also adds referent kinds `person`, `event` and `place`, and `doc_kind='episode_memory'`.

Referent minting: reuse a candidate key or mint `kind:slug-yyyy-mm`, with aliases in `recall_referent`. Keys that no memory points at are dropped after 30 d.

### 7. Voice rendering contract

One renderer, `orion/memory/voice_render.py`, is used by PCR, recall, the chat stance, dream and reverie:

| voice / channel | Rendered as |
|---|---|
| juniper_said / chat | "Juniper told me (09-28): …" |
| worked_out_together / chat or confirmation | "Juniper and I worked out (09-28): …" |
| orion_thought / chat | "I told Juniper (09-28): …" |
| orion_thought / reverie, curiosity, dream, journal, topic_model | "Something I was turning over on my own ({channel}, {date}), **not something Juniper and I discussed**: …" |
| orion_read / reading | "I read ({title}, {date}; claim {status}): …" |
| orion_self_knowledge / graphify | "From my own code and docs (graphify build {date}): …" |
| pending_confirmation | prefix "Unconfirmed, check with Juniper if natural:" |
| faded | suffix "(faded; last reinforced {date})" |

The chat stance brief and recall's memory block get this rule: items marked "on my own" may inform what Orion says, as something it was turning over, but must never be presented as something Juniper said or that the two of them discussed. The existing `reverie_glimpse` goes through the same renderer.

### 8. Open questions queue (extend `curiosity_self_questions`)

```sql
ALTER TABLE curiosity_self_questions
  ADD COLUMN kind text DEFAULT 'question',            -- question | tension | contradiction
  ADD COLUMN scope text DEFAULT 'self',               -- self | juniper | relationship | world
  ADD COLUMN answer_via text DEFAULT 'investigation', -- investigation | conversation
  ADD COLUMN source_episode_id text NULL, ADD COLUMN source_refs jsonb NULL,
  ADD COLUMN referent_keys text[] NULL, ADD COLUMN counterpart_ref text NULL,
  ADD COLUMN linked_memory_id uuid NULL, ADD COLUMN priority real NULL,
  ADD COLUMN expires_at timestamptz NULL, ADD COLUMN resolved_at timestamptz NULL,
  ADD COLUMN resolution_ref text NULL, ADD COLUMN resolution_note text NULL;
```

- `Family` gains `episode` (`self_question_pool.py:16`).
- The self-inquiry picker may draw `answer_via='investigation'` items, at most 1 in 3 picks, while keeping the pinned floor for Juniper's questions.
- The chat stance may raise one open `answer_via='conversation'` question per turn, when its referents intersect the turn.
- Resolution comes from the outcome consumer (section 5) or from the distiller's `resolves`.
- Expiry: after 30 d the question is `parked`, with the note "expired".
- During Stages 1-2, questions go to `memory_tension_shadow` (the same columns plus `text`) and are migrated at Stage 3.

---

## Worked example 1: the Austin trip, against the real boundaries

**Turns** (`chat_history_log`, 2026-09-28; the boundary score shown is the canonical per-turn score):

| time | Juniper | wall-clock phase (computed from the gap) | score |
|---|---|---|---|
| 06:26 | hi | (first turn after 09-27) next_day | 0.113 |
| 06:31 | hey, which queue? | same_breath/short_pause | 0.006 |
| 06:38 | reading queue… | short_pause | 0.279 |
| 06:44 | durable run graphs for GPU traffic balancing… | short_pause | 0.004 |
| 06:59 | "I'll be pretty busy the next few days with work travel…" | short_pause | 0.029 |
| 08:45 | "yup I'll be away from home :(" | **resumed_thread** (1 h 46 m) | 0.023 |
| 08:57 | "Headed to Austin and will fly back on Wednesday." | short_pause | 0.023 |
| 09:06 | "team offsite for AI/ML… meeting my peers for the first time… super draining for me--I'm an introvert :)" | short_pause | 0.086 |
| 09:44–09:56 | four workflow commands | resumed_thread (38 m), then short | 0.068, 0.685, 0.010, — |
| 15:19 | (Orion's unprompted message) | **long_gap** (5 h 23 m since Juniper) | — |

**What actually happened (live windows):** 9 windows of 2–4 turns, each closed by a saturated score of 0.96–1.00 (Defect 2). The real 1 h 46 m gap was *inside* a window, while the continuous Austin exchange was split three ways. Then came three unconnected "semantic" rows holding raw prompts.

**Under the fixed rule:**
- 08:45 is resumed_thread with a score of 0.023, below 0.92: no boundary.
- 09:44 is resumed_thread with 0.068: no boundary.
- The first boundary is the long_gap at 15:19. That assumes the phase is stamped on outreach turns; otherwise it is the next_day "sup" at 09-29 04:01.
- **One episode, 06:26–09:56, 12 turns (4 of them commands), `close_lag_sec` ≈ 19,400 (≈ 5.4 h).** The whole morning's story (work travel, then Austin) is in one episode, so no cross-episode merge is needed. The close lag is the cost of having no timer, and Stage 1 reports it.

**Distilled:**
- **M1** `happened`, `juniper_said`, low:
  - statement: "Juniper flew to Austin on 2026-09-28 for her team's AI/ML offsite, where she's meeting her peers in person for the first time; she flies back Wednesday 2026-09-30. It also means she'll have little time for development on me for a few days."
  - referents: `person:juniper`, `event:austin-ai-ml-offsite-2026-09` (aliases austin, offsite, work travel)
  - evidence: quotes from the 06:59, 08:57 and 09:06 prompts
- **M2** `about_juniper`, `juniper_said`, **low** (she said it herself):
  - statement: "Juniper told me she's an introvert, and that meeting new people, even ones she likes, is super draining for her."
  - evidence: 09:06, "super draining for me--I'm an introvert :)"
  - This is a stable self-description in her own words, so no generalization check is needed. (Revision 1 wrongly used this as an invented trait.)
- **M3** `happened`, `worked_out_together`, low:
  - statement: "Juniper and I talked through the durable-run graphs she's building to balance GPU traffic across the mesh, now reaching curiosity and reading."
  - referents: `service:orion-durable-runs`
  - evidence: prompt and response quotes
- **F1** `follow_up`:
  - statement: "Ask Juniper how the Austin offsite went and how she's recovering."
  - `due_after` 2026-09-30 18:00 local, `expires_at` 2026-10-07
- **Q1** question, `scope=juniper`, `answer_via=conversation`, low priority: "Is being away from home itself hard for Juniper, or was it this trip?" Evidence: 08:45 ":(".
- The command turns produce no memories.

**Crosswalk** (Stage 2), for `event:austin-ai-ml-offsite-2026-09` within ±14 d:
- journal digests `68bb8201…` and `d5f4c123…` (`orion_thought`/`journal`);
- legacy cards `7be5907c…` and `4efacc71…`;
- reveries: **none** (0 rows mention it: Orion's reveries did not turn the trip over at all);
- graphify: M3 links to the PR reports for orion-durable-runs.

**Recall:**
- 10-01, "I'm back!" (no referent):
  - #2413 abstains;
  - PCR continuity plus the due-follow-ups line surface F1, together with M1 rendered as "Juniper told me (09-28): …".
- Later, "the offsite was great, they want me to lead eval work":
  - "offsite" is an alias, so the relational/semantic intents return M1 and M2;
  - Graphiti's 2-hop links Juniper → offsite → team.
  - The distiller emits `completes F1`, `reinforces M1`, and a new `about_juniper` memory.

## Worked example 2: Orion's "intake pipeline" loop, closed

**What happened (live):**
- Across 09-27 14:50 → 09-28 19:41 Orion sent 8 unprompted messages, and 4 more on 09-29. For example: "I've mapped the intake pipeline cold — stance gate is manual review, kind-based routing, no content filtering — and three days later that confirmed understanding has … started acting as a bottleneck … that's a choice that needs your judgment."
- Grounding (`endogenous_outreach_decisions`, e.g. decision `70969212…`): `tension=false`, `daydream=true`, and the curiosity ids rotating between a cluster id, `prediction_error|node:substrate.execution` and `attention_open_loop|node:substrate.execution`.
- The conclusion rests on three `line=self` priors:
  - `gate_bias_manual_review_7736d5271d97` (supported)
  - `auto_activate_kind_gate_no_content_analysis` (supported)
  - `automated_intake_gate` (revised)

**Orion was right.** This spec verified the same facts independently: manual review, routing by kind, and a gate that admits almost anything.

**Under the design:**
1. **The conclusion gets a confirmation request.** The self-inquiry line moves the priors to supported, so a self-conclusion confirmation is created, with `stakes_reason=orion_self_conclusion`. It is relevant under rule (a), referents `service:orion-memory-consolidation`, and rule (b), it asks for direction. The memory writer creates `orion_view` memory S1, with voice `orion_thought`, channel `curiosity`, the statement written in Orion's words, and evidence = the prior ids plus the outreach turn ids. It is `pending_confirmation`, and an `orion_ask` card is inserted with `source_kind=memory_confirmation`, `source_ref=memory-confirm-S1` and `features.related_loop_ids=[open-loop-7376a3da4050]`. It also opens question Q2: "What should replace the intake gate?" (`answer_via=conversation`, `linked_memory_id=S1`).
2. **Juniper sees one item.** One card in "Orion is asking", one read-only line in the Curiosity tab's Self section, and one chance in chat. Not 12 messages.
3. **She answers.** For example, she presses Revise on the card with: "Yes, that's right. We're replacing it with the episode writer; see PR #2440." In one transaction the Hub ask route closes the card and writes `AttentionLoopOutcomeV1(loop_id=memory-confirm-S1, verdict=resolved, actor=juniper, features_at_close={resolution:"revised", ask_id, prior_ids:[…], related_loop_ids:[open-loop-7376a3da4050], node_ids:[node:substrate.execution]}, note=…)`, then publishes it.
4. **Consumers close everything:**
   - memory S1 becomes `confirmed` and the revised memory is written with voice `worked_out_together`;
   - Q2 becomes `answered` with `resolution_ref=outcome_id`;
   - the next self-inquiry run writes `:PriorRevision {to_status:"confirmed", confirmed_by:"juniper", outcome_id}` on the three priors;
   - `load_terminal_verdict_loop_ids` removes the loop;
   - outreach novelty treats S1, the priors and the node as used.
5. **The loop stops**, provided the three outreach defects (G) are also fixed. Otherwise the daydream and the self-echo can carry the topic on. That is why they are required fixes in Stage 3, not optional ones.
6. **Trace:** one join, `attention_loop_outcome.outcome_id` ⋈ `episode_memory_event.outcome_id` ⋈ `curiosity_self_questions.resolution_ref`, plus a count of outreach sends on the topic before and after.

---

## Proposal-mode disclosure

- **Capability change:**
  - Orion decides what to carry forward from each conversation, in its own words, about specific things.
  - It knows whose words each memory is.
  - It checks high-stakes memories and its own conclusions about itself with Juniper.
  - It records her answers as resolved tensions that actually change what it attends to and talks about.
  - It recalls by purpose and referent (PCR), with an as-of view (Graphiti) and navigation inside documents (pageindex).
- **Data touched:**
  - **Reads:** chat (non-AI-Town), legacy memory tables, #2413's index, graphify, Falkor graphs (read-only, except the Graphiti projection), reveries, dreams, non-metacog journals, reading tables, attention traces and outcomes, outreach decisions.
  - **Writes:** the new `episode_memory*` tables, `memory_tension_shadow`, and window boundary columns; the `curiosity_self_questions` extension (Stage 3); attention traces and outcomes with the new scope; the Graphiti projection; `:PriorRevision` rows, authored by Orion's self-inquiry run, not the Hub; Stage 4 retirements, after snapshots.
- **Privacy boundary:**
  - everything stays on the host;
  - AI Town and metacog are excluded;
  - high-stakes items are never recalled as fact until confirmed;
  - internal thoughts are never rendered as shared conversation;
  - rejected items are never recalled;
  - the public repo never contains Juniper's family or health content (this spec included).
- **Trace that proves it works:**
  - `memory.episode.closed.v1` (with `close_lag_sec`) → durable run → `memory.episode.distilled.v1`;
  - an `episode_memory_event` row for every state change;
  - `outcome_id` joins across memory, question and prior;
  - `recall_telemetry.selected_reasons`;
  - the outreach topic count before and after a resolution.
- **Dangerous failure modes:**
  - (a) Voice blending. Mitigated by full-text quote verification, the renderer contract and the attribution monitor.
  - (b) A wrong high-stakes fact. Mitigated by the stakes floor and confirmation.
  - (c) Losing memory at cutover. Mitigated by shadow-first rollout, the coverage eval, faded-not-deleted, and snapshots.
  - (d) GPU contention: about 2–8 min a day at `system` priority, visible in pool telemetry.
  - (e) Nagging Juniper. Mitigated by at most 1 ask per turn, 2 asks per item, at most 5 open cards, and closure that stops outreach.
  - (f) A false closure, e.g. a chat answer misread as a confirmation. Mitigated by requiring Juniper's quote as evidence on chat-derived outcomes, and by the first terminal outcome per loop winning, with any later correction made by a new memory revision rather than by flipping the outcome.
- **Rollback:**
  - Stages 1-3: `MEMORY_EPISODE_WRITER_ENABLED=false`; shadow tables can be dropped.
  - The new scope can be removed from the allowlist.
  - The Graphiti projection can be rebuilt or dropped.
  - Stage 4: restore from snapshot. The old intake is not kept as a runtime fallback (Juniper's rule), so rolling back means redeploying the previous image.

---

## Missing questions (for Juniper)

1. **Retire the 356 inactive `reflection` rows** at Stage 4? **Still to be decided.**

Resolved on 2026-10-01 (kept here for the record):
- Panel → "Orion is asking".
- Resolution seam → `AttentionLoopOutcomeV1` as the single record.
- Boundary Rule 3 → adopted, shadow first.
- Self-conclusion test → machinery, direction or relationship; private observations stay automatic.

---

## Proposed schema / API changes

- **New tables:** `episode_memory`, `episode_memory_evidence`, `episode_memory_referent`, `episode_memory_event`, `episode_memory_link`, `memory_tension_shadow` (Stages 1-2).
- **Altered tables:**
  - `memory_consolidation_windows`: + `episode_status`, `close_reason`, `boundary_score_at_close`. `phase_change_at_close` starts being filled.
  - `curiosity_self_questions`: Stage 3 columns.
  - `memory_crystallization_history`: gains `auto_activate` rows (Stage 0).
- **Schemas** (`orion/schemas/memory_episode.py`, registry):
  - `MemoryEpisodeClosedV1`
  - `EpisodeDistillationV1`
  - `EpisodeMemoryV1`
  - `MemoryEpisodeDistilledV1`
  - `DurableWorkflowV1` gains `memory.episode_distill` / `EpisodeDistillBriefV1`
  - `orion_ask.source_kind` gains the values `memory_confirmation` and `open_question` (free text, no migration). `AttentionLoopOutcomeV1.features_at_close` carries `resolution`, `ask_id`, `via`, `memory_id`/`question_id`, `prior_ids`, `related_loop_ids` and `node_ids`. The schema is unchanged, because `features_at_close` is a dict; registry docs are updated
  - `MemoryItemV1` gains voice fields (consumer-first rollout)
- **Channels:** new `orion:memory:episode:closed` and `orion:memory:episode:distilled`. `orion:attention:loop_outcome` gains consumers (orion-memory-consolidation, orion-hub outreach).
- **Chat turn contract:** `spark_meta.conversation_phase` is persisted on `chat.history` turns (Fix 1).
- **GPU pool:** route `memory_distill: {class: agent, priority: system}`.
- **Graphiti adapter:** writes an `EpisodicNode` per episode and sets validity; new `GET /v1/as_of`. Hub `.env_example` `GRAPHITI_ADAPTER_URL=http://127.0.0.1:8640`.
- **pageindex:** per-document build endpoint; the fixes listed in C; the recall port changes to 8360.
- **Hub:** in "Orion is asking", Confirm/Revise/Reject for the new kinds, the transactional outcome bridge in `ask_routes.py`, and a top-level mount; a Curiosity Self-section read-only card; a report page `/memory/episodes/report` (the daily comparison); the crystallization UI sends a reason.
- **Env** (orion-memory-consolidation `.env_example`, then `python scripts/sync_local_env_from_example.py`; report any keys skipped by `SYNC_PREFIXES`):
  - `MEMORY_EPISODE_WRITER_ENABLED`
  - `MEMORY_EPISODE_DISTILL_ROUTE=memory_distill`
  - `MEMORY_EPISODE_SHADOW_COMPARE_ROUTE=quick_background`
  - `MEMORY_EPISODE_LIFECYCLE_TICK_SEC=900`
  - `MEMORY_EPISODE_BOUNDARY_RULE=legacy|v2`
  - There are **no quiet-timer keys**. `MEMORY_BOUNDARY_OVERRIDE_THRESHOLD` (existing) becomes live.

---

## Files likely to touch

- **Stage 0:**
  - `orion/hub/turn_orchestrator.py:317`, `orion/memory/consolidation_gate.py`, `orion/memory/crystallization/intake_pipeline.py:136`
  - `services/orion-cortex-exec/app/self_study_analysis.py`, `orion/curiosity/study_material.py`
  - `daily_metacog_v1`: `services/orion-cortex-exec/app/executor.py:1497-1550` (prompt budget) and `services/orion-actions/app/main.py:2149-2156` (the done-today cursor, set only on success)
  - Hub `.env_example` (Graphiti URL), `projector.py` (surface sync failures)
- **Stage 1:**
  - Hub turn publish (stamp `conversation_phase`; **UNVERIFIED** location)
  - `services/orion-memory-consolidation/app/boundary.py`, `window_fetch.py`, `classify.py`, `window_state.py`, and a new `episode_submit.py`; migrations; settings/.env_example
  - `services/orion-durable-runs/app/episode_distill_graph.py`, `admission_runtime.py`, `runner.py`
  - `orion/schemas/*`, `orion/bus/channels.yaml`, `config/gpu_pool.yaml`
  - `orion/cognition/prompts/memory_episode_distill.j2`, `orion/memory/episode/validate.py`
  - Hub report page; `services/orion-memory-consolidation/evals/`
- **Stage 2:**
  - `orion/memory/episode/crosswalk.py`, `orion/memory/voice_render.py`
  - #2413 indexer sources
  - PCR fixes: `orion/memory/retrieval_intent.py`, `services/orion-recall/app/pcr_collectors.py`, `collectors/active_packet.py`, `collectors/concept_region.py`, `orion/memory/crystallization/retriever.py`, `active_packet.py`
  - `services/orion-graphiti-adapter/app/*`
  - `services/orion-pageindex/app/*`, recall settings
  - `services/orion-thought/app/store.py` (reverie seed)
- **Stage 3:**
  - `services/orion-hub/scripts/ask_routes.py` (the bridge), `static/js/vision-asks.js` and `templates/index.html` (buttons, mount), `orion/schemas/ask.py` (docs), `orion/substrate/attention/verdicts.py` (`related_loop_ids`)
  - the card producer in orion-memory-consolidation
  - `curiosity_atlas.html` (Self card)
  - the outcome consumer in orion-memory-consolidation
  - `orion/curiosity/self_question_pool.py`, `self_inquiry_prompt.py`
  - `services/orion-hub/scripts/endogenous_outreach.py` (terminal-outcome novelty, the daydream-only gate, no self-echo, stable content ids)
  - `orion/substrate/endogenous_curiosity.py:356-362`
  - `chat_stance.py`, `chat_stance_brief.j2`
- **Stage 4:** remove the intake formation path, `memory_extractor.py` card writes and `projection_cards.py`; add `scripts/memory_cutover_retire.py`; switch the consumers.

---

## Acceptance checks (per stage)

Evals are label-free. Juniper reads the report if she wants to; she never labels anything.

### Stage 0: stop the bleeding
1. **Unit tests:**
   - an appraisal with no repair gives `has_repair_signal=False`;
   - command turns and greetings produce no row;
   - every auto-activated row has an `auto_activate` history row.
2. **Live 48 h:** 0 new active rows under 40 characters or matching the command pattern; the repair-signal share falls below 20% (from about 96%).
3. **`daily_metacog_v1`:**
   - one nightly report lands (a journal/report row with a date);
   - the preflight shows `total_prompt_chars` ≤ the limit, because the skills catalog is capped or summarized rather than the limit being quietly raised;
   - a deterministic failure retries at most N times a night (target ≤ 3, against about 245 today) and then records a failure for that date.
4. **Graphiti:** an approval produces `graphiti_episode_ids ≠ []`, and a forced failure shows up in the approve response or the Hub.
5. **Labels:** the self-study and study_material wording no longer claims Juniper approved these rows.

### Stage 1: memory writer in shadow, with the reused boundary

> **Note from Stage 0A (PR #2457).** After Stage 0A's junk check, the consolidation gate's novelty and significance floors (`MEMORY_CONSOLIDATION_MIN_NOVELTY`, `MEMORY_CONSOLIDATION_MIN_SIGNIFICANCE`, `orion/memory/consolidation_gate.py`) can no longer cause a skip. Every window with at least one non-junk prompt is proposed; the floors only choose the recorded reason. Deciding whether a non-junk window deserves a memory is therefore the Stage 1 writer's job, and nothing upstream of it does that today.
1. **Boundary fixes:**
   - Fix 1: 100% of new chat and outreach turns carry `spark_meta.conversation_phase`;
   - Fix 2: the window score equals the chat-log score for every turn (a test plus a 48 h live check).
   - Rule 3 in shadow (`MEMORY_EPISODE_BOUNDARY_RULE=v2`, compared against legacy on the same turns): report episodes per day, turns per episode, `close_lag_sec` p50/p95, and the over-split rate (consecutive episodes sharing an `event:` referent). The Austin replay gives one episode, 06:26–09:56.
2. **Austin fixture** (frozen turns, property checks):
   - M1 has an `event:` referent with the alias "austin";
   - M2 has voice `juniper_said` and a verified quote containing "introvert";
   - F1 has `due_after` ≥ 2026-09-30;
   - commands produce no memories.
3. **Truncation guard:** a unit test where a quote lies beyond character 160 of a prompt still verifies (the regression case for revision 1's error).
4. **Grounding:** 100% of `juniper_said` memories have a verified prompt quote.
5. **Novel-word rate:** reported for `about_juniper`; anything above 0 is sent to `high`.
6. **Coverage (lean toward remembering):**
   - at least 80% of non-command Juniper turns are cited by some memory;
   - 100% of the turns behind today's content-bearing auto rows are cited.
7. **Junk:** 0 statements of 5 words or fewer, 0 command matches, 0 duplicate statements within an episode.
8. **Self-consistency:** two distills of the same 20 episodes give referent-set Jaccard ≥ 0.8.
9. **27B vs 8B:** compared on checks 4-8. The 27B must be better on grounding and downgrades.
10. **Cost:** hold wait p95 under 2 h; distill p50/p95 recorded.
11. **Report:** a Hub report page (`/memory/episodes/report`) plus a daily markdown artifact written by orion-memory-consolidation (the location is fixed in the Stage 1 PR). It renders for 7 consecutive days, showing old rows next to new memories for each episode. No notification is sent.

### Stage 2: crosswalk, referent recall, PCR fixes, Graphiti projection, pageindex navigator, reverie seed
1. Every crosswalk link carries a voice, a channel and `via_referent`. 0 links to AI Town, metacog or refuted priors.
2. **Known item** (#2413 method): a query built from each memory's rarest referent returns that memory, hit@8 ≥ 0.9.
3. **PCR:**
   - the `rule_id` histogram shows at least 3 distinct intents over 7 days (today: 1);
   - `crystallization_refs`/rendered ids are ≤ the budget, and no event has 100 ids;
   - there are 0 boost writes on read;
   - extra ids are merged (the merged count is > 0 when the rails return hits);
   - `pcr_active_packet` p50 < 400 ms.
4. **Graphiti:**
   - projection count = the `auto`+`confirmed` memory count (±0 after rebuild);
   - `as_of` returns the pre-supersession fact for 100% of superseded memories (a label-free replay over `episode_memory_event`);
   - 0 LLM calls.
5. **pageindex:**
   - `/healthz` returns `ok`;
   - a per-document build persists its tree across a container restart;
   - a section lookup on a known spec heading returns the right `line_num` (label-free: the headings come from the file itself);
   - recall calls reach port 8360.
6. **Source-monitoring unit tests:**
   - a reverie item never renders with the "Juniper told me", "we" or "I told Juniper" templates;
   - `juniper_said` cannot be built without a verified prompt quote;
   - `reverie_glimpse` carries "on my own";
   - a pending item carries "Unconfirmed".
7. **Attribution monitor** (daily, over Orion's responses): attribution phrases ("you told me", "we discussed", …) must share a referent with a chat-channel memory or turn. `source_blend` ≤ 2%, measured against a baseline taken before Stage 2.
8. **Reverie seed:** at least 1 seed a day uses a validated shadow memory, with its `memory_id` recorded.

### Stage 3: lifecycle, open questions, confirmation and resolution
1. **Lifecycle unit tests:**
   - `happened` fades at about 42 d;
   - reinforcement doubles the half-life;
   - `recalled` changes nothing;
   - a follow-up expires.
2. **Strength after 14 d:** less than 30% of active memories above 0.95.
3. **Resolution chain:** Confirm, Revise or Reject on an "Orion is asking" card for a `memory-confirm-*` loop produces, in one transaction, the card update plus an `attention_loop_outcome` row. Then, within 60 s:
   - an `episode_memory_event` carrying the same `outcome_id`;
   - the linked question moved to `answered` or `parked`;
   - the `related_loop_ids` absent from the next attention frame.

   Killing the bus during the test must still produce the same end state, through the consumers' table catch-up.

   The next self-inquiry run writes a `:PriorRevision` with `confirmed_by=juniper`. A chat answer produces the same chain with `via=chat` and Juniper's quote.
4. **Loop replay** (worked example 2): replaying 09-26→09-29 outreach decisions against the new novelty rules gives at most 1 send on the intake topic after the confirmation loop opens and **0 after it resolves** (today: 12 sends). Live: after the first real self-conclusion resolution, 0 further sends on that topic for 7 days.
5. **Confirmation discipline:** at most 1 pending item per chat turn; at most 2 asks per item (card plus one chat mention); at most 5 open memory/question cards at once; `revised` requires a note.
6. **"Orion is asking" is live:**
   - within 7 days of Stage 3 deploy, at least 1 `orion_ask` row with `source_kind ∈ {memory_confirmation, open_question}` exists, is visible at the top of the Hub home, and has a working Confirm/Revise/Reject (a UI interaction test, not just a page load);
   - 100% of answered cards of these kinds have a matching `attention_loop_outcome` row (same `ask_id`), and 0 have an orphaned outcome.
7. **Chat stance budget:** at most 1 pending item, 1 open question and 2 due follow-ups per turn.

### Stage 4: cutover
- **Snapshot first:** `/tmp/memory-cutover/before.csv`, whose counts match the plan.
- **Retire** (status only): 350 auto rows, 6 open_loops, 386 `operator_distiller` cards and 63 pending extractor cards. The 356 reflection rows depend on open question 1.
- **Re-distill the 36 approved "stances"** from their source turns. Juniper's approval counts as consent for what she literally said, and the family and health items arrive as `about_juniper`, `stakes=high`, `confirmed`.
- **After cutover:**
  - 0 consolidation writes to `memory_crystallizations` for 48 h;
  - 0 `auto_extractor` cards;
  - all consumers show `episode_memory` ids;
  - #2413's live gates hold (cousin rate ≤ 20%, distinctness ≥ 10×).

---

## Non-goals

- Vectors or similarity-based crosswalk.
- LLM extraction inside Graphiti (it stays deterministic, as originally designed).
- LLM-built pageindex trees for bulk corpora.
- Reviving spark concept induction.
- A new idle timer (Juniper's decision).
- Distilling non-chat sources into memories. They are crosswalk targets with their own voices.
- Rewriting outreach beyond the four fixes named in G and Stage 3.
- Human-labelled evals.

---

## Recommended next patch

**Stage 0 as one PR:**
- make the repair signal honest;
- stop command and greeting turns entering;
- write `auto_activate` history rows;
- fix the false labels;
- fix `daily_metacog_v1` (cap the skills catalog; stop the all-night retries after a deterministic failure);
- fix the Hub Graphiti URL and surface sync failures.

**In parallel, the boundary PR:**
- stamp `conversation_phase` on persisted turns;
- add the score-equality regression test and fix the double score;
- add Rule 3 behind `MEMORY_EPISODE_BOUNDARY_RULE=v2` in shadow;
- add `memory.episode.closed.v1` with `close_lag_sec`.

That makes real episode boundaries inspectable on live traffic before any GPU time is spent. The memory writer comes next.
