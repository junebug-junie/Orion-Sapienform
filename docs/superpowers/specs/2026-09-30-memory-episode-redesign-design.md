# How Orion forms memories: episodes, Orion's own words, and knowing whose words they are

Status: PROPOSAL (design only, no code). Needs Juniper's answers to "Missing questions" before Stage 1.
Date: 2026-09-30
Evidence base: read-only live queries on `conjourney` Postgres, FalkorDB `GRAPH.RO_QUERY`, `ssh circe@circe nvidia-smi`, and docker logs/inspect, all taken 2026-09-30 about 09:30 UTC. Code read against main at a005658db.
Related specs:
- `2026-07-07-consolidation-crystallization-gate-design.md` (the gate this replaces)
- `2026-08-14-crystallization-queue-auto-gate-analysis.md`
- `2026-09-29-recall-semantic-retrieval-pipeline-design.md` (#2413, "recall by referent, not resemblance"; the retrieval side this design plugs into)
- `2026-09-29-recall-retrieval-query-architecture-design.md` (approved; separates the search query from the prompt)

Anything not checked live is marked **UNVERIFIED**.

---

## Arsonist summary

Orion does not form memories. It photocopies Juniper's last message, and a pile of those photocopies has been passed off as memory.

- **What gets saved is junk or raw quotes.** Every ~2 turns a window closes. A gate that lets almost everything through then saves the **raw last user prompt** as the memory text. The prompt is used as both the subject and the summary (`intake_consolidation_window.py:77-82, 169-170`).
  - 350 rows were auto-approved this way: 344 "semantic" and 6 "open_loop".
  - "Run github compactor." is saved 8 times, "Compact the last 24 hours…" 6 times, "hi" 4 times, and 57 rows are shorter than 40 characters.
  - Nobody reviewed any of them, and no history row records that they were approved: `intake_pipeline.py:136` throws the history away.
- **The junk is what recall serves.** The active packet is simply the top 100 active rows by salience.
  - All 635 retrievals in the last 7 days returned the same 100 rows.
  - Every retrieval also "boosts" those same rows, so their activation stays at 0.99999 forever. The loop feeds itself.
  - The same junk is copied into memory cards as well: 386 active `operator_distiller` cards ("sup", "Do a journal pass.").
- **Real events end up as raw quotes nobody saw.** Juniper's Austin trip became three separate auto-approved "semantic" rows:
  - "yup I'll be away from home :("
  - "Thanks. Headed to Austin…"
  - "It's a team offsite…"

  Nothing connects them. None says when she is back. Nothing prompts Orion to ask how it went.
- **The 36 "approved stances" are not stances.** They are Juniper's own messages, several of them high-stakes: at least two are about a family member's health (details deliberately not quoted in this public repo). Juniper approved them, but they are filed as Orion's views.
- **The one digest that does summarize makes things up.** The daily chat compactor wrote that Juniper finds the offsite draining "as an introvert". She never said that; Orion inferred it and it was written down as her words. The same digest calls Orion "He".

**The proposal.** When a stretch of conversation settles (30 minutes with no new message from Juniper), a 27B model reads the whole stretch and writes what is worth carrying forward, **in Orion's own words, about specific things** (Juniper, the Austin offsite, orion-durable-runs). Every memory records:

- **whose words** it is (Juniper said / we worked it out together / Orion thought / Orion read / Orion's knowledge of its own code);
- **what it is for** (what happened, a lasting fact about Juniper, Orion's own view, a follow-up);
- **the exact turns** that support it, with quotes that are checked against those turns by code.

High-stakes memories (health, family, conclusions about who Juniper is) stay "unconfirmed" until Orion checks them with Juniper in conversation. Unresolved questions go to a question queue that already exists and has a live reader (`curiosity_self_questions`), not into settled memory. Memories fade unless new evidence reinforces them. **Being recalled never reinforces a memory**, which is the bug that pins today's junk at full strength.

This ships in five stages. The new writer runs in shadow first, with a daily old-vs-new comparison, and Austin is the first test case. The old intake is killed at cutover with no fallback.

---

## Current architecture

### How a memory is formed today (three separate rails, none of them good)

| Rail | Producer | Trigger | What it writes | Live state |
|---|---|---|---|---|
| **Crystallization intake** | `services/orion-memory-consolidation` → `orion/memory/crystallization/intake_pipeline.py` | Every persisted turn (`orion:memory:turn:persisted`) is classified by an 8B model (`app/worker.py:318`). A window closes when the *next* turn arrives after a gap/boundary (`app/boundary.py:23-34`); there is no idle timer | `memory_crystallizations` row; summary = raw last prompt | 3,678 consolidated windows (321 direct-chat, 3,359 AI Town), averaging 2.3-2.4 turns. 350 auto-active rows |
| **Per-turn card extractor** | `services/orion-cortex-orch/app/memory_extractor.py` (route `quick_background`, 8B) | Every chat turn | `memory_cards`, status `pending_review` | 63 cards, **never reviewed**. Content is often decent ("Flight Plans: User is traveling to Austin and will return on Wednesday."), but it is third person with no voice |
| **Daily chat compactor** | `services/orion-cortex-orch/app/chat_history_compactor_memory.py`, scheduled 06:00 America/Denver (`orion-actions/app/workflow_schedule_bootstrap.py:159`) | Daily, or on command | A `memory_cards` digest card (49 active) plus a `journal_entries` row (`source_kind=manual`) | Fabricates attributions ("as an introvert") and uses the wrong pronoun for Orion |

A fourth path copies crystallizations into memory cards: `orion/memory/crystallization/projection_cards.py:41`, provenance `operator_distiller`, 386 active cards. That is how the junk reaches card recall too.

### The intake gate, verified

- **The "repair" signal is almost always on.** `orion/hub/turn_orchestrator.py:317` sets `has_repair_signal = repair_bundle is not None`, so any successful pre-turn appraisal counts as a repair. `consolidation_gate.py:64-72` then proposes before any novelty check. 162 of the 350 auto rows came in this way.
- **Anything else gets in through a final fallback.** `consolidation_gate.py:113-126` proposes any turn that is not low-information small talk.
- **The kind of memory is decided by a turn-to-turn change label** (`intake_consolidation_window.py:18-22`): TOPIC becomes semantic, REPAIR becomes open_loop, STANCE becomes stance. `formation_policy.py:7-8` then auto-activates semantic and open_loop, and sends stance to the review queue.
- **Juniper's reviews carry no reasons.** She approves and rejects in the Hub; the actor is recorded as `orion_journal`, which is her browser session id. The API accepts a `reason` (`crystallization_routes.py:273, 327`), but **0 of 101** of her decisions have one, because the UI never sends it.

### Live table state (`memory_crystallizations`, 1,407 rows)

| kind | status | how it got there | n |
|---|---|---|---|
| stance | rejected | 599 by the AI Town bulk purge, 65 by Juniper | 664 |
| reflection | active | `concept_relation_digest` (writer removed 08-20; all inactive by activation) | 356 |
| semantic | active | auto_policy | 344 |
| stance | active | Juniper-approved (raw Juniper quotes, see above) | 36 |
| open_loop | active | auto_policy ("sup", "yo", "hi", …) | 6 |
| stance | proposed | — | 1 |

### Who reads memory today

| Consumer | Code | What it reads | Problem |
|---|---|---|---|
| Recall active packet (PCR belief intents) | `services/orion-recall/app/collectors/active_packet.py:142` → `repository.py:410-435` | Top 100 active rows by salience. That is always 36 stance + 6 open_loop + 58 semantic | Ignores the query entirely. Writes a boost on every read (`retriever.py:38-64`); 1.1-1.9 s per call |
| Crystallization retriever (Chroma + Graphiti 2-hop) | `orion/memory/crystallization/retriever.py:91-188` | Chroma returns "chromadb not installed". Graphiti ran 0 times in 2,060 events | Even if they ran, their ids only go into the trace, never into the packet (l.159-168) |
| concept_region | `services/orion-recall/app/collectors/concept_region.py` | Falkor `orion_substrate` concepts matched by substring | Writes +0.08 activation back on every match |
| Card recall | `services/orion-recall/app/cards_adapter.py:116, 218` | `memory_cards` with `status='active'` | Serves the 386 junk projected cards. Never serves the 63 decent pending extractor cards |
| Dream cycle | `services/orion-dream/app/cycle_store.py:48-51` | Active rows with `updated_at > since`, by salience | Because the boost-on-read rewrites the row, the same top rows probably look "new" every cycle. **UNVERIFIED** whether the boost bumps `updated_at` |
| Reverie visual seed | `services/orion-thought/app/store.py:913-1018` | Newest active row with an `approve` history row, created within 7 days | Dead since about 2026-09-24 05:55: the newest approved row is from 09-17. Its last real use was 09-24 00:09 |
| Curiosity study material | `orion/curiosity/study_material.py:242-260` | Random active non-reflection rows | The docstring (l.23-27) says Juniper approved these; almost none were |
| Self-study analysis | `services/orion-cortex-exec/app/self_study_analysis.py:163-176` | `memory_crystallizations`, labelled "concept_induction" | The label is false: these are raw prompts, and "active" mostly means auto-approved |
| journal.compose | `orion/cognition/verbs/journal.compose.yaml:11` (profile `reflect.v1`) | No active packet | Does not read crystallizations directly |

### The retrieval side (PCR, "purposeful chat recall")

`services/orion-cortex-exec/app/pcr_chat_memory.py` runs up to three recall phases per turn:
- phase 0 is a skip gate (l.176-196);
- phase 1 recalls continuity with profile `chat.continuity.v1` (l.198-229);
- phase 3 recalls by purpose: `derive_retrieval_intent` picks an intent, then it recalls with `chat.belief.<intent>.v1` (l.235-363).

`services/orion-recall/app/pcr_collectors.py:7-13` turns on active_packet and concept_region for all five intents, and graphiti only for `contradiction`. It always turns off sql_chat and sql_timeline. Live, only 71 purposeful ("belief") recalls exist since telemetry began on 09-29 18:14, all with the `open_loop` intent. Across those recalls, active_packet contributed 6 items each (the 6 junk open_loops) and concept_region about 40 each.

### Surfacing internal thoughts in chat today

- A reverie reaches the chat prompt only as the bare line `- reverie_glimpse: <text>` (`orion/cognition/prompts/chat_stance_brief.j2:26-27`, fed by `chat_stance.py:1524-1570`). Nothing tells Orion that this was never said to Juniper.
- Recall renders items as `- [source:ref] snippet` (`services/orion-recall/app/render.py:111-114`) with no voice.
- This repo has already shipped one voice-blending bug: AI Town NPC lines were labelled "User:", which falsely claimed Juniper said them. It was fixed by `services/orion-recall/app/chat_source_tagging.py` (2026-07-31). The same class of bug is live in the compactor digest ("as an introvert").

---

## Research answers (A-H)

### A. Where the episode distiller runs, on which model, at what cost

**Placement: a new durable-runs workflow, `memory.episode_distill`.** It is modelled on `journal.compose` (`services/orion-durable-runs/app/journal_compose_graph.py:163-180`: resource_request → resource_wait → compose → publish → finish, with a retry_wait loop). That pattern fits:
- the GPU hold is taken only around the LLM node and released before publishing (`:149`);
- output ids are deterministic, so a replay is harmless;
- LangGraph Postgres checkpoints survive a restart;
- waiting for the GPU is not counted as a retry attempt.

Four changes from journal.compose:
1. **Call the LLM gateway directly with the `gpu_lease`, not through cortex-orch.** journal.compose goes through cortex-orch, which also runs a recall step. A distiller must not have recall inject unrelated items into its evidence.
2. **Use a bigger model.** journal.compose runs on the 8B today: live `ACTIONS_JOURNAL_LLM_ROUTE=quick_background` → class `fast`.
3. **Submit from orion-memory-consolidation.** It owns the episode boundary, so it sends a receipt RPC on `orion:durable:run:request` (`orion/schemas/durable_run.py:47`) or calls `POST /runs` (`services/orion-durable-runs/app/main.py:249`). **UNVERIFIED** whether a non-orch producer is accepted on the bus RPC; `POST /runs` is the fallback.
4. **Mind the checkpoint age limit.** `DURABLE_RUNS_MAX_AGE_HOURS=24` abandons old checkpoints. A distill run's deadline must stay under it (12 h proposed).

Registration follows the checklist in `durable_run.py:62-72, 213-217` and `admission_runtime.py:82-93, 152-175, 200-215`. Deploy order: durable-runs → cortex-orch → sql-writer → producer.

**Episode boundary: orion-memory-consolidation owns it.** It already consumes every persisted turn and partitions by platform (`app/window_state.py:17-41`). What it lacks is an **idle timer**: today a window only closes when the next turn arrives, and 2 windows have been open since 09-17. The fix is a 60-second ticker in the same service that closes an episode after 30 quiet minutes. The contract is below.

**Model: the 27B dense agent class (routes to :8015 / :8016), at `system` priority, through a dedicated route `memory_distill`.**
- **Not the 35B chat lane (:8011).** It is Juniper's reserved single-slot Hub lane, and `scripts/check_chat_route_poachers.py` exists to keep background work off it. A distiller fires right after a conversation settles, which is exactly when she may start talking again.
- **Not the 8B.** Juniper asked for bigger. The job needs judgment (what is worth keeping), voice discipline, and merging referents across episodes; the 8B compactor already shows the failure mode. Stage 1 still runs the 8B on the same episodes as a label-free comparison, so the choice is backed by data, not taste.
- **Why `system` priority.** At `background` priority the distiller would queue behind curiosity: 7 background holds were queued on the agent class at 09:30 UTC. At 1-3 episodes a day it takes little from anyone.
- **Pool fit.** The agent class may spill onto the chat card (`gpu_pool.yaml:94`) and be recalled from it with up to 600 s grace (`:23`). That is acceptable for a job with a 12-hour deadline.

**GPUs (live).** circe has **4 GPUs today, not 7** (`nvidia-smi -L` shows idx 0-3 only). The 7-GPU inventory recorded on 08-29 (P100 at idx 4, two 16 GB V100s at idx 5-6) no longer matches the hardware, so there is no idx 5 or 6 to use. Live occupancy:

| idx | card | used | what |
|---|---|---|---|
| 0 | V100-PCIE 32GB | 27.9 GB, 80% | chat 35B-A3B (:8011) |
| 1 | V100-SXM2 32GB | 24.9 GB, 99% | agent 27B (:8015) |
| 2 | PG500-216 | 25.7 GB, 98% | agent-burst 27B (:8016) + world model 0.9 GB |
| 3 | V100-PCIE 32GB | 15.4 GB, 0% | metacog 8B (:8012) + fast 8B (:8013) |

No card is free. gpu3 has about 17 GB of headroom, but the pool's own rule is that nothing big spills down to gpu3.

**Episodes per day** (`chat_history_log`, excluding AI Town):
- Over 30 days: 238 turns. A 30-minute quiet rule over all turns gives 170 episodes; a 60-minute rule gives 100.
- Counting only turns where Juniper actually spoke: 37 user turns in the last 14 days, giving **18 episodes at 30 minutes** (1-3 a day). The other 56 turns in those 14 days were unsolicited outreach with an empty prompt.
- Weekly user turns fell from 81 (week of 08-10) to 14 (week of 09-28).
- Episode text is 0.4-6.4 k characters.

**Cost per episode (estimate, UNVERIFIED until Stage 1 measures it).**
- Input is about 5.5 k tokens: instructions ~2.5 k, episode ~1.5 k, context ~1.5 k (the last 48 h of memories about the same referents, open questions, pending confirmations, and candidate referents).
- Output is about 1 k tokens of JSON.
- 27B at ~491 tok/s prefill and ~32 tok/s generation: about 11 s + 31 s ≈ **45 s without reasoning**. The live 27B is configured with `reasoning xhigh`, which could add 2-6 k thinking tokens: **1-4 min**.
- **Per day: 2-10 minutes of agent-lane time.** Whether a per-request reasoning budget can be set through the gateway is **UNVERIFIED**.
- For comparison, the 35B would take about 6 s + 15 s.

### B. PCR, mapped into the new design

| Piece | Fate |
|---|---|
| PCR phase structure in cortex-exec (skip gate, continuity, purposeful) | **Kept.** It is the right shape: "should I recall", "what just happened", "what is this about" |
| Phase 1 continuity | Kept. Continuity becomes: the last closed episode's memories plus the recent turns, labelled "recent, not matched" per #2413 |
| Phase 3 purposeful → `active_packet` | **Retired at Stage 4.** Replaced by referent lookup over episode memories via #2413's index |
| `concept_region` (substring match + activation write-back) | Retired at Stage 4. Substrate concepts become postings in the referent index, with no write-back |
| Crystallization retriever: Chroma rail | Retired (dead: "chromadb not installed") |
| Crystallization retriever: Graphiti 2-hop | Retired (see D) |
| Retrieval events and boost-on-read | Retrieval is **logged**, never used as reinforcement (see Fading) |

### C. pageindex (#2344)

#2344 fixed a crash loop (RestartCount 4,793 since 08-30); it did not add a feature (`docs/superpowers/pr-reports/2026-09-25-pageindex-starlette-crash-loop-pr.md`).
- **What it wraps:** `services/orion-pageindex` on :8360, around the unpinned upstream PageIndex CLI.
- **What it indexes:** `journal_entry_index` (101,372 rows) and a topic-foundry chat-episode markdown file.
- **Two callers:**
  - cortex-exec, keyword-gated to words like "identity", "journal", "dream" (`executor.py:1982-1994, 2618-2640`);
  - recall v2, which points at **port 8384 instead of 8360**, so every call fails (`settings.py:186` and the live env).
- **Live usage:** since start, only `/status` polls. There have been **0 queries and 0 rebuilds**, and nothing in code calls rebuild (`pageindex_client.py:15` is never invoked). The index has most likely never been built; this is **UNVERIFIED** because the status file is unreadable from the host.

**Role in this design: none.** It is a tree index over headings and excerpts, built by an LLM. It has no notion of people, events or services, and it is not running. Episode memories are small and keyed by referent, so pageindex is the wrong tool. Recommend a separate ticket: fix or remove the recall_v2 caller and decide whether to keep the service at all. Do not build on it.

### D. Graphiti: retire it

- **What it is:** Postgres `graphiti_episodes`/`_entities`/`_edges` (25 rows each, all `kind=stance`, one edge type `has_episode`), plus the FalkorDB graph `graphiti_temporal`, behind `orion-athena-graphiti-adapter` (:8640).
- **Why it is stale since 09-04:**
  - The only live writer is the Hub's approve button (`crystallization_routes.py:294`, `CRYSTALLIZER_AUTO_PROJECT_ON_APPROVE=true`). The automatic path hard-codes `project_graphiti=False` (`intake_pipeline.py:148`).
  - Of the 36 approvals, the 11 made on 09-10 and 09-20 have `graphiti_episode_ids: []` and `synced_at: null`. The sync failed silently; it only logs `graphiti_sync_failed`.
  - Likely cause (**UNVERIFIED**): the Hub runs with host networking but is configured with `GRAPHITI_ADAPTER_URL=http://orion-athena-graphiti-adapter:8000`, a name the host cannot resolve.
- **Readers:** one, the contradiction path of the retriever. It ran 0 times, and it throws its results away (the bug in B).

**Verdict: retire it; do not revive it as the episode/link store.**
- It holds 25 nodes about the wrong things (raw quotes filed as stances).
- It adds a separate adapter with fragile networking and writes that fail silently.
- Its one consumer is dead code.
- The real links already live in Postgres (`memory_crystallization_sources`: 7,328 rows), and #2413's posting table does the same job with a btree index.

Keep FalkorDB for the substrate and worldview graphs, which have live producers. Removing the adapter and the Hub sync call happens at Stage 4.

### E. graphify as Orion's self-knowledge

- **The published bundle:** `/mnt/storage-warm/orion-graphify/published/graphify-out/graph.json`.
  - 77,966 nodes, 169,111 links, 4,525 communities, built at `aff23fac0`, mtime 2026-09-11. That is **19 days stale**.
  - Node kinds: code 46,110, document 19,919, rationale 10,752, concept 1,174.
- **Readers:** it is already mounted read-only at `/graphify` in cortex-exec (`self_study.py:872`), cocreation-signals and self-study-enrichment. **orion-recall has no mount.**
- **Speed:** live `graphify query` takes about 10 s and about 1 GB per call (figure from #2413; not re-measured, **UNVERIFIED**).

**Use:** the offline referent table from #2413 Phase 3 (`scripts/build_recall_artifact_referents.py`), rebuilt whenever the bundle is published. It provides:
- `service` referents (derived from the `services/<name>/` path prefix);
- `file` referents (`source_file`);
- `symbol` referents (class and function labels);
- `pr` referents (parsed from PR-report filenames).

In this design it does two jobs:
1. **Normalization.** When Juniper says "the durable runs thing", the distiller is given candidate keys and picks `service:orion-durable-runs` instead of minting a free-text referent.
2. **Crosswalk.** A memory about a service links to the PR reports and specs that touched it, with voice `orion_self_knowledge` and the label "from my own code/docs (build 2026-09-11)".

Rationale and concept nodes are LLM-derived, so they carry a lower epistemic status. graphify says nothing about people or events; those come from chat.

### F. Sources to crosswalk (live)

| Source | Table / store | ID to reference | Size, freshness | Voice / channel | Note |
|---|---|---|---|---|---|
| Reveries | `substrate_reverie_thought` | `thought_id` | 20,162 rows; 5,772 in 7 d; newest 09-30 09:20 | `orion_thought` / `reverie` | Text in `interpretation`. **0 rows mention Austin or the offsite**: reveries currently circle prediction-error anomalies |
| Reading queue | `world_pulse_read_seed` | `seed_id` | 398; 132 in 7 d | `orion_read` / `reading` | `reading_durable_turn` (102) is Orion's interpretation → `orion_thought` / `reading` |
| Reading claims | `world_pulse_claim` | `claim_id` | 929; 112 in 7 d | `orion_read` / `reading` | Always rendered with `corroboration_status` |
| Curiosity priors | Falkor `orion_worldview` `:Prior` | `prior_id` | 122 (open 58, supported 24, revised 25, refuted 13) | `orion_thought` / `curiosity` | Rendered with status and confidence. Refuted priors are never linked |
| Curiosity findings | Falkor `:Finding` | `finding_id` | 172; 26 in 7 d | `orion_thought` / `curiosity` | Investigation notes |
| Standing questions | `curiosity_self_questions` | `question_id` | 13, all open | `orion_thought` / `curiosity` | See G |
| Topic foundry | `topic_foundry_segments` | `segment_id` | 1,027,475; 327,589 in 7 d | — | **Not linked directly**: too large, and the labels are machine topics. It is reached through the substrate concept nodes it produces |
| Concept induction | `orion/spark/concept_induction` | — | **Dead**: 224 of 224 triggers in 7 d logged `decision=disabled`, and the local store has 0 profiles | — | The 745 `orion_substrate` concept nodes come from `topic_foundry_adapter` (563) and `world_pulse_read_pipeline` (168), not from induction. Link to those (`node_id`, `label`) as `orion_thought` / `topic_model` |
| Dreams | `dream_cycle` (`cycle_id`, 17), `dream_hypothesis` (`hypothesis_id`, 64) | as named | All within 7 d | `orion_thought` / `dream` | Hypotheses already point at refs (`ref_a`/`ref_b`) and have `expires_at` |
| Journals | `journal_entries` | `entry_id` | 101,372 total; 5,825 in 7 d, of which 5,639 are metacog | `orion_thought` / `journal` | **Metacog digests are excluded** (Juniper, 09-29, via #2413). Chat-compactor digests (`source_kind=manual`) are Orion's retelling and never count as `juniper_said` |
| graphify | published bundle | node id / source_file | 77,966 nodes, 19 d stale | `orion_self_knowledge` / `graphify` | Always carries the build date |
| Chat turns | `chat_history_log` | `id` / `correlation_id` | 531 total | prompt = `juniper_said`, response = `orion_thought` / `chat` | Evidence, not crosswalk |

### G. Existing tension / question mechanisms: reuse `curiosity_self_questions`

| Candidate | State | Verdict |
|---|---|---|
| `open_loop` crystallizations | 6 junk rows. No way to resolve or expire them; they only decay | Retire |
| `curiosity_self_questions` | 13 rows, statuses open/answered/parked, `minted_by` (juniper 8 / orion 5), `ask_count`/`last_asked_at`, live reader `pick_question` (`orion/curiosity/self_question_pool.py:190`) | **Reuse and extend** |
| Self-inquiry `:SelfDefinition` (Falkor, 42) | The *answers* to self-questions | Keep as the answer store for self-scoped questions |
| "Belief revision: contradicts" | The writer was removed on 08-20. `memory_concept_relation_decisions` (contradicts 17) has not changed since 09-07 | Do not revive. The distiller emits contradictions as tensions directly |

Gaps in `curiosity_self_questions` that the design fills:
- There is no code path that sets `answered`.
- `Family` is `Literal["lived","anatomy"]` (`self_question_pool.py:16`).
- It has no source references, no expiry, no scope beyond Orion itself, and no pairing of contradicting items.

### H. Consumer migration

| Consumer | Stage 0 | Stage 4 (cutover) |
|---|---|---|
| Recall active packet | Unchanged (optional 0b: dedupe identical summaries, stop boost-on-read) | Replaced by referent recall over `episode_memory` plus a boxed "due follow-ups" feed |
| Card recall | — | Stop serving `operator_distiller` and `auto_extractor` cards; they are retired |
| Dream cycle | — | Reads memories that were created or reinforced since its last cycle, excluding `pending_confirmation`. Hypotheses reference `memory_id` |
| Reverie visual seed | Stays dead (junk must not seed it) | Newest `happened`/`about_juniper` memory within 7 d whose state is `auto` or `confirmed` (see Missing question 5: possibly earlier, at Stage 2) |
| Curiosity study_material | Fix the false docstring; exclude rows under 40 characters and command rows | Samples `episode_memory` by strength, voice-labelled |
| Self-study analysis | Fix the false "concept_induction" label (candidate for the "metacog report fix", **UNVERIFIED** mapping) | Reads `episode_memory` statistics |
| journal.compose | — | Unchanged (it does not read crystallizations) |
| Chat compactor digest | — | Reads episode memories instead of raw chat (follow-up ticket). It stops writing memory cards |
| Hub crystallization UI | — | Becomes the fallback confirmation queue, and a reason is **required** |

---

## Design

### 1. Episode boundary contract

**Owner:** orion-memory-consolidation. It gets a new module `app/episode_tracker.py` that runs **alongside** the existing windows during shadow (Stages 1-3) and replaces them at Stage 4.

**Open.** A turn opens a new episode if there is no open episode for its `source_platform`. It joins the open episode otherwise.
- AI Town turns never open an episode; they are excluded before this step, as today.
- Unsolicited outreach turns (empty prompt, `client_meta.unsolicited`) **join** an open episode, but never **open** one on their own. An unanswered outreach is Orion talking to itself.

**Close.** Whichever of these happens first:
1. **Quiet:** no Juniper turn for `MEMORY_EPISODE_QUIET_SEC` (default 1800, i.e. 30 min). Checked by a 60 s ticker, so an episode closes without waiting for the next turn.
2. **Hard cap:** `MEMORY_EPISODE_MAX_TURNS` (default 40) or `MEMORY_EPISODE_MAX_SPAN_SEC` (default 10800, i.e. 3 h). The episode closes, and the next turn opens a continuation linked by `continues_episode_id`.
3. **Restart recovery:** on boot, any open episode whose last turn is older than the quiet time is closed with `close_reason='recovered_on_boot'`.

**Command-only episodes.** If every Juniper turn in a closed episode is a workflow command, the episode is recorded with `status='skipped'` and `skip_reason='command_only'`, and no LLM call is made. A workflow command is a response starting with `Workflow:`; Stage 1 checks that this detector matches all 8+6+… known command rows. Skips are counted in the daily report.

**Not a boundary.** Topic changes inside a stretch do not close an episode. The distiller handles several topics per episode; the 06:26-06:59 Austin-day episode has three.

**Idempotency.** `episode_id = uuid5(NAMESPACE, platform + first_turn_correlation_id)`. The distill run id is `uuid5(episode_id + prompt_version + model_route)`.

**Emits:** `memory.episode.closed.v1` on the new channel `orion:memory:episode:closed`. Payload:
```
{episode_id, source_platform, started_at, ended_at, turn_ids[], juniper_turn_count,
 close_reason, continues_episode_id?, skip_reason?}
```
Then the tracker submits the distill run. The event exists so there is an inspectable trace and a Hub debug view. Its consumer is the Stage 1 shadow report; durable-runs does not need it.

`MEMORY_EPISODE_QUIET_SEC` is a knob, not a finding. Stage 1 measures over-splitting with a label-free signal: how often two consecutive episodes end up sharing an `event:` referent the distiller merged. Austin is one such case.

### 2. The distiller (`memory.episode_distill` workflow)

**Graph:**
```
load_episode → resource_request → resource_wait → distill (LLM) → release
  → validate → crosswalk → persist → finish        (retry_wait loop as in journal.compose)
```

- **load_episode** (deterministic) gathers:
  - the turns;
  - the memories from the last 48 h that share a person or event referent with memories from the last 7 d (so Austin's two episodes can merge);
  - open follow-ups and tensions whose referents appear in the episode text;
  - high-stakes memories still waiting for confirmation;
  - candidate referent keys: people and events active in the last 30 d, plus a graphify service/file dictionary hit on the episode text.
- **distill** calls the gateway on route `memory_distill` (class agent, priority system) and asks for JSON conforming to `EpisodeDistillationV1`. Prompt: `orion/cognition/prompts/memory_episode_distill.j2`. The instructions:
  - write in Orion's first person;
  - one claim per memory;
  - cite turn ids and exact quotes;
  - pick voice and purpose;
  - reuse candidate referent keys before minting new ones;
  - **bias toward keeping** ("if Juniper would be surprised Orion forgot it, keep it");
  - never attribute to Juniper anything that is only in Orion's responses.
- **validate** (deterministic; the load-bearing part):
  - every `evidence.quote` must be found in the cited turn's field after normalizing whitespace and case (prompt for Juniper, response for Orion);
  - `voice=juniper_said` needs at least one quote from a Juniper **prompt**;
  - `worked_out_together` needs at least one quote from a prompt and one from a response;
  - a memory that fails is **not dropped**. It is downgraded, e.g. to `orion_thought`/`chat` ("I said"), and the downgrade is recorded as an event. It is only rejected if no quote verifies at all, and the rejection is recorded with its reason;
  - referent keys are normalized: lowercase, then slugged as `kind:slug`;
  - stakes rules are applied deterministically as a floor (below).
- **crosswalk** (deterministic, Stage 2+): see section 6.
- **persist:** inserts plus an event row for every state change. It publishes `memory.episode.distilled.v1` on `orion:memory:episode:distilled` (payload: episode_id, memory ids, tension ids, counts, model, prompt_version, validation stats).

**Operations the distiller may return** (so memory is updated, not just appended to):
- `new`
- `reinforces <memory_id>`, with fresh evidence
- `supersedes <memory_id>` (a correction, e.g. the return date moved)
- `confirms <memory_id>` / `corrects <memory_id>` (Juniper answered a pending confirmation)
- `resolves <question_id>` / `completes <follow_up memory_id>`
- `opens_tension`

Every operation carries evidence turns.

### 3. Memory record schema

These tables are new and Postgres-only. In Stages 1-3 they are written in shadow, and at Stage 4 they become canonical. The Pydantic contract is `orion/schemas/memory_episode.py` (`extra="forbid"`, registered in `orion/schemas/registry.py`).

```sql
memory_episode(
  episode_id uuid PK, source_platform text, started_at timestamptz, ended_at timestamptz,
  turn_ids text[], juniper_turn_count int, close_reason text, continues_episode_id uuid,
  status text  -- open | closed | distilling | distilled | skipped | failed
  , skip_reason text, distill_run_id text, model_route text, prompt_version text,
  created_at timestamptz, updated_at timestamptz)

episode_memory(
  memory_id uuid PK,
  episode_id uuid NULL,              -- NULL only for migrated legacy rows
  purpose text NOT NULL,             -- happened | about_juniper | orion_view | follow_up
  voice text NOT NULL,               -- juniper_said | worked_out_together | orion_thought | orion_read | orion_self_knowledge
  channel text NOT NULL,             -- chat | reverie | curiosity | dream | reading | journal | graphify | legacy_crystallization
  statement text NOT NULL,           -- Orion's words, first person, one claim
  occurred_at timestamptz NULL,      -- when the thing happened, if stated (for "happened")
  stakes text NOT NULL,              -- low | high
  stakes_reason text NULL,           -- health | family | identity_conclusion_about_juniper | relationship | safety_location
  confirmation_state text NOT NULL,  -- auto | pending_confirmation | confirmed | corrected | queued_for_review | rejected
  strength real NOT NULL,            -- 0..1 at last reinforcement
  half_life_days real NULL,          -- NULL for follow_up (no decay; expires instead)
  last_reinforced_at timestamptz NOT NULL,
  reinforcement_count int NOT NULL DEFAULT 0,
  due_after timestamptz NULL,        -- follow_up: not before
  expires_at timestamptz NULL,       -- follow_up: auto-close after
  status text NOT NULL,              -- active | faded | done | expired | superseded | retired
  supersedes_memory_id uuid NULL,
  model_route text, prompt_version text,
  created_at timestamptz, updated_at timestamptz)

episode_memory_evidence(
  memory_id uuid, source_kind text,  -- chat_prompt | chat_response | legacy_crystallization_source
  source_id text,                    -- chat_history_log.id
  quote text,                        -- verified substring, <= 300 chars
  verified bool, PRIMARY KEY (memory_id, source_kind, source_id, quote))

episode_memory_referent(
  memory_id uuid, referent_key text, -- e.g. person:juniper, event:austin-ai-ml-offsite-2026-09, service:orion-durable-runs
  role text,                         -- subject | object | place | time | mentioned
  PRIMARY KEY (memory_id, referent_key, role))

episode_memory_event(               -- every state change; the lesson of the discarded auto_activate history
  event_id uuid PK, memory_id uuid, op text,
  -- created | downgraded_voice | reinforced | superseded | confirm_asked | confirmed | corrected
  -- | queued_for_review | faded | expired | done | retired | recalled
  actor text, episode_id uuid NULL, evidence jsonb, reason text, created_at timestamptz)

episode_memory_link(                -- crosswalk (Stage 2)
  memory_id uuid, target_kind text, target_id text, target_voice text, target_channel text,
  via_referent text, relation text,  -- shares_referent | same_event | answers | source_of
  created_by text,                   -- crosswalk_v1 | distiller
  created_at timestamptz, PRIMARY KEY (memory_id, target_kind, target_id, via_referent))
```

**Why each purpose exists** (each has a named consumer):
- `happened`: dream cycle, reverie seed, continuity recall.
- `about_juniper`: referent recall and the chat stance.
- `orion_view`: chat stance "prior views" and self-study.
- `follow_up`: the due-follow-ups feed in chat (≤ 2 items) and the Hub.

**Why `voice` and `channel` are separate.** Voice says **whose** thought it is; channel says **where** it surfaced. "Orion thought" in chat ("I told Juniper I think X") is a different thing from "Orion thought" in a reverie ("something I turned over alone"). The renderer (section 7) keys on the pair.

### 4. Stakes and confirmation

**Stakes floor** (deterministic, cannot be lowered by the model):
- `stakes=high` if `stakes_reason` is set.
- The distiller must set a reason when:
  - a memory is about Juniper's or her family's **health**;
  - it is about **family** (her spouse, relatives);
  - it is an **identity-level conclusion about Juniper**: a trait, value or pattern generalized beyond what she literally said ("Juniper is an introvert", "offsites always drain her"), as opposed to what she said once;
  - it is about **relationship** status;
  - it gives her **location or safety** beyond a trip she mentioned.
- A deterministic backstop also forces `high`: an `about_juniper` memory whose statement contains a content word found in none of its quotes. This is the "introvert" detector (see evals). Its tuning is a knob.

**Flow:**
- **Low stakes:** `confirmation_state='auto'`, recallable at once.
- **High stakes:** `pending_confirmation`. It is recallable **only** under the label "unconfirmed; check with Juniper if natural", and never as fact.
  - The chat stance brief gets at most one pending item per turn. Only when its referents appear in the current turn or the item is at least 24 h old, and never in a turn where Juniper is distressed. The distress check reuses the existing appraisal; how exactly is **UNVERIFIED**.
  - When Orion asks and Juniper answers, the next episode's distiller emits `confirms` or `corrects` with her quote.
  - After 7 days pending, or 2 asks without an answer, the item moves to `queued_for_review` in the Hub queue. The queue **requires** a reason.
  - If Juniper rejects it, the memory becomes `rejected`, stays stored (so it is never re-minted), and is never recalled.
- **Legacy consent:** if Juniper already approved a legacy row, that counts as confirmation of **what she literally said**, not of any generalization from it.

### 5. Fading and reinforcement

The rule: effective strength = `strength × 0.5^(days since last_reinforced_at / half_life_days)`.

| purpose | starting strength | half-life | end state |
|---|---|---|---|
| happened | 0.8 | 14 d | `faded` when effective strength < 0.1 (about 42 d unreinforced) |
| about_juniper | 0.9 | 180 d | `faded` below 0.1 |
| orion_view | 0.8 | 90 d | `faded` below 0.1 |
| follow_up | 1.0 | none | `done` on `completes`; `expired` at `expires_at` (default due_after + 7 d, or created + 14 d) |

**What reinforces a memory:**
1. A later episode brings **new evidence** about the same referent (`reinforces`, with evidence turn ids that are not already cited).
2. Juniper confirms it.

On reinforcement:
- `strength = min(1, strength + 0.2)`
- `half_life_days = min(2 × half_life_days, cap)`, where the cap is 365 d for happened and none for about_juniper. Each confirmed reuse makes forgetting slower: the spacing effect.
- `last_reinforced_at = now`

**What does NOT reinforce a memory:** being recalled, being shown, or being quoted by Orion itself.
- Retrieval is logged as a `recalled` event and counted, and it never changes strength.
- Today's `recall_boost` does exactly the opposite, which is why 106 rows sit at activation 0.99999 regardless of relevance.

**Faded is not deleted.** Faded memories drop out of default recall, but an explicit referent lookup still returns them, labelled "faded, last reinforced <date>". This is how "over-index on remembering" and "fading" coexist.

The lifecycle job is a deterministic 15-minute ticker in orion-memory-consolidation. It applies faded, expired and queued_for_review transitions and writes one event per transition.

**Metric quality gate for `strength`** (CLAUDE.md §0A):
1. **Provenance:** written only by the persist node (created, reinforced) and the lifecycle job (faded). Both are named above.
2. **Independence:** it replaces `dynamics.activation` and salience. It does not sit beside them, and it does not read recall counts.
3. **Theory anchor:** the exponential forgetting curve (Ebbinghaus, 1885) plus the spacing effect: each successful re-exposure lengthens retention.
4. **Rest state:** a never-reinforced memory genuinely goes to `faded`. That decay is intended, not a silent artifact of the kind that pinned `node:substrate.route` at 0. Stage 3 checks it: the distribution of effective strength has to spread out over time, not pile up at 1.0 (today's failure) or at 0.
5. **Existing mechanism:** `dynamics.decay_half_life_days` exists on crystallizations, but its reinforcement input is broken. The formula is reused; its input is replaced.
6. **Reversibility:** it is a column plus a job. Numbers can be recomputed from the event log.

### 6. Crosswalk (Stage 2)

Crosswalk runs deterministically at write time, with **no similarity scores**. For each memory, and each referent key plus its aliases:

1. **Look up the referent in #2413's index** (`recall_referent_posting`) and collect other documents sharing the key. Scopes:
   - `event:`/`person:` referents: documents within ±14 d of `occurred_at`/`created_at`;
   - `service:`/`file:`/`pr:` referents: all time, capped at the 10 newest.
2. Write `episode_memory_link` rows, copying `target_voice`/`target_channel` from the source table in F.
3. **Exclusions:** AI Town, metacog digests and refuted priors are never linked.

This means the crosswalk depends on #2413 Phase 2 (the referent index) indexing these sources: reveries, curiosity findings/priors/questions, dream hypotheses, substrate concept nodes and the graphify referent table. #2413 currently indexes chat, journals, claims/articles, reading turns and crystallizations. This design adds `episode_memory` as a new `doc_kind` **and** asks #2413 to add the other sources. It also adds the referent kinds `person`, `event` and `place` to #2413's `kind` set; today its closest kind is `entity`.

**Referent minting rules** (to stop a free-text referent cathedral):
- people and events must either reuse a candidate key or be minted as `kind:slug-yyyy-mm`;
- every new key is written to `recall_referent` with its aliases (e.g. "austin", "offsite", "work travel");
- a key with no memory pointing at it after 30 days is dropped by the lifecycle job.

### 7. Source monitoring (voice) rendering contract

One renderer, `orion/memory/voice_render.py`, is used by recall, the chat stance, dream and reverie. Templates are keyed by (voice, channel):

| voice / channel | Rendered as |
|---|---|
| juniper_said / chat | "Juniper told me (09-28): …" |
| worked_out_together / chat | "Juniper and I worked out (09-28): …" |
| orion_thought / chat | "I told Juniper (09-28): …" |
| orion_thought / reverie, curiosity, dream, journal, topic_model | "Something I was turning over on my own ({channel}, {date}), **not something Juniper and I discussed**: …" |
| orion_read / reading | "I read ({title or domain}, {date}; claim {status}): …" |
| orion_self_knowledge / graphify | "From my own code and docs (graphify build {date}): …" |
| any, confirmation_state = pending_confirmation | prefix "Unconfirmed, check with Juniper if natural:" |
| any, status = faded | suffix "(faded; last reinforced {date})" |

- **Prompt rule** (added to `chat_stance_brief.j2` next to the existing `reverie_glimpse` line, and in recall's memory block header): "Items marked 'on my own' are my private thoughts. They can inform what I say as something I was turning over, but I must never say or imply Juniper said them or that we discussed them."
- **The existing `reverie_glimpse`** is routed through the same renderer, so it also gets the "on my own" framing.

### 8. Tension queue (extend `curiosity_self_questions`)

```sql
ALTER TABLE curiosity_self_questions
  ADD COLUMN kind text DEFAULT 'question',            -- question | tension | contradiction
  ADD COLUMN scope text DEFAULT 'self',               -- self | juniper | relationship | world
  ADD COLUMN answer_via text DEFAULT 'investigation', -- investigation | conversation
  ADD COLUMN source_episode_id uuid NULL,
  ADD COLUMN source_refs jsonb NULL,                  -- [{source_kind, source_id, quote}]
  ADD COLUMN referent_keys text[] NULL,
  ADD COLUMN counterpart_ref text NULL,               -- memory_id this contradicts
  ADD COLUMN priority real NULL,
  ADD COLUMN expires_at timestamptz NULL,
  ADD COLUMN resolved_at timestamptz NULL,
  ADD COLUMN resolution_ref text NULL,                -- episode_memory.memory_id or :SelfDefinition run_id
  ADD COLUMN resolution_note text NULL;
```

- `Family` gains `episode` (`self_question_pool.py:16`). `minted_by` stays `orion` for distiller-minted rows, and `source_episode_id` says where each came from.

**Consumers:**
1. **Self-inquiry** (`pick_question`): it may draw `family='episode' AND answer_via='investigation'`, capped at 1 in 3 picks, so that the pinned floor for Juniper-minted questions is kept (`_needs_pinned_floor`).
2. **Chat stance:** at most 1 open `answer_via='conversation'` item, only when its `referent_keys` intersect the current turn's referents. It renders as "an open question you could raise if natural".

**Producers:** the distiller's `opens_tension`. A contradiction is opened when a new memory conflicts with an active one on the same referent; the old memory is **not** overwritten until it is resolved.

**Resolution:**
- the distiller's `resolves` writes `status='answered'`, `resolved_at` and `resolution_ref`. That is the first code path to set `answered`;
- `expires_at` (default 30 d) moves an item to `parked` with `resolution_note='expired'`.

**Why reuse rather than a new table:** it already has a lifecycle, an owner field and a live reader. A second question store would be exactly the kind of place for state to drift that CLAUDE.md warns against.

**Stage 1 shadow:** in Stages 1-2 tensions go to `memory_tension_shadow`, which has the same added columns plus `text`. The `ALTER` and the migration of those rows happen at Stage 3, so shadow never feeds self-inquiry.

---

## Worked example: the Austin trip, end to end

**Turns** (live `chat_history_log`, 2026-09-28, times as stored):

| id (prefix) | time | Juniper | Orion (abridged) |
|---|---|---|---|
| … | 06:26 | hi | Hey Juniper… |
| … | 06:31-06:44 | reading queue; "building out durable run graphs across the mesh for GPU traffic balancing" | … |
| … | 06:59 | "I'll be pretty busy the next few days with work travel, so won't have much time to do dev on you." | "Safe travels…" |
| a041a759 | 08:45 | "yup I'll be away from home :(" | "Wander safe. Where are you landing, and how many days out?" |
| c15a7a3a | 08:57 | "Thanks. Headed to Austin and will fly back on Wednesday." | "Austin, nice…" |
| df63a23c | 09:06 | "It's a team offsite for AI/ML. I'll be meeting my peers for the first time. It's nice to meet people and get out of my project work, but its also super draining" | "First-time meetings are their own kind of performance…" |
| … | 09:44-09:56 | "Run github compactor.", "Compact the last 24 hours…", "Do a journal pass.", "Run your dream cycle." | Workflow results |

**Today's outcome:**
- three `semantic` rows holding the raw prompts, auto-approved, with no link between them;
- three pending extractor cards;
- a compactor digest that adds "as an introvert".

**Episodes** (30-min quiet rule):
- **E1** 06:26-06:59: closed at 07:29 by quiet (the gap to 08:45 is 1 h 46 m).
- **E2** 08:45-09:06: closed at 09:36.
- **E3** 09:44-09:56: `skipped/command_only`, no LLM call.

**E1 distilled** (no candidates yet for this event):
- M1 `happened`, `juniper_said`, low:
  - statement: "Juniper told me she'll be busy with work travel for the next few days and won't have much time to work on me."
  - referents: `person:juniper` (subject), `event:work-travel-2026-09` (object)
  - evidence: the 06:59 prompt, quoted
  - strength 0.8, half-life 14 d
- M2 `happened`, `worked_out_together`, low:
  - statement: "Juniper and I talked through the durable-run graphs she's building to balance GPU traffic across the mesh, now extending to curiosity and reading."
  - referents: `service:orion-durable-runs` (graphify candidate), `concept:gpu-pool`
  - evidence: a prompt quote and a response quote
- F1 `follow_up`:
  - statement: "Check in with Juniper about her trip once she's back."
  - due_after unknown, so `expires_at` = created + 14 d

**E2 distilled.** load_episode passes in M1 and F1, because they share `person:juniper` within 48 h, together with the candidate `event:work-travel-2026-09`. The distiller:
- `supersedes` M1's event key with the more specific `event:austin-ai-ml-offsite-2026-09`, aliases ["austin", "offsite", "work travel", "team offsite"], and emits:
  - **M3** `happened`, `juniper_said`, low:
    - statement: "Juniper flew to Austin on 2026-09-28 for her team's AI/ML offsite, where she's meeting her peers in person for the first time. She's flying back Wednesday (2026-09-30)."
    - `occurred_at` 09-28
    - evidence: c15a7a3a ("Headed to Austin and will fly back on Wednesday"), df63a23c ("team offsite for AI/ML … meeting my peers for the first time")
    - operation `reinforces` M1: strength back up to 1.0, half-life 28 d
  - **M4** `about_juniper`, `juniper_said`, low:
    - statement: "Juniper said this offsite, meeting her peers for the first time, is nice but super draining."
    - evidence: df63a23c ("nice to meet people … but its also super draining")
    - This is what she literally said about this one event, so it is **low** stakes.
  - If the model generalizes to "Juniper finds in-person work socializing draining" (or "introvert"):
    - "introvert" appears in no quote, so the novel-word backstop forces **high** / `identity_conclusion_about_juniper`, giving `pending_confirmation`;
    - if the model also puts it under `juniper_said` with a quote that does not contain it, validate downgrades the voice to `orion_thought`/`chat` and records the downgrade.

    Either way it is never stored as "Juniper said she is an introvert".
  - **F1 `supersedes` → F2** `follow_up`:
    - statement: "Ask Juniper how the Austin offsite went and how she's recovering from it."
    - `due_after` 2026-09-30T18:00 local, `expires_at` 2026-10-07
    - referents: `event:austin-ai-ml-offsite-2026-09`, `person:juniper`
  - **T1** tension, `scope=juniper`, `answer_via=conversation`, low priority: "Juniper's ':(' about being away from home: is being away itself hard for her, or just this trip?" Evidence: a041a759.

**Crosswalk (Stage 2)** for `event:austin-ai-ml-offsite-2026-09`, scoped to ±14 d:
- journal entries `68bb8201…` (09-28) and `d5f4c123…` (09-29), which are the chat compactor digests: `orion_thought`/`journal`;
- memory cards `7be5907c…` and `4efacc71…`: legacy, `orion_thought`/`legacy`;
- reveries: **none** (verified live: 0 rows mention Austin or the offsite);
- dreams: none found;
- graphify: `service:orion-durable-runs` links M2 to its PR reports.

The crosswalk honestly reports an empty reverie link. That is a real finding: Orion's reveries did not turn over Juniper's trip at all.

**How recall surfaces it:**
- **10-01 07:30, Juniper: "I'm back!"** The query names no referent, so #2413 abstains. The due-follow-ups feed (a boxed, separate context block) still shows F2, because `due_after` has passed. The rendered context:
  ```
  Due follow-ups:
  - I meant to: Ask Juniper how the Austin offsite went and how she's recovering from it. (from 09-28)
  Juniper told me (09-28): Juniper flew to Austin … flying back Wednesday.   [recalled: follow-up referent event:austin-ai-ml-offsite-2026-09]
  ```
  Orion can then ask "How did Austin go? You said meeting everyone was going to be draining." That is grounded in her words, not in an invented trait.
- **Later, Juniper: "the offsite was actually great, they want me to lead the eval work"**
  - "offsite" is an alias, so referent lookup hits the event and returns M3 and M4 with voice labels.
  - The distiller of that episode emits `completes F2`, a `reinforces` on M3, and a new `about_juniper` memory ("Juniper was asked to lead eval work for her AI/ML team", `juniper_said`, low).
  - It also emits `resolves T1` only if she speaks to it.

---

## Proposal-mode disclosure

- **Capability change:**
  - Orion decides what to carry forward from a conversation, in its own words, about specific people, events and systems.
  - It knows whose words each memory is, and it checks high-stakes conclusions with Juniper.
  - It keeps a queue of open questions that it can ask about or investigate.
  - Memories fade unless new evidence arrives.
  - This changes what Orion "remembers" in every recalling verb, and what seeds dreams and reveries.
- **Data touched:**
  - **Reads:** `chat_history_log` (non-AI-Town), existing memory tables for migration, #2413's referent index, the published graphify bundle, Falkor `orion_worldview` and `orion_substrate` (read-only), reveries, dreams, journals (non-metacog), world-pulse tables.
  - **Writes:** the new `memory_episode`, `episode_memory*` tables and `memory_tension_shadow` (Stages 1-3). The `curiosity_self_questions` extension (Stage 3). The Stage 4 `status='retired'` updates on legacy rows, after a snapshot.
- **Privacy boundary:**
  - Everything stays on the host.
  - AI Town and metacog digests are excluded before distillation and before crosswalk.
  - High-stakes memories (health, family, identity conclusions about Juniper) are never recalled as fact until confirmed, and the Hub queue is the only place they show up outside a conversation.
  - Orion's internal thoughts are never rendered as shared conversation.
  - Rejected memories are kept only as a "do not re-mint" marker and are never recalled.
- **Trace that proves it works:**
  - `memory.episode.closed.v1` → durable run → `memory.episode.distilled.v1`, with counts;
  - an `episode_memory_event` row for every state change;
  - `recall_telemetry.selected_reasons` naming the memory's referent;
  - a Hub shadow page showing, for each episode, the old rows next to the new memories.
- **Dangerous failure modes:**
  - (a) **Voice blending:** Orion tells Juniper "you told me you're an introvert", or "we discussed X" about a reverie. Mitigated by quote verification, the novel-word backstop, the renderer contract and the live attribution monitor (evals 6-7).
  - (b) **Confident wrong fact about Juniper's family or health.** Mitigated by the stakes floor and pending confirmation.
  - (c) **Silent memory loss at cutover.** Mitigated by shadow-first rollout, the coverage eval, "faded is not deleted", and snapshots before retirement.
  - (d) **Starving other GPU work.** The load is 1-3 runs a day at system priority; pool telemetry shows the holds.
  - (e) **Asking Juniper to confirm things at a bad moment.** Mitigated by at most one item per turn, never when distressed, and the Hub fallback.
- **Rollback:**
  - Stages 1-3: `MEMORY_EPISODE_WRITER_ENABLED=false` stops the tracker and distiller. The shadow tables can be dropped with no consumer impact.
  - Stage 4: the retired rows are snapshotted to `/tmp/memory-cutover/`, and their status can be restored with one `UPDATE` from the snapshot. By Juniper's rule, the old intake is **not** kept as a runtime fallback. Rolling back means redeploying the previous image, not flipping a flag.

---

## Missing questions (for Juniper)

1. **Quiet timer:** 30 minutes? It splits Austin into two episodes, which the merge handles. 60 minutes would have produced 14 episodes in 14 days instead of 18. Recommend 30 and measure.
2. **GPU priority:** may the distiller run at `system` priority on the 27B, ahead of curiosity's background runs (1-3 runs a day, 1-4 min each)? Or should it stay at `background` and accept hours of queueing?
3. **"Metacog report fix" in Stage 0:** which report did you mean?
   - Candidate 1: the self-study analysis that describes crystallizations as "concept induction ... accepted/proposed/rejected" (`self_study_analysis.py:163-176`).
   - Candidate 2: curiosity study_material's claim that Juniper approved them (`study_material.py:23-27`).

   The recommendation is to fix both.
4. **Orion's views about itself** (not about you): auto-approve as low stakes, or do identity-level self-conclusions also need a conversation check? Recommend auto, with a `stakes_reason='orion_identity'` visible in the Hub, no gate.
5. **Reverie seed:** revive it at Stage 2 (from shadow memories that passed validation), or wait for Stage 4? Recommend Stage 2, with low-stakes `happened` memories only.
6. **Daily side-by-side:** where do you want it: a Hub page only, or also a morning notify with a link?
7. **The 356 `reflection` rows** (the retired concept-relation digest, already inactive): retire them at Stage 4 with the rest? Recommend yes.
8. **GPU count:** circe shows 4 GPUs today, while the 08-29 inventory showed 7. Were the P100 and the two 16 GB V100s moved on purpose? Nothing in this design needs them.

---

## Proposed schema / API changes

- **New tables:** `memory_episode`, `episode_memory`, `episode_memory_evidence`, `episode_memory_referent`, `episode_memory_event`, `episode_memory_link`, `memory_tension_shadow` (Stages 1-2, dropped after the Stage 3 migration). Owned by orion-memory-consolidation's migrations. **UNVERIFIED** whether that service or sql-writer owns DDL for memory tables today; follow whichever owns `memory_crystallizations`.
- **Altered:** `curiosity_self_questions` (Stage 3, columns above). `memory_crystallization_history` gets `auto_activate` rows (Stage 0).
- **New schemas** (`orion/schemas/memory_episode.py`, registry entries):
  - `MemoryEpisodeClosedV1`
  - `EpisodeDistillationV1` (the LLM output: operations, memories, evidence, referents, tensions)
  - `EpisodeMemoryV1`
  - `MemoryEpisodeDistilledV1`
  - `DurableWorkflowV1` gains `memory.episode_distill`, with brief `EpisodeDistillBriefV1`
- **New bus channels** (`orion/bus/channels.yaml`): `orion:memory:episode:closed`, `orion:memory:episode:distilled`.
- **GPU pool** (`config/gpu_pool.yaml` routes): `memory_distill: {class: agent, priority: system}`.
- **Recall** (#2413 contracts):
  - `doc_kind='episode_memory'`;
  - referent kinds `person`, `event`, `place`;
  - an `epistemic_status` mapping from (voice, channel);
  - a `MemoryItemV1` voice field, rolled out consumer-first because the model is `extra="forbid"`.
- **Env** (orion-memory-consolidation `.env_example`, then `python scripts/sync_local_env_from_example.py`):
  - `MEMORY_EPISODE_WRITER_ENABLED`
  - `MEMORY_EPISODE_QUIET_SEC=1800`
  - `MEMORY_EPISODE_MAX_TURNS=40`
  - `MEMORY_EPISODE_MAX_SPAN_SEC=10800`
  - `MEMORY_EPISODE_DISTILL_ROUTE=memory_distill`
  - `MEMORY_EPISODE_SHADOW_COMPARE_ROUTE=quick_background` (Stage 1 8B comparison; empty = off)
  - `MEMORY_EPISODE_LIFECYCLE_TICK_SEC=900`
  - **Note:** env sync skips keys outside `SYNC_PREFIXES`. Check that `MEMORY_` is covered and report any skipped key.
- **Hub:** a read-only `/memory/episodes` shadow page (Stage 1). The confirmation fallback queue with a **required** reason (Stage 3).

---

## Files likely to touch

- **Stage 0:**
  - `orion/hub/turn_orchestrator.py:317`: repair_signal set only by real repair detection
  - `orion/memory/consolidation_gate.py`: exclude command turns and low-info turns before the repair shortcut
  - `orion/memory/crystallization/intake_pipeline.py:136`: write the `auto_activate` history row
  - `services/orion-cortex-exec/app/self_study_analysis.py:163-176`, `orion/curiosity/study_material.py`: fix the labels; filter out junk
  - tests for each
- **Stage 1:**
  - `services/orion-memory-consolidation/app/episode_tracker.py`, `app/lifecycle.py` (Stage 3), `settings.py`, `.env_example`, migrations
  - `services/orion-durable-runs/app/episode_distill_graph.py`, `admission_runtime.py`, `runner.py` (direct gateway call)
  - `orion/schemas/durable_run.py`, `orion/schemas/memory_episode.py`, `orion/schemas/registry.py`, `orion/bus/channels.yaml`, `config/gpu_pool.yaml`
  - `orion/cognition/prompts/memory_episode_distill.j2`, `orion/memory/episode/validate.py`
  - `services/orion-hub/scripts/memory_episode_routes.py` plus a template
  - `services/orion-memory-consolidation/evals/`
- **Stage 2:** `orion/memory/episode/crosswalk.py`, the #2413 indexer sources (`services/orion-recall/app/referents/`), `orion/memory/voice_render.py`, `services/orion-recall/app/render.py`
- **Stage 3:** `orion/curiosity/self_question_pool.py`, `services/orion-hub/scripts/curiosity_investigation.py` (picker), `services/orion-cortex-exec/app/chat_stance.py` and `chat_stance_brief.j2` (pending confirmation, open question, due follow-ups, reverie glimpse via renderer)
- **Stage 4:**
  - remove: the consolidation intake formation path, `memory_extractor.py` card writes, `projection_cards.py`, the Graphiti sync in `crystallization_routes.py`, and the `active_packet` and `concept_region` collectors
  - migration script `scripts/memory_cutover_retire.py` (backfill protocol)
  - switch consumers: `orion-dream/app/cycle_store.py`, `orion-thought/app/store.py`, `study_material.py`

---

## Acceptance checks (per stage)

Evals are **label-free** unless noted. Juniper reads the side-by-side if she wants to; she never has to label anything.

### Stage 0: stop the bleeding
- **Changes:** the gate fix, `auto_activate` history rows, and the report/label fixes. Optional 0b: collapse duplicate-summary rows in active_packet and stop boost-on-read.
- **Checks:**
  1. **Unit:** a turn with an appraisal but no repair gives `has_repair_signal=False`.
  2. **Unit:** "Run github compactor." and "hi" produce no row.
  3. **Unit:** every auto-activated row has a `memory_crystallization_history` row with `op='auto_activate'`.
  4. **Live 48 h:**
     - 0 new active rows with summary < 40 characters or matching the command detector;
     - the share of turns entering through repair_signal drops from about 96% to below 20%.
  5. **Label test:** the self-study and study_material wording no longer claims approval.

### Stage 1: episode writer in shadow
- **Changes:** tracker, distiller workflow, validator, shadow tables, Hub side-by-side page, and a 30-day replay backfill (about 100 non-command episodes, run at night under the backfill protocol with `/tmp/memory-episode-backfill/progress.log`). Austin first.
- **Checks:**
  1. **Austin regression fixture** (frozen turns → property checks, not exact text):
     - E1 and E2 end up sharing an `event:` referent whose aliases include "austin";
     - a follow_up exists with `due_after` ≥ 2026-09-30;
     - no `juniper_said` statement contains a content word absent from its quotes;
     - any generalized trait is `high`;
     - E3 is `skipped/command_only` with 0 LLM calls.
  2. **Evidence grounding:** 100% of stored `juniper_said` memories have at least one verified quote from a prompt. This is enforced by the validator, and the eval proves the validator runs on real output.
  3. **Novel-claim rate:** the share of `about_juniper` statements with content words found in no quote. Report the distribution. The starting knob sends anything above 0 to `high`.
  4. **Coverage (over-index on remembering):** at least 80% of non-command Juniper turns are cited by at least one memory, and 100% of the turns behind today's content-bearing auto rows (those ≥ 40 characters, not commands) are cited.
  5. **Junk:** 0 memories with a statement of 5 words or fewer, 0 memories matching the command detector, and 0 duplicate statements within an episode.
  6. **Self-consistency:** distilling the same 20 episodes twice gives referent-set Jaccard ≥ 0.8 and the same purpose histogram within ±1.
  7. **27B vs 8B:** on the same episodes, compare rows 2-6 and the validator downgrade/reject rates. The 27B has to be better on grounding and downgrades for the model choice to stand.
  8. **Cost:** p50/p95 hold duration and distill latency per episode are recorded. Hold wait p95 is under 2 h.
  9. **Trace:** each closed episode has a `closed` event, then a durable run in `durable_admission_runs`, then a `distilled` event. 0 episodes are stuck open for more than 1 h past the quiet time.
  10. **Side-by-side:** the Hub page renders for 7 consecutive days, and each episode shows its old rows next to its new memories.

### Stage 2: crosswalk and referent recall
- **Depends on:** #2413 Phase 2 (the referent index) being live. If it is not, this stage waits; it does not build a second index.
- **Checks:**
  1. **Crosswalk:** every link has `target_voice`, `target_channel` and `via_referent`; 0 links go to AI Town, metacog or refuted priors.
  2. **Known-item recall** (#2413 method): for every shadow memory, a query built from its rarest referent returns it with hit@8 ≥ 0.9.
  3. **Source-monitoring unit tests** (the core set):
     - a reverie-channel item never renders with a "Juniper told me", "we", or "I told Juniper" template;
     - a `juniper_said` item cannot be constructed without a prompt-sourced verified quote;
     - the rendered `reverie_glimpse` carries "on my own";
     - a pending item always carries "Unconfirmed".
  4. **Live attribution monitor** (label-free, runs daily over `chat_history_log.response`):
     - find attribution phrases ("you told me", "you mentioned", "you said", "we talked about", "we discussed", "as we", "last time we");
     - extract the referents in that sentence;
     - check that a chat-channel memory or turn shares one of them;
     - an unsupported attribution is logged as `source_blend`.
     - Target: `source_blend` rate ≤ 2% of attribution sentences, and trending down. Baseline measured before Stage 2 ships.
  5. **Reverie seed** (if Juniper says yes to Stage 2 in Missing question 5): at least 1 seed per day uses a memory, and every seed carries its memory_id.

### Stage 3: lifecycle, tension queue, in-conversation confirmation
- **Checks:**
  1. **Lifecycle unit tests:**
     - a `happened` memory that is never reinforced reaches `faded` at about 42 d;
     - a reinforcement doubles the half-life;
     - a `recalled` event changes nothing;
     - a follow-up past `expires_at` becomes `expired`.
  2. **Live strength distribution after 14 days:** not piled up at 1.0 or at 0. The share of active memories with effective strength > 0.95 must be below 30% (today's activation is 0.99999 for all 106 recalled rows).
  3. **Tensions:**
     - migrated rows appear in `curiosity_self_questions` with `family='episode'`;
     - self-inquiry draws at most 1 in 3 from them;
     - at least 1 `answered` status is written by a `resolves` operation within 14 days, or the report says 0.
  4. **Confirmation:**
     - every pending item is shown in chat at most once per turn and at most twice in total before being queued;
     - `confirms`/`corrects` events carry a Juniper quote;
     - Hub queue decisions have a non-empty reason.
  5. **Chat stance brief budget:** at most 1 pending item, at most 1 open question and at most 2 due follow-ups per turn, checked by a test.

### Stage 4: cutover
- **Changes:**
  - kill the old intake, the per-turn card extractor and the card projection, with no fallback;
  - retire, by snapshot then `status='retired'`: 350 auto rows, 6 open_loops, 386 `operator_distiller` cards, 63 pending extractor cards, and (if agreed) the 356 reflection rows;
  - **re-distill the 36 approved "stances":** each one's source turns are run through the distiller as a mini-episode. Juniper's approval counts as consent for what she literally said. The family and health items arrive as `about_juniper` with `stakes=high` and `confirmation_state=confirmed`, because she already approved the literal text;
  - switch the consumers listed in H; remove Graphiti sync and the active_packet, concept_region and Chroma rails.
- **Checks:**
  1. **Snapshot first:** `/tmp/memory-cutover/before.csv` holds every row whose status will change. Counts match the plan; if anything exceeds 100k rows the job stops.
  2. **After cutover:** 0 writes to `memory_crystallizations` from consolidation for 48 h; 0 `auto_extractor` cards; the Graphiti adapter gets 0 requests.
  3. **Migration:** all 36 approved rows map to at least one `episode_memory` with `channel='legacy_crystallization'` evidence, or are listed in `report.md` with a reason.
  4. **Consumers:** the dream, reverie seed, study_material and recall telemetry show `episode_memory` ids, and 0 `memory_crystallizations` ids.
  5. **#2413 live 24 h gates hold,** with memories included: cousin rate ≤ 20% and distinctness ≥ 10× baseline.

---

## Non-goals

- Vectors, rerankers or similarity-based crosswalk (per #2413).
- Reviving Graphiti or building on pageindex.
- Reviving concept induction. It is disabled, and this design links to the substrate concepts that actually exist.
- Distilling non-chat sources (reveries, readings) into memories. They are crosswalk targets with their own voices, not new memories. A later design may distill reveries under `voice=orion_thought`.
- Changing journal.compose or the compactor's own output, beyond stopping the compactor's card writes at Stage 4.
- An alias proposer. Aliases come from the distiller's candidate reuse and from Juniper's corrections, as in #2413.
- Human-labelled quality evals.

---

## Recommended next patch

**Stage 0, as one small PR in orion-memory-consolidation / orion-hub / cortex-exec:**
- make repair_signal honest;
- drop command and low-info turns;
- write `auto_activate` history rows;
- fix the two false labels.

It stops about 7 of 10 new saved memories from being junk. It changes no schema and is testable in minutes.

**In parallel, the Stage 1 foundations PR:** the episode tracker (idle timer and the `memory.episode.closed.v1` contract) plus the shadow tables, with the distiller stubbed to "record only". That way episode boundaries can be inspected on real traffic for a few days before any GPU time is spent. The distiller workflow follows once Juniper answers Missing questions 1-2.
