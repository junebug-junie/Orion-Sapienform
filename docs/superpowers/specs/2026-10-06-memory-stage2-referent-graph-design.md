# Memory Stage 2: one list of the things Orion knows about, and recall that starts from them

Status: **PROPOSAL** (design mode + proposal mode). Nothing here is implemented.
Date: 2026-10-06
Parent: `2026-09-30-memory-episode-redesign-design.md` (rev 3, APPROVED 2026-10-01), the "Stage 2" rows of its plan.
Builds on: `2026-09-29-recall-semantic-retrieval-pipeline-design.md` (#2413, "recall by referent, not resemblance"). No vectors, no similarity, anywhere in this design.
Evidence base: read-only queries on the `conjourney` Postgres, FalkorDB `GRAPH.RO_QUERY`, the published graphify bundle, LangGraph checkpoints of the three live distill runs, and the code on main @ f0fc91dc1. All taken 2026-10-06. Anything not checked live is marked **UNVERIFIED**.

The repo is public. Memories about Juniper's family, body or whereabouts are counted here, never quoted.

---

## Arsonist summary

Stage 1 works: since 10-02, after each conversation, Orion writes down what is worth keeping, in its own words. Each memory names the specific things it is about (`project:hecate`, `person:vincent`, `place:chicago`). Today nothing uses those names. Recall still serves the same top 100 old rows by salience, whatever Juniper says.

Stage 2 makes those names the backbone of memory:

1. **One list of things.** A Postgres table holds every person, place, machine, project, service, concept and event Orion knows about. Each has its other names: "Hecate", "the Inspur" and "Inspur NF5288M5" all point at one machine. The memory writer proposes the names. Code accepts a name only if Juniper actually said it. When a name could mean two things, Orion asks; it never merges silently.
2. **Links from everything to those things.** Memories, episodes, Orion's reveries, readings, curiosity notes and topic-model concepts are each linked to the things they mention, and each link records whose words it is.
3. **Recall starts from the things a message names.** "How's Hecate doing?" finds Hecate, walks one step to the Hecate memories and their quotes, and returns them with the reason ("you said Hecate") and the voice ("Juniper told me…"). Orion's 180 reveries about Hecate come back labelled as Orion's own thoughts, never as something Juniper said.
4. **Graphiti mirrors the timeline**, so Orion can answer "what did I know about X last week". It runs with no LLM, and Orion sets the validity dates.
5. **The reverie image seed** switches from old crystallizations to validated new memories.

It runs in shadow next to today's recall, with a side-by-side report. It replaces the old recall only when measured gates pass.

**Three live findings shape the design:**
- **The writer already proposes aliases, and Stage 1 throws them away.** The raw 27B output for the Hecate episode carries `aliases: ["Inspur NF5288M5", "AGX-2 GPU", "8x smx2 gpus"]`. But `validate.py:308-315` keeps only `(key, role)`, and `episode_memory_referent` has no alias column. Six referents across the three runs had aliases. All were lost.
- **Some proposed aliases are wrong as identities.** "my boss" for Rachel only means something relative to Juniper. "a Marriott" names a class of hotel, not the hotel. "autonomous robot" for the camera project names a different future thing. So aliases need a grounding rule and a collision path. A plain "accept what the LLM said" rule would not do.
- **The writer files machines as projects** (`project:hecate`, `project:circe`). Its kind list has no `machine` (`validate.py:44`).

---

## Current architecture

### What Stage 1 actually writes (live, 2026-10-06)

| Table | Rows | Notes |
|---|---|---|
| `episode_distill_run` | 3 (10-04 01:26, 10-05 02:56, 10-05 03:53) | 27B Q4 on route `memory_distill`; LLM 35 s / 78 s / 118 s; hold wait 0.5–1.5 s |
| `episode_memory` | 25 | 15 `happened`/`juniper_said`, 7 `about_juniper`/`juniper_said`, 2 `happened`/`orion_thought`, 1 `follow_up`. All `stakes=low`, `auto`, `active` |
| `episode_memory_referent` | 32 rows, 14 distinct keys | No alias column. Indexed on `referent_key` |
| `episode_memory_evidence` | 26 quotes, all `verified=t` | 22 from prompts, 4 from responses. `source_id` = `chat_history_log.id` |
| `episode_memory_event` | 25, all `created` | No supersede, fade or confirm events yet |
| `memory_tension_shadow` | 2 open questions | Both `scope=juniper`, `answer_via=conversation`, with `referent_keys` |
| `episode_memory_link` (rev 3 §2) | **not created** | Stage 2 |

**Referent keys the writer emitted**, with key/role/count:
- `person:juniper` (subject 5, about 3), `project:hecate` (about 5), `person:juniper-sister` (about 3, a family member)
- `event:austin-offsite` (event 2), `event:joker-2-viewing` (event 2), `person:vincent` (participant 2), `place:the-wade` (location 2), `project:circe` (about 2)
- 1 each: `concept:space` (topic), `person:orion` (about), `person:rachel` (participant), `place:chicago` (location), `place:jackalope-bar` (location), `project:orion-camera` (project)

What this tells us:
- **Roles are free text.** Seven distinct values appear: subject, about, participant, location, event, topic, project.
- **Event keys are not dated.** The prompt asks for `event:<slug>-<yyyy>-<mm>`; the writer emitted `event:austin-offsite`.
- **Some named people became no referent.** In "Jay, Jay, Armando, and Vincent", only Vincent got a key.

**Aliases the writer emitted**, recovered from the `answer_text` channel of `checkpoint_writes` for the three `memdistill-*` threads:

| Key | Aliases emitted | Grounded in Juniper's words? |
|---|---|---|
| `project:hecate` | "Inspur NF5288M5", "AGX-2 GPU", "8x smx2 gpus" | Yes. Prompt `ddc979a1…`: "I got us an Inspur NF5288M5 AGX-2 GPU that holds 8x smx2 gpus. We'll call it Hecate" |
| `person:juniper-sister` | "my sister" | Yes, but relative to the speaker |
| `place:the-wade` | "the Wade", "a Marriott" | "the Wade" yes; "a Marriott" is a class (**UNVERIFIED** whether it is in a quote) |
| `event:joker-2-viewing` | "Joker 2" | Yes |
| `event:austin-offsite` | "Austin team", "offsite" | "offsite" yes |
| `person:rachel` | "my boss" | Yes, but relative to the speaker |
| `place:jackalope-bar` | "Jackalope bar" | Yes |
| `project:orion-camera` | "camera", "autonomous robot" | Yes, but "autonomous robot" is a different future thing |
| `event:chicago-simulation` (question only) | "simulation", "client use case" | Generic words |

`validate.py` drops every one of these. The persisted `referents` field is `list[tuple[str, str]]` (`validate.py:103, 308-315`), and `store.py` inserts only `(memory_id, key, role)`.

**The "new server" case.** Juniper never wrote "the new server". Zero chat prompts match `new server|the inspur`. The phrase appears only in Orion's own memory statement ("onboarding processes for the new server", memory `63d7e72b`). So today "the new server" can only resolve through Orion's own words. The design handles that case explicitly (section 1.4).

### Where referents already appear in other stores (live counts)

| Source | Hecate | Inspur | Vincent | Jackalope | Wade |
|---|---|---|---|---|---|
| `substrate_reverie_thought` (Orion's reveries) | **180** (first 10-04 02:10, 41 min after Juniper's message; newest 10-06 02:05) | 1 | 0 | 0 | 0 |
| `journal_entries`, non-metacog | 6 (manual 2, orion_day 2, scheduler 2) | 3 | 2 | 2 | 2 |
| `chat_history_log` | 9 (1 prompt; 8 are Orion's unprompted outreach, with empty prompt) | 1 | 1 | 1 | 3 |

The reveries are Orion turning Hecate over on its own: "the unresolved tension between substrate structure and Hecate signals". They are exactly what must never come back as "Juniper said".

**Topic-model concepts** (Falkor `orion_substrate`, `:Concept`):
- 636 from `topic_foundry_adapter` (newest 10-05 04:10), 201 from `world_pulse_read_pipeline`, 11 from `substrate_runtime_worker`, 4 from `seed_concepts_loader`.
- The labels are topic phrases: "Home server setup", "GPU performance and work", "Disk damage from AC failure".
- Each concept's `evidence_refs_json` holds `<run_id>:topic:<n>`. That joins to `topic_foundry_segments(run_id, topic_id)`, whose `provenance.row_ids` are `chat_history_log` ids.
- Live check: the Hecate turn `ddc979a1…` sits in 4 segments, which map to the concept **"GPU performance and work"** (`sub-concept-topicfoundry-7ed137b5…-6`). This is a real, typed link from a concept to a memory through a shared source turn, with no similarity involved.

**graphify** (published bundle `/mnt/storage-warm/orion-graphify/published/graphify-out/graph.json`):
- Built at `aff23fac0`, file dated 2026-09-11 (25 days stale). 77,966 nodes and 169,111 links.
- It covers 95 `orion-*` service directories, 6,107 source files and 530 PR-report files. 4,828 of the files have a basename that is unique in the bundle; `__init__.py` (236), `README.md` (106), `settings.py` (83) and `main.py` (81) are not unique.
- Its 1,174 `concept` nodes are mostly schema field names ("name", "category", "priority"). They are not referents.
- `main` has 2,377 "Merge pull request #N" commits.

**Rarity of names in Orion's own words.** The corpus is 574 non-AI-Town chat turns + 2,748 non-metacog journals + 1,029 world-pulse claims = 4,351 documents. Document frequency over chat and journals: hecate 15, inspur 4, chicago 13, offsite 14, server 27, camera 38. At #2413's df ≤ 0.5% cut (≤ 21 docs), "hecate" and "inspur" count as rare and "server" and "camera" do not. That is why "the new server" needs an alias or a phrase rule. A rare-term rule alone will not catch it.

### Recall today (live, 7 days)

- **The active packet is still query-blind.** `memory_crystallization_retrieval_events`: 569 retrievals, every one exactly 100 ids, only 103 distinct ids across them.
- **The intent is still always `open_loop`.** `recall_telemetry` for `stance_react`: 636 `chat.continuity.v1` (Phase 1) and 608 `chat.belief.open_loop.v1` (Phase 3). Zero relational, semantic, procedural or contradiction profiles. The cause is the one rev 3 named: `retrieval_intent.py:128` checks `open_loops` first.
- **Latency:**

  | Phase | p50 | p95 |
  |---|---|---|
  | Phase 1 (continuity) | 532 ms | 1,084 ms |
  | Phase 3 (purposeful) | 1,199 ms | 2,150 ms |

- **#2413 is not built.** The tables `recall_referent`, `recall_referent_posting` and `recall_term_stats` do not exist, and there is no `services/orion-recall/app/referents/`. `MemoryItemV1` has no voice or reason field, and `orion/memory/voice_render.py` does not exist.

### Graphiti today

- The Hub URL fix shipped in **#2457** (commit `ffcf99e88`, "Graphiti URL reachable from host network; surface sync failures"). Hub `.env_example:911` is `GRAPHITI_ADAPTER_URL=http://127.0.0.1:8640`.
- **Live proof it works:** Juniper approved crystallization `ae612548…` at 2026-10-05 04:31:10.63. A `graphiti_temporal` Entity node appeared at 04:31:10.98. The graph went from 25 to 28 nodes / 28 `RELATES_TO` edges.
- **The temporal fields are still unused:** 0 edges have `valid_at`/`invalid_at`.
- The adapter (app-net, `/health` OK) exposes `/v1/episodes`, `/v1/rebuild`, `/v1/neighborhood/{crystallization_id}` and `/v1/search`. orion-durable-runs is also on app-net.

### Reverie visual seed today: alive again, but repetitive

Rev 3 said this seed was dead since about 09-24. **That is no longer true:**
- `reverie_visual_chain` has memory-seeded chains every day since 09-30 (4, 7, 8, 7, 10, 9, 1 per day).
- The reader (`services/orion-thought/app/store.py:917-1018`) takes the newest *approved* crystallization. Juniper's 10-05 approval revived it.
- But in the last 7 days only **2 distinct** memories seeded all of those chains, the latest one roughly 20 times.

### Current gap, in one line

The new memories know what they are about, but nothing turns that into a shared list of things, links other sources to those things, or lets recall use them.

---

## Missing questions (answered here from evidence; open ones at the end)

| Question | Decision | Why |
|---|---|---|
| Wait for #2413's index? | **No.** This spec builds the referent store. #2413 Phases 2-3 are amended to read and write the same tables | Juniper's approved idea; #2413 is unbuilt (checked) |
| Where does the referent store live? | Postgres, as a shared library `orion/memory/referents/`. Three writers: the distill persist node (durable-runs), the mention indexer (memory-consolidation) and the offline seeders (scripts). Recall only reads | The writer already runs in durable-runs' persist node (`episode_distill_graph.py:170-178`), in one transaction |
| Who may add aliases? | The writer proposes. An alias becomes **active** only if it appears verbatim in a verified quote of **Juniper's** prompt. Orion-only aliases stay `proposed` | This narrows #2413's "aliases are not self-authored": the alias is Juniper's own word, and Orion only noticed it |
| Kind for machines | Add `machine`. `project` and `machine` with the same slug are one thing (kind refinement, not a merge) | The writer filed Hecate and Circe as projects |
| Store aliases from the 3 existing runs? | **Yes.** The backfill re-parses `answer_text` from the LangGraph checkpoints through the fixed validator | The raw outputs are still there (checked) |
| Is `episode_memory_link` (rev 3 §2) needed? | **No stored table.** Links are a join, memory → referent → mention, exposed as a view | One fact per row; nothing to keep in sync |
| Graphiti in the chat request path? | Only for the contradiction intent's "as of" view, with a 200 ms timeout. 1-hop and 2-hop walks read Postgres | At this scale Postgres answers both in milliseconds. Graphiti earns its place on the as-of view or goes back to optional (rev 3's honest risk) |
| Old crystallizations at cutover? | Matched by the same referent/phrase rules against their `summary`, labelled `legacy_crystallization`. **The top-100-by-salience path is deleted**, not kept as a fallback | Kill means kill. Stage 4 still retires the rows |

---

## Design

### 1. The canonical referent store (Postgres is the truth)

#### 1.1 Tables

```sql
CREATE TABLE referent (
  referent_id   uuid PRIMARY KEY,            -- uuid5(ns, first canonical key); never changes
  canonical_key text NOT NULL UNIQUE,        -- kind:slug, e.g. machine:hecate
  kind          text NOT NULL,               -- person | place | machine | project | service | concept | event | file | pr
  display_name  text NOT NULL,               -- "Hecate"
  status        text NOT NULL DEFAULT 'active', -- active | provisional | retired
  source        text NOT NULL,               -- episode_writer | graphify | topic_foundry | git_log | seed
  source_build  text NULL,                   -- e.g. 'graphify aff23fac0 2026-09-11'; 'topic_foundry run <id>'
  external_ref  text NULL,                   -- graphify node id / Falkor node_id / PR number
  first_seen_at timestamptz NOT NULL,        -- earliest evidence (memory occurred_at, mention ts, build date)
  last_seen_at  timestamptz NOT NULL,
  memory_count  int NOT NULL DEFAULT 0,      -- maintained by persist; feeds idf
  created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now());

CREATE TABLE referent_alias (
  referent_id  uuid NOT NULL REFERENCES referent,
  alias_norm   text NOT NULL,                -- lowercased, whitespace-collapsed, punctuation-trimmed
  alias_text   text NOT NULL,                -- as first seen
  alias_class  text NOT NULL,                -- name | descriptor | key   (deterministic, 1.3)
  state        text NOT NULL,                -- active | proposed | ambiguous | rejected
  grounded_in  text NULL,                    -- 'chat_prompt:<chat_history_log.id>' when active by grounding
  proposed_by  text NOT NULL,                -- episode_writer | graphify | topic_foundry | juniper | seed
  valid_until  timestamptz NULL,             -- descriptors only (1.3)
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (referent_id, alias_norm));
CREATE INDEX ON referent_alias (alias_norm) WHERE state = 'active';

CREATE TABLE referent_key_redirect (          -- old key -> referent (kind refinements, Juniper-confirmed merges)
  old_key text PRIMARY KEY, referent_id uuid NOT NULL REFERENCES referent, reason text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now());

CREATE TABLE referent_event (                  -- every identity decision, for audit and rollback
  event_id uuid PRIMARY KEY, referent_id uuid NULL, op text NOT NULL,
  -- op: created | alias_added | alias_activated | alias_ambiguous | kind_refined | merge_proposed
  --     | merge_confirmed | merge_rejected | retired
  actor text NOT NULL, detail jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now());

ALTER TABLE episode_memory_referent ADD COLUMN referent_id uuid NULL REFERENCES referent;  -- filled by persist + backfill
CREATE INDEX ON episode_memory_referent (referent_id);
```

- **Kinds** are the brief's seven plus `file` and `pr`, which graphify needs. Each kind has a consumer: the ranking weight (1.5), the mention rules (2.3), and the voice renderer.
- **Roles are normalized at validate time** to `subject | about | participant | location`. The writer's `event`, `topic` and `project` roles map to `about`. Roles are kept for Graphiti edge direction (section 3). Ranking ignores them.
- **Self referents are seeded and fixed**: `person:juniper` and `person:orion`. They are on nearly every memory, so their ranking weight is ~0 (1.5). They can still filter.

#### 1.2 Resolution in the writer's persist node (deterministic, same transaction)

For each referent `{key, role, aliases}` the validator keeps:

1. **Exact key.** The normalized key equals a `referent.canonical_key` → that referent.
2. **Redirect.** The key is in `referent_key_redirect` → that referent.
3. **Kind refinement.** Same slug, and both kinds are in the artifact family {project, machine, service} → the same referent.
   - Kind moves to the more specific one (machine or service over project). Write `kind_refined`, and add the old key as a redirect.
   - This is how `project:hecate` becomes `machine:hecate` without a merge decision.
   - Same slug across *different* families (`person:austin` vs `place:austin`) gives separate referents. They are not ambiguous; they are different things.
4. **Name-alias match.** The key's slug, or one of its **grounded** aliases (1.3), exactly equals an `active` name alias of exactly one existing referent of a compatible kind → that referent.
   - Write `alias_added` for any new aliases.
   - This is still deterministic exact matching, which the approved idea allows.
5. **Ambiguous** (any of the following) → create the new referent as `status='provisional'`, attach the memory to it, and open a tension:
   - the match finds two or more referents;
   - the match finds a referent of an incompatible kind;
   - a descriptor matches two or more live referents.

   The tension goes to `memory_tension_shadow` (`kind='tension'`, `text` = "Is 'X' the same as 'Y'?", `referent_keys` = both, `source_refs` = the quotes). Its scope:
   - `scope=juniper`, `answer_via=conversation` for person/place/event. At Stage 3 it becomes an "Orion is asking" card.
   - `scope=self`, `answer_via=investigation` for machine/service/file/pr/concept. At Stage 3 it becomes a curiosity self-question.
6. **Otherwise mint** a new referent with `display_name` from the first grounded name alias, or else the slug title-cased.

**Never silent merging.** A merge happens only through the outcome consumer at Stage 3: Juniper's `AttentionLoopOutcomeV1` on the tension's loop, or Orion's investigation answer with its evidence. It writes `merge_confirmed` and a redirect. Until then, recall treats a provisional referent and its candidate twin as two things. When a query names either one, `why` shows "possibly the same as X (unconfirmed)".

#### 1.3 Which proposed aliases count (alias admission, deterministic)

| Rule | Effect | Live example |
|---|---|---|
| **Grounding.** The alias text occurs (case-insensitive, word-bounded) in a verified evidence quote whose `source_kind='chat_prompt'` | `state='active'`, `grounded_in` set | "Inspur NF5288M5", "8x smx2 gpus", "AGX-2 GPU" → active on Hecate |
| Not grounded in a prompt (only in a response, the statement, or nowhere) | `state='proposed'`. Visible in telemetry and the report; **never used to resolve** | "a Marriott" if absent from the quote; "the new server" from Orion's statement |
| **Descriptor class.** The alias starts with a determiner or possessive (the, a, an, my, our, her, his, their, this, that), or every token in it is common (df > 0.5% in `recall_term_stats`) | `alias_class='descriptor'`, `valid_until = last use + 90 d` | "my boss", "my sister", "the new server", "camera" |
| **Descriptor resolution.** A descriptor resolves only while valid **and** only when it is active on exactly one referent | Otherwise `alias_ambiguous` → tension | If Juniper later calls a second machine "the new server", Orion asks which |
| **Collision.** An active alias proposed for referent B already exists, active, on referent A | `state='ambiguous'` on B → tension (1.2 step 5) | When `machine:robot` someday gets the alias "autonomous robot", it collides with the camera project and Orion asks |
| **Rare-token index.** For a name alias, each token with df ≤ 0.5% that appears in the aliases of exactly one referent is also matched on its own | Query "the Inspur" → "inspur" → Hecate | "inspur" df 4; "nf5288m5" df ≤ 1 |

Known limit, stated plainly: grounding cannot tell "AGX-2 GPU" (a part of Hecate) from a name for Hecate. The cost is small. A query naming the GPU board finds the Hecate memories, and `why` says which alias matched. The collision rule is what catches a part-of alias once that part gets its own referent.

`alias_class` is not a free label. The recall extractor (1.5) reads it to decide whether a match needs the uniqueness and validity checks.

#### 1.4 The "new server" path, end to end

There are two honest routes, and both are deterministic:
- **Today:** the **phrase lane** (1.5 c). The bigram "new server" is rare in Orion's corpus. It occurs in memory `63d7e72b`'s statement, which is linked to Hecate. So a query containing "the new server" finds that memory, and through its referent the other Hecate memories. `why` reads: "phrase 'new server' in my memory of 10-04 → Hecate".
- **After Juniper says it:** the next distill proposes "the new server" as a descriptor alias grounded in her prompt. It becomes active for 90 days, while it stays unique among machines.

#### 1.5 How referents are seeded

| Seed | Producer | When | What | Label |
|---|---|---|---|---|
| Writer backfill | `scripts/backfill_referents_from_episodes.py` (one-off, idempotent) | PR A deploy | The 14 keys in `episode_memory_referent`, plus aliases re-parsed from `checkpoint_writes.answer_text` for `memdistill-*` threads through the fixed validator. Fills `episode_memory_referent.referent_id` | `source=episode_writer` |
| Writer, live | durable-runs persist node (1.2) | Every distill | Referents + aliases + resolution events | `source=episode_writer` |
| graphify services | `scripts/build_referents_from_graphify.py` | On graphify publish (hooked into the publish script) and nightly if the bundle's `built_at_commit` changed | 95 `service:orion-*`. Name alias = the directory name, plus the name with "orion-" stripped, but only if no other referent claims it | `source=graphify`, `source_build='graphify <commit> <date>'` |
| graphify files | same | same | `file:<full path>`. Basename alias only when unique in the bundle (4,828 of 6,107) | same |
| PRs | same script, `git log --merges origin/main` | same | `pr:<n>` with alias "#n" / "pr n", `external_ref` = merge sha. The PR-report path is a mention target (2.2) | `source=git_log` |
| Topic concepts | `scripts/build_referents_from_topic_foundry.py`, then the indexer's cursor (2.3) | After each completed `topic_foundry_runs` row | 636 `:Concept` nodes from `topic_foundry_adapter`. `concept:<slug(label)>`; the label is the only alias (class name, active by fiat as Orion's own vocabulary) | `source=topic_foundry`, `external_ref=node_id` |
| Rarity table | `scripts/build_recall_term_stats.py` (#2413's `recall_term_stats`) | Nightly | df per term and bigram over the 4,351-doc corpus; AI Town and metacog excluded | — |

Not seeded: graphify's `concept` file-type nodes (they are schema field names, checked) and `rationale` nodes (LLM-derived). World-pulse concepts are left out until a reading-side consumer needs them.

**No vector similarity for identity, anywhere.** Every resolution step above is string equality after normalization.

### 2. Edges: what links to what, who writes each link, and when

#### 2.1 One mention table

```sql
CREATE TABLE referent_mention (
  referent_id  uuid NOT NULL REFERENCES referent,
  target_kind  text NOT NULL,   -- chat_turn | reverie | journal | reading_claim | reading_snapshot | curiosity_finding
                                -- | curiosity_prior | curiosity_question | dream_hypothesis | topic_segment | pr_report | spec
  target_id    text NOT NULL,
  target_ts    timestamptz NOT NULL,
  voice        text NOT NULL,   -- juniper_said | orion_thought | orion_read | orion_self_knowledge
  channel      text NOT NULL,   -- chat | reverie | journal | reading | curiosity | dream | topic_model | graphify
  via          text NOT NULL,   -- alias_exact | rare_token | co_evidence | graphify_source_file | git_merge
  matched_text text NULL,       -- the alias or token that matched
  producer     text NOT NULL,   -- referent_mention_indexer | build_referents_from_graphify | ...
  created_at   timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (referent_id, target_kind, target_id, via));
CREATE INDEX ON referent_mention (target_kind, target_id);
CREATE INDEX ON referent_mention (referent_id, target_ts DESC);

CREATE TABLE referent_indexer_cursor (source text PRIMARY KEY, last_ts timestamptz, last_id text, updated_at timestamptz);
```

`voice` and `channel` reuse rev 3's vocabulary and the renderer keys on them (section 7 of rev 3). Every value has a producer (this table's writers) and a consumer (the renderer and recall).

#### 2.2 Edge types and their producers

| Edge | Stored as | Producer (service, when) |
|---|---|---|
| memory → referent | `episode_memory_referent` (+ `referent_id`) | Distill persist node, orion-durable-runs, same transaction as the memory |
| episode → referent | **view** `episode_referent_v` (union of its memories' referents) | Derived, nothing to write |
| question → referent | `memory_tension_shadow.referent_keys` (exists) | Distill persist node |
| chat turn → referent | `referent_mention` (`chat_turn`, voice `juniper_said` for prompt matches, `orion_thought`/chat for response matches; one row per side, `target_id` = `<id>:prompt` or `<id>:response`) | Mention indexer, orion-memory-consolidation, 60 s cursor |
| reverie → referent | `referent_mention` (`reverie`, `orion_thought`/reverie) | Mention indexer, 60 s cursor over `substrate_reverie_thought.interpretation` |
| journal → referent | `referent_mention` (`journal`, `orion_thought`/journal; non-metacog only) | Mention indexer |
| reading claim / snapshot → referent | `referent_mention` (`orion_read`/reading) | Mention indexer over `world_pulse_claim.payload_json` and `reading_document_snapshot` titles |
| curiosity finding / prior / question → referent | `referent_mention` (`orion_thought`/curiosity) | Mention indexer, 10-min read-only pass over Falkor `orion_worldview` and `curiosity_self_questions`. Refuted priors never |
| dream hypothesis → referent | `referent_mention` (`orion_thought`/dream) | Mention indexer |
| concept → chat turn → memory | `referent_mention` (`topic_segment`, `via=co_evidence`; `target_id` = the chat_history_log id from segment provenance) | Mention indexer, after each completed topic-foundry run. The memory join is `episode_memory_evidence.source_id = target_id` |
| PR / file / service → spec or PR report | `referent_mention` (`pr_report`/`spec`, `orion_self_knowledge`/graphify) | `build_referents_from_graphify.py` on publish |

**The crosswalk is now a view, not a stored link table:**

```sql
CREATE VIEW memory_crosswalk_v AS
SELECT r.memory_id, mn.target_kind, mn.target_id, mn.voice, mn.channel, mn.via, ref.canonical_key AS via_referent, mn.target_ts
FROM episode_memory_referent r
JOIN referent ref ON ref.referent_id = r.referent_id
JOIN referent_mention mn ON mn.referent_id = r.referent_id
JOIN episode_memory m ON m.memory_id = r.memory_id
WHERE ref.canonical_key NOT IN ('person:juniper','person:orion')
  AND (ref.kind NOT IN ('person','event','place')
       OR mn.target_ts BETWEEN coalesce(m.occurred_at, m.created_at) - interval '14 days'
                           AND coalesce(m.occurred_at, m.created_at) + interval '14 days');
```

This keeps rev 3 §6's rule (±14 d for people, events and places) as SQL instead of a second table. Rev 3's "10 newest" for artifact referents is applied at read time (1.5).

#### 2.3 Mention rules (what the indexer may link)

- **Names only.** Only `active` aliases of class `name` (plus rare tokens) are matched, word-bounded and case-insensitive. Descriptors are never matched in other sources. A reverie that says "the new server" is not a mention of Hecate.
- **People only in Orion's own world.** `person:` referents are not matched in `orion_read` (external articles). A news story about another Vincent is not Juniper's colleague.
- **Exclusions at index time:** AI Town (source contains "town"), metacog journals, refuted priors. Same as #2413 and rev 3.
- **Caps:** none at write time, since the table is cheap. Read-time caps are in 4.3.

#### 2.4 On-write crosswalk and backfill

- **On write.** When persist writes `alias_added`/`alias_activated` or `created`, the indexer handles the event next tick. It runs a **bounded alias backfill**: the new alias, over each source's last 90 days, at most 2,000 rows scanned per source per tick.
  - Example: Hecate's referent is created on 10-05 at backfill. The indexer then finds the 180 reveries since 10-04 02:10 that mention it.
- **Initial backfill** follows CLAUDE.md §14, with progress in `/tmp/referent-mention-backfill/progress.log`.
  - No pre-snapshot is needed: it only INSERTs into new tables. The count of target rows is recorded up front.
  - Source sizes: about 20k reveries, 574 chat turns, 2,748 journals, 1,029 claims, 636 concepts.
  - The posting count is **UNVERIFIED**. Estimate: tens of thousands. The report records the real number.
- **Order:** referents are seeded first (PR A/B). The indexer then runs a full pass from cursor zero.

### 3. Graphiti as the temporal mirror (LLM-free)

**What is mirrored** (group id `episode_memory`, separate from the 28 legacy crystallization nodes):
- **One `EpisodicNode` per episode** with memories: uuid = uuid5(episode_id), `valid_at` = episode start.
- **One `EntityNode` per referent** that has at least one mirrored memory: uuid = `referent_id`, name = `display_name`, labels = `[kind]`.
- **One `EntityEdge` per memory:** uuid = `memory_id`, `fact` = statement, `episodes=[episode uuid]`, attributes `{voice, purpose, channel, confirmation_state}`.
  - The source is the referent with role `subject`, else `person:juniper` for `juniper_said`/`worked_out_together`, else `person:orion`.
  - The target is each remaining referent. A memory with no other referent gets an edge to its first `about` referent.
  - Multi-referent memories become several edges sharing `memory_id` in attributes.
- **Only** `auto`/`confirmed` memories with status `active`/`faded` are mirrored. `pending_confirmation` and `rejected` are never mirrored.

**Validity, set by Orion, never by Graphiti's LLM:**
- `valid_at` = `occurred_at`, else `created_at`.
- On `superseded`, `corrected`, `rejected` or `retired`: `invalid_at` = the event time.
- On `faded`: `expired_at` = the event time; the edge stays and is still queryable.

**When:** the **projector**, a cursor loop in orion-memory-consolidation over `episode_memory_event` (cursor table `memory_projection_cursor`). It is idempotent because the uuids are deterministic. A lost tick is caught up later. `POST /v1/rebuild?group=episode_memory` replays from the tables.
- This replaces rev 3's "project node inside the distill graph". A cursor loop survives the adapter being down, and validity changes come from the lifecycle job too, not only from the distiller.

**Write path:** a new adapter endpoint, `POST /v1/memory_projection`, takes explicit nodes and edges. It saves them with graphiti-core's node/edge `save` calls and makes no `add_episode` or `add_triplet` call (those invoke the LLM, rev 3 §D). Whether graphiti-core 0.19's `EntityEdge.save` needs an embedding for `fact` is **UNVERIFIED**. If it does, the adapter's existing CPU embedder (bge on vector-host) fills it. That embedding is stored for Graphiti's internals only and is **never read for recall or identity**.

**The "what did I know about X last week" query:**
- `GET /v1/as_of?referent=<canonical_key>&at=<ts>` returns edges with `valid_at ≤ at` and (`invalid_at` null or > at), and `expired_at` shown as "faded".
- The Postgres twin, which is canonical and used by the eval:

```sql
SELECT m.* FROM episode_memory m JOIN episode_memory_referent r USING (memory_id)
WHERE r.referent_id = $1 AND coalesce(m.occurred_at, m.created_at) <= $2
  AND m.confirmation_state IN ('auto','confirmed')
  AND NOT EXISTS (SELECT 1 FROM episode_memory_event e WHERE e.memory_id = m.memory_id
                  AND e.op IN ('superseded','corrected','rejected','retired') AND e.created_at <= $2);
```

- Example: "what did I know about Hecate on 09-29?" returns nothing, plus "first heard of it 10-04" (`referent.first_seen_at`). That is the honest answer, and both stores must agree on it.

**Consumers:**
- PCR contradiction intent (4.2), with a 200 ms timeout. On timeout, the contradiction lane renders its Postgres items only and logs `graphiti_timeout`.
- The Hub memory report's as-of view.

**Honest risk, carried forward from rev 3:** if 30 days of the equivalence eval (6.5) show Graphiti and Postgres always agree, the request path drops Graphiti and it stays a debug mirror.

### 4. Recall by referent

#### 4.1 The collector: `episode_referent`, in orion-recall

```
query text (retrieval_query from the companion spec)
 1. extract (in-process, ≤ 5 ms)
    a. alias n-grams: 1–5 token windows, longest match first, against an in-memory map
       alias_norm → [(referent_id, class)], built from referent_alias WHERE state='active',
       refreshed every 60 s (a few thousand rows)
       - descriptor hits must be valid and unique (1.3), else marked ambiguous
    b. explicit ids (#2413 regex): "#2287", "PR 2287", orion-* services, file paths
    c. rare tokens (df ≤ 0.5%) not covered by (a), and rare bigrams (the phrase lane)
    → if nothing: ABSTAIN (continuity still runs; no filler)
 2. walk 1 hop (one Postgres round trip)
    referent → episode_memory_referent → episode_memory (status active|faded, not rejected)
             → episode_memory_evidence (quotes)
    rare tokens / phrases → episode_memory.statement and evidence.quote via a 'simple'-config
             tsvector GIN index (exact lexemes, no stemming) → that memory, and through its
             referents, their memories (marked hop=2, "via phrase")
    referent → memory_crosswalk_v / referent_mention, for the channels the intent allows (4.2)
    legacy lane (until Stage 4): the same aliases/tokens → memory_crystallizations.summary (FTS),
             channel legacy_crystallization
 3. rank (lexicographic, deterministic)
    (a) number of distinct query referents matched, weighted:
        Σ idf(referent), idf = ln(N_memories / (1 + memory_count)); self referents → 0
    (b) purpose fits the intent (4.2)
    (c) effective strength = strength × 0.5^(days since last_reinforced_at / half_life)
    (d) recency (occurred_at, else created_at)
 4. caps: ≤ 6 memories; ≤ 2 per referent only when the query names 2+ referents (diversity);
    ≤ 3 crosswalk items; ≤ 1 legacy item per referent
 5. emit MemoryItemV1 with recall_reason {referents:[canonical_key], matched:[text], via:alias|token|phrase|id,
    hop:1|2, ambiguous:bool} plus voice, channel, confirmation_state, faded
```

- **Every result carries why and whose words.** The voice renderer (`orion/memory/voice_render.py`, rev 3 §7) turns that into, for example: "Juniper told me (10-04): she got an Inspur NF5288M5 and named it Hecate. [recalled: you said 'Hecate']".
- **Rendering never reinforces.** The collector writes `recalled` events in a batch, after the reply, and never touches `strength`.

#### 4.2 PCR phases and the new intent → memory mapping

**Phase 1 (continuity, before stance)** replaces today's 1,200-token continuity block and the active packet's top 100:
- the current turn's referent hits (4.1), at most 3, because referent lookup needs no stance;
- the last closed episode's memories for this session/platform, at most 3, `happened` first;
- due follow-ups (`due_after ≤ now`, active), at most 2;
- recent sql_chat as today.

Budget: 300 tokens.

**Phase 3 (purposeful, after stance)** replaces `active_packet` and `concept_region`:

| Intent | Memory purposes read | Voices / channels allowed | Crosswalk channels | Extra |
|---|---|---|---|---|
| continuity | `happened` (last episode), due `follow_up` | all chat | none | — |
| relational | `about_juniper`, `orion_view`, `happened` with a `person:` referent other than self | `juniper_said`, `worked_out_together`, `orion_thought`/chat | journal (Orion's retelling) | — |
| semantic | `happened`, `about_juniper` | all | reading, curiosity, topic_model, reverie, graphify | 2-hop via a shared referent (Postgres) |
| procedural | `follow_up`, `happened` with service/file/pr/machine referents | all | graphify (PR reports, specs) | pageindex section lookup once that track lands |
| open_loop | open `follow_up`, `memory_tension_shadow` questions whose `referent_keys` intersect the query | all | curiosity questions | — |
| contradiction | memories sharing a referent that are `pending_confirmation`, `corrected` or superseded | all | none | Graphiti `as_of` for the top referent (200 ms) |

**Intent derivation fix** (rev 3 §B, now concrete):
- `open_loop` fires only for persistent loops (not `current_turn_v1`, not `already_known`) or an open `follow_up` whose referents appear in the turn.
- Relational and topic rules are evaluated before it.
- **New rule:** if 4.1 extracted a referent of kind person (non-self) → `relational`; of kind machine/service/file/pr → `procedural`; of any other kind → `semantic`. This puts the referent signal directly into the intent choice.
- `rule_id` is logged per call.

**The concept_region collector is deleted at cutover.** The concepts it matched by substring are now referents with exact aliases and co-evidence links.

#### 4.3 Latency budget

| Step | Budget (p95) | Basis |
|---|---|---|
| Extraction (in-memory) | 5 ms | Dictionary lookups over ≤ 5-grams of a ≤ 500-char query |
| 1-hop memory + evidence + phrase lane | 30 ms | One SQL statement over indexed tables. At ~1–2 episodes a day × ~8 memories, the store is ~5k memories a year (**UNVERIFIED** growth) |
| Crosswalk read | 30 ms | Indexed by `referent_id`, `target_ts DESC`, with a LIMIT |
| Legacy lane | 40 ms | FTS over about 1.4k crystallizations |
| Graphiti as-of (contradiction only) | 200 ms hard timeout | Not measured. **UNVERIFIED** |
| **Collector total, without Graphiti** | **≤ 100 ms** | #2413's referent-stage budget |
| Phase 1 total | p50 < 250 ms | Today 532 ms |
| Phase 3 total | p50 < 400 ms, p95 < 800 ms | Today 1,199 / 2,150 ms. Removing the 100-row boost writes and the embed call is most of the win |

### 5. Reverie visual seed from validated shadow memories

Juniper approved this at Stage 2 in rev 3. A new reader, `load_episode_memory_seed()`, is added in `services/orion-thought/app/store.py` beside the old one:
- **Eligible:**
  - `purpose IN ('happened','about_juniper')`;
  - `confirmation_state IN ('auto','confirmed')`, `status='active'`, `stakes='low'`;
  - `created_at > now() - 7 d`;
  - no `downgraded_voice` or `rejected_invalid` event;
  - every evidence quote `verified`.
- **Privacy rule until Juniper's stakes policy lands:** exclude memories with any `person:` referent other than `person:juniper`/`person:orion`.
  - Why: Stage 1's validator leaves stakes to the distiller (`validate.py:19-20, 317-322`), and all 3 family memories are stored `low`/`auto`.
  - This is a structural rule on referent kind, not a word list, and it is removed once the stakes floor exists.
- **Rotation:** pick the eligible memory seeded least recently (ties: newest). This fixes "2 distinct seeds in 7 days".
- **Trace:** `VisualSourceV1(source_kind='episode_memory', source_id=memory_id)` in `chain_json`. The existing "Orion remembers: " prefix and 180-char CLIP cap are kept, and the text goes through the voice renderer's short form.
- **Switch:** `REVERIE_VISUAL_MEMORY_SEED_SOURCE=episode_memory` (ships ON). After 7 days of at least 1 episode-memory seed a day, the crystallization reader is deleted (kill means kill).

### 6. Evals (label-free, `services/orion-recall/evals/referent/`)

1. **Known item, from the writer's own referents.**
   - For every referent with ≥ 1 memory, and every active name alias, the query is "what do you remember about {alias}". The gold set is the memories linked to that referent.
   - Metrics: hit@8 and recall@8. Gate: hit@8 ≥ 0.9.
   - Named cases: "Hecate", "the Inspur" and "Inspur NF5288M5" each return all 5 Hecate memories. "Jackalope" returns the karaoke memory.
   - A phrase case: "the new server" returns `63d7e72b`, then the other Hecate memories at hop 2.
   - Honest caveat: this tests the lookup, not whether the writer chose the right referents. A known-item test built from the writer's referents cannot catch a writer that tags the wrong thing. The next eval partly covers that.
2. **Quote-held-out known item.** The query uses a rare term from a memory's Juniper quote that is in neither its statement nor its aliases. This is reported, not gated. It measures the alias/paraphrase gap #2413's vector verdict watches.
3. **Cousin rate against today.**
   - Replay the last 7 days of `stance_react` Phase 3 queries (`recall_telemetry.query`) through both today's active packet and the shadow collector.
   - Cousin = a rendered, non-feed item that shares no referent (alias, id or rare token) with the query.
   - Today's figure is measured in PR D's baseline (#2413 measured 98.4% across verbs).
   - Gate: shadow ≤ 20%. Distinctness ≥ 10× today's 103/569-retrievals.
4. **Source monitoring** (unit + replay):
   - a reverie mention (e.g. any of the 180 Hecate reveries) never renders with "Juniper told me", "we" or "I told Juniper";
   - `juniper_said` never renders without a verified prompt quote;
   - a `pending_confirmation` item always carries "Unconfirmed";
   - an `orion_read` item always carries its title and claim status;
   - a person mention is never indexed from `orion_read`.

   Replay gate: 0 violations over 7 days of shadow output.
5. **Graphiti equivalence.** For each referent and each day of the last 30, `as_of` against the Postgres twin gives identical memory-id sets. Fixture cases: a supersede, a fade, a rejection, and "before first_seen" (Hecate on 09-29 is empty in both). Gate: 100% equal, and 0 LLM calls (counted at the gateway, route tag).
6. **Identity safety.**
   - 0 merges without a `merge_confirmed` event;
   - every `ambiguous` alias has an open tension row;
   - re-running the backfill twice gives byte-identical `referent`/`referent_alias` tables.
7. **Latency.** Per-step timings in `recall_telemetry.timings_ms` (`referent_extract`, `referent_lookup`, `referent_crosswalk`, `referent_legacy`, `graphiti_as_of`), checked against 4.3.
8. **Abstention honesty.** The share of queries with no referent that abstain is reported. So is the share with a referent the index does not know: a Juniper prompt that contains a rare token but gets no match. That is the "aliases cannot keep up" signal.

**Metric gate for the ranking signal** (matched-referent count weighted by referent idf; CLAUDE.md):
1. **Provenance:** `referent.memory_count`, maintained by the persist node.
2. **Independence:** it replaces salience/activation ranking for these items; it does not sit beside it.
3. **Theory:** inverse document frequency (Spärck Jones 1972), over Orion's own memories.
4. **Live data:**
   - With 25 memories, `person:juniper` (8 referent rows) gets idf ≈ 1.0. It is forced to 0 as a self referent.
   - `project:hecate` (5 of 25): ≈ 1.4. `place:jackalope-bar` (1 of 25): ≈ 2.5.
   - So the signal is not flat. With no referent the sum is 0 and recall abstains: a real rest state.
5. **Existing mechanism:** #2413's df table, reused.
6. **Reversibility:** a computed column and a ranking function. `RECALL_PCR_MEMORY_MODE=legacy` restores today.

### 7. Rollout

#### 7.1 Shadow, side by side

- `RECALL_PCR_MEMORY_MODE=shadow` (ships ON). After each PCR recall returns, a background task runs the `episode_referent` collector on the same query and intent. It never runs before the reply, and is bounded by a semaphore of 2.
- It writes one row to `recall_referent_shadow`:

  ```sql
  recall_referent_shadow(corr_id text PK, created_at timestamptz, verb text, phase text, intent text, rule_id text,
    query text, query_referents jsonb, abstained bool, live_ids text[], shadow_items jsonb,  -- id, why, voice, channel, hop
    cousin_live int, cousin_shadow int, timings_ms jsonb)
  ```

- **The side-by-side report** extends Stage 1's report (`orion/memory/episode/report.py`, Hub `/memory/episodes/report`, plus the daily markdown artifact). It gets a "recall" section. For each day it shows:
  - the 10 most recent chat recalls with live vs shadow items;
  - the cousin rate, distinctness, abstain rates and latency;
  - every identity tension opened.

  It is a report only; no notification is sent.

#### 7.2 Cutover criteria (all must hold over 7 consecutive days of shadow)

1. Known-item hit@8 ≥ 0.9 (eval 1).
2. Shadow cousin rate ≤ 20%, and below live by at least 50 points (eval 3).
3. 0 source-monitoring violations (eval 4).
4. Collector p95 ≤ 100 ms without Graphiti; Phase 3 p50 projected < 400 ms.
5. **No loss:** every live item that *did* share a referent with its query also appears in the shadow set, or is explained in the report (e.g. capped).
6. 0 unconfirmed merges (eval 6).

#### 7.3 Cutover

- Flip `RECALL_PCR_MEMORY_MODE=referent`.
- In the **same PR**:
  - delete `active_packet`'s salience query and boost-on-read;
  - delete the `concept_region` collector;
  - delete the crystallization retriever's Chroma/Graphiti rails.

  Nothing is left as a fallback.

#### 7.4 Rollback

- Before cutover: set `RECALL_PCR_MEMORY_MODE=legacy` to stop shadow work. All new tables are additive and derived. The referent tables can be rebuilt from `episode_memory_referent`, the checkpoints, graphify and topic-foundry by re-running the seed scripts.
- After cutover: rollback means reverting the cutover PR and redeploying orion-recall (one commit), since the old path is deleted, not dormant.
- The Graphiti group `episode_memory` can be dropped and rebuilt.

#### 7.5 PRs, in order

| # | PR | Contents | Acceptance |
|---|---|---|---|
| A | `feat(memory): canonical referent store + alias persistence` | Migration (1.1). `orion/memory/referents/{normalize,resolve,store}.py`. The validator keeps aliases, adds `machine`, normalizes roles. Persist resolves (1.2) and opens tensions. Prompt v3 adds `machine` and descriptor guidance. `scripts/backfill_referents_from_episodes.py`. Schema registry/docs | 14 keys → referents; `project:hecate`/`project:circe` become `machine:` via `kind_refined` + redirects. Hecate has 3 active grounded aliases. "my boss" is an active descriptor with `valid_until`. A forced collision fixture opens a tension and merges nothing. The backfill is idempotent (run twice, same rows). The Stage 1 distill tests still pass |
| B | `feat(memory): seed referents from graphify, git log, topic-foundry; term stats` | 3 seed scripts, the `recall_term_stats` builder, the graphify publish hook | 95 services, about 2,377 PRs, 6,107 files (4,828 basename aliases), 636 concepts, each labelled with its build. Re-run is a no-op. No graphify `concept`/`rationale` nodes |
| C | `feat(memory): referent mention indexer (crosswalk)` | Indexer loop in orion-memory-consolidation (+ read-only `FALKORDB_URI`, `.env_example`, env sync), `referent_mention`, the views, the alias-backfill trigger, the initial backfill under §14 | `machine:hecate` reverie mentions ≈ the live ILIKE count (180 ± matches excluded by word boundary). The Hecate memory joins the concept "GPU performance and work" through `co_evidence`. 0 AI Town/metacog/refuted rows. 0 person mentions from `orion_read`. Indexer lag p95 < 120 s |
| D | `feat(recall): recall by referent in shadow + voice renderer + intent fix` | `orion/memory/voice_render.py`; `services/orion-recall/app/referents/`; `MemoryItemV1` gains `recall_reason`/voice fields (consumer-first, registry); the `retrieval_intent.py` fix + referent rule; `recall_referent_shadow`; the evals (6.1–6.4, 6.7, 6.8) with a baseline report; the report section | Evals 1, 3, 4 run and report. `rule_id` histogram ≥ 3 intents in 7 days. Shadow rows exist for ≥ 95% of stance_react recalls. Live recall latency unchanged (shadow runs after the reply) |
| E | `feat(graphiti): episode-memory temporal mirror + as_of` | Adapter `POST /v1/memory_projection`, `GET /v1/as_of`, group-scoped rebuild; the projector loop in orion-memory-consolidation | Mirrored edge count = `auto`+`confirmed` active/faded memory edges (±0 after rebuild). Eval 5 passes on fixtures. 0 LLM calls |
| F | `feat(thought): reverie visual seed from episode memories` | The 5 reader, flag, trace | ≥ 1 seed a day with `source_kind=episode_memory`. 0 seeds from memories with a non-self person referent. ≥ 5 distinct memory ids over 7 days (if ≥ 5 are eligible) |
| G | `feat(recall): cut PCR over to recall by referent` | Flip the mode; delete active_packet salience + boost, concept_region, retriever rails; delete the old reverie reader | 7.2 met before merge. After 24 h live: 0 retrieval events with 100 ids, 0 boost writes, Phase 3 p50 < 400 ms |

A → B → C are sequential, since each needs the previous one's referents. D can start after A, and its crosswalk read lights up after C. E and F only need A. G waits on the shadow week.

---

## Proposed schema / API changes

- **New tables:** `referent`, `referent_alias`, `referent_key_redirect`, `referent_event`, `referent_mention`, `referent_indexer_cursor`, `memory_projection_cursor`, `recall_referent_shadow`, `recall_term_stats` (from #2413).
- **New views:** `episode_referent_v`, `memory_crosswalk_v`.
- **Altered tables:**
  - `episode_memory_referent` + `referent_id`;
  - a 'simple'-config tsvector GIN index on `episode_memory.statement` and `episode_memory_evidence.quote`;
  - `recall_telemetry` + `query_referents jsonb`, `abstained bool`, `selected_reasons jsonb`, `ranking_mode text` (#2413).
- **Dropped from rev 3:** the `episode_memory_link` table (replaced by the view).
- **Schemas:**
  - `DistillReferentV1` keeps `aliases`, and `ValidatedMemory.referents` becomes `list[ValidatedReferent{key, role, aliases, grounded_aliases}]`;
  - `MemoryItemV1` + optional `recall_reason`, `voice`, `channel`, `confirmation_state`, consumer-first;
  - registry docs updated.
- **#2413 amended:** its `recall_referent`/`recall_referent_posting` are this spec's `referent`/`referent_alias`/`referent_mention`. Its `recall_term_stats` and Phase 0 harness are reused.
- **Graphiti adapter:** `POST /v1/memory_projection`, `GET /v1/as_of`, `POST /v1/rebuild?group=`.
- **Bus:** no new channels. The indexer and projector are table-cursor loops, so they survive lost messages.
- **Env** (each `.env_example`, then `python scripts/sync_local_env_from_example.py`; report keys skipped by `SYNC_PREFIXES`):
  - orion-recall: `RECALL_PCR_MEMORY_MODE=shadow` (legacy|shadow|referent), `RECALL_REFERENT_ALIAS_REFRESH_SEC=60`, `RECALL_RARE_TERM_MAX_DF_FRAC=0.005`, `RECALL_GRAPHITI_AS_OF_TIMEOUT_MS=200`.
  - orion-memory-consolidation: `MEMORY_REFERENT_INDEXER_ENABLED=true`, `MEMORY_REFERENT_INDEXER_TICK_SEC=60`, `MEMORY_GRAPHITI_PROJECTOR_ENABLED=true`, `FALKORDB_URI` (read-only use), `GRAPHITI_ADAPTER_URL=http://orion-athena-graphiti-adapter:8000` (app-net).
  - orion-thought: `REVERIE_VISUAL_MEMORY_SEED_SOURCE=episode_memory`.
  - All flags ship ON.

## Files likely to touch

- **A:**
  - `services/orion-sql-db/manual_migration_referent_v1.sql` (+ rollback)
  - `orion/memory/referents/` (new), `orion/memory/episode/validate.py`, `orion/memory/episode/store.py`
  - `orion/schemas/memory_episode.py`, `orion/cognition/prompts/memory_episode_distill.j2`
  - `scripts/backfill_referents_from_episodes.py`
  - tests in `orion/memory/episode/tests/`, `orion/memory/referents/tests/`
- **B:** `scripts/build_referents_from_graphify.py`, `scripts/build_referents_from_topic_foundry.py`, `scripts/build_recall_term_stats.py`, the graphify publish script hook
- **C:** `services/orion-memory-consolidation/app/referent_indexer.py` (new), `settings.py`, `.env_example`, `docker-compose.yml`, `README.md`, tests
- **D:**
  - `orion/memory/voice_render.py`, `services/orion-recall/app/referents/{extract,lookup,rank}.py`
  - `services/orion-recall/app/worker.py`, `pcr_collectors.py`, `orion/memory/retrieval_intent.py`, `orion/core/contracts/recall.py`, `orion/schemas/registry.py`
  - `services/orion-recall/evals/referent/`, `orion/memory/episode/report.py`
- **E:** `services/orion-graphiti-adapter/app/{main.py,backends/graphiti_core.py}`, `services/orion-memory-consolidation/app/graphiti_projector.py`
- **F:** `services/orion-thought/app/store.py`, `visual_chain.py`, `.env_example`
- **G:** `services/orion-recall/app/collectors/active_packet.py`, `collectors/concept_region.py`, `orion/memory/crystallization/retriever.py`, `services/orion-thought/app/store.py`

## Non-goals

- Vectors or embeddings for identity, linking or ranking. The Graphiti-internal `fact` embedding, if graphiti-core requires it, is never read by recall.
- LLM entity resolution, LLM dedupe or `add_episode`/`add_triplet` in Graphiti.
- Silent merges. Merges happen only on a confirmed outcome.
- The "Orion is asking" producer and the outcome consumer (Stage 3). Stage 2 only writes the tensions to `memory_tension_shadow`.
- The pageindex fixes from rev 3 §C. That is a separate pageindex-owned track. The procedural intent calls it once it lands.
- Distilling non-chat sources into memories. They stay linked sources with their own voices.
- Retiring legacy crystallizations (Stage 4). Stage 2 only stops serving them by salience.
- The stakes policy (Juniper's open decision from Stage 1 review). This spec's only stopgap is the reverie-seed person rule.
- Killing `recall_v2`'s inline shadow diagnostic (separate ticket).

## Acceptance checks (Stage 2 as a whole)

1. **A-G each meet their row in 7.5.**
2. **Hecate end to end** (live, after D):
   - a chat turn "is Hecate flashed yet?" gives a shadow row with `query_referents=[machine:hecate]` and all 5 Hecate memories (3 rendered "Juniper told me (10-04)…", 2 rendered "I told Juniper…" or as Orion's own follow-up), plus at most 3 reverie items rendered "Something I was turning over on my own (reverie, …)";
   - the reason names the alias "hecate";
   - the intent is `procedural` by the referent rule, unless an earlier rule fires (`rule_id` logged either way).
3. **Same for "the Inspur"** (via the rare token) **and "the new server"** (via the phrase lane, hop 2).
4. **No referent:** "how are you feeling tonight" → `abstained=true` and no memory filler.
5. **Graphiti:** `as_of(machine:hecate, 2026-09-29)` is empty in both stores, and `as_of(machine:hecate, 2026-10-06)` returns 5 edges.
6. **Reverie:** a `reverie_visual_chain` row carries `source_kind=episode_memory` and a `memory_id` that resolves in `episode_memory`.
7. **The whole stage made 0 LLM calls** outside the existing distiller.

## Proposal-mode disclosure

- **Capability change:** Orion gets a durable list of the people, places, machines, projects and ideas in its life, with the names Juniper uses for them. Recall starts from the things a message names, and every recalled line says why it came back and whose words it is. Orion can answer "what did I know about X then". Reverie images come from real recent memories.
- **Data touched:**
  - **Reads:** `episode_memory*`, LangGraph checkpoints of distill runs (once, for alias recovery), non-AI-Town chat, non-metacog journals, reveries, dreams, world-pulse claims, reading snapshots, Falkor `orion_worldview` and `orion_substrate` (read-only), topic-foundry segments, the published graphify bundle, `git log`.
  - **Writes:** the new referent tables, `episode_memory_referent.referent_id`, `recall_referent_shadow`, `recall_telemetry` columns, `memory_tension_shadow` (identity tensions), the Graphiti `episode_memory` group, `recalled` events.
- **Privacy boundary:**
  - everything stays on the host;
  - AI Town and metacog never enter mentions;
  - people are never linked from external reading;
  - pending and rejected memories are never mirrored;
  - the reverie seed excludes memories about people other than Juniper and Orion until a stakes policy exists;
  - this public spec names referent keys only and quotes no family, health or location content.
- **Trace that proves it works:**
  - `referent_event` rows for every identity decision;
  - `referent_mention.producer`/`via` on every link;
  - `recall_referent_shadow.shadow_items[].why`;
  - `recall_telemetry.selected_reasons`;
  - `VisualSourceV1.source_id=memory_id`;
  - Graphiti edge uuid = `memory_id`.
- **Dangerous failure modes:**
  - (a) **A wrong merge** puts two people's memories under one name. Mitigated: no merge without a confirmed outcome; collisions open tensions.
  - (b) **Source blending:** a reverie is recalled as Juniper's words. Mitigated by voice on every mention, the renderer contract, and eval 4 as a cutover gate.
  - (c) **A bad alias drags in wrong memories**, e.g. "autonomous robot". Mitigated by grounding, the `why` shown on every item, collision tensions and the descriptor expiry.
  - (d) **Silent narrowing:** a turn that talks around a thing without naming it gets nothing. Mitigated by the visible abstention, the abstain-with-rare-token rate, and the phrase lane.
  - (e) **Stale graphify** (25 days today) cites changed code. Mitigated by the build label on every graphify referent and mention.
- **Rollback:**
  - `RECALL_PCR_MEMORY_MODE=legacy` before cutover; revert PR G after it;
  - `MEMORY_REFERENT_INDEXER_ENABLED=false` / `MEMORY_GRAPHITI_PROJECTOR_ENABLED=false`;
  - `REVERIE_VISUAL_MEMORY_SEED_SOURCE=crystallization` until PR G deletes it;
  - every new table is derived and can be dropped and rebuilt by the seed scripts.

## Open product questions (for Juniper)

1. **When should recall by referent become primary: at Stage 2 when the gates in 7.2 pass, or at Stage 4 as rev 3 planned?**
   - Recommendation: Stage 2, when the gates pass.
   - Why: today's path is query-blind (569 of 569 retrievals return the same-shaped 100 rows) and reinforces junk on every read.
   - The legacy lane keeps old crystallizations reachable by referent until Stage 4 retires them.
2. **Descriptor names like "my boss" or "my sister": may Orion keep them as live names (90-day expiry, grounded in your words), or should descriptors always be confirmed by you before they resolve?**
   - Recommendation: keep them live with expiry. They are your own words, and a collision always becomes a question.

## Recommended next patch

**PR A, the canonical referent store with alias persistence.** It is small, it unblocks everything else, and it stops losing data today: every distill run currently throws away the aliases the writer produced. Acceptance is the PR A row in 7.5, centred on Hecate. Its three aliases come back from the checkpoints, `project:hecate` refines to `machine:hecate`, and a forced name collision opens a tension instead of merging.
