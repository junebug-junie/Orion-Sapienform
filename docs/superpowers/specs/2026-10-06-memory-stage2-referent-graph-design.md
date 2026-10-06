# Memory Stage 2: the things Orion knows about, as one graph, and recall that starts from them

Status: **PROPOSAL, revision 2** (design mode + proposal mode). Nothing here is implemented.
Date: 2026-10-06
Parent: `2026-09-30-memory-episode-redesign-design.md` (rev 3, APPROVED 2026-10-01), its "Stage 2" rows.
Builds on:
- `2026-09-29-recall-semantic-retrieval-pipeline-design.md` (#2413, "recall by referent, not resemblance"). No vectors and no similarity anywhere in this design.
- `docs/plans/substrate/2026-10-06-reading-property-graph-design.md` and `…/2026-10-06-neighborhood-read-contract.md` (**#2497, merged**). This design follows their doctrine:
  - Postgres holds text, evidence and the append-only audit trail.
  - Falkor (the substrate graph) holds what can be walked.
  - A relationship becomes walkable only once its assertion is accepted.

Evidence base:
- read-only queries on the `conjourney` Postgres and FalkorDB `GRAPH.RO_QUERY`;
- read-only neighborhood replays (`scripts/replay_substrate_neighborhood.py`);
- the published graphify bundle;
- LangGraph checkpoints of the three live distill runs;
- code on main @ 4d03ae877.

All of it was taken on 2026-10-06. Anything not checked live is marked **UNVERIFIED**.

The repo is public. Memories about Juniper's family, body, feelings or whereabouts are counted here, never quoted.

## What changed in revision 2

Revision 1 built a second graph of the same things #2497 governs. It used Postgres tables `referent` and `referent_mention`, walked them in Postgres, and added a Graphiti mirror. It also planned a second replacement for `concept_region`. Revision 2 removes that duplication:

1. **Referents are substrate nodes.**
   - People, places, machines, projects, services and events become `EntityNodeV1`. They use `entity_type`, a free-text field the codec already persists, so `machine` needs no schema change.
   - Concepts become `ConceptNodeV1`.
   - The Postgres `referent` table is gone.
2. **Links are graph edges with voice, channel, producer and reason.**
   - A memory, reverie, reading, curiosity note, journal or dream is an `EvidenceNodeV1`. Its text stays in Postgres.
   - The link from a referent to that evidence is a provenance edge (`observed_in`, `edge_role=provenance`).
   - Every claim-bearing step goes through #2497's assertion lifecycle: proposed, then provisional/canonical, or rejected/deprecated. The claim-bearing steps are alias→referent resolution, merges, and referent↔referent relationships.
   - `referent_mention` is gone.
3. **Postgres keeps four things:** the alias lifecycle, the evidence text and quotes, the episode_memory rows, and one shared append-only journal. Section 1.3 justifies keeping aliases there.
4. **Recall by referent walks through #2497's bounded neighborhood API**, plus one bounded evidence-handle read. It is the single replacement for both `concept_region` and the active packet. The latency budget is measured against real neighborhood replays, and those are slow today (section 4.3).
5. **The Graphiti mirror is dropped from Stage 2.** Substrate edges already store `valid_from`/`valid_to` natively, and the Postgres journal holds every state change. Together they answer "what did I know about X last week" (section 3). We do not keep two time-aware stores.
6. **Stakes is decided, not open.** Juniper's policy: health, family/relationships, feelings and identity conclusions get confirmed in conversation. A fix PR is in flight. The reverie seed is now gated on that policy (section 5).
7. **The PR sequence is now A–H.** PR A is the shared substrate contract patch that both this design and #2497's reading work need.

---

## Arsonist summary

Stage 1 works. Since 10-02, after each conversation, Orion writes down what is worth keeping, in its own words. Each memory names the things it is about: `project:hecate`, `person:vincent`, `place:chicago`. Nothing uses those names yet. Recall still serves the same top 100 old rows by salience, whatever Juniper says.

Stage 2 makes those names part of Orion's one graph:

1. **Each thing becomes a node in the substrate graph.**
   - "Hecate", "the Inspur" and "Inspur NF5288M5" are names for one machine node.
   - A name only counts if Juniper actually said it.
   - When a name could mean two things, it becomes a *proposal* that Orion asks about. Nothing merges, or becomes walkable, until it is accepted.
2. **Everything Orion has about a thing hangs off its node as evidence:** memories, reveries, readings, curiosity notes. Each link says whose words they are.
3. **Recall starts from the things a message names.**
   - "How's Hecate doing?" goes from the Hecate node to its evidence, and through #2497's bounded neighborhood to related things.
   - Each item comes back with why it was recalled and whose words it is.
   - Orion's 180 reveries about Hecate come back labelled as its own thoughts, never as something Juniper said.
4. **"What did I know about Hecate last week?" is one graph query.** It reads edge validity dates plus the journal. No second time store.
5. **The reverie image seed uses validated new memories.** High-stakes ones are used only after Juniper confirms them.

It runs in shadow next to today's recall. It replaces the old recall only when measured gates pass.

**Live findings that shape it:**
- **The writer already proposes aliases, and Stage 1 throws them away.** In the raw 27B output for the Hecate episode, `aliases` = "Inspur NF5288M5", "AGX-2 GPU", "8x smx2 gpus". `validate.py:308-315` keeps only `(key, role)`.
- **Some proposed aliases are wrong as identities.** "my boss" only means something relative to Juniper. "a Marriott" names a class of hotel. "autonomous robot" for the camera project names a different future thing. So aliases need grounding and a collision path.
- **The substrate graph has almost nothing walkable by default.** All 1,610 Entity nodes and 851 of 855 Concept nodes are `proposed`, but #2497's neighborhood defaults to `provisional`/`canonical`. A default read focused on topic-foundry's "circe" entity returns `focal_unavailable_or_filtered`.
- **Neighborhood reads are slow on the chat path today:** 352–674 ms for one focal node from the host. The cost is round trips, not query time. A single bounded Cypher query runs in 2–4 ms inside Falkor.

---

## Current architecture

### What Stage 1 writes (live, 2026-10-06)

| Table | Rows | Notes |
|---|---|---|
| `episode_distill_run` | 3 (10-04 01:26, 10-05 02:56, 10-05 03:53) | 27B Q4 on route `memory_distill`; LLM 35 s / 78 s / 118 s; hold wait 0.5–1.5 s |
| `episode_memory` | 25 | 15 `happened`/`juniper_said`, 7 `about_juniper`/`juniper_said`, 2 `happened`/`orion_thought`, 1 `follow_up`. All `stakes=low`, `auto`, `active`, because they were written before Juniper's stakes policy was implemented |
| `episode_memory_referent` | 32 rows, 14 distinct keys | No alias column. Indexed on `referent_key` |
| `episode_memory_evidence` | 26 quotes, all `verified=t` | 22 from prompts, 4 from responses. `source_id` = `chat_history_log.id` |
| `episode_memory_event` | 25, all `created` | No supersede, fade or confirm events yet |
| `memory_tension_shadow` | 2 open questions | `scope=juniper`, `answer_via=conversation`, with `referent_keys` |

**Keys emitted** (key: role count):

| Key | Role(s) |
|---|---|
| `person:juniper` | subject 5, about 3 |
| `project:hecate` | about 5 |
| `person:juniper-sister` | about 3 |
| `event:austin-offsite` | event 2 |
| `event:joker-2-viewing` | event 2 |
| `person:vincent` | participant 2 |
| `place:the-wade` | location 2 |
| `project:circe` | about 2 |
| `concept:space`, `person:orion`, `person:rachel`, `place:chicago`, `place:jackalope-bar`, `project:orion-camera` | 1 each |

What this shows:
- Roles are free text (7 values).
- Event keys are not dated, even though the prompt asks for a `-yyyy-mm` suffix.
- Machines are filed as `project`. The kind list in `validate.py:44` has no `machine`.

**Aliases the writer emitted**, recovered from `checkpoint_writes.answer_text` for the `memdistill-*` threads:

| Key | Aliases | Grounded in Juniper's words? |
|---|---|---|
| `project:hecate` | "Inspur NF5288M5", "AGX-2 GPU", "8x smx2 gpus" | Yes. Prompt `ddc979a1…`: "I got us an Inspur NF5288M5 AGX-2 GPU that holds 8x smx2 gpus. We'll call it Hecate" |
| `person:juniper-sister` | "my sister" | Yes, but relative to the speaker |
| `place:the-wade` | "the Wade", "a Marriott" | "the Wade" yes. "a Marriott" is a class (**UNVERIFIED** whether it is quoted) |
| `event:joker-2-viewing` | "Joker 2" | Yes |
| `event:austin-offsite` | "Austin team", "offsite" | "offsite" yes |
| `person:rachel` | "my boss" | Yes, but relative to the speaker |
| `place:jackalope-bar` | "Jackalope bar" | Yes |
| `project:orion-camera` | "camera", "autonomous robot" | Yes, but "autonomous robot" is a different future thing |
| `event:chicago-simulation` (question only) | "simulation", "client use case" | Generic words |

"The new server": Juniper never wrote it. Zero prompts match `new server|the inspur`. The phrase appears only in Orion's own memory statement `63d7e72b`.

### The substrate graph today (Falkor `orion_substrate`, live)

- **Nodes:**
  - 2,494 Evidence (2,508 are `topic_foundry_run_topic`, all `proposed`; the two counts were taken a few minutes apart);
  - 1,610 Entity: all `entity_type=unknown`, `proposed`, `anchor_scope=world`, from `topic_foundry_adapter`, unique by lowercased label;
  - 855 Concept: 851 `proposed`, 4 `canonical`.
- **Edges (37,630):**

  | From → to | Predicate | Count |
  |---|---|---|
  | Concept → Entity | `associated_with` | 17,481 |
  | Concept → Concept | `co_occurs_with` | 13,459 |
  | Concept → Concept | `associated_with` | 3,379 |
  | Evidence → Concept | `supports` | 3,310 |

- **0 edges have `valid_from` set.** The codec persists it natively (`falkor_codec.py:90-91`, and for edges at `:354-355`); nothing writes it.
- **Entity identity today is the label.**
  - `reconcile.py:143-156` keys an entity as `entity|{scope}|{subject}|label:{label}`.
  - Concepts can also merge by embedding cosine (`_concept_embedding_match_key`).
  - Both conflict with "no silent merge, no vector identity". That is true for our nodes, and #2497 notes it in general.
  - Topic-foundry already has Entity nodes labelled "juniper", "circe" and "orion".
- **Codec:**
  - It durably supports only `concept`, `entity` and `evidence` (`DURABLE_NODE_KINDS`, `falkor_codec.py:120`).
  - For entities it stores `entity_type` and `aliases_json`. The latter is a JSON string, not indexable.
  - For evidence it stores `evidence_type` and `content_ref`.
  - `metadata` is not persisted generically.
  - `SubstrateEdgeV1` has no acceptance state and no `edge_role` (#2497 §"Existing contracts").
- **Writers** go through `SubstrateGraphMaterializer.apply_record` (`orion/substrate/materializer.py:41`).

### #2497 (merged): what it gives and what it leaves for later

**What it gives:** `read_neighborhood(NeighborhoodRequestV1)`:
- internal and boundary edges with separate budgets, round-robin over focal node, direction and predicate;
- only Concept/Entity endpoints in the requested promotion states;
- receipts: `complete_for_request`, `truncated`, `degraded`, `reason`;
- no hydration and no cache;
- caps of 16 focal nodes and 256 edges/neighbors.

Evidence and other non-semantic endpoints never compete for budgets (`neighborhood_backends.py`, the `source.node_kind IN ['concept','entity']` filter).

**What it leaves for later patches:**
- the assertion pipeline: `Assertion` node kind, `assertion_subject`/`assertion_object` edges, and proposal/decision/materialization events with a journal;
- native `edge_role`;
- `visibility_scope`;
- the provenance resolver (`EvidenceLineageV1`);
- moving existing callers, including `concept_region`, onto the new read.

**Its own replay:** 8 hub-heavy focal nodes, budgets 12/16/16, 1,186.61 ms, truncated, not degraded.

**Neighborhood replays I ran** (read-only, host Python against `redis://localhost:6380`, `proposed` opted in, 3 runs each):

| Focal node | Budgets | Elapsed ms | Boundary edges | Truncated |
|---|---|---|---|---|
| Concept "GPU performance and work" | 12/16/16 | 621, 526, 674 | 15 | no |
| Entity "circe" (topic-foundry) | 12/16/16 | 410, 388, 396 | 16 | yes |
| Entity "circe" | 4/8/8 | 351, 406, 359 | 8 | yes |
| Entity "circe", default states | 4/8/8 | 6 | 0 | `focal_unavailable_or_filtered` |

For comparison, one bounded Cypher query (focal → latest 6 evidence nodes) runs in 2.6–4.1 ms inside Falkor, and about 100 ms through `docker exec`. The neighborhood cost is the number of round trips. It makes one query per focal node, per predicate group, per edge page, and **per neighbor node** (`neighborhood.py:166-170` fetches each outside endpoint separately).

### Other stores that mention referents (live counts)

| Source | Hecate | Inspur | Vincent | Jackalope | Wade |
|---|---|---|---|---|---|
| `substrate_reverie_thought` | **180** (first 10-04 02:10, 41 min after Juniper's message) | 1 | 0 | 0 | 0 |
| `journal_entries`, non-metacog | 6 | 3 | 2 | 2 | 2 |
| `chat_history_log` | 9 (1 prompt; 8 are Orion's unprompted outreach) | 1 | 1 | 1 | 3 |

**How rare names are in Orion's own corpus.** The corpus is 574 non-AI-Town chat turns, 2,748 non-metacog journals and 1,029 claims: 4,351 documents. Document frequency over chat and journals:

| Term | Documents |
|---|---|
| hecate | 15 |
| inspur | 4 |
| chicago | 13 |
| offsite | 14 |
| server | 27 |
| camera | 38 |

At #2413's cut (df ≤ 0.5%, so ≤ 21 docs), "hecate" and "inspur" are rare and "server" is not.

**graphify:** the published bundle was built at `aff23fac0` (file dated 2026-09-11, 25 days stale). It has 95 `orion-*` services and 6,107 source files, of which 4,828 have a unique basename. Its `concept` file-type nodes are schema field names, not referents. `main` has 2,377 "Merge pull request #N" commits.

### Recall today (live, 7 days)

- **The active packet ignores the query.** All 569 retrievals return exactly 100 ids, and only 103 distinct ids appear across them.
- **Phase 3 always runs as `open_loop`.** 608 of 608 recalls used `chat.belief.open_loop.v1`, because `retrieval_intent.py:128` checks open loops first.
- **Latency:**

  | Phase | p50 | p95 |
  |---|---|---|
  | Phase 1 | 532 ms | 1,084 ms |
  | Phase 3 | 1,199 ms | 2,150 ms |

- **Not built:** #2413's tables, `MemoryItemV1` voice/reason fields, `orion/memory/voice_render.py`.

### Graphiti today

- **The Hub URL fix shipped in #2457** (`ffcf99e88`). Live proof: Juniper's approval of crystallization `ae612548…` at 2026-10-05 04:31:10.63 produced a `graphiti_temporal` node at 04:31:10.98. The graph now has 28 nodes and 28 edges.
- **Its time fields are unused:** 0 edges have `valid_at`/`invalid_at`.
- **Its only writer** is the legacy crystallization-approval sync.

### Reverie visual seed today

- **It is alive again since 09-30**, contrary to rev 3. It produced 4–10 memory-seeded chains a day.
- It reads the newest *approved* crystallization (`services/orion-thought/app/store.py:917-1018`).
- **But only 2 distinct memories seeded it in 7 days.**

---

## Missing questions (answered here from evidence)

| Question | Decision | Why |
|---|---|---|
| One graph or two? | **One.** Referents are substrate Entity/Concept nodes; memories and other sources are Evidence nodes | #2497's doctrine. It also avoids two plans for one recall seam |
| Who builds the assertion core? | **PR A, a shared substrate contract patch** used by both this design and #2497's reading pipeline | Memory needs it first, and #2497 left it unbuilt. Building it twice would recreate the conflict |
| Aliases: node property or Postgres? | **Postgres is the source of truth for the lifecycle and the lookup.** Active aliases are copied onto `EntityNodeV1.aliases` for display | See 1.3. Each alias's state, grounding and validity are audit data. `aliases_json` is a non-indexable JSON string. `ConceptNodeV1` has no `aliases` field |
| Does a mention (memory→referent link) need acceptance? | **Acceptance is enforced upstream, on the alias and the node.** A mention is a provenance receipt (where a name appears), not a claim. It is created only through an accepted alias pointing at a provisional/canonical node | One proposal per name, not 20k proposals for reveries. The claim-bearing step is "this name means this thing" |
| Referent↔referent relationships? | **`co_occurs_with` assertions only**, from a verified Juniper quote naming both. Accepted under a named, tested policy (2.2). Never `causes`, `part_of`, and so on | #2497: `co_occurs_with` "only records source co-occurrence". It asks for any auto-accept policy to be named and tested, not quietly enabled |
| Should new referent nodes enter dynamics/attention? | **Not in Stage 2.** They are excluded from the dynamics/attention eligibility predicate by provenance source kind, with a before/after test | #2497 rule 8: keep cognitive eligibility unchanged in the first migration |
| Is Graphiti needed for the as-of view? | **No** (section 3) | Native edge validity plus the journal answer it in one query |
| Old crystallizations at cutover? | A Postgres full-text lane over `memory_crystallizations.summary` until Stage 4. **The top-100-by-salience path is deleted** | Kill means kill |

---

## Design

### 1. Referents as substrate nodes

#### 1.1 Node mapping (existing models and codec)

| Writer kind | Model | `entity_type` | `anchor_scope` |
|---|---|---|---|
| person (not Juniper or Orion) | `EntityNodeV1` | `person` | `juniper` |
| place | `EntityNodeV1` | `place` | `juniper` |
| event | `EntityNodeV1` | `event` | `juniper` |
| machine (new value; also `project:` slugs refined to it) | `EntityNodeV1` | `machine` | `orion` |
| project | `EntityNodeV1` | `project` | `juniper` or `orion`, by first evidence |
| service / file / pr | `EntityNodeV1` | `service` / `file` / `pr` | `orion` |
| concept | `ConceptNodeV1` | — | `juniper` |
| `person:juniper`, `person:orion` | `EntityNodeV1` | `person` | `juniper` / `orion`, seeded once |

- **`EventNodeV1` is not used.** The codec cannot persist it, and #2497's neighborhood only walks Concept/Entity. An event is an Entity with `entity_type=event`.
- **`entity_type` values come from a closed set** in `orion/memory/referents/kinds.py`. The writer validates them and the schema docs list them. The Pydantic field stays a free string, so the schema does not change.
- **Node id:** `referent-<uuid5(ns, canonical key at mint)>`. It is never recomputed from a later label. Two things with the same name get different keys only through the ambiguity path (1.2), which #2497 requires ("IDs are not hashes of labels").
- **Identity key (reconcile).** A new branch in `reconcile.py` gives nodes whose `provenance.producer == "memory.referents"` the key `referent|<node_id>`. That fences them off from:
  - the label-identity merge (`reconcile.py:143-156`);
  - the embedding-cosine concept merge. Our concepts carry no embedding, and a test asserts `_concept_embedding_match_key` returns `None` for them.
- **Promotion state** follows #2497's single lifecycle:

  | State | When | Walkable? |
  |---|---|---|
  | `provisional` | Minted from a grounded writer key or alias | Yes |
  | `proposed` | Minted on the ambiguity path | No, until decided |
  | `canonical` | Confirmed by Juniper | Yes |
  | `deprecated` | Merged away (a redirect alias points to the survivor) | No |
  | `rejected` | Juniper said it is not a thing | No |

- **Provenance:** `producer="memory.referents"` and `source_kind="episode_memory_referent"`. `authority="user_asserted"` when grounded in a Juniper prompt, else `local_inferred`. `evidence_refs` = the memory ids.
- **Roles** are normalized at validate time to `subject | about | participant | location`. They are kept on the provenance edge, not used for ranking.

#### 1.2 Resolution in the writer's persist node (deterministic)

The persist node in orion-durable-runs writes Postgres only, in one transaction:
- the memory rows (as today);
- `episode_memory_referent.node_id`;
- alias rows;
- journal events (1.4).

A projector then materializes the graph (2.3). This is the transactional-outbox shape #2497 requires.

For each referent `{key, role, aliases}`:
1. **Exact key, or a redirect.** The key exists as an alias row of class `key` in state provisional/canonical → that node.
2. **Kind refinement.** Same slug, and both kinds are in {project, machine, service} → that node. Write `kind_refined` and add the old key as a `key` alias. This is how `project:hecate` becomes `machine:hecate`. It is not a merge.
3. **Grounded name alias.** The key's slug, or one of its grounded aliases (1.3), exactly equals a provisional/canonical name alias of exactly one node of a compatible kind → that node.
4. **Ambiguous** → mint a new node as `proposed`, attach the memory to it, and write an **identity proposal**. A referent is ambiguous if any of these hold:
   - it matches two or more nodes;
   - it matches a node of an incompatible kind;
   - it is a descriptor that matches two or more live nodes;
   - its slug equals the label of an existing *non-memory* node, such as topic-foundry's "circe".

   The proposal is a `SubstrateGraphProposalV1` with `proposal_kind=referent_identity` (1.4). It carries both node ids, the quotes, and the question "Is 'X' the same as 'Y'?".

   It is mirrored to `memory_tension_shadow`:
   - person, place, event: `scope=juniper`, `answer_via=conversation`. At Stage 3 this becomes an "Orion is asking" card.
   - machine, service, file, pr, concept: `scope=self`, `answer_via=investigation`. At Stage 3 this becomes a curiosity self-question.
5. **Otherwise** mint a `provisional` node.

**Nothing merges without a decision.**
- A merge is a `SubstrateGraphDecisionV1` with `resolution=merge`. It comes from Juniper's `AttentionLoopOutcomeV1` (Stage 3), or, for self-scope kinds, from Orion's investigation answer with its evidence.
- The projector then:
  - moves the evidence edges to the survivor;
  - sets the loser to `deprecated` and writes a redirect alias;
  - emits `SubstrateGraphMaterializationV1` with the real ids.
- Until then the proposed node is invisible to the neighborhood, because of the default states. Recall shows it only when a query names it directly, as "possibly the same as X (unconfirmed)".

#### 1.3 Aliases: Postgres holds the lifecycle; the node holds a display copy

```sql
CREATE TABLE referent_alias (
  node_id      text NOT NULL,                -- substrate node id
  alias_norm   text NOT NULL,                -- lowercased, whitespace-collapsed, punctuation-trimmed
  alias_text   text NOT NULL,
  alias_class  text NOT NULL,                -- key | name | descriptor
  promotion_state text NOT NULL,             -- proposed | provisional | canonical | rejected | deprecated (#2497 vocabulary)
  grounded_in  text NULL,                    -- 'chat_prompt:<chat_history_log.id>'
  proposed_by  text NOT NULL,                -- episode_writer | graphify | git_log | juniper | seed
  valid_until  timestamptz NULL,             -- descriptors only
  created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (node_id, alias_norm));
CREATE INDEX ON referent_alias (alias_norm) WHERE promotion_state IN ('provisional','canonical');
ALTER TABLE episode_memory_referent ADD COLUMN node_id text NULL;
CREATE INDEX ON episode_memory_referent (node_id);
```

**Why aliases live in Postgres, not as node properties:**
1. The per-turn extractor needs an exact-string index across all names. `aliases_json` is a JSON string property and cannot be indexed.
2. Each alias has its own lifecycle: state, grounding quote, validity window, who proposed it. #2497 puts lifecycle and audit in Postgres.
3. `ConceptNodeV1` has no `aliases` field at all.

The projector writes the provisional/canonical names onto `EntityNodeV1.aliases`, so the Atlas and the graph workbench show them. That copy is for display only and is rebuilt from the table.

**Admission rules** (deterministic). The alias state is the acceptance gate for every mention.

| Rule | Effect | Live example |
|---|---|---|
| **Grounding:** the alias occurs (word-bounded, case-insensitive) in a verified evidence quote with `source_kind='chat_prompt'` | `provisional`. This is an auto-accept policy, `alias_grounding_v1` ("it is Juniper's own word"), with tests | "Inspur NF5288M5", "8x smx2 gpus", "AGX-2 GPU" on Hecate |
| Not grounded in a prompt | `proposed`. Never used to resolve or to index | "a Marriott" if unquoted; "the new server" from Orion's statement |
| **Descriptor:** starts with a determiner or possessive (the, a, an, my, our, her, his, their, this, that), or every token is common (df > 0.5%) | `alias_class='descriptor'`, `valid_until = last use + 90 d`. Resolves only while valid and unique | "my boss", "my sister", "camera" |
| **Collision:** an alias proposed for node B is already provisional/canonical on node A | The alias on B is `proposed` → identity proposal (1.2 step 4) | If a robot node someday gets "autonomous robot", Orion asks |
| **Rare tokens:** a token of a name alias with df ≤ 0.5% that belongs to exactly one node's aliases | The extractor matches it on its own | "the Inspur" → "inspur" (df 4) → Hecate |
| Juniper confirms or corrects (Stage 3) | `canonical` / `rejected` | — |

Known limit: grounding cannot tell that "AGX-2 GPU" is part of Hecate rather than its name. The cost is small. A query naming the board finds the Hecate memories, and `why` names the alias. The collision rule catches it once the board gets its own node.

#### 1.4 One shared journal (Postgres, append-only)

#2497 proposed reading-specific `ReadingGraphProposalV1/DecisionV1/MaterializationV1`. This design generalizes them:
- **Schemas:** `SubstrateGraphProposalV1`, `SubstrateGraphDecisionV1` and `SubstrateGraphMaterializationV1`, with a field `proposal_kind ∈ {relationship_assertion, referent_identity}`.
- **Table:** one table, `substrate_graph_journal`. Columns: event_id PK, kind, proposal_id, assertion/node ids, expected revision, actor/authority, `visibility_scope`, evidence refs, payload jsonb, recorded_at.
- **Channels:** as in #2497, renamed to `orion:substrate:graph:{proposal,decision,materialized}`.

Reading uses `relationship_assertion`. Memory uses both kinds. Alias transitions (`alias_added`, `alias_state_changed`, `kind_refined`) are journal rows too. That gives one audit trail, not `referent_event` plus a separate reading journal.

### 2. Edges

#### 2.1 Provenance edges: referent → evidence, with voice on every link

**Evidence nodes.** Every source item that names a referent becomes one `EvidenceNodeV1`:
- `evidence_type` ∈ {`episode_memory`, `chat_turn`, `reverie`, `journal`, `reading_claim`, `reading_snapshot`, `curiosity_finding`, `curiosity_question`, `dream_hypothesis`};
- `content_ref` = `<table>:<id>`, e.g. `episode_memory:01c4d9ef…` or `reverie:<thought_id>`. The text stays in Postgres;
- PR A adds `voice` and `channel` to the codec's evidence encoding as native properties. They go on a closed allowlist; this is never a metadata dump.

**Provenance edges.** There is one edge per (referent, evidence) pair: `Entity|Concept --observed_in--> Evidence`, using the existing predicate. Each edge carries:
- native `edge_role=provenance` (PR A adds `edge_role` to edges, as #2497 requires);
- `temporal.valid_from` = the source time, and `valid_to` when the evidence is superseded or retracted;
- the provenance `producer`;
- `via` ∈ {`writer`, `alias_exact`, `rare_token`} and `matched_text`, as native properties.

These edges have an Evidence endpoint, so `read_neighborhood` never walks them and they never compete for semantic budgets (the endpoint-kind filter in `neighborhood_backends.py`). PR A also excludes them from dynamics and pressure (#2497 rule 8).

| Edge | Producer (service, when) |
|---|---|
| referent → `episode_memory` evidence | Projector in orion-memory-consolidation, from the journal, seconds after each distill persist |
| referent → `chat_turn` (prompt side `juniper_said`; response side `orion_thought`/chat) | Mention indexer in orion-memory-consolidation, 60 s cursor |
| referent → `reverie` (`orion_thought`/reverie) | Mention indexer over `substrate_reverie_thought.interpretation` |
| referent → `journal` (non-metacog; `orion_thought`/journal) | Mention indexer |
| referent → `reading_claim` / `reading_snapshot` (`orion_read`/reading) | Mention indexer. When #2497's reading pipeline lands, its `SourceDocument`/excerpt Evidence nodes are reused instead of minting a second node for the same source |
| referent → `curiosity_finding` / `curiosity_question` (`orion_thought`/curiosity) | Mention indexer, a 10-minute read-only pass over Falkor `orion_worldview` and `curiosity_self_questions`. Never from refuted priors |
| referent → `dream_hypothesis` (`orion_thought`/dream) | Mention indexer |
| episode → referent | Not stored. `episode_memory.episode_id` groups the memory evidence nodes |
| graphify service / file / pr | **Lazy** (2.4) |

**Mention rules:**
- Only provisional/canonical aliases of class `key` or `name` are matched, plus rare tokens. Descriptors are never matched.
- `person` referents are never linked from `orion_read`.
- Nothing is indexed from AI Town, metacog or refuted priors.

**On-write crosswalk.**
- When an alias becomes provisional, the indexer backfills that alias over each source's last 90 days, capped at 2,000 rows per source per tick.
- It keeps a cursor per source (`referent_indexer_cursor`, Postgres).
- It is idempotent because ids are deterministic: `ev-<uuid5(content_ref)>` and `edge-<uuid5(node_id|content_ref|observed_in)>`.

The memory's own `episode_memory_referent` row and the Falkor edge record the same fact. The row is the writer's record; the edge is its projection. The reconciler (2.3) repairs the edge from the row, never the other way round.

#### 2.2 Semantic edges: referent ↔ referent, only through assertions

**What Stage 2 creates.** Exactly one kind of semantic relationship: **`co_occurs_with` between two referents, other than Juniper and Orion, named in the same verified Juniper prompt quote**. Each is:
- an Assertion node (PR A: durable `assertion` kind, with `assertion_subject`/`assertion_object` edges);
- with `predicate=co_occurs_with`;
- supported by the memory's Evidence node (`supports`).

**Acceptance policy `source_cooccurrence_v1`.** It is named, tested, limited, and switchable on its own via `MEMORY_COOCCURRENCE_AUTO_ACCEPT`.
- An assertion is accepted as `provisional` only when all of these hold:
  - both endpoints are provisional/canonical;
  - the source is a verified prompt quote;
  - the predicate is `co_occurs_with`;
  - the memory has no more than 6 such assertions.
- Anything else stays `proposed`.

**Projection.** An accepted assertion is projected as a semantic edge with `edge_role=semantic_projection`, `assertion_id` and `assertion_revision`, which is #2497's rule. This gives the neighborhood something real to walk:
- Vincent ↔ Austin offsite ↔ Rachel;
- Vincent ↔ Jackalope bar;
- Hecate ↔ Circe.

Other predicates (`part_of`, `causes`, `subtype_of`) are out of scope. They stay with #2497's reviewed reading pipeline.

#### 2.3 The projector (Postgres journal → Falkor)

The projector is a cursor loop in orion-memory-consolidation over `substrate_graph_journal` and `episode_memory_event`. It:
- writes nodes, evidence and edges through `SubstrateGraphMaterializer.apply_record`;
- records a `SubstrateGraphMaterializationV1` with the canonical ids the materializer actually returned. If those differ from the deterministic ids, it fails closed and logs;
- sets `valid_to` on evidence edges when a memory is superseded, corrected, rejected or retired;
- updates node `aliases` and `promotion_state`.

It is idempotent and can be rebuilt from Postgres. It replaces revision 1's Graphiti projector.

#### 2.4 Lazy seeds, not bulk seeds

Revision 1 seeded 95 services, about 2,377 PRs and 6,107 files as rows. As substrate nodes, that would add about 8.5k nodes that no memory names to the cognitive graph. Instead:

- **graphify and git-log names start dormant.** `scripts/build_referent_candidates.py` writes alias rows with `node_id` NULL, `proposed_by=graphify|git_log`, and `source_build='graphify aff23fac0 2026-09-11'`. It covers services, PRs, and files (basenames only when unique).
- **A node is minted on first use.** The first time a memory or a Juniper prompt names a dormant name, a graphify node is minted as `provisional`, with the build label. It gets `observed_in` edges to the spec or PR report as `orion_self_knowledge`/graphify evidence.
- **Topic-foundry concepts are already substrate nodes, so we do not copy them.** When a writer `concept:` key matches a topic concept's label, that is an identity proposal (1.2 step 4), never a silent merge.

#### 2.5 What is gone from revision 1

- the `referent`, `referent_mention`, `referent_key_redirect` and `referent_event` tables;
- the `memory_crosswalk_v` view;
- the topic-segment co-evidence link, which returns only when topic concepts have a promotion path (a #2497 follow-up);
- the Graphiti projector and `as_of` endpoint.

### 3. "What did I know about X last week?" without Graphiti

Edges natively carry `valid_from`/`valid_to` (codec `:354-355`), and the journal records every state change with its time. The projector sets both (2.3), so the as-of view is one query:

```cypher
MATCH (r:SubstrateNode {node_id: $node_id})-[e:observed_in]->(ev:Evidence)
WHERE ev.evidence_type = 'episode_memory'
  AND e.valid_from <= $at AND (e.valid_to IS NULL OR e.valid_to > $at)
RETURN ev.content_ref, e.valid_from ORDER BY e.valid_from DESC LIMIT 20
```

A Postgres hydration of those memory ids follows. Each memory's confirmation state at `$at` comes from `episode_memory_event`.

Example: "What did I know about Hecate on 09-29?" returns nothing, plus "first heard of it 10-04" (the minimum `valid_from`). On 10-06 it returns the 5 Hecate memories.

**Recommendation: retire the Graphiti mirror's role in Stage 2.**
- Graphiti's claimed advantages were validity on edges and multi-hop walks. The substrate now has both: native validity and #2497's bounded neighborhood.
- Graphiti's live time fields are unused (0 of 28 edges).
- Keeping it would mean a second time-aware copy of the same facts.

The legacy approval sync runs untouched until Stage 4 retires the crystallization approval UI. Retiring the adapter itself is a Stage 4 decision, outside this spec.

### 4. Recall by referent: one seam replacing `concept_region` and the active packet

#### 4.1 The collector (`referent_region`, in orion-recall)

```
query text (retrieval_query, companion spec)
 1. extract focal referents (in-process, ≤ 5 ms)
    a. alias n-grams (1–5 tokens, longest first) against an in-memory map built every 60 s from
       referent_alias WHERE promotion_state IN ('provisional','canonical');
       descriptors only if valid and unique; proposed nodes only when named directly,
       flagged "unconfirmed identity"
    b. explicit ids (#2413 regex): "#2287", orion-* services, file paths -> dormant candidates resolve too
    c. rare tokens / rare bigrams (df ≤ 0.5%) not covered by (a): Postgres 'simple' FTS over
       episode_memory.statement + episode_memory_evidence.quote -> those memories' node_ids
       ("the new server" -> memory 63d7e72b -> Hecate)
    -> focal set (≤ 16, ranked by idf); none -> ABSTAIN
 2. evidence handles (ONE Cypher read, new bounded store method read_evidence_handles):
    focal (+ selected neighbors) -observed_in-> Evidence, filtered by evidence_type/voice/channel
    for the intent (4.2), valid now, newest first, ≤ 6 per referent
 3. neighborhood (Phase 3 only): read_neighborhood(focal, states=(provisional, canonical),
    budgets 4/8/8, direction=both) -> related referents via accepted assertions; receipt kept.
    Neighbors feed step 2 at hop 2 (≤ 2 per neighbor)
 4. hydrate (Postgres, by content_ref; the one provenance resolver from #2497 §5):
    memory rows + verified quotes; reverie/journal/claim text, truncated
 5. legacy lane until Stage 4: same aliases/tokens -> memory_crystallizations.summary (FTS)
 6. rank (lexicographic): Σ idf of matched focal referents (self referents -> 0);
    hop (focal 1.0, neighbor 0.5); purpose fits intent; effective strength; recency
 7. caps: ≤ 6 memories; ≤ 2 per referent only when 2+ referents are named; ≤ 3 other-voice
    evidence items; ≤ 1 legacy item per referent
 8. emit MemoryItemV1 + recall_reason {referents, matched, via, hop, assertion_state of the edge
    used, neighborhood receipt flags} + voice, channel, confirmation_state
```

This is #2497's "relevance-scoped neighborhood" for `concept_region`, made concrete:
- focal nodes are picked by exact alias/label match (no substring, no embedding);
- expansion is `read_neighborhood`;
- relation fragments carry assertion state and evidence handles;
- truncated and degraded receipts are passed through. A degraded read renders "related things unavailable", never an empty "nothing related".

Rendering never reinforces a memory. `recalled` events are batch-written after the reply. Every line goes through `orion/memory/voice_render.py` (rev 3 §7).

#### 4.2 PCR phases and the intent → memory mapping

**Phase 1 (continuity, before stance)** runs steps 1, 2 and 4 only, with no neighborhood. Contents, within 300 tokens:
- up to 3 referent hits;
- the last closed episode's memories (up to 3);
- due follow-ups (up to 2);
- recent sql_chat.

**Phase 3 (purposeful):**

| Intent | Memory purposes | Evidence types / voices allowed | Neighborhood |
|---|---|---|---|
| continuity | `happened` (last episode), due `follow_up` | `episode_memory`, chat | no |
| relational | `about_juniper`, `orion_view`, `happened` with a person other than Juniper or Orion | `episode_memory`, `chat_turn`, `journal` | yes (people and events) |
| semantic | `happened`, `about_juniper` | all, including `reverie`, `reading_*`, `curiosity_*`, graphify | yes |
| procedural | `follow_up`, `happened` with machine/service/file/pr | `episode_memory`, graphify evidence | yes |
| open_loop | open `follow_up`, `memory_tension_shadow` questions whose referents overlap the query | `episode_memory`, `curiosity_question` | no |
| contradiction | memories sharing a referent that are pending, corrected or superseded | `episode_memory` + the as-of query (section 3) | no |

**Intent fix:**
- `open_loop` fires only for persistent loops, or for an open `follow_up` whose referents appear in the turn.
- Relational and topic rules are evaluated first.
- **Referent rule:**
  - a person other than Juniper or Orion → `relational`;
  - a machine, service, file or pr → `procedural`;
  - any other kind → `semantic`.
- `rule_id` is logged on every call.

#### 4.3 Latency budget, measured against neighborhood reads

| Step | Budget (p95) | Basis |
|---|---|---|
| Extract | 5 ms | In-memory dictionary over ≤ 5-grams |
| Evidence handles (1 Cypher query) | 30 ms | A comparable bounded query takes 2.6–4.1 ms inside Falkor |
| Rare-token FTS + Postgres hydration | 30 ms | Indexed; about 5k memories a year (**UNVERIFIED** growth) |
| Legacy lane | 40 ms | FTS over about 1.4k crystallizations |
| **Neighborhood** (Phase 3) | **150 ms, hard timeout** | See below |
| Phase 1 total | p50 < 250 ms | Today 532 ms. No neighborhood |
| Phase 3 total | p50 < 400 ms, p95 < 800 ms | Today 1,199 / 2,150 ms |

**Why the neighborhood needs work first.** Today it takes 351–674 ms for one focal node (my replays) and 1,187 ms for #2497's 8-node hub set. The cost is round trips: one query per neighbor (`neighborhood.py:166-170`), per predicate group and per edge page.

**PR E** batches the neighbor and focal-node fetches into one `IN $ids` query per round (`nodes([...])`). That stays inside #2497's contract, and its receipts do not change.

If PR E cannot bring a 4/8/8 single-focal read under 150 ms p95 on replay, the neighborhood step stays **shadow-only** and the cutover goes ahead without it. Neighbors are an extra hop; recall of the named things themselves does not depend on them.

### 5. Reverie visual seed from validated memories

A new reader, `load_episode_memory_seed()`, goes in `services/orion-thought/app/store.py`.

**Eligible memories:**
- `purpose IN ('happened','about_juniper')`, `status='active'`, created in the last 7 days;
- no `downgraded_voice` or `rejected_invalid` event, and every quote verified;
- **gated on Juniper's stakes policy:** either `stakes='low' AND confirmation_state='auto'`, or any stakes with `confirmation_state='confirmed'`.

Under that policy, memories about health, family or relationships, feelings, and identity conclusions are `high` and `pending_confirmation`. They seed an image only after Juniper confirms them in conversation.

**Dependency:** the stakes-fix PR (in flight) must re-stake the 25 existing rows. They were all written `low`/`auto` before the policy existed, including 3 family memories and 1 about feelings. PR G does not merge before that re-stake is live.

**Behaviour:**
- **Rotation:** pick the eligible memory seeded least recently. This fixes "2 distinct seeds in 7 days".
- **Trace:** `VisualSourceV1(source_kind='episode_memory', source_id=memory_id)`.
- **Switch:** `REVERIE_VISUAL_MEMORY_SEED_SOURCE=episode_memory`, shipped ON. The crystallization reader is deleted at cutover (PR H).

### 6. Evals (label-free, `services/orion-recall/evals/referent/`)

1. **Known-item test from the writer's referents.**
   - For each provisional/canonical node with ≥ 1 memory, and each of its active name aliases, the query is "what do you remember about {alias}" and the gold set is the node's memories.
   - Gate: hit@8 ≥ 0.9.
   - "Hecate", "the Inspur" and "Inspur NF5288M5" each return all 5 Hecate memories; "the new server" returns `63d7e72b`.
   - Caveat: this tests the lookup, not whether the writer tagged the right things.
2. **Quote-held-out known item.** The query uses a rare term from a memory's Juniper quote that is in neither its statement nor its aliases. It measures the alias gap. Reported, not gated.
3. **Cousin rate against today.**
   - Replay 7 days of `stance_react` Phase 3 queries through both paths. A cousin is a rendered, non-feed item that shares no referent with the query.
   - Gates: shadow ≤ 20%, and distinctness ≥ 10× today's (103 distinct ids over 569 retrievals).
4. **Source monitoring.** Gate: 0 violations over 7 days.
   - None of the 180 Hecate reveries ever renders as "Juniper told me", "we" or "I told Juniper".
   - `juniper_said` never renders without a verified prompt quote.
   - Pending items carry "Unconfirmed".
   - `orion_read` items carry their title and claim status.
   - No person mention is indexed from `orion_read`.
5. **Graph discipline:**
   - 0 merges without a decision.
   - 0 walkable semantic edges without an accepted assertion: every `edge_role=semantic_projection` edge has an assertion in provisional/canonical at the same revision.
   - Reconcile never merges a `memory.referents` node by label or embedding. Fixture: the writer's `machine:circe` next to topic-foundry's "circe" opens an identity proposal.
   - A projector rebuild from Postgres gives identical node and edge id sets.
6. **Cognitive isolation.** On a fixed fixture, dynamics and attention outputs are identical before and after projecting the referent, evidence and assertion nodes and the provenance edges (#2497 acceptance 8).
7. **As-of consistency.** For each memory node and each of the last 30 days, the as-of result equals the Postgres replay of `episode_memory_event`. Fixtures cover a supersede, a rejection, and a date before the first evidence (Hecate on 09-29 is empty).
8. **Neighborhood receipts and latency.**
   - Replay the shadow focal sets through `read_neighborhood` and report p50/p95, the truncated rate and the degraded rate.
   - A degraded read never renders as "nothing related".
   - Collector p95 ≤ 100 ms without the neighborhood; the neighborhood ≤ 150 ms after PR E.
9. **Abstention honesty.** Report how often a query with no referent abstains, and how often a Juniper prompt has a rare token but no match.

**Metric gate for the ranking signal** (Σ idf of matched referents):
1. **Provenance:** `memory_count` per node, computed from `episode_memory_referent.node_id` and cached with the alias map.
2. **Independence:** it replaces salience ranking for these items.
3. **Theory:** inverse document frequency (Spärck Jones 1972), over Orion's own memories.
4. **Live data:**
   - Juniper has 8 referent rows across the 25 memories, so idf ≈ 1.0, but it is forced to 0 as a self referent.
   - Hecate (5 of 25) ≈ 1.4; Jackalope (1 of 25) ≈ 2.5.
   - With no referent the sum is 0 and recall abstains, a real rest state.
5. **Existing mechanism:** #2413's df table.
6. **Reversibility:** `RECALL_PCR_MEMORY_MODE=legacy`.

### 7. Rollout

#### 7.1 Shadow

`RECALL_PCR_MEMORY_MODE=shadow` ships ON. After each PCR recall returns, a background task (semaphore 2) runs `referent_region` and writes one row:

```sql
recall_referent_shadow(corr_id text PK, created_at timestamptz, verb text, phase text, intent text, rule_id text,
  query text, focal jsonb, abstained bool, live_ids text[], shadow_items jsonb,     -- id, why, voice, channel, hop
  neighborhood_receipt jsonb, cousin_live int, cousin_shadow int, timings_ms jsonb)
```

The Stage 1 report (`/memory/episodes/report` plus the daily markdown) gains a "recall" section showing, per day:
- live vs shadow for the last 10 chat recalls;
- cousin rate, distinctness, abstention and latency;
- neighborhood receipts;
- open identity proposals.

It is a report only; no notification is sent.

#### 7.2 Cutover criteria (all must hold for 7 consecutive days)

1. Known-item hit@8 ≥ 0.9.
2. Shadow cousin rate ≤ 20%, and at least 50 points below live.
3. 0 source-monitoring violations.
4. Collector p95 ≤ 100 ms without the neighborhood. The neighborhood counts only if it holds ≤ 150 ms p95; otherwise it stays shadow-only.
5. **No loss:** every live item that shared a referent with its query is in the shadow set, or the report explains why.
6. Evals 5 and 6 are clean.

#### 7.3 Cutover

Flip `RECALL_PCR_MEMORY_MODE=referent`. In the same PR, delete:
- the active packet's salience query and boost-on-read;
- the `concept_region` collector;
- the retriever's Chroma/Graphiti rails;
- the old reverie reader.

Nothing is kept as a fallback.

#### 7.4 Rollback

- **Before cutover:** set `RECALL_PCR_MEMORY_MODE=legacy`. Also set `MEMORY_REFERENT_PROJECTOR_ENABLED=false`, `MEMORY_REFERENT_INDEXER_ENABLED=false` or `MEMORY_COOCCURRENCE_AUTO_ACCEPT=false` as needed.
- **Graph:** projected nodes and edges are identified by `producer` and journal receipts. Deprecating them (#2497's retraction) removes their projections and keeps the history.
- **After cutover:** revert PR H and redeploy orion-recall.

#### 7.5 PRs, in order

| # | PR | Contents | Acceptance |
|---|---|---|---|
| **A** | `feat(substrate): shared assertion core` (contract patch, co-owned with #2497) | Durable `assertion` node kind (codec encode/decode); `assertion_subject`/`assertion_object` edges; native `edge_role` on edges; evidence `voice`/`channel` native properties; `substrate_graph_journal` + `SubstrateGraph{Proposal,Decision,Materialization}V1` (registry, channels, fixtures); `reconcile.py` identity branch for `memory.referents` nodes; the dynamics/attention eligibility predicate excludes assertions, provenance edges, evidence and `memory.referents` nodes | Codec round-trips every new field. Old edges decode as `edge_role=legacy_unreviewed`. Dynamics before/after fixture is identical. Reconcile fixture shows no label or embedding merge for fenced nodes. Reading can use the same contracts unchanged |
| **B** | `feat(memory): referents as substrate nodes + alias lifecycle` | `referent_alias` and `episode_memory_referent.node_id`; validator keeps aliases, adds `machine`, normalizes roles; persist writes aliases and journal (1.2); projector loop in orion-memory-consolidation (2.3) with memory evidence nodes and `observed_in` edges; `co_occurs_with` assertions under `source_cooccurrence_v1`; checkpoint alias-recovery backfill | 14 keys → nodes (Hecate and Circe are `entity_type=machine` via `kind_refined`). Hecate has 3 provisional grounded aliases. The writer's `machine:circe` vs topic-foundry "circe" opens an identity proposal and merges nothing. 25 memory evidence nodes have a valid `valid_from`. Accepted assertions include Vincent–austin-offsite and Hecate–Circe. Running the backfill twice gives an identical journal and graph |
| **C** | `feat(memory): referent mention indexer` | Indexer loop (+ read-only `FALKORDB_URI` for `orion_worldview`, `.env_example`, env sync); dormant graphify/git-log candidates (2.4); 90-day alias backfill under CLAUDE.md §14 | `machine:hecate` reverie evidence ≈ the live ILIKE count of 180, minus word-boundary misses. 0 from AI Town, metacog or refuted priors. 0 person edges from `orion_read`. Indexer lag p95 < 120 s. A graphify PR node is minted only on first mention |
| **D** | `feat(memory): voice renderer + intent fix` | `orion/memory/voice_render.py`; the `retrieval_intent.py` fix and referent rule; `MemoryItemV1` gains `recall_reason`/voice fields (consumer-first, registry) | `rule_id` histogram shows ≥ 3 intents over 7 days. Source-monitoring unit tests pass |
| **E** | `perf(substrate): batch neighborhood node fetches + read_evidence_handles` (in #2497's code) | One `IN $ids` node query per round in `neighborhood.py`; bounded `read_evidence_handles(node_ids, evidence_types, voices, per_node_limit)` in the Falkor, SPARQL and memory backends; hydration through the provenance resolver | Replay: 4/8/8 single-focal ≤ 150 ms p95. Receipts identical to before on fixtures. Evidence-handle read ≤ 30 ms p95 |
| **F** | `feat(recall): referent_region in shadow` | `services/orion-recall/app/referents/`; `recall_referent_shadow`; evals 1–9; report section | Shadow rows for ≥ 95% of stance_react recalls. Live latency unchanged. The evals report |
| **G** | `feat(thought): reverie seed from episode memories` | The section 5 reader, gated on the stakes-fix re-stake | ≥ 1 seed a day with `source_kind=episode_memory`. 0 seeds from `pending_confirmation` memories. ≥ 5 distinct ids over 7 days (when ≥ 5 are eligible) |
| **H** | `feat(recall): cut PCR over to recall by referent` | Flip the mode; delete the old paths (7.3) | 7.2 met. Over 24 h live: 0 retrieval events with 100 ids, 0 boost writes, Phase 3 p50 < 400 ms |

**Order:**
- A comes first. It blocks B, and #2497's reading pipeline too.
- B, then C.
- D runs in parallel with B.
- E runs in parallel with B and C.
- F needs B, D and E.
- G needs B and the stakes fix.
- H comes after the shadow week.

---

## Reconciliation with #2497: what is settled and what is still a real conflict

**Settled by this revision:**
- one graph, with referents as Entity/Concept nodes;
- evidence text in Postgres, reached through Evidence handles;
- only accepted assertions produce walkable semantic edges;
- one recall seam, through `read_neighborhood`;
- one journal;
- no Graphiti for time.

**Still genuinely in tension. These need the #2497 owner, Juniper, or both:**
1. **Who builds the assertion core, and in what order.** #2497 left it for a later reading patch, but memory needs it first. This spec proposes PR A as the shared contract. It also renames #2497's `ReadingGraph*V1` events to `SubstrateGraph*V1` with a `proposal_kind`, which changes #2497's proposed contract.
2. **Auto-acceptance.** #2497 recommends operator review as the initial gate, and asks that any self-accept policy be named and tested. This spec defines two named policies, each with tests and a kill switch:
   - `alias_grounding_v1`: an alias is provisional when it appears in Juniper's verified quote;
   - `source_cooccurrence_v1`: `co_occurs_with` only, from one verified prompt quote, at most 6 per memory.

   Without them, nothing memory-derived would ever be walkable, because no review surface exists for these yet.
3. **Default eligibility hides all existing topic structure.** 851 of 855 concepts and 1,610 of 1,610 entities are `proposed`. After cutover, chat loses the topic concepts `concept_region` surfaces today, until #2497's promotion path exists. This spec does **not** opt recall into `proposed`; #2497 calls that a diagnostic opt-in. This is a known behaviour loss.
4. **Neighborhood latency.** Measured at 351–674 ms (one focal node) and 1,187 ms (8 focal nodes), against a 150 ms chat budget. PR E's batching changes #2497's code. If it falls short, the neighborhood stays shadow-only in recall.
5. **Label and embedding identity merges** in `reconcile.py` contradict both specs' "no automatic identity by label or embedding". This spec fences off only its own nodes. Fixing topic-foundry's paths is #2497/substrate debt.
6. **`visibility_scope`** is proposed in #2497 but not in the schema. Until it lands, memory evidence nodes about Juniper rely on the internal API boundary; anchor scope is not an access control.

---

## Proposed schema / API changes

- **Substrate (PR A):**
  - durable `assertion` node kind, with `assertion_subject`/`assertion_object` edges;
  - native `edge_role` (`semantic_projection | provenance | ontology_membership | legacy_unreviewed`);
  - native `voice`/`channel` on evidence;
  - `SubstrateGraph{Proposal,Decision,Materialization}V1`, `substrate_graph_journal`, and channels `orion:substrate:graph:{proposal,decision,materialized}`;
  - a reconcile identity branch;
  - an exclusion in the eligibility predicate.
- **Store API:**
  - new `read_evidence_handles(...)` (PR E);
  - batched node fetch inside `read_neighborhood`, with no contract change.
- **Postgres:**
  - new tables: `referent_alias`, `referent_indexer_cursor`, `recall_referent_shadow`, `recall_term_stats` (#2413);
  - `episode_memory_referent.node_id`;
  - a 'simple' tsvector GIN index on statement and quote;
  - `recall_telemetry` gains `query_referents`, `abstained`, `selected_reasons`, `ranking_mode`.
- **Schemas:**
  - the validated referent keeps `aliases`;
  - `MemoryItemV1` gains `recall_reason`, `voice`, `channel`, `confirmation_state` (consumer-first);
  - the closed set of `entity_type` values is documented.
- **Dropped from revision 1 and rev 3:** `referent`, `referent_mention`, `referent_key_redirect`, `referent_event`, `episode_memory_link`, the Graphiti `memory_projection`/`as_of` endpoints, `memory_projection_cursor`.
- **Env.** Update each `.env_example`, then run `python scripts/sync_local_env_from_example.py` and report any keys skipped by `SYNC_PREFIXES`. All flags ship ON.
  - orion-recall: `RECALL_PCR_MEMORY_MODE=shadow`, `RECALL_REFERENT_ALIAS_REFRESH_SEC=60`, `RECALL_RARE_TERM_MAX_DF_FRAC=0.005`, `RECALL_NEIGHBORHOOD_TIMEOUT_MS=150`.
  - orion-memory-consolidation: `MEMORY_REFERENT_PROJECTOR_ENABLED=true`, `MEMORY_REFERENT_INDEXER_ENABLED=true`, `MEMORY_REFERENT_INDEXER_TICK_SEC=60`, `MEMORY_COOCCURRENCE_AUTO_ACCEPT=true`, `FALKORDB_URI`, `FALKORDB_SUBSTRATE_GRAPH=orion_substrate`.
  - orion-thought: `REVERIE_VISUAL_MEMORY_SEED_SOURCE=episode_memory`.

## Files likely to touch

- **A:**
  - `orion/core/schemas/cognitive_substrate.py`;
  - `orion/substrate/falkor_codec.py`, `falkor_store.py`, `graphdb_store.py`, `materializer.py`, `reconcile.py`;
  - eligibility in `dynamics.py` and in attention;
  - `orion/schemas/registry.py`, `orion/bus/channels.yaml`;
  - `services/orion-sql-db/manual_migration_substrate_graph_journal_v1.sql`;
  - tests and fixtures.
- **B:**
  - `orion/memory/referents/` (new), `orion/memory/episode/{validate,store}.py`, `orion/schemas/memory_episode.py`;
  - `orion/cognition/prompts/memory_episode_distill.j2`;
  - `services/orion-memory-consolidation/app/referent_projector.py`;
  - `scripts/backfill_referents_from_episodes.py`;
  - a migration.
- **C:** `services/orion-memory-consolidation/app/referent_indexer.py`, `settings.py`, `.env_example`, `docker-compose.yml`, `README.md`, `scripts/build_referent_candidates.py`.
- **D:** `orion/memory/voice_render.py`, `orion/memory/retrieval_intent.py`, `orion/core/contracts/recall.py`.
- **E:** `orion/substrate/neighborhood.py`, `neighborhood_backends.py`, `store.py`, the provenance resolver.
- **F:** `services/orion-recall/app/referents/`, `worker.py`, `pcr_collectors.py`, `evals/referent/`, `orion/memory/episode/report.py`.
- **G:** `services/orion-thought/app/store.py`, `.env_example`.
- **H:** `services/orion-recall/app/collectors/{active_packet,concept_region}.py`, `orion/memory/crystallization/retriever.py`, `services/orion-thought/app/store.py`.

## Non-goals

- Vectors or embeddings for identity, linking or ranking.
- A second graph store, or any Graphiti role in Stage 2.
- Silent merges.
- Opting recall into `proposed` nodes.
- Predicates other than `co_occurs_with` from memory.
- Bulk-seeding graphify artifacts into the cognitive graph.
- The Stage 3 "Orion is asking" producer and outcome consumer. Stage 2 writes only proposals and `memory_tension_shadow` rows.
- The pageindex fixes (a separate track).
- Fixing topic-foundry's label and embedding merges.
- Retiring the Graphiti adapter (a Stage 4 decision).
- Distilling non-chat sources into memories.

## Acceptance checks (Stage 2 as a whole)

1. Each PR A–H meets its row in 7.5.
2. **Hecate end to end (live shadow).** The query "is Hecate flashed yet?" gives:
   - `focal=[referent-…hecate]`;
   - all 5 Hecate memories: 3 rendered "Juniper told me (10-04)…", 2 as Orion's own words;
   - at most 3 reveries, rendered "Something I was turning over on my own (reverie, …)";
   - Circe as a hop-2 neighbor through an accepted `co_occurs_with` assertion, with the assertion id in `why`;
   - the neighborhood receipt attached;
   - intent `procedural`, unless an earlier rule fires.
3. "the Inspur" (rare token) and "the new server" (phrase lane) reach the same memories.
4. A query with no referent ("how are you feeling tonight") gives `abstained=true` and no filler memories.
5. **As-of:** Hecate on 09-29 is empty; on 10-06 it returns 5 memories.
6. **Reverie:** a `reverie_visual_chain` row carries `source_kind=episode_memory` and an eligible memory id. 0 rows come from pending memories.
7. No LLM calls outside the existing distiller. 0 Graphiti writes from memory code.

## Proposal-mode disclosure

- **Capability change:**
  - The people, places, machines, projects and ideas in Orion's life become nodes in its one graph, with Juniper's names for them.
  - Everything Orion has about each thing hangs off it, labelled with whose voice it is.
  - Recall starts from the things a message names, and says why each item came back and whose words it is.
  - Orion can say what it knew about something at a past time.
  - Reverie images come from real recent memories.
- **Data touched:**
  - **Reads:** `episode_memory*`, the distill checkpoints (once), non-AI-Town chat, non-metacog journals, reveries, dreams, claims, snapshots, Falkor `orion_worldview`, the graphify bundle, `git log`.
  - **Writes:** `referent_alias`, `substrate_graph_journal`, the indexer cursors, the shadow table and telemetry columns, `memory_tension_shadow`, and Falkor `orion_substrate`. The Falkor writes are new Entity/Concept/Evidence/Assertion nodes and provenance, assertion and semantic-projection edges, all with `producer=memory.referents`.
- **Privacy boundary:**
  - Everything stays on the host.
  - Evidence text never enters the graph; only `content_ref` does.
  - AI Town and metacog are never indexed.
  - People are never linked from external reading.
  - High-stakes memories (under Juniper's policy) render "Unconfirmed" and never seed images until confirmed.
  - `visibility_scope` is still pending (#2497), so access relies on the internal API boundary.
  - This public spec names keys only.
- **Trace:**
  - journal rows for proposal → decision → materialization, with the real ids;
  - `edge_role` and assertion ids on edges;
  - `recall_referent_shadow.shadow_items[].why`;
  - neighborhood receipts;
  - `VisualSourceV1.source_id`.
- **Dangerous failure modes:**
  - (a) **A wrong merge.** Mitigated by merging only on decisions, plus reconcile fencing.
  - (b) **Source blending.** Mitigated by voice on every evidence node, the renderer contract, and eval 4 as a cutover gate.
  - (c) **A bad alias.** Mitigated by grounding, `why` on every item, collision proposals and descriptor expiry.
  - (d) **New nodes silently changing attention or dynamics.** Mitigated by the eligibility exclusion and eval 6.
  - (e) **A partial neighborhood shown as "nothing related".** Mitigated by rendering receipts; degraded is never shown as empty.
  - (f) **Stale graphify.** Mitigated by the build label.
  - (g) **Auto-accept policies letting junk structure in.** Mitigated by two narrow named policies with kill switches, covering only `co_occurs_with` relationships.
- **Rollback:** see 7.4. Each subsystem has its own flag. Graph projections are retracted by producer and receipt, never by deleting the graph.

## Open product questions (for Juniper)

1. **When should recall by referent become primary: at Stage 2, once the 7.2 gates pass, or at Stage 4 as rev 3 planned?** Recommendation: Stage 2. Today's path ignores the query (569 of 569 retrievals return the same-shaped 100 rows) and reinforces junk on every read.
2. **Should descriptor names like "my boss" or "my sister" resolve while live (90-day expiry, grounded in your words, collisions become questions), or only after you confirm them?** Recommendation: resolve while live.

## Recommended next patch

**PR A: the shared substrate assertion core, agreed with #2497's owner first.** It unblocks both the memory referents and the reading pipeline, and it is where the one-graph decision actually becomes code.

**In parallel, PR D:** the voice renderer and the intent fix. It does not depend on PR A and is useful immediately.
