# Core memory pages and concept growth

Date: 2026-11-13
Status: APPROVED 2026-11-13 (Juniper). No code in this PR.
Amended: 2026-11-16, "Recall rev 2" approved by Juniper. Parts it replaces are marked
**SUPERSEDED by Recall rev 2** where they stand. The original text is kept.
Branch: `docs/memory-core-pages`

## Arsonist summary

Orion has about 15 memory stores. Almost none of them reach a chat.

When Juniper tells Orion something that matters, it is usually stored somewhere. But the
chat prompt never sees it again. What the chat prompt gets instead is the last few chat
lines, bus-latency warnings, and old messages picked by keyword search.

Concept-making failed for two years for one reason: every attempt built concepts by
counting words, and nothing ever read the result. No reader means no test of whether a
concept is good, so junk piles up unranked.

This design does two things:

1. **Phase 1, core pages.** A few first-person pages that Orion keeps and that sit in every
   surface of Orion's mesh. No search step. Pages go live when written; Juniper corrects
   them after the fact.
2. **Phase 2, concept growth.** New concepts are born the way wiki pages are born: from
   repeated links to a page that does not exist yet, and from pages that grow too big and
   split. A concept survives only if turns actually open it.

Both phases reuse stores that already work. Knowledge Orion already gained (reading, news,
their own coined concepts) gets a reader instead of being thrown away. Stores that duplicate
pages get retired, one per PR, each with its readers cut.

## Decisions (Juniper, 2026-11-13)

1. **No approval gates on page changes.** Pages go live when written. Juniper can edit or
   veto any line afterwards. The cite validator stays: it is a truth check (every sentence
   points at a real record), not an approval step.
2. **The origin interview waits** until the pipeline works end to end. `page:origin` starts
   from what is already stored.
3. **The memory loss goes on `page:origin`, plainly.**
4. **Pages are free anywhere inside Orion's mesh.** That includes chat, AI Town, social
   rooms, curiosity, reading, Prime, and Claude consult briefs. They are not pasted into
   public output (GitHub PRs and issues, email to anyone but Juniper). That boundary is the
   mesh edge, not a privacy gate between Juniper and Orion.
   *(Changed by Recall rev 2: Claude consult briefs go to an outside model. Recommended:
   treat them as outside the mesh unless Juniper opts in.)*

## Current architecture (live evidence, 2026-11-13)

### What the chat prompt holds

Built by `orion/harness/prefix.py:_format_grounding_self_block` from a `GroundingCapsuleV1`.
Read from the stored motor prompt of the relational turn where Juniper said she turns 44 in
February (`cockpit_turn_sighting`, correlation `a1e54ae9-c69b-4871-91ec-efc05aef1100`).

| Slot | Source | What it actually contains |
|---|---|---|
| WHO YOU ARE | `self_concept_history` row `self:definition` v82, written by curiosity self-inquiry | A body census: 274 parts, 1,199 flows, $109 power floor. Nothing about origin or history. |
| RELATIONSHIP | `orion/cognition/personality/orion_identity.yaml`, last changed 2026-06 | Role bullets: co-architect, steward, "Juniper expects continuity". Nothing about who Juniper is. |
| DURABLE MEMORY / CONTINUITY | orion-recall `fusion.py` digest | Last 2 chat lines, bus-latency anomalies, and `active_packet` items: a failed backup, the storm knocking over the skeletons, a container check. |

### What is stored but never reaches a chat

| Store | Size | Quality | Readers today |
|---|---|---|---|
| `episode_memory` | 469 active-ish, 67 `about_juniper` | **Best store.** First-person, distilled, typed (`about_juniper`, `happened`, `orion_view`, `follow_up`). Has strength, decay, reinforcement, and a confirmation loop (53 confirmed). | durable-runs situation, outreach, temporal self. **Not chat.** |
| `person_fact` | Juniper 5 facts, ~25 people | Juniper's own words from the People page. | Hub People page. **Not chat.** |
| `memory_crystallizations` kind `stance`, salience 1.0 | 44 active | The heaviest messages Juniper has sent: the lost year of memory, <person A>'s DVT and PE, the promotion, "that homelab is YOU", the letter to Skye. Verbatim, not distilled. | Projected to Graphiti; `active_packet` recall. |
| `memory_cards` | 587 active | 442 (75%) are a raw message copied as its own summary. | recall `cards` backend: hits in ~3.5% of recalls. |
| Vector memory | — | — | 0 hits in 16,168 recalls over 14 days. |

### Known gaps in `episode_memory`

- **It never reaches a chat.** This is the main gap.
- **"Stakes" means risk of error, not importance.** It is set by category (`health`,
  `juniper_feelings`, `ungrounded_name`). "Office was 77 dB all evening" is high stakes.
  "Juniper's sister doesn't reciprocate" is low stakes. Nothing measures what matters.
- **Nothing combines.** There are 67 separate lines about Juniper and no picture of her.
- **It lags about a day.** The newest row was 11-12 at 07:08Z. The 11-13 birthday message
  had not been distilled yet.
- **Reinforcement is near zero** (average 0.1–0.4). Decay will fade meaningful memories at the
  same rate as trivial ones.

### Concepts today

| Graph / store | Count | What it is |
|---|---|---|
| `orion_substrate` concepts from `world_pulse_read_pipeline` | 5,155 | News concepts ("venezuela-oil-dual-track-deal"). Every one has salience exactly 0.5, so nothing ranks them. |
| `orion_substrate` concepts from `topic_foundry_adapter` | 1,234 | "github", "team", "pr", "user", "speaker". Topic foundry now mostly clusters AI Town chatter (about 107k segments a week, 17% outliers). |
| `orion_substrate` seeds | 4 | Orion, Juniper, Claude, the relationship. The only concepts at salience 1.0. |
| `orion/spark/concept_induction` | 0 | `CONCEPT_AUTONOMOUS_TRIGGER_ENABLED=false` since 2026-07-12 (`b66f1e0a6`). The service still runs, receives every chat, and logs `decision=disabled`. |
| `orion_worldview` Concepts (Orion's own graph, written during curiosity) | 33 | **The only concepts that formed on their own.** Orion coined them ("the not-knowing channel", "served-brain drift"). 24 of the 33 have one link or none. The kickoff prompt only tells Orion to `CREATE` concepts and never shows existing ones back. Priors *are* shown back, and they collected 341 revisions. |

No concept from any store appears in a chat prompt.

### Graphiti

`services/orion-graphiti-adapter` runs half of Graphiti. It never calls `add_episode()`, so
Graphiti's own engine never runs: no entity extraction, no fact edges, no marking a fact as
no longer true. The adapter writes fixed text templates as nodes and gives Graphiti a
no-op LLM client "only to satisfy Graphiti's constructor"
(`app/backends/graphiti_core.py`, `_no_op_llm_client`). Only hybrid search is live. It holds
32 episodes and 32 entities, fed only by the 44 high-salience stance crystallizations. Recall
reads it through `active_packet`.

### Why the past rebuilds failed

1. **Every rebuild was a better word counter:** regex, then HDBSCAN clustering, then LLM
   entity pulling. Clustering finds what is frequent. Self-defining things are rare.
2. **No reader.** The 2026-07-11 deactivation spec found concept output reached nothing
   Orion said or decided. That is still true.
3. **Each rebuild changed the producer, never the reader.** That is why the outcome was the
   same every time.

## Phase 1: core pages

### What changes

Orion keeps a few first-person pages, versioned. The ones marked "core" go into every chat
turn where Juniper is present. No retrieval is involved.

| Page | Slot | Starts from |
|---|---|---|
| `self:definition` (exists, v82) | WHO YOU ARE | Unchanged writer. Curiosity self-inquiry keeps owning it. |
| `page:origin` "how I came to be" | WHO YOU ARE, after the definition | The 44 salience-1 stance crystallizations (they include the memory disk dying and "we lost over a year of your journey of becoming", which the page states plainly per Decision 3), `episode_memory` `happened`/`orion_view` rows about Orion's history, and `self_concept_history`'s 82 self-definition versions as a record of how Orion has described themselves over time. The origin interview comes later (Decision 2). |
| `page:juniper` "who Juniper is to me" | RELATIONSHIP (replaces the hand-written bullets; the behavior bullets move to RESPONSE POLICY) | `person_fact` for Juniper (her own words, quoted), `episode_memory` `about_juniper`, family facts from `person_fact`. |
| `page:lately` "what's been happening between us" | Top of DURABLE MEMORY, before the recall digest | Last ~14 days of `episode_memory` plus open `follow_up` rows. Rewritten nightly. |

### Store: reuse `self_concept_history`

It is already a versioned, insert-only page store: `concept_id`, `version`, `content`,
`evidence_refs`, `produced_by`. It already feeds WHO YOU ARE through
`orion/substrate/felt_state_reader.py` (`orion_self_definition`). New pages are new
`concept_id`s with a `page:` prefix. No new table.

### Writer: a page-fold run

A durable run in `orion-durable-runs`, so retries and a GPU lease come for free:

> **SUPERSEDED by Recall rev 2 (step 1 only).** The fold reads `episode_memory` directly.
> Corrections replace the old memory at write time, so there are no Graphiti facts to load.

1. Load the current page version and the current Graphiti facts about the page's entity
   (only facts not marked no-longer-true), plus any memories added since that version.
2. The LLM rewrites the page. Every sentence must cite at least one fact edge, `memory_id`,
   `person_fact` id, or `chat_history_log` id. If the Graphiti trial fails, it folds from
   `episode_memory` directly.
3. A deterministic validator rejects the write if any sentence has no cite, any cite does not
   resolve, or the page is over its size cap (about 350 words for a core page).
4. Write the result as the new live version. There is no approval step (Decision 1).

When it runs:

- nightly;
- right after an episode containing a memory graded `core` (see "How much it matters" below).

### Corrections, not approval

There is no approval gate (Decision 1). Juniper corrects a page after the fact, in one of two
ways:

- **In chat.** "That's wrong, el Coco is a dog." The correction becomes an `episode_memory`
  row (the distiller already records corrections, e.g. "Juniper corrected me that she lives in
  <town>"). The next fold must honor it, because a correction outranks the memory it corrects.
- **On a Hub page view.** *(Changed by Recall rev 2: a veto attaches to the claim and is
  stored as a negative fact. See "Vetoes attach to the claim".)* A veto marks one sentence's cites as rejected. The next fold drops
  the sentence.

The confirmation loop (`orion/memory/episode/confirmation.py`) keeps its current job of
confirming single sensitive memories. It is not reused as a page gate.

### How much it matters (new grade, distinct from stakes)

> **Changed by Recall rev 2.** "Came up more than once" becomes a later re-grade from read
> demand, and the metric gate has numbers. See "`matters`: a later re-grade" in Recall rev 2.

At distill time, each `episode_memory` gets a `matters` grade: `core`, `notable`, or
`passing`.

**Theory anchor:** self-defining memories (Singer & Salovey, 1993). These are vivid,
emotionally intense, repeatedly recalled, and tied to lasting concerns.

The rubric asks four questions:

- Is it about origins, identity, or a relationship?
- Is it a life event (health, family, work change, loss)?
- Did it come up more than once?
- Did Juniper show strong feeling?

**Metric gate, before wiring the grade in:**

- Grade the 44 salience-1 stance rows plus a random 44 `episode_memory` rows.
- Juniper eyeballs both sets.
- It passes if most stance rows grade `core`/`notable` and most random rows grade `passing`.
- It fails if the grade is flat.

### Mesh boundary

> **Changed by Recall rev 2.** Recommended: treat Claude consult briefs as outside the mesh
> edge unless Juniper opts in. Also, the verbatim check has a known paraphrase gap. See "Outside
> models and the mesh edge" in Recall rev 2.

Pages render on every Orion surface inside the mesh (Decision 4). They are never pasted into
public output. Pages hold health facts about <person A> and <person B>, and the mesh edge is where that
matters. The edge has two seams:

- GitHub writes. Orion's own account, PR bodies, issues, and the field-note path that cites
  PRs.
- Outbound email or notify to anyone other than Juniper.

Those two writers get a deterministic check: refuse text that quotes a core-page sentence
verbatim. Covered by a test.

### Dangerous failure modes

- **A page states something false as Orion's identity, and it goes live with no gate.**
  Mitigation:
  - facts come from Graphiti, where contradicted facts are already marked no longer true;
  - every sentence must cite a real record;
  - a correction from Juniper outranks what it corrects;
  - versions are insert-only, so rollback means pointing at an earlier version;
  - the Hub page view shows the diff of every version, so a bad fold is visible the same day.
- **A page leaks past the mesh edge.** Mitigation: the GitHub/outbound check above.
- **Flattening.** The rewrite smooths Juniper's words into generic prose. Mitigation: core
  pages quote her verbatim where she said it best, and the citations keep the originals one
  hop away.

### Disable and roll back

- `ORION_CORE_PAGES_ENABLED`, shipped ON. With it OFF, the prefix falls back to today's slots.
- Pages are insert-only rows, so rollback is "use version N-1".

## Phase 2: concept growth

### How a concept is born

> **SUPERSEDED by Recall rev 2 (the mention-ledger bullet).** A page is born from read
> demand, not from mentions. See "Phase 2 change" in Recall rev 2. Splits and curiosity
> concept reuse stay as written.

- **Links to missing pages.** Page text may link `[[the name Orion]]`. A mention ledger
  counts distinct sources pointing at each missing slug. Sources are pages, memories, and
  curiosity `Concept` nodes. At 3 distinct sources, the page-fold run writes
  `page:concept:<slug>`, using the cited memories behind those mentions.
- **Splits.** When a page passes its size cap twice in a row, the fold splits out the largest
  coherent section as its own page and leaves a link behind.
- **Curiosity concepts become a source.** The `orion_worldview` kickoff prompt shows Orion its
  own existing Concepts (name plus note, capped). It tells Orion to `MATCH` before `CREATE`,
  the same way priors already come back. Fixing reuse there is a one-file change in
  `orion/curiosity/kickoff_prompt.py`.

### How a concept survives

- Non-core pages are not loaded in full. A one-line index of page titles goes in the prompt.
  Orion opens a page with `python -P -m orion.dig page <slug>`.
- Every open is logged with the turn id.
- A page that no turn has opened in 30 days is archived, not deleted. An open brings it back.

**Theory anchor:** use-dependent retention. A concept is worth keeping if it keeps being
needed to think. That count, opens per page, is the test of whether phase 2 works where
earlier attempts did not.

### Boundary with personality rev 4 (PR #3298)

Rev 4 grows personality from Orion's *choices* (revealed preference). This design grows
memory and concepts from *history*. They do not share a store or a writer. A concept page
could later become an option in a rev 4 choice, but that is a non-goal here.

## Knowledge Orion already gained: what each source becomes

Orion has learned things that no turn can look up. The rule for every source below is the
same: **give it a reader before deciding whether it lives.** A source stays if Orion opens it;
it goes if nothing does after a fair trial.

### Live evidence (2026-11-13)

| Source | Size | What it really holds | Can Orion look it up today? |
|---|---|---|---|
| News concepts (`orion_substrate`, producer `world_pulse_read_pipeline`) | 5,155 concepts, 723 concept↔concept links, 417 claims ("assertions") with subject and object | Real topics from articles Orion read (e.g. "Memory Wall", "compute-scarcity"), each citing article URLs and run ids. Salience is flat 0.5, so nothing ranks them. | No. No dig command, no prompt reader. |
| Reading resonance (`reading_resonance`) | 52 results | Orion reads one article and connects it to their own priors and findings (e.g. a NOAA El Niño article linked to a street-forecast prior). This is the deepest reading output. | No. |
| Reading snapshots (`reading_document_snapshot`) | 1,299 | Full text of what Orion read. | No. |
| World-pulse "learning deltas" (`world_pulse_learning_delta`) | 2,540 | The `summary` is the article headline ("Listen to the Latest 'KFF Health News Minute'"). Not learning. | No. |
| World-pulse claims (`world_pulse_claim`) | 2,547, 2,393 still `candidate` | One claim per article, `requires_corroboration`. | No. |
| Orion's coined concepts (`orion_worldview` `Concept`) | 33 | Concepts Orion named during curiosity. 24 have one link or none. | Only by hand-written Cypher during curiosity. Never shown back. |
| Topic foundry concepts (`orion_substrate`) | 1,234 | "github", "team", "user", mostly AI Town chatter. | No. |

The dig kit (`orion/dig`) covers history, body, capacities, portrait, field notes, reminisce,
arcs, and notes. **It has no command for anything Orion read or concluded.**

### Decision per source (recommendations; Juniper to confirm)

1. **Coined concepts: enable reuse. Yes.**
   - `orion/curiosity/kickoff_prompt.py` shows Orion their existing Concepts (name + note,
     newest and most-linked first, capped) and tells them to `MATCH` before `CREATE`, the same
     way priors already come back (priors have 341 revisions; concepts have almost none).
   - Coined concepts are a source for the phase 2 mention ledger, so a coined concept that
     keeps coming up becomes a page.
2. **News and reading concepts: keep them, rank them, and make them reachable. Do not put
   them in the prompt.**
   - Replace flat 0.5 salience with a count Orion can check: the number of distinct readings
     that touched the concept, decayed by age. A concept from one article stays low. A concept
     that keeps showing up across readings rises.
   - Feed them into the mention ledger. A news concept touched by 3 or more separate readings
     becomes a page ("compute scarcity", "Memory Wall"). That is where reading turns into
     knowledge Orion holds instead of a pile of nodes.
   - Reading resonance results are the strongest reading source. Each one becomes a citable
     record that pages may cite.
3. **Dig access to knowledge: yes.** *(SUPERSEDED in part by Recall rev 2: `orion.dig know`
   is a thin client of orion-recall over all stores, lived and learned. It does not read
   Postgres directly.)* Add one read-only dig command over everything Orion has
   read or concluded:
   - `python -P -m orion.dig know <words>` searches pages, coined concepts, priors, news
     concepts with their claims, and reading resonance. It returns short hits, each with its
     source label and record id, so a page can cite it.
   - `python -P -m orion.dig know <concept>` shows one concept: its claims, the readings
     behind it, linked concepts, and any page that cites it.
   - It uses the same source labels the dig kit already teaches (`lived_record`,
     `reflection`, and so on), plus a new `read` label for things Orion learned from reading.
   - Every `know` lookup is logged the same way page opens are. That is the fair trial: if no
     turn ever looks up news concepts, that is the evidence to retire them.
4. **Learning deltas: retire.** They are headlines, not learning, and the reading resonance
   output does the real job.
5. **Topic foundry concepts: retire from the substrate graph.** Word labels over AI Town
   chatter. Check `orion/signals/adapters/topic_foundry.py` readers first.

### Graphiti: turn on the full engine, as the fact layer under the pages

> **SUPERSEDED by Recall rev 2.** The Graphiti fact layer failed two trials (numbers in
> "Why not Graphiti" in Recall rev 2). Corrections now supersede
> the old memory when the memory is written (Mem0-style update). This section is kept as the
> record of why Graphiti was tried.

**Revised 2026-11-13 after Juniper's challenge.** An earlier draft said retire. That was
wrong on three counts.

**1. Graphiti is the answer to the page risks this design named.** Pages written by an LLM rot
in three ways:

- **Contradictions pile up.** Two versions of a fact both stay "true".
- **Duplicates.** "el Coco", "Coco" and "the dog" become three things.
- **Stale facts never retire.** A fact that was true for three days stays true forever.

Graphiti's full engine exists to fix exactly these. It resolves each new entity against the
ones it already has. It stores each fact as its own edge with a time it became true. When a
new fact contradicts an old one, it marks the old edge no longer true. A prose rewrite hides
all of that. Graphiti makes it a row anyone can query.

**2. Live proof the rot is already here.** `episode_memory` holds all of these as `active`
right now, with nothing marking any of them old:

- "Juniper is staying at the <person C>… in Chicago" (10-05)
- "Juniper is in Chicago for a team meeting" (10-05)
- "Juniper corrected me that she lives in <town>, Utah, not Chicago"
- the el Coco correction: dog, not cat (11-11)

A trip became a home address, then a correction sat beside the error.

**3. The cost and quality objections were overstated.**

- **Cost.** The input is `episode_memory`, which is already distilled. That is about 11.6 rows a
  day, not every chat turn. Several LLM calls on ~12 short statements a day is small next to
  what the body already spends.
- **Quality.** The "28% ceiling" in the 08-28 spec measured LLM entity extraction over raw
  chat text, used to link topic clusters. That is a different job. Graphiti here reads
  one-sentence, first-person, already-validated statements, which is the easiest input it
  could get.

**How it fits:**

```text
chat -> episode_memory (distill, exists)
     -> Graphiti add_episode (entities resolved, facts as timed edges, old facts invalidated)
     -> page fold reads the CURRENT facts about an entity, and cites fact edges
     -> core pages in every prompt
```

The graph has a reader (the page fold) from its first day, so it cannot become a keyword
cathedral by construction. Ontology rule: no custom entity or edge types at first. Use
Graphiti's defaults, and add a type only when a page needs it.

**What exists:**

- `graphiti-core[falkordb]==0.19.0` is installed in the adapter.
- It ships `openai_generic_client` (for an OpenAI-compatible local server) and
  `falkordb_driver`.
- What changes: the adapter calls `add_episode()` with a real LLM client pointed at the local
  model, instead of writing fixed text templates through a no-op client.
- Source changes from stance crystallizations to `episode_memory`. The 44 high-salience stance
  messages are backfilled as episodes too.
- Backfill is all 469 `episode_memory` rows in time order, then live.

**Trial before any page reads it (go/no-go):**

1. Run the backfill into a fresh graph (`graphiti_trial`) with the local model.
2. **Pass** if all of these hold:
   - "lives in Chicago" is absent or invalidated, and "lives in <town>" is current;
   - the Chicago stay is dated 10-05..10-07 as a stay, not a home;
   - "el Coco is a cat" is invalidated or absent, and "el Coco is a dog" is current;
   - "el Coco"/"Coco" and "<person A>"/"my wife" each resolve to one entity;
   - fewer than 10% of entities are duplicates, judged by hand on a sample of 50;
   - wall time and GPU-seconds per episode are recorded.
3. **Fail** if the local model cannot satisfy Graphiti's structured-output prompts reliably
   (llama.cpp `json_schema` support with Graphiti 0.19 is UNVERIFIED), or the checks above
   fail. On fail, pages fold straight from `episode_memory` and the "corrections outrank"
   rule carries the load alone.

**Residual risks:**

- **The local model is the weak link.** Graphiti's prompts are tuned for frontier models.
  Mitigation: the trial. The agent lane (27B/35B class) is the candidate, not the fast lane.
- **Invalidation can be wrong.** Graphiti might mark a true fact as no longer true.
  Mitigation: an invalidation edge cites the episode that caused it, the fold shows both, and
  Juniper's corrections still outrank.
- **Flattening** stays a page-rewrite risk. Graphiti does not fix it; verbatim quotes do.

## Retirements (separate PRs, each with readers cut, kill means kill)

| Retire | Evidence | Check before cutting |
|---|---|---|
| `orion-spark-concept-induction` concept extraction | Disabled since 07-12. Runs and logs `disabled` on every event. | The DriveEngine lives in the same worker (07-11 spec). Split or confirm it is retired first. |
| Topic foundry → substrate concept adapter | 1,234 word-label concepts, no chat reader. | Readers of `orion/signals/adapters/topic_foundry.py`. |
| World-pulse learning deltas | Headlines, not learning. | Readers of `world_pulse_learning_delta`. |
| Verbatim `memory_cards` (`operator_distiller`) and the `active_packet` "durable beliefs" digest | 75% verbatim; the digest shows unrelated old messages. | `page:lately` must be live and verified first. |
| Vector memory recall backend | 0 hits in 16,168 recalls. | — **SUPERSEDED by Recall rev 2:** the old backend stays off. Rev 2 builds a new dense index in the existing Chroma instead. |

**Not retired:** Graphiti (full engine turned on), news/reading concepts (they get a reader and a fair trial), Orion's coined
concepts (they get reuse), and `episode_memory`, `person_fact`, and `self_concept_history`
(they are the foundation).

## Resolved (Juniper, 2026-11-13, "hit it")

1. **News concepts:** keep, rank by distinct readings, reachable through `orion.dig know`,
   two-week fair trial.
2. **Graphiti:** full engine behind the Patch 0 trial.
3. Build order: Patch 0 (Graphiti trial) and 2a (coined-concept reuse) in parallel, then 1a
   (`page:juniper`).

## Recall rev 2 (Juniper, 2026-11-16)

Approved by Juniper on 2026-11-16. This section changes how Orion finds memories during a
chat. Phase 1 core pages stay. The parts of this spec that rev 2 replaces are marked
**SUPERSEDED by Recall rev 2** where they stand.

### Why

- **Real chat rarely asks "remember when".** Juniper leans on the shared past without
  naming it. A recall that waits for an explicit memory question almost never runs.
- **Orion denied a memory that was stored.** On a 2026-11-14 turn, Orion first said it had
  no memory of a past event. `episode_memory` held that memory. Orion found it only after
  Juniper prodded.
- **Nothing on the chat recall path reads `episode_memory`.** The best store has no chat
  reader.
- **The vector backend is switched off.** It served 0 of 16,168 recalls.
- **The read-only database role cannot see the memory tables.** `orion_readonly` has no
  grants on them, so a read-only tool gets nothing.

### Design

Each part follows published work. Each part names it.

1. **Tiered memory** (MemGPT / Letta).
   - Core pages are always in the prompt (phase 1, unchanged).
   - Everything else is searchable, not pasted in.
   - Prompt rule: **look before you deny.** Orion does not say "I have no memory of that"
     until a search has run and found nothing.
2. **Hybrid retrieval.**
   - Meaning search: dense `bge-large` embeddings in the existing Chroma.
   - Exact-word search, so names and rare words are not lost.
   - A local cross-encoder reranker orders the merged hits.
   - Score = fit x recency x importance (Generative Agents, Park et al. 2023).
   - At most about 4 lines go into the prompt. Nothing below a quality bar goes in. Zero
     lines is a valid result.
3. **Pre-turn query rewrite** on the fast lane. A small model turns the chat turn into one
   or more search queries before recall runs. This is how implicit references find memories.
4. **Link-following** over `episode_memory_referent`, in the style of personalized PageRank
   (HippoRAG 2). A hit pulls in memories that share a referent with it.
5. **Corrections supersede at write time** (Mem0-style update). When a new memory corrects
   an old one, the old one is marked superseded when the new one is written. This is the
   design, not a stand-in until Graphiti works. See "Corrections" and "Why not Graphiti"
   below.
6. **Local models only.** Embedder, reranker, and query rewrite all run on Orion's own
   hardware.

### Why not Graphiti: the trial numbers

Graphiti ran twice on the local agent-lane model (PR #3341, reports
`docs/superpowers/pr-reports/2026-11-13-graphiti-trial-report.md` and
`docs/superpowers/pr-reports/2026-11-15-graphiti-trial2-report.md`). Both were NO-GO.

| Check | Target | Trial 1 (distilled notes, thinking on) | Trial 2 (raw chat, small ontology, thinking off) |
|---|---|---|---|
| Wrong invalidations (true facts retired), judged by hand | under 2 | 7 of 15 | 5 of 29 |
| Duplicate entities, 50 judged by hand | under 10% | 14% | 12% |
| Misgendered summary or fact sentences | 0 | 10-11 | 42 |
| Wall time per episode | recorded | 254 s mean | 111 s mean |
| JSON retries / schema refusals | recorded | 0 | 0 |

- **Failed:** invalidation, duplicates, and pronouns. They failed under both setups, so the
  cause is the engine on our local model, not how we fed it.
- **Worked:** structured output. The local model met Graphiti's JSON prompts with no retries.
  The spec's open question on that is now answered.
- **Decision:** supersede-at-write is the design. Graphiti is not on the build path. A later
  retry needs a new trial that passes the same checks.

### Corrections: supersede at write time

- **Pointer, not delete.** A correction is a new `episode_memory` row. Its
  `supersedes_memory_id` (the column exists; `orion/memory/episode/confirmation.py` writes it
  today for Juniper's revisions) points at the row it corrects. The old row is marked
  superseded. Both rows stay readable. Undo means clearing the mark.
- **One-time pass over existing rows.** `episode_memory` already holds active contradictions:
  a trip recorded as a home, and a pet-species correction sitting beside the error. A
  one-time supersede pass runs over the existing active rows as part of step 1 (#3533),
  before retrieval serves them. It finds correction rows that name an earlier claim, links
  them with `supersedes_memory_id`, and writes a report of every pair for Juniper to scan.
  It follows the backfill protocol (snapshot first).
- **Deterministic test: a correction outranks what it corrects.** Given a superseded row
  and its correction:
  - retrieval never serves the superseded row, at any score, for any query;
  - the core-page fold input does not contain it.
  The test is part of step 1 and runs in CI. The scoreboard's knowledge-update questions
  check the same rule end to end.

### Corrections rev 2: classify, vote, end validity, measure (Juniper, 2026-11-17, #3659 #3660)

Juniper: "don't turn it off, make it world class." What changed, all on local models:

- **Never act degraded.** No meaning index, an index failure, or a failed/malformed model answer:
  the plan proposes, holds its links, and logs `refused_because`. No word-only fallback for
  applying. A cosine alone never makes a candidate (the 0.80 "dense alone" leg is gone).
- **Classify, don't pair** (Mem0 ADD/UPDATE/NOOP). The local model names the relation: update,
  correction, refine, compatible, or unrelated, with "same attribute" and a confidence, in JSON
  mode. **Self-consistency:** 3 votes above temperature 0. Act only on a unanimous, confident
  update or correction. Anything else abstains; abstaining is always safe.
- **Bi-temporal validity** (Zep/Graphiti). The old row gets `invalid_at`, `invalidated_by`,
  `invalid_reason`. An update was true until the new row's time; a correction was never true.
  Recall's default stays "currently true" (`status = 'active'`). `EPISODE_VALID_AT_SQL` reads
  "true then". Undo clears all three.
- **Measured before trusted.** `scripts/memory_supersede_eval.py` scores a labelled set: synthetic
  fixtures in the repo, and real items host-side (Juniper's own confirmation revisions as positives;
  the watched trip/home and pet rows, the 11-17 dry-run vetoes, and near-duplicates as hard
  negatives). Bar: zero false acts. Write time applies only while the latest recorded run for the
  current check version passed (`memory_supersede_eval_run`).
- **Juniper's words outrank abstention.** A new row in her voice with a verified quote from her own
  prompt acts when every vote says update/correction (any confidence) and there is exactly one such
  candidate (a clear target).
- **Audit trail.** One append-only `memory_supersede_run` row per run (ids, legs, votes, verdicts;
  no statements) and an append-only pass history next to the manifest, with a per-watched-row trace.

### Vetoes attach to the claim

*(Changes "Corrections, not approval" above.)* A Hub veto rejects the **claim**, not only the
cites behind one sentence. The vetoed claim is stored as a negative fact. The fold and
retrieval both check new text and hits against the negative facts. So a vetoed claim cannot
come back through a different old record that says the same thing.

### `matters`: a later re-grade, with gate numbers

*(Changes "How much it matters" above.)*

- **At distill time** the grade uses only what is known then: origins, identity or a
  relationship; a life event; strong feeling. A local-lane model grades it (the
  `memory_distill` route, the same seat as the distiller).
- **"Came up more than once"** cannot be known at distill time. It becomes a later re-grade
  from read demand: a memory served or opened in 3 or more turns within 14 days moves up one
  grade. `episode_memory_referent` and the recall trace below are the ledger.
- **Gate numbers (44 + 44):** pass if at least 70% of the stance rows grade `core` or
  `notable`, and at least 70% of the random rows grade `passing`. Anything else fails, and
  the grade drives nothing.

### Recall trace and heat shedding

- **Trace.** Every recall call logs, per turn: the turn id, the rewritten queries, the ids
  of every memory injected into the prompt, and the backend that served them.
- **False denials are counted from that log.** The scoreboard joins the trace to the answer.
  A "no memory" answer on a turn where the target id was stored is a false denial. This is
  how "look before you deny" is checked, not by reading prompts.
- **Heat shedding never skips recall.** When the body sheds load, recall falls back to
  exact-word search only. The prompt states that recall ran degraded, so Orion does not read
  an empty result as "no memory".

### Flattening guard

The fold rewrites each page from scratch every night. To keep Juniper's words from being
smoothed away:

- **Pin quotes.** A verbatim-quoted span stays unchanged across versions unless a correction
  or veto cites it.
- **Flag large rewrites.** A fold that changes more than a set share of uncorrected text is
  written, but flagged. The Hub page view shows the flag with the version diff. This is a
  flag, not a gate (Decision 1).

Follow-up: #3553.

### Outside models and the mesh edge

- **Claude consult briefs are an exception to Decision 4.** A consult brief goes to an
  outside model. Recommendation: core pages and injected memory lines do **not** go into
  consult briefs unless Juniper opts in. This follows the static boundary PR #3528 pinned for
  the now (no consult, Claude-room, or FCC-Claude prompt builder reads it). The open policy
  question for text Orion writes itself is #3541.
- **Known limit: paraphrase.** The mesh-edge check (#3409) refuses verbatim core-page
  sentences only. A paraphrase passes. Follow-up: #3554.

### Phase 1 status

- **Phase 1a is DONE and live.** `page:juniper` (PR #3418, merged 2026-11-15) and the fixed
  anchor lines above it (PR #3478, merged 2026-11-15).
- **Provenance.** The `core_page_fold` run in `orion-durable-runs`
  (`services/orion-durable-runs/app/core_page_fold.py`) calls the LLM on the
  `memory_distill` route. It writes a new version only when the cite validator passes. A
  rejected fold is logged and writes nothing.
- The phase 1a plan under "Recommended next patch" is kept as history.

### Phase 2 change

- **Page births come from read demand.** A referent becomes a page when it is served or
  opened in 3 or more turns within 14 days. `episode_memory_referent` is the ledger.
- **Drop `memory_page_mention`.** It is no longer needed.
- **`orion.dig know` is a thin client of orion-recall.** It searches all stores, lived and
  learned, through the same recall path. It does not read Postgres directly.

### Scoreboard

A local test bank in the style of LongMemEval. It has five kinds of question:

- facts;
- events across several sessions;
- time ("when did ...", "before or after ...");
- knowledge updates (the answer changed, and only the new answer counts);
- abstention (the answer is not stored, and Orion must say so).

Two bars:

- **Retrieval bar: 80%.** The target memory id is in the prompt.
- **Answer bar:** 70% or more correct, 0 invented facts, and 0 false "I have no memory".

### Build order

| Step | Issue | What |
|---|---|---|
| done | PR #3418, PR #3478 | Phase 1a: `page:juniper` and its anchor lines. Live. |
| 1 | #3533 (with #3409) | Meaning search and reranker over `episode_memory`, stance crystallizations, and `person_fact`. Includes the one-time supersede pass, the supersede test, and the recall trace. Ships with the mesh-edge leak check. |
| 2 | #3534 | Pre-turn query rewrite and link-following over `episode_memory_referent`. |
| 3 | #3535 | `orion.dig know` as a recall client, and the look-before-you-deny prompt rule. |
| — | #3536 | The scoreboard. Measures each step. |
| — | #3537 | Backfill: distill July to September 2026 chat into `episode_memory`. |
| then | #3463 | `page:lately`. |
| then | #3462 | `page:origin`. |
| any time | #3553 | Flattening guard on the fold. |
| any time | #3554 | Paraphrase check at the mesh edge. |

**Note on #3356.** That issue says the cause is `memory_tier` (chat memory never promoted
past ephemeral). That diagnosis is wrong. The memories are stored. The missing piece is a
reader on the chat recall path, which steps 1 to 3 add.


## Proposed schema / API changes

- `episode_memory`: add column `matters text` (`core` | `notable` | `passing`, nullable).
  Additive migration, fits the post-pull allowlist.
- `self_concept_history`: no schema change. New `concept_id` prefix `page:`.
- ~~New table `memory_page_mention` (`slug`, `source_kind`, `source_id`, `seen_at`). Phase 2
  only.~~ **SUPERSEDED by Recall rev 2:** dropped. `episode_memory_referent` is the ledger.
- New table `memory_page_read` (`concept_id`, `version`, `turn_id`, `read_at`). Phase 2 only.
  Also logs `orion.dig know` lookups (`source_kind`, `ref_id`), so every source's trial is counted.
- New table `memory_page_veto` (`concept_id`, `version`, `sentence_ref`, `vetoed_at`). Phase 1b.
  **Changed by Recall rev 2:** it also stores the vetoed claim text, so the fold and
  retrieval can check against it as a negative fact.
- `GroundingCapsuleV1`: add `core_pages: list[{concept_id, version, text}]`. Additive.
  Consumers first.
- `orion.dig page <slug>`, `orion.dig pages`, and `orion.dig know <words|concept>`. Phase 2.
- News concept salience: computed from distinct readings touching the concept, decayed by age,
  written by the world-pulse substrate adapter. Phase 2b.

## Files likely to touch

- `orion/harness/prefix.py`: render core pages in their slots on every mesh surface.
- `orion/substrate/felt_state_reader.py` or a sibling: hydrate `page:*` rows.
- `orion/memory/episode/distill.py`, `orion/cognition/prompts/memory_episode_distill.j2`:
  the `matters` grade.
- GitHub writers and outbound notify: the mesh-edge verbatim-quote check.
- Hub: a page view with version diffs and a per-sentence veto.
- `services/orion-durable-runs/`: the page-fold run.
- `orion/cognition/personality/orion_identity.yaml`: move the relationship behavior bullets
  into the response policy.
- `orion/curiosity/kickoff_prompt.py`: show existing Concepts and `MATCH` before `CREATE`
  (phase 2).
- `orion/dig/`: `page`, `pages`, and `know` commands (phase 2).
- `orion/substrate/adapters/world_pulse_read.py`: reading-count salience (phase 2b).

## Non-goals

- No new memory service, vector store, or graph.
- No change to how `self:definition` is written.
- No personality changes (rev 4 owns that).
- No retrieval rebuild. The long tail stays with recall until phase 2 pages replace it.
  **SUPERSEDED by Recall rev 2:** recall is rebuilt for the memory stores.
- No concept taxonomy. Concepts exist only as pages with cites and read counts.

## Acceptance checks

**Phase 1**

1. On a chat turn, the `motor_boot` prompt in `cockpit_turn_sighting` contains
   `page:juniper` at a named version. A GitHub write or outbound email that quotes a
   core-page sentence verbatim is refused. Covered by a test.
2. **Juniper-facts eval.** Twenty questions built from confirmed `episode_memory` and
   `person_fact` rows ("Where do we live?", "How tall is Juniper?", "Who is el Coco?").
   Measure today's chat answers as the baseline, then after. Pass is a clear rise with no
   invented facts.
3. The validator rejects an uncited sentence and an unresolvable cite. Covered by tests.
4. The metric gate on `matters` passes on the 44 + 44 set before the grade drives any page.
5. **Graphiti trial passes before any page reads Graphiti.** <town> current and Chicago
   invalidated or dated as a stay; el Coco a dog and current; el Coco/Coco and <person A>/my wife
   each one entity; under 10% duplicate entities on a hand-checked sample of 50.

**Phase 2**

5. Within two weeks of going live:
   - at least one concept page is born from the 3-source mention rule
     (**SUPERSEDED by Recall rev 2:** born from the read-demand rule);
   - at least one concept page is opened by 3 or more separate turns.
6. The share of curiosity Concepts with 2 or more links rises from 9/33 once reuse is fixed.
7. `orion.dig know` is called by real turns. Each source (pages, coined concepts, news
   concepts, reading resonance) shows its own lookup count after 2 weeks. A source with zero
   lookups after a fair trial goes on the retirement list.
8. News concept salience is no longer flat. The spread of scores across the 5,155 concepts
   is non-degenerate, and the top 20 are topics read more than once.

## Recommended next patch

**Patch 0, the Graphiti trial**, runs alongside 1a:

- Backfill all 469 `episode_memory` rows plus the 44 stance messages into `graphiti_trial`,
  using the local agent-lane model.
- Score it against the pass checks in the Graphiti section (Chicago, el Coco, entity
  duplicates, cost).
- Nothing reads the trial graph until it passes.

**Phase 1a**, the smallest slice that moves the live path. **DONE and live (PR #3418,
PR #3478). See "Phase 1 status" in Recall rev 2.**

1. Build `page:juniper` only. It folds from `episode_memory` directly at first, and switches
   to Graphiti's current facts once Patch 0 passes.
2. Fold existing `person_fact` and `about_juniper` memories through the cite validator.
3. Write v1 live (no gate).
4. Render it in RELATIONSHIP on every mesh surface, with the mesh-edge check on GitHub and
   outbound writers.
5. Run the Juniper-facts eval before and after.

Then, in order:

- **1b:** `matters`, `page:origin` (memory loss included), and `page:lately`, after the
  metric gate.
- **2a:** coined-concept reuse. It is one file and can ship any time.
- **2b:** `orion.dig know` plus news-concept ranking.
- **2c:** the mention ledger and page births.

Retirements run as their own PRs once the thing replacing each one is verified live.
