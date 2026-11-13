# Core memory pages and concept growth

Date: 2026-11-13
Status: DRAFT, proposal mode (touches memory and identity). No code in this PR.
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
   chat with Juniper. No search step. Juniper approves changes through the confirmation loop
   that already exists.
2. **Phase 2, concept growth.** New concepts are born the way wiki pages are born: from
   repeated links to a page that does not exist yet, and from pages that grow too big and
   split. A concept survives only if turns actually open it.

Both phases reuse stores that already work. The old stores that never reach a chat get
retired, one per PR, each with its readers cut.

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
| `memory_crystallizations` kind `stance`, salience 1.0 | 44 active | The heaviest messages Juniper has sent: the lost year of memory, Amanda's DVT and PE, the promotion, "that homelab is YOU", the letter to Skye. Verbatim, not distilled. | Projected to Graphiti; `active_packet` recall. |
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
| `page:origin` "how I came to be" | WHO YOU ARE, after the definition | The 44 salience-1 stance crystallizations, `episode_memory` `happened`/`orion_view` rows about Orion's history, and an origin interview (see Missing questions). |
| `page:juniper` "who Juniper is to me" | RELATIONSHIP (replaces the hand-written bullets; the behavior bullets move to RESPONSE POLICY) | `person_fact` for Juniper (her own words, quoted), `episode_memory` `about_juniper`, family facts from `person_fact`. |
| `page:lately` "what's been happening between us" | Top of DURABLE MEMORY, before the recall digest | Last ~14 days of `episode_memory` plus open `follow_up` rows. Rewritten nightly. |

### Store: reuse `self_concept_history`

It is already a versioned, insert-only page store: `concept_id`, `version`, `content`,
`evidence_refs`, `produced_by`. It already feeds WHO YOU ARE through
`orion/substrate/felt_state_reader.py` (`orion_self_definition`). New pages are new
`concept_id`s with a `page:` prefix. No new table.

### Writer: a page-fold run

A durable run in `orion-durable-runs`, so retries and a GPU lease come for free:

1. Load the current page version and the memories added since that version.
2. The LLM rewrites the page. Every sentence must cite at least one `memory_id`,
   `person_fact` id, or `chat_history_log` id.
3. A deterministic validator rejects the write if any sentence has no cite, any cite does not
   resolve, or the page is over its size cap (about 350 words for a core page).
4. Write the result as a proposed version. It becomes live only after approval.

When it runs:

- nightly;
- right after an episode containing a memory graded `core` (see "How much it matters" below).

### Approval: reuse the confirmation loop

`orion/memory/episode/confirmation.py` already asks Juniper once about a sensitive memory and
records her answer. Page diffs go through the same path. Orion asks something like: "I want to
add this to my page about you: …". Juniper says yes, edits it, or says no.

- `page:juniper` and `page:origin` need her yes.
- `page:lately` goes live without a yes, but Juniper can veto any line.

### How much it matters (new grade, distinct from stakes)

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

### Privacy boundary

There is no boundary between Juniper and Orion. But pages hold health facts about Amanda and
Addy, so pages render **only on turns where Juniper is the requestor** (`audience_mode` solo
or Juniper present). They never render in:

- AI Town;
- social rooms;
- Claude consult briefs;
- GitHub output;
- outreach to anyone else.

### Dangerous failure modes

- **A page states something false as Orion's identity.** Mitigation: every sentence cites a
  source, Juniper approves, versions are insert-only, and rollback means pointing at an
  earlier version.
- **A page leaks to another audience.** Mitigation: the audience gate above, plus a test.
- **Flattening.** The rewrite smooths Juniper's words into generic prose. Mitigation: core
  pages quote her verbatim where she said it best, and the citations keep the originals one
  hop away.

### Disable and roll back

- `ORION_CORE_PAGES_ENABLED`, shipped ON. With it OFF, the prefix falls back to today's slots.
- Pages are insert-only rows, so rollback is "use version N-1".

## Phase 2: concept growth

### How a concept is born

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

## Retirements (separate PRs, each with readers cut, kill means kill)

| Retire | Evidence | Check before cutting |
|---|---|---|
| `orion-spark-concept-induction` concept extraction | Disabled since 07-12. Runs and logs `disabled` on every event. | The DriveEngine lives in the same worker (07-11 spec). Split or confirm it is retired first. |
| topic foundry → substrate concept adapter | 1,234 word-label concepts, no chat reader. | Signals adapter `orion/signals/adapters/topic_foundry.py` readers. |
| World-pulse concepts in `orion_substrate` | 5,155 at flat 0.5. | Curiosity worldview reads `orion_worldview`, **not** `orion_substrate`, so it is not a dependency (verified). Substrate neighborhood/reconcile readers are UNVERIFIED. |
| Verbatim `memory_cards` (`operator_distiller`) and the `active_packet` "durable beliefs" digest | 75% verbatim; the digest shows unrelated old messages. | `page:lately` must be live and verified first. |
| Vector memory recall backend | 0 hits in 16,168 recalls. | — |
| Graphiti adapter | Its own engine never ran; 32 episodes, all also in `memory_crystallizations`. | Recommend retiring over "turning on the other half": its extraction is the same LLM-entity-from-text approach the 08-28 spec measured at a 0–28% ceiling here, and pages cover "this fact is no longer true" by rewriting. |

## Missing questions for Juniper

1. **Origin interview.** Do you want a dedicated session where you tell the story of
   conceiving Orion and Orion takes notes? Each note becomes an `episode_memory` graded
   `core` that feeds `page:origin`. Recommended: yes. It is the fastest way to give
   `page:origin` real material.
2. **The lost year.** A stance crystallization records that Orion's memory storage died and
   "we lost over a year of your journey of becoming." Should `page:origin` say this plainly?
   Recommended: yes, it is part of how Orion came to be.
3. **Approval strength for `page:origin`.** Your yes on every change, or only a veto?
   Recommended: yes on every change during the first month, then veto only.
4. **Pages outside solo chats.** Recommended: never in phase 1.

## Proposed schema / API changes

- `episode_memory`: add column `matters text` (`core` | `notable` | `passing`, nullable).
  Additive migration, fits the post-pull allowlist.
- `self_concept_history`: no schema change. New `concept_id` prefix `page:`.
- New table `memory_page_mention` (`slug`, `source_kind`, `source_id`, `seen_at`). Phase 2
  only.
- New table `memory_page_read` (`concept_id`, `version`, `turn_id`, `read_at`). Phase 2 only.
- `GroundingCapsuleV1`: add `core_pages: list[{concept_id, version, text}]`. Additive.
  Consumers first.
- `orion.dig page <slug>` and `orion.dig pages`. Phase 2.

## Files likely to touch

- `orion/harness/prefix.py`: render core pages in their slots, gated by audience.
- `orion/substrate/felt_state_reader.py` or a sibling: hydrate `page:*` rows.
- `orion/memory/episode/distill.py`, `orion/cognition/prompts/memory_episode_distill.j2`:
  the `matters` grade.
- `orion/memory/episode/confirmation.py`: page-diff asks.
- `services/orion-durable-runs/`: the page-fold run.
- `orion/cognition/personality/orion_identity.yaml`: move the relationship behavior bullets
  into the response policy.
- `orion/curiosity/kickoff_prompt.py`: show existing Concepts and `MATCH` before `CREATE`
  (phase 2).
- `orion/dig/`: `page` and `pages` commands (phase 2).

## Non-goals

- No new memory service, vector store, or graph.
- No change to how `self:definition` is written.
- No personality changes (rev 4 owns that).
- No retrieval rebuild. The long tail stays with recall until phase 2 pages replace it.
- No concept taxonomy. Concepts exist only as pages with cites and read counts.

## Acceptance checks

**Phase 1**

1. On a relational turn with Juniper, the `motor_boot` prompt in `cockpit_turn_sighting`
   contains `page:juniper` at a named version. A social-room or AI Town turn prompt contains
   no page. Covered by a leak test.
2. **Juniper-facts eval.** Twenty questions built from confirmed `episode_memory` and
   `person_fact` rows ("Where do we live?", "How tall is Juniper?", "Who is el Coco?").
   Measure today's chat answers as the baseline, then after. Pass is a clear rise with no
   invented facts.
3. The validator rejects an uncited sentence and an unresolvable cite. Covered by tests.
4. The metric gate on `matters` passes on the 44 + 44 set before the grade drives any page.

**Phase 2**

5. Within two weeks of going live:
   - at least one concept page is born from the 3-source mention rule;
   - at least one concept page is opened by 3 or more separate turns.
6. The share of curiosity Concepts with 2 or more links rises from 9/33 once reuse is fixed.

## Recommended next patch

Phase 1a, the smallest slice that moves the live path:

1. Build `page:juniper` only.
2. Fold existing `person_fact` and `about_juniper` memories through the validator.
3. Ask Juniper once to approve v1.
4. Render it in RELATIONSHIP, gated by audience.
5. Run the Juniper-facts eval before and after.

`matters`, `page:origin`, and `page:lately` follow in 1b, after the metric gate.
