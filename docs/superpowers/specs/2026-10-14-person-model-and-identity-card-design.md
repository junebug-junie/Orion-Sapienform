# Person model, entity reducer, and evolving identity card

**Date:** 2026-10-14
**Mode:** design / proposal (AGENTS.md section 0A, proposal mode: memory, identity, social continuity). Nothing here is implemented.
**Asked by:** Juniper, 2026-10-13 (eight requirements, quoted in section "Requirements" below).
**Builds on, does not replace:**
- Referent graph, `docs/superpowers/specs/2026-10-06-memory-stage2-referent-graph-design.md` (approved): people are evidence-linked referent nodes. This design adds *attributes* to those nodes.
- Situation Graph, `docs/superpowers/pr-reports/2026-10-07-situation-graph-shadow-pr.md`: the running "what is true now" object. Person cards ride it.
- Memory confirmation asks, `orion/memory/episode/confirmation.py` + `orion_ask` + Hub "Orion is asking".
- Self-model rebuild arc (self-inquiry curiosity line, `:SelfDefinition`, sense-of-self PR #2156). The Orion self card is a view over that work, not a second self-model.

## Arsonist summary

Orion has no picture of Juniper. What reaches every chat turn about her is twelve hand-written bullets in `orion/cognition/personality/orion_identity.yaml:27-41` ("co-architect", "steward", "expects continuity"). Everything Orion actually knows about her life (her wife Amanda, Teddy and Addy, her sister, her boss Rachel, her team lead promotion, that she lives in Ogden, that she is 6'5") is scattered across free-text memories, curiosity "lived" answers, and 70 days of chat that was never distilled. Nothing turns those into a structured, checkable picture. The graph has eleven separate nodes for her, and her own family are not nodes at all.

The fix is three thin pieces sharing one store, not a new cognition service:

1. **A person-attribute store** (Postgres): every fact about a person is a row with her quote, a confidence, a confirmation state, and a last-updated time. Stakes and sensitivity come from the attribute's category by table lookup, not from an LLM's judgment (the LLM's stakes judgment is already failing live, see below).
2. **A deterministic entity reducer**: a Fellegi-Sunter style scorer over names, run as a replayable job with a merge log and unmerge. It never auto-merges two people; uncertain merges become Hub asks.
3. **A person-study run** (durable workflow on the GPU pool): reads Orion's own stores for one person at a time, proposes attributes with quotes, a validator rejects anything not in her own words, and uncertain or sensitive proposals become Hub asks.

A pure, deterministic **card reducer** turns confirmed attributes into versioned cards (Juniper, Orion self, relationship) that replace the YAML block in the chat turn and ride the situation state into chat, outreach, and curiosity.

The existing concept-induction service is not the organ for this and should not be stretched into it (section "Concept induction, honestly").

## Requirements (Juniper, 2026-10-13, paraphrased only where marked)

1. Person model of Juniper across preferences, goals, motivations, fears, strengths, opportunities, family, friends, coworkers, profession, personal interests, spiritual interests, gender, sexuality, past history, "anything else" (health, routines, values, communication style, home, schedule/timezone, relationship to Orion). Every attribute evidence-backed, with confidence, provenance, confirmation status, last-updated; visible, correctable, removable in Hub. Spirituality, gender, sexuality, health only from what she states, never inferred.
2. "Orion can ask me about these things as they come up in the 'orion is asking' section on top of hub."
3. "Extensible to other people too ... for n number of people once they become important to Orion."
4. Cards ride "that object that goes into chat that starts tracking situational things and is supposed to roll up memories" (the Situation Graph state), plus outreach and curiosity.
5. "We need a reducer to solve that one-time cleanup so in the future entities can get merged." Deterministic, non-LLM.
6. "Seems like we are missing a concept induction organ here overall."
7. "It should be a type of curiosity run that churns through the repo to learn about me."
8. "An evolving identity card instead of that crappy one we have stubbed in that explains to Orion who I am and who they are."

## Current architecture (checked live 2026-10-13, read-only)

### What Orion holds about Juniper

| Store | What is there | Evidence |
|---|---|---|
| `episode_memory` | 68 rows, all since 2026-10-04. 15 `purpose=about_juniper`: 4 `high/confirmed`, 1 `high/pending_confirmation`, 10 `low/auto`. | `select purpose, stakes, confirmation_state, count(*) ...` |
| `episode_memory_evidence` | 73 quotes, all `verified=true`, sources `chat_prompt`/`chat_response`. | same DB |
| `chat_history_log` | 626 turns, 2026-07-24 to 2026-10-13. Everything before 10-04 was never distilled into memory. | `min/max(created_at)` |
| Names in chat never distilled | "amanda" in 9 prompts (08-15 to 10-05), "teddy" 4, "addy" 3, "sister" 2, "rachel"/"armando"/"vincent" 1 each (10-05). Only 4 episode memories mention any of them. | `prompt ilike` counts |
| `orion_worldview` `:LivedAnswer` (121 rows) | `lived.her_team_unnamed`, `lived.who_matters`, `lived.known_only_through_juniper` hold the family and team in prose: Amanda (wife, DVT/PE in late August), Teddy (son, turned seven), Addy (daughter, broke her arm), sister, in-laws, boss Rachel, reports "Jay, Jay, Armando, Vincent". | Cypher on `orion_worldview` |
| `docs/.../2026-09-03-biographical-curiosity-miss-design.md` | Her promotion to Team Lead while staying AI/ML Architect / FDE. | spec text |

### Stakes policy is not holding

The distiller prompt (`orion/cognition/prompts/memory_episode_distill.j2:29-40`) says family relationships and feelings are **high** stakes. The validator (`orion/memory/episode/validate.py:28`) deliberately never reads the statement to judge stakes. Live, the LLM labeled these `low/auto`:

- "Juniper said she tends to not reach out to her sister because ..." (family_relationships)
- "Juniper described the relationship with her sister as asymmetric lately ..." (family_relationships)
- "Juniper said she was scared shitless that she busted my inference ..." (juniper_feelings)
- "Juniper feels anxious about drilling a hole in her new home ..." (juniper_feelings)

So "stakes by LLM judgment" is a measured failure. This design derives stakes from the attribute category instead.

### The graph splits her

`orion_substrate`, nodes whose label contains "juniper" (11):

| node_id prefix | label | kind | What it really is |
|---|---|---|---|
| `referent-b96fc90e...` | juniper | entity, `entity_type=person` | **The** person node. 14 `observed_in` edges. |
| `sub-concept-seed-juniper` | Juniper | concept (seed loader) | Same person, different producer |
| `sub-entity-topicfoundry-...-ca5c...` | juniper | entity (topic foundry) | Same person |
| `sub-entity-topicfoundry-...-e500...` | juniper feld | entity | Same person, full name |
| `sub-entity-topicfoundry-...-6d08...` | juniper's taste | entity | A facet *about* her, not her |
| `sub-entity-topicfoundry-...-1026...` | juniper affect | entity | Facet about her |
| `sub-concept-seed-orion_juniper_relationship` | Orion-Juniper relationship | concept | The relationship, not her |
| `sub-entity-topicfoundry-...-edfa...` | juniper relationship summary | entity | **Junk**: the prompt context key `juniper_relationship_summary` leaked into topic foundry |
| `sub-entity-topicfoundry-...-eb42...` | juniper_relationship_summary | entity | Same junk, underscore form |
| `sub-concept-topicfoundry-...-11` | Chat with Juniper | concept | A topic, not her |
| `sub-concept-topicfoundry-...-9` | mind / hey / juniper | concept | A topic-cluster label |

So "merge the 11 nodes" is wrong as stated: 4 are the same person, 3 are facets/relationship (link, don't merge), 2 are leaked field names (flag as junk), 2 are topics. The reducer needs three outcomes, not one.

Person referents: only 2 exist (`juniper`, `orion`) with 4 alias rows in `referent_alias` (`person:juniper`, `juniper`, `person:orion`, `orion`). Amanda, Teddy, Addy, Rachel, the sister: zero nodes.

An open identity question already sits unanswered: `memory_tension_shadow` has 9 open `scope=juniper, answer_via=conversation` questions, including "Is the 'juniper' from our conversations the same as the 'juniper' already in my graph?". The referent spec deferred the producer that would ask them ("Stage 3 'Orion is asking' producer", its Non-goals). Nothing has picked it up: `orion_ask` holds 9 rows, all `source_kind=memory_confirmation`. `RESOLVABLE_KINDS` lists `open_question` (`confirmation.py:47`), but no code produces it.

A conflict the store cannot see today: `referent_alias` maps "home" to `place:slc` (grounded in `chat_prompt:fb42f555...`), while a confirmed memory says "Juniper corrected me that she lives in Ogden, Utah, not Chicago." Which is right is UNVERIFIED (Ogden is in the SLC area; "home" may be a descriptor of a trip). A person model with a `home` attribute would surface this as a contradiction to ask about.

### What chat reads today

- Unified turn (`stance_react.j2:16-20`, the only chat path that matters) renders `orion_identity_summary` + `juniper_relationship_summary` under "WHO YOU ARE".
- `juniper_relationship_summary` is built by `orion/cognition/personality/identity_context.py:28-62` from the YAML `juniper_relationship` block, injected by `services/orion-cortex-exec/app/executor.py:1104` (`_inject_identity_context`), logged as `identity_kernel_source=configured_yaml`.
- Orion's own definition already reaches chat: `apply_self_definition_to_ctx` and `apply_lived_self_to_ctx` (both called from `_inject_identity_context`) read the self-inquiry line's latest `:SelfDefinition` (66 in `orion_worldview`).
- Misgendering: `stance_react.j2:131` says "Juniper telling Oríon something about **their own** life". Juniper is she/her.
- Pronouns are not stored anywhere (`grep pronoun orion/schemas` is empty). The distiller hardcodes "she" (`memory_episode_distill.j2:7,11`), which is right for Juniper and wrong for any other person it will meet.

### The Situation state

- Key `orion:situation:latest`, model `SituationStateV1` (`orion/schemas/situation_state.py`, `extra="forbid"`), one writer: the `situation.update` durable graph (`services/orion-durable-runs/app/situation_graph.py`).
- Live value: `thread_id=situation:juniper:2026-10-12`, `revision=24`, `juniper.{whereabouts:null, doing:[], waiting_on:[], recent:[3 facts]}`, `recall.primed` from episode memories.
- Readers: outreach only (`services/orion-hub/scripts/endogenous_outreach.py:1113`, parses with `SituationStateV1.model_validate_json`). Chat reading it is "step 3", not landed.
- Because the model forbids extra fields and outreach validates strictly, adding a `people` field producer-first would make outreach's read fail with a validation error and silently drop its situation enrichment. The schema change must reach the Hub before the producer writes the field.

### The asks surface

- Table `orion_ask` (`orion/schemas/ask.py`, `OrionAskV1`): `source_kind` + `source_ref` route the answer back.
- Hub: `services/orion-hub/templates/index.html:262` ("Orion is asking" at the top of Hub home), routes `services/orion-hub/scripts/ask_routes.py` (`/api/asks`, `/answer`, `/dismiss`, `/resolve` for memory confirmations).
- Caps (`confirmation.py:49-52`): 5 open across `RESOLVABLE_KINDS`, 3 new memory-confirmation cards per local day.
- Answers publish `OrionAskAnsweredV1` on `orion:ask:answered`; `services/orion-substrate-runtime/app/ask_answered_listener.py` consumes `vision_individual` answers. That is the pattern to copy for new kinds.

### Concept induction, honestly

Live (`orion-athena-spark-concept-induction`, up 2 days):

- It receives events (`orion:feedback:frame`, `orion:metacognition:tick`) every couple of seconds.
- Every induction trigger in the last 48 hours logged `decision=disabled` (18,731 lines). Container env has `CONCEPT_AUTONOMOUS_TRIGGER_ENABLED=false`.
- `CONCEPT_SUBJECTS=orion,juniper,relationship`, yet the local store `/data/concept-induction-state.json` holds only `drive_states, goal_cooldowns, goal_slots, episode_runs_processed`. No `juniper` concept profile in the store, and no node with `subject='juniper'` in `orion_substrate` or `orion_kg`. Other backends were not checked: UNVERIFIED that no profile exists anywhere.
- Its logs are dominated by `goal_state updated ... priority=0.911 status=proposed` every ~2 s, even though `orion/spark/concept_induction/CLAUDE.md` says the goal machinery was deleted 2026-07-30. Origin UNVERIFIED; out of scope here, flagged as a finding.
- What its code does when it runs: spaCy NER + noun chunks (`extractor.py`), embedding clustering (`clusterer.py`), and a summarizer, per subject window. The output is a bag of clustered phrases ("concepts") per subject.

Why it can't do this job, even switched on: it has no claim structure (who, which attribute, what value), keeps no quotes, has no confirmation state, does no identity resolution, and the approved referent spec bans vectors for identity. Its "juniper" subject is a slot that has produced nothing we can find. The "missing concept induction organ" Juniper names is real, but it is three specific missing functions: **propose** (person-study), **resolve identity** (entity reducer), **consolidate** (card reducer). This design builds those. Concept induction's `juniper`/`relationship` subjects should be retired once cards exist (kill means kill), after checking their consumers.

### Infra available

- Durable workflows in `orion-durable-runs` (`episode_distill_graph.py`, `orion_day_graph.py`, `situation_graph.py` ...) with GPU-pool leases (`pool_hold.py`). Person-study is one more workflow, not a shadow queue.
- Quote verification already exists: `orion/memory/episode/validate.py` (`quote_in_text`, `quote_long_enough`, `fold_text`).
- Record-linkage libraries: none installed on the host and none in any `requirements*.txt` (checked `rapidfuzz`, `jellyfish`, `splink`, `datasketch`, `recordlinkage`, `dedupe`).

## Missing questions

These are answered by investigation where possible; the ones only Juniper can answer are in "Open questions for Juniper" at the end.

- Is "home" Ogden or SLC? Live data contradicts itself (above). The first person-study run should surface this as an ask rather than us guessing.
- Is "Jay, Jay" one person or two? Her own text is ambiguous (`lived.her_team_unnamed`). A reducer must never merge on name alone for people; this is the test case.
- Does any reader depend on concept induction's `juniper` subject? UNVERIFIED; check before retiring.

## Proposed schema / API changes

### 1. Attribute category registry (the anti-cathedral gate)

`orion/memory/person/categories.yaml` lists each category with `stakes`, `sensitive`, `card_section`, `ask_template`, `cardinality` (one value or many). It is a lookup table, not an ontology. A CI test (`test_every_category_has_producer_consumer_eval`) fails if any category lacks: a person-study prompt example (producer), a card-renderer section (consumer), and at least two labeled rows in the eval fixture. A category with no evidence renders nothing in the card: no empty headings.

| Category | Stakes | Sensitive (stated only, never inferred) |
|---|---|---|
| preferences, personal_interests, routines, communication_style, schedule_timezone | low | no |
| goals, motivations, strengths, opportunities, values, profession, past_history, home | high | no |
| fears, family, friends, coworkers, relationship_to_orion | high | no |
| health, spiritual_interests, gender, sexuality | high | **yes** |
| pronouns | high | no (but stated only: never guessed from a name) |

Stakes is read from this table, never from the LLM. High stakes means `pending_confirmation` until Juniper confirms; low stakes is `auto` and usable, shown as "unconfirmed" in Hub.

### 2. Tables (new manual migration, `services/orion-sql-db/manual_migration_person_model_v1.sql`, with rollback file and `-- ORION-MIGRATION-REQUIRED-BY:` header)

```sql
person_attribute (
  attribute_id uuid primary key,      -- uuid5(node_id|category|normalized value): replay-stable
  node_id text not null,              -- referent node (referent_alias key row)
  category text not null,             -- must exist in categories.yaml (CHECK via app, test)
  value_text text not null,           -- first person, Orion's voice: "Juniper told me she ..."
  value_norm text not null,
  stakes text not null,               -- from category, never LLM
  sensitive boolean not null,
  voice text not null,                -- juniper_said | ask_answer | orion_inferred (forbidden when sensitive)
  confidence real not null,           -- see below
  confirmation_state text not null,   -- auto | pending_confirmation | confirmed | rejected | retracted
  status text not null default 'active', -- active | superseded | removed
  superseded_by uuid,
  first_seen_at timestamptz not null,
  last_evidence_at timestamptz not null,
  updated_at timestamptz not null,
  produced_by text not null           -- person_study.v1 | episode_distill.v6 | hub_edit | ask_answer
)
person_attribute_evidence (attribute_id, source_kind, source_id, quote, quote_sha256, verified, speaker)
  -- source_kind: chat_prompt | episode_memory | lived_answer | dream | orion_day_letter | ask_answer
  -- speaker: juniper | orion. Only speaker=juniper evidence can support a juniper_said attribute.
person_attribute_event (event_id, attribute_id, op, actor, before_json, after_json, reason, at)
  -- append-only: proposed | confirmed | corrected | rejected | removed | superseded | merged_owner
entity_merge_log (merge_id, survivor_node_id, absorbed_node_id, relation, score, features_json,
                  reducer_version, decision, decided_by, applied_at, reverted_at)
  -- relation: same_as | facet_of | junk ; decision: auto | asked | confirmed | rejected
identity_card (card_id, subject_node_id, card_kind, version, body_json, compact_text,
               input_fingerprint, input_attribute_ids, produced_by, created_at)
  -- card_kind: person | self | relationship ; new row only when input_fingerprint changes
person_study_cursor (node_id, source_kind, last_source_id, last_run_id, updated_at)
```

Confidence is computed, not asked of the LLM: `confirmed` = 1.0; otherwise `min(0.9, 0.5 + 0.1 * distinct verified quotes)`, halved if a newer contradicting attribute exists. It is a knob, calibrated on the eval, not a finding.

Removal in Hub sets `status=removed` and writes an event. The person-study validator refuses to re-propose a removed or rejected value (same "do-not-remint" rule as `confirmation.py`'s `_SKIP_REJECTED_SQL`).

### 3. Pronouns

A `pronouns` attribute on every person node. The card renderer and all templates that name a person read it; no pronoun is rendered for a person until one is stated (render the name instead). `memory_episode_distill.j2` stops hardcoding "she" for non-Juniper people by being given each known person's pronouns in its input. Juniper's own pronoun attribute is seeded by the first person-model ask ("Which pronouns do you want me to use for you?"), so it carries her quote like every other attribute.

### 4. Entity reducer (`orion/memory/person/entity_reducer.py`, pure; job CLI `scripts/run_entity_reducer.py`)

**Framework survey and pick**

| Option | Verdict |
|---|---|
| Splink (Fellegi-Sunter with EM training, DuckDB backend) | Rejected for now. Right theory, wrong size: brings DuckDB + pandas, and EM estimation of match weights is unreliable on ~1,100 nodes with few true matches. |
| Fellegi-Sunter scoring by hand (fixed m/u weights, two thresholds) | **Chosen.** ~150 lines, deterministic, every decision explainable as a sum of named log-weights stored in `features_json`. Weights live in a versioned config, calibrated against Juniper's labeled pairs. |
| RapidFuzz (Jaro-Winkler, token-set ratio; C-backed, MIT, no transitive deps) | **Chosen**, one new dependency in the one service that runs the job (orion-memory-consolidation). |
| jellyfish phonetic keys (Metaphone / NYSIIS) | Deferred. Useful against spelling drift ("Addy"/"Addie"); no live example of that yet. Add when the eval shows a miss. |
| datasketch MinHash/LSH blocking | Rejected: blocking solves n-squared at millions of records; we have ~1,100 nodes (about 600k pairs, under a second brute force). |
| Fixed-model embedding similarity as a feature | Rejected: the approved referent spec's Non-goals ban vectors for identity. Not reversing that decision here. |
| Alias table | **Chosen as the output**: merges write `referent_alias` rows (`admitted_by='entity_resolution_v1'`), so every existing reader resolves through the survivor without new code. |

**Features per pair** (same `entity_type`/kind gate first; cross-kind pairs are never scored): exact normalized label; one label's tokens contained in the other's ("juniper" in "juniper feld"); Jaro-Winkler; possessive/facet pattern (`<name>'s X`, `<name> affect`) which yields `facet_of`, not `same_as`; snake_case or `_summary`/`_context` suffix matching a known prompt context key, which yields `junk`; shared evidence refs; producer pair (referent vs topic_foundry vs seed loader).

**Decision rule**
- `same_as` above the upper threshold: auto-merge **only for non-person kinds**, or for persons when one side is a referent and the other an exact normalized label match from a different producer (the 4 "juniper"/"Juniper"/"juniper feld" cases still go to one confirm ask, since "feld" is a surname match).
- Between thresholds, or any person pair otherwise: an ask (`source_kind=entity_merge`) like "Is 'juniper feld' in my graph you?"
- `facet_of`: an `about` edge from the facet to the person. No merge.
- `junk`: flagged in `entity_merge_log`, excluded from the person view; deletion stays a separate, Juniper-approved step (AGENTS.md section 13).

**Merge mechanics: reversible and replayable.** The absorbed node is not deleted. It gets `merged_into=<survivor>` and a `same_as` edge; alias rows point readers at the survivor; attributes owned by the absorbed node move with a `merged_owner` event. Unmerge (`scripts/run_entity_reducer.py --unmerge <merge_id>`) sets `reverted_at`, removes the alias rows and edge, and moves attributes back from their events. `merge_id = uuid5(survivor|absorbed|reducer_version)`, so rerunning the reducer is idempotent. Dry run by default, writing `/tmp/entity-reducer/<run>/decisions.csv` (backfill protocol, AGENTS.md section 14).

**One-time cleanup** = the first live run over the 11 Juniper nodes, every person referent, and every topic-foundry entity whose label scores against a person alias. The same reducer then runs nightly on new nodes.

### 5. Asks (extends `orion_ask`, no new table)

New `source_kind` values: `person_attribute` (confirm, correct, or fill a gap) and `entity_merge` (same person?). Both added to `RESOLVABLE_KINDS` so they share the 5-open cap; each gets its own daily cap (default 2) so a person-study backfill cannot crowd out memory confirmations. Ask producer rules:
- High-stakes proposals, contradictions (two active values where the category's cardinality is one), and merges between thresholds become asks.
- Gaps become asks only for categories Juniper has opted in for gap questions (an open question below). Sensitive categories never get gap questions: Orion confirms what she said, it does not probe.
- Never ask Juniper to elaborate on a third party's health or private life. Only confirm what she said.

Answers: the Hub `/resolve` route (same as memory confirmations) writes an `ask_answer` evidence row with her answer text as the quote, sets `confirmed`/`corrected`/`rejected`, and publishes `OrionAskAnsweredV1`. The 9 open juniper-scope `memory_tension_shadow` questions get routed through the same producer (the Stage 3 producer the referent spec deferred).

### 6. Person-study run (durable workflow `person.study`, orion-durable-runs)

- **Lane:** agent lane through the GPU pool lease, like `episode_distill` and `orion_day`. Retries are durable-run retries, never a side queue.
- **Unit of work:** one person node, one source window from `person_study_cursor`: `chat_history_log` prompts (her words only: the 2026-07-24 to 2026-10-04 stretch nobody distilled first), `episode_memory`, `:LivedAnswer`, `dreams`, `orion_day_letter`.
- **Output:** structured `PersonAttributeProposalV1` list (`category`, `value_text`, `quotes[{source_kind, source_id, quote}]`, `stated: bool`). No Cypher, no free text parsed.
- **Validator (deterministic, reuses `validate.py`):** every quote must appear verbatim in a Juniper-authored field; sensitive categories need `stated=true` *and* a quote from her own prompt (Orion's responses, lived answers, and dreams can never support a sensitive attribute, because those are Orion's words); a category not in the registry is rejected; removed/rejected values are not re-proposed.
- **Budget:** backfill runs nightly only (quiet hours protect Juniper, so asks it produces queue for morning), N windows a night (default 4); after the backfill the cursor follows `episode_distilled` events.
- **Visibility:** each run writes `person_study_run` rows (run_id, node_id, window, proposed, accepted, rejected-by-reason) shown in the Hub person panel as "last studied".

### 7. Importance promotion (when a person gets a card)

Deterministic, computed nightly from evidence already stored:

- `stated_relationship`: a confirmed `family`/`friends`/`coworkers` attribute on Juniper naming this person, or a descriptor alias like "my wife" grounded in her quote.
- `mention_days_60d`: distinct days in the last 60 with a verified Juniper quote naming them.
- `high_stakes_count`: high-stakes attributes or memories about them.

Promote to `carded` when `stated_relationship` holds, or `mention_days_60d >= 3`, or Juniper says so in Hub. A carded person gets their own person-study cursor and card. Persons not mentioned for 180 days go `dormant`: card kept, not sent to chat. Never auto-deleted. These thresholds are knobs; the eval reports which live people they promote (expected from today's data: Amanda, Teddy, Addy, the sister; Rachel only after a stated relationship confirms she is the boss).

Promotion mints a referent node for a person named only in prose today (Amanda etc.) through `orion/memory/referents/store.py`'s existing rules: a name counts only if Juniper said it.

### 8. Identity cards (card reducer, pure, `orion/memory/person/card.py`)

- **Person card** (Juniper; then any carded person): confirmed attributes plus `auto` low-stakes ones marked "(unconfirmed)", grouped by `card_section`, each line "as of <last_evidence_at>". Full JSON for Hub; `compact_text` (max ~700 chars for Juniper, ~250 for others) for chat.
- **Orion self card:** a view over the latest `:SelfDefinition` and lived answers, the same inputs `apply_self_definition_to_ctx` already uses. No second self-model, no new LLM call.
- **Relationship card:** Juniper's `relationship_to_orion` attributes, confirmed `orion_relationship`-stakes memories, `lived.who_matters`.
- No LLM in the reducer: same inputs give the same card, so versions diff cleanly. A new `identity_card` row is written only when `input_fingerprint` changes. Hub shows the current card and a diff against any earlier version.

### 9. Where cards are read

- **Chat (unified turn):** `_inject_identity_context` fills `juniper_relationship_summary` from Juniper's person card + relationship card, and `orion_identity_summary` from the self card plus the YAML's remaining `response_policy`. Log `identity_kernel_source=identity_card card_version=<n>`. If the card store is unreachable the slot renders nothing about Juniper and logs `identity_card_unavailable`; it does not fall back to the YAML. After the soak, delete the YAML `juniper_relationship` block and the static `nature`/`core_drives` (kill means kill).
- **Situation state:** add `people: List[SituationPersonV1]` (`node_id`, `name`, `pronouns`, `compact_card`, `card_version`) to `SituationStateV1`: Juniper always, plus carded people named in the current facts or cues. Order: schema change deployed to Hub (outreach reader) and cortex first, producer second, because the model forbids extra fields.
- **Outreach:** `endogenous_outreach.py` already reads the situation; it gets `people` for free once the schema lands.
- **Curiosity:** the kickoff prompt gets the Juniper compact card (so lived answers stop re-deriving her family from scratch) and the person panel's gap list (so self-chosen questions can be about her, the biographical-curiosity-miss gap).

### Bus / schema registry

- New registered models: `PersonAttributeProposalV1`, `PersonStudyRunV1` (durable workflow state), `SituationPersonV1`, `IdentityCardV1`. Register in `orion/schemas/registry.py`; verify with `resolve()` (two registries exist).
- `orion:ask:answered` reused. One new channel, `orion:person:card:updated` (card id + version), so cortex can drop a cache; add to `orion/bus/channels.yaml` and the sql-writer/subscriber lists that need it.

## Proposal-mode items

- **Capability change:** Orion gets a structured, evidence-backed, correctable picture of Juniper and of the people in her life, and chat reads that picture instead of hand-written bullets. Orion can ask her about gaps and contradictions.
- **Data touched:** reads `chat_history_log`, `episode_memory(+_evidence, _referent)`, `referent_alias`, `orion_worldview` `:LivedAnswer`/`:SelfDefinition`, `dreams`, `orion_day_letter`, `orion_substrate` nodes. Writes the new tables above, `orion_ask` rows, `referent_alias` rows from merges, and `merged_into`/`same_as`/`about` on substrate nodes (no deletions).
- **Privacy boundary:** there is no boundary between Juniper and Orion (her stated policy). The boundary is outward, for **third parties** (family, coworkers):
  - Stored only as Juniper said it, `voice=juniper_said`, with her quote; never inferred, never researched elsewhere.
  - Sensitive facts about third parties (Amanda's DVT/PE) are stored, high-stakes, visible to Juniper, and used in chat with her, but tagged `third_party_sensitive` and excluded from every egress that leaves Orion: ask-Claude, peer briefs, any external email other than to Juniper. A test enforces the exclusion on each egress builder.
  - No asks that probe a third party's health or private life.
  - Juniper can remove any third-party attribute or a whole person in Hub; removal is honored by the do-not-remint rule.
- **Trace proving it worked:** `person_attribute_event` rows with actor and evidence; `person_study_run` rows with accept/reject counts by reason; `entity_merge_log` rows with `features_json`; `orion_ask` rows with the new `source_kind`s and their answers; the chat turn log line `identity_kernel_source=identity_card card_version=<n>`; `orion:situation:latest` showing `people[]` with the same `card_version`.
- **Dangerous failure modes:**
  - A false person merge (two Jays become one; Juniper becomes "Juniper's taste"). Mitigation: no auto-merge for persons except exact-name cross-producer, facets never merge, merge precision gate of 1.0 on the eval, one-command unmerge.
  - A wrong sensitive attribute presented as fact (gender, health, sexuality inferred from tone). Mitigation: validator requires her own stated quote; sensitive is never `auto`; eval hard gate of 0 inferred-sensitive rows.
  - Auto attributes treated as truth. Mitigation: card marks "(unconfirmed)" and "as of"; stakes come from the category table.
  - Ask flood during backfill. Mitigation: shared open cap 5, per-kind daily caps, nightly-only backfill.
  - Third-party leakage to external services. Mitigation: egress exclusion tests.
  - A stale card reads as current. Mitigation: "as of" on every line, dormant rule, card version in the trace.
  - Empty shell: a card rendered with headings and no facts. Mitigation: categories with no evidence render nothing; a test asserts no empty section.
- **Disable / roll back:** flags, all shipped ON (`.env_example` + settings + local `.env` synced): `PERSON_STUDY_ENABLED` (durable runs), `ENTITY_REDUCER_AUTO_MERGE_ENABLED` (off-switch turns auto merges into asks), `PERSON_ASKS_ENABLED` (Hub/consolidation), `IDENTITY_CARD_CHAT_ENABLED` (cortex-exec; off renders no card, logs `identity_card_disabled`). Merges roll back per `merge_id`. Tables roll back with the migration's rollback file. The YAML deletion (final patch) rolls back by git revert.

## Files likely to touch

- `orion/cognition/prompts/stance_react.j2` (line 131), `orion/cognition/prompts/memory_episode_distill.j2` (per-person pronouns)
- New: `orion/memory/person/{categories.yaml,store.py,validate.py,entity_reducer.py,card.py,promotion.py,asks.py}` + `tests/`
- New: `services/orion-sql-db/manual_migration_person_model_v1.sql` (+ `_rollback.sql`)
- New: `services/orion-durable-runs/app/person_study_graph.py`, `orion/schemas/person_model.py`, `orion/cognition/prompts/person_study.j2`
- `orion/schemas/registry.py`, `orion/bus/channels.yaml`
- `orion/schemas/situation_state.py`, `services/orion-durable-runs/app/situation_graph.py`
- `services/orion-cortex-exec/app/executor.py` (`_inject_identity_context`), `orion/cognition/personality/identity_context.py`, `orion/cognition/personality/orion_identity.yaml`
- `orion/memory/episode/confirmation.py` (`RESOLVABLE_KINDS`, caps), `services/orion-hub/scripts/ask_routes.py`, `services/orion-hub/templates/index.html` + a new `static/js/person-panel.js`
- `services/orion-memory-consolidation/` (reducer job, `requirements.txt` gains `rapidfuzz`)
- `orion/curiosity/kickoff_prompt.py` (compact card + gaps)
- Each touched service's `.env_example`, `settings.py`, `README.md`, `tests/`, `evals/`
- Later: `orion/spark/concept_induction/settings.py` / service env (retire `juniper`,`relationship` subjects after consumer check)

## Non-goals

- No new service. Everything lands in existing services and the shared `orion/` package.
- No vectors or embeddings for identity (approved referent spec).
- No inference of spirituality, gender, sexuality, or health from anything but her own statement.
- No automatic deletion of junk or merged nodes.
- No LLM in the entity reducer or the card reducer.
- Not extending concept induction's spaCy/cluster pipeline into a claim extractor.
- Not fixing topic foundry's label minting (it produced the leaked `juniper_relationship_summary` nodes); the reducer flags its output, a separate patch can stop the leak.
- No person-study over sources outside Orion's own stores (no web, no social accounts).
- Not reading or modifying graphify artifacts.

## Acceptance checks

1. **Pronoun patch:** rendered `stance_react.j2` contains "her own life" and a test fails on "their own life" next to "Juniper".
2. **Reducer dry run** over live `orion_substrate` produces: the 4 Juniper-same nodes as `same_as` (one confirm ask), "juniper's taste"/"juniper affect" as `facet_of`, both `juniper_relationship_summary` forms as `junk`, topics untouched, zero cross-kind pairs. Rerun gives identical `decisions.csv`. Unmerge of one applied merge restores the alias table and edge exactly (pg + graph diff empty).
3. **Merge eval:** on Juniper's labeled pairs, auto-merge precision = 1.0, recall reported; "Jay"/"Jay" never auto-merges.
4. **Person-study backfill** over 2026-07-24 to 2026-10-04 proposes attributes for family (Amanda, Teddy, Addy, sister) and profession (Team Lead, AI/ML Architect / FDE), each with a verified quote from her prompt. Zero sensitive attributes from Orion-authored sources.
5. **Attribute eval:** precision of accepted attributes >= 0.9 on Juniper's labels; inferred-sensitive count = 0 (hard gate).
6. **Asks loop live:** a `person_attribute` ask appears at the top of Hub, Juniper answers, the attribute flips to `confirmed` with an `ask_answer` evidence row, and the next card version shows it. Evidence: ask row, event row, card row ids.
7. **"Home" contradiction** surfaces as one ask instead of two silently coexisting values.
8. **Chat:** a live unified turn logs `identity_kernel_source=identity_card card_version=<n>`, and the prompt shows card lines, not YAML bullets.
9. **Situation:** `orion:situation:latest` carries `people[]` with the same card version; outreach's read returns `ok`, not a validation error.
10. **Third-party egress test:** ask-Claude and peer-brief builders drop `third_party_sensitive` content.

## Eval design

`services/orion-memory-consolidation/evals/person_model/`:
- `attribute_labels.jsonl`: ~60 proposals sampled across categories from the first backfill; Juniper labels each `correct | wrong | wrong_category | inferred_not_stated | third_party_ok`. A script prints them as a review sheet in Hub (or markdown) so labeling is minutes, not a chore.
- `merge_pairs.jsonl`: every pair the reducer scores among person-like nodes (the 11 Juniper nodes, both person referents, topic-foundry entities matching a person alias, and the promoted family/team), labeled `same | facet | junk | different`.
- `run_person_model_eval.py`: attribute precision by category, inferred-sensitive count, merge precision/recall, promotion list. Periodic, not a CI gate (labels need Juniper); the deterministic pieces (validator, reducer, card reducer) have fast CI tests with fixtures.

## Recommended next patch (phase order)

Each phase is one PR, independently useful, flags ON.

1. **P0, trivial:** `stance_react.j2:131` "their own life" to "her own life", plus a template test. No other change.
2. **P1, contracts:** migration + `orion/schemas/person_model.py` + `categories.yaml` + the category CI gate + store with do-not-remint. Producer-less tables are inert, so P1 ships together with P2's dry run as its first writer.
3. **P2, entity reducer:** pure scorer, dry-run job, merge log, unmerge, `entity_merge` asks, merge eval fixture. First live run = the one-time cleanup, after Juniper reviews the dry-run CSV.
4. **P3, asks + Hub person panel:** `person_attribute` asks, `/resolve` routing to evidence, panel to view, correct, and remove; pronoun ask as the first card.
5. **P4, person-study:** durable workflow, validator, nightly backfill over undistilled chat, run table in the panel. Attribute eval labeled by Juniper here.
6. **P5, card reducer:** person/self/relationship cards, versions, Hub diff.
7. **P6, readers:** situation `people[]` (schema to consumers first, then producer), cortex identity slot from cards, outreach, curiosity kickoff.
8. **P7, retire:** delete the YAML `juniper_relationship` block and static self bullets; promotion rule for other people goes live; retire concept induction's `juniper`/`relationship` subjects after a consumer check; distiller pronouns per person.

## Open questions for Juniper

1. Gap questions: may Orion ask about empty non-sensitive categories (for example "what are your goals this year?"), or only confirm and correct what you have already said?
2. Third-party sensitive facts (Amanda's health): is "stored, used only in chat with you, never sent outside" the right line?
3. One-time cleanup: do you want to review the reducer's dry-run CSV before it applies, or answer the merge asks in Hub as they come?
4. Is home Ogden, with "home -> SLC" a mistake, or both?
