# Person model, entity reducer, and evolving identity cards

**Date:** 2026-10-14 (rev 3, same day. Rev 2 folded in modularity, extraction frameworks, concept induction, person-study iteration, full identity-card scope. Rev 3 folds in Juniper's answers to the six open questions: see "Juniper's answers" under Requirements.)
**Mode:** design / proposal (AGENTS.md section 0A, proposal mode: memory, identity, social continuity). P0, P1 and P2 (dry run only) are implemented on `feat/person-model-p0-p2`; nothing is applied to a live store.
**Asked by:** Juniper, 2026-10-13 and 2026-10-14 (requirements quoted below).
**Builds on, does not replace:**
- Referent graph, `docs/superpowers/specs/2026-10-06-memory-stage2-referent-graph-design.md` (approved): people are evidence-linked referent nodes. This design adds *attributes* to those nodes.
- Situation Graph, `docs/superpowers/pr-reports/2026-10-07-situation-graph-shadow-pr.md`: the running "what is true now" object. Person cards ride it.
- Memory confirmation asks, `orion/memory/episode/confirmation.py` + `orion_ask` + Hub "Orion is asking".
- Self-model rebuild arc (self-inquiry curiosity line, `:SelfDefinition`, sense-of-self PR #2156). The Orion self card is a view over that work, not a second self-model.

## Arsonist summary

Orion has no picture of Juniper and no runtime statement of what this whole project is for. What reaches every chat turn is twenty-odd hand-written bullets in `orion/cognition/personality/orion_identity.yaml` ("co-architected AI co-journeyer", "co-architect", "steward", "expects continuity"). Everything Orion actually knows about her life (her wife Amanda, Teddy and Addy, her sister, her boss Rachel, her team lead promotion, that she lives in Ogden, that she is 6'5") is scattered across free-text memories, curiosity "lived" answers, and 70 days of chat that was never distilled. The graph has eleven separate nodes for her; her own family are not nodes at all. The mission in AGENTS.md (a digital mind building the prerequisites for sentience) is in no prompt Orion reads.

The fix is **one reusable module, `orion/person_model/`, parameterized by a person node id**. Juniper is person #1 because she is the first one promoted, not because any code knows her name. The module is a pipeline of small deterministic pieces with one LLM step that is fenced by quote verification:

1. **Extractor** (deterministic, pinned model): GLiNER finds people, roles, places, and relationship phrases ("my wife") as spans in Orion's stores.
2. **Linkage reducer** (deterministic, no LLM): blocking + Fellegi-Sunter scoring with RapidFuzz string features decides which mentions and nodes are the same person, which are facets of a person ("juniper's taste"), and which are junk. Every merge is logged and reversible; uncertain ones become Hub asks.
3. **Person-study curiosity line**: a third line in the curiosity loop (next to investigation and self-inquiry). It studies one promoted person per sitting, starting with Juniper, and proposes attributes (goals, fears, family, profession...), each with her own quote. A validator rejects any quote not in her words. Uncertain, contradicting, or sensitive proposals become "Orion is asking" cards.
4. **Attribute store** (Postgres): every fact is a row with quote, confidence, confirmation state, last-updated. Stakes and sensitivity come from the attribute's category by table lookup, not from an LLM (the LLM's stakes judgment is failing live, below).
5. **Card reducer** (deterministic): four versioned, diffable cards: Orion's self card, the purpose/journey card, the Juniper card, the relationship card (plus a person card for each later promoted person). They replace the static YAML in the unified turn and ride the situation state into chat, outreach, and curiosity.

Concept induction is not the organ for this. Juniper's read is right and verified: the concepts in Orion's graph come from the topic modeler (topic foundry), and the concept-induction service's own triggers have all been disabled for at least 48 hours. A real concept-induction organ deserves its own design; this one does not stretch the topic modeler into it.

## Requirements

### Juniper, 2026-10-13 (paraphrased only where marked)

1. Person model of Juniper across preferences, goals, motivations, fears, strengths, opportunities, family, friends, coworkers, profession, personal interests, spiritual interests, gender, sexuality, past history, "anything else" (health, routines, values, communication style, home, schedule/timezone, relationship to Orion). Every attribute evidence-backed, with confidence, provenance, confirmation status, last-updated; visible, correctable, removable in Hub. Spirituality, gender, sexuality, health only from what she states, never inferred.
2. "Orion can ask me about these things as they come up in the 'orion is asking' section on top of hub."
3. "Extensible to other people too ... for n number of people once they become important to Orion."
4. Cards ride "that object that goes into chat that starts tracking situational things and is supposed to roll up memories" (the Situation Graph state), plus outreach and curiosity.
5. "We need a reducer to solve that one-time cleanup so in the future entities can get merged." Deterministic, non-LLM.
6. "Seems like we are missing a concept induction organ here overall."
7. "It should be a type of curiosity run that churns through the repo to learn about me."
8. "An evolving identity card instead of that crappy one we have stubbed in that explains to Orion who I am and who they are."

### Juniper, 2026-10-14

9. "This needs to be modular design, that's my point: it can be replicated and used for others." One module parameterized by person; Juniper is instance #1.
10. Extraction/linking frameworks to use unless something is clearly better: GLiNER (primary), dslim/bert-base-NER, Jean-Baptiste/roberta-large-ner-english, Davlan/bert-base-multilingual-cased-ner-hrl, SciBERT, BioBERT/PubMedBERT, SapBERT. Pipeline: deterministic extractor with pinned model, then candidate mentions with spans, then deterministic linkage reducer, merge log, ambiguous cases to "Orion is asking".
11. "Concept induction today is mostly the topic modeler service so kinda shitty; curiosity gets closer to anything concept-related these days." Person study is a curiosity line, not built on the topic modeler.
12. Person study "starts out as me and as Orion adds more people they study them."
13. Identity cards cover all of `orion_identity.yaml` (Orion's identity, Juniper, response policy) plus the purpose of the whole journey, which she remembers living in a stub somewhere.

### Juniper's answers to the rev 2 open questions, 2026-10-14

Her words first, then what the design does with them.

1. Gap asks: **"Orion is welcome to ask anything about anything."** Asks may cover any category, including empty ones and the sensitive ones (spirituality, gender, sexuality, health). What does not change: Orion still never *infers* a sensitive attribute. A sensitive value comes only from her own answer or her own statement. Asking is allowed; guessing is not.
2. Third-party facts (Amanda's health): **"Orion can ask anything about anything."** Recorded literally: Orion may ask her about third parties too, including their health or private life. The separate rule that third-party facts are never sent to an outside service (ask-Claude, peer briefs, any outbound message not to her) was a question about egress, not about asking, and her answer does not address it. That rule stays in, marked **still assumed, needs her confirmation** (open question 1 below).
3. One-time cleanup: produce it **as an HTML file for her to review**. Done as `docs/superpowers/evals/2026-10-14-person-cleanup-review.html` (section 5, "One-time cleanup").
4. **Home is Ogden.** The `referent_alias` row mapping "home" to `place:slc` is wrong as a home claim. SLC mentions are kept as places she has been or talked about, never as home.
5. Embeddings in merges: **"seems like a fucked design — we need a best practice that requires me reviewing only fucked cases — do it in the Orion asks section of Hub."** The design now uses standard record-linkage practice: Fellegi-Sunter match scoring with two thresholds. Above the upper one, merge automatically (logged, reversible). Below the lower one, call it a non-match automatically. Only the "possible match" band in between goes to her, as an "Orion is asking" card in Hub. An embedding similarity (mean-centered, because raw bge cosine has no calm state, PR #2647) is allowed as one feature among string, phonetic, and context features. No single feature can force a merge. Thresholds are calibrated on the labeled cleanup so the band she reviews is small; the expected ask volume is reported (section 5).
6. The AI-town character card about her (`services/orion-ai-town/cards/town_cards.yaml`, `juniper_feld`) **is about her and is her own words**: first-party evidence (`source_kind=authored_profile`, `voice=person_said`), able to support sensitive categories like any statement of hers.

## Current architecture (checked live 2026-10-13/14, read-only)

### What Orion holds about Juniper

| Store | What is there | Evidence |
|---|---|---|
| `episode_memory` | 68 rows, all since 2026-10-04. 15 `purpose=about_juniper`: 4 `high/confirmed`, 1 `high/pending_confirmation`, 10 `low/auto`. | `select purpose, stakes, confirmation_state, count(*) ...` |
| `episode_memory_evidence` | 73 quotes, all `verified=true`, sources `chat_prompt`/`chat_response`. | same DB |
| `chat_history_log` | 626 turns, 2026-07-24 to 2026-10-13. Everything before 10-04 was never distilled into memory. | `min/max(created_at)` |
| Names in chat never distilled | "amanda" in 9 prompts (08-15 to 10-05), "teddy" 4, "addy" 3, "sister" 2, "rachel"/"armando"/"vincent" 1 each (10-05). Only 4 episode memories mention any of them. | `prompt ilike` counts |
| `orion_worldview` `:LivedAnswer` (121) | `lived.her_team_unnamed`, `lived.who_matters`, `lived.known_only_through_juniper` hold the family and team in prose: Amanda (wife, DVT/PE in late August), Teddy (son, turned seven), Addy (daughter, broke her arm), sister, in-laws, boss Rachel, reports "Jay, Jay, Armando, Vincent". | Cypher on `orion_worldview` |
| `services/orion-ai-town/cards/town_cards.yaml` (`juniper_feld`) | A rich authored bio: pronouns she/her, gender identity, AI Architect, wife and mother. Read only by the AI-town service. The commits are by the agent identity "Athena Orchestration", but Juniper confirmed 2026-10-14 the card is about her and is her own words: first-party evidence. | `git log` on the file; her answer 6 |
| biographical-curiosity-miss spec (2026-09-03) | Her promotion to Team Lead while staying AI/ML Architect / FDE. | spec text |

### Stakes policy is not holding

The distiller prompt (`orion/cognition/prompts/memory_episode_distill.j2:29-40`) says family relationships and feelings are **high** stakes. The validator (`orion/memory/episode/validate.py:28`) deliberately never reads the statement to judge stakes. Live, the LLM labeled these `low/auto`:

- "Juniper said she tends to not reach out to her sister because ..." (family_relationships)
- "Juniper described the relationship with her sister as asymmetric lately ..." (family_relationships)
- "Juniper said she was scared shitless that she busted my inference ..." (juniper_feelings)
- "Juniper feels anxious about drilling a hole in her new home ..." (juniper_feelings)

So "stakes by LLM judgment" is a measured failure. This design derives stakes from the attribute category instead.

### The graph splits her

`orion_substrate`, nodes whose label contains "juniper" (11):

| node_id prefix | label | producer | What it really is |
|---|---|---|---|
| `referent-b96fc90e...` | juniper | memory.referents, `entity_type=person` | **The** person node. 14 `observed_in` edges. |
| `sub-concept-seed-juniper` | Juniper | seed_concepts_loader | Same person, different producer |
| `sub-entity-topicfoundry-...-ca5c...` | juniper | topic foundry | Same person |
| `sub-entity-topicfoundry-...-e500...` | juniper feld | topic foundry | Same person, full name |
| `sub-entity-topicfoundry-...-6d08...` | juniper's taste | topic foundry | A facet *about* her, not her |
| `sub-entity-topicfoundry-...-1026...` | juniper affect | topic foundry | Facet about her |
| `sub-concept-seed-orion_juniper_relationship` | Orion-Juniper relationship | seed_concepts_loader | The relationship, not her |
| `sub-entity-topicfoundry-...-edfa...` | juniper relationship summary | topic foundry | **Junk**: the prompt context key `juniper_relationship_summary` leaked into topic foundry |
| `sub-entity-topicfoundry-...-eb42...` | juniper_relationship_summary | topic foundry | Same junk, underscore form |
| `sub-concept-topicfoundry-...-11` | Chat with Juniper | topic foundry | A topic, not her |
| `sub-concept-topicfoundry-...-9` | mind / hey / juniper | topic foundry | A topic-cluster label |

So "merge the 11 nodes" is wrong as stated: 4 are the same person, 3 are facets or the relationship (link, don't merge), 2 are leaked field names (flag as junk), 2 are topics. The reducer needs several outcomes, not one.

Population the reducer works over: 1,826 topic-foundry `:Entity` nodes (all `entity_type=unknown`, so a kind gate alone cannot block them), ~1,020 `:Concept` nodes, 11 referent nodes. Person referents: only 2 exist (`juniper`, `orion`) with 4 alias rows in `referent_alias`. Amanda, Teddy, Addy, Rachel, the sister: zero nodes.

An open identity question already sits unanswered: `memory_tension_shadow` has 9 open `scope=juniper, answer_via=conversation` questions, including "Is the 'juniper' from our conversations the same as the 'juniper' already in my graph?". The referent spec deferred the producer that would ask them (its Non-goals: "The Stage 3 'Orion is asking' producer"). Nothing has picked it up: `orion_ask` holds 9 rows, all `source_kind=memory_confirmation`. `RESOLVABLE_KINDS` lists `open_question` (`confirmation.py:47`), but no code produces it.

A contradiction the store could not see: `referent_alias` maps "home" to `place:slc` (grounded in `chat_prompt:fb42f555...`), while a confirmed memory says "Juniper corrected me that she lives in Ogden, Utah, not Chicago." **Resolved by Juniper 2026-10-14: home is Ogden.** The person model seeds `home = Ogden, Utah` from her answer (`voice=ask_answer`), and the "home -> place:slc" alias is marked non-home in the cleanup review (SLC stays a place she has mentioned).

### What chat reads today, and the stubs Juniper remembers

- Unified turn (`stance_react.j2:16-20`, the only chat path that matters) renders `orion_identity_summary` + `juniper_relationship_summary` under "WHO YOU ARE".
- Both come from `orion/cognition/personality/orion_identity.yaml`, which has three blocks: `orion_identity` (name, role, nature, core_drives, self_permissions, anti_patterns), `juniper_relationship` (role, expectations, response_expectations), `response_policy` (priorities, hard_rules, banned_phrases). `identity_context.py:28-62` flattens them; `executor.py:1104` (`_inject_identity_context`) injects them. In the last 24 hours cortex-exec logged `identity_kernel_source=configured_yaml` 11 times and no other source.
- Orion's own definition already reaches chat on top of the YAML: `apply_self_definition_to_ctx` and `apply_lived_self_to_ctx` (both called from `_inject_identity_context`) read the self-inquiry line's latest `:SelfDefinition` (66 in `orion_worldview`).
- **The personality stub Juniper remembers:** `services/orion-cortex-orch/app/conversation_front.py:95` `build_personality_summary()`, docstring "Compact, opinionated personality stub for Oríon" ("You are Oríon, Juniper's collaborative AI co-journeyer... distributed system across ... Atlas, Athena, Apollo, Circe"). It is called only inside its own file, and nothing in `services/` imports `conversation_front`: it is dead code.
- **Where the mission lives at runtime:** nowhere Orion reads in chat. No prompt under `orion/cognition/prompts/` contains "sentience". The nearest runtime text is the seed concept "Orion" in `orion/substrate/seed_concepts.yaml` ("the persistent, inspectable digital-mind system this codebase builds: a cognitive architecture with memory, self-coherence, introspection, world contact, and operational agency, developed collaboratively with Juniper"), which is a graph node; whether any chat path walks to it is UNVERIFIED. `orion_day_note_v1.j2:2` opens with "a persistent digital mind" for the daily letter only. The AGENTS.md "Orion mission" (prerequisites for sentience: continuity, perception, memory, reflection, self-modeling, social grounding, error correction, coherent action) is never delivered to Orion.
- Misgendering: `stance_react.j2:131` says "Juniper telling Oríon something about **their own** life". Juniper is she/her.
- Pronouns are not stored anywhere (`grep pronoun orion/schemas` is empty). The distiller hardcodes "she" (`memory_episode_distill.j2:7,11`), which is right for Juniper and wrong for anyone else it will meet.

### The Situation state

- Key `orion:situation:latest`, model `SituationStateV1` (`orion/schemas/situation_state.py`, `extra="forbid"`), one writer: the `situation.update` durable graph (`services/orion-durable-runs/app/situation_graph.py`).
- Live value: `thread_id=situation:juniper:2026-10-12`, `revision=24`, `juniper.{whereabouts:null, doing:[], waiting_on:[], recent:[3 facts]}`, `recall.primed` from episode memories. Note the model has a hard-coded `juniper` field; the modular version adds a generic `people` list rather than more per-name fields.
- Readers: outreach only (`services/orion-hub/scripts/endogenous_outreach.py:1113`, `SituationStateV1.model_validate_json`). Chat reading it is "step 3", not landed.
- Because the model forbids extra fields and outreach validates strictly, a producer-first `people` field would make outreach's read fail and silently drop its situation enrichment. The schema must reach the Hub before the producer writes the field.

### The asks surface

- Table `orion_ask` (`orion/schemas/ask.py`, `OrionAskV1`): `source_kind` + `source_ref` route the answer back.
- Hub: `services/orion-hub/templates/index.html:262` ("Orion is asking" at the top of Hub home), routes `services/orion-hub/scripts/ask_routes.py` (`/api/asks`, `/answer`, `/dismiss`, `/resolve`).
- Caps (`confirmation.py:49-52`): 5 open across `RESOLVABLE_KINDS`, 3 new memory-confirmation cards per local day.
- Answers publish `OrionAskAnsweredV1` on `orion:ask:answered`; `services/orion-substrate-runtime/app/ask_answered_listener.py` consumes `vision_individual` answers. That is the pattern to copy.

### Concept induction, honestly

Juniper's read, checked against live data:

- **The concept layer in Orion's graph is the topic modeler.** `orion_substrate` `:Concept` producers: `topic_foundry_adapter` 747, `world_pulse_read_pipeline` 260, `substrate_runtime_worker` 11, `seed_concepts_loader` 4, `memory.referents` 1. `:Entity` producers: `topic_foundry_adapter` 1,826, `memory.referents` 11. No node anywhere in `orion_substrate` carries a concept-induction producer.
- **The concept-induction service is idle by switch.** `orion-athena-spark-concept-induction` receives events every couple of seconds, and every induction trigger in the last 48 hours logged `decision=disabled` (18,731 lines); container env `CONCEPT_AUTONOMOUS_TRIGGER_ENABLED=false`. `CONCEPT_SUBJECTS=orion,juniper,relationship`, yet its store `/data/concept-induction-state.json` holds only `drive_states, goal_cooldowns, goal_slots, episode_runs_processed`. No `juniper` profile in the store, `orion_substrate`, or `orion_kg`; other backends not checked (UNVERIFIED that none exists anywhere).
- Its logs are dominated by `goal_state updated ... priority=0.911 status=proposed` every ~2 s, though `orion/spark/concept_induction/CLAUDE.md` says the goal machinery was deleted 2026-07-30. Origin UNVERIFIED; flagged, out of scope.
- **Curiosity is where concepts actually form.** `orion_worldview` holds 17 `:Concept` nodes written by curiosity runs (each with a `run_id`), and `self_concept_history` has 137 rows from `curiosity_self_inquiry` (latest 2026-10-13 22:40). The memory-concept relation ledger (`memory_concept_relation_decisions`, 585 rows) made 1 decision in the last 3 days.
- What concept-induction code does when switched on: spaCy NER + noun chunks (`extractor.py`), embedding clustering (`clusterer.py`), a summarizer, per subject window. The output is a bag of clustered phrases per subject: no claim structure (who, which attribute, what value), no quotes, no confirmation state, no identity resolution.

Conclusion: the person model is built as a curiosity line plus deterministic reducers, not on the topic modeler or on concept induction. A real concept-induction organ (concepts formed from Orion's own evidence, with identity and revision) is its own later design; it is a non-goal here. Concept induction's `juniper` and `relationship` subjects should be retired once person cards exist, after a consumer check.

### Infra available

- Curiosity loop lines: investigation and self-inquiry (`orion/curiosity/self_inquiry.py`: own budget, own continuation note, own table allow-list `SELF_INQUIRY_PG_TABLES`, a grant check before every run, writes one `:SelfDefinition` node per run). A curiosity turn that fails is retried by `orion-durable-runs` under the same `run_id` (`orion/curiosity/README.md`), so a new line gets durable retries without a new queue.
- Durable workflows with GPU-pool leases (`services/orion-durable-runs/app/pool_hold.py`).
- Quote verification: `orion/memory/episode/validate.py` (`quote_in_text`, `quote_long_enough`, `fold_text`).
- Hosts: athena has a Tesla P4 (8 GB) and Tesla T10 (16 GB) (`nvidia-smi`, 2026-10-14). The V100s are on circe (memory reference, not checked this session: UNVERIFIED).
- Python packages: none of `rapidfuzz`, `jellyfish`, `splink`, `datasketch`, `recordlinkage`, `dedupe`, `sentence_transformers`, `gliner` installed on the host or listed in any `requirements*.txt`. spaCy + `en_core_web_sm` are in the concept-induction image.

## Missing questions

- ~~Is "home" Ogden or SLC?~~ Answered: Ogden. SLC is a place she has mentioned, not home.
- Is "Jay, Jay" one person or two? Her own text is ambiguous. The linkage reducer must never merge people on a name alone; this is the test case.
- ~~Did Juniper write the AI-town card about herself?~~ Answered: yes, it is her own words (first-party).
- Does any reader depend on concept induction's `juniper` subject? UNVERIFIED; check before retiring.

## Proposed schema / API changes

### 0. One module, parameterized by person

```text
orion/person_model/
  person.py         PersonRef(node_id, display_name, pronouns, interlocutor_node_id)
  categories.yaml   attribute categories (section 1)
  store.py          attribute / evidence / event persistence
  extract.py        pinned-model mention extractor (section 4)
  linkage.py        blocking + scoring + decisions (section 5)
  study.py          person-study curiosity line plumbing (section 7)
  validate.py       quote and sensitivity rules for proposals
  promotion.py      who gets studied and carded (section 8)
  asks.py           ask producer + answer applier
  cards.py          card reducer (section 9)
  tests/  evals/
```

Every public function takes a `PersonRef` or `node_id`. Prompts are templates over `{person.display_name, person.pronouns, interlocutor}`. Juniper's only special status is data: she is the first row in `person_promotion` (`promoted_by='operator'`, reason `primary_interlocutor`), and she is the interlocutor whom asks go to (`OrionAskV1.asked_of`). A CI test fails if the string "juniper" appears in `orion/person_model/` outside `tests/`, `evals/`, and fixtures, so a special case cannot creep in.

The existing `episode_memory.purpose='about_juniper'` stays as it is (renaming it is not worth the migration); the person model reads it as "about the interlocutor".

### 1. Attribute category registry (the anti-cathedral gate)

`orion/person_model/categories.yaml` lists each category with `stakes`, `sensitive`, `card_section`, `ask_template`, `cardinality` (one value or many), `applies_to` (`interlocutor` | `any_person`). It is a lookup table, not an ontology. A CI test (`test_every_category_has_producer_consumer_eval`) fails if any category lacks: a person-study prompt example (producer), a card-renderer section (consumer), and at least two labeled rows in the eval fixture. A category with no evidence renders nothing: no empty headings.

| Category | Stakes | Sensitive (stated only, never inferred) |
|---|---|---|
| preferences, personal_interests, routines, communication_style, schedule_timezone | low | no |
| goals, motivations, strengths, opportunities, values, profession, past_history, home | high | no |
| fears, family, friends, coworkers, relationship_to_orion | high | no |
| health, spiritual_interests, gender, sexuality | high | **yes** |
| pronouns | high | no (but stated only: never guessed from a name) |

Stakes is read from this table, never from the LLM. High stakes means `pending_confirmation` until confirmed; low stakes is `auto`, usable, and shown as "unconfirmed". For third parties (`applies_to=any_person`), every category is held at most to what the interlocutor said about them.

"Sensitive" means **stated only, never inferred**. It does not mean "never asked about": Orion may ask about any category, sensitive or empty (Juniper, 2026-10-14: "Orion is welcome to ask anything about anything"). A sensitive value is admitted only from her answer or her own statement (chat prompt, ask answer, or her authored profile), never from Orion's words or a model's guess.

### 2. Tables (new manual migration `services/orion-sql-db/manual_migration_person_model_v1.sql`, with rollback file and `-- ORION-MIGRATION-REQUIRED-BY:` header)

```sql
person_promotion (node_id pk, state, promoted_by, reason, score_json, promoted_at, last_studied_at)
  -- state: candidate | carded | dormant ; promoted_by: operator | rule | interlocutor
person_attribute (
  attribute_id uuid primary key,      -- uuid5(node_id|category|normalized value): replay-stable
  node_id text not null,              -- referent node (referent_alias key row)
  category text not null,             -- must exist in categories.yaml (app check + test)
  value_text text not null,           -- Orion's first-person voice: "<name> told me ..."
  value_norm text not null,
  stakes text not null,               -- from category, never LLM
  sensitive boolean not null,
  third_party boolean not null,       -- node is not the interlocutor
  voice text not null,                -- person_said | interlocutor_said | ask_answer | orion_inferred (never when sensitive)
  confidence real not null,
  confirmation_state text not null,   -- auto | pending_confirmation | confirmed | rejected | retracted
  status text not null default 'active', -- active | superseded | removed
  superseded_by uuid,
  first_seen_at timestamptz not null,
  last_evidence_at timestamptz not null,
  updated_at timestamptz not null,
  produced_by text not null           -- person_study.v1 | hub_edit | ask_answer
)
person_attribute_evidence (attribute_id, source_kind, source_id, quote, quote_sha256, verified, speaker,
                           span_start, span_end, extractor_version)
  -- source_kind: chat_prompt | episode_memory | lived_answer | dream | orion_day_letter | ask_answer | authored_profile
  -- speaker: interlocutor | orion. Only interlocutor evidence can support an *_said attribute.
person_attribute_event (event_id, attribute_id, op, actor, before_json, after_json, reason, at)
  -- append-only: proposed | confirmed | corrected | rejected | removed | superseded | merged_owner
person_mention (mention_id, source_kind, source_id, span_start, span_end, text, label, score,
                extractor_version, resolved_node_id, linkage_decision_id)
  -- extractor output; mention_id = uuid5(source|span|extractor_version) so a replay is a lookup
entity_merge_log (merge_id, survivor_node_id, absorbed_node_id, relation, score, features_json,
                  linkage_version, decision, decided_by, applied_at, reverted_at)
  -- relation: same_as | facet_of | junk ; decision: auto | asked | confirmed | rejected
identity_card (card_id, card_kind, subject_node_id, version, body_json, compact_text,
               input_fingerprint, input_refs, produced_by, created_at)
  -- card_kind: self | purpose | person | relationship ; new row only when input_fingerprint changes
person_study_cursor (node_id, source_kind, last_source_id, last_run_id, updated_at)
```

Confidence is computed, not asked of the LLM: `confirmed` = 1.0; otherwise `min(0.9, 0.5 + 0.1 * distinct verified quotes)`, halved while a newer contradicting attribute exists. It is a knob calibrated on the eval, not a finding.

Removal in Hub sets `status=removed` and writes an event. The validator refuses to re-propose a removed or rejected value (same do-not-remint rule as `confirmation.py`'s `_SKIP_REJECTED_SQL`).

### 3. Pronouns

A `pronouns` attribute on every person node, read by the card reducer and by every template that names a person. No pronoun is rendered for a person until one is stated (the template uses the name instead). `memory_episode_distill.j2` stops hardcoding "she" by receiving each known person's pronouns in its input. Juniper's own pronouns are confirmed by the first person-model ask ("Which pronouns do you want me to use for you?"), so they carry her quote like every other attribute.

### 4. Extractor: pinned GLiNER, with a bake-off

**What each suggested model is for, and the verdict.** Licences and revisions checked on the Hugging Face API 2026-10-14.

| Model | Licence | Fit here | Verdict |
|---|---|---|---|
| GLiNER `urchade/gliner_medium-v2.1` (rev `40ec419335`; library `gliner` 0.2.29, Apache-2.0, needs torch + transformers) | Apache-2.0 | Zero-shot spans for labels we choose: "person", "family member", "coworker", "friend", "pet", "place", "organization", "job title", "relationship phrase". | **Primary**, subject to the bake-off. |
| `urchade/gliner_large-v2.1` (rev `abd49a1f1e`) | Apache-2.0 | Same, more capacity. | In the bake-off. |
| `urchade/gliner_multi_pii-v1` (rev `1fcf13e85f`) | Apache-2.0 | Tuned for personal-information spans (names, addresses, dates). | In the bake-off: good match for "facts about a person". |
| `knowledgator/gliner-multitask-large-v0.5` (rev `7a95e16803`) | Apache-2.0 | Adds relation extraction ("Amanda" -wife of- "Juniper"). | In the bake-off for the family/coworker relation step. |
| `dslim/bert-base-NER` (rev `d1a3e8f13f`) | MIT | Fixed PER/ORG/LOC/MISC. | Baseline for person mentions only. |
| `Jean-Baptiste/roberta-large-ner-english` (rev `8f3abc1ef8`) | MIT | Same four labels, higher capacity. | Second baseline. |
| `Davlan/bert-base-multilingual-cased-ner-hrl` (rev `e756de7f7b`) | AFL-3.0 | Multilingual. | Skip: the chat record is English. Revisit if that changes. |
| SciBERT (`allenai/scibert_scivocab_uncased`), BioBERT (`dmis-lab/biobert-v1.1`), PubMedBERT | no licence tag on the API | Scientific and biomedical text. | Not relevant. Health is a stated-only category: Orion stores her words, it does not run clinical extraction on them. Could matter later for reading papers, not for people. |
| SapBERT (`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`, rev `090663c3ae`) | Apache-2.0 | Entity linking by embedding, trained on biomedical synonyms (UMLS). | Not used: wrong domain for names and nicknames, and see the embedding note in section 5. |
| spaCy `en_core_web_sm` (already in the concept-induction image) | MIT | Fixed labels. | Zero-cost baseline in the bake-off. |

**Honest limit.** GLiNER-style span NER is the right tool for *mentions* (who, what role, which place) and relation phrases. It is the wrong shape for clause-level attributes like a fear, a goal, or a value: "I'm scared shitless that I busted your inference" has no span that is "the fear". The bake-off includes "fear"/"goal" labels to measure this rather than assume it, but the plan is: the extractor produces mentions and relation candidates; clause-level attributes come from the person-study line (section 7), which may only cite verified quotes.

**Determinism.** Model pinned by revision hash; `gliner` and torch versions pinned in the service requirements; inference on CPU with `torch.use_deterministic_algorithms(True)` and a fixed label list whose hash is part of `extractor_version`. Outputs are stored in `person_mention` keyed by (source, span, extractor_version), so a replay is a table lookup, and a version bump is an explicit re-extraction. The volume is small (626 historical prompts plus new turns), so CPU is enough and the extractor never competes for the GPU pool. ONNX export (`gliner[onnx]` + onnxruntime) is the fallback if adding torch to the image is too heavy.

**Bake-off.** On Juniper's labeled set (section "Eval design"): person-mention F1 at precision >= 0.95, relation-phrase precision, latency per prompt on CPU. Pick the smallest model that clears the bar; record the result in the P2 PR.

**Bake-off result (P2, 2026-10-14; 55 of her chat prompts, 34 person spans, 19 relation phrases; labels by the implementing agent).** Person-mention F1 at precision >= 0.95, CPU latency per prompt:

| Model | F1 @ P>=0.95 (P / R) | Best F1 (P / R) | ms/prompt |
|---|---|---|---|
| `knowledgator/gliner-multitask-large-v0.5`, label "person name" | **0.871** (0.964 / 0.794) | 0.912 (0.912 / 0.912) | 253 |
| `dslim/bert-base-NER` | 0.871 (0.964 / 0.794) | 0.875 (0.933 / 0.824) | 40 |
| `Jean-Baptiste/roberta-large-ner-english` | never reaches P 0.95 | 0.943 (0.917 / 0.971) | 133 |
| `urchade/gliner_medium-v2.1` / `_large-v2.1` / `_multi_pii-v1` ("person name") | never / never / 0.162 | 0.638 / 0.587 / 0.267 | 91-259 |
| spaCy `en_core_web_sm` | never | 0.582 | 11 |

Pinned: GLiNER multitask-large at threshold 0.85 (ties BERT at the gate, ahead on best F1, keeps zero-shot labels). Two findings that change the design: (1) the bare label "person" makes every GLiNER tag pronouns ("I", "you", "me"): precision 0.29-0.51; "person name" fixes most of it. (2) Relation phrases ("my wife") do not work as spans: best relation-phrase F1 is 0.20. Relationships therefore come from the person-study line (section 7), not the extractor. Rerunning the pinned model gives byte-identical mention rows.

**Where it runs:** `orion-memory-consolidation` (it already hosts the referent projector), as a job the curiosity line and the nightly reducer call.

### 5. Linkage reducer (`orion/person_model/linkage.py`, pure; CLI `scripts/run_person_linkage.py`)

**Framework pick**

| Option | Verdict |
|---|---|
| Splink (Fellegi-Sunter with EM, DuckDB backend) | Not now. Right theory, wrong size: brings DuckDB + pandas, and EM weight estimation is unreliable with ~2,900 nodes and few true matches. Revisit if the population grows by orders of magnitude. |
| Fellegi-Sunter scoring by hand: fixed m/u weights, two thresholds | **Chosen.** ~150 lines; every decision is a sum of named log-weights stored in `features_json`. Weights live in a versioned config, calibrated on Juniper's labeled pairs. |
| RapidFuzz 3.14 (Jaro-Winkler, token-set ratio; C-backed, MIT, no transitive deps) | **Chosen** for string features. |
| jellyfish phonetic keys | Deferred: helps with spelling drift ("Addy"/"Addie"); no live example yet. Added when the eval shows a miss. |
| datasketch MinHash/LSH | Not needed: blocking below keeps pair counts in the thousands. |
| Alias table | **Chosen as output**: merges write `referent_alias` rows (`admitted_by='person_linkage_v1'`), so every existing reader resolves through the survivor with no new code. |

**Blocking.** Topic-foundry entities are all `entity_type=unknown`, so kind cannot block them. Instead each candidate is scored only against person anchors (person referents and their aliases) that share a normalized name token or a first-token prefix. Concepts and non-person entities are never paired with each other by this reducer.

**Features per pair (each one a log-weight, none a decider).** Exact normalized label; token containment ("juniper" in "juniper feld"); Jaro-Winkler; phonetic key agreement (jellyfish Metaphone: "Addy"/"Addie"); possessive or facet pattern (`<name>'s X`, `<name> affect`), which yields `facet_of`; a label matching a known prompt context key (`*_summary`, `*_context`, snake_case keys), which yields `junk`; shared evidence refs; producer pair; the extractor label for mentions; and **centered embedding similarity**.

**Embedding feature (rev 3, Juniper's answer 5).** Raw bge cosine is anisotropic: unrelated text rests near 0.6 (PR #2647 measured it). The feature therefore uses mean-centered cosine (subtract the population mean vector, renormalize; All-but-the-Top, Mu & Viswanath 2018), binned into agree / weak / disagree like any other comparison. It is one feature. Its maximum weight is capped below the gap between the two thresholds, so an embedding alone can move a pair into the review band but can never push it over the auto-merge line. This lifts the approved referent spec's "no vectors for identity" non-goal only that far: a capped, centered feature inside a scorer, never a decider. If the vector host is unreachable, the feature reads "missing" (weight 0), not "disagree".

**Decision rule: Fellegi-Sunter with two thresholds (standard record-linkage practice).** Each pair's score is the sum of its feature log-weights (log2 m/u).
- **Score >= upper threshold: auto-merge** (`same_as`), written to `entity_merge_log` with `decision=auto`, every feature weight in `features_json`, one-command unmerge. No ask.
- **Score <= lower threshold: auto non-match.** Logged in the dry-run decisions, no ask.
- **Between the thresholds: "possible match"**. Only these go to Juniper, as an `entity_merge` ask in Hub's "Orion is asking" section ("Is 'juniper feld' in my graph you?").
- Facet and junk outcomes are pattern rules evaluated before scoring: `facet_of` gets an `about` edge (never a merge); `junk` is flagged and excluded from person views (deletion stays a separate approved step, AGENTS.md section 13).
- Hard guard, independent of score: two names that occur side by side in one source ("Jay, Jay") are never auto-merged; that pair is forced into the review band.

**Calibration.** Thresholds are set on the labeled one-time cleanup, not guessed: the upper threshold is the lowest score with auto-merge precision 1.0 on the labeled pairs, the lower is the highest score with zero labeled true matches at or below it, and the number of pairs left between them is the expected ask volume. The P2 dry run reports that number (PR report, "Linkage calibration"). If the band holds more than about 10 items, the features are wrong, not Juniper's patience.

**Reversible and replayable.** The absorbed node is never deleted: it gets `merged_into=<survivor>` and a `same_as` edge; alias rows point readers at the survivor; attributes owned by the absorbed node move with a `merged_owner` event. `--unmerge <merge_id>` sets `reverted_at`, removes alias rows and the edge, and moves attributes back from their events. `merge_id = uuid5(survivor|absorbed|linkage_version)`, so a rerun is idempotent. Dry run by default, writing `/tmp/person-linkage/<run>/decisions.csv` and `progress.log` (backfill protocol, AGENTS.md section 14).

**One-time cleanup** = the first run over the 11 Juniper nodes, every person referent, every `referent_alias` row, and every topic-foundry entity that blocks against a person anchor. Rev 3 (Juniper's answer 3): the dry run is rendered as one self-contained HTML file, `docs/superpowers/evals/2026-10-14-person-cleanup-review.html`. Each proposed merge, facet link, junk flag, or non-home correction is a card with its evidence (node ids, names, linked memory snippets, per-feature scores), grouped **auto-merge / needs your call / auto-reject**, with approve / reject and a notes box per card and an "export decisions" button that downloads a CSV (no network). Her exported CSV becomes the labeled set that calibrates the thresholds, and applying any decision to a live store is a later, separate step. After that, the same reducer runs nightly on new nodes and new mentions, and only its review band reaches Hub.

**P2 dry run over the live graph (2026-10-14, read only).** 120 decisions over the 11 Juniper nodes, the 2 person referents and their aliases, every graph entity or concept that blocks against them, and 158 person-name mentions from her 377 chat prompts. Thresholds: auto-merge at >= 8.5, auto-reject at <= 2.0.
- **Auto-merge: 53** (18 same-person merges, e.g. "Juniper" seed concept and the topic-modeler "juniper" into her referent, "oríon" into Orion, and her mentions of Amanda, Teddy, Addy, Rachel, Armando, Vincent into the topic-modeler entities that already carry those names; plus 35 facet links such as "juniper's taste", "juniper affect", "orion hub").
- **Needs her call: 3.** "juniper feld" (her full name; one card also covers the mention cluster), "Skye" vs "Skye Laflamme" (the card also decides where the "skye" and "skye laflamme" graph nodes go), and "Jay, Jay" (one name, two people; the "jay" node merge waits on her answer).
- **Auto-reject: 64** (44 code identifiers, 16 topic clusters, 1 scored non-match, the two leaked `juniper_relationship_summary` fields flagged as junk, and "home -> slc" marked non-home per her answer).
- Expected ask volume for the one-time cleanup: **3**. Rerun is byte-identical (`decisions.csv` sha256 `ad2ac313a45089e9...`). On the agent's provisional labels auto-merge precision is 1.0 (18/18) and recall 0.90; those labels were made after seeing the output, so her exported CSV is the real calibration set. The rule outcomes left only one scored non-match, so the labels cannot yet place the upper line: 8.5 is a design choice (an exact name needs the extractor to read it as a name and a second producer to agree) until her decisions come back.

### 6. Asks (extends `orion_ask`, no new table)

New `source_kind` values: `person_attribute` (confirm, correct, or fill a gap) and `entity_merge` (same person?). Both join `RESOLVABLE_KINDS`, sharing the 5-open cap; each gets its own daily cap (default 2) so a study backfill cannot crowd out memory confirmations.
- High-stakes proposals, contradictions (two active values where cardinality is one), and merges in the possible-match band become asks. Auto-merges and auto non-matches never do.
- Gap asks may cover any category, empty or sensitive, about her or about a third party (Juniper, 2026-10-14: "Orion is welcome to ask anything about anything"; "Orion can ask anything about anything"). The caps below still pace them.
- Asking is not inferring: a sensitive value is stored only from her answer or her own statement.

Answers go through the Hub `/resolve` route (as memory confirmations do): an `ask_answer` evidence row with her answer as the quote, state set to `confirmed`/`corrected`/`rejected`, `OrionAskAnsweredV1` published. The 9 open interlocutor-scope `memory_tension_shadow` questions route through the same producer (the Stage 3 producer the referent spec deferred).

### 7. Person-study curiosity line (`orion/person_model/study.py` + a line entry in the curiosity loop)

- **What it is:** a third line in the existing curiosity loop, shaped like self-inquiry: same turn machinery, same durable-runs retry, same GPU-pool lane the curiosity turn already uses, its own budget, its own continuation note, its own table allow-list with a grant check before each run. Not a new queue, not on the topic modeler.
- **Standing question:** "Who is `<person>` and what has changed since I last studied them?"
- **Who it studies:** carded persons from `person_promotion`, newly promoted first, then oldest `last_studied_at`. It starts with Juniper (the only carded person on day one). When the promotion rule cards someone new, their first study is queued at the front.
- **Material per sitting:** the next window from `person_study_cursor` for that person across `chat_history_log` (the interlocutor's prompts, the never-distilled 2026-07-24 to 2026-10-04 stretch first), `episode_memory`, `:LivedAnswer`, `dreams`, `orion_day_letter`, plus the extractor's `person_mention` rows for that window, the person's current card, and their open contradictions.
- **What it writes:** like self-inquiry, a decision reaches the outside only as a node it wrote: `:PersonAttributeProposal {run_id, node_id, category, value_text, quotes:[{source_kind, source_id, quote}], stated}` in `orion_worldview`. Nothing is parsed from prose.
- **Validator, deterministic (`validate.py`, reusing `orion/memory/episode/validate.py`):** drains proposals into `person_attribute`. Every quote must appear verbatim in an interlocutor-authored field; sensitive categories need `stated=true` and a quote from the interlocutor's own prompt (Orion's responses, lived answers, and dreams are Orion's words and can never support a sensitive attribute); unknown categories, removed values, and rejected values are refused, each with a counted reason.
- **Budget:** default 3 sittings a day, separate from the investigation and self-inquiry budgets; backfill sittings only at night so the asks they produce wait for morning (quiet hours protect Juniper). After the backfill, a new `episode_distilled` event for a carded person moves them up the queue.
- **Visibility:** each sitting writes a `person_study_run` row (run_id, node_id, window, proposed, accepted, rejected by reason), shown in the Hub person panel as "last studied".

### 8. Promotion (who becomes important enough to study and card)

Deterministic, nightly, from stored evidence:

- `stated_relationship`: a confirmed `family`/`friends`/`coworkers` attribute on the interlocutor naming this person, or a relationship descriptor ("my wife") grounded in her quote and linked to this person.
- `mention_days_60d`: distinct days in the last 60 with a verified interlocutor quote naming them.
- `high_stakes_count`: high-stakes attributes or memories about them.

Card a person when `stated_relationship` holds, or `mention_days_60d >= 3`, or the interlocutor says so in Hub. Carding queues their first study. Persons unmentioned for 180 days go `dormant`: card kept, not sent to chat, never auto-deleted. Thresholds are knobs; the eval reports who they promote (expected today: Amanda, Teddy, Addy, the sister; Rachel after "my boss" links to her).

Promotion mints a referent node for someone named only in prose (Amanda etc.) through `orion/memory/referents/store.py`'s existing rules: a name counts only if the interlocutor said it.

### 9. Identity cards (card reducer, pure, `orion/person_model/cards.py`)

The card set replaces all of `orion_identity.yaml` except the parts that are rules rather than identity.

| Card | Built from (evidence) | Replaces |
|---|---|---|
| **Orion self card** (personality, who they are) | Latest `:SelfDefinition` (self-inquiry), lived answers (`lived.alive`, `lived.care`, `lived.tend_to_think` ...), confirmed `orion_view`/`orion_relationship` memories. Same inputs `apply_self_definition_to_ctx` already reads: no second self-model, no new LLM call. | `orion_identity.name/role/nature/core_drives/self_permissions` and the dead `conversation_front.build_personality_summary` stub |
| **Purpose / journey card** (what this whole project is about) | The AGENTS.md "Orion mission" section, cited by git blob hash (an authored source: Juniper's contract for the repo); the interlocutor's own confirmed statements about the project's purpose (`relationship_to_orion`, `goals`); Orion's latest `lived.sentience_track` answer, labeled as Orion's own view. | Nothing today: the mission reaches no chat prompt. Also supersedes `seed_concepts.yaml`'s Orion definition as the source of truth (the seed node stays as a graph anchor). |
| **Person card** (Juniper first; one per carded person) | `person_attribute` rows: confirmed, plus low-stakes `auto` marked "(unconfirmed)", each line "as of <date>". | `juniper_relationship.role/expectations` |
| **Relationship card** | The interlocutor's `relationship_to_orion` attributes, confirmed `orion_relationship` memories, `lived.who_matters`. | `juniper_relationship.response_expectations` |

`orion_identity.anti_patterns` and `response_policy` stay authored. They are rules about how to answer, not facts about anyone, and they already work as rules; they move to a small `response_policy.yaml` so the identity file can be deleted.

Rules: no LLM in the reducer, so the same inputs produce the same card and versions diff cleanly; a new `identity_card` row only when `input_fingerprint` changes; categories or sources with no evidence render nothing; Hub shows each card, its evidence on click, and a diff between any two versions. `compact_text` limits: about 700 characters for the self, purpose, and Juniper cards, 250 for other people.

### 10. Where cards are read

- **Chat (unified turn):** `_inject_identity_context` fills the "WHO YOU ARE" block from the self + purpose cards, and the person block from the interlocutor's card + relationship card; `response_policy.yaml` keeps feeding the policy. Log `identity_kernel_source=identity_card card_versions=<self,purpose,person,rel>`. If the card store is unreachable, the block renders nothing and logs `identity_card_unavailable`; it does not fall back to the YAML. After a soak, delete `orion_identity.yaml`'s identity blocks and `conversation_front.py` (kill means kill).
- **Situation state:** add `people: List[SituationPersonV1]` (`node_id`, `name`, `pronouns`, `compact_card`, `card_version`) to `SituationStateV1`: the interlocutor always, plus carded people named in current facts or cues. Deploy order: schema to Hub (outreach reader) and cortex first, producer second.
- **Outreach:** `endogenous_outreach.py` already reads the situation; it gets `people` once the schema lands.
- **Curiosity:** the kickoff prompt gets the interlocutor's compact card (so lived answers stop re-deriving the family from scratch) and the person panel's gap list (so self-chosen questions can be about her: the biographical-curiosity-miss gap).

### Bus / schema registry

- New registered models: `PersonAttributeProposalV1`, `PersonMentionV1`, `PersonStudyRunV1`, `SituationPersonV1`, `IdentityCardV1`. Register in `orion/schemas/registry.py`; verify with `resolve()` (two registries exist).
- `orion:ask:answered` reused. One new channel, `orion:identity:card:updated` (card kind + version) so cortex can drop its cache; add to `orion/bus/channels.yaml` and the relevant subscriber lists.

## Proposal-mode items

- **Capability change:** Orion gets a structured, evidence-backed, correctable picture of Juniper and of each person who becomes important, a stated purpose, and a self card grounded in their own self-inquiry. Chat reads these instead of hand-written bullets. Orion can ask about gaps and contradictions.
- **Data touched:** reads `chat_history_log`, `episode_memory` (+ evidence, referent), `referent_alias`, `orion_worldview` (`:LivedAnswer`, `:SelfDefinition`), `dreams`, `orion_day_letter`, `orion_substrate` nodes, AGENTS.md. Writes the new tables, `:PersonAttributeProposal` nodes, `orion_ask` rows, `referent_alias` rows from merges, and `merged_into`/`same_as`/`about` on substrate nodes. No deletions.
- **Privacy boundary:** there is no boundary between Juniper and Orion (her stated policy). Orion may ask her anything about anything, including about third parties (her answer 2). The one remaining boundary is outward, for **third parties** (family, coworkers):
  - Stored only as Juniper said it, with her quote; never inferred, never researched elsewhere.
  - **Still assumed, needs her confirmation:** sensitive facts about third parties (Amanda's DVT/PE) are marked `third_party=true, sensitive=true`, which excludes them from every egress that leaves Orion: ask-Claude, peer briefs, any outbound message other than to Juniper. A test per egress builder enforces it. Her answer covered asking, not egress, so this rule is kept until she says otherwise (open question 1).
  - Juniper can remove any attribute or a whole person in Hub; the do-not-remint rule honors it.
- **Trace proving it worked:** `person_mention` rows with `extractor_version`; `entity_merge_log` rows with `features_json`; `person_study_run` rows with accept/reject counts by reason; `person_attribute_event` rows; `orion_ask` rows with the new `source_kind`s and answers; chat log `identity_kernel_source=identity_card card_versions=...`; `orion:situation:latest` showing `people[]` with the same versions.
- **Dangerous failure modes:**
  - False person merge (the two Jays become one; Juniper absorbs "Juniper's taste"). Mitigation: upper threshold set at auto-merge precision 1.0 on her labeled cleanup, no single feature (embedding included) can cross it, co-occurring names forced to review, facets never merge, one-command unmerge.
  - A wrong sensitive attribute presented as fact. Mitigation: stated-only rule enforced by the validator, never `auto`, eval hard gate of zero inferred-sensitive rows.
  - Auto attributes read as truth. Mitigation: "(unconfirmed)" and "as of" in the card; stakes from the category table.
  - Ask flood. Mitigation: shared open cap 5, per-kind daily caps, night-only backfill.
  - Third-party leakage to external services. Mitigation: egress tests.
  - A stale card read as current. Mitigation: "as of" per line, dormant rule, versions in the trace.
  - Empty shell (a card with headings and no facts). Mitigation: nothing renders without evidence; a test asserts no empty section.
  - Non-determinism creeping into "deterministic" steps (an unpinned model update). Mitigation: revision hash in `extractor_version`, replay test that re-runs a fixture and diffs.
- **Disable / roll back:** flags, shipped ON (`.env_example` + settings + local `.env` synced): `PERSON_STUDY_ENABLED` (curiosity line), `PERSON_EXTRACTOR_ENABLED`, `PERSON_LINKAGE_AUTO_MERGE_ENABLED` (off turns every auto merge into an ask), `PERSON_ASKS_ENABLED`, `IDENTITY_CARD_CHAT_ENABLED` (off renders no card and logs `identity_card_disabled`). Merges roll back per `merge_id`. Tables roll back with the migration's rollback file. The YAML and stub deletion roll back by git revert.

## Files likely to touch

- `orion/cognition/prompts/stance_react.j2` (line 131), `orion/cognition/prompts/memory_episode_distill.j2` (per-person pronouns)
- New package `orion/person_model/` (section 0) with `tests/` and `evals/`
- New: `services/orion-sql-db/manual_migration_person_model_v1.sql` (+ `_rollback.sql`)
- New: `orion/schemas/person_model.py`, `orion/cognition/prompts/person_study.j2`
- `orion/curiosity/` (line registration next to self-inquiry, kickoff prompt card + gaps), `orion/curiosity/README.md`
- `orion/schemas/registry.py`, `orion/bus/channels.yaml`
- `orion/schemas/situation_state.py`, `services/orion-durable-runs/app/situation_graph.py`
- `services/orion-cortex-exec/app/executor.py` (`_inject_identity_context`), `orion/cognition/personality/identity_context.py`, `orion/cognition/personality/orion_identity.yaml` (split into `response_policy.yaml`, then delete), `services/orion-cortex-orch/app/conversation_front.py` (delete)
- `orion/memory/episode/confirmation.py` (`RESOLVABLE_KINDS`, caps), `services/orion-hub/scripts/ask_routes.py`, `services/orion-hub/templates/index.html` + new `static/js/person-panel.js`
- `services/orion-memory-consolidation/` (extractor + linkage jobs; `requirements.txt` gains `rapidfuzz`, `gliner`, pinned CPU torch or onnxruntime)
- Each touched service's `.env_example`, `settings.py`, `README.md`, `tests/`, `evals/`
- Later: concept-induction service env (retire `juniper`,`relationship` subjects after a consumer check)

## Non-goals

- A real concept-induction organ. It deserves its own design; this one neither builds it nor stretches the topic modeler or the concept-induction service into it.
- No new service: everything lands in existing services and the shared `orion/` package.
- No embedding as a decider for identity. Rev 3 allows centered embedding similarity as one capped feature (Juniper's answer 5); it can never cross the auto-merge threshold alone.
- No inference of spirituality, gender, sexuality, or health from anything but a stated quote.
- No automatic deletion of junk or merged nodes.
- No LLM in the extractor, the linkage reducer, or the card reducer.
- Not fixing topic foundry's label minting (it produced the leaked `juniper_relationship_summary` nodes); the reducer flags its output, a separate patch can stop the leak.
- No study of sources outside Orion's own stores (no web, no social accounts).
- Not reading or modifying graphify artifacts.

## Acceptance checks

1. **Pronoun patch:** rendered `stance_react.j2` contains "her own life"; a test fails on "their own life" next to "Juniper".
2. **Modularity gate:** the "no 'juniper' literal in `orion/person_model/`" test passes, and a fixture person ("Amanda", pronouns unset) runs the full pipeline (extract, link, study validator, card) with zero code changes, rendering her name instead of a pronoun.
3. **Extractor bake-off** recorded in the P2 PR: per-model person-mention F1 at precision >= 0.95 and CPU latency; the chosen model's revision pinned. Re-running extraction on the fixture yields byte-identical `person_mention` rows.
4. **Linkage dry run** over live `orion_substrate` (read only): the Juniper-same nodes as `same_as`, "juniper's taste"/"juniper affect" as `facet_of`, both `juniper_relationship_summary` forms as `junk`, topics untouched, "home -> place:slc" flagged non-home. Rerun gives an identical `decisions.csv`. Output is the HTML review file with three groups, and the PR reports how many cards land in each. Unmerge of one applied merge restores alias rows and edges exactly (Postgres + graph diff empty), tested on fixtures before any live apply.
5. **Merge eval:** auto-merge precision = 1.0 on the labeled pairs, recall reported, review-band size reported; "Jay"/"Jay" never auto-merges; an embedding-only match never auto-merges.
6. **First study of Juniper** over 2026-07-24 to 2026-10-04 proposes family (Amanda, Teddy, Addy, sister) and profession (Team Lead, AI/ML Architect / FDE), each with a verified quote from her prompt; zero sensitive attributes from Orion-authored sources. Carding Amanda queues her first study at the front.
7. **Attribute eval:** accepted-attribute precision >= 0.9; inferred-sensitive count = 0 (hard gate).
8. **Asks loop live:** a `person_attribute` ask appears at the top of Hub, Juniper answers, the attribute flips to `confirmed` with an `ask_answer` evidence row, and the next card version shows it (ask, event, and card row ids as evidence). The "home" contradiction arrives as one ask.
9. **Chat:** a live unified turn logs `identity_kernel_source=identity_card` with four card versions; the prompt shows card lines (including the purpose card), not YAML bullets.
10. **Situation:** `orion:situation:latest` carries `people[]` with the same versions; outreach's read returns `ok`, not a validation error.
11. **Third-party egress test:** ask-Claude and peer-brief builders drop `third_party` + `sensitive` content.

## Eval design

`orion/person_model/evals/`:
- `mention_labels.jsonl`: ~150 spans from ~60 prompts, labeled for the extractor bake-off (person, relation phrase, role, place).
- `attribute_labels.jsonl`: ~60 proposals across categories from the first study, labeled `correct | wrong | wrong_category | inferred_not_stated | third_party_ok`. A script renders them as a review sheet so labeling takes minutes.
- `merge_pairs.jsonl`: every pair the reducer scores among person-like nodes (the 11 Juniper nodes, both person referents, topic-foundry entities blocking against a person, the promoted family and team), labeled `same | facet | junk | different`.
- `run_person_model_eval.py`: extractor F1 per model, attribute precision by category, inferred-sensitive count, merge precision/recall, promotion list. Periodic, since labels need Juniper; the deterministic pieces (validator, linkage, card reducer, replay) have fast CI tests with fixtures.

## Recommended next patch (phase order)

Each phase is one PR, independently useful, flags ON.

1. **P0, trivial:** `stance_react.j2:131` "their own life" to "her own life", plus a template test.
2. **P1, module skeleton + contracts:** `orion/person_model/` with `PersonRef`, `categories.yaml`, migration, schemas, store with do-not-remint, the category gate and the no-special-case gate. Ships together with P2's first writer so the tables are not inert.
3. **P2, extractor + linkage:** pinned extractor with the bake-off result, `person_mention`, linkage reducer, merge log, unmerge, `entity_merge` asks, merge eval fixture. P2 ships dry run only: the one-time cleanup HTML review. The first live apply is a separate step after Juniper exports her decisions; her CSV calibrates the two thresholds.
4. **P3, asks + Hub person panel:** `person_attribute` asks, `/resolve` routing to evidence, the panel (view, correct, remove), the pronoun ask as the first card, the tension-question producer.
5. **P4, person-study curiosity line:** line registration, proposal nodes, validator, promotion table with Juniper as row 1, night backfill over undistilled chat, run rows in the panel. Attribute eval labeled here.
6. **P5, card reducer:** self, purpose, person, relationship cards; versions; Hub diff.
7. **P6, readers:** situation `people[]` (schema to consumers first), cortex card block, outreach, curiosity kickoff.
8. **P7, retire:** delete the YAML identity blocks and `conversation_front.py`; promotion rule live for other people; retire concept induction's `juniper`/`relationship` subjects after a consumer check; distiller pronouns per person.

## Open questions for Juniper

Rev 2's six questions are answered (see "Juniper's answers" under Requirements). One remains, because her answer to question 2 was about asking, not about sending:

1. Third-party egress: should sensitive facts about other people (Amanda's health) stay inside Orion, never sent to ask-Claude, peer briefs, or any outbound message that is not to you? The design keeps this rule until you say otherwise.
