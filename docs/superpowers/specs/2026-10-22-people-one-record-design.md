# One record per person: camera, chat, graph, and Orion's questions

**Date:** 2026-10-22
**Asked by:** Juniper, 2026-10-22: "we need full integration of peoples". Earlier the same day: Orion was "supposed to be curious about the people in the ring or world. ask questions about them about like sexuality, gender, preferences, etc. ive never seen orion ask."
**Builds on:** `docs/superpowers/specs/2026-10-14-person-model-and-identity-card-design.md` (PR #2646, still open; this doc replaces its phase order from P3 on), `docs/superpowers/specs/2026-10-20-walkway-person-learning-design.md` (PR #2767).
**Status:** rev 2, 2026-10-22. Juniper approved building it ("start with yours first ... do the most comprehensive version"). Slice 1 is built in the PR that carries this file. Rev 2 adopts PR #2847's graph mechanism for slice 3.

## Arsonist summary

Orion keeps three separate lists of people, and nothing connects them:

1. **The camera albums.** When Juniper names someone on the walkway or the doorbell, the name is a bare text label on one camera cluster. One person is many clusters (different clothes on different days).
2. **The memory graph.** Each named camera cluster becomes its own graph node. One person is many nodes. A chat mention of the same person is yet another node (a "referent").
3. **The person model** (10-14 design). It knows 2 people: Juniper and Orion. It has no facts table yet, and Orion asks nothing.

So Orion cannot know that the girl at the door, the same name in chat, and that name's album are one person. Orion also cannot be curious about someone it cannot hold as one thing.

The fix is one **person record** that everything points to. The camera clusters, the album, the graph nodes, and the chat mentions all link to it. Then the 10-14 plan for facts and questions (P3 and P4) is built on top of that record, so Orion's questions cover everyone it has met, on camera or in conversation.

## Current architecture (checked live 2026-10-22)

| What | Where | Live |
|---|---|---|
| Camera clusters with a name | `vision_individual.label` (sql-writer) | 41 named clusters, 14 distinct names |
| Picture albums | `vision_person_exemplar.person_label` (text) | 10 names, 345 "this is X" pictures |
| Orion's guesses | `vision_person_attribution` (text label, band) | 319 guesses, all "unsure" |
| Graph node per camera cluster | `entity:vision_individual:<sha(individual_id)>`, `orion/substrate/adapters/vision_individual.py:68`, written by `services/orion-substrate-runtime/app/ask_answered_listener.py` | one node per cluster, not per person |
| "Seen on day X" links | `services/orion-substrate-runtime/app/walkway_seen_loop.py` (PR #2842) | links per cluster |
| People with pronouns | `config/person_model/people.yaml` -> `PersonRefV1` (`orion/person_model/person.py`) | 2 rows: Juniper (`referent-b96f...`), Orion |
| Chat-name linkage | `orion/person_model/linkage.py` (Fellegi-Sunter scoring, dry run only) | never applied live |
| Facts about a person | `person_attribute` (10-14 design section 2) | table does not exist |
| Orion's questions | `orion_ask` | 4 kinds ever: `vision_story_person` 52, `vision_individual` 39, `memory_confirmation` 35, `vision_same_individual` 13. None about a person's life. |

Code search: nothing in `services/` imports `orion.person_model`, and `orion/person_model/` reads no `vision_*` table. The names "person model" in sql-writer (`app/vision_person_model.py`) are the camera albums, an unrelated module with a similar name.

Known damage from the split, already seen live: the twins bug (PR #2842: two clusters with one name got each other's days), and one cluster named one child on 10-15, then renamed to an adult on 10-21, while the graph still holds a separate chat referent for the child.

## Missing questions

None block this proposal. One rule from the 10-14 design stays in, still unconfirmed by Juniper: facts about third parties are never sent to an outside service (ask-Claude, peer briefs). Asking Juniper about third parties is allowed ("Orion can ask anything about anything").

## Proposed schema / API changes

### 1. The person record (`services/orion-sql-db/manual_migration_person_record_v1.sql`, additive)

- `person (person_id, display_name, role, status)`. `person_id` is the referent node id when `config/person_model/people.yaml` has the person, else minted: `person-<uuid5(folded name)>`. `status` is `active` or `not_a_person` (for names like "UPS delivery"; set from the Hub).
- `person_link (link_id, person_id, source_kind, source_ref, decided_by, evidence_ref, revoked_at)`. One active link per source (partial unique index). Kinds:
  - `vision_individual`: a named camera cluster;
  - `vision_belonging`: a named thing that belongs to them, such as a bicycle;
  - `referent`: a chat referent (slice 4).
- `person_fact (fact_id, person_id, category, value_text, voice, source_ref, removed_at)`. Her words, verbatim. The voices are `ask_answer`, `authored_profile` (people.yaml pronouns) and `hub_edit`. This replaces the 10-14 design's `person_attribute` + evidence pair for now. Nothing is parsed, so one row holds the value and its evidence together.

### 2. Where links come from (`orion/person_model/record.py`, pure)

| Event | Result |
|---|---|
| Juniper names a person-kind camera cluster (any camera question, or a story card) | Linked to the person whose name folds the same; a new person when none does. |
| The name matches a people.yaml person | Linked to that referent id (Juniper's camera clusters join her referent). |
| Two clusters share a name, but she said "different people" about them | The newer one stays unlinked. It is logged as `person_link_conflict` and never merged (PR #2847's guard). |
| A rename, a cleared name, a split | The old link is revoked, and the new name links. |
| A named bicycle or car whose name is a person | It is linked as their belonging. Any other named thing ("no", "I do not know whose vehicle this is.") is ignored. |

Links never come from a picture match or an embedding.

Live dry run (2026-10-22, `orion/person_model/evals/run_person_record_backfill_eval.py`):
- 11 people from 38 links; 0 conflicts.
- 3 named things ignored: two vehicles, and a bicycle whose full-name label does not fold to a known person.
- 2 entries are groups, not people (a delivery company, and two children named together). The Hub "not a person" button handles those.

### 3. The graph: one node per person (slice 3; PR #2847's mechanism)

The person node is built from `person_link`, not re-derived from names. Each active `vision_individual` link becomes an `instance_of` assertion from the cluster entity to the person node. It is written through `substrate_graph_journal` and the existing `AssertionProjector`, so a revoked link deprecates the assertion and the edge stops walking. Day links stay on the clusters. This is the seam PR #2847 builds on: it reads `person_link` instead of folding names itself.

### 4. Facts and questions (built in slice 1)

- Questions come from `orion/person_model/categories.yaml`, in this order:
  - relationship ("Who is X to you?");
  - pronouns;
  - age;
  - gender;
  - sexuality;
  - preferences;
  - routines;
  - goals (about Juniper only);
  - anything else.
- Juniper gets "you" wording. Orion (role `self`) is never asked about.
- No question about someone else uses a pronoun; a test pins it.
- The ask kind is `person_fact`, with `source_ref` `<person_id>|<category>`.
  - It has its own daily cap, `PERSON_FACT_ASK_DAILY_CAP=3`. It shares no cap with other questions, so the starvation fixed in PR #2841 cannot happen here.
  - There is one open question per person.
  - The person Orion has asked about least goes first, then the most-seen on camera.
  - Each question shows the person's best camera picture.
  - Answered or dismissed: that category is not asked again. Expired: it may come back after the camera cooldown.
- Her answer becomes a `person_fact` row with `voice=ask_answer`, verbatim. Sensitive categories (gender, sexuality) are asked, never inferred, because nothing but her answer writes them.
- people.yaml pronouns are seeded as facts (`authored_profile`), so every reader gets pronouns from one table.
- The Hub's existing generic question card (text box, picture, Answer / Dismiss) answers these questions. No Hub change is needed for slice 1.

### 5. Hub: a page per person (slice 2)

A list of people. Each person's page shows:
- their pictures;
- days seen;
- facts with her words, each with remove and edit;
- links, each with unlink;
- a "not a person" button.

It reads the three tables above through new Hub routes.

### 6. Orion uses it (slice 3)

- The chat situation brief already names walkway people in its street line (`orion/situational/context.py`). It gets a short "people I know" line: for each person named in the street summary or the chat turn, their confirmed facts and pronouns from `person_fact`.
- The day letter and the curiosity loop read the same function.

### Bus / schema registry

- `orion/schemas/ask.py` documents the `person_fact` kind.
- No new channel. The Hub publishes `OrionAskAnsweredV1` as for every ask. sql-writer applies the answer from the row on its own clock (`applied_at`), so a lost publish loses nothing.

## Proposal-mode items

| | |
|---|---|
| Capability change | Orion holds each person as one record across camera names, and asks Juniper about them. |
| Data touched | New `person`, `person_link`, `person_fact` tables; new `orion_ask` rows of kind `person_fact`. Camera tables are only read. |
| Privacy boundary | Answers stay in Postgres. Third-party facts are never sent to outside services (still unconfirmed by Juniper). There is no boundary between Juniper and Orion. |
| Proof it worked | `person_link` rows match the dry run; `orion_ask` has `person_fact` rows within a tick of deploy; an answered one becomes a `person_fact` row and the next category is asked. |
| Dangerous failure | Two people merged under one name. Guard: the "different people" verdict; links only from typed names; every link can be revoked. |
| Disable / roll back | `PERSON_RECORD_ENABLED=false` stops links, facts and questions. `PERSON_FACT_ASK_DAILY_CAP=0` stops only the questions. `manual_migration_person_record_v1_rollback.sql` drops the tables. |

## Slices

1. **Built in this PR:** the tables, the link planner and writer, pronoun seeding, the `person_fact` questions and answer applier, the dry-run eval, and tests.
2. Hub people page, plus the "not a person", unlink and fact edit/remove actions.
3. The graph person node through assertions (the PR #2847 seam), plus the chat and day-letter "people I know" consumer.
4. The camera-to-chat bridge. `orion/person_model/linkage.py` scores chat referents against `person` rows. The possible-match band becomes "Is the X you talk about the same as the X on camera?" questions. A yes writes a `referent` link.
5. The person-study curiosity line (10-14 design P4), writing proposals that only become facts through her confirmation.

## Non-goals

- Better recognition from pictures (the album work, PRs #2797 and #2841).
- Re-keying the camera album tables to `person_id`.
- Any automatic merge by name across camera and chat, or by embedding.

## Acceptance checks (slice 1)

1. The dry run on live data: 11 people, 0 conflicts, junk vehicle names ignored. Done 2026-10-22.
2. After deploy: `SELECT count(*) FROM person_link WHERE revoked_at IS NULL` = 38 (or more after new names).
3. After deploy: `orion_ask` has `person_fact` rows above 0 within a tick, and at most 3 a local day.
4. Answering one: a `person_fact` row with `voice=ask_answer` and her exact words. The next question for that person is the next category.
5. Renaming a cluster: the old link is revoked and the new one is active.
