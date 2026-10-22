# One record per person: camera, chat, graph, and Orion's questions

**Date:** 2026-10-22
**Asked by:** Juniper, 2026-10-22: "we need full integration of peoples". Earlier the same day: Orion was "supposed to be curious about the people in the ring or world. ask questions about them about like sexuality, gender, preferences, etc. ive never seen orion ask."
**Builds on:** `docs/superpowers/specs/2026-10-14-person-model-and-identity-card-design.md` (PR #2646, still open; this doc replaces its phase order from P3 on), `docs/superpowers/specs/2026-10-20-walkway-person-learning-design.md` (PR #2767).
**Status:** proposal (AGENTS.md section 0A: this changes memory, identity and curiosity).

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

### 1. The person record (new migration `manual_migration_person_record_v1.sql`, additive only)

```sql
create table person (
  person_id     text primary key,        -- the referent node id when one exists, else minted
  display_name  text not null,
  status        text not null default 'active',   -- active | dormant | merged
  merged_into   text references person(person_id),
  created_at    timestamptz not null default now(),
  updated_at    timestamptz not null default now()
);

create table person_link (
  person_id    text not null references person(person_id),
  source_kind  text not null,   -- vision_individual | vision_label | referent | graph_entity
  source_ref   text not null,   -- individual_id | label (casefolded) | referent node id | graph node id
  decided_by   text not null,   -- answer | auto_exact | merge_ask | hub_edit
  evidence_ref text,            -- ask_id / merge-log id
  created_at   timestamptz not null default now(),
  revoked_at   timestamptz,
  primary key (source_kind, source_ref)   -- a source belongs to one person at a time
);
```

- `person_id` reuses an existing referent node id, so Juniper keeps `referent-b96f...` and the pronoun rows in `people.yaml` keep working. A camera-only person gets a minted referent id through `orion/memory/referents/store.py`'s existing minting rules. There is no second id scheme.
- Revoking a link (unlink in Hub, or a split) sets `revoked_at`. Nothing is deleted.

### 2. Where links come from (all deterministic)

| Event | Link written |
|---|---|
| Juniper names a camera cluster (any of the 3 camera question kinds, or a story card) | `vision_individual:<id>` and `vision_label:<name>` -> the person whose active `vision_label` link matches the name (casefolded). No match: a new person. |
| A typed name matches **two** people (two people with one name) | No link. A `person_same` question: "Is this the same [name] as <picture of the other one>?" |
| A cluster is renamed  | The old link is revoked, the new one written. |
| A cluster is split (`app/vision_split.py`) | The moved pictures' new cluster gets no link until it is named. |
| The chat-name linkage (`linkage.py`) puts a chat referent and a camera name in the "possible match" band | A `person_same` question, never an automatic merge. A name alone is never enough: a name in chat and the same name on camera are linked only by Juniper's answer, or by an exact referent node match. |

The album keeps its text `person_label` for this slice. The album code reads it through `person_link`, so a rename moves the album with it. Re-keying the album tables to `person_id` is a later slice. Non-goal here.

### 3. The graph: one node per person

- The person node is the referent node (`person_id`).
- Each camera cluster node gets an `appears_as` edge from its person node, written by the existing `ask_answered_listener.py` path. No new writer.
- `walkway_seen_loop.py` links "seen on day X" to the **person** node, so a person seen as 3 clusters on one day gets 1 link, not 3. This closes the twins class of bug for good, not per case.

### 4. Facts and questions (10-14 design P3, now for every person)

- `person_attribute` and `person_attribute_evidence`, as specified in the 10-14 design section 2, keyed by `person_id`.
- New ask kinds: `person_attribute` (fill a gap, confirm, correct) and `person_same`.
  - Each kind has its own daily cap (`PERSON_ATTRIBUTE_ASK_DAILY_CAP=3`, `PERSON_SAME_ASK_DAILY_CAP=2`). The caps are not shared, so the bug fixed in PR #2841 cannot happen again.
- **Who gets asked about:** the most-seen people first. Sort by distinct camera days, plus distinct chat-mention days, over 60 days.
  - The first question about a new person is the one that changes the most: "Who is <name> to you?"
  - Then empty categories one at a time, in `categories.yaml` order: pronouns, gender, sexuality, relationship to Juniper, preferences, routines, and so on.
- Every question carries the person's best picture when one exists, so Juniper knows who it is about.
- **Sensitive categories** (gender, sexuality, spirituality, health) are asked, never inferred. A value is stored only from Juniper's answer or her own words in chat. The validator refuses Orion's words and model guesses.
- Answers go through Hub's existing `/resolve` route into `person_attribute_evidence` with `voice=ask_answer`. Pronouns answered here update the same `PersonRefV1` that chat prompts read.

### 5. Hub: a page per person

One panel listing people. Each person's page shows:

- pictures from the album;
- days seen, by camera;
- facts, each with Juniper's quote;
- links (clusters, chat names), each with an unlink button;
- edit and remove for each fact.

It reads the tables above. It does not invent a new view store.

### 6. Orion uses it (the consumer, in this slice)

- The walkway rollup line names people through `person.display_name`. Today's line: "[person A] at 10:17, [person B] at 10…".
- The chat prompt's people block (10-14 design P6) gets confirmed facts for any person named in the turn.
- Without this consumer, the facts table would be write-only, which is the keyword-cathedral pattern AGENTS.md section 0A bans.

### Bus / schema registry

- `OrionAskAnsweredV1` already carries `source_kind`/`source_ref`. The two new kinds are added to `orion/schemas/ask.py`'s documented kinds and to the substrate-runtime listener.
- No new channel.

## Proposal-mode items

| | |
|---|---|
| Capability change | Orion holds each person as one thing across camera, chat and graph, and asks Juniper about them. |
| Data touched | New `person`, `person_link`, `person_attribute(_evidence)` tables. Graph edges `appears_as`. `orion_ask` rows. Album tables are only read. |
| Privacy boundary | Answers stay in Postgres and the graph. Third-party facts are never sent to outside services (still unconfirmed by Juniper). There is no boundary between Juniper and Orion. |
| Proof it worked | A named cluster gets a `person_link` row. `walkway_seen` writes to one node per person. A `person_attribute` question is asked and answered, and its answer appears in the Hub person page and in the next chat prompt that names that person. |
| Dangerous failure | A wrong automatic link: two people merged into one, and facts about one shown for the other. Mitigation: names never auto-link across sources. Only an exact existing link or Juniper's answer creates one. Every link can be revoked, and Hub unlink is one click. |
| Disable / roll back | `PERSON_RECORD_ENABLED=false` stops new links and questions; the cap settings at 0 stop only the questions. Every table is additive. The rollback migration drops them; nothing else depends on them until the slice ships. |

## Files likely to touch

- `services/orion-sql-db/manual_migration_person_record_v1.sql` (+ rollback)
- `orion/person_model/record.py` (new: resolve or create a person, write and revoke links; pure plus one store)
- `orion/person_model/asks.py` (new: the two question producers and the answer applier)
- `orion/person_model/categories.yaml` (exists in the 10-14 design as planned; created here)
- `services/orion-sql-writer/app/vision_individuals.py` (on a name answer: call `record.link_name`)
- `services/orion-substrate-runtime/app/ask_answered_listener.py`, `walkway_seen_loop.py` (person node, `appears_as`)
- `services/orion-hub/scripts/ask_routes.py`, `templates/index.html`, a new `static/js/people.js`
- `orion/schemas/ask.py`, `orion/schemas/person_model.py`
- the matching `.env_example`, `settings.py` and README files; tests; `orion/person_model/evals/`

## Non-goals

- Orion recognizing people better from pictures. That is the album learning work (PR #2797, #2841).
- Re-keying the album tables to `person_id`.
- The person-study curiosity line that reads chat history to propose facts (10-14 design P4). In this slice, facts come from questions only. P4 comes next, on the same tables.
- Identity cards (10-14 design P5).
- Automatic merges by name or by embedding.

## Acceptance checks

1. Backfill dry run: every named camera cluster maps to a person. The report lists who would be linked to whom, and every clash is a `person_same` question, not a merge. Expected live: 14 names -> 13 or 14 people, plus any clashes.
2. A new camera name answered in Hub -> a `person_link` row within 1 tick, and an `appears_as` edge in the graph.
3. `walkway_seen` on a day with one person seen as 2 clusters -> 1 edge.
4. `orion_ask` has `person_attribute` rows above 0 within a day of deploy. Each one has a picture when the person has an album.
5. Answering "she/her" for someone -> a `person_attribute` row with `voice=ask_answer`, and the chat prompt uses those pronouns for that person.
6. Validator eval: a sensitive value proposed from Orion's own words is refused (counted reason).
7. Unlink in Hub -> `revoked_at` set. The facts stay with the person. The cluster becomes unnamed again.

## Eval design

- `orion/person_model/evals/run_person_record_backfill_eval.py`: read-only over live data. Reports links, clashes, people with no picture, and the question queue order. Exit nonzero if any link was made without an exact match or an answer.
- `run_person_ask_queue_eval.py`: replays the question producers on a throwaway database copy, in the same way as `run_vision_confirm_asks_replay_eval.py`.

## Recommended next patch

One PR, in this order:

1. The migration and `record.py`.
2. The backfill dry run against live data, with its report for Juniper to read.
3. The live link writer on camera name answers, plus the graph `appears_as` edges and person-level `walkway_seen`.
4. `person_attribute` tables, the two question kinds with their own caps, and the `/resolve` routing.
5. The Hub people panel.
6. The rollup and chat consumers.

The first live link apply waits for Juniper to read the backfill report, as the 10-14 cleanup did.
