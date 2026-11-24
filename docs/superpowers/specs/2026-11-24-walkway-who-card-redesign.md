# Walkway "who is this?" card redesign (proposal)

Status: PROPOSAL. Juniper approves before anything here is built.
Date: 2026-11-24. Follows the bug fix "each person row shows their own pictures"
(branch `fix/walkway-story-card-per-person-pictures`, called PR 1 below).

## Arsonist summary

The walkway story card asks too much at once and asks it blind. One story can
list six people. Each row has a name box and three buttons. Before PR 1 only one
row had a picture, and nothing said which row it belonged to. The buttons under
the sentence ("Wrong name", "Wrong motion", "Wrong other") mix three questions
into one click, and there was no way to say "yes, that name is right".

The answers show it. Juniper pressed "Not sure" 412 times on story cards. Those
412 clicks wrote 748 "unsure" picture answers. She pressed "Someone else" 58
times (131 "different" answers) and typed a name 270 times. On story cards she
has never once been able to say "yes, that's them". "Different" answers are not
harmless: they split camera groups (60 splits so far, `vision_split.py`), so a
blind "different" moves pictures to the wrong place.

PR 1 fixes the worst part: every row now shows its own pictures, its own box
numbers and a letter, and a named row has "Yes, that's <name>". This proposal is
the next step: ask about one person at a time, grade the sentence in small yes/no
parts, and show two groups that are already the same person as one person.

The biggest surprise in the live data: the two "Addy" rows on Juniper's example
story are already linked as the same person. 304 of the 313 "same person" links
join two groups that carry the same typed name (mostly made by the same-name
link, `_link_same_name`, when she types a name she used before).
The card simply does not show the link, so it asks about Addy twice. Only 3 names
still have same-name groups that are not linked (Bethany 3 groups, Juniper on the
doorbell 2, Maeve 2). So "merge" is mostly a display problem, not a data problem.

## Current architecture

- **Card:** `services/orion-hub/static/js/walkway-stories.js`, under the live
  picture in the Vision panel ("Walkway (live)").
- **Routes:** `services/orion-hub/scripts/motion_story_routes.py`.
  - `GET /api/vision/motion-stories` lists stories, the gate tally, and per story
    `people` (one per `vision_individual` in the story's crops, plus companions
    from stories it walked together with). After PR 1 each person also has
    `pictures`, `boxes`, `picture_crop_ids`, `pictured`.
  - `POST .../{story_id}/verdict` writes one story verdict
    (`right | wrong_name | wrong_motion | wrong_other`, or "can't tell") to
    `vision_motion_story_verdict`. The gate needs 50 checked, 80% right, 0 wrong names.
  - `POST .../{story_id}/people/{individual_id}` writes an already-answered
    `orion_ask` row: a name, and "yes" (PR 1), are `vision_individual` answers
    (a yes also carries one `same` picture verdict per picture shown);
    "someone else" and "not sure" are `vision_story_person` rows whose
    `answer_detail.pictures` hold one `different | unsure` verdict per picture shown.
- **Writer:** `services/orion-sql-writer/app/vision_individuals.py`
  (`_apply_answers` -> `_apply_typed_name` / `_record_pictures` into
  `vision_ask_picture_verdict`), then `app/vision_split.py` splits on
  "different" pictures, with an audit row in `vision_individual_split` and
  `scripts/undo_vision_split.py` to reverse one split.
- **Same-person links:** `vision_individual_pair_verdict` (`same | different`).
  A typed name makes a `same` link to an earlier group Juniper typed the same name
  for (`_link_same_name`). Labels copy across `same` links (`label_copied`).
  There is no crop merge; groups stay separate and are joined by links.
- **Consumers of picture verdicts:** the split reads `different`; the person
  model (`vision_person_model.py`) reads `different` as "not <name>"
  exemplars; the People panel (`orion/person_model/pictures.py`) shows `same`
  pictures, but only `thumb:` refs, so story-card answers (`crop:` refs) never
  appear there. Nothing reads `unsure`.

## Live numbers (read-only, conjourney, 2026-11-24)

| What | Count |
|---|---|
| Picture verdicts, all cards | 918: unsure 749, different 144, same 25 |
| ... from story cards (`vision_story_person`) | unsure 748, different 131, same 0 |
| Story-card clicks | name 270, not sure 412 (on 124 stories), someone else 58 |
| Story verdicts | right 260, wrong_name 5, wrong_other 2, can't tell 1 |
| Splits from "different" | 69 rows: 60 split, 6 skipped, 3 refused; 0 undone |
| Same-person links | same 313 (304 join two groups with one typed name), different 106 |
| Typed-name groups with more than one camera group | 30 names; 27 already one linked person |
| Person stories, last 30 days | 448 stories, 1,656 person rows (3.7 per story) |
| ... rows with no picture of their own | 345 (21%), now greyed by PR 1 |
| ... stories with more than one walker group | 361 (81%) |
| ... walker rows that repeat an already-linked same-name person | 148 rows on 97 stories |
| People per story | 1: 64, 2: 102, 3: 109, 4: 61, 5: 26, 6+: 86 |

Read: most stories show the same person split over two to four camera groups
(the tracker re-assigns mid-passage). The card treats each group as a person.

## Proposed design (plain English first)

### 1. One person at a time

The card shows one person, not a list. A person here is one linked group
(all camera groups joined by `same` links), not one camera group.

- Top: a tight crop of this person from this story (their largest strip crop),
  and the big picture with only their box numbers called out.
- Question: "Is this Addy?" when the group has a name; "Who is this?" when not.
- Three answers:
  - **Yes, that's Addy** -> PR 1's yes: a `vision_individual` answer with the
    name shown, plus `same` picture verdicts on the pictures shown.
  - **No, it's ...** -> opens a short list: the names Orion knows (most seen first)
    plus "someone not listed" (type a name) and "I don't know who". Picking a name
    writes `different` with that name; "I don't know who" writes `different`
    without a name. Both feed the split exactly as today.
  - **Can't see them** -> `unsure`. Says plainly: "Orion won't ask about these
    pictures again."
- Then "Next person (2 of 4)". A person without a picture of their own is
  skipped, never shown as a question (PR 1 rule).
- A group Orion already linked as one person is asked once. The answer is
  written for each camera group in it (one `orion_ask` row per group, same
  `story_id`), so the writer and split need no change.

### 2. The sentence, graded in parts

Replace "Right / Wrong name / Wrong motion / Wrong other" with yes/no rows, each
asked only when the sentence says it:

- "Right people?" (the name or "a person", and "with another person")
- "Right direction?" ("right to left", "toward the house")
- "Right place?" ("on the porch")
- "Right length?" ("over 6 s") - optional; propose leaving it out unless Juniper wants it.

Store each as a row in a new table `vision_motion_story_part_verdict
(story_id, part, verdict yes|no|cant_tell, rules_version, answered_at)`. The
gate's existing verdict is derived, so the gate and the 260 old verdicts keep
working:

- all parts yes -> `right`; people no -> `wrong_name`; direction or place no
  -> `wrong_motion`; any other no -> `wrong_other`; any can't tell -> can't tell.

"Wrong other" disappears from the card. The 2 old `wrong_other` rows stay.

### 3. Two groups that are one person

- **Already linked (the common case, 27 of 30 names):** show them as one person
  ("Addy - seen as 2 camera groups in this story"). No question.
- **Same name, not linked (3 names today):** one extra question after the person
  answers: "Orion has another Addy group. Same person?" with both crops side by
  side. Yes writes a `vision_same_individual` answer (the existing "same
  person?" path, `_apply_pair_answer`), which creates the `same` link.
- **Undo, mirroring the split undo:** add `scripts/undo_vision_link.py
  <ask_id>` and a Hub "Undo" next to "Linked as the same person" for 7 days. It
  deletes the `same` pair verdict that ask made and re-runs the label copy so a
  `label_copied` name the link gave is removed. It refuses when a later answer
  built on the link (a name typed after it, or a split whose `linked_pair` names
  it), and says which one. The pair-verdict row already carries `ask_id`, so no
  new audit table is needed; the undo writes one `[VISION_LINK_UNDONE]` log line
  and sets `orion_ask.answer_detail.undone_at`.
- No crop merge. Groups stay separate; links join them. This keeps every merge
  reversible.

### 4. The 748 "unsure" answers

Nothing reads `unsure`, so the backlog is inert: it never split a group and never
taught the person model. Proposal:

- Leave the rows. Do not re-ask them in bulk.
- A story with only `unsure` answers goes back into the unchecked queue once, so
  the new card can ask it properly (124 stories, about 2 weeks at her current
  pace of ~150 answers a week).
- Count `unsure` separately in the card's tally line ("can't see: N") so a high
  rate is visible. A rising rate is the signal that pictures are too small or
  too dark, not that Juniper is unsure.

## Missing questions (for Juniper)

1. "No, it's ..." - pick from known names, or always type? A list is faster but
   can put a wrong name one tap away.
2. Should "Yes, that's Addy" also count as "right people" for the sentence, so
   one click answers both? (Saves a click; mixes two questions again.)
3. Which sentence parts do you want graded: people, direction, place, length?
4. Same-name groups that are not linked (Bethany, Maeve, Juniper on the
   doorbell): ask on the card, or let Orion link them without asking, as it
   already does when you type a name?
5. Re-ask the 124 "not sure only" stories, or leave them closed?

## Proposed schema / API changes

- New table `vision_motion_story_part_verdict` (additive migration, `--
  ORION-MIGRATION-REQUIRED-BY: orion-hub`).
- `POST /api/vision/motion-stories/{story_id}/parts` `{part, verdict}`; writes the
  part row and upserts the derived story verdict in one transaction.
- `GET /api/vision/motion-stories` adds per story `persons` (linked groups:
  `{person_key, individual_ids, label, pictures, boxes}`) and `parts` (which
  parts the sentence has, and any saved answers). `people` stays for one release.
- `POST .../people/{individual_id}` accepts `action: "no"` with `name` (known or
  typed), mapped to today's `someone_else`; and `group: true` to write the answer
  to every camera group in the linked person.
- New `POST /api/vision/individual-links/{ask_id}/undo`.
- No new bus channel. No new verdict type: `same | different | unsure` only.

## Files likely to touch

- `services/orion-hub/static/js/walkway-stories.js` (+ `.test.js`)
- `services/orion-hub/scripts/motion_story_routes.py`
- `services/orion-hub/tests/test_motion_story_routes.py`,
  `test_walkway_stories_browser_smoke.py`, `test_walkway_stories_js.py`
- `services/orion-sql-db/manual_migration_vision_motion_story_part_verdict_v1.sql`
- `services/orion-sql-writer/app/vision_individuals.py` (link undo helper)
- `scripts/undo_vision_link.py`
- `orion/vision/motion_story_gate.py` (derived verdict, if the mapping lives there)
- `services/orion-hub/README.md`

## Non-goals

- No crop merge of camera groups.
- No change to the split rules or the 7-day crop retention.
- No new verdict type or new ask kind.
- No change to the gate numbers (50 / 80% / 0 wrong names).
- Not the People panel. Its `thumb:`-only picture filter is a separate ticket.

## Proposal-mode checklist

- **Capability that changes:** Juniper can confirm a person, correct one, or say
  she can't see them, one person at a time; she can grade a sentence in parts;
  she can join or un-join two same-name groups.
- **Data touched:** `orion_ask` (new answered rows, same kinds as today),
  `vision_ask_picture_verdict` (via the writer), `vision_individual_pair_verdict`
  (`same` links, and deletes on undo), new `vision_motion_story_part_verdict`,
  `vision_motion_story_verdict` (derived).
- **Privacy boundary:** unchanged. Camera pictures of people at the house stay in
  the Hub and the database; nothing new leaves the host. Names are Juniper's own
  answers.
- **Trace that proves it worked:** per answer, the `orion_ask` row with
  `answer_detail.story_id`, then the writer's `[VISION_PICTURE_VERDICT]` /
  `[VISION_SAME_NAME_LINK]` / `[VISION_LINK_UNDONE]` log lines; the tally line on
  the card; the acceptance queries below.
- **Dangerous failure:** a wrong "same" link joins two real people, and their
  names copy across both groups; or a "No, it's X" picked from a list by
  mistake splits a group. Both already happen today with typed answers; the list
  makes the mistake one tap closer.
- **Disable / roll back:** the card change is one static file and one route
  module (revert the PR). Links undo per ask (`undo_vision_link.py`), splits per
  split (`undo_vision_split.py`). The new table is additive; dropping it loses
  only part answers, and the derived story verdicts stay.

## Acceptance checks

Numbers are from the live database on 2026-11-24. Re-measure 14 days after deploy.

1. **No blind answers:** 0 story-card picture verdicts on a crop the card did not
   show (check: every `vision_story_person` picture `crop_id` is in that story's
   strip crops or its thumb crop). Today: not checkable for old rows.
2. **"Yes" is used:** `same` verdicts from story cards go from 0 to at least 30%
   of story-card picture verdicts.
3. **Fewer "not sure":** story-card `unsure` share drops from 748 / 879 (85%) to
   under 40%.
4. **Fewer questions per story:** asked persons per story drop from 3.7 rows
   (1,656 / 448) to under 2.5, because linked groups are asked once and unpictured
   rows are skipped (21% today).
5. **Repeat rows gone:** 0 stories show the same linked person twice (148 rows on
   97 stories today).
6. **Gate still moves:** the existing tally reads the same before and after for
   old verdicts (260 right, 5 wrong_name, 2 wrong_other); new part answers produce
   story verdicts.
7. **Undo works:** a test links two groups, undoes it, and the pair verdict row
   and the copied label are gone; an undo after a dependent name refuses and names it.
8. **Browser smoke** at 360 px: one person shown, Yes / No / Can't see, Next,
   sentence parts; no horizontal overflow.

## Recommended next patch

Patch A (small, no schema): group already-linked camera groups into one person
row on the current list card, ask once, and write the answer to each group. This
removes the "Addy, Addy" double ask today (148 rows on 97 stories) without the
one-at-a-time layout. Patch B: the one-at-a-time card and "No, it's ...". Patch
C: sentence parts table and derived verdict. Patch D: link undo.
