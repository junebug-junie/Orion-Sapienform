# Your clues as evidence in who-is-this guesses

Status: PROPOSAL (proposal mode: identity / person model). Ticket: #3768.
Date: 2026-11-24.

## Arsonist summary

Orion guesses who someone is from two things: how much the picture looks like
a person's album, and who is usually on that camera. When a back is turned, the
album match is weak and Orion has to ask.

Juniper already tells Orion how she can tell. The "How could you tell?" box
(PR #3769) has 7 reasons saved: a white dog, a stroller, a purple puffy jacket,
shorty shorts, a tight ponytail, tall or short, walking with a usual companion, leaving the
house after dark. Nothing reads them.

This proposal makes those clues count as evidence, both for a name and against
it, and learns how much each clue is worth per person from her confirmed
answers. Her reason is the starting belief, never a hard rule.

If it does not name people correctly more often on replay than the album
alone, it does not ship.

## Current architecture

- **Albums.** `services/orion-sql-writer/app/vision_person_model.py`: one album
  of crops per named person (+1 "this is them", -1 "not them"), from her
  answers only. `attribute()` scores each unnamed cluster against each album
  and puts it in a band: accept ("probably X"), unsure (ask), reject.
- **Camera priors.** `orion/person_model/camera_priors.py`: who is usually on a
  camera. Used only when no album match is confident.
- **Facts.** `person_fact` rows (`orion/person_model/facts.py`), in her own
  words, shown and edited on the Hub People panel
  (`services/orion-hub/scripts/people_routes.py`). `person_record.py` already
  turns answered asks into `person_fact` rows.
- **One fact already changes a guess.** The age veto
  (`docs/superpowers/specs/2026-11-13-person-age-band-soft-veto-design.md`):
  each crop's stored SigLIP vector (an image embedding the camera already
  saves) is compared with text like "a photo of a child". Pure numpy, no model
  loaded. 83% right on named crops. It can only strike a name.
- **Story reasons.** `orion_ask.answer_detail->>'why'` on walkway story answers
  (PR #3769). Stored, shown back, read by nothing.
- **Companions.** Walkway stories link people who walked past together
  (`together_story_ids`).

## Missing questions

1. Which clue kinds can be read from a picture well enough to use? The age read
   is 83% right. A stroller or a white dog may read well; "shorty shorts" or a
   ponytail from the back may not. Patch 2 measures each one before it is used.
2. A reason mixes facts about the person ("I rarely push a stroller") with
   notes on this one picture ("it is dark out"). Patch 1 keeps the reason
   verbatim; the clue list in patch 2 is what separates them, and Juniper sees
   and edits it.
3. Clothes change. A jacket is a strong clue in winter and none in summer. The
   learned weight must forget: only recent confirmed sightings count
   (default: 60 days for clothes, no limit for a dog or a companion).

## Proposed schema / API changes

### Patch 1: reasons become facts (small, no change to guesses)

- `person_record.py` turns each story reason into a `person_fact` row on the
  person it is about: the name she typed, or for "Someone else" / "Not sure",
  the person Orion had guessed (the reason is still about them).
- Category `how_i_tell` (added to `orion/person_model/categories.yaml`, never
  asked as a question: `about_other: null`). Voice `story_reason`.
  `source_ref` = the ask id, so it is written once.
- Shows on the person's People page with the other facts. Removable there.

### Patch 2 revised (2026-11-24, after measuring live data)

The plan below assumed every clue needs a new picture reader. Live data says
otherwise:

- The detector already boxes `stroller` (since 2026-11-18) and `dog`. They need
  no SigLIP read: a clue is "a box of that kind in the same frame, within one
  person-height sideways".
- The walked-together link (`together_pairs`, >= 2 shared moving instants) misses
  most dogs: a dog is often a one-frame passage. So clues are read per frame,
  not from `together_story_ids`.
- Answer key: the person album (`vision_person_exemplar` +1, not also -1).
  That is ~99.8% crops of clusters she named that passed the album purity
  check, not pictures confirmed one by one. Good enough to measure; not perfect.
- Unit: a passage (one cluster, one stream, one minute). Frames of one walk
  past are not independent; per-crop counts overstated the evidence ~2x.

Measured with `services/orion-sql-writer/evals/run_clue_rates_eval.py`
(2026-11-18..24, 372 street passages; all stroller boxes fall in ~23 minutes,
so this is about 20 street stroller events in one week):

| clue | street rate | regulars (>= 10 passages) | weight of evidence |
|---|---|---|---|
| stroller | 7.8% | 0-4% | -3.4 for the most-seen regular; -0.5 to -1.4 for the rest (thin) |
| dog | 11.3% | 0-27% | -1.7 to +0.9 (person-dependent, positive for some) |
| named dog | 4.0% | 0-17% | -0.7 to +1.5 (weak) |

Street includes regulars not named yet, which shrinks weights toward 0.

So: stroller is evidence "not them" in direction, but the size rests on ~20
events; patch 3 must learn it per person from passages, not take these
numbers as fixed. Dog clues can point either way and stay low weight.
Clothes, hair and build still have no answer key: they wait for a labelled
set, and the SigLIP plan below applies only to them.

### Clothes and hair, measured (2026-11-25): not usable yet

`services/orion-sql-writer/evals/run_clothing_clue_eval.py` scores six generic phrases
from her reasons (puffy purple jacket, red hair, tight ponytail, blue denim shirt,
tights, shorts) against the SigLIP vectors already stored for every crop (minus a
neutral "a photo of a person"), per passage, person album vs street (AUC; 0.5 =
nothing). Answer key = the album, no new labels. Height and build are not
readable from one crop vector and were not tested.

Live, last 30 days (2,375 street passages, 14 people with >= 10 passages):

- A person's own stated phrases point the right way but weakly: the best case,
  three phrases combined, reaches AUC 0.67 (rank 1 of 14; next 0.59) over 30 days
  and ties (0.64) over 14 days. Another stated set reaches 0.61-0.70 and ranks 2nd.
- A summer phrase in November scores 0.33 for its person: clothes clues are seasonal.
- A decoration scores 0.70 on "red hair": part of the read is light / camera, not person.

For comparison the stroller clue is ~30:1 against a regular per sighting. So
clothes and hair are NOT wired into guesses. Re-run the eval as albums grow or the
season changes; a phrase needs a clear per-person lead before it is considered.

### Patch 2 (original plan): a clue list per person, read from pictures

- `orion/person_model/clues.yaml` (committed, generic): the clue kinds that
  can be read, each with its text prompts, the same way the age read works.
  Starting set, from her 7 reasons: `stroller`, `dog_white`, `dog_any`,
  `puffy_jacket`, `shorts`, `ponytail`. Height and hair colour only if they
  pass the gate below.
- `orion/vision/clue_read.py`: pure numpy, like `age_band.py`. Crop vector +
  committed text vectors -> `{clue: present | absent | unclear}`.
- Gate per clue (offline eval on crops she has named, before it is used):
  right at least 80% of the time on crops where it is not `unclear`. A clue
  that fails is left out. Results go in the design PR, not in prose.
- `person_clue` (new table): `person_id, clue, stance (usually | rarely),
  source (her reason | Hub edit), created_at, removed_at`. Built from her
  `how_i_tell` facts by a short LLM pass that only maps her words onto the
  clue list (no new clues invented). Juniper sees and edits it on the People
  page. Nothing is stored unless it maps to a listed clue.
- Two clues need no picture read: `with:<person>` (from `together_story_ids`)
  and `at:<camera>/<hours>` (from the sighting's stream and time).

### Patch 3: clues as evidence in the guess

- In `attribute()`: each person's score gets a clue term, added in log-odds
  (weights of evidence), after the album score and before banding:
  `clue term = sum over clues read on this cluster of log(P(clue | it is X) / P(clue | street))`.
- `P(clue | street)`: how often the clue reads on unnamed street clusters
  (last 30 days).
- `P(clue | it is X)`: starts from her stance (`usually` = 0.8,
  `rarely` = 0.05), then is updated by her confirmed sightings of X
  (Beta counts: her stance is worth 5 sightings). So her rule is the prior and
  her answers move it.
- Cap: the clue term can move a score by at most one band step. Clues alone
  never name anyone with no album match: a white dog is not enough for
  "probably X" if X's album says no.
- The guess carries `clues: [{clue, read, weight}]` so the reason is visible,
  and the ask card says it: "Back turned, white dog. Is this <name>?"

No bus or schema change in patches 1-2 beyond the new table and fact category.
Patch 3 adds `clues` to the guess dict that the Hub reads.

## Files likely to touch

- `orion/person_model/categories.yaml`, `services/orion-sql-writer/app/person_record.py` (patch 1)
- `orion/person_model/clues.yaml`, `orion/vision/clue_read.py`, `orion/vision/clue_text_vectors.json`,
  `scripts/generate_clue_text_vectors.py`, a `manual_migration_person_clue_v1.sql`,
  `services/orion-hub/scripts/people_routes.py` + People panel JS (patch 2)
- `services/orion-sql-writer/app/vision_person_model.py`, the ask card text (patch 3)
- Evals: `orion/person_model/evals/run_clue_read_eval.py`, `run_clue_guess_replay_eval.py`

## Non-goals

- No gender read, ever (same reason as the age design: it misgenders).
- No face recognition change. Albums stay as they are.
- No clues invented by Orion. Clues come from her words, mapped onto a fixed list.
- No naming from clues alone.
- Not for the office camera in patch 3 (its prior path was never replayed),
  same exemption as the age veto.

## Privacy, danger, rollback

- Data touched: her reasons (already stored), person facts, stored crop vectors.
  No new pictures, nothing leaves the host.
- Dangerous failure: two people who share clues (same dog, same jacket) get
  merged in guesses. Guard: the band cap, and her "not them" answers still
  block a name for that cluster.
- Off switch: `clue_evidence_enabled` in `PersonModelConfig` (ships on, per
  house rule); off = today's guesses exactly.
- Patch 1 is additive facts; removing a fact on the People page undoes it.

## Acceptance checks

1. Patch 1: each of the 7 saved reasons appears once on the right person's
   People page (live query on `person_fact` where voice = `story_reason`).
2. Patch 2: per-clue read accuracy on her named crops, in the PR. Only clues at
   >= 80% are enabled.
3. Patch 3 replay on every story and who-is-this answer she has given:
   - right-name rate, album + clues vs album alone, overall and for
     back-turned / no-face clusters;
   - wrong-name rate must not rise (that is the dangerous one).
   Ships only if right-name rate rises and wrong-name rate does not.
4. Live, after 2 weeks: her "yes" rate on Orion's who-is-this asks, before vs
   after. Filed as a `verify-live` ticket.

## Recommended next patch

Patch 1: story reasons become `how_i_tell` facts on the right person. It
changes no guess, is useful on the People page on its own, and is what
patches 2-3 learn from.
