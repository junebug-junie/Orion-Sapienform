# Splitting walkway individuals from Juniper's answers (proposal)

Date: 2026-10-16. Status: proposal (design only). Identity change: proposal
mode per AGENTS.md section 0A. Nothing here is built yet.

## Arsonist summary

Orion groups walkway sightings into "individuals". Some individuals hold
more than one person. Juniper's answers already say which pictures belong to
someone else, but Orion only records those answers. It never acts on them.

This proposal makes Orion act on them. When Juniper marks a picture "someone
else", Orion moves that picture (and a few close face matches) into a new
individual. It keeps a full record of the move so that one command can undo
it.

The replay on today's live data shows two hard truths:

1. **A split fixes the record, not the recognition.** After the split, 0 of
   the next 516 person crops joined the new individuals. Every new crop went
   back to the big mixed individuals, because the body embedding cannot tell
   these people apart. (The body embedding is the "what the whole person
   looks like" vector.)
2. **No face threshold separates Juniper's labelled pairs today.** One pair
   she called "same" scores lower than two pairs she called "different". If
   we raise the threshold, we lose real matches and still do not stop the
   wrong merges.

So the first patch is worth building, but only for what it does. Ask cards,
the street line, and names will stop showing a person under someone else's
name. It will not make Orion recognise that person next time. That needs a
second, separate change to how crops get matched, and it needs more answers
first.

## Current architecture

- **Clusterer:** `services/orion-sql-writer/app/vision_individuals.py`
  `apply_batch`. The loop is `vision_individuals_loop.py`, and it runs every
  60 s.
  - A crop with a face joins the nearest face centroid at cosine >= 0.60
    (`VISION_INDIVIDUALS_FACE_MATCH_THRESHOLD`).
  - Otherwise it joins the nearest body centroid at >= 0.80
    (`VISION_INDIVIDUALS_MATCH_THRESHOLD`).
  - Otherwise it opens a new individual.
  - Two crops from one frame can never join the same individual.
  - A centroid is the running mean of unit vectors. Each centroid keeps a
    count (`centroid_n`, `face_n`).
- **Membership is not stored.** `vision_crop_observation` has no
  `individual_id` column. A sighting keeps only its time span, counts, and its
  newest crop's `thumb_ref` / `face_thumb_ref` / `evidence_ref`. To find
  "which crops belong to 62cb5fcc", you must replay the clusterer. The replay
  is exact today (see "Replay eval"), but it only works while crops are still
  kept. Crops are kept for 7 days.
- **Answers:** `_apply_answers`.
  - A "who is this?" answer sets `vision_individual.label` and copies it to
    every individual linked by a "same" pair verdict (`_propagate_label`).
  - Each picture verdict goes to `vision_ask_picture_verdict` (`_record_pictures`),
    which says "record only: nothing here splits an individual".
  - A "same person?" answer goes to `vision_individual_pair_verdict`. A
    "same" verdict links two individuals and never merges them. A
    "different" verdict only stops Orion asking about that pair again.
- **Consumers keyed on `individual_id`:**
  - Hub asks (pictures come from sighting thumbs).
  - The street line `orion/situational/perception_reader.py`. It already shows
    each name once (it compares names case-insensitively).
  - Rhythm `vision_rhythm.py`, with subject `individual:<id>`, read from sightings.
  - Attention scores on sightings.
  - Substrate entity nodes. The node id is a hash of `individual_id`
    (`orion/substrate/adapters/vision_individual.py`), written by
    `services/orion-substrate-runtime/app/ask_answered_listener.py`.
  - Forecasts `services/orion-actions/app/walkway_forecast.py`, from sightings.

### Live state used here (read-only, 2026-10-16 ~02:12 UTC)

- **Person crops:** 4,988 walkway person crops with a body embedding. 155 of
  them also have a face embedding. These crops make up 40 person
  individuals. The biggest are 1,166, 758, 738, 334, and 297 crops.
- **Labelled individuals:**

  | Individual | Label | Crops | Face crops |
  |---|---|---|---|
  | 62cb5fcc | Maeve (typed) | 758 | 25 |
  | f5242d40 | Teddy (typed) | 297 | 35 |
  | 4354c877 | Amanda Feld (copied) | 84 | 49 |
  | 9bea16d5 | Amanda Feld (typed) | 54 | 23 |

- **Picture verdicts:**
  - Ask 9dd6a5ad on 62cb5fcc: picture 0 is the same person (Maeve). Pictures
    1 and 2 are a different person, named Esme. Pictures 1 and 2 are body crops
    with no face.
  - Ask 64490457 on f5242d40: pictures 0 and 1 are the same person (Teddy).
    Picture 2 is a different person, named Esme. All three pictures have a face.
- **Pair verdicts:**

  | Pair | Verdict | Face cosine |
  |---|---|---|
  | 4354c877 / 9bea16d5 | same | 0.533 |
  | 62cb5fcc / f5242d40 | different | 0.758 |
  | 4354c877 / f5242d40 | different | 0.405 |

## Replay eval (read-only, live data)

**Method:**

1. Export the walkway person crops and individuals with `COPY ... TO stdout`.
   This is a read-only export.
2. Re-run the matching rule in 60 s landing-time batches. The rule is face
   first at 0.60, then body at 0.80, with the same-frame exclusion.
3. Record crop -> individual for every crop.

**Fidelity:** the replay rebuilt **40 of 40** live person individuals exactly,
with the same `centroid_n` and centroid cosine > 0.999. The crop membership
below is therefore the real membership.

The scripts are in the session scratchpad. They are not committed, because
this is a design-only patch. Patch 1 turns them into
`services/orion-sql-writer/evals/run_vision_split_replay_eval.py`.

### What a split would produce

Split rule tested: move the crops Juniper named "different". Also move a
member face crop if it meets both conditions below. (A "seed" is a crop
Juniper named. The "keep seed" is the crops she called "same". The "move
seed" is the crops she called "different".)

- Its face cosine to the move seed is >= 0.60.
- Its face cosine is at least 0.10 higher to the move seed than to the keep
  seed.

| Source individual | Crops moved by answer | Face crops that follow | Stay | New vs remaining, body cosine | New vs remaining, face cosine |
|---|---|---|---|---|---|
| 62cb5fcc "Maeve" | 2 | 0 (no face seed for Esme) | 755 | 0.977 | n/a |
| f5242d40 "Teddy" | 1 | 3 (4 at margin 0.05) | 287 | 0.944 | 0.728 |

The two new "Esme" pieces score 0.889 against each other on body cosine.
There is no face cosine, because one piece has no face.

### Why body-embedding "nearest neighbour" must not decide who follows

- Inside 62cb5fcc, the live body centroid scores 0.977 against the Esme
  pictures and only 0.851 against the Maeve picture. If each of its 758
  crops went to whichever named seed is nearer by body embedding, **734 of
  758 crops would leave "Maeve"** (715 with a 0.05 margin). That does not
  split off Esme. It shows the whole individual is a body-embedding blob
  that got the name "Maeve" from its 25 face crops.
- Two different people in the same frame score body cosine p10 0.673,
  p50 0.752, p90 0.832 (n = 2,910 pairs). **20% of known-different pairs are
  at or above the 0.80 body threshold.** At this camera distance, the body
  threshold sits inside the range for different people.

### Does the split hold? (forward replay)

I applied the split at Juniper's answer time (23:22 UTC 10-15) and continued
the replay over the next 516 person crops:

| Individual | New crops |
|---|---|
| 62cb5fcc | +2 |
| f5242d40 | +19 |
| Esme piece from 62cb5fcc | +0 |
| Esme piece from f5242d40 | +0 |

The replay cannot show whether Esme walked past after 23:22. It does show
that nothing changed for incoming crops. The new individuals are close to
the big ones on body cosine (0.944 and 0.977), so incoming crops still go to
the bigger ones.

### Face threshold: what the labelled data says

All pairs labelled so far:

| Pair | Juniper says | Face cosine |
|---|---|---|
| 4354c877 / 9bea16d5 (individual centroids) | same | 0.533 |
| Teddy picture 0 / Teddy picture 1 (crops) | same | **0.151** |
| 62cb5fcc / f5242d40 (centroids) | different | **0.758** |
| Maeve picture / Esme picture (on Teddy's ask) | different | **0.646** |
| Teddy picture 0 / Esme picture | different | 0.582 |
| Maeve picture / Teddy picture 0 | different | 0.511 |
| 4354c877 / f5242d40 (centroids) | different | 0.405 |
| Maeve picture / Teddy picture 1 | different | 0.145 |
| Teddy picture 1 / Esme picture | different | 0.125 |

Earlier measurement, in the PR #2698 report and re-checked here on 44 pairs:

- Different faces in the same frame: p50 0.27, p90 0.48.
- The same face 3 s or less apart: p10 0.39, p50 0.65, p90 0.76.

The labelled "same" and "different" sets overlap completely. No threshold
separates them.

The Teddy-vs-Teddy score of 0.151 is suspicious. The cards show body
thumbnails, but the face embedding comes from whatever face the detector
found inside the person box. That face may belong to someone else, for
example a child being carried. This is Missing question 1.

## The proposal

### 1. What a split does

**Trigger.** A `vision_ask_picture_verdict` row with `verdict='different'`
whose crop is still a member of the asked individual. A verdict on a crop
that has already been moved or pruned does nothing, and the split trace logs
why.

**Group.** One split per (asked individual, name). Pictures with the same
typed name move together. Pictures marked "different" with no name each
form one unnamed group per ask. One ask with two different names makes two
splits.

**Who keeps the id.** The side Juniper confirmed keeps the id. That side
holds the "same" pictures, or the typed label if no picture is "same". It
keeps the `individual_id`, its label, and its substrate entity node. The
moved crops get a **new** `individual_id`.

Identity follows her answer, not crop count. For 62cb5fcc, that matters:
most of its body crops look more like Esme than Maeve, but the name came
from Maeve's face.

**Which crops move.** Two kinds of crops move:

- **(a)** The crops she named.
- **(b)** A member *face* crop, when all three conditions below hold:
  - The move seed has a face.
  - The crop's face cosine to the move-seed face centroid is >=
    `VISION_SPLIT_FOLLOW_MIN_FACE_COSINE` (0.60).
  - That cosine beats the keep-seed face centroid by >=
    `VISION_SPLIT_FOLLOW_MARGIN` (0.10).

Body-only crops never follow, for the reason measured above. Today, (b) adds
3 crops on f5242d40 and 0 on 62cb5fcc.

**Centroids.** Recompute both centroids exactly from member crops: the sum of
unit vectors, then renormalize. Do not try to subtract crops from the running
mean. `centroid_n` and `face_n` become the member counts. `distinct_days`,
`first_seen_at`, and `last_seen_at` are recomputed from member crops. The
stationary-run fields reset on the new individual.

**Sightings.** Rebuild both individuals' sightings over the time windows that
contain moved crops, with the same grouping rule as `apply_batch`
(merge gap 60 s):

- The source's rebuilt sighting keeps its `sighting_id`.
- Groups of moved crops get new sightings.
- Thumb, face-thumb, and evidence refs are re-picked from members. This is
  the step that stops the next ask card from showing Esme under "Maeve".
- Attention scores are recomputed for rebuilt sightings.
- `vision_events` rows already emitted stay as history. A split never edits
  history.
- Sightings older than crop retention (7 days) cannot be rebuilt, because
  their crops are gone. They stay with the source. The audit row records
  how many sightings were out of reach.

**Rhythm and forecasts.** These modules are not rewritten. They read sightings,
so the next cycle picks up the rebuilt sightings. Expectations that were
already graded stay graded. The new individual has too few occurrences to
get a rhythm at all (fewer than `min_occurrences`).

**Substrate.**

- The source's entity node is untouched: same id, same name.
- A named new individual gets its own entity node, through the same writer
  that a typed name uses. The patch must check whether `ask_answered_listener`
  needs a new event kind, or whether the split can emit the existing
  ask-answered event with `names={new_id: name}`. It must use the second if
  the schema allows it.
- An unnamed new individual gets no node, as today.
- **Never rename a named entity.** A split cannot change the label of the
  side that keeps the id.

### 2. Name-driven vs verdict-driven

- **"Someone else" with a name.** The new individual gets that name as a
  typed label, applied with `_apply_typed_name`. If another individual on the
  same stream and kind already has that exact typed name, the split adds a
  `vision_individual_pair_verdict` row `same` between the two. It does **not**
  move crops into the other individual.
  - Today: the second Esme split links to the first. The existing group logic
    then treats both as one Esme. The street line already shows each name
    once.
  - Reason: two of today's four labelled people already have two
    individuals each ("Amanda Feld"). Linking keeps every split undoable on
    its own. A merge would mix two splits' crops.
- **"Someone else" without a name.** The new individual has no label. It is
  eligible for the normal "who is this?" ask once it has enough faces, under
  the existing rules.
- **Should a "same" group be merged for matching?** Not in patch 1. The
  measured "same" pair (0.533) sits *below* the face threshold, so it would
  route new crops to one member or the other. Merging the matching would
  change which member gets them, and would not change whether the group
  gets them. Counting the group as one person is already a consumer job.
  The street line does it by name. Revisit when there are 20 or more "same"
  pairs.

### 3. Preventing re-merge

- **Cannot-link.** Every split writes a `different` pair verdict between the
  source and the new individual. Nothing merges individuals today, so this
  record is what any future merge must respect.
- **Patch 2, online matching.** A cannot-link also does something for crop
  routing, as an **ambiguity rule**. A crop's best face match is X at
  similarity s. X has a cannot-link with Y. If Y scores within 0.10 of s, the
  crop does not join X by face. It falls through to the body rule. This only
  fires where Juniper has said two individuals are different, so it cannot
  disturb anything else.
  - It is in patch 2, not patch 1, because the forward replay shows the
    face path is not where re-mixing happens. The body path is.
- **Face threshold calibration rule.** The threshold moves only when all of
  these hold:
  1. There are at least 20 labelled "same" face pairs and at least 20
     labelled "different" face pairs. These come from pair verdicts, plus
     picture verdicts expanded into crop pairs: "same" pictures in one ask
     pair with each other, "same" pictures pair with "different" pictures,
     and pictures with the same typed name pair across asks.
  2. Some threshold t lets through at most 5% of "different" pairs at or
     above it and keeps at least 50% of "same" pairs. The proposed value is
     the smallest such t, plus 0.02.
  3. The proposal goes to Juniper as a card that shows both error rates.
     Juniper approves it. Then the value is changed in `.env_example` and
     `.env`. Nothing writes the threshold by itself.

  With today's data, the rule gives **hold**: 2 same pairs and 7 different
  pairs, and the two sets do not separate. The rule runs inside the replay
  eval, so every run prints the current verdict.
- **The body threshold (0.80) is the real cause of the blobs.** 20% of
  known-different pairs pass it. Changing it is outside this proposal,
  because it changes every individual, not only labelled ones. It is listed
  as a missing question and a follow-up.

### 4. Automatic or ask first

**Recommend automatic, on her answer.**

- A picture marked "someone else" already *is* the confirmation: she looked
  at that exact picture.
- A second "split?" card would ask the same question twice. That is the
  repeated yelling AGENTS.md section 0A says to replace with a gate.
- Face crops that follow are Orion's own inference (0 and 3 crops today).
  They move automatically too, but the audit row marks each one
  `moved_by='face_follow'`. Undo restores them along with everything else.

Kill switch: `VISION_SPLIT_ON_VERDICT_ENABLED`. Per standing practice it ships
`true` in both `.env_example` and settings. When it is `false`, picture
verdicts are recorded exactly as today and nothing moves.

### 5. Reversibility

**Table `vision_individual_split`:**

| Column | What it holds |
|---|---|
| `split_id` | Primary key |
| `source_individual_id` | The individual that was split |
| `new_individual_id` | The individual created by the split |
| `ask_id` | The answer that triggered it |
| `name` | The typed name, if any |
| `created_at` | When the split happened |
| `undone_at` | When it was undone, if it was |
| `moved` | jsonb list of `{crop_id, moved_by: verdict \| face_follow}` |
| `before` | jsonb: both individuals' centroids, counts, day and seen fields, and the source's affected sighting rows, in full |
| `after` | jsonb: the same fields after the split |
| `linked_pair` | The pair verdict rows it wrote |
| `sightings_out_of_retention` | Count of sightings too old to rebuild |

**Undo:** `scripts/undo_vision_split.py <split_id>`. It refuses to run
without `--yes`, and prints a dry run first. It does these steps in order:

1. Move the crops back.
2. Restore `before` exactly.
3. Delete the new individual's sightings and the pair verdicts the split
   wrote.
4. Clear the new individual's label.
5. Set `undone_at`.

Undo refuses if crops have joined the new individual since the split. In
that case it names the crops, and Juniper decides. It never drops that data
silently.

**Substrate node on undo.** A node created for the new individual is not
deleted, because substrate history is append-only. It is marked superseded
through the same writer, with `evidence_refs=[split:<id>]`. Patch 1 must
confirm that this mark exists in the adapter. If it does not, undo logs that
the node is now orphaned. This is the one non-clean edge.

**Trace:** a log line `[VISION_SPLIT]` with `split_id`, `ask_id`, `source`,
`new`, `moved_verdict`, `moved_follow`, `stay`, `body_cos`, and `face_cos`.
Undo writes the same line, tagged `[VISION_SPLIT_UNDONE]`.

### 6. Privacy boundary

- **Data touched:** walkway crop membership, individual centroids, sightings,
  labels, pair verdicts, and substrate entity nodes. All of it is already in
  `conjourney`, and it stays there.
- **What crosses a boundary:** no new pictures or vectors. The audit table
  holds crop ids and centroids. It does not hold images. Its retention
  should follow sightings (90 days), because it holds centroids, which are
  biometric-adjacent.
- **Names:** names come only from Juniper's typed answers, as today.
- **Juniper and Orion:** there is no privacy boundary between them by
  standing decision. The boundary that matters is this: names and face
  vectors never leave the host DB, and no external call is added.

### 7. Dangerous failure mode

**The wrong side keeps the name.** If the "who keeps the id" rule picked by
crop count, splitting 62cb5fcc by body neighbour would move 734 crops away.
The name "Maeve" would stay on 24 crops, or end up on Esme's body crops. Any
later chat or street line would then name the wrong child.

**Guards:**

- Identity follows the confirmed side.
- Body-only crops never follow.
- A split that would move more than 50% of the source's crops is refused
  and logged `[VISION_SPLIT_REFUSED] reason=majority`. Under the proposed
  rule this cannot happen today: the most a split moves is 4 of 291.
- A gate test pins all three guards.

**Second danger: re-splitting in a loop.** A replayed answer could split
again. To prevent that, a verdict whose crop is already outside the asked
individual does nothing, and `ask_id` is unique per source in the audit
table.

## Missing questions

1. Do the cards show the face the face embedding came from? Teddy picture 0
   vs Teddy picture 1 scores 0.151. If the cards show body thumbnails but the
   face embedding is from another face in the box, every face calibration
   number is noisy. That would need a check of `face_thumb_ref` vs `thumb_ref`
   on these crops, in the Hub, before patch 2.
2. Should the body threshold be raised, or should body-only crops be made
   unable to carry a name? The measured overlap says the blobs come from the
   body path. That change touches every individual, so it needs its own
   proposal.
3. Should membership be backfilled? The forward write is cheap. A one-time
   backfill by replay is exact today (40/40), but it is a production write
   under the AGENTS.md section 14 backfill protocol: snapshot first, then
   Juniper approves.

## Proposed schema / API changes

- **`vision_crop_observation.individual_id text NULL`**, with index
  `(individual_id, observed_at)`. The clusterer writes it in the same
  transaction as the batch. This is the prerequisite: without it, a split
  needs a full replay. It is a migration with header
  `-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer`.
- **New table `vision_individual_split`**, as described above.
- **No new bus channel and no new schema**, if the existing ask-answered
  event can carry the new individual's name to the substrate listener. If it
  cannot, the patch adds a field there and updates the registry and
  `channels.yaml` in the same change.
- **New env keys** (orion-sql-writer):
  - `VISION_SPLIT_ON_VERDICT_ENABLED=true`
  - `VISION_SPLIT_FOLLOW_MIN_FACE_COSINE=0.60`
  - `VISION_SPLIT_FOLLOW_MARGIN=0.10`
  - `VISION_SPLIT_MAX_MOVE_FRACTION=0.5`

## Files likely to touch

- `services/orion-sql-writer/app/vision_individuals.py`: write membership,
  add a pure `plan_split(...)`, add `_apply_splits` after `_record_pictures`,
  rebuild sightings.
- `services/orion-sql-writer/app/settings.py`, `.env_example`, local `.env`
  (sync script), `README.md`.
- `services/orion-sql-writer/sql/manual_migration_*.sql`: the membership
  column and the split table.
- `services/orion-sql-writer/tests/test_vision_split.py`: plan, guards,
  undo, and idempotence.
- `services/orion-sql-writer/evals/run_vision_split_replay_eval.py`: the
  replay above, made permanent. It also prints the calibration rule's
  verdict.
- `scripts/undo_vision_split.py`.
- Possibly `services/orion-substrate-runtime/app/ask_answered_listener.py` and
  `orion/schemas/ask.py`, only if the event cannot carry the new individual.

## Non-goals

- No change to the body or face match thresholds in this proposal.
- No merging of individuals, and no merge UI.
- No confirm card for splits.
- No rewrite of history: emitted `vision_events`, graded rhythm
  expectations, and substrate history stay as they are.
- No splitting of sightings older than crop retention.
- No new pictures, no new external calls.

## Acceptance checks

1. **Gate tests.** The test file is `tests/test_vision_split.py`, run with
   `pytest services/orion-sql-writer/tests/test_vision_split.py -q`. It must
   show all of these:
   - A named "different" verdict creates one new individual with exactly
     the named crops, plus face crops that pass both follow conditions.
   - Body-only crops never follow.
   - The confirmed side keeps its id, its label, and its entity node.
   - A majority move is refused.
   - Running the same answer twice changes nothing the second time.
   - Undo restores `before` byte-for-byte.
2. **Replay eval.** On live data (read-only),
   `run_vision_split_replay_eval.py` reproduces this doc's numbers:
   - 62cb5fcc: 2 moved, 0 followers.
   - f5242d40: 1 moved, 3 followers.
   - Body cosines 0.977 and 0.944.
   - Calibration verdict "hold".
3. **Live, after deploy.** The live path must show all of these:
   - `[VISION_SPLIT]` lines for asks 9dd6a5ad and 64490457.
   - Two `vision_individual_split` rows.
   - Two new individuals labelled "Esme", linked by a `same` pair verdict.
   - `different` pair verdicts from each Esme individual to its source.
   - The next "who is this?" card for 62cb5fcc or f5242d40 shows none of the
     moved crops' thumbs. Verify the image refs, not only the payload.
4. **Undo.** `scripts/undo_vision_split.py` restores one split on a scratch
   copy of the tables.
5. **Honest outcome metric.** For 7 days, report how many new crops join
   each Esme individual. The expected value is about 0, because of the
   forward replay above. If it stays at 0, patch 2 is needed, and that is
   the evidence for it.

## Recommended next patch

1. Add a nullable `individual_id` column to `vision_crop_observation`. The
   clusterer writes it with every batch.
2. Add a `vision_individual_split` audit table and `undo_vision_split.py`.
   The undo restores the exact "before" state.
3. On a "someone else" picture answer, move that crop and its close face
   matches (never body-only crops) into a new individual. The confirmed side
   keeps the id and the name.
4. A typed name becomes a label on the new individual, plus a "same" link to
   an existing individual with that name. Every split also writes a
   "different" link back to the source.
5. A kill switch ships ON. The replay eval pins today's numbers (2 + 0 and
   1 + 3 crops) and prints "hold" for the face threshold.
