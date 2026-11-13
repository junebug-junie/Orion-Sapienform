# The body-match line for walkway individuals (proposal)

Date: 2026-10-16. Status: proposal (design only). Identity change: proposal
mode per AGENTS.md section 0A. Nothing here is built yet.

Companion to `2026-10-16-vision-individual-split-proposal.md` (PR #2704) and
the split-on-verdict patch (PR #2709, merged, not yet deployed). Names are
placeholders: name A, B, C stand for labels Juniper typed. Individuals are
referred to by id prefix.

## Arsonist summary

Orion decides "is this the same person as before?" mostly from a picture of
the whole body. Two people are treated as one when their body pictures score
0.80 or more on a 0-to-1 likeness scale. That line is too low for this
camera, and its shape is not the main problem. Its height is.

What the replay of today's live data shows:

1. **Individual 62cb5fcc is two people, not a crowd.** About 690 of its 758
   crops are one child sitting in a chair at one spot for 40 minutes (name C;
   Juniper's two "someone else" pictures are in this group). The other crops
   are name A, who walked past earlier and whose face gave the individual its
   name. The body line at 0.80 glued the chair-sitter onto name A.
2. **A running average of several people looks like everyone.** The
   individual's body vector is the average of its members. Different people
   on this camera score 0.71 to 0.75 against each other. The average of a
   mix of such people scores about 0.84 to 0.87 against a typical new person
   (square root of the pairwise score; measured 0.939 against 0.938
   predicted). That is above 0.80, so a mixed individual pulls in more
   people. Below about 0.87 the line cannot stop this.
3. **Other line shapes do not help.** Per-zone and per-size lines: the
   different-person scores are the same in every zone and at every box size.
   A margin over the runner-up shatters every passing into pieces. Track
   continuity alone does not separate children walking together. Centroid
   guards (frozen, sliding window, exemplars) change little.
4. **What does help:** two lanes. A crop continuing a person seen in the
   last 10 seconds, close by, joins at 0.88. Recognising someone after a gap
   needs 0.90 (or a face). On the replay this halves the merging of people
   seen together (13.6% to 6.3%), and passings break apart slightly less
   often than today (15.6% vs 17.6%).
5. **The patch only changes new crops.** 62cb5fcc's stored body vector
   already points at the chair-sitter: 96% of her crops would still join it
   at 0.90. So the same patch must also re-anchor each named individual's
   body vector on its own face crops (then 1% would join). That is a small
   production write (4 rows) and needs Juniper's yes.
6. **The 0.151 face pair is a real face of the right child, seen in
   profile.** It is not a wrong face. The face model scores a frontal and a
   profile view of one child at 0.15. See "The 0.151 face pair".

## Current architecture

- **Matcher:** `services/orion-sql-writer/app/vision_individuals.py`
  `apply_batch`, run every 60 s by `vision_individuals_loop.py`.
  1. A crop with a face joins the nearest face average at >= 0.60
     (`VISION_INDIVIDUALS_FACE_MATCH_THRESHOLD`).
  2. Otherwise it joins the nearest body average at >= 0.80
     (`VISION_INDIVIDUALS_MATCH_THRESHOLD`), with no limit on time or place.
  3. Otherwise it opens a new individual.
  4. Two crops from one frame never join one individual.
  5. The body and face averages are running means of unit vectors.
- **Body vector:** orion-vision-host `crop_embeddings.py`, through the
  `embed_image` profile. The profile asks for SigLIP2
  (`google/siglip2-so400m-patch14-384`), which does not load under the
  pinned transformers 4.49.0. The live log says so on every start
  (`[MODEL] embedder load failed ... falling back
  id=google/siglip-so400m-patch14-384`, 2026-10-15 21:09 UTC). So the live
  body vector is **SigLIP v1**, a general picture-and-caption model, not a
  person re-identification model. It squashes the tall box crop into a
  384 x 384 square. 1152 numbers per crop.
- **Face vector:** MTCNN finds the most confident face inside the person
  box (probability >= 0.95, short side >= 40 px), and InceptionResnetV1
  (`facenet-pytorch`, VGGFace2 weights) turns it into 512 numbers.
- **Camera:** walkway frames at 2560 x 1920. People's boxes are 161 to 353
  px tall (p10 to p90). Frames with people come about 2.5 s apart at best
  (median gap 2.6 s).
- **Membership:** PR #2709 adds `vision_crop_observation.individual_id`.
  Its migration is not applied yet, so membership still comes from replay.

## Measurement method (read-only)

- **Data:** read-only `COPY` export at 2026-10-16 03:11 UTC: 4,988 walkway
  person crops with a body vector (155 with a face), 2026-10-15 04:57 to
  2026-10-16 01:53 UTC. All crops since the walkway started, so a replay
  from empty starts from the same state as live.
- **Replay:** the matcher rule in 60 s landing-time batches, with the
  same-frame rule. The baseline rebuilds the live individuals: 40 person
  individuals, with sizes 1166, 758, 738 (same as live).
- **Scripts:** in the session scratchpad only, not committed (design-only
  patch). The recommended patch turns them into an eval (below).

### What the scores mean

There are no person labels for most crops, so purity and fragmentation use
proxies built from the camera's own geometry:

- **Track pair.** A crop and the crop in the next frame (<= 3 s later)
  whose box moved at most one box height and kept its size, matched both
  ways with no close second candidate. Proxy for "same person": 2,538 pairs.
- **Tracklet.** A chain of track pairs. Two tracklets that share a frame
  are two different people: 1,561 such pairs.
- **Mixed (purity, lower is better).** Share of those different-people
  tracklet pairs that end up sharing an individual anywhere.
- **Passing split (fragmentation, lower is better).** Share of track pairs
  that end up in different individuals.
- **Re-seen (coverage).** Share of crops in an individual seen on two or
  more visits more than 10 minutes apart. Today's 96% is mostly the blobs,
  so this is a ceiling, not a goal.
- **Verdict breaks.** Juniper's six answered pictures as labelled crops.
  "Must" pairs: name B's two pictures, and the three name C pictures with
  each other (4 pairs). "Cannot" pairs: pictures with different names
  (11 pairs). A break is a must pair split or a cannot pair joined.

The "mixed" score only sees people who were in one frame together. It
cannot see two people merged across different visits. That error is
estimated from the score distributions below, not counted.

### Body scores: same person vs different people

| Pair kind | n | p10 | p50 | p90 | share >= 0.80 |
|---|---|---|---|---|---|
| Same person, next frame (track pair) | 2,538 | 0.848 | 0.928 | 0.967 | 97% |
| Different people, same frame | 2,910 | 0.673 | 0.752 | 0.832 | 20% |

Line height vs the two kinds:

| Line | Same person kept | Different people let through |
|---|---|---|
| 0.80 | 97% | 20.3% |
| 0.85 | 90% | 6.7% |
| 0.88 | 81% | 3.1% |
| 0.90 | 71% | 1.8% |
| 0.92 | 57% | 0.7% |

By zone and by box size the different-people pass rate at 0.80 is
patio 19%, walkway 21%, mailbox 26%; boxes under 150 px 16%, 150-250 px 20%,
250-400 px 22%, over 400 px 31%. No zone or size has a different curve,
so per-zone or per-size lines have nothing to work with. They were not
replayed.

## Replay results

Same data, same scores, one line per rule.

| Rule | Individuals | With >= 5 crops | Mixed | Passing split | Re-seen | Must breaks | Cannot breaks |
|---|---|---|---|---|---|---|---|
| **Today: body 0.80** | 40 | 27 | 13.6% | 17.6% | 96% | 2/4 | 4/11 |
| Body 0.85 | 131 | 54 | 9.5% | 24.5% | 64% | 2/4 | 1/11 |
| Body 0.88 | 323 | 81 | 6.3% | 30.1% | 59% | 2/4 | 1/11 |
| Body 0.90 | 605 | 84 | 4.8% | 35.5% | 53% | 3/4 | 1/11 |
| Body 0.92 | 1,085 | 100 | 2.7% | 44.9% | 45% | 4/4 | 1/11 |
| Margin 0.02 over runner-up | 2,745 | 54 | 1.3% | 77.9% | 35% | 3/4 | 1/11 |
| Margin 0.05 over runner-up | 3,513 | 17 | 0.4% | 92.0% | 26% | 4/4 | 1/11 |
| Average frozen after 20 crops | 45 | 31 | 16.1% | 26.9% | 87% | 2/4 | 1/11 |
| Average of last 20 crops | 36 | 24 | 11.8% | 16.1% | 80% | 2/4 | 4/11 |
| Nearest of last 20 crops | 29 | 19 | 12.2% | 16.8% | 98% | 2/4 | 1/11 |
| Body 0.90 + average of last 20 | 589 | 87 | 4.5% | 37.4% | 53% | 3/4 | 1/11 |
| Body only within 30 s; after a gap, face only | 522 | 147 | 17.3% | 11.5% | 5% | 3/4 | 1/11 |
| Within 10 s and 1.5 box heights at 0.80; after that 0.90 | 258 | 81 | 16.2% | 11.2% | 61% | 4/4 | 0/11 |
| Within 10 s and 1.5 box heights at 0.85; after that 0.90 | 361 | 80 | 9.2% | 10.5% | 65% | 1/4 | 3/11 |
| **Within 10 s and 1.5 box heights at 0.88; after that 0.90 (recommended)** | **493** | **88** | **6.3%** | **15.6%** | **55%** | **3/4** | **1/11** |
| Face only, no body matching | 4,872 | 8 | 0.3% | 98.3% | 2% | 4/4 | 1/11 |

How to read it:

- **Height matters most.** Raising the line from 0.80 to 0.90 cuts mixing
  by about two thirds, but doubles passing splits.
- **Margin** gives the best purity but turns a passing into many pieces.
  Children in similar clothes are always a close runner-up.
- **Track continuity alone makes mixing worse** (17%). Children walking
  together are each other's only nearby candidates, so the lane hands one
  child's crop to the other.
- **Centroid guards** barely move anything at 0.80. Above about 0.87 the
  running average stops attracting new people, so the guard has nothing
  left to fix (4.5% vs 4.8% at 0.90).
- **The recommended two-lane rule** keeps the purity of a flat 0.88 line
  (6.3%). It splits passings less than today (15.6% vs 17.6%), because a
  close, recent person is not stolen by an old individual with a better
  score.
- **The one cannot break that never goes away** (name A vs name C's face
  picture on name B's card) is a face match at 0.646. It is the face line's
  error, not the body line's. The PR #2709 cannot-link rule may catch it.
- **The must breaks** are face-line problems too: name B's two pictures
  score 0.151 on the face (profile, below), and name C's face picture
  joins name A by face.

### What the recommended rule does to 62cb5fcc and f5242d40

Replayed from empty with the recommended rule:

- **62cb5fcc "name A" (758 crops)** becomes 30 pieces:
  - Name A's piece has 20 crops, 19 of them with a face. 15 of these came
    from 62cb5fcc. The name now rides on face evidence.
  - 687 crops go to one individual (699 crops in total). It is one child in
    a chair at about x=1820, y=900 from 22:07 to 22:54 UTC. Juniper's two
    "someone else" pictures are in it. That is name C, as one person.
  - 20 crops become single-crop individuals.
- **f5242d40 "name B" (297 crops, six visits)** becomes 71 pieces:
  - Name B's piece has 13 crops (6 with a face).
  - 15 pieces have 5 or more crops, and 41 are single crops.
  - Name B's profile picture lands in another piece of 8 crops (face 0.151).
  - Name C's face picture still joins name A's piece through the face line.

With a flat 0.90 line, the result is close: name A's piece has 21 crops
(20 with a face), and the chair-sitter is one individual of 643.

**Forward-only caveat.** The live 62cb5fcc body average was built from the
mix. 96% of the chair-sitter's crops score >= 0.90 against it, so any new
crops like hers would still join "name A". An average built from 62cb5fcc's
25 face crops only lets 1% of them through. For f5242d40 the figures are
95% and 35%. Hence the re-anchor step in the patch.

## The 0.151 face pair (open question 1 from the split proposal)

**Verdict: the face crop is a real face of the right child. The low score
is the pose, not a wrong face.**

- The cards on ask 64490457 showed the **face** thumbnails
  (`image_ref` = the crop's `face_thumb_ref`; the body `thumb_ref` was empty
  because body thumbnails are rate-limited). So Juniper judged the same
  pixels that made the face vector.
- Picture 0 (crop `9cc76180...:2`): a frontal face, about 55 x 66 px, MTCNN
  probability 0.991.
- Picture 1 (crop `8035f96e...:0`): the same child's head turned down and
  away, close to a profile, about 45 x 67 px, near the 40 px floor. MTCNN
  probability 0.951.
- Other face crops in f5242d40 back this up. Three more face thumbnails
  were checked by eye; they look like the same child:
  - A three-quarter view scores 0.55 against picture 1 and 0.16 against
    picture 0.
  - A head-down view scores 0.10 against picture 1 and 0.66 against
    picture 0.
  - A frontal view scores 0.51 against picture 1 and 0.45 against picture 0.
- Picture 1's best match in the whole walkway set is 0.727, a crop in the
  same individual. Inside f5242d40 (35 face crops, and it is itself
  mixed), face pairs score p10 0.145, p50 0.405, p90 0.589.
- Body score of the two pictures: 0.825.

What this means: VGGFace2 InceptionResnetV1 on 45-60 px faces gives one
child anywhere from 0.10 to 0.73 depending on head angle. The 0.60 face line
misses most same-child pairs across poses. Those crops then fall through to
the body line, which is how the body line gets so much work. Raising or
lowering the face line will not fix pose. A pose gate could fix it
(MTCNN already returns the five landmarks), but that is not in this patch.

Images were read in place from `/mnt/telemetry/orion-vision-host/crop_thumbs/`.
Nothing was copied. `face_frames/` holds cam0 only, so it played no part here.

## A better body vector (not first)

- **What is used:** SigLIP v1 so400m, a general model. It was trained to
  match pictures with captions, not to tell two people apart. Its
  same-frame different-people median is 0.75, and its next-frame
  same-person p10 is 0.85: a gap of only 0.10.
- **Obvious candidate:** OSNet (`osnet_x1_0`, person re-identification,
  about 2.2 M parameters, about 9 MB of weights). The vision-host container
  already has torch 2.2.2+cu121 and timm. `torchreid` 0.2.5 is on PyPI. The
  T10 shows 11,991 / 16,384 MiB used (about 4.3 GB free). OSNet at 256 x 128
  in batches of 8 should need well under 0.5 GB. **UNVERIFIED:** not loaded
  or measured here.
- **Why it cannot be scored today:** there are no stored frames, and only
  390 of 4,988 person crops have a body thumbnail (rate-limited). Not one
  track pair and not one same-frame pair has thumbnails on both sides. So
  no labelled pair exists to score OSNet against SigLIP offline.
- **Path:** a later patch stores an OSNet vector next to the SigLIP one
  (shadow, not used for matching) for a week. Then this same replay is
  re-run on OSNet vectors. Separately, the SigLIP2 load failure is a real
  bug: the profile asks for one model and silently runs another.

## Proposal (proposal mode)

### What capability changes

The body line becomes two lanes:

- **Continuation lane.** If the crop sits within
  `VISION_INDIVIDUALS_CONTINUATION_MAX_DISP` (1.5) box heights of an
  individual's last box, and that individual was seen within
  `VISION_INDIVIDUALS_CONTINUATION_GAP_SEC` (10 s), the crop joins the
  best such individual at >= `VISION_INDIVIDUALS_CONTINUATION_THRESHOLD`
  (0.88). Only those individuals are candidates.
- **Re-identification lane.** Otherwise, the crop joins the best of all
  other individuals at >= `VISION_INDIVIDUALS_MATCH_THRESHOLD`, moved
  0.80 -> 0.90.
- **Face.** The face lane is unchanged and still runs first.
- **Same frame.** The same-frame rule is unchanged.

Theory anchor for 0.90: a running average of people whose pairwise score
is p scores about sqrt(p) against its members. Measured p is 0.71 to 0.75,
so sqrt(p) is 0.84 to 0.87. The re-identification line must sit above that,
or mixed individuals keep growing. 0.90 clears it with margin.

### Re-anchor (one-shot)

`scripts/reanchor_vision_individual_centroids.py`. For each **labelled**
person individual, it rebuilds the body average from the crops that carry a
face, using the replay membership (exact today) or `individual_id` once
PR #2709's migration is applied. Then it writes `centroid` (keeping
`centroid_n` as the face-crop count). Today that is 4 rows. It runs dry by
default. `--yes` writes after a snapshot to `/tmp/vision-reanchor/`, under
the AGENTS.md section 14 protocol. Running it is a production write and
needs Juniper's yes.

### Data touched

- `vision_individual`: new individuals, and centroids, as today.
- Labelled rows: centroid only, through the re-anchor.

No new pictures, no new vectors, no new external call, and nothing leaves
`conjourney`.

### Privacy boundary

There is no new boundary. Body vectors and face vectors already live in
`conjourney`. The re-anchor snapshot holds 4 centroids under `/tmp`. Delete
it after 7 days (it is biometric-adjacent).

### Trace that proves it worked

- A per-batch log line:
  `[VISION_MATCH] stream=walkway face=<n> continuation=<n> reid=<n> new=<n>`.
- A new `assigned_by_continuation` count on `BatchResult`, next to
  `assigned_by_face`.
- On live data, the replay eval prints the table row for the deployed rule
  and the baseline row.

### Dangerous failure modes

1. **Fragmentation hides people from asks and rhythm.** Individuals per day
   go from about 40 to about 500, most of them small. Ask gates (>= 10
   sightings, >= 5 distinct days) and rhythm minimums mean fewer subjects
   qualify, so Orion asks about fewer people.
   - This is the safe direction. Two pieces of one person can be joined
     later by a face or by Juniper's "same" answer. Two people in one
     piece cannot be separated without membership.
   - Guard: the eval prints individuals per day. If more than 1,500 person
     individuals appear in 24 h, the `[VISION_MATCH]` line warns.
2. **The re-anchor names the wrong side.** If a labelled individual's face
   crops are themselves mixed (f5242d40 holds name C's face picture), its
   rebuilt average leans toward the wrong child.
   - Guard: the re-anchor uses only face crops whose face score to the
     individual's face average is >= 0.60.
   - The dry run prints, for every labelled individual, how many of its
     crops would still pass 0.90 against the new average.
   - Juniper approves the dry run before `--yes`.

### Disable / roll back

- `VISION_INDIVIDUALS_CONTINUATION_ENABLED=false` and
  `VISION_INDIVIDUALS_MATCH_THRESHOLD=0.80` restore exactly today's rule.
  The replay eval asserts this: the baseline row is 40 individuals with top
  sizes 1166 / 758 / 738.
- The re-anchor writes the old centroid to the snapshot file.
  `--restore <snapshot>` puts it back.
- Ships ON, per standing practice.

## Missing questions

1. **Fragmentation vs mixing.** Does Juniper accept about 500 small person
   individuals per day, with fewer asks, in exchange for half the mixing?
   The alternative (flat 0.90) mixes a little less (4.8%) but splits
   passings twice as often.
2. **Re-anchor now?** Without it, the patch does not change 62cb5fcc's
   future. With it, 4 rows change.
3. **OSNet shadow vectors?** This is the only path to a body vector made
   for telling people apart. It costs about 0.5 GB on the T10 (UNVERIFIED)
   and one more array column.
4. **Side finding: the skeleton is not caught by the stationary rule.**
   Individual fefb24d7 (1,166 crops, the patio skeleton prop, thumbnails
   checked) has `stationary=false`. Its box centre moves 109 px (standard
   deviation) across the day. Probably the detector box swaps between the
   skeleton and the chair. That needs its own small fix.

## Proposed schema / API changes

- **No schema change**, no bus channel, and no migration.
- **New env keys** (orion-sql-writer `.env_example`, settings, compose,
  README, and local `.env` through the sync script):
  - `VISION_INDIVIDUALS_CONTINUATION_ENABLED=true`
  - `VISION_INDIVIDUALS_CONTINUATION_THRESHOLD=0.88`
  - `VISION_INDIVIDUALS_CONTINUATION_GAP_SEC=10`
  - `VISION_INDIVIDUALS_CONTINUATION_MAX_DISP=1.5`
- **Changed default:** `VISION_INDIVIDUALS_MATCH_THRESHOLD` 0.80 -> 0.90
  (its meaning narrows to "re-identification after a gap").
- **`apply_batch` signature:** add `continuation_threshold`,
  `continuation_gap_sec` and `continuation_max_disp` (None = off). The last
  box per individual comes from its latest or touched sighting
  (`last_box_xyxy`), which `apply_batch` already holds.

## Files likely to touch

- `services/orion-sql-writer/app/vision_individuals.py`: the continuation
  lane in `apply_batch`, the `[VISION_MATCH]` line, and
  `assigned_by_continuation`.
- `services/orion-sql-writer/app/vision_individuals_loop.py`: pass the new
  settings.
- `services/orion-sql-writer/app/settings.py`, `.env_example`,
  `docker-compose.yml`, `README.md`, and local `.env` (sync script).
- `services/orion-sql-writer/tests/test_vision_body_match.py`: gate tests.
- `services/orion-sql-writer/evals/run_vision_body_match_replay_eval.py`:
  this replay, driving the deployed `apply_batch`, in the same style as
  `run_vision_split_replay_eval.py`.
- `scripts/reanchor_vision_individual_centroids.py`.

## Non-goals

- No new body model in this patch (OSNet is a shadow-vector follow-up).
- No face-line change and no pose gate.
- No per-zone or per-size lines (the data shows no difference to exploit).
- No margin rule.
- No re-clustering of history beyond the 4-row re-anchor.
- No merging of fragments.
- No skeleton or stationary fix here.

## Acceptance checks

1. **Gate tests** (`pytest services/orion-sql-writer/tests/test_vision_body_match.py -q`):
   - A crop 0.89 from an individual seen 5 s ago, 1 box height away, joins
     it.
   - The same crop 20 s later, or 3 box heights away, opens a new
     individual.
   - A crop 0.91 from an individual seen an hour ago joins it.
   - With the continuation lane off and the line at 0.80, the result is
     byte-identical to today's `apply_batch` on a fixture batch.
   - The same-frame rule still holds inside the continuation lane.
2. **Replay eval** (read-only, live DB, while crops from 2026-10-15/16
   are retained):
   - The baseline row reproduces 40 / 13.6% / 17.6%.
   - The deployed row reproduces 493 / 6.3% / 15.6% (within 0.5 points).
   - Exit 0 = reproduced, 1 = a number moved, 2 = data pruned.
3. **Live, 24 h after deploy:**
   - `[VISION_MATCH]` lines show non-zero `continuation=` and `reid=`.
   - The eval run over the new 24 h prints the mixed score for live data.
     The live run must score at or below the baseline row computed on that
     same window.
4. **Re-anchor (only after Juniper says yes):**
   - The dry run prints 4 labelled rows.
   - For 62cb5fcc, at most 5% of the chair-sitter's crops pass 0.90
     against the new average (replay figure: 1%).
   - `--restore` brings back the snapshot byte-for-byte.

## Recommended next patch

1. In `apply_batch`, add a continuation lane: a crop near (1.5 box heights)
   an individual seen in the last 10 s joins it at 0.88.
2. Move the re-identification line from 0.80 to 0.90. Face first, same-frame
   rule, and everything else stay as they are.
3. Kill switch: `VISION_INDIVIDUALS_CONTINUATION_ENABLED=false` plus
   threshold 0.80 restores today's rule exactly. Ships ON.
4. Add an eval that replays live crops and prints mixed / split / count
   for the baseline and the deployed rule. On today's data the expected
   result is 13.6% -> 6.3% mixed and 17.6% -> 15.6% split.
5. Add a dry-run re-anchor script for the 4 labelled individuals. Juniper
   approves before it writes. This is what stops 62cb5fcc from pulling in
   the chair-sitter.
