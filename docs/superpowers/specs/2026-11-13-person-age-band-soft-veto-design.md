# Age band as a soft veto on person guesses (no gender)

Status: PROPOSAL (perception of people; CLAUDE.md proposal mode). Not implemented.
Date: 2026-11-13. Builds on the walkway person learning spec
(`2026-10-20-walkway-person-learning-design.md`, PR #2767), the one-record
spec (`2026-10-22-people-one-record-design.md`) and the nightly
`make eval-person-recognition`.

## Arsonist summary

Juniper asked for a cheap age and gender check so Orion stops mixing people up.
I tried three cheap ways to do it on the pictures Orion already keeps. I scored
each way against the ages and genders Juniper herself told Orion (`person_fact`).

- **Age works, and it needs no new model.** Orion already keeps a numeric
  fingerprint of every person picture (the SigLIP vector that the camera
  clustering uses). That fingerprint can be compared with the phrases "a photo
  of a child" and "a photo of an adult". This gets the child/adult band right on
  83% of the pictures Juniper has named, and on 89% of the pictures where the read is
  confident (one side at 0.75 or more). For a whole named cluster it is right 91% of the time. It
  costs 0.3 ms per picture on CPU. It works on all 16,052 person pictures,
  not only on the 1.5% that show a face.
- **A face-based age reader (InsightFace) fails here.** It never read anyone as
  younger than 22. It read every child (7 to 9 years old) as an adult.
- **A body-based age reader (MiVOLO) is worse than the fingerprint.** It can only see
  the small 160 px thumbnails Orion keeps. It read 11% of children's pictures as
  children. On the same 593 pictures, the fingerprint was right on 78% and MiVOLO on 57%. It
  is also roughly 1,000x slower, it needs its own container, and its weights
  are non-commercial.
- **Gender must not be built.** On body pictures, the fingerprint read gets women
  and girls right only 58% of the time (men and boys 95%). For one girl it was
  right 29% of the time. That is misgendering, not noise. It is the same failure that got the
  old affect model removed (it misgendered Juniper, 2026-08-26). Do not add it, not even
  as a hidden signal.
- **What age buys is small but clean.** The nightly eval holds out each named cluster
  and checks whether Orion would guess the right person. In a replay of the latest eval
  run, an age veto struck 8 of its 99 wrong guesses (12% of the 69 it could score) and 0 of its 77 right guesses.
  On Juniper's "same person" answers, it would have wrongly split 4 of 104 pairs (3.8%). That
  is the real cost. It does not change the eval's gate ("no wrong confident names"),
  because Orion makes no confident names yet. As a check for mixed clusters, it catches the
  one case it is built for (a child's cluster holding an adult's picture). Beyond that, it mostly
  flags big clusters, so it is weak evidence.

Recommendation: **GO, small: an age-band veto that is computed when a guess is made,
from the stored fingerprint. It is off on cameras with a known regular (the office),
and it is used only for people whose age Juniper gave. NO-GO on gender. NO-GO on a new model
or service.**

## Current architecture

- **Crops.** `vision_crop_observation` (orion-sql-writer) holds every detected
  person crop. Each crop has these parts:
  - `embedding`: the SigLIP vector, 1152-d, L2-normalised (`runner.py`
    `get_image_features`, then normalised in `crop_embeddings.py`). The vision host
    tries `google/siglip2-so400m-patch14-384` first (`runner.py:253`). That model has
    never loaded, so every vector since 2026-10-16 says
    `embedding_model_id = google/siglip-so400m-patch14-384`, which is the fallback.
    4,990 earlier vectors (2026-10-15/16) have a NULL model id. Only 4 named album crops are among them.
  - An optional face vector and face thumbnail (MTCNN + InceptionResnetV1). Live counts of face vectors:
    walkway 438, doorbell 113, office (`cam0`) 28.
  - An OSNet shadow vector.
  - `thumb_ref`: a 160 px JPEG, rate-limited. Only 2,946 of 16,354 person crops have one.
  - Person crops are never pruned (`VISION_CROP_RETENTION_KEEP_LABELS=person`).
- **Clusters.** `vision_individual` is one appearance cluster. Splits
  (`vision_individual_split`) move crops off a mixed cluster after a
  "someone else" picture verdict.
- **Albums and guesses.** `services/orion-sql-writer/app/vision_person_model.py`
  builds albums from Juniper's answers (`vision_person_exemplar`). It makes
  one guess per unnamed cluster in `attribute()`, and `write_attributions()` upserts the guess into
  `vision_person_attribution`. `attribute()` takes `blocked` (cluster -> names
  Juniper said it is not). That set is applied before the camera-prior pick
  (`ruled_out`, lines 375-377), so a blocked name also loses its prior. Today
  every guess is "unsure". No accept bar has armed yet (expected around 12-08).
- **Ground truth Juniper gave.** `person_fact`: 14 `age` rows for 13 people
  (one row is a duplicate), plus `gender` / `pronouns` rows.
- **Existing age/gender estimators: none.** The repo search
  (`graphify query`, then `rg -i "genderage|insightface|mivolo|fairface"`)
  finds no age or gender model. The one related rule points the other way: the
  affect reader must never state gender
  (`services/orion-juniper-affective-state/evals/test_vision_affect_quality_eval.py::test_model_never_asserts_gender_or_identity`).

## What I measured (offline, read-only, 2026-11-13)

Data: a CSV export of these tables:
- `vision_crop_observation` (person crops with a vector);
- `vision_individual`;
- `vision_person_exemplar`;
- `person` / `person_link`;
- `person_fact`;
- `vision_individual_split`;
- `vision_individual_pair_verdict`;
- the latest `vision_person_model_eval` row.

There were no writes and no deploys. The models ran on CPU in a throwaway container from the
vision-host image. The crop thumbnails were mounted read-only. The host load average was about
200 on 96 cores, so read the speed numbers as upper bounds.

Truth: the age band is child when under 13, else adult. It comes from Juniper's `age` answers
(13 people, 2,595 album crops). Gender comes from her `gender` answer, or else from
pronouns (3,015 album crops). People without an answer are not scored. "Confident"
means one side at 0.75 or more, everywhere below unless a different value is given.

### 1. Does it return anything (coverage)

| Method | Input | Crops it can read | Per crop (CPU) |
|---|---|---|---|
| SigLIP zero-shot on the stored vector | vector already in Postgres | 16,052 / 16,052 (100%) | 0.3 ms |
| InsightFace `genderage` (buffalo_s) | face thumbnail | 238 (1.5%) | 24 ms |
| MiVOLO v2 (body, + face when present) | 160 px body thumbnail | 2,946 (18%); only 63 also had a face | 243 ms idle, 507 ms under load |

Full frames are not kept (a 5-minute ring). A model that needs pixels can only see
the small stored thumbnails. The SigLIP vector was made from the full-resolution
crop, so it carries more than the thumbnail does.

### 2. Is it right (against Juniper's answers)

| Method | Age band, per crop | Age band, confident reads | Age band, per named cluster (3+ crops) | Gender, per crop |
|---|---|---|---|---|
| SigLIP zero-shot | 2,163 / 2,595 = 83% | 1,824 / 2,044 = 89% | 83 / 91 = 91% | 2,198 / 3,015 = 73% |
| InsightFace | 16 / 78 = 21% (no child ever read as a child) | same | 3 / 16 | 35 / 96 = 36% |
| MiVOLO v2 | 336 / 593 = 57% (children: 32 / 289 = 11%) | 318 / 563 | 27 / 56 = 48% | 477 / 726 = 66% (female 54%) |
| Linear probe on the stored vector, leave-one-person-out | 59.5% | | | not run |

On the 593 album crops both could read, SigLIP got 462 right (78%) and MiVOLO 336 (57%).

SigLIP split by truth: children 80%, adults 88%. Two people pull the number down.
One girl reads as a child only 62% of the time. One adult (a delivery worker in a
uniform) reads as an adult only 64% of the time. Every other person with 20+ crops
scores 92-100%. Of the 2,595 scored crops, 2,591 carry the pinned SigLIP v1 id. The
NULL-id rows are not a factor.

SigLIP gender split: male 95%, female 58%. Four women and girls have 20+ crops. Their
per-person scores are 29%, 69%, 72% and 74%. That pattern is misgendering, not noise.

I also trained a linear probe on Juniper's labels, leaving one person out each time. It did worse
than the zero-shot text read (59.5%). With 13 people, the probe learns who the person is, not
how old they are. So there is nothing to train. The text prompts are the whole model.

Office camera (`cam0`, infrared at night): 360 of 479 crops read confidently as adults and 11 confidently
as children. The camera's regular is an adult with no `age` answer, so this is not scored. It is UNVERIFIED.

### 3. Within-cluster agreement and the known mixed clusters

There are 24 known mixed clusters, found from Juniper's answers: clusters that were split, plus
clusters whose album crops name two people. There are 187 pure clusters: named, one person, never split.
A cluster is scorable when it has 2+ crops with a read: 19 of the 24 mixed clusters and 134 of
the 187 pure clusters. Flag a cluster when it holds at least one child read and one adult read,
each at confidence 0.9:

| Cluster size | Mixed flagged | Pure flagged |
|---|---|---|
| 2-9 crops | 1 / 4 | 1 / 62 |
| 10-39 | 1 / 5 | 6 / 51 |
| 40+ | 5 / 10 | 5 / 21 |
| total | 7 / 19 | 12 / 134 |

The motivating cluster is one of the 24, and it is caught. It held a boy's crops and one adult's crop.
20 crops read 0.98-1.0 child, and the adult crop read 0.03. Juniper's answers cannot confirm it
is age-mixed, because the adult in it has no `age` answer. By her answers, none of the 24
is provably mixed by age band. So the other 6 mixed flags are not proven age catches. They
follow cluster size: a big cluster has more stray reads. The flag is good for exactly one
shape of mistake (a child's cluster that holds an adult's crop, or the reverse). It is not a
general purity detector.

### 4. Would a veto change the nightly eval

**Replay on the latest stored eval (2026-11-12, `median-topk-street-v2-people-split-face90`).**
That run held out 200 named clusters, one at a time:
- 176 got an "unsure" guess: 77 named the right person and 99 the wrong one.
- 24 had no album to guess from.
- 0 got a confident name (an accept).

For each guess, I asked one question. Is the mean read of the cluster's own crops confident
(0.75 or more on one side, 2+ crops), and does it disagree with the guessed person's age band
from Juniper's answer? Only guesses for people with an `age` answer can be scored: 69 wrong and 52 right.

| Veto | Wrong top guesses struck | Right top guesses struck |
|---|---|---|
| SigLIP age band | 8 / 69 scorable (8 / 99 overall) | **0 / 52** |
| SigLIP apparent gender (not proposed) | 30 / 69 | 2 / 52 |
| MiVOLO age (not proposed) | 28 / 42 | 4 / 34 |
| InsightFace age (not proposed) | 5 / 10 | 5 / 7 |

The age veto is precise but small. It does not move the gate (0 wrong accepts before and
after), because nothing is accepted yet. MiVOLO strikes more wrong guesses only because it calls
almost every child an adult, so it also strikes right guesses.

**Full rerun with the veto inside the eval loop:** in progress at the time of writing (the eval
takes about 100 minutes). It re-runs the real eval read-only, with the veto injected, for both arms.
This section will be updated with its numbers.

This replay calls `attribute()` the way the eval does: with no camera priors and an empty
`blocked`. So it says nothing about the office camera's prior path (see the design below).

### 5. Pair verdicts

`vision_individual_pair_verdict`, cluster means, 2+ crops each:
- Of the pairs Juniper called "same", a confident age mismatch would wrongly split 4 of 104 (3.8%).
- Of the pairs she called "different", it catches 6 of 39.
- A gender mismatch would wrongly split 8 of 104.

This 3.8% is the honest false-veto rate. The 0 of 52 above is the better case.

## Licences

- SigLIP (`google/siglip-so400m-patch14-384`): Apache-2.0, already deployed.
- InsightFace model packs: non-commercial research use only (code is MIT).
- MiVOLO v2 weights (`iitolstykh/mivolo_v2`): CC BY-NC 4.0. Code Apache-2.0.
  It needs `timm==0.8.13.dev0` and `transformers==4.51.0`. The vision host pins
  `timm==1.0.9` and `transformers==4.49.0`, so MiVOLO would need its own
  container.
- FairFace: CC BY 4.0, but face-only. Faces are 1.5% of crops, and the face path
  fails on children here, so I did not run it.

## Proposed design (age band only)

### Producer: a pure function

`orion/vision/age_band.py` (new, about 80 lines):

- `AGE_BAND_PROMPTS`: the six prompt strings, three for child and three for adult.
- `AGE_BAND_MODEL_ID = "google/siglip-so400m-patch14-384"`.
- The text vectors are committed once in `orion/vision/age_band_text_vectors.json`:
  two re-normalised mean vectors of 2 x 1152 floats, plus `logit_scale`. A one-off
  script under `scripts/` regenerates the file. A test pins the file to the model id and the prompt hash.
  The sql-writer never loads a model.
- `p_child(vec, model_id) -> Optional[float]` is pure numpy. It computes the sigmoid of
  `logit_scale * (cos(child) - cos(adult))`. The bias cancels in a two-way comparison.
  It returns `None` unless `model_id == AGE_BAND_MODEL_ID`. That covers two cases: if SigLIP2 ever
  loads, the vectors move to a different space, and NULL-id rows are skipped.
- `cluster_age_band(reads, conf=0.75, min_n=2) -> Optional[str]` returns `"child"`,
  `"adult"` or `None`. It returns `None` when there are fewer than `min_n` reads or the mean is
  between 0.25 and 0.75. "Unknown" is a normal answer. The defaults are the replayed
  values. Raising `min_n` to 3 needs its own replay first.

Nothing is written per crop or per person. The read is recomputed from the stored
vector each time a guess runs (0.3 ms per crop).

### Consumer 1: guess veto in `attribute()`

- **New arguments, not `blocked`.** `attribute(..., person_bands: Dict[str, str] = None,
  cluster_bands: Dict[str, str] = None)`. `person_bands` maps a person to `child`/`adult`
  and comes only from Juniper's `person_fact` age answers. There is no album-consensus fallback: the
  replay did not test one, and comparing two reads from the same fingerprint
  gives correlated errors. `cluster_bands` maps a cluster to its band, from
  `cluster_age_band`. Both are loaded by one helper, `load_age_bands(conn, clusters)`. The live tick and the eval
  both call that helper, so they share the same code.
- **Rule.** A person whose band disagrees with the cluster's band is left out of
  `scored` for that cluster only. This happens after the prior pick, not through `ruled_out`.
- **Prior cameras are exempt.** On a stream with a camera prior (today the office, `cam0`),
  the veto does nothing. The prior path was never replayed, infrared crops are unmeasured,
  and `basis='prior'` guesses feed office presence lines
  (`orion/situational/perception_reader.py:1252`) and confirm asks. 73 live guesses rest on the prior.
- The veto can only remove a candidate. It never names one.
- Flag: `VISION_PERSON_AGE_VETO_ENABLED=true` in the orion-sql-writer `.env_example`,
  `settings.py` and compose. It is on by default, per the ship-flags-on rule.

### Consumer 2: one purity nudge, not a splitter

This lives in the individuals reducer's existing split-ask path. A named cluster with one child read and one
adult read (confidence 0.9) becomes a candidate for the existing "Is this the same
person?" picture card. The card shows the two crops that disagree. It never splits on
its own, and it stays under the card's existing daily cap. Ship it second, and only if Juniper wants
more picture cards.

### Schema change and trace

- One additive column: `services/orion-sql-db/manual_migration_vision_person_age_veto_v1.sql`
  holds `ALTER TABLE vision_person_attribution ADD COLUMN IF NOT EXISTS age_veto jsonb`, with
  `-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer`. The statement is on the post-pull auto-apply
  allowlist, the same pattern as `manual_migration_vision_person_basis_v1.sql`. There is a matching
  `_rollback.sql` (`DROP COLUMN`, marked `-- DESTRUCTIVE`).
- The column holds `{"struck": [names], "rule": "age_band_v1"}` and nothing else. The age
  read and the cluster band are not stored, so no stored age estimate exists for a passerby.
  A struck name still implies a band for that cluster. That is the one thing kept, and the
  retention rule below limits it.
- `write_attributions()` sets `age_veto = EXCLUDED.age_veto` on upsert, NULL included, so a
  stale strike never stays on a rewritten row. The writer checks that the column exists first
  (an `attribution_has_age_veto` guard, like `attribution_has_basis`), so a deploy before
  the migration does not break guesses.
- If the veto strikes every candidate, no attribution row is written
  (`if not scored: continue`). That case is counted as `age_veto_all_struck` in the
  `run_person_model` stats, together with `age_vetoes`, and it is logged with the cluster id.
- The nightly eval runs both arms in one pass (veto off and veto on) and reports
  `age_vetoes`, wrong and right guesses struck, and `unsure_right_person` for each arm.
  Its existing McNemar compares only `correct_accept`, so it is empty while there are 0 accepts.

### Privacy, retention, display

- Passersby are third parties, and some are children. No per-crop, per-cluster or
  per-person age read is stored. The only thing stored is which names a veto struck, on a
  guess row.
- Retention: when a guess row is revoked, `age_veto` is set to NULL in the same statement.
  `vision_person_attribution` keeps revoked rows forever today (6,519 rows, 4,739
  revoked), so without this the strike would be kept forever.
- Display: the age band is never shown as a fact about a person. It is not shown on
  People pages, ask cards, chat, day letters or the graph. No Hub page reads
  `vision_person_attribution` today. The debug surface is the column itself, the stats
  counters and the eval metric. Nothing new is added to the UI.
- No gender, apparent or otherwise, is estimated, stored or displayed. Gender stays
  what Juniper or the person said (`person_fact`).

## Non-goals

- No apparent-gender estimate in any form (see the measurements).
- No new model, service, GPU lease or container.
- No identity labels from age. A veto removes a candidate. It never names anyone.
- No automatic splits.
- No age read on prior cameras (the office).
- No album-consensus band for people without an `age` answer.
- Not fixing cross-day body matching. That is the real reason no accept bar has armed.

## Acceptance checks

1. Unit: `p_child` on synthetic vectors built from the committed text vectors
   gives the expected side. No real crop vectors are committed: they belong to passersby and children.
2. Unit: `p_child` returns `None` for a NULL or different `model_id`.
3. Unit: `cluster_age_band` returns `None` for 1 read, for 2 reads with a mean between 0.25 and 0.75,
   and a band for 2 confident reads.
4. Unit: `attribute()` with the veto never adds a name. It only removes candidates. A cluster
   whose band is `None` gets the same guess with and without the veto. On a stream with
   a prior, the guess (including `basis`) is the same with and without the veto.
5. Unit: the upsert resets `age_veto` to NULL when a later cycle strikes nothing. A revoke nulls it.
   The writer works when the column is absent.
6. Eval: in one pass, the veto-on arm has 0 wrong accepts, and its `unsure_right_person` is not
   lower than the veto-off arm's.
7. Live (after deploy): at least one non-null `age_veto` row within 7 days. A manual look
   at the struck crops confirms the strike was right.
8. Grep gate: no `gender` key appears in `orion/vision/age_band.py`, the guess trace, or the
   purity-nudge card payload.

## Failure modes and rollback

- Dangerous failure: the veto strikes the true person, so Orion keeps failing to recognise
  someone. The measured rate is 3.8% of "same person" pairs, and 0 of 52 right guesses in the replay. It is
  bounded: the veto needs a confident cluster mean over 2+ crops, and only people with a stated
  age can be struck. Check 6 catches a regression on the next night.
- A child who grows past 13 keeps the old `age` answer. An `age` answer goes stale, and the
  veto will strike wrongly once it does. Mitigation: use only `age` answers newer than 2 years.
  Check 7 is the live check.
- Tall children and teenagers sit between the bands. The band has only two sides, with a wide "unknown"
  middle, so ambiguous clusters get no veto.
- Rollback: set `VISION_PERSON_AGE_VETO_ENABLED=false` and restart orion-sql-writer. To remove
  the stored strikes as well, apply the `_rollback.sql`.

## Recommended next patch

One PR in orion-sql-writer plus `orion/vision/age_band.py`. It contains:
- the pure function and the committed text vectors;
- the `attribute()` veto behind the flag, with the prior-camera exemption;
- the `age_veto` column, the upsert and revoke handling;
- the eval with both arms;
- tests 1-5.

The purity nudge (consumer 2) is a second, smaller PR, and only if Juniper wants it.
