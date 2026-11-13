# Age band as a soft veto on person guesses (no gender)

Status: PROPOSAL (perception of people; CLAUDE.md proposal mode). Not implemented.
Date: 2026-11-13. Builds on the walkway person learning spec
(`2026-10-20-walkway-person-learning-design.md`, PR #2767), the one-record
spec (`2026-10-22-people-one-record-design.md`) and the nightly
`make eval-person-recognition`.

## Arsonist summary

Juniper asked for a cheap age and gender check so Orion stops mixing people up.
I measured three cheap ways to do it on the crops Orion already stored, and
scored each one against the ages and genders Juniper herself gave Orion
(`person_fact`).

- **The cheapest way already works for age, and it needs no new model.**
  Every person crop already carries a SigLIP vector (the picture embedding
  the walkway reducer clusters on). Comparing that stored vector with the
  text "a photo of a child" against "a photo of an adult" gets the age band
  right on 83% of album crops (89% when the read is confident), and on 91% of
  named clusters. It costs 0.3 ms per crop on CPU, and it covers every person
  crop that has a vector (16,052 crops), not only the 2% that show a face.
- **The face model is useless here.** InsightFace's age/gender model never
  read anyone as younger than 22. It read every child (7 to 9 years old) as an
  adult. It also only runs on the 238 crops with a face (1.5%).
- **MiVOLO (face + body)** reads bodies without a face, but only from the
  160 px thumbnails Orion keeps. It read 11% of children's crops as children.
  On the same 593 crops, the stored-vector read got 78% right and MiVOLO got 57%. It is also
  about 1,000x slower, needs its own container (pins clash with the vision host), and its
  weights are non-commercial.
- **Gender must not be built.** On body crops, the stored-vector read gets women
  and girls right only 58% of the time (men and boys 95%). It reads women and
  girls as male much of the time. For one girl it was right 29% of the time.
  This is the same failure that got the old affect model removed (it
  misgendered Juniper, 2026-08-26). An apparent-gender read would add
  misgendering to Orion's perception and give little in return. Do not add it, not even as a hidden
  signal.
- **What age buys:** the age veto in the nightly eval
  (numbers in "Eval impact" below) strikes some wrong guesses and no right guesses.
  It does not change the gate today, because nothing is accepted yet. The
  impure-cluster check mostly tracks cluster size, so it is weak evidence.

Recommendation: **GO, small, for age band only, as a veto computed at decision
time from the stored vector. Store nothing new about any person. NO-GO on gender.
NO-GO on a new model or service.**

## Current architecture

- **Crops.** `vision_crop_observation` (orion-sql-writer) holds every detected
  person crop: SigLIP vector (`embedding`, 1152-d, L2-normalised,
  `google/siglip-so400m-patch14-384`, produced in
  `services/orion-vision-host/app/runner.py` `get_image_features`), optional
  face vector + face thumbnail (MTCNN + InceptionResnetV1, walkway only),
  OSNet shadow vector, `thumb_ref` (160 px JPEG, rate-limited: 2,946 of
  16,354 person crops have one). Person crops are never pruned
  (`VISION_CROP_RETENTION_KEEP_LABELS=person`).
- **Clusters.** `vision_individual` is one appearance cluster. Splits
  (`vision_individual_split`) move crops off a mixed cluster after a
  "someone else" picture verdict.
- **Albums and guesses.** `services/orion-sql-writer/app/vision_person_model.py`
  builds albums from Juniper's answers (`vision_person_exemplar`) and makes
  one guess per unnamed cluster in `attribute()`. `attribute()` already takes
  `blocked`: cluster -> names Juniper said it is not. Today the guesses are all
  "unsure": no accept bar has armed (expected around 12-08).
- **Ground truth Juniper gave.** `person_fact` holds `age` (14 rows) and
  `gender` / `pronouns` (27 rows) for the people she named, via the
  `person_fact` asks.
- **Existing age/gender estimators: none.** The repo search
  (`graphify query`, then `rg -i "genderage|insightface|mivolo|fairface"`)
  finds no age or gender model. The one related rule is the opposite: the
  affect reader must never state gender
  (`services/orion-juniper-affective-state/evals/test_vision_affect_quality_eval.py::test_model_never_asserts_gender_or_identity`).

## What I measured (offline, read-only, 2026-11-13)

Data: a CSV export of `vision_crop_observation` (person crops with a vector),
`vision_individual`, `vision_person_exemplar`, `person`/`person_link`,
`person_fact`, `vision_individual_split`, `vision_individual_pair_verdict`, and
the latest `vision_person_model_eval` row. No writes, no deploys. The models ran
in a throwaway container from the vision-host image, on CPU, with the crop
thumbnails mounted read-only. The host load average was about 200 on 96 cores, so the speed numbers are upper bounds.

Truth: age band = child under 13, else adult, from Juniper's `age` answers
(13 people, 2,595 album crops). Gender from her `gender` answer, else from
pronouns (3,015 album crops). People without an answer are not scored.

### 1. Does it return anything (coverage)

| Method | Input | Crops it can read | Per crop (CPU) |
|---|---|---|---|
| SigLIP zero-shot on the stored vector | vector already in Postgres | 16,052 / 16,052 (100%) | 0.3 ms |
| InsightFace `genderage` (buffalo_s) | face thumbnail | 238 (1.5%) | 24 ms |
| MiVOLO v2 (body, + face when present) | 160 px body thumbnail | 2,946 (18%); only 63 also had a face | 243 ms idle, 507 ms under load |

Full frames are not kept (a 5-minute ring), so a model that needs pixels can
only see the small stored thumbnails. The SigLIP vector was made from the full
crop at full resolution, so it carries more than the thumbnail does.

### 2. Is it right (against Juniper's answers)

| Method | Age band, per crop | Age band, confident reads | Age band, per named cluster (3+ crops) | Gender, per crop |
|---|---|---|---|---|
| SigLIP zero-shot | 2,163 / 2,595 = 83% | 1,824 / 2,044 = 89% | 83 / 91 = 91% | 2,198 / 3,015 = 73% |
| InsightFace | 16 / 78 = 21% (no child ever read as a child) | same | 3 / 16 | 35 / 96 = 36% |
| MiVOLO v2 | 336 / 593 = 57% (children: 32 / 289 = 11%) | 318 / 563 | 27 / 56 = 48% | 477 / 726 = 66% (female 54%) |
| Linear probe on the stored vector, leave-one-person-out | 59.5% | | | not run |

On the 593 album crops both could read, SigLIP got 462 right (78%) and MiVOLO 336 (57%).

Split by truth (SigLIP): children 80%, adults 88%. Two people pull the number down.
One girl reads as a child only 62% of the time. One adult (a delivery worker in a
uniform) reads as an adult only 64% of the time. All other people with 20+ crops
score 92-100%.

Gender split (SigLIP): male 95%, female 58%. Per person, for the people with 20+ crops:
29%, 69%, 72%, 74% for four women and girls. That pattern is
misgendering, not noise.

I also trained a probe on Juniper's labels (leave one person out). It did worse than the
zero-shot text read (59.5%). With 13 people it learns who the person is, not how old they are. So
there is nothing to train. The text prompts are the whole model.

### 3. Within-cluster agreement and the known mixed clusters

Mixed clusters, known from Juniper's answers: 24 (clusters that were split,
plus clusters whose album crops name two people). Pure: 187 named clusters
with one person and no split. Flag a cluster when it holds at least one
confident child read AND one confident adult read (confidence 0.9):

| Cluster size | Mixed flagged | Pure flagged |
|---|---|---|
| 2-9 crops | 1 / 4 | 1 / 62 |
| 10-39 | 1 / 5 | 6 / 51 |
| 40+ | 5 / 10 | 5 / 21 |

The motivating case is caught. That is the cluster that held a boy's crops and one adult
crop: 20 crops read at 0.98-1.0 child, and the one adult crop read 0.03.
But none of the 24 known mixed clusters is mixed by age band according to
Juniper's answers. The flags on the other mixed clusters
come mostly from cluster size: a big cluster has more stray reads. So the
flag is good for exactly one shape of mistake (a child cluster that holds an
adult crop, or the reverse). It is not a general purity detector.

### 4. Would a veto change the nightly eval

**Replay on the latest stored eval (2026-11-12, `median-topk-street-v2-people-split-face90`).**
That run scored 200 named clusters held out one at a time. It gave 176 "unsure" guesses
(77 named the right person, 99 the wrong one) and 0 accepts, and 24 clusters had no album.
For each guess, I checked whether the cluster's own crops confidently disagree with the guessed
person's age band (from Juniper's answer):

| Veto | Wrong top guesses struck | Right top guesses struck |
|---|---|---|
| SigLIP age band | 8 / 69 scorable | **0 / 52** |
| SigLIP apparent gender (not proposed) | 30 / 69 | 2 / 52 |
| MiVOLO age (not proposed) | 28 / 42 | 4 / 34 |
| InsightFace age (not proposed) | 5 / 10 | 5 / 7 |

The age veto is precise but small. It never struck a right guess, and it removed 12% of the wrong
ones it could score. It does not move the gate (0 wrong accepts before and
after), because nothing is accepted yet. MiVOLO strikes more wrong guesses only because it calls
almost every child an adult, so it also strikes right guesses.

FULL_RERUN

### 5. Pair verdicts

`vision_individual_pair_verdict`: of the cluster pairs Juniper called "same", a
confident age mismatch would wrongly split 4 / 104 (3.8%). Of the pairs she
called "different", it catches 6 / 39. A gender mismatch would wrongly split 8 / 104.

## Licences

- SigLIP (`google/siglip-so400m-patch14-384`): Apache-2.0, already deployed.
- InsightFace model packs: non-commercial research use only (code is MIT).
- MiVOLO v2 weights (`iitolstykh/mivolo_v2`): CC BY-NC 4.0. Code Apache-2.0.
  Needs `timm==0.8.13.dev0` and `transformers==4.51.0`. The vision host pins
  `timm==1.0.9` and `transformers==4.49.0`, so MiVOLO would need its own
  container.
- FairFace: CC BY 4.0, but face-only. Faces are 1.5% of crops, and the face path
  here fails on children, so I did not run it.

## Proposed design (age band only)

### Producer: a pure function, no new stored attribute

`orion/vision/age_band.py` (new, about 60 lines):

- `AGE_BAND_PROMPTS`: the six prompt strings (three for child, three for adult).
- `age_band_text_vectors()`: loads the SigLIP text tower once and caches the two
  mean text vectors, plus `logit_scale`/`logit_bias`, in a small JSON file
  committed next to it (`orion/vision/age_band_text_vectors.json`, 2 x 1152
  floats). The sql-writer then never loads a model. A test pins the file to the
  model id and prompt hash.
- `p_child(vec) -> float`: sigmoid of the logit difference. Pure numpy.
- `cluster_age_band(vecs, conf=0.75, min_n=2) -> {"band": "child"|"adult"|None,
  "p_child": float, "n": int}`. It returns `None` when the reads are few or not
  confident. "Unknown" is a normal answer.

Nothing is written per crop or per person. The estimate is recomputed from the
stored vector each time a guess or a purity check runs (0.3 ms per crop).

### Consumer 1: guess veto in `attribute()`

In `vision_person_model.attribute()`, before scoring, add the name of each album person
whose band confidently disagrees with the cluster's band to the
cluster's `blocked` set. The person's band comes from Juniper's `person_fact`
age answer. When there is none, it comes from the album consensus (5+ crops, confident mean).
The veto can only remove a candidate. It never names one. Behind
`VISION_PERSON_AGE_VETO_ENABLED=true` (orion-sql-writer `.env_example`,
`settings.py`, compose), on by default per the ship-flags-on rule.

The eval gets the same veto through the same function. `MODEL_VERSION` bumps
(`...-age-veto`), so the nightly McNemar compares the old and new runs on the same clusters.

### Consumer 2: one purity nudge, not a splitter

In the individuals reducer's existing split-ask path: a named cluster with a
confident child read AND a confident adult read (confidence 0.9) becomes a candidate for
the existing "Is this the same person?" picture card. The card shows the two
crops that disagree. It never splits on its own. The card stays under its existing daily cap.

### Schema change

One additive column, and nothing else in Postgres:
`ALTER TABLE vision_person_attribution ADD COLUMN IF NOT EXISTS age_veto jsonb`
(a `manual_migration_*.sql` with `-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer`.
The statement is on the post-pull auto-apply allowlist). The column holds
`{"struck": [names], "band": "child"|"adult", "p_child": float, "n": int}`.
It is set only when the veto struck someone, and it is the trace that proves the veto fired.
The table has no JSON detail column today (`individual_id, person_label, channel,
score, band, model_version, decided_at, revoked_at, basis`). The nightly eval
also reports `age_vetoes` (count) in its metrics JSON. There is no bus channel or registry change.

### Privacy, retention, display

- Passersby are third parties. No age band is stored for anyone, passerby or
  household. It is a recomputed number, it lives only for the length of one
  decision, and it adds no retention beyond the crop vector that already exists.
- Only the veto trace is kept (`vision_person_attribution.age_veto`, and only on
  rows where it struck someone). It inherits the guess row's retention.
- Display: the age band is never shown as a fact about a person. It is never shown on
  People pages, ask cards, chat, day letters, or the graph. The Hub debug view of a
  guess may show "age-band veto struck: X" with the number. That is all.
- No gender, apparent or otherwise, is estimated, stored, or displayed. Gender stays
  only what Juniper or the person said (`person_fact`).

## Non-goals

- No apparent-gender estimate in any form (see measurements).
- No new model, service, GPU lease, or container.
- No identity labels from age. A veto removes a candidate. It never names anyone.
- No automatic splits.
- No per-crop or per-person age table.
- Not fixing cross-day body matching (the real reason no accept bar has armed).

## Acceptance checks

1. Unit: `p_child` on the committed text vectors reproduces this eval's per-crop
   reads (fixture of 20 stored vectors with their scores) to 1e-4.
2. Unit: `cluster_age_band` returns `None` for 1 crop, and for a mean between 0.25 and 0.75.
3. Unit: `attribute()` with the veto never adds a name. It only removes candidates. A
   cluster whose band is `None` gets an identical guess with and without the veto.
4. Eval: `make eval-person-recognition` with the veto has 0 wrong accepts,
   and `unsure_right_person` is not lower than without it, on the same night.
5. Live (after deploy): at least one `vision_person_attribution` row with a
   non-null `age_veto` within 7 days, and a manual look at the struck crops
   confirms the strike was right.
6. Grep gate: no `gender` key appears in `orion/vision/age_band.py` or the guess
   trace.

## Failure modes and rollback

- Dangerous failure: a child is read as an adult (or the reverse) and the true person is struck,
  so Orion keeps failing to recognise someone. Bounded: the veto only touches confident cluster means
  (2+ crops), and 0 of 52 right guesses were struck in the replay. Check 4
  catches a regression on the next night.
- Tall children and teenagers sit between the bands. The band is two-way, with a wide
  "unknown" middle, so ambiguous clusters get no veto.
- Rollback: `VISION_PERSON_AGE_VETO_ENABLED=false`, then restart orion-sql-writer.
  Nothing is stored, so there is nothing to clean up.

## Recommended next patch

One PR in orion-sql-writer plus `orion/vision/age_band.py`: the pure function, the
committed text vectors, the `attribute()` veto behind the flag, the `age_veto` column,
the eval wiring, and tests 1-3. The purity nudge (consumer 2) comes as a second, smaller
PR, and only if Juniper wants more picture cards.
