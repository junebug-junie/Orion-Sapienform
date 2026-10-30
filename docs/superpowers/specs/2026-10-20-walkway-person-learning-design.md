# Walkway person learning: Orion learns who people are from Juniper's answers

Status: design, not built. Date: 2026-10-20. Builds on
`2026-09-22-walkway-camera-busy-world-design.md` (individuals, asks) and PR #2765
(photos kept forever, per-label snapshot slot, story links).

## Arsonist summary

Today Orion matches. It does not learn. Each new walkway crop joins the closest
stored "individual" (a cluster of crops) or starts a new one. A name Juniper
gives labels one cluster. Nothing carries that name to the next day.

Live, 2026-10-20: 275 person clusters, 13 named. Juniper is 2 clusters; another named person is 4.
Juniper named herself on a story on 10-19; the clusters Orion opens tomorrow
will not know it.

This design gives each named person an **album** of confirmed crops and a
**confidence bar learned from Juniper's yes/no answers**. A new person is
compared against every album. Above the bar: "probably Juniper". Below the
reject bar: not Juniper. In between: Orion asks **"Is this Juniper?"** -- and
the answer grows the album and moves the bar. A held-out eval proves whether
it is learning.

There is not enough labelled data to train a neural network today (202
labelled crops, 49 with a face, 14 same-person answers). Training is Phase 3
and is gated on the eval.

**Urgent (Phase 0):** labelled examples are being deleted right now. Crop rows
(and their embeddings) are pruned after 7 days (`VISION_CROP_RETENTION_DAYS=7`).
Two named people already have 0 surviving crops; a third has 4 left of 30
sightings; a fourth person's 10-15 crops go within a day. Every lost labelled crop is a
lost training example.

## Current architecture

- **Producer.** `orion-vision-host` `app/crop_embeddings.py`: per embedded
  crop, a body vector (`embedding`), an OSNet re-id vector (`osnet_embedding`,
  shadow), and for walkway person crops a face vector (`face_embedding`) when
  a face is found. Written to `vision_crop_observation` by `orion-sql-writer`.
- **Clusterer.** `orion-sql-writer` `app/vision_individuals.py`: face centroid
  match first (`VISION_INDIVIDUALS_FACE_MATCH_THRESHOLD=0.60`), else body
  centroid (`..._MATCH_THRESHOLD=0.90`), else a new individual. Centroids are
  running means -- the only "learning" today.
- **Asks.** "Who is this?" (one cluster, typed name -> `vision_individual.label`)
  and "Are these the same person?" (`vision_individual_pair_verdict`, 6 same /
  8 different). Story cards name a cluster too (`motion_story_routes.py`).
- **Separate, untouched:** `orion-vision-host` `identity_face` gallery
  (`identity_gallery.py`): one enrolled subject (Juniper, 14 samples), never
  grows at runtime, per the 2026-08-21 spec §4.
- **Gap.** A name lives on a cluster, not on a person. No cross-cluster
  matching against named people, no per-person threshold, no measure of
  whether recognition improves.

## Missing questions

1. **Does the 2026-08-21 "gallery does not grow / no stranger gallery" rule
   still bind here?** That rule governs the vision-host `identity_face`
   gallery. The walkway individuals work (2026-10-15) already stores every
   crop embedding, strangers included. This design leaves `identity_face`
   alone and builds albums only in sql-writer from Juniper's answers.
   **Decided (Juniper, 2026-10-20): yes, Orion builds answer-built albums
   here.** `identity_face` stays as it is.
2. **Can anything but a face recognise someone across days?** Body and OSNet
   vectors are mostly clothing. Juniper has 125 labelled crops but 1 with a
   face. If body vectors only match same-outfit, albums will mostly help
   same-day. Phase 1's eval measures this before anything is shown as a name.
3. **Should "probably Juniper" reach other consumers** (story names, rollups,
   chat presence)? This design: Hub only, until the eval gate passes.

## Proposed schema / API changes

### Phase 0 -- stop deleting labelled examples (one env key)

- `VISION_CROP_RETENTION_KEEP_LABELS=person` (orion-sql-writer). The crop
  prune skips rows whose `label` is in this list. Person crops measured
  2026-10-20: 4778 on the first day (burst), then 69-1036/day, ~5 KB each
  (mostly the embedding): about 1-2 GB/year. Vehicles keep the 7-day prune.
- Ship ON. `0`/empty = old behaviour. **Shipped: PR #2776, deployed 2026-10-20.**

### Phase 1 -- albums, learned bars, eval

New tables (additive, `manual_migration_vision_person_album_v1.sql`,
`-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer, orion-hub`):

```sql
CREATE TABLE vision_person_exemplar (
  exemplar_id   text PRIMARY KEY,
  person_label  text NOT NULL,          -- "Juniper"
  crop_id       text NOT NULL,          -- vision_crop_observation.crop_id
  polarity      smallint NOT NULL,      -- +1 is this person, -1 is NOT
  source        text NOT NULL,          -- cluster_label | confirm_yes | confirm_no | pair_same
  ask_id        text,                   -- the answer that put it here
  added_at      timestamptz NOT NULL DEFAULT now(),
  revoked_at    timestamptz
);
CREATE TABLE vision_person_bar (
  person_label  text PRIMARY KEY,
  channel       text NOT NULL,          -- face | body | osnet (eval-chosen)
  accept_at     real,                   -- NULL = not calibrated, ask only
  reject_below  real,
  n_genuine     integer NOT NULL,
  n_impostor    integer NOT NULL,
  model_version text NOT NULL,
  fitted_at     timestamptz NOT NULL
);
CREATE TABLE vision_person_attribution (
  individual_id text NOT NULL,
  person_label  text NOT NULL,
  score         real NOT NULL,
  band          text NOT NULL,          -- accept | unsure
  model_version text NOT NULL,
  decided_at    timestamptz NOT NULL,
  revoked_at    timestamptz,
  PRIMARY KEY (individual_id, person_label, model_version)
);
CREATE TABLE vision_person_model_eval (
  run_id text PRIMARY KEY, model_version text NOT NULL, ran_at timestamptz NOT NULL,
  metrics jsonb NOT NULL                 -- see Acceptance checks
);
```

`vision_individual.label` stays Juniper's answers only. A model guess never
writes it.

**Album fill (deterministic).**
- A cluster Juniper named directly (`label_ask_id` set, not `label_copied`) ->
  its crops become `+1 cluster_label` exemplars.
- "Is this X?" yes -> the shown crops `+1 confirm_yes`; the cluster is
  labelled through the existing answer path.
- "Is this X?" no -> the shown crops `-1 confirm_no` for X.
- Pair verdict "same" between a named cluster and an unnamed one -> the
  unnamed cluster's crops `+1 pair_same`.

**Score.** For an unnamed cluster C and a person P: per channel, the mean of
the top-3 cosines between C's recent crops and P's `+1` exemplars. Face when
both sides have faces; otherwise the eval-chosen fallback channel.

**Learned bars (per person, refit each tick when the album changed).**
- Genuine scores: each `+1` exemplar scored against the rest of P's album,
  leaving out its whole cluster (no cluster-mate leakage).
- Impostor scores: every other person's `+1` exemplars and P's `-1`
  exemplars, scored against P's album.
- `accept_at` = conformal bound on false accepts: the
  `ceil((1-a)(n+1))`-th smallest impostor score, a = 0.05. Needs
  n_impostor >= 19; otherwise NULL (no accept band; ask only).
- `reject_below` = the a-quantile of genuine scores (same rule, n_genuine >= 19).
- Three bands, never two (the 2026-08-21 rule: `unsure` must be common).

**Ask.** New ask kind `vision_person_confirm`: "Is this Juniper?" with the
cluster's crops, answers yes / no / can't tell. Chosen from unsure-band
clusters, closest to the middle of the band first. Shares the existing
`ORION_ASK_DAILY_CAP`. When no person has a bar yet, the existing "who is
this?" ask runs unchanged.

**Hub.** Ask card renders the new kind. Story card and ask card show an
active `accept` attribution as "probably Juniper (Orion's guess)", with a
"not Juniper" button that writes a `-1` exemplar and revokes the attribution.

Env (orion-sql-writer, ship ON per standing rule):
`VISION_PERSON_MODEL_ENABLED=true`, `VISION_PERSON_FALSE_ACCEPT_ALPHA=0.05`,
`VISION_PERSON_TOP_K=3`. Off = no attributions, no new asks, tables untouched.

### Phase 2 -- reach other consumers

Only after the Phase 1 eval gate passes: story naming rule, rollups and chat
presence may read `accept` attributions, marked as a guess. Separate PR.

### Phase 3 -- trained model

When there are >= 300 `+1` exemplars across >= 5 people: train a small
linear projection on the frozen face/body vectors (metric learning, triplet
loss, CPU). Ships only through the same eval gate as a new `model_version`.

### Phase 1 calibration finding (2026-10-20, live dry run)

Impostors = "every other person's +1 exemplars and P's -1 exemplars" (above)
is not enough. Fitted that way, one person's bar accepted 23 of 252 unnamed
street clusters: the street is a different population from the 6 other named
people. Built instead (PR on `feat/vision-person-model`):

- `accept_at` is fitted on CLUSTER scores (the decision unit) of the street
  (every unnamed recent cluster), other named people's clusters and ruled-out
  clusters. Unnamed clusters that really are P only raise the bar.
- The accept bar is dropped unless one of P's own clusters, held out with
  every album crop from its days, scores above it.
- A cluster's score is the median of its crops' scores, not top-k over pairs
  (top-k inflates with cluster size).

Result on live data: no person has an accept bar; every guess is `unsure`;
leave-one-cluster-out eval 0 wrong accepts, 5 of 8 unsure guesses name the
right person. Cross-day cluster separation: body AUC 0.76 (4 genuine
clusters), face 0.69 (3). "Probably X" will appear only once answers grow the
albums enough that a held-out visit beats the street.

## Metric gate (Phase 1 score)

1. **Provenance.** Vectors: `crop_embeddings.py` (`embed_fn`, `_attach_faces`,
   `_attach_osnet`) -> `vision_crop_observation`. Labels: `orion_ask.answer_detail`,
   `vision_individual_pair_verdict`.
2. **Independence.** The score reuses the clusterer's vectors, so it is not a
   new sensor. It is a different decision (person, not cluster). The eval
   holds out whole clusters so the clusterer's own grouping cannot leak.
3. **Theory.** Open-set verification with per-identity thresholds; conformal
   bound on false-accept rate; top-k prototype matching for few-shot classes.
4. **Live sanity (today).** 202 labelled crops, 49 with faces (Juniper: 1 of
   125). Expect `unsure` to dominate and face to be the only cross-day
   channel. The first eval run must report cross-day genuine vs impostor
   separation per channel; a channel that only separates same-day does not
   get an accept band.
5. **Existing mechanism.** `identity_face` (one subject, untouched),
   `vision_individuals` centroids, pair verdicts, `label_copied`.
   `orion/person_model/` is text-name linkage, not vision.
6. **Reversibility.** New tables only; one flag; attributions carry
   `model_version` and `revoked_at`. Dropping the feature drops four tables.

## Files likely to touch

- `services/orion-sql-writer/app/vision_individuals.py` (crop prune keep-list; album fill on answers)
- `services/orion-sql-writer/app/vision_person_model.py` (new: score, bars, attribution, pick asks -- pure functions + one DB cycle)
- `services/orion-sql-writer/app/settings.py`, `.env_example`, `.env`, `docker-compose.yml`
- `services/orion-sql-db/manual_migration_vision_person_album_v1.sql`
- `services/orion-sql-writer/evals/run_person_recognition_eval.py` (new)
- `services/orion-sql-writer/tests/test_vision_person_model.py` (new)
- `orion/schemas/ask.py` (new ask kind) + registry
- `services/orion-hub/scripts/ask_routes.py`, `motion_story_routes.py`, `static/js/walkway-stories.js`

## Non-goals

- No change to the vision-host `identity_face` gallery.
- No merging of clusters. Attribution names a cluster; it never rewrites centroids.
- No model guess written to `vision_individual.label`.
- No neural training before Phase 3's data threshold and eval gate.
- No new consumers of "probably X" before Phase 2.

## Acceptance checks

- **Phase 0:** after deploy, `SELECT min(created_at) FROM vision_crop_observation WHERE label='person'`
  stops advancing; vehicle rows still prune.
- **Phase 1 tests:** bars NULL below 19 impostors; three bands; a `-1`
  exemplar lowers a cluster's score for that person; held-out cluster never
  scores against itself; flag off = zero writes.
- **Phase 1 eval** (`run_person_recognition_eval.py`, leave-one-cluster-out
  over every labelled cluster), writes a `vision_person_model_eval` row with:
  wrong-accept count, correct-accept rate, unsure rate, per-channel cross-day
  separation. Gate: wrong accepts = 0 on the held-out set, and any
  `model_version` change must beat the previous one on a paired test
  (McNemar, p < 0.05), not on a lower average.
- **Live:** an `accept` attribution and a `vision_person_confirm` ask appear
  for a real unnamed cluster; Juniper's answer adds an exemplar row and the
  refit bar row's `fitted_at` moves. If `unsure` is under 20% of scored
  clusters, treat the bars as miscalibrated.

## Handoff: everything still open (2026-10-20)

**Status update 2026-10-29.** Checked against the live database and logs:
A.1 done (crop_thumbs 13,790, face_frames 5.4G). A.2 applied 2026-10-29: 76 person
stories (all ended 10-17..10-19, before #2765) got 130 links; a re-run plans 0
(snapshot and undo: `/tmp/story-evidence-relink/`). B verified: the container is
`orion-athena-sql-writer` (64 `[MOTION_STORY] relinked` lines in 24 h); person thumb
share 54/136 in a day; every person story since 10-20 is linked except 1 of 594; all 76
"who is this?" asks since 10-20 carried pictures. C.1 acceptance passed (oldest person
crop still 10-15, vehicles pruned to 10-22). C.3 done (ask is `withdrawn`). C.6 done:
linked from the tracking list in PR #2752. D question 3 is still open: nothing reads
"probably X" guesses yet (no PR as of 2026-10-29).

Context for a fresh agent. PR #2765 (photos kept forever, per-label snapshot
slot, story-link relink) is merged and its images are deployed. This spec is
PR #2767 (squash-merge it: early branch commits held names the privacy gate
later removed). Items below are not done unless marked.

### A. Operator steps (Juniper runs; agent sessions are blocked from these)

1. **Restart to pick up keep-forever.** Primary checkout `.env` files were
   already switched to `0` on 2026-10-20 (backups in the session scratchpad);
   the running containers still hold 10/14/30 days until restarted:
   `scripts/safe_docker_build.sh orion-vision-host up -d && scripts/safe_docker_build.sh orion-sql-writer up -d`
   Verify after: `ls /mnt/telemetry/orion-vision-host/crop_thumbs | wc -l` stays
   >= 1458 and `du -sh /mnt/telemetry/orion-vision-host/face_frames` stays
   >= 2.4G over the next hour.
2. **Story-link backfill** (must run before crop rows age out, 7 days from
   2026-10-15..19). 72 of 147 person stories in the last 5 days were still
   unlinked after deploy. Dry run, then `--apply`:
   `.venv/bin/python services/orion-sql-writer/scripts/backfill_story_evidence_links.py --dsn postgresql://postgres:postgres@localhost:55432/conjourney --job-dir /tmp/story-evidence-relink [--apply]`
   Undo: `--restore /tmp/story-evidence-relink`.

### B. Live verification of #2765 (UNVERIFIED as of 2026-10-20 04:00 UTC)

- `[MOTION_STORY] relinked` log line in orion-sql-writer. (Story
  `3ca2d1c15fd53c423a38400ddc582709` did gain its link after deploy.)
- Person crops get thumbnails while vehicles are present:
  `SELECT label, count(*), count(thumb_ref) FROM vision_crop_observation WHERE created_at > now()-interval '1 day' GROUP BY 1`
  -- person thumb share should rise well above the pre-deploy 106/973.
- A new "who is this?" ask carries `image_refs` from a sighting older than 3 days.

### C. Build items (status 2026-10-20, end of day)

1. **Phase 0: DONE.** PR #2776, deployed, live container has
   `VISION_CROP_RETENTION_KEEP_LABELS=person`. Acceptance (oldest person crop
   stops moving forward while vehicles keep pruning) is checkable from 2026-10-22.
2. **No picture, no ask: DONE.** PR #2782, deployed. Also fixed: both
   "who is this?" openers fetched only budget-many candidates, so a skipped
   one held the day's slot on every tick. The replay eval now fails any
   picture-less ask. The Phase 1 `vision_person_confirm` opener must follow
   the same rule (skip, never `LIMIT` by budget before the skip).
3. **Retire the picture-less open ask: WAITING ON JUNIPER (operator step).**
   The plan above was wrong: the re-ask cooldown counts from the ask's
   `created_at` (`_NOT_ASKED_SQL`), so `expired` still blocks until about
   2026-11-18. `withdrawn` is not in the blocking list and the Hub only shows
   `open`. Approved by Juniper, blocked for agent sessions (production write):
   `psql postgresql://postgres:postgres@localhost:55432/conjourney -c "UPDATE orion_ask SET status='withdrawn' WHERE ask_id='ask-9645cdc1bdd64660b80a07380b7c71ef' AND status='open'"`
   The next tick opens a new ask with the individual's 4 sighting thumbs.
4. **Impure clusters: MEASURED, build with Phase 1.** Read-only, 2026-10-20,
   over surviving crops with embeddings (202 labelled crops in 11 named
   clusters; 774 in 240 unnamed). Reference scales from Juniper's data:
   - Same person (crops in one sighting, crop vs crop), p10 / p50: body 0.836 / 0.894,
     OSNet 0.635 / 0.738, face 0.286 / 0.546.
   - Different people ('different' pair verdicts, crop vs crop), p50 / p90: body 0.703 / 0.814,
     OSNet 0.601 / 0.687, face 0.268 / 0.417. Centroid vs centroid (n = 4), body p50 0.821.
   - Purity test: 2-means split of a cluster's crops; mixed if the two halves'
     centroids are below the same-person p10 and each half has >= 2 crops.
     Body: 1 of 52 clusters with >= 4 crops, **1 of 6 named** (`f5694d84...`,
     3 vs 2 crops at 0.81). OSNet: 1 of 51, 0 named. Face: 0 of 3.
   - The C.3 individual's 10 surviving crops look like one person (body split
     0.94); the two odd thumbs are probably from crops already pruned.
   - Caveat: only 4 'different' verdicts calibrate the scale, so this is a
     screen, not a verdict. Phase 1 design consequence: a cluster that fails
     the split test contributes no `+1` exemplars (abstain) until Juniper
     answers; add the "these are different people" answer as planned. Script:
     re-derive from this description; it was a one-off scratch file.
5. **Disk alarm: ALREADY EXISTS.** `scripts/disk_threshold_watchdog.py`, cron
   every 15 min, covers `/mnt/telemetry`, Hub attention card at 90 %. Last run
   2026-10-20 06:00 UTC: 25 % used (336 G free). No change needed.
6. **Tracking doc: WAITING** on PR #2752 to merge, then link this spec there.
7. **Cleanup: DONE.** Worktrees for #2765, #2776, #2782 removed.

### D. Decisions waiting on Juniper

- ~~Missing question 1~~ answered 2026-10-20: albums allowed.
- Missing question 3 (when "probably X" may reach other consumers).

## Recommended next patch

(Updated 2026-10-20: Phase 0 and C.2 shipped.) Phase 1, after Juniper answers
section D question 1. Include the C.4 abstain rule.

Original recommendation:

Phase 0 alone, today: `VISION_CROP_RETENTION_KEEP_LABELS=person` in
orion-sql-writer with a test that person crops survive the prune and vehicle
crops do not. Every day it waits deletes more labelled examples. Then C.2
(no picture, no ask) in the same service.
