# Walkway follow-ups: context spec for three items

Status: design only, 2026-10-20. No code, no database writes, no deploy.
Live numbers below come from read-only queries run 2026-10-20 03:24–03:40 UTC
(2026-10-19 21:24–21:40 MDT). Each query is in the appendix.

This spec gives context for three follow-ups. Each one later gets its own
design doc or patch:

1. **Reach.** Seven parts of Orion get some or none of what the walkway camera sees.
2. **People are over-counted.** The hour/day rollups count one person many times.
3. **Invented motion.** The vision council can repeat a caption model's guess as fact.

The tracking list for item 1 is
`docs/superpowers/plans/2026-10-18-walkway-reach-into-orion-queue.md`
(branch `docs/walkway-reach-queue`, PR #2752). This spec does not edit that
branch. Link this spec from its item rows when that PR next moves.

---

## How the three items depend on each other

Items 2 and 3 decide whether what reaches Orion is true. Item 1 decides how much
reaches Orion. If item 1 goes first, every new consumer copies today's wrong
numbers and today's unlabelled guesses into chat, dreams, journals and memory.
Each copy is harder to take back than the source.

- **Chat stance, day letter, journals, curiosity day card** read rollup counts.
  They need item 2 (an honest count, or an honest range).
- **Reveries, dreams, curiosity material, street line** read `vision_events`
  narratives. They need item 3 (a council row must never carry a caption's guess
  as fact).
- **Named-people links** (`observed_in`) need item 2 too: a split person
  becomes several unnamed "regulars".
- Item 3's motion cross-check needs motion stories (`vision_motion_story`). Item 2's
  best count also comes from motion stories. One read helper can serve both.

### Recommended order

1. **Item 3, patch A**: keep captions out of the council's walkway input. It is small,
   deterministic, and stops a false-fact path at the source.
2. **Item 2, patch A**: the honesty rule. Orion-facing text shows a range, not one
   inflated number. No new estimator is needed for this.
3. **Item 2, patch B**: a read-only eval that scores the count estimators against
   hand-checked hours. Pick the estimator from its result.
4. **Item 3, patch B**: a deterministic motion cross-check on `camera_caption` rows.
5. **Item 1**: one faculty at a time, in the order in the item 1 section.

---

## Decisions recorded (Juniper, 2026-10-20)

1. **YES.** Every Hub "Who is this?" answer adds the same-name `same` link to the
   existing individuals that already carry that typed name. This applies to all
   answers, not only story-card names (PR #2753). Being implemented in a separate PR now.
2. **YES.** The post-pull migration step (`scripts/apply_post_pull_migrations.py`,
   PR #2703) waits for a running `pg_dump` backup to finish and then retries. It must
   not fail on `lock_timeout`. Incident: 2026-10-17 04:00 UTC, the backup held locks
   for about 45 minutes, the migration FAILED, and sql-writer and hub refused to
   start. Being implemented in a separate PR now.
3. **YES.** The `doorbell` stream in sql-writer `VISION_MOTION_STORY_STREAMS=walkway,doorbell`
   is a real camera. Keep it. Follow-up: the template must match the live intended
   default. Checked 2026-10-20 on `main`: `services/orion-sql-writer/.env_example`
   line 363, `settings.py` line 613 default, the local `.env`, and the running
   `orion-athena-sql-writer` container all say `walkway,doorbell`. So on `main`
   no difference exists today. If a difference was seen, it was on another branch;
   that follow-up is to check that branch before it merges. No env change here.
   Note: the doorbell stream wrote 0 crop observations in the last 72 h (it is
   event-driven), so it adds no rows to the counts below.

---

## Item 2 — People are over-counted

### Arsonist summary

The walkway rollup says "79 people" for one hour. No single frame in that hour held
more than 5 people. The count is the number of person clusters, and the clusters
reducer makes a new cluster for most single detections. So "passings" mostly
measures how often the matcher failed, not how many people walked by. This number
already reaches the day letter, the journals, the curiosity day card and the Hub.

### Current architecture

- A **passing** is one individual seen in one clock hour
  (`orion/vision/walkway_rollup.py` lines 18–22). The rollup reducer is
  `services/orion-sql-writer/app/vision_rollup.py`. It writes `vision_rollup` rows
  (`grain` hour/day/week/month) with `kinds.<kind>.passings/distinct/sightings`,
  `peak_at_once`, and a text `line`.
- Individuals come from `services/orion-sql-writer/app/vision_individuals.py`. The
  matcher tries a face line (cosine 0.60), then a continuation line (0.88, within
  10 s and 1.5 box heights), then a re-id line against cluster centroids (0.90).
  Cannot-link pairs come from Hub verdicts (PRs #2709, #2719).
- The body vector is SigLIP v1 (fallback; SigLIP2 fails to load). OSNet vectors
  are written in shadow (PR #2720). The comparison eval
  (`services/orion-sql-writer/evals/run_osnet_vs_siglip_eval.py`) says INSUFFICIENT
  DATA as of 2026-10-20: 136 same-person and 54 different-person pairs; it needs
  200 of each.
- Detector cadence is about one walkway frame every 5 s.
- Motion stories (`services/orion-sql-writer/app/vision_motion.py`, table
  `vision_motion_story`) link boxes by position and speed. Each story gets a
  `drop_reason` (`glimpse`, `standing`, `prop`, `edge_clipped`, `out_of_scope`)
  or is published. Stories are Hub-only until the spot-check gate passes
  (needs 50 checked; 27 checked as of 2026-10-20 per the rollup's `stories` field).
- The rollup docstring calls passings "an upper bound" (line 35). The Orion-facing
  `line` does not say so. It prints "92 passing (79 people, …; most at once 5 people)".

### Live numbers (2026-10-20 03:25 UTC)

Per-hour estimates for persons on the walkway (worst hours, local time MDT):

| Hour (MDT) | Person clusters (= rollup passings) | Most people in one frame | Frames with a person | Presence runs | Sum of run peaks | Multi-frame stories | Glimpse stories |
|---|---|---|---|---|---|---|---|
| 10-16 12 | 79 | 5 | 198 | 7 | 16 | (stories not yet built) | |
| 10-16 11 | 27 | 2 | 78 | 13 | 17 | | |
| 10-16 09 | 22 | 3 | 51 | 9 | 12 | | |
| 10-17 14 | 15 | 2 | 18 | 10 | 12 | 3 | 10 |
| 10-19 11 | 13 | 2 | 53 | 14 | 15 | 6 | 15 |
| 10-19 16 | 11 | 3 | 124 | 9 | 13 | 12 | 7 |

Definitions: a **presence run** is a stretch of frames with at least one person,
with no gap over 30 s. **Sum of run peaks** adds the most people in one frame for
each run. **Multi-frame stories** are published plus `standing`, `edge_clipped`
and `out_of_scope` stories.

Last 72 h, walkway persons:

- 89 new person clusters. 77 of them (87 %) have exactly one sighting.
  Only 4 have a name.
- 151 person sightings. The median sighting has 1 observation and 0 s dwell.
- Rollup person passings: 137 over 71 hours.
- Sum over hours of "most people in one frame": 47 (a hard floor).
- Presence runs: 87. Sum of run peaks: 115.
- Person motion stories: 144. Of these, 101 are `glimpse` (one frame, mean 1.2
  detections), 26 published (mean 4.2 detections, 10 s), 12 `standing`.
  0 stories carry an `individual_id`.

What this shows:

- The clusters count is an over-count when people linger. In 10-16 12 the runs
  lasted 124 s on average; 198 person frames became 79 clusters. Most clusters are
  one detection that matched nothing.
- **Motion stories are not a better count as they are today.** 144 person stories
  is more than 89 clusters, because most stories are one-frame glimpses. Only
  stories with more than one frame (38 in 72 h) look like passages. The brief's
  idea "count passages from the motion tracker" holds only after glimpses are
  merged or excluded, and that choice needs its own check.
- No estimator is proven. The floor (most at once) is certain. Every other number
  is a guess until it is scored against hand-checked hours.

### Missing questions

1. Is a glimpse story a real passer-by that the detector saw once, or detector
   flicker on someone already counted? Hand-check a sample of glimpses.
2. Why does the continuation line (10 s window) miss frames 5 s apart? Is the
   cause box jump, cosine below 0.88, or the 1.5 box-height limit? The matcher
   does not log which line failed.
3. Does one person walking back within the same hour count once (passing rule) or
   twice (presence runs)? The rule says once. Presence runs say twice.
4. Which number should Orion say: a range, or a single best guess with a method?

### Options

- **A. Report a range (honesty rule).** Low = most at once. High = clusters.
  Text: "at least 5 and at most 79 people; the camera cannot tell them apart well
  yet". Cheap, true today, no new estimator.
- **B. Count multi-frame stories as passages.** Needs: glimpse handling, and a
  dedupe of a person whose story breaks in two. Theory anchor: a passage is a
  continuous track through the scene; motion stories already build tracks.
- **C. Presence runs times peak.** Simple and deterministic. Weak when two people
  overlap in one run, or one person enters twice.
- **D. Per-hour dedupe in the matcher.** Merge clusters in one hour when their
  boxes never appear in the same frame and their vectors are close. Risky: it can
  merge two real strangers, and it can never be stronger than the vectors.
- **E. Better re-id with OSNet.** Only after the OSNet eval reaches 200/200 pairs
  and beats SigLIP v1 with a significance test (see memory note on model gates).

### Metric quality gate (CLAUDE.md §0A) a design doc must answer

1. Provenance: point at the exact function that produces each estimate.
2. Independence: clusters, stories and runs all come from the same detector boxes.
   They are not independent signals. They are three readings of one sensor.
3. Theory anchor: a passage is a continuous track (option B) or a contiguous
   occupancy (option C). Say which and why.
4. Live sanity: show the estimate reads 0 in an empty hour and does not grow with
   dwell time (a person standing for 2 minutes must stay 1).
5. Existing mechanism: `peak_at_once` and motion stories already exist. Reuse them.
6. Reversibility: keep the estimate in rollup JSON with a `rules_version` bump. Do
   not bake it into a schema column until it is proven.

### Proposed schema / API changes

- `vision_rollup.rollup` JSON (no column change): add per kind
  `count_range: {low, high, method_low, method_high}` and later `estimate` with
  `method` and `rules_version`. Bump `rules_version`.
- `line` text: show the range for people. Example: "people: at least 5, at most 79
  (the camera splits one person into several)". Vehicles keep their current count
  until checked the same way.
- No bus or schema registry change in patch A.

### Files likely to touch

- `orion/vision/walkway_rollup.py` (range fields, line text)
- `services/orion-sql-writer/app/vision_rollup.py` (rules version, recompute)
- `services/orion-sql-writer/tests/` (rollup tests)
- `services/orion-sql-writer/evals/` (new read-only count eval)
- Readers that print the number: `orion/orion_day/gather.py`, the walkway journal
  in `services/orion-actions/app/`, Hub "Walkway history" panel.

### Non-goals

- No matcher threshold change in patch A.
- No OSNet switch until its eval passes.
- No new table.

### Acceptance checks

- For every hour, `low <= high`, and `low` equals the most people in one frame.
- The 10-16 12 MDT hour reads "at least 5, at most 79", not "79 people".
- No Orion-facing line shows a single people count without its range.
- Count eval: on at least 10 hand-checked hours, report each estimator's error.

### Recommended next patch

Honesty rule: rollup JSON gets `count_range` for persons (low = most at once,
high = clusters) and every Orion-facing line prints the range. Then a read-only
eval that scores clusters, stories and runs against hand-checked hours.

---

## Item 3 — The council passes caption guesses on as fact

### Arsonist summary

A small caption model (Qwen2-VL-2B) describes one frame and guesses motion from a
still picture. It said "a car is driving" 9 times; in all 9 frames the vehicles the
detector saw belong to motion stories marked `parked`. Those captions are saved as
`camera_caption` rows and labelled "motion not verified". That part works. But the
same caption text is also put into the vision council's prompt. The council
sometimes copies the caption's wording into its own row, and that row has no label.
Readers show council rows as plain fact.

### Current architecture (traced)

1. The frame router asks Qwen for a caption. The window's caption gate
   (`services/orion-vision-window/app/caption_gate.py`) accepts it when a detector
   label agrees. An accepted caption goes into `window.summary.captions`.
2. The window also writes the caption as its own `vision_events` row
   (`services/orion-vision-window/app/caption_events.py`): `event_type="camera_caption"`,
   tags `camera_caption`, `motion_unverified`, `detector:<label>`, and evidence
   refs `artifact:`, `frame:`, `frame_ts:`, `box:`, `crop:`, `model:`.
3. The council (`services/orion-vision-council/app/main.py`) runs when the
   evidence-transition gate (`evidence_transition.py`) sees a change in believed
   labels, or when the refresh timer fires.
4. `interpretation.py` `_compact_window_context` puts up to 3 captions into the
   prompt (`summary.captions`). The prompt rule says "soft hints only; never sole
   basis for activity claims". The 8B model does not always follow it.
5. `evidence_grounding.py` `enforce_evidence_grounding` checks only two things:
   person words without `person` in `hard_labels`, and activity verbs from a short
   list (`watching|reading|using|talking|listening|playing|browsing`). Motion verbs
   (driving, walking, pushing, riding, parked) are not checked. The
   `caption_inferred` tag is added only for those activity verbs with a person
   present and a caption in the window.
6. Note: the change gate uses `believed_hard_labels`; grounding uses raw
   `hard_labels`. They can disagree.
7. `project_interpretation_to_events` writes each event candidate as a
   `vision_events` row with the candidate's tags. No caption provenance is added.
8. Readers: `orion/situational/street_material.py` `dedupe_street_captions` labels
   `camera_caption` rows with `label_camera_caption` ("camera caption (one frame;
   … motion not verified)"). Council rows (`visual_observation`) pass through
   unlabelled. Same for reverie (`services/orion-thought/app/reverie.py` via
   `vision_reader.py`) and curiosity material (`orion/curiosity/study_material.py`).
   No reader checks the `caption_inferred` tag.

### Live numbers (2026-10-20 03:24–03:35 UTC)

- `camera_caption` rows all time: 83 (first 2026-10-17 13:51 UTC). 36 of 83 use a
  motion verb. 9 say "driving".
- Cross-check of the 9 "driving" captions: each caption's `frame_ts` matched to
  vehicle crops within 1 s, then to the motion stories that hold those crops. All 9
  frames have vehicle crops in stories with `speed_class=parked`; the rest are in no
  story. 0 of 9 have a moving vehicle story. Limit: this matched every vehicle
  in the frame, not the caption's own `box:`. One of the 9 names a mail truck,
  which may have been stopping and starting.
- Walkway council rows all time: 7,585. 72 h: 4,790, of which 3,313 are the same
  sentence ("A vehicle is observed in the walkway area.").
- Council rows that share an `artifact` with a caption: 27. Of those, 21 are the
  grounded fallback "A person was detected on camera." (tag `host_detect`), which
  carries no caption text. About 5 copy caption wording with no tag. Example:
  caption 2026-10-19 22:46:48 UTC "A person is pushing a lawn mower along a
  sidewalk next to a street." became council row 22:47:03 "A person is pushing a
  lawn mower on the sidewalk adjacent to a street.", tags `[]`, same artifact
  `e4bf5ae3…`. Another: council row 22:49:12 "A vehicle is parked on a residential
  street with a fence and trees in the background." copies a caption word for word.
- Council rows with a motion or parking word (walking, pushing, parked, driving,
  riding, mowing), all time: 8. 5 are from 10-16 ("persons observed walking"),
  3 are from 10-19 22:47–22:49 (caption copies: 2 "pushing", 1 "parked").
  0 say "driving".
- 3 council rows are tagged `caption_inferred` ("A person is using a blue lawn
  mower."). No reader uses that tag.
- Council log, last 72 h: 93 `injected:person_presence`, 3 `capped_confidence`,
  0 `dropped:*` notes.

So: the copy path is real and proven by artifact id. Today it is rare (about 8
rows in 7,585). The "driving car as fact" case is not seen in council rows yet; it
lives in the caption rows, which are labelled. The risk grows with item 1: more
consumers read council rows as fact.

### Missing questions

1. Does the room camera (`cam0`) need captions in the council prompt? Patch A
   below can be walkway-only.
2. Should the council see the caption at all, given the window already writes it
   as its own labelled row?
3. When a caption says "parked" and the story says `parked`, may the caption row
   drop "motion not verified"? (A confirmed claim is still a model's word.)
4. The gate accepted "A car is parked…" captions tagged only `detector:person`.
   Is agreement on any label enough, or must the caption's subject match?

### Options

- **A. Keep captions out of the council input for walkway.** The caption already
  reaches Orion through its own labelled row. The council then describes detector
  labels only. Kill, not fence: no "soft hint" fallback.
- **B. Carry provenance.** Tag any council candidate whose narrative overlaps a
  window caption (word overlap, deterministic) as `caption_derived`, and make every
  reader label such rows the way it labels `camera_caption`. More moving parts;
  every reader must learn the tag.
- **C. Motion cross-check (deterministic).** For each `camera_caption` with a motion
  verb, find the crop matching the caption's `box:` at `frame_ts` (IoU), find its
  motion story, and tag `motion_contradicted` (story `parked`/`standing`) or
  `motion_supported` (story moving). Readers then say "caption said driving; the
  car did not move". Needs motion stories for that stream.

A and C together cover the risk. B is needed only if A is rejected.

### How to measure

- **Copy rate**: council rows (not `camera_caption`, not `host_detect`) that share
  an artifact with a caption, or whose word overlap with a caption in the same
  window is above a set threshold, per 1,000 council rows. Today about 8 per 7,585.
  Target after patch A: 0 on walkway.
- **Unsupported motion**: council rows with a motion verb whose evidence frame's
  crops belong only to `parked`/`standing` stories or to no story. Target: 0.
- **Contradicted captions**: share of motion-verb `camera_caption` rows tagged
  `motion_contradicted` after patch C. Today, by hand: 9 of 9 "driving".

### Metric quality gate questions

1. Provenance: `speed_class` from `vision_motion.py`; caption boxes from
   `caption_events.py` evidence refs.
2. Independence: the story's motion and the caption come from different models
   (detector track vs VLM). That is why the cross-check means something.
3. Theory anchor: motion is displacement over time; one frame cannot show it.
4. Live sanity: confirm `parked` stories are not stamped on cars that did move
   (spot-check 20 with the Hub strip).
5. Existing mechanism: `motion_unverified` tag, `label_camera_caption`, motion
   stories. Reuse all three.
6. Reversibility: tags only. No column.

### Proposed schema / API changes

- Patch A: one council setting, for example `COUNCIL_CAPTION_STREAMS` (streams
  whose captions go into the prompt; default `cam0`). Env parity in
  `services/orion-vision-council/.env_example`, `settings.py`, and the local `.env`.
- Patch C: new tag values on `camera_caption` rows (`motion_contradicted`,
  `motion_supported`). `label_camera_caption` reads them. No bus change.

### Files likely to touch

- `services/orion-vision-council/app/interpretation.py` (`_compact_window_context`,
  `_derive_scene_summary`, which falls back to `captions[0]`)
- `services/orion-vision-council/app/settings.py`, `.env_example`, `README.md`, tests
- Patch C: `services/orion-sql-writer/app/vision_motion*.py` or a small reducer,
  `orion/situational/street_material.py` (`label_camera_caption`)

### Non-goals

- No change to the caption model or the caption gate in these patches.
- No change to room-camera behaviour unless question 1 says so.
- No relabelling of old rows.

### Acceptance checks

- Unit test: a walkway window with a caption builds a prompt with no caption text,
  and `_derive_scene_summary` never returns caption text on walkway.
- Live: 24 h after deploy, copy rate on walkway is 0 (query in appendix, A3).
- Patch C: the 9 known "driving" captions are tagged `motion_contradicted` in a
  replay test; the street line says so.

### Recommended next patch

Walkway captions leave the council prompt (setting `COUNCIL_CAPTION_STREAMS`,
default `cam0`). Then a deterministic motion cross-check tags each motion-verb
caption as supported or contradicted by its motion story.

---

## Item 1 — Seven parts of Orion and the walkway

Sources: the tracking list (snapshot 2026-10-18 00:15 UTC) plus checks re-run on
2026-10-20. "Re-checked" means a fresh live read this session. Everything else is
carried from the tracking list and is UNVERIFIED for today.

Since 2026-10-18: walkway rollups went live (PR #2755) with consumers in the
nightly/weekly/monthly journal (with an anti-invention guard), a day letter line,
the curiosity day card, the Hub "Walkway history" panel, and `observed_in` day
links (fixed by PR #2756). Story cards join to recognition (PR #2753). Captions are
saved as `camera_caption` rows (PR #2740).

| Part | Today | Gap | "Maximized" means | Needs item |
|---|---|---|---|---|
| Chat stance | Street line renders (last 15 min, names first, captions). Reaching a real chat turn: UNVERIFIED. | No trace that a turn used it; no day view; expectations part empty. | A turn log records which street facts went in; the line can say "today so far" from rollups. | 2, 3 |
| Curiosity | Captions (6 h) + walkway surprise; day card from rollups. | Kickoff prompt content not logged. | Investigations can ask rollup and story questions, with the evidence logged. | 2, 3 |
| Attention | Fenced by design. Leaks into every frame as a suppressed "body signal". | Wrong label (camera filed as body). | Decide: keep fenced and fix the label, or admit it as a world source with a camera-down check. | — |
| Substrate layers | Surprise node live, 0 edges (re-checked). 7 named entity nodes now have 13 `observed_in` edges total (re-checked). | Surprise node unlinked; unnamed regulars have no nodes. | Sightings refresh entity activation; surprise links to the day it happened. | 2 |
| Reveries | Last 2 captions: 98 `reverie_street_context` lines in 24 h, each `captions=2 graded=0` (re-checked). | No day view; no graded predictions yet. | Reverie reads the day's rollup line and graded predictions. | 2, 3 |
| Dreams | Graded walkway predictions only. `dream_replay_item` walkway rows: 0 (re-checked). `vision_percept_expectation`: 4 open rows now, all vehicle (was 0 on 10-18; re-checked). | Nothing graded yet; captions are not a source. | Met/missed predictions replay; surprising days replay. | 2 |
| Memory and day letter | Day letter now has a walkway line from rollups. Episodes still room-only (`ROOM_STREAMS={"cam0"}`). | Day letter number is the over-count. Episodes ignore the street. | Day letter shows honest ranges; notable stories become episode material. | 2, 3 |

### Per-part notes for the later design docs

- **Chat stance.** Code: `orion/situational/perception_reader.py`
  `fetch_street_summary` → `orion/situational/context.py` `_build_street_fields` →
  `services/orion-cortex-exec/app/chat_stance.py`. First patch is a trace, not more
  content: record the street fields a turn received (event or log with correlation
  id). Gate questions: what counts as "used"; does the line change replies at all
  (A/B on replay).
- **Curiosity.** `services/orion-hub/scripts/curiosity_investigation.py`,
  `orion/curiosity/kickoff_prompt.py`, `orion/substrate/endogenous_curiosity.py`.
  Log the actual kickoff material. Gate: the walkway surprise signal must be able
  to read calm (CLAUDE.md §0A item 4); check its rest value on a quiet night.
- **Attention.** `orion/attention/field_attention/builder.py` host-novelty path,
  `orion/substrate/attention_broadcast.py`. Smallest patch: correct the label and
  `source_kind`. Admitting the walkway is a proposal-mode change (autonomy loop).
- **Substrate layers.** `services/orion-substrate-runtime/app/worker.py`,
  `orion/substrate_ladder_liveness.py`. Writer for entity nodes today:
  `ask_answered_listener.py` plus the rollup `observed_in` links. Wait for item 2
  before linking unnamed clusters: today they are mostly one-detection splits.
- **Reveries.** `services/orion-thought/app/reverie.py`. Add the rollup day line.
  Needs item 3 patch A first so council copies of captions stop.
- **Dreams.** `services/orion-dream/app/cycle.py`, `replay.py`. Becomes real once
  expectations get graded (they now exist: 4 open). Check that a vehicle routine
  is not a parked car seen every day (rollup `excluded.parked_repeat`).
- **Memory and day letter.** `orion/orion_day/gather.py`,
  `orion/temporal_self/sources.py`. Episodes from the street are a proposal-mode
  change (memory). Day letter range text comes from item 2 patch A.

### Suggested order inside item 1

1. Chat stance trace (proves the one path that touches conversation).
2. Attention label fix (one-line class of change, removes a wrong fact).
3. Reveries and day letter, after items 2A and 3A.
4. Dreams, once predictions are graded.
5. Substrate links for unnamed people, after item 2's count eval.
6. Episodes from the street (proposal mode first).

### Item 1 as a design-mode summary

- **Arsonist summary**: the walkway is watched well and told badly. Most parts get a
  15-minute slice or nothing, and nothing records whether the slice was used.
- **Missing questions**: should the walkway ever win attention? Should street
  events become episodes?
- **Proposed schema/API changes**: a per-turn street-context trace (log or event);
  no new tables.
- **Files likely to touch**: listed per part above.
- **Non-goals**: no new consumer before items 2A and 3A; no attention admission
  without proposal mode.
- **Acceptance checks**: each part's design doc names one live trace that proves the
  path moved (CLAUDE.md §0A "runtime truth").
- **Recommended next patch**: chat-stance trace — log the street fields each chat
  turn received, keyed by correlation id.

---

## Appendix: queries (all `BEGIN READ ONLY`, run 2026-10-20 ~03:25 UTC)

Database: `docker exec orion-athena-sql-db psql -U postgres -d conjourney`.

**A1. Clusters vs most at once vs stories, per hour (item 2).**

```sql
with clus as (select date_trunc('hour', started_at) h, count(distinct individual_id) clusters
  from vision_individual_sighting s join vision_individual i using(individual_id)
  where s.stream_id='walkway' and i.kind='person' and started_at > now()-interval '96 hours' group by 1),
fr as (select h, max(n) max_frame, count(*) frames from (
  select date_trunc('hour', observed_at) h, artifact_id, count(*) n from vision_crop_observation
  where stream_id='walkway' and label='person' and observed_at > now()-interval '96 hours' group by 1,2) x group by 1)
select c.h, c.clusters, f.max_frame, f.frames from clus c join fr f using (h) order by c.clusters desc;
```

**A2. Presence runs (30 s gap) and sum of run peaks.**

```sql
with f as (select observed_at t, count(*) n from vision_crop_observation
  where stream_id='walkway' and label='person' and observed_at > now()-interval '72 hours' group by observed_at),
g as (select t, n, case when t - lag(t) over (order by t) > interval '30 seconds'
  or lag(t) over (order by t) is null then 1 else 0 end newrun from f),
r as (select t, n, sum(newrun) over (order by t) run from g)
select count(distinct run), (select sum(peak) from (select max(n) peak from r group by run) z) from r;
```

Story breakdown: `select coalesce(drop_reason,'(published)'), count(*) from vision_motion_story where stream_id='walkway' and kind='person' and started_at > now()-interval '72 hours' group by 1;`

**A3. Council rows that share an artifact with a caption (item 3, copy rate).**

```sql
select c.created_at, c.narrative, v.narrative, v.tags
from vision_events c join vision_events v on v.event_type<>'camera_caption' and v.stream_id='walkway'
  and v.evidence_refs ? replace(c.evidence_refs->>0,'artifact:','')
where c.event_type='camera_caption' and c.stream_id='walkway';
```

**A4. "Driving" captions vs motion stories (item 3, cross-check).**

```sql
with caps as (select e.created_at, e.narrative,
  (select replace(x,'frame_ts:','')::timestamptz from jsonb_array_elements_text(e.evidence_refs) x
   where x like 'frame_ts:%' limit 1) fts
  from vision_events e where e.event_type='camera_caption' and e.narrative ~* '\mdriv')
select caps.created_at, string_agg(distinct coalesce(s.speed_class,'no_story'), ',')
from caps left join vision_crop_observation c on c.stream_id='walkway' and c.label='vehicle'
  and c.observed_at between caps.fts - interval '1 second' and caps.fts + interval '1 second'
left join vision_motion_story s on c.crop_id = any(s.crop_ids)
group by 1;
```

**A5. Item 1 re-checks.** FalkorDB (`GRAPH.RO_QUERY orion_substrate`):
`MATCH (a)-[r:observed_in]->(d) WHERE a.node_id STARTS WITH 'entity:vision_individual' RETURN count(r)` → 13;
`MATCH (n {node_id:'node:substrate.perception_walkway'}) OPTIONAL MATCH (n)-[r]-() RETURN count(r)` → 0.
SQL: `select count(*) from vision_percept_expectation` → 4;
`select count(*) from dream_replay_item where row_to_json(dream_replay_item)::text ilike '%walkway%'` → 0.
Logs: `docker logs --since 24h orion-athena-thought | grep -c reverie_street_context` → 98.
Council log notes: `docker logs --since 72h orion-athena-vision-council | grep grounding`.
