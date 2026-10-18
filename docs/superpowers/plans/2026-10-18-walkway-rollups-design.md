# Walkway rollups by hour, day, week and month

Juniper, 2026-10-18 (explicit build): "build the whole daily walkway summary, but actually
roll that into hour intervals, and also have a day version, a week version, and a month
version." Tracking list: `docs/superpowers/plans/2026-10-18-walkway-reach-into-orion-queue.md`
(branch `docs/walkway-reach-queue`), item "Memory and the day letter" and the named-people links.

## What it does

The outdoor camera's sightings are counted by code into one row per hour, day, week and
month (`vision_rollup`). Each row holds the numbers and one plain sentence. Days are built
from their hours, weeks and months from their days, so every grain agrees. Hours the camera
did not deliver frames say "camera was down", never "nobody came".

Five readers use the rows in this patch:

| Reader | What it gets |
|---|---|
| Nightly walkway journal (orion-actions, 22:30) | Day entry every night, week entry Sunday, month entry on the last day; the numbers are the rollup's, and composed prose that states any other number is replaced by the template line. |
| Day letter (`orion/orion_day/gather.py`) | The day's line, cited as `[walkway_day:<date>]`. |
| Curiosity study material | The newest day's line, next to the 6 h captions. |
| Hub "Walkway (live)" view | "Walkway history" panel, tabs Hours / Days / Weeks / Months. |
| Substrate (orion-substrate-runtime) | One day event node per complete day, with an `observed_in` edge from each named person's existing entity node. |

Not touched: attention (stays fenced), episodes. Dreams: follow-up (a replay source needs a
dream-side candidate shape; not trivial).

## Counting rules (`orion/vision/walkway_rollup.py`)

- **Passing** = one individual seen in one clock hour. Day = sum of hours; week/month = sum of days.
- **Distinct** = distinct individual clusters in the period.
- **Not counted** (tallied under `excluded`): stationary sightings/individuals, prop ids
  (`VISION_MOTION_PROP_INDIVIDUAL_IDS`), no-embed zones, and vehicles seen in 4+ different
  hours of the trailing 24 h (`parked_repeat`).
- **Camera minutes** = minutes with a `vision_scene_inventory` row. < 15 of 60 = down;
  < 50 = partial. A child period with no row = "No record".
- **Peak at once** = most of each kind in one vision window (`vision_scene_inventory.counts`).
- **Regulars** = unnamed individuals seen on 2+ days (day: also in one of the 6 earlier day
  rows; week/month: 2+ days inside the period).
- **Usual** = mean over earlier periods where the camera was up (hour: same local hour on up
  to 7 earlier days; day: 7 days; week: 4; month: 3), only with 3+ periods.
- **Captions**: distinct `camera_caption` rows, 3 per hour, 5 per day/week/month, always
  labelled as one-frame captions.
- **Stories**: included only when the motion-story spot-check gate passed
  (`orion/vision/motion_story_gate.py`, shared with the Hub card). Today: closed (8 checked).

## Metric quality gate

1. **Provenance.** Passings/distinct/excluded: `vision_individual_sighting` + `vision_individual`
   (written by `services/orion-sql-writer/app/vision_individuals.py`). Camera minutes and peaks:
   `vision_scene_inventory` (one row per vision window, persisted by orion-sql-writer from
   orion-vision-window). Captions: `vision_events` `camera_caption` (orion-vision-window).
2. **Independence.** Passings and distinct come from the same sightings: two views of one
   source, not two signals. Camera minutes and peaks come from a different producer (scene
   inventory, written whether or not anything was seen), so they can check the sightings.
3. **Theory anchor.** Counting distinct tracked objects per time bin (occupancy counting).
   Coverage is kept apart from the count because a missing frame is missing data, not a
   zero: an hour without frames cannot be "empty".
4. **Live data.** Read-only replay of 10-15..10-17 (below). Not degenerate: 0 to 92 passings
   per hour; nights reach a true rest state (camera up, 0 passings, "nobody passed").
   Camera-down hours match the scene-inventory gaps (10-17 09:00-12:00 MDT, 10-14 before the
   camera was mounted). **Found during the check:** the individuals reducer over-splits people
   (10-16 12:00 MDT: 79 person clusters in an hour while no frame held more than 5 people).
   So passings are an upper bound; `peak_at_once` (other producer) is the lower bound, and
   lines show it for the busiest period. Vehicle peaks include parked cars and are not shown.
5. **Existing mechanism.** `orion/situational/camera_eyes.py` counts distinct individuals since
   local midnight (no history, no grains). `vision_percept_expectation` is per-subject
   predictions (empty today). Nothing summarized the walkway over time.
6. **Reversibility.** One new table (`DROP TABLE vision_rollup`); every reader tolerates its
   absence; kill switches `VISION_ROLLUP_ENABLED`, `WALKWAY_SEEN_ENABLED`. Nothing is baked
   into a training default.

## Schema / API

- New table `vision_rollup` (`services/orion-sql-db/manual_migration_vision_rollup_v1.sql`,
  auto-apply clean). PK (stream_id, grain, period_start).
- New journal trigger kinds `walkway_day`, `walkway_week`, `walkway_month`
  (`orion/journaler/schemas.py`, `worker.py`, `dispatch_registry.py`: digest, no email).
- `OrionDayMaterialV1.walkway_day` (optional, default None; same rollout rule as
  `corrections`: rebuild orion-durable-runs before or with orion-hub).
- `StudyMaterial.street_day` (dataclass, Hub-internal).
- `GET /api/vision/rollups?stream_id=&grain=&limit=` (Hub).
- No bus channel or schema-registry change.

## Acceptance checks

- `pytest services/orion-sql-writer/tests/test_vision_rollup.py` (DST 23/25 h days, week across
  a month boundary, exclusions, camera-down vs empty, build order, real Postgres with
  `ORION_TEST_PG_URI`).
- `python services/orion-sql-writer/evals/run_vision_rollup_replay_eval.py --days 3` PASS.
- Journal guard test: "empty on at least five occasions" is replaced by the template.
- Hub browser smoke at 360 px.

## Known limits / follow-ups

- Passings over-count people when the individuals reducer splits one person (see gate 4).
- Retina decode corruption is only in the retina's live /health; it shows here only as
  missing minutes when frames are dropped.
- Dreams replay source: not built.
- Labels are read at build time: a rename shows up in rows rebuilt after it (recent hours,
  today), not in closed older rows.
