## Summary

- Walkway photos are now kept forever. Crop thumbnails, face frames and story pictures are no longer deleted by age. `0` means "keep forever" and is the new default for `VISION_CROP_THUMB_RETENTION_DAYS`, `VISION_FACE_FRAMES_RETENTION_DAYS` and `VISION_STORY_STRIP_RETENTION_DAYS`.
- "Who is this?" asks can show older pictures again. With keep-forever, `ask_image_refs` uses no age limit. This fixes `ask-9645cdc1bdd64660b80a07380b7c71ef`, which was posted with no picture: its only thumbnails were 3.6 days old, and the old rule only allowed thumbnails up to 3 days old.
- People get their own snapshot slot. The crop-thumbnail rate limit was one slot per camera every 10 s, shared by every label, so parked cars took most of it. It is now one slot per camera and label.
- Walkway stories keep their person links. A story records each crop's individual when it closes, but the individuals reducer often links the crops a few seconds later. The story loop now fills those empty links on later ticks. The Hub also fills them when it reads a story, which covers the few seconds before the story loop does.
- Adds a one-time backfill script for stories older than 6 h, with a snapshot and a restore. It was not run against production.

## Outcome moved

- Ask cards: an individual whose thumbnails are older than 3 days now gets a picture. Before this patch, the card had no picture at all.
- Thumbnail share (live 2026-10-16..20, one shared slot): vehicles had 32,799 crops and 197 thumbnails. People had 973 crops and 106 thumbnails. Juniper's 27-sighting individual had thumbnails on only 4 sightings. Each label now has its own slot.
- Story people (live, read-only dry run of the backfill, 2026-10-20): 2,243 stories from the last 7 days have 3,802 evidence crops that were linked after the story closed (person=111, vehicle=2,075, dog=25, stroller=20, bicycle=12). One of them is story `3ca2d1c15fd53c423a38400ddc582709`, whose 3 crops resolve to `cb2b4de4...` ("Juniper"). After this patch the Hub shows these people and their naming buttons.

## Current architecture

- orion-vision-host `ThumbStore` (app/crop_embeddings.py) deleted thumbnails after 10 days. `FaceFrameStore` (app/face_frames.py) deleted face frames after 14 days. A value of `0` meant "delete immediately".
- orion-sql-writer `prune_strips` deleted a story's pictures 30 days after a verdict. Story retention deleted unchecked story rows after 90 days, and then the orphan sweep deleted their picture folders.
- `ask_image_refs` only offered thumbnails from sightings that ended within (thumb retention - ask expiry) = 3 days.
- `ThumbRateLimiter.take(stream_id)` had one slot per stream.
- `vision_motion_story.evidence` was written once, when the story closed. The Hub (`people_in`/`_enrich`) read only that saved copy.

## Architecture touched

- orion-vision-host: thumbnail and face-frame pruners (0 = forever; leftover `.tmp` files from a crashed write are still removed after 1 day), and the per-label thumbnail key.
- orion-sql-writer: `ask_image_refs` keep-forever path; `prune_strips` and `story_prune_sql` keep-forever path; new `relink_stories` step in the motion-story tick.
- orion-hub: `_late_links` / `fill_late_links` in `_enrich` and in the POST person route.
- No bus, schema or migration changes.

## Design choice: where the story link lives

The crop rows are deleted after 7 days (`VISION_CROP_RETENTION_DAYS`), but stories live 90 days or longer. A link that is only looked up from crop rows at read time would disappear after 7 days. So the durable copy is written into the story row by the producer: `relink_stories` runs on every motion-story tick, covers stories that ended in the last 6 h, and fills only empty links. It updates a row only if its evidence is still exactly what it read. The Hub read is a short-lived extra that covers the time before the next tick. It runs inside a savepoint and fails open. Older stories (6 h to 7 days) need the one-time backfill, which must run before their crop rows are deleted.

## Files changed

- `services/orion-vision-host/app/crop_embeddings.py`: 0 = keep forever; `thumb_slot_key(stream, label)`; docstrings.
- `services/orion-vision-host/app/face_frames.py`: 0 = keep forever (new default).
- `services/orion-vision-host/app/runner.py`, `app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: defaults set to 0, with docs.
- `services/orion-vision-host/tests/test_crop_thumbnails.py`, `tests/test_face_frames.py`: tests for keep-forever and for a vehicle not blocking a person.
- `services/orion-sql-writer/app/vision_individuals.py`: `ask_image_refs` uses no age limit at 0; config default 0.
- `services/orion-sql-writer/app/vision_motion_loop.py`: `relink_stories`, `fill_evidence_links`, `story_prune_sql`, keep-forever `prune_strips`.
- `services/orion-sql-writer/app/vision_story_strip.py`, `app/settings.py`, `.env_example`, `README.md`: defaults set to 0, with docs.
- `services/orion-sql-writer/scripts/backfill_story_evidence_links.py`: new one-time backfill (dry run, `--apply`, `--restore`).
- `services/orion-sql-writer/tests/test_vision_story_strip.py`, `test_vision_motion.py`, `test_vision_confirm_asks.py`, `test_vision_walkway_zones_thumbs_cursor.py`: tests.
- `services/orion-hub/scripts/motion_story_routes.py`: read-time late links.
- `services/orion-hub/scripts/ask_routes.py`: docstring.
- `services/orion-hub/tests/test_motion_story_routes.py`: a story frozen before its crops were linked lists the person and can be named.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: `vision_motion_story.evidence[*].individual_id` can change from null to an id after the story is written. Null-only: an existing id is never changed. With strip retention 0, story retention skips stories that still have a saved picture.
- Compatibility notes: the Hub reads the same fields as before.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- Changed meaning or default: `VISION_CROP_THUMB_RETENTION_DAYS` (vision-host and sql-writer) 10 -> 0; `VISION_FACE_FRAMES_RETENTION_DAYS` 14 -> 0; `VISION_STORY_STRIP_RETENTION_DAYS` 30 -> 0. `0` now means keep forever. Before this patch, `0` meant delete immediately on vision-host.
- `.env_example` updated: yes (orion-vision-host, orion-sql-writer).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: ran it. These keys are outside its default scope, and it never overwrites a value that differs. **The live `.env` values (10 / 14 / 30) were deliberately left unchanged.** Old vision-host code reads `0` as "delete everything now". If vision-host restarted on the old image with `0` in `.env`, every thumbnail and face frame would be deleted. Change the values to 0 only after the new images are deployed (commands below).
- skipped keys requiring operator action: the three keys above, in the live `.env` files, after deployment.

## Tests run

```text
orion-vision-host (in the service image, PYTHONPATH=repo): 368 passed, 1 skipped
orion-sql-writer full suite (throwaway Postgres): 14 failed, 980+ passed. The same 14 tests fail on clean main; they depend on the environment and are not related to this patch.
orion-sql-writer vision tests with ORION_TEST_PG_URI (confirm_asks, split, motion, story_strip, individuals, zones_thumbs_cursor): all pass
orion-hub tests/test_motion_story_routes.py with ORION_MEMORY_EPISODE_TEST_DATABASE_URL: 23 passed
Mutation checks: the new tests fail when relink is disabled, when the keep-forever guards are removed, and when the Hub late-link lookup is disabled.
Repo gates: check_env_template_parity PASS; env/compose/single-source parity tests: 56 passed.
```

## Evals run

```text
No eval harness covers retention or story linking. Live evidence instead: a read-only dry run of the backfill against production found 2,243 stories / 3,802 crops to fill, including story 3ca2d1c1... -> Juniper's individual.
```

## Docker/build/smoke checks

```text
Not deployed (by instruction). vision-host tests ran inside the orion-vision-host-vision-host image.
```

## Review findings fixed

A code-review subagent ran on commit adb9b7ebe.

- Finding (must-fix 1+2): old vision-host code reads `0` as "delete everything". The sync script also never overwrites a value that differs, so the live `.env` stays at 10/14/30.
  - Fix: the live `.env` was deliberately not changed. "Restart required" below gives the order: deploy the new images first, then switch the 4 keys to 0. Both services switch `VISION_CROP_THUMB_RETENTION_DAYS` in the same step.
  - Evidence: `test_retention_zero_never_deletes_a_photo` and `test_default_retention_keeps_frames_forever` cover the new code.
- Finding (3): the keep-forever DELETE referenced `evidence_strip` even when the strip migration was missing, so every tick would fail.
  - Fix: `story_prune_sql(cfg, strip_columns=...)` falls back to the plain prune when the columns are missing, and the prune is wrapped in try/except with a warning.
  - Evidence: `test_strips_are_kept_forever_by_default`.
- Finding (4): with strips switched off, story retention deleted stories that had pictures.
  - Fix: `LoopConfig.strip_keep_forever` is taken from `VISION_STORY_STRIP_RETENTION_DAYS`, not from whether strips are on.
  - Evidence: same test (`strip=None` keeps the clause).
- Finding (5): the relink query did not use the index, and the story table now grows.
  - Fix: the query filters on `stream_id = ANY(:streams)`.
  - Evidence: live EXPLAIN shows a Bitmap Index Scan on `vision_motion_story_stream_time_idx`, 4.2 ms for 441 candidate rows.
- Finding (6): the reviewer said CI does not run the new Postgres tests.
  - Fix: not needed. `orion-sql-writer-tests.yml` already runs `test_vision_story_strip.py` and `test_vision_motion.py` with `ORION_TEST_PG_URI`, the split migration and the strip_v2 migration, and it fails on any skip. These tests were also run locally against a throwaway Postgres (results above).
- Finding (7): there was no daily figure for disk growth.
  - Fix: a bound is now stated. Thumbnails are written only on triggered frames. The shared slot used about 75 per day (303 over 4 days). With one slot per label, the limit is at most 5 labels x that rate, about 400 per day at about 5 KB, so 2 MB/day or less. Face frames are about 170 MB/day. Follow-up: a disk-usage alarm on `/mnt/telemetry`.
- Finding (8): the jsonpath missed entries that had no `individual_id` key.
  - Fix: the jsonpath is now `!exists(@.individual_id) || @.individual_id == null` in both the loop and the backfill.
- Finding (10): restore could skip rows without saying so.
  - Fix: it now prints "restored N of M" and exits 1 on a shortfall.
  - Evidence: the backfill test asserts this.
- Finding (11, nit): the Hub runs one lookup per companion story. Left as is: it is correct, and there are only a few companion stories per card.

## Restart required

Deploy from the clean primary main checkout after merge:

```bash
scripts/safe_docker_build.sh orion-vision-host up -d --build
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

Then switch on keep-forever in the live `.env` files and restart:

```bash
sed -i 's/^VISION_CROP_THUMB_RETENTION_DAYS=.*/VISION_CROP_THUMB_RETENTION_DAYS=0/; s/^VISION_FACE_FRAMES_RETENTION_DAYS=.*/VISION_FACE_FRAMES_RETENTION_DAYS=0/' services/orion-vision-host/.env && scripts/safe_docker_build.sh orion-vision-host up -d
sed -i 's/^VISION_CROP_THUMB_RETENTION_DAYS=.*/VISION_CROP_THUMB_RETENTION_DAYS=0/; s/^VISION_STORY_STRIP_RETENTION_DAYS=.*/VISION_STORY_STRIP_RETENTION_DAYS=0/' services/orion-sql-writer/.env && scripts/safe_docker_build.sh orion-sql-writer up -d
```

One-time backfill (run within 7 days, while the crop rows exist). Dry run first, then apply:

```bash
python services/orion-sql-writer/scripts/backfill_story_evidence_links.py --dsn postgresql://postgres:postgres@localhost:55432/conjourney --job-dir /tmp/story-evidence-relink
python services/orion-sql-writer/scripts/backfill_story_evidence_links.py --dsn postgresql://postgres:postgres@localhost:55432/conjourney --job-dir /tmp/story-evidence-relink --apply
```

## Risks / concerns

- Severity: medium. Concern: deploy order. Old vision-host code reads `0` as "delete now". Mitigation: the live `.env` was not changed; the `sed` step runs only after the new image is up.
- Severity: low. Concern: disk growth. Face frames are about 170 MB/day (2.4 GB over 14 days measured), about 62 GB/year, against 336 GB free. Thumbnails grow by 2 MB/day or less. Mitigation: any positive retention value turns pruning back on. Follow-up: a disk-usage alarm.
- Severity: low. Concern: story rows that still have a picture are no longer deleted at 90 days. Mitigation: the rows are small, and strips have only existed since 10-17 (28 rows so far).

## PR link

(filled after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
