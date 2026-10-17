# Walkway stories: a picture of the moment, a dark-scene flag, and "Can't tell"

## Summary

- Each published walkway story now gets its own pictures, taken from the real frames of the walk-by: up to 4 crops around the person (256 px tall) and one full frame (640 px wide) with all the story's boxes drawn. The Hub card shows these, labelled "Picture from <time>". It no longer draws boxes over today's view.
- orion-sql-writer saves the pictures. Its motion loop already ticks every 15 s, and the walkway frames live only about 60 s, so each tick first copies a crop and a small frame for every new walker box, then builds the story's strip when the story closes. A box whose frame was already gone says "frame already gone". Nothing is faked.
- Each strip carries two brightness numbers. The story is marked `dark` when the area where the person walked is darker than 0.06 (median brightness, 0 = black, 1 = white). Dark stories are still published. The card's new "only checkable (daylight, with picture)" filter is on by default, so Juniper's queue holds only stories she can judge.
- A fifth button, "Can't tell", appears on stories with no picture or a dark scene. It records that she could not judge the story. It removes the story from the "unchecked" queue and never counts as right or wrong. The tally line shows "can't tell N (not counted)".
- The Hub serves the pictures through a new route that accepts only a 32-hex story id and the five file names the writer creates, and never follows a link.

## Outcome moved

Before: the one published story (00:47 UTC, dusk) had no picture of its own. At night the card showed its boxes over a dark, current view, so Juniper could not tell a person walked. Her spot-check, which gates stories into Orion, could not proceed.

After: every story published after deploy carries the pictures of its own moment, kept until she has checked it. Stories she cannot judge are filtered out of her queue by default, or marked "Can't tell" without moving the tally.

## Current architecture

- `vision_crop_observation` rows (one per detected box, `observed_at` = the retina's `frame_ts`) feed `app/vision_motion_loop.py`. It links boxes into passages and writes `vision_motion_story` rows about 41 s or more after a passage's last box.
- Frames: the walkway retina writes `walkway_walkway_<epoch_ms>_<hex>.jpg` into `/mnt/telemetry/vision/frames`. The edge prunes that shared directory at its own 60 s retention (measured live: oldest walkway frame about 60 s old).
- Hub card `static/js/walkway-stories.js` drew a box map over `/api/vision/frames/latest/image` (today's view) plus vision-host crop thumbnails, which exist for about 6% of boxes.

## Architecture touched

- **Which service saves the strips, and why: orion-sql-writer.**
  - It is the only service that knows which boxes become a published story.
  - Its 15 s tick reaches each new box while the box's frame still exists. A crop row lands about 2 s after its frame, and frames live about 60 s.
  - Story close and strip retention stay in the service that owns the story table.
  - vision-host was the other option. It has each frame in memory, but it knows nothing about passages, so it would have to save pictures of every box blind, and a third service would have to join them.
  - Cost: sql-writer now mounts the frames directory read-only and the strip directory read-write.
- orion-sql-db: one additive migration (`manual_migration_vision_motion_story_strip_v2.sql`).
- orion-hub: one new route, a list filter, the "Can't tell" verdict, the card, and the template checkbox.
- No bus channel and no schema-registry change (same pattern as slice 1: a database loop that the Hub reads).

## Files changed

- `services/orion-sql-writer/app/vision_story_strip.py` (new): frame lookup by name, capture to `_pending`, strip assembly, luminance, safe directory removal, pending prune.
- `services/orion-sql-writer/app/vision_motion_loop.py`: the capture step (before stories), the assemble step (after), strip retention and the orphan sweep, the `[STORY_STRIP]` trace, and a check for missing columns (stories keep running if the migration is missing).
- `services/orion-sql-writer/app/settings.py`, `.env_example`, `docker-compose.yml`: 6 keys, ON by default; frames mounted `:ro`, strip directory mounted read-write.
- `services/orion-sql-writer/requirements.txt`: `Pillow==11.3.0`.
- `services/orion-sql-writer/README.md`: the new section, with the darkness measurements.
- `services/orion-sql-writer/tests/test_vision_story_strip.py` (new): 10 pure tests + 5 real-Postgres tests.
- `services/orion-sql-writer/evals/run_story_strip_darkness_eval.py` (new): a read-only coverage and darkness-by-verdict eval.
- `services/orion-sql-db/manual_migration_vision_motion_story_strip_v2.sql` (new).
- `services/orion-hub/scripts/motion_story_routes.py`: the strip route, `checkable_only`, `cant_tell`, tally `cant_tell`.
- `services/orion-hub/app/settings.py`, `.env_example`, `docker-compose.yml`: `HUB_VISION_STORY_STRIP_DIR`, mounted read-only.
- `services/orion-hub/static/js/walkway-stories.js` (+ `.test.js`), `templates/index.html`: the card.
- `services/orion-hub/tests/test_motion_story_routes.py`, `test_walkway_stories_browser_smoke.py`: the new behaviour.
- `.github/workflows/orion-sql-writer-tests.yml`, `orion-memory-episode-tests.yml`: the v2 migration and the new tests in lanes that fail if they skip.

## Schema / bus / API changes

- Added columns:
  - `vision_motion_story.evidence_strip` jsonb, `frame_luminance` real, `scene_luminance` real, `dark` boolean.
  - `vision_motion_story_verdict.cant_tell` boolean.
- Added HTTP:
  - `GET /api/vision/motion-stories/{story_id}/strip/{crop_0..3|frame}.jpg`.
  - `GET /api/vision/motion-stories?checkable_only=true|false` (default true).
  - `POST .../verdict {"verdict": "cant_tell"}`.
- Behaviour changed:
  - `unchecked_only` now means "no verdict row at all", so a "Can't tell" story leaves the queue.
  - The list returns `verdict: "cant_tell"` for those stories.
- Compatibility:
  - The migration is additive and passes the post-pull auto-apply filter. `auto_apply_refusals` returns `[]`, both with and without the live catalog.
  - "Can't tell" uses a new column, not a wider CHECK, because changing the CHECK would need `DROP CONSTRAINT`, which the auto-apply refuses.

## Env/config changes

- Added keys (orion-sql-writer):
  - `VISION_STORY_STRIPS_ENABLED=true`
  - `VISION_STORY_FRAMES_DIR=/mnt/telemetry/vision/frames`
  - `VISION_STORY_STRIP_DIR=/mnt/telemetry/vision/story_strips`
  - `VISION_STORY_STRIP_RETENTION_DAYS=30`
  - `VISION_STORY_STRIP_PENDING_SEC=3600`
  - `VISION_STORY_STRIP_DARK_LUMINANCE=0.06`
- Added keys (orion-hub): `HUB_VISION_STORY_STRIP_DIR=/mnt/telemetry/vision/story_strips`.
- Removed / renamed: none.
- `.env_example` updated: yes.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys orion-sql-writer orion-hub`: yes. All 7 keys were added to the primary checkout's `.env` files. No `--force`.
- Skipped keys: none.

## Darkness threshold (measured)

- Live walkway frame at 20:12 MDT (02:12 UTC): the camera was still in colour (colour spread 14.5, not IR). The porch light kept the whole-frame mean at 0.255. So the whole-frame mean cannot say "dark": it is stored, but it is not used for `dark`.
- The same night frame, inside the padded areas where the 10 clean non-prop stories happened: the median was 0.004-0.03 on the street, and 0.08-0.12 where a lit porch pillar fills the area.
- Daytime and dusk crop thumbnails from the clean period (13:09-18:47 MDT, n=31, box only): the median was 0.11-0.52. The low end is a dark-clothed person, not a dark scene.
- Threshold: 0.06. It sits between the night street (≤ 0.03) and the darkest daytime box (0.11).
- **UNVERIFIED:** no daytime full frame existed to measure, because frames live 60 s. The 18:47-20:12 MDT twilight, and whether the camera ever switches to IR, are not measured. `evals/run_story_strip_darkness_eval.py` re-checks 0.06 against Juniper's "Can't tell" verdicts once she has some.

## Tests run

```text
<filled below>
```

## Evals run

```text
<filled below>
```

## Docker/build/smoke checks

```text
<filled below>
```

## Review findings fixed

<filled below>

## Restart required

Deploy from the clean primary `main` checkout after the merge. The migration auto-applies on pull. Then:

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

Live check after deploy:
- The sql-writer logs show no `[STORY_STRIP] missing columns`.
- `ls /mnt/telemetry/vision/story_strips/_pending` fills while someone walks past.
- After the next published walk-by, `[STORY_STRIP] story_id=... crops_saved=4 frame=True` appears.
- On "Walkway (live)", the card shows "Picture from <time>".

## Risks / concerns

<filled below>

## PR link

<filled below>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
