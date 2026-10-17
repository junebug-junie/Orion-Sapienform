# Walkway stories: a picture of the moment, a dark-scene flag, and "Can't tell"

## Summary

- Each published walkway story now gets its own pictures, taken from the real frames of the walk-by: up to 4 crops around the person (256 px tall) and one full frame (640 px wide) with all the story's boxes drawn. The Hub card shows these, labelled "Picture from <time>". It no longer draws boxes over today's view.
- orion-sql-writer saves the pictures. Its motion loop already ticks every 15 s, and the walkway frames live only about 60 s, so each tick first copies a crop and a small frame for every new walker box, then builds the story's strip when the story closes. A box whose frame was already gone says "frame already gone". Nothing is faked.
- Each strip records how much visible detail the person's crops hold (contrast). The story is marked `dark` when that contrast is below 0.10. Brightness alone could not tell a lit-but-empty night lawn from a dark coat in daylight, so contrast is what decides. Dark stories are still published. The card's new "only checkable (daylight, with picture)" filter is on by default, so Juniper's queue holds only stories she can judge.
- A fifth button, "Can't tell", appears on every story (the review asked for it on all of them: an unclear daylight picture should not be forced into "wrong"). It records that she could not judge the story. It removes the story from the "unchecked" queue and never counts as right or wrong. The tally line shows "can't tell N (not counted)".
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
- `services/orion-sql-writer/tests/test_vision_story_strip.py` (new): 12 pure tests + 8 real-Postgres tests.
- `services/orion-sql-writer/evals/run_story_strip_darkness_eval.py` (new): a read-only coverage and darkness-by-verdict eval.
- `services/orion-sql-db/manual_migration_vision_motion_story_strip_v2.sql` (new).
- `services/orion-hub/scripts/motion_story_routes.py`: the strip route, `checkable_only`, `cant_tell`, tally `cant_tell`.
- `services/orion-hub/app/settings.py`, `.env_example`, `docker-compose.yml`: `HUB_VISION_STORY_STRIP_DIR`, mounted read-only.
- `services/orion-hub/static/js/walkway-stories.js` (+ `.test.js`), `templates/index.html`: the card.
- `services/orion-hub/tests/test_motion_story_routes.py`, `test_walkway_stories_browser_smoke.py`: the new behaviour.
- `.github/workflows/orion-sql-writer-tests.yml`, `orion-memory-episode-tests.yml`: the v2 migration and the new tests in lanes that fail if they skip.

## Schema / bus / API changes

- Added columns:
  - `vision_motion_story.evidence_strip` jsonb, `frame_luminance` real, `scene_luminance` real, `scene_contrast` real, `dark` boolean.
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
  - `VISION_STORY_STRIP_DARK_CONTRAST=0.10`
- Added keys (orion-hub): `HUB_VISION_STORY_STRIP_DIR=/mnt/telemetry/vision/story_strips`.
- Removed / renamed: none.
- `.env_example` updated: yes.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys orion-sql-writer orion-hub`: yes. All 7 keys were added to the primary checkout's `.env` files. No `--force`. An earlier draft key, `VISION_STORY_STRIP_DARK_LUMINANCE`, was added by the first sync and then removed by hand when the measurement changed to contrast. It was never in any merged code.
- Skipped keys: none.

## Darkness threshold (measured)

The threshold is a contrast, not a brightness. Here is why, from live data:

- **Whole frame (night, 20:12 MDT):** the camera was still in colour (colour spread 14.5, so not IR). The porch light held the whole-frame mean at 0.255. The whole-frame mean never reads dark, so it is stored as a diagnostic only.
- **Brightness of the person's area (night):** the first design used the median brightness of the walk area, with a threshold of 0.06. I ran the real code on tonight's live frames at the 00:47 story's place, and it read 0.116, so it was not marked dark. But the crop showed nothing a person could judge: a porch-lit lawn. A dark-clothed person in daylight reads 0.114 (13:33 MDT thumbnail). Brightness cannot tell these two apart.
- **Contrast (RMS contrast, the standard deviation of brightness; theory: visibility of detail follows contrast, not mean level, Peli 1990):**
  - Night: all 10 clean non-prop story places on the live night frame read 0.012-0.066, run through this code (`assemble_strip` on real walkway frames found by name).
  - Day and dusk: the 31 clean crop thumbnails from 13:09-18:47 MDT, resized the same way, read 0.133-0.30 (p10 0.187).
- **Threshold: 0.10.** It sits in the gap between the night maximum (0.066) and the day minimum (0.133).

**UNVERIFIED:**
- No daytime full frame existed to measure, because frames live 60 s. The day side comes from box-only thumbnails, not padded crops.
- The 18:47-20:12 MDT twilight was not measured, and neither was whether the camera ever switches to IR.
- `evals/run_story_strip_darkness_eval.py` re-checks 0.10 against Juniper's "Can't tell" verdicts once she has some.

## Metric quality gate (scene_contrast, frame_luminance)

1. **Where the number comes from:** `vision_story_strip.luma_stats` reads the luma histogram of each saved crop JPEG. `assemble_strip` takes the median of those values over the crops. If no crop was saved, it measures each box's padded area on the 640 px frame instead.
2. **Independence:** this is the only pixel-derived number on a story. Every other story field is box geometry. `frame_luminance` (whole-frame mean) comes from a different area of the same frame, so it is not redundant. It is a diagnostic only: nothing reads it.
3. **Theory:** RMS contrast measures how visible detail is (Peli 1990). The yes/no question is "can a human see the person", and contrast answers it.
4. **Live data:** the night and day ranges are above, and they do not overlap. A calm or rest state is not the issue here. A "flat" crop (no contrast) is exactly the dark case, and it is reachable.
5. **Existing mechanism:** nothing in the repo measured scene darkness for the walkway. vision-host thumbnails store no brightness or contrast.
6. **Reversibility:** one nullable column plus one env key. The Hub filter reads only `dark`, so the threshold can change without touching the schema.

## Tests run

```text
services/orion-sql-writer/tests/test_vision_story_strip.py + test_vision_motion.py
  (real Postgres, ORION_TEST_PG_URI, throwaway postgres:16)          53 passed
services/orion-hub/tests/test_motion_story_routes.py (real Postgres)  12 passed
node --test static/js/walkway-stories.test.js                          7 passed
test_walkway_stories_browser_smoke.py (Chromium, 360 px)             passed 3/3 reruns
orion-static-gates.yml, every run step executed locally (34 steps)    all green
check_async_routes_not_blocking / check_migration_required_by / check_env_template_parity /
  check_env_key_single_source / check_settings_defaults orion-hub     PASS
auto_apply_refusals(v2 migration), with and without live catalog       []
Full sql-writer suite: 14 failures, identical set on clean main
  (biometrics_*, chat_history_response_identity, gpu_pool_state_history, journal_entry,
   notify_attention_*, subscription_matches_declared_consumers -- local-env sensitive)
```

## Evals run

```text
Live darkness measurement (READ ONLY, scratch copies only, no DB writes):
  real walkway frames found by name for 10 clean non-prop story places, run through
  capture_pending + assemble_strip:  contrast 0.012-0.066, all dark=True
  31 clean daytime/dusk crop thumbnails through luma_stats: contrast 0.133-0.30
run_story_strip_darkness_eval.py against the throwaway DB: runs; reports coverage and the
  verdict split; correctly flags a strip whose frame file is missing (exit 1).
Not run against live: the live DB does not have the v2 columns until the migration applies.
```

## Docker/build/smoke checks

```text
docker build -f services/orion-sql-writer/Dockerfile (worktree)   OK; in the image:
  import PIL (11.3.0), app.vision_story_strip, app.vision_motion_loop  OK  (test image removed)
docker compose config (sql-writer, hub) with the primary .env files: frames mounted read_only,
  story_strips mounted (rw for sql-writer, ro for hub), all 7 keys present
No deploy, no restart, no live DB writes (task rule).
Live frame-name matching of a NEW crop: UNVERIFIED. A read-only probe watched for
  walkway crops for 90 min after 02:20 UTC; nobody walked past at night.
```

## Review findings fixed

The code review ran in a read-only subagent on the first commit. It found no blockers. Path handling, SQL, async and XSS were all clean.

- **Finding: the darkness number measured the scene, not the walker; lit night areas passed.**
  - Fix: `dark` now comes from crop contrast (see "Darkness threshold"). I had found the same failure live just before the review returned.
  - Evidence: `test_a_lit_lawn_at_night_is_dark_although_it_is_as_bright_as_a_dark_coat_by_day`, plus the live night run (10/10 dark).
- **Finding: the gate could pass on daylight stories only, and an unclear lit picture was forced into "wrong".**
  - Fix: "Can't tell" is now on every story. The tally returns `not_checkable`, and the gate line says how many dark or picture-less stories it does not cover.
  - Evidence: JS test `gate line says how many...`; PG test (`not_checkable == 4`); smoke.
- **Finding: the capture cursor could skip a crop for good.** `created_at` is the start time of the inserting transaction, so a row can commit after the cursor has already passed it.
  - Fix: each read re-covers 10 s behind the cursor, and it skips the newest second.
  - Evidence: `test_a_crop_committed_late_behind_the_cursor_is_still_captured` (real Postgres).
- **Finding: the prop filter never fired, because fresh crops have no `individual_id` yet.** The patio skeleton would be decoded on every frame.
  - Fix: capture matches crops to stationary or prop boxes by overlap, and records `prop_overlap`.
  - Evidence: `test_prop_boxes_are_not_captured_and_say_why`.
- **Finding: a crop that landed after the first capture read could be marked `not_captured` for good.**
  - Fix: capture runs again just before strips are built.
- **Finding: a strip that kept failing stayed NULL forever, with a half-written folder.**
  - Fix: after half the pending window, the story gets a final `{"error": ...}` record, and the folder is removed.
  - Evidence: `test_a_strip_that_keeps_failing_gets_a_final_error_record`.
- **Finding: "never deleted while unchecked" was overstated.**
  - Fix: README, settings and `.env_example` now say the true rule. A strip is kept until 30 days after its story AND a verdict exists ("Can't tell" counts), or until story retention deletes the story row itself.
- **Nits fixed:**
  - Crops are now copied atomically.
  - `retention_days` is carried in the record, and the card reads it instead of hard-coding 30.
  - `frame_luminance` is labelled a diagnostic.
  - The smoke also checks that the page itself has no sideways scroll at 360 px.
- **Not changed:** the Hub route checks for a symlinked folder, then opens the file with `O_NOFOLLOW`. A fully race-free `dir_fd` open is not needed, because only sql-writer writes there. Noted as low.

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

- Severity: medium. **The capture race is UNVERIFIED live.**
  - Concern: frames live about 60 s. Capture reaches a crop within about 17 s (a 15 s tick plus about 2 s for the crop row to land).
  - Mitigation: if sql-writer is down or behind, the strip says "frame already gone" or "not saved"; it never shows a wrong picture. After deploy, `run_story_strip_darkness_eval.py` shows the saved vs `frame_gone` split.
- Severity: medium. **The day side of the threshold comes from box-only thumbnails (n=31), not full frames.**
  - Mitigation: a re-check against "Can't tell" verdicts through the eval. The threshold is one env key.
- Severity: low. **sql-writer now decodes one full-size walkway JPEG per detected walker frame.** That is about 0.2 per second at most, and props are skipped. Disk use is about 5 small JPEGs (about 100 KB) per published story, plus an hour of pending files.
- Severity: low. **Stories published before deploy have no strip.** That includes the 00:47 UTC one. The card says "No picture saved for this story" for them, and "Can't tell" is available.
- Severity: low. **The local `.env` has `VISION_MOTION_STORY_STREAMS=walkway,doorbell`, which differs from the example.** That predates this patch. The capture code handles any stream by its frame prefix. Doorbell frame naming is **UNVERIFIED**.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2748

🤖 Generated with [Claude Code](https://claude.com/claude-code)
