# Walkway motion stories, slice 1 + the Hub spot-check card (Hub only)

## Summary

- orion-sql-writer now turns each person who walks past on the walkway into one plain sentence, for example "A person with a stroller walked left to right and away from the house on the walkway, over 5 s (3 detections)." It uses arithmetic on the detector's boxes only: no model, no embeddings, no new identity.
- Boxes are linked into a passage by where they are and how fast they move, not by individual id. An individual id supplies only a name, and only under a strict rule. Every id on the passage must carry the same name. That name must cover at least 2/3 of the crops. No box may touch the frame edge.
- A box cut by the left, right or bottom frame edge gives no zone and no travel. Props (the patio skeleton, stationary individuals and their boxes) never become stories. Slice 1 publishes walking-pace stories only.
- The stories are **Hub only**. Nothing in Orion's cognition reads them. A new "Walkway stories" card sits under the live walkway picture. It shows each story with its boxes and any crop thumbnails, and it records Right / Wrong name / Wrong motion / Wrong other. It also shows the running tally against Juniper's gate: at least 50 checked, at least 80% right, zero wrong names.
- New tables `vision_motion_story`, `vision_motion_story_cursor` and `vision_motion_story_verdict`. The migration auto-applies on pull.

**Where it is in the Hub:** open the Hub home page and scroll to "Orion's Vision". In the camera drop-down, choose **"Walkway (live)"**. The "Walkway stories" card appears directly under the live picture, with the tally line at the top. Choosing any other view hides it.

## Outcome moved

Before this patch, Orion could say who was on the walkway, but nothing ever read two boxes of the same object in time order. Now each walk-by becomes a sentence you can trace back to its boxes, and Juniper has a way to measure whether those sentences are right before Orion sees them.

## Current architecture

- `vision_crop_observation` (one row per detected box) feeds `vision_individuals.py` (ids, sightings, stationary props) and `vision_rhythm.py`.
- The Hub's Vision panel had a "Walkway (live)" picture (PR #2687). It had no motion view.

## Architecture touched

- orion-sql-writer:
  - New pure module `app/vision_motion.py`.
  - New clocked loop `app/vision_motion_loop.py`, wired in `app/main.py`, ticking every 15 s.
- orion-sql-db: one additive migration.
- orion-hub:
  - New routes `scripts/motion_story_routes.py`.
  - New card `static/js/walkway-stories.js` plus a template block.
- No bus channel and no schema-registry entry. This follows the existing walkway reducers: a database loop in orion-sql-writer that the Hub reads.

## Files changed

- `services/orion-sql-writer/app/vision_motion.py`: linking, clipping, speed classes, zones, props, strict naming, together pairs, sentence.
- `services/orion-sql-writer/app/vision_motion_loop.py`:
  - The cursor-based reducer.
  - `ON CONFLICT DO NOTHING` inserts.
  - An advisory lock.
  - Retention (never deletes a story that has a verdict).
  - `MigrationMissing` back-off.
  - The `[MOTION_STORY]` trace line.
- `services/orion-sql-writer/app/main.py`: starts and stops the loop.
- `services/orion-sql-writer/app/settings.py`, `.env_example`, `docker-compose.yml`: 12 keys, with the kill switch ON.
- `services/orion-sql-writer/README.md`: the motion stories section.
- `services/orion-sql-writer/tests/test_vision_motion.py`: 33 tests, including real Postgres.
- `services/orion-sql-writer/evals/run_vision_motion_story_replay_eval.py`: read-only replay over live crops.
- `services/orion-sql-db/manual_migration_vision_motion_story_v1.sql`: 3 new tables and CONCURRENTLY indexes on the new tables. Header `REQUIRED-BY: orion-sql-writer, orion-hub`.
- `services/orion-hub/scripts/motion_story_routes.py` and `scripts/main.py`: list, verdict upsert and tally.
- `services/orion-hub/static/js/walkway-stories.js` (+ `.test.js`), `templates/index.html`: the card.
- `services/orion-hub/tests/test_motion_story_routes.py`: pure tests plus real Postgres.
- `services/orion-hub/tests/test_walkway_stories_browser_smoke.py`: Chromium at 360 px.
- `.github/workflows/orion-sql-writer-tests.yml`, `orion-memory-episode-tests.yml`, `schedule-browser-smoke.yml`: the new tests in lanes that fail if they skip.

## Schema / bus / API changes

- Added tables:
  - `vision_motion_story`: sentence, structured fields, `crop_ids`, and `evidence` jsonb (box, zone, clipped flag, individual id and thumb hash for each crop).
  - `vision_motion_story_cursor`.
  - `vision_motion_story_verdict`.
- Added HTTP:
  - `GET /api/vision/motion-stories?limit&clean_only&unchecked_only`.
  - `POST /api/vision/motion-stories/{story_id}/verdict {verdict}`.
- Removed / renamed: none.
- Behaviour changed: none for existing paths.
- Compatibility: additive. The Hub returns 503 `motion_story_schema_missing` until the migration lands.

## Env/config changes

- Added keys (orion-sql-writer):
  - `VISION_MOTION_STORIES_ENABLED=true`
  - `VISION_MOTION_STORY_INTERVAL_SEC=15`
  - `VISION_MOTION_STORY_CLOSE_GAP_SEC=11`
  - `VISION_MOTION_STORY_MIN_DETECTIONS=3`
  - `VISION_MOTION_STORY_STREAMS=walkway`
  - `VISION_MOTION_STORY_SETTLE_SEC=30`
  - `VISION_MOTION_STORY_INITIAL_LOOKBACK_SEC=3600`
  - `VISION_MOTION_STORY_RETENTION_DAYS=90`
  - `VISION_MOTION_FRAME_WIDTH=2560`
  - `VISION_MOTION_FRAME_HEIGHT=1920`
  - `VISION_MOTION_CLEAN_SINCE=2026-10-16T18:59:00Z`
  - `VISION_MOTION_PROP_INDIVIDUAL_IDS=fefb24d7e4074e9f827ad711874f6edf`
- Removed / renamed: none. No orion-hub keys.
- `.env_example` updated: yes.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys orion-sql-writer`: yes. All 12 keys were added to the primary checkout's `.env`.
- Diverged lines that already existed (not touched, need review by hand, unrelated to this PR): `SQL_WRITER_SUBSCRIBE_CHANNELS` and `SQL_WRITER_ROUTE_MAP_JSON`. Local and example differ in order. Local also has a duplicate `orion:chat:response:feedback`.

## Tests run

```text
services/orion-sql-writer/tests/test_vision_motion.py      33 passed (incl. real Postgres, ORION_TEST_PG_URI)
services/orion-hub/tests/test_motion_story_routes.py       8 passed (real Postgres)
node --test static/js/walkway-stories.test.js              5 passed
test_walkway_stories_browser_smoke.py (Chromium, 360 px)   passed 5/5 reruns
scripts/check_async_routes_not_blocking.py                 clean
scripts/check_migration_required_by.py                     OK
orion.sql_migration_drift.auto_apply_refusals              [] (known and unknown catalog); post-pull classify -> apply
check_env_template_parity / env_key_single_source / metric lineage / definition drift   PASS
CI (first push): 19/19 green; the Postgres lanes ran the new tests without skipping (144 + 18 passed)
```

Pre-existing failures that are not from this patch: 8 sql-writer tests fail the same way on `main` in the primary checkout. They are `test_biometrics_*`, `test_chat_history_response_identity_merge`, `test_gpu_pool_state_history` and `test_journal_entry_payload_boundary`. All are sensitive to the local environment.

## Evals run

```text
run_vision_motion_story_replay_eval.py (READ ONLY, clean frames 2026-10-16T18:59Z .. ~23:50Z)
  crops 4703, passages 312, published 3, glimpse 53, prop 252, parked 2, standing 1, edge_clipped 1
  names shown 0; withheld by the strict rule: two_named 1, other_id 2
  detected-frame interval p50 5.0 s, p90 5.15 s; stepwise == whole-window; PASS

--include-corrupt --since 2026-10-15 (calibration only, never accuracy)
  crops 24384, passages 2987, published 68 (person 66, dog 2, all walking), companion 3,
  out_of_scope 2, prop 1831, standing 163, glimpse 792; names shown 0; PASS
```

S9 and S10 from the design doc on the clean replay:
- **S9** is no longer a story. It was the parked car's box jittering.
  - The new gate scales with the gap, so the 10 s link breaks.
  - The middle box joins the parked car's own passage, which is a prop.
  - What is left is two 2-detection glimpses.
- **S10** is now "A person walked left to right and away from the house on the patio, over 27 s (12 detections)", with no name.
  - 8 of its 12 boxes touch the bottom edge.
  - It carries two named ids, so the name is withheld as `two_named`.
  - The 4 unclipped boxes shrink and move up the frame, so "away from the house" is plausible. It is **UNVERIFIED** against pixels.

Accuracy of every story is **UNVERIFIED** until Juniper's spot-check reaches the gate.

## Docker/build/smoke checks

```text
No docker build run here (no deploy per task). Compose change is one environment line with a default.
```

## Review findings fixed

The code review ran in a subagent, read only. It found no blockers. All 10 findings are fixed:

1. **An unnamed id from someone else could sit next to a named one and still get the name.** This is a wrong-name risk.
   - Fix: every non-null id must carry the same name. Otherwise the name is withheld with the new reason `other_id`.
   - Evidence: `test_name_needs_two_thirds_...` covers `["ind-a"]*4 + ["x","x"]`. The live replay now withholds 2 as `other_id`.
2. **Two ids with the same name counted as "two named".**
   - Fix: the rule compares names (case-folded), not ids.
   - Evidence: a new test case with two ids for "Alex".
3. **Scope was wider than "walker stories only".** Vehicle "crept" stories were published.
   - Fix: slice 1 publishes only person, dog, stroller and walked bicycle at walking pace. Everything else is stored as `out_of_scope`.
   - Evidence: `test_vehicles_are_out_of_scope_in_slice_one` and `test_three_fast_detections_run`.
4. **One walk-by could become two stories.** The person's story and the dog's or bicycle's own story were both published.
   - Fix: a non-person passage with a person partner is stored as `companion`.
   - Evidence: the together test. The calibration replay shows `companion 3`.
5. **Crops that land after the settle time are lost.**
   - Fix: documented as a known limit in the README.
6. **A new database engine every tick.**
   - Fix: `pool_size=1, max_overflow=1`.
7. **The stored name does not change after a rename.**
   - Fix: documented in the README.
8. **Retention pruned stories and orphaned their verdicts.**
   - Fix: the prune keeps every story that has a verdict.
   - Evidence: `test_retention_keeps_stories_juniper_has_checked` (real Postgres).
9. **The browser smoke could skip silently.**
   - Fix: the CI step runs with `-rs` and greps for `SKIPPED`.
10. **The Hub listed stories from every rules version, but the tally counted only the newest.**
    - Fix: the list is filtered to the tally's version.
    - Evidence: `test_tally_counts_only_the_newest_rules_version`.

I also found and fixed a flaky layout assertion in the smoke. It compared pixel positions while the page was still loading. It now checks document order.

## Restart required

Deploy order: the migration auto-applies on pull, then orion-sql-writer, then orion-hub. Run these from the clean primary `main` checkout after the merge:

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

Live check after deploy:
- `[MOTION_STORY] stream=walkway ... cursor=` appears in the sql-writer logs every 15 s.
- `SELECT count(*) FROM vision_motion_story WHERE published` grows when someone walks past.
- The card appears under "Walkway (live)".

## Risks / concerns

- Severity: medium. There are very few stories to check yet. Only 3 published stories exist on clean frames, because nobody walked past after the stream fix. The 50-story gate will take days of daylight.
- Severity: low. Thumbnails exist for only about 6% of crops. Most stories show only the box map, and the map is drawn over today's view, not the moment itself. Judging motion from boxes is possible. Judging a name needs a thumbnail.
- Severity: low. Motion toward or away from the camera is foreshortened, so it reads slower than it is. This underestimates speed. It never invents running.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2741

🤖 Generated with [Claude Code](https://claude.com/claude-code)
