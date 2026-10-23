## Summary

- A camera caption that claims motion ("A car is driving down the street") is now checked against the walkway motion stories. If the caption's own car sat in a `parked` story, the row gets the tag `motion_contradicted`; if it sat in a moving story, `motion_supported`. If it cannot be decided, the row gets nothing.
- Matching uses the caption's own boxes (its `box:` and `crop:` evidence refs) against the detector crops of the same frame, by IoU (box overlap). It never looks at every vehicle in the frame.
- The check runs inside the motion-story tick in orion-sql-writer, after the stories close, because a story closes minutes after its caption is written.
- Every reader that shows captions (chat brief street line, reverie, curiosity) now says "caption said driving; the vehicle did not move" instead of "motion not verified" for a checked row. All three go through `label_camera_caption`, so only that seam changed.
- The motion words are derived from the motion-story vocabulary that already exists (`SPEED_CLASSES`: walking, running, riding, creeping, driving, pushed). No new word list. Words outside it ("passing", "mowing") stay unchecked on purpose.

Spec: PR #2763, item 3, patch C.

## Outcome moved

Before: the 9 known "driving" captions read as "motion not verified". After: in the replay test all 9 come out `motion_contradicted`, and the street line says "caption said driving; the vehicle did not move".

Live, read only, last 72 h (2026-10-20 17:05 to 2026-10-23 17:05 UTC), same code as the reducer (after review fixes):

- walkway, 289 caption rows: 32 contradicted, 14 supported, 130 undecided, 113 with no motion claim.
- doorbell, 15 caption rows: 3 contradicted, 3 supported, 2 undecided, 7 with no motion claim.

Walkway "driving" claims: 27 contradicted (story `parked`), 6 supported (4 `creeping`, 2 `driving`), 107 undecided (the caption's own vehicle box was in no story, or in a glimpse that may be the passing car).

## Current architecture

- `services/orion-vision-window/app/caption_events.py` writes each accepted caption as a `vision_events` row (`camera_caption`, tags `motion_unverified`, `detector:<label>`, refs `frame_ts:`, `box:`, `crop:`).
- `services/orion-sql-writer/app/vision_motion_loop.py` builds motion stories every 15 s; each story has `speed_class` and `crop_ids`.
- Readers label captions with `orion/situational/street_material.py` `label_camera_caption` ("motion not verified"). Nothing connected the two.

## Architecture touched

- orion-sql-writer: new pure module + clocked step in the motion-story tick.
- `orion/situational/street_material.py`: tag constants (one definition, writer and readers), tag parser, label wording.
- No bus, schema or column change. Tags only.

## Files changed

- `services/orion-sql-writer/app/vision_caption_motion.py`: claim detector (derived from `SPEED_CLASSES`), ref parser, IoU match, verdict, reads, and `caption_motion_step`.
- `services/orion-sql-writer/app/vision_motion_loop.py`: calls the step per stream after the stories; `LoopConfig` fields; `[CAPTION_MOTION]` trace.
- `services/orion-sql-writer/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: three new settings, documented.
- `orion/situational/street_material.py`: `CAPTION_MOTION_*` tags, `CaptionMotionCheck`, `caption_motion_check_from_tags`, `label_camera_caption(motion_check=...)`; `dedupe_street_captions` and `street_caption_line` pass it through.
- `services/orion-sql-writer/evals/run_caption_motion_crosscheck_eval.py`: read-only replay; prints counts; `--fixture` dumps the replay inputs.
- `services/orion-sql-writer/tests/test_vision_caption_motion.py` + `tests/fixtures/caption_motion_driving_captions.json`: replay of the 9 live captions and unit tests.
- `orion/situational/tests/test_street_material.py`, `services/orion-hub/tests/test_curiosity_street_captions_read.py`: reader tests.
- `scripts/sync_local_env_from_example.py`: `VISION_CAPTION_MOTION_` sync prefix (the keys matched no prefix, so the sync skipped them).
- `.github/workflows/orion-sql-writer-tests.yml`: runs the new test; path filter includes `street_material.py`.

## Schema / bus / API changes

- Added: tag values on `camera_caption` rows: `motion_contradicted`, `motion_supported`, `motion_check:<kind>:<claim word>:<speed_class>` (contradicted claims first).
- Removed: none.
- Renamed: none.
- Behavior changed: `label_camera_caption` replaces "motion not verified" with the check result when a row has a verdict tag and a readable detail tag.
- Compatibility notes: untagged rows render exactly as before. `motion_unverified` stays on the row (the writer's own tag is not removed). `orion-actions` `walkway_forecast.py` calls `label_camera_caption` without the new keyword and is unchanged.

## Env/config changes

- Added keys: `VISION_CAPTION_MOTION_CHECK_ENABLED=true`, `VISION_CAPTION_MOTION_LOOKBACK_SEC=21600`, `VISION_CAPTION_MOTION_MIN_IOU=0.5` (orion-sql-writer).
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (`orion-sql-writer: +3 keys`).
- skipped keys requiring operator action: none.

## Metric quality gate (CLAUDE.md §0A)

1. Provenance: claim from the caption text (`caption_events.build_caption_event` narrative); boxes from its `box:`/`crop:` refs; motion from `vision_motion.build_stories` `speed_class`, read through `vision_motion_story.crop_ids`.
2. Independence: the caption is a VLM reading one frame; the story is detector boxes tracked over time. Different models, different inputs. That is why one can check the other.
3. Theory anchor: motion is displacement over time. One frame cannot show it; a track can.
4. Live sanity: not degenerate. Both verdicts occur (walkway 45 contradicted / 21 supported), and a caption with no motion claim reads as nothing (113 of 287). The spec's spot-check that `parked` stories are not stamped on cars that moved (20 with the Hub strip) is a visual check: UNVERIFIED. Most contradicting stories are vehicles the reducer already treats as props (known stationary cars).
5. Existing mechanism: reuses `motion_unverified`, `label_camera_caption`, motion stories, `SPEED_CLASSES`, `box_iou`. No detector of motion words existed elsewhere (`evidence_grounding.py` checks activity verbs only).
6. Reversibility: tags only. Kill switch `VISION_CAPTION_MOTION_CHECK_ENABLED=false`; readers fall back to "motion not verified" for untagged rows.

## Tests run

```text
pytest services/orion-sql-writer/tests/test_vision_caption_motion.py   31 passed
ORION_TEST_PG_URI=<throwaway postgres:16 + vision migrations> pytest tests/test_vision_caption_motion_pg.py   3 passed
pytest services/orion-sql-writer/tests/test_vision_motion.py test_vision_rollup.py (with the above)   passed
pytest orion/situational/tests   passed
pytest services/orion-hub/tests/test_curiosity_street_captions_read.py   5 passed
pytest orion/curiosity   passed
pytest services/orion-thought/tests -k "street or vision or caption"   42 passed
pytest services/orion-actions/tests -k "walkway_forecast or caption"   23 passed
orion-static-gates.yml: all 42 commands   rc=0
python scripts/check_env_template_parity.py   rc=0
```

## Evals run

```text
PYTHONPATH=. python services/orion-sql-writer/evals/run_caption_motion_crosscheck_eval.py --hours 72
  walkway 289: contradicted 32, supported 14, undecided 130, pending 0, no_claim 113
  doorbell 15: contradicted 3, supported 3, undecided 2, pending 0, no_claim 7
... --since 2026-10-17 --until 2026-10-20T03:35 --streams walkway
  claim 'driving' contradicted story=parked 9   (the 9 known captions; 9 of 9)
```

## Docker/build/smoke checks

```text
Not deployed (not asked). Candidate SELECT and EXPLAIN of the UPDATE ran against live Postgres in a
READ ONLY transaction: 31 walkway candidates in 6 h, plan OK. The SQL itself is exercised by
test_vision_caption_motion_pg.py against a throwaway Postgres 16 (also in CI's vision_pg step). Live tagging: UNVERIFIED until deploy.
```

## Review findings fixed

- Finding (must-fix): a parked car plus a passing car seen as a glimpse (no speed) in the caption's boxes was tagged contradicted, though the glimpse may be the driving car.
  - Fix: a story with no usable speed among the matched crops now blocks a contradiction (`check_claim`).
  - Evidence: `test_a_glimpse_of_the_claimed_kind_blocks_contradiction`; live walkway contradicted 45 -> 32; the 9 known captions stay contradicted.
- Finding (should-fix): glimpse, too-long and edge-clipped stories have a speed class that cannot vouch for the frame.
  - Fix: `usable_speed` treats them as no speed (reads `drop_reason`). The passage-average limit (drive in, then park) is documented as a known limit.
  - Evidence: `test_usable_speed_drops_stories_that_cannot_vouch_for_the_frame`, `test_glimpse_story_leaves_the_row_untagged` (real Postgres).
- Finding (should-fix): the 500-row cap read the oldest rows first, and final rows were skipped only after the LIMIT.
  - Fix: newest first, final rows skipped in SQL (`event_id <> ALL(:skip)`), warning when the cap is hit.
  - Evidence: `test_caption_motion_step_tags_once_and_skips_final_rows` asserts the skip list.
- Finding (should-fix): the SQL was only faked in tests.
  - Fix: `tests/test_vision_caption_motion_pg.py` (append once, own tags kept, second tick reads nothing, update guard, glimpse untagged); added to CI.
  - Evidence: 3 passed on a throwaway Postgres 16.
- Finding (should-fix, low): a redelivered caption (`worker.py` `sess.merge`) replaces the tags.
  - Fix: documented in README; the step re-tags inside its 6 h lookback.
- Finding (nit): story order was not deterministic. Fix: `ORDER BY ended_at, story_id`.
- Finding (nit): a stream with no story cursor was re-read each tick. Fix: early return. Evidence: `test_stream_without_a_story_cursor_reads_nothing`.
- Finding (nit): regex misses and false hits (negation, "moving", "drove"). Not fixed: documented as known limits; growing the word list is a keyword cathedral.

## Restart required

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

From the clean primary `main` checkout after merge. Readers (`orion-cortex-exec`, `orion-thought`, `orion-hub`) pick up the new label on their next rebuild; until then they show "motion not verified" as before.

## Risks / concerns

- Severity: medium. Concern: the story's speed class is the average over the whole passage, so a car that drove in and then parked reads `parked` and its caption can be tagged contradicted. Mitigation: documented; the better fix is the crop's own movement near `frame_ts` from the story evidence. The spec's 20-story spot-check is still UNVERIFIED.
- Severity: low. Concern: undecided is common (130 of 289 walkway rows) because fast cars are glimpses with no speed. Mitigation: undecided means no tag, so the reader keeps "motion not verified".
- Severity: low. Concern: if the sql-writer re-writes a caption row with the same event_id, its tags are replaced. Mitigation: the step re-reads untagged rows in its 6 h lookback and tags them again.
- Severity: low. Concern: claims outside the story vocabulary ("passing", "mowing") are not checked. Mitigation: deliberate; growing a word list is a keyword cathedral.

## PR link

PR_LINK

🤖 Generated with [Claude Code](https://claude.com/claude-code)
