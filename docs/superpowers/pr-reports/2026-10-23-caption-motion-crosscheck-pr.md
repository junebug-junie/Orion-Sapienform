## Summary

- A camera caption that claims motion ("A car is driving down the street") is now checked against how the caption's own car actually moved in the 10 seconds around the frame. Clearly still: the row gets the tag `motion_contradicted`. Clearly moving: `motion_supported`. Too little evidence, or in between: nothing.
- The check reads the car's own track near the frame, not the story's whole-passage speed class. The story class is an average, so a car that drove in and then parked reads `parked`; using it would label a true "driving" caption false.
- A street line that states one frame's people count ("Two persons are observed walking") now says "(count from one frame)", so it no longer reads as a head count. The detector is the rollup's own `bare_people_counts` (PR #2781). No new word list.
- Matching uses the caption's own boxes (its `box:` and `crop:` evidence refs) against the detector crops of the same frame, by IoU (box overlap). It never looks at every vehicle in the frame.
- The check runs inside the motion-story tick in orion-sql-writer, after the stories close, because a story closes minutes after its caption is written.
- Every reader that shows captions (chat brief street line, reverie, curiosity) now says "caption said driving; the vehicle did not move" instead of "motion not verified" for a checked row. All three go through `label_camera_caption`, so only that seam changed.
- The motion words are derived from the motion-story vocabulary that already exists (`SPEED_CLASSES`: walking, running, riding, creeping, driving, pushed). No new word list. Words outside it ("passing", "mowing") stay unchecked on purpose.

Spec: PR #2763, item 3, patch C.

## Outcome moved

Before: the 9 known "driving" captions read as "motion not verified". After: in the replay test all 9 come out `motion_contradicted`, and the street line says "caption said driving; the vehicle did not move".

Live, read only, fixed window 2026-10-20 17:30 to 2026-10-23 17:30 UTC, same code as the reducer:

| | walkway contradicted | supported | undecided | no claim | doorbell contradicted | supported | undecided |
|---|---|---|---|---|---|---|---|
| story average (first version) | 32 | 14 | 130 | 114 | 3 | 3 | 2 |
| local movement (this version) | 10 | 21 | 146 | 114 | 0 | 3 | 5 |

Per caption, story average -> local movement: 10 stay contradicted; 3 flip contradicted -> supported (all "a woman is pushing a stroller": the stroller's story averaged `parked`, but it was moving at the frame -- the exact failure); 20 contradicted -> undecided; 3 doorbell contradicted -> undecided; 7 undecided -> supported; 3 supported -> undecided.

9-caption replay (the known "driving" captions, 2026-10-17..20): 9 of 9 contradicted before, 9 of 9 contradicted after. Each parked car moved 0.002-0.014 body heights over 12-20 s with boxes on both sides of the frame.

One-frame people counts, walkway + doorbell street rows, last 7 days: 7 of 8,139 state a bare count (6 council rows, 1 camera caption). Before: shown as fact. After: all 7 carry "(count from one frame)".

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

1. Provenance: claim from the caption text (`caption_events.build_caption_event` narrative); boxes from its `box:`/`crop:` refs; motion from the story evidence boxes (`vision_motion.build_stories`, `evidence[].t/box/clipped`) within 10 s of the frame, through the reducer's own `kinematics`/`speed_class`.
2. Independence: the caption is a VLM reading one frame; the story is detector boxes tracked over time. Different models, different inputs. That is why one can check the other.
3. Theory anchor: motion is displacement over time. One frame cannot show it; a track can.
4. Live sanity: not degenerate. Both verdicts occur (walkway 10 contradicted / 21 supported), and a caption with no motion claim reads as nothing (114 of 291). Contradicted needs: 3+ boxes, one at least 4 s before and one at least 4 s after the frame, net travel and total path both <= 0.25 body heights. The spec's spot-check that `parked` stories are not stamped on cars that moved (20 with the Hub strip) is a visual check: UNVERIFIED. Most contradicting stories are vehicles the reducer already treats as props (known stationary cars).
5. Existing mechanism: reuses `motion_unverified`, `label_camera_caption`, motion stories, `SPEED_CLASSES`, `box_iou`. No detector of motion words existed elsewhere (`evidence_grounding.py` checks activity verbs only).
6. Reversibility: tags only. Kill switch `VISION_CAPTION_MOTION_CHECK_ENABLED=false`; readers fall back to "motion not verified" for untagged rows.

## Tests run

```text
pytest services/orion-sql-writer/tests/test_vision_caption_motion.py   33 passed
ORION_TEST_PG_URI=<throwaway postgres:16 + vision migrations> pytest tests/test_vision_caption_motion_pg.py   3 passed
pytest services/orion-cortex-exec/tests/test_chat_stance_street_context.py test_situation_perception_context.py   59 passed
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
PYTHONPATH=. python services/orion-sql-writer/evals/run_caption_motion_crosscheck_eval.py \
    --since 2026-10-20T17:30:00+00:00 --until 2026-10-23T17:30:00+00:00
  walkway 291: contradicted 10, supported 21, undecided 146, pending 0, no_claim 114
  doorbell 15: contradicted 0, supported 3, undecided 5, pending 0, no_claim 7
... --since 2026-10-17 --until 2026-10-20T03:35 --streams walkway
  claim 'driving' contradicted local=parked 9   (the 9 known captions; 9 of 9)
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

Second round (local movement + people count):

- Finding (must-fix): a car whose first box is the frame (drove in fast, then parked) or whose last box is the frame (pulled away) read as still and contradicted a true claim.
  - Fix: a still verdict needs a box at least 4 s before AND at least 4 s after the frame.
  - Evidence: `test_still_only_after_or_only_before_the_frame_is_none`; the 9 fixture cars all straddle the frame.
- Finding (should-fix): a there-and-back track has net travel ~0.
  - Fix: still also needs total path <= 0.25 body heights. Evidence: `test_there_and_back_track_is_not_still`.
- Finding (should-fix): the one-frame count note and the motion verdict pushed the street brief past its 420-char budget (456 worst case).
  - Fix: note shortened to "count from one frame"; `street_caption_line` takes the extra label length out of the caption text (floor 40 chars), so the line keeps its old budget. `perception_reader.py` not touched.
  - Evidence: `test_worst_case_labels_still_fit_the_street_budget` (7 labels + verdict + 2 count notes <= 420).
- Finding (should-fix): the 3-box / 8-s minimum also blocked "supported", so walking claims were almost never supported.
  - Fix: a moving verdict needs 2 boxes; the minimums apply to still only. Evidence: `test_two_moving_boxes_are_enough_to_support`; walkway supported 7 -> 21.
- Finding (nit): a timestamp with no offset would raise TypeError and abort the tick. Fix: read as UTC. Evidence: `test_naive_timestamps_are_read_as_utc`.
- Finding (nit): the count note was computed on clipped text. Fix: computed on the raw narrative. Evidence: `test_count_note_uses_the_unclipped_text`.
- Finding (nit): window comment said ~5 s cadence; live is ~2.5 s. Fixed.
- Finding (nit): `bare_people_counts` false positives ("1 person", "2 out of 3 people"). Not changed: it is shared with the rollup's anti-invention guard; harmless here (adds a note, never removes text).

## Restart required

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

From the clean primary `main` checkout after merge. Readers (`orion-cortex-exec`, `orion-thought`, `orion-hub`) pick up the new label on their next rebuild; until then they show "motion not verified" as before.

## Risks / concerns

- Severity: low. Concern: a car that stops for exactly the 20 s around the frame (a stop sign) reads still and contradicts "driving". Mitigation: needs boxes on both sides and <= 0.25 body heights of total path; such a caption is wrong about that frame anyway. The spec's 20-story spot-check is still UNVERIFIED.
- Severity: low. Concern: undecided is common (146 of 291 walkway rows) because fast cars are seen in only one or two frames. Mitigation: undecided means no tag, so the reader keeps "motion not verified".
- Severity: low. Concern: if the sql-writer re-writes a caption row with the same event_id, its tags are replaced. Mitigation: the step re-reads untagged rows in its 6 h lookback and tags them again.
- Severity: low. Concern: claims outside the story vocabulary ("passing", "mowing") are not checked. Mitigation: deliberate; growing a word list is a keyword cathedral.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2877

🤖 Generated with [Claude Code](https://claude.com/claude-code)
