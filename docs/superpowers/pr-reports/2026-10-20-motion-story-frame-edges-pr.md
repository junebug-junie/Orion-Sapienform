## Summary

- Motion stories now measure every box against the frame size of the camera that saw it. Before this, the doorbell was measured as if its frame were the walkway's 2560x1920.
- The writer now keeps the detector's frame size on each crop row (`vision_crop_observation.frame_width/height`). The vision host already sent it; the writer dropped it.
- Each story records where it came in and where it left (`entry_edge`, `exit_edge`), the last foot point, the exit speed in frame widths per second, and the frame size it was measured in. These are the phase 1 facts for the street-continuity handoff (`docs/superpowers/specs/2026-10-20-street-continuity-handoff-design.md`, section 1, PR #2803).
- Fallback order for the frame size: the crop's own size, then `VISION_MOTION_FRAME_SIZE_BY_STREAM`, then the global `VISION_MOTION_FRAME_WIDTH/HEIGHT`.
- One additive migration. The writer keeps working without it: it logs the missing columns and writes the older columns only.

## Outcome moved

- Live bug fixed: on the doorbell, a walker who crosses 10% of the frame read as "no lateral direction" (160 px / 2560 px = 6%, under the 8% threshold), and a box at the doorbell's right or bottom edge never counted as clipped. Regression tests cover both.
- Walkway stories are unchanged: the replay eval gives byte-identical fields on every story that existed in both runs (3831 walkway stories, 17 doorbell stories, 0 changed).
- New facts are now available for the handoff reducer: edges, exit foot, exit speed, frame size and its source.

## Current architecture

`app/vision_motion.py` (pure) links `vision_crop_observation` boxes into passages and builds one story per passage. `app/vision_motion_loop.py` reads crops per stream and writes `vision_motion_story`. Both used `MotionConfig.frame_w/frame_h` from `VISION_MOTION_FRAME_WIDTH/HEIGHT` (2560x1920) for every stream. `vision.crop.observation.v1` carries `frame_width/frame_height` (set by `orion-vision-host/app/runner.py` from the real image). `app/vision_crop_persist.py` used it only for zone checks and did not store it.

## Architecture touched

- orion-sql-writer only: crop persist, motion pure logic, motion loop, story strips (per-stream frame size), settings, env, compose, README, eval, tests.
- Migration: `services/orion-sql-db/manual_migration_vision_motion_story_edges_v1.sql`.
- CI: the sql-writer Postgres job applies the new migration before the vision tests.

## Files changed

- `services/orion-sql-writer/app/vision_motion.py`: `frame_of`, `crop_clipped`, `passage_frame`, `edge_of`, `exit_foot`, `exit_speed`. Kinematics, clipping, zones and evidence use each box's own frame. New story fields.
- `services/orion-sql-writer/app/vision_motion_loop.py`: reads crop frame size when the columns exist; per-stream config (`LoopConfig.motion_for`, `strip_for`); `parse_frame_sizes`; insert with or without the new columns; logs a missing migration each tick.
- `services/orion-sql-writer/app/vision_story_strip.py`: `PendingCrop` carries the crop's frame size; capture scales by it.
- `services/orion-sql-writer/app/vision_crop_persist.py`: stores `frame_width/frame_height`. Checks once whether the columns exist (absent is re-checked every 300 s, so applying the migration needs no restart).
- `services/orion-sql-writer/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: two new keys and docs.
- `services/orion-sql-writer/evals/run_vision_motion_story_replay_eval.py`: per-stream frame size, `--frame-size`, edge/frame report, new failure checks.
- `services/orion-sql-writer/tests/test_vision_motion.py`, `tests/test_vision_story_strip.py`: 14 new tests (13 motion, 1 strip) (pure, settings, migration shape, Postgres with and without the migration).
- `services/orion-sql-db/manual_migration_vision_motion_story_edges_v1.sql`: new.
- `.github/workflows/orion-sql-writer-tests.yml`: apply the new migration in the Postgres job.

## Schema / bus / API changes

- Added (`vision_crop_observation`, all nullable): `frame_width INTEGER`, `frame_height INTEGER`. The frame size the detector saw. NULL on older rows.
- Added (`vision_motion_story`, all nullable, NULL on older rows):
  - `entry_edge TEXT`, `exit_edge TEXT`: `left | right | top | bottom | inside`, from the first / last box. Left, right, bottom: the box side is within `VISION_MOTION_EDGE_ZONE_FRAC` (0.05) of the frame; the nearest wins; ties go left, right, bottom. Top: only when none of those and the foot point is within that share of the top (a head cut by the top edge is not an exit). Else `inside`.
  - `exit_foot_x REAL`, `exit_foot_y REAL`: the last box's foot point (bottom centre), 0..1 of its frame. On a clipped box it is on the frame edge.
  - `exit_speed_fw_per_sec REAL`: lateral foot speed over the last 4 or fewer unclipped boxes, in frame widths per second, UNSIGNED. Direction comes from `exit_edge` / `dir_lateral`. NULL with fewer than 2 unclipped boxes.
  - `frame_w INTEGER`, `frame_h INTEGER`: the frame size the story was measured in.
  - `frame_size_source TEXT`: `detection | stream_setting | default`.
- Removed / renamed: none.
- Behavior changed: doorbell direction and clipping once its crops carry their frame size. Walkway: none (its frames are 2560x1920, confirmed live on the bus).
- Bus: no change. `vision.crop.observation.v1` already had `frame_width/frame_height`.
- `rules_version` stays 1. Juniper's spot-check tally and the rollup story gate count one version at a time; a bump would hide the 28 existing verdicts. On the walkway, where all verdicts are, every story reads the same as before.
- Compatibility: the Hub selects named columns only, so the new columns change nothing there.

## Env/config changes

- Added keys: `VISION_MOTION_FRAME_SIZE_BY_STREAM` (default empty), `VISION_MOTION_EDGE_ZONE_FRAC` (default 0.05).
- Removed / renamed keys: none.
- `.env_example` updated: yes (`services/orion-sql-writer/.env_example`).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes, both keys added to the primary checkout's `services/orion-sql-writer/.env`.
- Skipped keys requiring operator action: none. (`SQL_WRITER_SUBSCRIBE_CHANNELS`, `GRAMMAR_RETENTION_PERIODIC_MAX_CYCLE_SEC`, `SQL_WRITER_ROUTE_MAP_JSON` were already diverged before this branch and were not touched.)

## Measured doorbell frame size

UNVERIFIED. No stored source holds it. The 33 doorbell crops carry no frame size (the column did not exist). The frame files are pruned after 300 s, and the last doorbell burst was at 17:33 UTC. The bus window cache holds only walkway/cam0 entries. Lower bound from the boxes: at least 1608x1395. After deploy, the first doorbell crop records its real size; then this query gives it:

```sql
SELECT frame_width, frame_height, count(*) FROM vision_crop_observation WHERE stream_id = 'doorbell' AND frame_width IS NOT NULL GROUP BY 1, 2;
```

Live evidence that the producer sends the size: a 40 s sample of `orion:vision:crops:sql-write` showed `walkway 2560 1920` on every message.

## Tests run

```text
pytest services/orion-sql-writer/tests/test_vision_motion.py (no PG)                     -> 42 passed, 5 skipped
throwaway postgres:16, CI migration set + edges_v1, ORION_TEST_PG_URI set:
  test_vision_confirm_asks, _split, _osnet_shadow, _body_match, _reanchor, _motion,
  _story_strip, _rollup                                                                 -> 242 passed, 0 skipped
  after review fixes, same set + test_vision_walkway_persist                            -> 258 passed, 0 skipped
  test_vision_motion.py alone                                                            -> 47 passed
pytest services/orion-sql-writer/tests (full)                                            -> 960 passed, 14 failed
  (biometrics, chat history, notify, gpu pool, subscription tests; none touch vision code.
   Same suite on the primary main checkout: 15 failed. Not caused by this branch.)
python scripts/check_env_template_parity.py                                              -> PASS
python scripts/check_env_key_single_source.py                                            -> OK
python scripts/check_migration_required_by.py                                            -> OK
orion.sql_migration_drift.auto_apply_refusals(edges_v1)                                  -> []
git diff --check                                                                         -> clean
```

## Evals run

```text
run_vision_motion_story_replay_eval.py --stream walkway (read only, since 2026-10-16T18:59Z)
  before: passages=3831 published=36 PASS
  after:  passages=3858 published=36 PASS (27 new stories closed between runs)
          0 of 3831 shared stories changed any field
          frame_sizes: 2560x1920/default 3858 (crops written before the migration have no size)
          entry edges (published): inside 34, bottom 2; exit edges: inside 35, right 1
          exit speed fw/s (published, n=36): p10 0.007, p50 0.0398
run_vision_motion_story_replay_eval.py --stream doorbell
  before: passages=17 published=1 PASS
  after:  passages=18 published=1 PASS, 0 of 17 shared stories changed
```

Finding from the eval: 35 of 36 published walkway stories exit "inside". The detector looks about every 5 s, so the last box of a walker is usually 10-40% of the frame from the edge (measured from the boxes). That is what the boxes show, so it is recorded as `inside`. The handoff reducer can use `dir_lateral`, `exit_foot_x` and `exit_speed_fw_per_sec` to project the exit. See Risks.

## Docker/build/smoke checks

```text
No image build: no dependency or Dockerfile change. Runtime proof is UNVERIFIED until deploy (see Restart required).
```

## Review findings fixed

Review ran in a subagent against `origin/main...HEAD`. No material correctness bugs. It confirmed that SQLAlchemy `text()` ignores the extra bind keys, the column cache race is harmless, walkway output is unchanged, no reader uses `SELECT *` on either table, and the migration fits the auto-apply allowlist.

- Finding (medium): story strips still scaled boxes by the stream or global frame, so doorbell strip crops were cut in the wrong place.
  - Fix: capture reads each crop's `frame_width/frame_height` (`CAPTURE_SQL`, `PendingCrop.frame_w/frame_h`). Assembly uses the story's `frame_w/frame_h`. Both fall back to the stream setting, then the global size.
  - Evidence: `test_capture_scales_by_the_crops_own_frame_size`. On a 1600x1200 frame, the crop with its own size is mostly the red box (share > 0.3, padding lowers it); the crop with the old global scaling has < 5% red.
- Finding (nit): the eval re-check rebuilt crops with the story's frame even when that frame was a fallback.
  - Fix: it passes the frame only when `frame_size_source == "detection"`.
  - Evidence: replay eval PASS on both streams.
- Finding (nit): walkway "unchanged" depends on the walkway detector sending exactly 2560x1920.
  - Fix: checked live. A 40 s sample of `orion:vision:crops:sql-write` showed `walkway 2560 1920` on every message. Post-deploy check: `SELECT stream_id, frame_width, frame_height, count(*) FROM vision_crop_observation WHERE frame_width IS NOT NULL GROUP BY 1,2,3;`
- Finding (nit): the Hub tally does not filter by stream, so future doorbell verdicts would mix stories from before and after the fix at `rules_version=1`.
  - Fix: not changed. Today the doorbell has 0 verdicts. Older rows have NULL `frame_size_source`, so they can be told apart. Listed in Risks.
- Finding (nit): "top" almost never fires; a walker going away toward the horizon reads "inside".
  - Fix: documented in the migration, README and Risks. The handoff reducer must not read "inside" as only "door, stop or lost".
- Finding (nit): the missing-migration error logs every tick. Kept on purpose: it is the same pattern as the strip check, and the migration auto-applies on pull.

## Restart required

Run from the clean primary `main` checkout after merge and `git pull`. The post-pull rebuild auto-applies the migration first (additive, REQUIRED-BY orion-sql-writer). If it does not, apply it by hand:

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 < services/orion-sql-db/manual_migration_vision_motion_story_edges_v1.sql
```

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

## Risks / concerns

- Severity: medium. Concern: published walkway stories almost always exit `inside` (35 of 36). The handoff reducer cannot rely on `exit_edge` alone. Mitigation: `exit_foot_x`, `dir_lateral` and `exit_speed_fw_per_sec` are recorded so it can project; a projected-exit field is a follow-up decision for the reducer PR.
- Severity: low. Concern: the doorbell frame size is UNVERIFIED, and the 33 old doorbell crops will keep using 2560x1920 unless `VISION_MOTION_FRAME_SIZE_BY_STREAM` gets a doorbell entry. Mitigation: new crops carry the real size; set the map entry once the query above returns it.
- Severity: low. Concern: `rules_version` stays 1, so if doorbell verdicts start, the Hub tally will mix doorbell stories from before and after the fix. Mitigation: today there are 0 doorbell verdicts; pre-fix rows have NULL `frame_size_source`.
- Severity: low. Concern: "inside" also covers a walker who goes away toward the horizon, because "top" needs the foot near the top edge. Mitigation: documented; the handoff reducer must treat "inside" as "not at an edge", not as "door".

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2822

🤖 Generated with [Claude Code](https://claude.com/claude-code)
