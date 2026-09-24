# PR #2287: walkway camera, design + implementation

## Summary

- The design doc (`docs/superpowers/specs/2026-09-22-walkway-camera-busy-world-design.md`) now ships with its implementation: all ten ideas, built on a shared contract commit and four parallel branches, then two review rounds.
- **Individuals:** the GPU host fingerprints each person/dog/vehicle crop. sql-writer groups the crops into "the same one again" (appearance clusters, never faces) and their visits, with zone and dwell time.
- **Clocks:** a rhythm reducer learns when things usually happen (needs 5+ days of support). It predicts tomorrow's windows and grades each one met / missed / unscorable. "Expected, didn't come" becomes a vision event, and an open window makes the router look harder.
- **Asking:** Orion opens a question when an unnamed individual has been seen 10+ times over 5+ days (own daily cap of 2, restart-proof). A Hub card shows a thumbnail and takes Juniper's answer. The answer names the individual and becomes a substrate entity.
- **Readers:** things Orion couldn't name feed curiosity study material and a daily journal block. A street summary reaches the situation brief, so it reaches stance too. A nightly forecast/grade journal entry.
- **Cleanup:** the RTSP password no longer leaks into `stream_id`; the dead `orion-security-watcher` is retired; the untrained world model is marked parked.

## Outcome moved

Orion goes from "labels in a frame" to rows it writes from observation and predictions it can be wrong about: `vision_individual(_sighting)`, `vision_percept_expectation` (graded), `orion_ask` (answered), `vision_unresolved`. None of it is live yet. See "Restart required" and acceptance checks 1-8 in the spec.

## Current architecture

Before: cam0/carbon → edge → router → vision-host (detect/caption/embed) → window (5-30s rollups + scene census) → council (one sentence) → scribe → `vision_events`. There was no re-identification, no time-of-day learner, and no ask/answer channel. Vision reached no curiosity or journal reader. The RTSP URL (with password) was stored as `camera_id`/`stream_id`. `orion-security-watcher` ran with no producer.

## Architecture touched

- **The patio privacy rule** is enforced at three layers:
  - host: no embedding or thumbnail for any box touching a no-embed zone;
  - sql-writer: recomputes the zones and strips those fields;
  - Postgres: CHECK constraints on a `zone_no_embed` flag.
  Readers only ever see a patio *count*.
- **Room readers** (situation brief, reverie) now read only room-camera `vision_events`, using a new `stream_id` column. Rows with no camera are accepted only if written before the 2026-09-24 cutoff, so a stale deploy fails closed.
- **Services changed:**
  - vision-host, vision-window, vision-frame-router, vision-edge, vision-council
  - sql-writer (four clocked reducers)
  - hub (ask routes, card, thumbnail route)
  - substrate-runtime (answer → `EntityNodeV1`)
  - actions (perception-gaps block, forecast/grade journal)
  - cortex-exec / `orion/situational` (street summary)
  - thought (room filter)
  - curiosity study material

## Files changed

About 190 files; `git diff --stat origin/main...HEAD` lists them. Key ones:

- **Contract:**
  - `orion/schemas/vision.py` (VisionObject zone/embedding/thumb_ref, VisionCropV1, VisionCropObservationV1, PerceptExpectationV1, VisionUnresolvedV1)
  - `orion/schemas/ask.py`, `orion/schemas/registry.py`, `orion/bus/channels.yaml`
  - `services/orion-sql-db/manual_migration_walkway_camera_v1.sql`
  - `config/vision_zones.yaml`, `orion/vision/zones.py`, `orion/vision/stream_ids.py`
- **Host:** `services/orion-vision-host/app/crop_embeddings.py` (batched crop embeddings, ThumbStore, rate limit), `runner.py`, `config/vision_profiles.yaml`.
- **Window / router / edge:**
  - `services/orion-vision-window/app/crops.py`
  - `services/orion-vision-frame-router/app/expectation.py` + the `walkway` policy in `config/vision_frame_router.yaml`
  - `services/orion-vision-edge` (`vision-edge-walkway` compose profile, camera name instead of URL)
- **sql-writer:**
  - `app/vision_crop_persist.py`
  - `app/vision_individuals.py` + `_loop.py`
  - `app/vision_attention_score.py`
  - `app/vision_rhythm.py` + `_loop.py` (plus the 60s expect-key refresh)
  - `models/vision_walkway.py`
  - `scripts/report_vision_individuals.py`
- **Hub:** `scripts/ask_routes.py`, `static/js/vision-asks.js`, `templates/index.html`.
- **substrate-runtime:** `app/ask_answered_listener.py`, `orion/substrate/adapters/vision_individual.py`, perception `stream_id` sanitising (`worker.py`, `store.py`).
- **council:** `app/unresolved.py`.
- **Readers:**
  - `orion/situational/perception_reader.py`, `context.py`
  - `orion/curiosity/study_material.py`, `kickoff_prompt.py`
  - `services/orion-actions/app/{perception_gap_journal,walkway_forecast,vision_pg}.py`
  - `orion/journaler/schemas.py`
  - `services/orion-thought/app/vision_reader.py`
- **Retired:**
  - `services/orion-security-watcher/`
  - `orion/signals/adapters/security_watcher.py`
  - `VisionGuardSignal` / `VisionGuardAlert`
  - the `vision:guard:*` channels
- **Gates/CI:**
  - `scripts/sync_local_env_from_example.py` (REOLINK_URL / WALKWAY_RTSP_URL never synced)
  - `.github/workflows/orion-sql-writer-tests.yml` (new tests)
  - `config/metrics/metric_definitions.lock.json`

## Schema / bus / API changes

- **Added:**
  - **Channels:** `orion:vision:crops:sql-write`, `orion:vision:unresolved:sql-write`, `orion:ask:answered`.
  - **Kinds:** `vision.crop.observation.v1`, `vision.unresolved.v1`, `orion.ask.v1`, `orion.ask.answered.v1`.
  - **Tables:** `vision_crop_observation`, `vision_individual`, `vision_individual_sighting`, `vision_individuals_cursor`, `vision_percept_expectation`, `vision_rhythm_cursor`, `vision_unresolved`, `orion_ask`.
  - **Column:** `vision_events.stream_id`.
  - **Event types on `vision_events`:** `expected_absent`, `arrived_as_expected`, `attention_worthy`.
  - **Hub routes:** `GET /api/asks`, `POST /api/asks/{id}/answer|dismiss`, `GET /api/vision/crop-thumbs/{sha256}`.
  - **Redis key:** `orion:vision:expect:<stream_id>` (TTL = open window).
- **Removed:** `vision.guard.*` channels; `VisionGuardSignal` / `VisionGuardAlert`.
- **Behavior changed:**
  - The edge publishes the camera name as `camera_id`, never the URL.
  - The cam0 perception baseline re-keys from the URL to `cam0` (one cold start).
  - Room percept readers ignore walkway rows.
- **Compatibility:**
  - All schema fields are additive.
  - sql-writer adds `vision_events.stream_id` at boot in its own transaction, and drops the field from writes until the column exists.
  - `orion:ask:opened` was deliberately NOT added. Nothing would consume it; the Hub polls the table.

## Env/config changes

- **Added keys** (all in `.env_example` with comments):
  - sql-writer: `VISION_INDIVIDUALS_*`, `VISION_RHYTHM_*`, `VISION_ASK_*`, `ORION_ASK_DAILY_CAP`, `VISION_CROP_RETENTION_DAYS`, `VISION_SIGHTING_RETENTION_DAYS`, `VISION_EXPECT_REFRESH_INTERVAL_SEC`, `VISION_CROP_THUMB_RETENTION_DAYS`, `VISION_LOCAL_TZ`, and more.
  - vision-host: `VISION_CROP_THUMB_DIR`, `VISION_CROP_THUMB_RETENTION_DAYS`, `VISION_CROP_THUMB_MIN_INTERVAL_SEC`.
  - vision-window: `WINDOW_CROP_OBSERVATIONS_ENABLED`, `CHANNEL_CROP_OBSERVATIONS_PUB`.
  - router: `ROUTER_EXPECTATION_STEERING_ENABLED`, `ROUTER_EXPECTATION_REFRESH_SEC`.
  - edge: `WALKWAY_RTSP_URL` (empty), `WALKWAY_EDGE_PORT`.
  - council: `CHANNEL_VISION_UNRESOLVED`, `COUNCIL_UNRESOLVED_ENABLED`, `COUNCIL_UNRESOLVED_MIN_INTERVAL_SEC`.
  - substrate-runtime: `CHANNEL_ASK_ANSWERED`, `ENABLE_ASK_ANSWERED_LISTENER`.
  - hub: `HUB_VISION_CROP_THUMB_DIR`.
  - actions: `ACTIONS_JOURNAL_PERCEPTION_GAPS_ENABLED`, `ACTIONS_WALKWAY_*`, `POSTGRES_URI`.
  - cortex-exec: `ORION_SITUATION_STREET_STREAM_IDS`, `ORION_VISION_EVENTS_LEGACY_CUTOFF`.
  - thought: `ORION_REVERIE_PERCEPTION_STREAM_IDS`, `ORION_VISION_EVENTS_LEGACY_CUTOFF`.
- **Removed keys:** none shipped (`ORION_ASK_OPENED_CHANNEL` was added and dropped within this branch; removed from the local `.env`).
- **`.env_example` updated:** yes.
- **Local `.env` synced:** yes, with `sync_local_env_from_example.py --all-keys <service>` per service. Most new keys fall outside the default prefixes.
- **Operator action:**
  - `REOLINK_URL` / `WALKWAY_RTSP_URL` are now never-sync; the edge template holds a placeholder.
  - Set `WALKWAY_RTSP_URL` in `services/orion-vision-edge/.env` by hand when the camera is mounted.
  - `SQL_WRITER_SUBSCRIBE_CHANNELS` / `SQL_WRITER_ROUTE_MAP_JSON` show as diverged locally. That is harmless: code force-adds the walkway channels and merges its default routes.

## Tests run

```text
35 changed test files across hub, sql-writer, council, window, router, edge,
substrate-runtime, thought, actions, cortex-exec, orion/{vision,situational,curiosity},
tests/ -- all pass on the merged tree (per-file sweep, PYTHONPATH=.:<service>).
vision-host (needs torch): 227 passed, 1 skipped inside the existing image, worktree mounted read-only.
sql-writer CI list + walkway tests + eval: 134 passed. node --test (hub asks): 13 passed.
Static gates (orion-static-gates.yml): definition drift PASS, metric lineage PASS,
env template parity, async routes, compose relative mounts, hostname refs, journal
dispatch, schedule collisions, inner-state, sentience static, system-health producers,
control-surface parity, chat-route poachers, stdlib shadow -- all 0.
Pre-existing failures (identical on origin/main, not touched here): sql-writer 12,
hub 37, substrate-runtime 18, cortex-exec 179, tests/test_channel_prefix_guardrail,
tests/test_single_consumer_channels_gate.
```

## Evals run

```text
services/orion-sql-writer/evals/test_walkway_reducers_eval.py: simulated 21-day street
(weekday dog, random-time neighbour, 15 passers-by/day). Dog recovered (15 visits);
the random neighbour gets no prediction; held-out week 5/5 met.
Throwaway postgres:16 end-to-end runs of the real reducer SQL. They covered:
- migration idempotent twice, and an upgrade from the original contract draft
- no-embed CHECK rejects embeddings and thumbnails
- an ask opened with a thumb image; its answer set the label
- met/missed grading
- Redis expect key set, then deleted once met
- a late crop processed; 5 late crops merged into 1 sighting
- the room percept ignores a newer walkway row
```

## Docker/build/smoke checks

```text
No docker builds or deploys. vision-host tests ran in the existing image read-only.
Everything live is UNVERIFIED until the migration is applied and the services rebuilt.
```

## Review findings fixed

Each branch ran its own code-review subagent. Then two integration reviews ran on the merged branch; every finding is fixed.

- **Finding:** walkway and patio narratives could surface as "what Orion sees in the room".
  - **Fix:** `vision_events.stream_id`. The council and reducers stamp it; the room readers filter to room cameras. A row with no camera after the cutoff fails closed.
  - **Evidence:** reader tests on real SQL; postgres smoke.
- **Finding:** the ask card could never show a picture (edge frames are deleted after 60s).
  - **Fix:** the host writes content-addressed crop thumbnails, only for embeddable crops, rate-limited, kept 10 days (longer than the 7-day ask). A narrow Hub route serves them: hex ids only, O_NOFOLLOW, regular files only, size cap.
  - **Evidence:** host, route and JS tests.
- **Finding:** `vision_unresolved` could carry patio text or a whole-frame image.
  - **Fix:** for streams with a no-embed zone, only `no_label` is recorded, counting boxes in embeddable zones only. No captions, no free text, no image.
  - **Evidence:** council tests.
- **Finding:** a patio box could have its bottom-centre outside the patio and still include patio pixels.
  - **Fix:** `intersects_no_embed` withholds the embedding and thumbnail for any box that overlaps the patio.
  - **Evidence:** zones and host tests.
- **Finding:** the boot DDL for `stream_id` could roll back silently with the big bootstrap block.
  - **Fix:** it runs in its own transaction with an ERROR log, and writes drop the field until the column exists.
  - **Evidence:** postgres smoke, with and without the column.
- **Finding:** the scribe hop drops `stream_id` if it isn't rebuilt.
  - **Fix:** legacy-cutoff guard plus the rebuild list.
- **Finding:** late crops split one passing into many sightings, and late patio crops moved presence backwards.
  - **Fix:** nearest-candidate sighting matching, a landing-time cursor, and a max() on presence last-seen.
  - **Evidence:** regression tests from the repro.
- **Finding:** the expect key could start 15 minutes late.
  - **Fix:** a separate 60s refresh loop, gated on the rhythm loop being on.
- **Finding:** "patio" was hardcoded in the DB CHECKs.
  - **Fix:** a `zone_no_embed` flag instead; tested with a synthetic second zone.
- **Finding:** `orion:ask:opened` had no consumer (the metric-lineage orphan ratchet caught it).
  - **Fix:** the channel and its publisher were removed; asks are rows only.
- **Finding:** the patio headcount was lost between writer and reader (`subject.count` vs `count`).
  - **Fix:** the reader lifts the count out of `subject`.
  - **Evidence:** regression test.
- **Finding:** a zone point on the frame's bottom edge counted as outside every zone.
  - **Fix:** `zone_for_box` clamps into the frame.
  - **Evidence:** test.
- **Finding:** a `--force` env sync could overwrite the live camera URL with the placeholder.
  - **Fix:** the RTSP keys are never-sync.
  - **Evidence:** test.

## Restart required

Order matters. Apply the migration first, then rebuild from a worktree after merge:

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_walkway_camera_v1.sql
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-vision-scribe up -d --build
scripts/safe_docker_build.sh orion-vision-council up -d --build
scripts/safe_docker_build.sh orion-vision-host up -d --build
scripts/safe_docker_build.sh orion-vision-window up -d --build
scripts/safe_docker_build.sh orion-vision-frame-router up -d --build
scripts/safe_docker_build.sh orion-vision-edge up -d --build vision-edge
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
scripts/safe_docker_build.sh orion-signal-gateway up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-actions up -d --build
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
scripts/safe_docker_build.sh orion-thought up -d --build
docker rm -f orion-athena-security-watcher
# once WALKWAY_RTSP_URL is set in services/orion-vision-edge/.env and the polygons are traced:
scripts/safe_docker_build.sh orion-vision-edge --profile walkway up -d --build vision-edge-walkway
```

Editing `config/vision_zones.yaml` later means rebuilding vision-host, vision-council and sql-writer; the file is baked into all three images.

## Risks / concerns

- **Severity: high. The camera password is in git history.** It was in the old edge `.env_example`. Rotate the cam0 camera password.
- **Severity: medium. The zone polygons are placeholders.** `config/vision_zones.yaml` has guessed shapes. Until Juniper traces the real patio from a frame, the patio rule protects the wrong area.
- **Severity: medium. Cleanup of leaked rows is not done.** The existing rows still hold the URL: 483,545 in `substrate_perception_embedding_baseline` and 315,770 `camera_id` values in `vision_scene_inventory`. Section-14 backfill protocol before either:
  - `DELETE FROM substrate_perception_embedding_baseline WHERE stream_id LIKE '%://%';`
  - `UPDATE vision_scene_inventory SET camera_id = stream_id WHERE camera_id LIKE '%://%';`
- **Severity: medium. Appearance vectors go to every artifacts subscriber.** Walkway (non-patio) crop vectors ride the general `orion:vision:artifacts` broadcast. They are transient pub/sub, but any bus mirror sees them. A narrower channel is a follow-up if that matters.
- **Severity: low. The numbers are guesses to tune from data.** Match threshold 0.80, merge gap 60s, attention weights, thumbnail interval 10s. `scripts/report_vision_individuals.py` is the tool for tuning them.
- **Severity: low. Lost answers are not replayed.** If the Hub publish fails or substrate-runtime is down, no entity node is written. The label is still applied from the row.
- **Severity: low. No memory card.** No bus path creates person memory cards, so none was built. Adding one needs proposal mode.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2287

🤖 Generated with [Claude Code](https://claude.com/claude-code)
