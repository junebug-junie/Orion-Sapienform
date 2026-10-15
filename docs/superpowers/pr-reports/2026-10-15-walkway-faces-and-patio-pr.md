## Summary

Juniper lifted three walkway-camera privacy limits (Utah: home cameras not pointed into homes are lawful; the walkway is public). This PR:

- **Patio fingerprinting on.** The patio zone now gets crop embeddings, thumbnails and faces like the rest of the walkway, and patio people become individuals. The zone stays (dwell, attention, and its "someone is on the patio" head count via a new `presence: true` zone flag). A new migration drops the database rules that forbade this.
- **Faces for anyone on the walkway.** Each embedded walkway person crop also gets a face vector and a small face picture, from the same face models Juniper's identity check already loads. The individuals reducer remembers a face per individual and trusts a confident face match over clothing/appearance; appearance still works when no face is visible. Naming still only happens through the ask card, which now prefers the face picture. Juniper's enrolled gallery and the cam0/carbon identity check are untouched.
- **Full captions.** With no no-embed zone left, the council's walkway unresolved items carry the council's own words, the caption and the frame like every other camera.
- **Stationary fixtures (added from live data the same night).** A Halloween skeleton at the patio table was detected as a `person` in the same spot all night. A box that has not moved for 30 minutes is now a fixture: still recorded, but it never triggers an ask, an attention event, a rhythm prediction, a patio head count, or walkway "someone is here" presence. The first time the box moves it stops being a fixture.

## Outcome moved

- Patio visitors become individuals Orion can learn and ask about (previously only a count).
- Two passes by the same person in different clothes now join one individual when their face is visible (end-to-end check below: `assigned_by_face=7`, one individual, two sightings).
- The skeleton prop no longer becomes "a person seen every night" that would hit the ask thresholds, drive attention, and keep the walkway/patio reading "present" with nobody there.

## Current architecture

Walkway camera -> frame router (`want_crop_embeddings`) -> vision-host crop SigLIP vectors + thumbnails (patio boxes withheld by `embed: false`) -> vision-window `VisionCropObservationV1` -> sql-writer `vision_crop_observation` (CHECK: no-embed rows keep nothing) -> individuals reducer (appearance-only clustering; patio = count-only presence row flagged `no_embed_zone`) -> asks. Council unresolved items for the walkway were restricted to bare `no_label` counts. Faces existed only in `identity_face` (whole frame, Juniper only).

## Architecture touched

- `config/vision_zones.yaml`, `orion/vision/zones.py` (`presence`, `counts_presence`), new `orion/vision/stationary.py`.
- orion-vision-host: `crop_faces`/`_attach_faces`, `_crop_face_fn`, artifact builder.
- Contract: `VisionObject`/`VisionCropV1` additive face fields.
- orion-vision-window: forwards face fields; `FixtureTracker` for walkway presence.
- orion-sql-writer: persists face fields; individuals reducer (face index, stationary runs, head counts, asks); rhythm excludes stationary.
- orion-vision-council: docs + tests only (behaviour change comes from the zones config).
- DB: `manual_migration_walkway_privacy_v2.sql`.

## Files changed

- `config/vision_zones.yaml`: patio `embed: true`, `presence: true`.
- `config/vision_frame_router.yaml`: walkway `want_crop_faces: true` (both tiers).
- `config/vision_profiles.yaml`: `crop_face_profile/labels/min_prob/min_side_px`.
- `orion/vision/zones.py`: `presence` flag, `counts_presence`.
- `orion/vision/stationary.py` (new): IoU run rule.
- `orion/schemas/vision.py`: `face_embedding`, `face_score`, `face_thumb_ref` on `VisionObject` and `VisionCropV1`.
- `services/orion-vision-host/app/crop_embeddings.py`, `app/runner.py`, `app/artifacts.py`: faces on embedded person crops.
- `services/orion-vision-window/app/crops.py`, `app/fixtures.py` (new), `app/main.py`, `app/settings.py`, `.env_example`, `docker-compose.yml`.
- `services/orion-sql-writer/app/vision_crop_persist.py`, `app/vision_individuals.py`, `app/vision_individuals_loop.py`, `app/vision_rhythm.py`, `app/settings.py`, `.env_example`.
- `services/orion-sql-db/manual_migration_walkway_privacy_v2.sql` (new).
- Tests: host `test_crop_faces.py` (new), `test_crop_embeddings_patio.py`; sql-writer `test_vision_walkway_faces_stationary.py` (new, real skeleton boxes in `tests/fixtures/walkway_patio_skeleton_boxes.csv`), `test_vision_walkway_zones_thumbs_cursor.py`; window `test_stationary_fixtures.py` (new), `test_crop_observations.py`; council `test_unresolved.py`; router `test_walkway_policy_and_expectation.py`; `orion/vision/tests/test_zones.py`, `test_stationary.py` (new).
- `.github/workflows/orion-sql-writer-tests.yml`: runs the new tests; triggers on `orion/vision/**` and the zones file.
- READMEs (host, window, sql-writer, router, council) and the walkway spec amendment.

## Schema / bus / API changes

- Added: `face_embedding`, `face_score`, `face_thumb_ref` on `VisionObject` (extra=ignore) and `VisionCropV1` (extra=forbid). DB columns: `vision_crop_observation.face_*`; `vision_individual.face_centroid, face_n, still_box, still_since, still_last, stationary`; `vision_individual_sighting.face_thumb_ref, stationary`. Presence snapshot key `head_count_zone`.
- Removed: CHECKs `vision_crop_observation_no_embed_zone_keeps_nothing`, `vision_individual_sighting_no_embed_zone_keeps_nothing`.
- Renamed: none. No channel changes; registry entries unchanged (same models).
- Behavior changed: patio boxes are embedded; walkway unresolved items carry full text/caption/frame; stationary individuals excluded from asks/attention/rhythm/head counts.
- Compatibility: `VisionCropV1` is extra=forbid, so **deploy orion-sql-writer before orion-vision-window** (an old writer would reject crops with face fields). The new writer refuses to run the individuals cycle until v2 is applied (logs `migration not applied`, backs off). Patio head-count rows still carry `no_embed_zone: true` because `orion/situational/perception_reader.py` (not touched here) finds them by that flag.

## Env/config changes

- Added keys: orion-sql-writer `VISION_INDIVIDUALS_FACE_MATCH_THRESHOLD=0.60`, `VISION_STATIONARY_MIN_SEC=1800`, `VISION_STATIONARY_IOU=0.5`, `VISION_STATIONARY_MAX_GAP_SEC=300`; orion-vision-window `WINDOW_STATIONARY_STREAMS=walkway` + the same three `VISION_STATIONARY_*`.
- Removed / renamed keys: none.
- `.env_example` updated: yes (both services; window compose passes the new keys).
- local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys orion-sql-writer orion-vision-window`: yes, 8 keys added (4 per service). Pre-existing unrelated divergences reported (`SQL_WRITER_SUBSCRIBE_CHANNELS`, `SQL_WRITER_ROUTE_MAP_JSON`), not touched.
- skipped keys requiring operator action: none.

### Metric gate (stationary rule)

1. Provenance: `orion/vision/stationary.py:advance`, fed by `vision_individuals.py:advance_stationary` (per assigned crop) and `vision-window app/fixtures.py:observe_window` (per window).
2. Independence: derived only from detector box geometry over time; not a transform of the appearance/face vectors or the attention score it gates.
3. Theory anchor: a visit implies motion; a box with IoU >= 0.5 to its run's first box for 30 min has not changed place (IoU is the standard box-overlap measure).
4. Live data (read-only, 2026-10-15 04:57-05:23 UTC): 79 patio `person` boxes in 26 min, max gap 105 s, IoU vs first box min 0.676 / median 0.97 (two truncated frames at ~0.68-0.70, which is why the threshold is 0.5 not 0.7). Rest state: a walking person's run restarts every observation (test `test_a_walking_person_is_never_stationary`); the rule returns to "not stationary" the first time a box moves.
5. Existing mechanism: none (`rg stationary|fixture` in vision code found nothing).
6. Reversibility: `VISION_STATIONARY_MIN_SEC=0` disables both consumers; columns are additive.

## Tests run

```text
orion-sql-writer tests+evals: 14 failed, 783 passed (baseline origin/main: 14 failed, 757 passed; identical failure list)
orion-vision-host (in orion-vision-host-vision-host image): 4 failed, 249 passed (tests/test_crop_faces.py alone: 10 passed) (baseline: same 4 failed, 238 passed)
orion-vision-window: 146 passed (baseline 137)
orion-vision-council: 110 passed (baseline 110)
orion-vision-frame-router: 86 passed
orion/vision + orion/situational/tests/test_street_summary.py + tests/test_curiosity_unresolved_percepts.py: 64 passed
All steps of .github/workflows/orion-static-gates.yml run locally: 0 failed
check_env_template_parity: PASS; check_env_key_single_source: OK; compose parity: env_file (N/A)
```

## Evals run

```text
services/orion-sql-writer/evals/test_walkway_reducers_eval.py: pass (part of the sql-writer run above)
End-to-end on a throwaway Postgres 16 (not the live DB): v1 + v2 applied, v2 applied twice (idempotent),
real skeleton boxes x2 + a visitor seen twice in different clothes with the same face, one individuals cycle:
  summary assigned=166 new_individuals=2 assigned_by_face=7 stationary_crops=67 asks_opened=1 attention events=0
  skeleton individual stationary=True; visitor one individual (face_n=8, 2 sightings);
  walkway:patio head count = absent/0 (prop excluded); the only ask is the visitor, image = face thumbnail.
Real-model smoke (vision-host image, CPU, --network none): crop_faces on a real cam0 face frame -> prob 0.9997,
  512-d unit vector; blank crop -> None.
```

## Docker/build/smoke checks

```text
No image built and nothing deployed (deploy is out of scope for this branch).
Host tests and the facenet smoke ran inside the existing orion-vision-host-vision-host image, read-only mount.
```

## Review findings fixed

A fresh review subagent found no blockers. It raised four "should" findings and four nits:

- Finding: the prop's first 30 minutes (split into ~10 sightings by detector gaps longer than the 60 s merge gap) stayed ordinary sightings. Those alone could meet `VISION_ASK_MIN_SIGHTINGS=10` in one night.
  - Fix: when an individual turns stationary, every sighting of that still run is re-flagged stationary with its attention score cleared, both in the batch and in the DB (`_mark_run_sightings`, plus an UPDATE in `_one_stream`).
  - Evidence: `test_gapped_cadence_reflags_the_runs_earlier_sightings`, which fails when the re-flag is removed (mutation check).
  - Not fixed: attention events already written in that first window stay. They are insert-only, and it happens once per prop because the stationary flag sticks.
- Finding: the face-match threshold (0.60) has never been checked against walkway faces, and tiny upscaled faces can merge different people.
  - Fix: `crop_face_min_side_px` raised from 20 to 40.
  - Evidence: the threshold is stated as UNVERIFIED (see risks).
- Finding: the review could not see the vision-host face tests running anywhere.
  - Fix/evidence: they ran inside the vision-host image: `tests/test_crop_faces.py` 10 passed; full host suite 249 passed, with the same 4 failures as baseline.
- Finding: an old writer would reject every crop observation, because the window always sent `face_*: null`.
  - Fix: the window leaves out empty face keys (`_crop_payload`).
  - Evidence: `test_published_payload_omits_empty_face_keys_for_old_writers`.
  - Also: the sql-writer README deploy note now states the real order (migration first, then writer, then window), and its numbering is fixed.
- Nits fixed: the claim about reusing the cached face models now notes it only holds on the same device; the window's in-memory fixture runs (reset on restart, forgotten after 24 h) are documented; the stale window docstring is updated; the migration rollback note now explains how to restore the CHECKs.
- Nits not fixed:
  - Head counts can still include the prop if its crops split onto a second, non-stationary individual. Logged as a risk.
  - `orion/situational/perception_reader.py` has a stale privacy comment. That file belongs to another lane.

## Restart required

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_walkway_privacy_v2.sql
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-vision-host up -d --build
scripts/safe_docker_build.sh orion-vision-window up -d --build
scripts/safe_docker_build.sh orion-vision-council up -d --build
scripts/safe_docker_build.sh orion-vision-frame-router up -d --force-recreate
```

Order matters: migration, then sql-writer, then the producers. The router's config is a single-file bind mount, so it needs a recreate after the main checkout pulls.

## Risks / concerns

- Severity: medium. Concern: a real person who sits perfectly still on the patio for 30+ minutes is treated as a fixture (not counted as present, no attention) until they move; if they leave without the detector seeing them move, the next time their individual appears in the same spot it is immediately stationary. Mitigation: geometry-only rule, tunable/disable via `VISION_STATIONARY_MIN_SEC`; documented.
- Severity: medium. Concern: face hit rate on walkway crops is UNVERIFIED live (faces at walkway distance may be under MTCNN's 20 px minimum). Mitigation: appearance matching unchanged when no face; `crop_faces` counters (`faces`, `face_thumbs`) in the detect artifact's `crop_embeddings` block show the rate after deploy.
- Severity: medium. Concern: facenet now loads on the walkway detect device (~600 MB VRAM estimate in the identity_face profile) if identity_face was not already resident there. UNVERIFIED on the live GPU. Mitigation: disabling the `identity_face` profile turns crop faces off (no error).
- Severity: low. Concern: `orion/situational/perception_reader.py` (owned by another lane) still finds head-count rows by `no_embed_zone` and does not filter stationary sightings from its "who is around" line, so the skeleton individual can appear there. Mitigation: kept `no_embed_zone: true` on head-count rows; follow-up for that reader: filter `NOT s.stationary` and switch to `head_count_zone`.
- Severity: medium. Concern: the face match threshold (0.60) is UNVERIFIED on walkway faces, and a face match overrides appearance, so a wrong match merges two people. Mitigation: minimum face side raised to 40 px; env knob; `assigned_by_face` in the reducer tick log.
- Severity: low. Concern: in a prop's first 30 minutes, attention events may already have been written; they are not retracted. Its sightings are re-flagged, so asks and rhythm are not affected. Head counts can still include the prop if its crops split onto a second individual. Mitigation: once flagged stationary, the flag sticks while the box stays put.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2683

🤖 Generated with [Claude Code](https://claude.com/claude-code)
