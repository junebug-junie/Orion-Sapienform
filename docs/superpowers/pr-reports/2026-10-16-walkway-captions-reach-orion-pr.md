# fix(vision-window): accepted walkway captions reach vision_events and Orion's street line

## Summary

- The walkway caption gate accepted its first Qwen caption live (22:53 UTC 2026-10-16), and Orion never saw it. The caption went into the window summary, and the only reader of that field (the vision council) skipped the window because the scene labels had not changed (`stable_scene`). No `vision_events` row, so no street line, reverie, dream or curiosity input.
- orion-vision-window now writes each accepted caption itself: one `vision_events` row with `event_type=camera_caption`, through the existing scribe path. Provenance goes into `tags` (`camera_caption`, `motion_unverified`, `detector:<label>`) and `evidence_refs` (artifact, frame path, frame time, each moving detector box, crop boxes, model).
- Every street reader now labels these rows as a caption of one frame, not a fact: `camera caption logged 16:53 (one frame; detector saw person, stroller; motion not verified): "A person is pushing a stroller…"`. Council rows render as before. Every consumer keeps at most one camera caption, and the caption SQL fetches captions and council rows as separate limits, so a busy street cannot crowd the council line out.
- The window logs every accepted caption (`[WINDOW] caption_accepted stream=... labels=... text='...'`, 160 chars) so it can be checked without searching other services.

## Outcome moved

Before: accepted caption -> window summary -> council `stable_scene` skip -> lost. 0 of 1 accepted captions reached Orion.
After: accepted caption -> `orion:vision:events` -> scribe -> `vision_events` (`camera_caption`) -> street line / reverie / curiosity, labelled. Live end to end is UNVERIFIED until the window, cortex-exec, orion-thought and hub images are rebuilt and the next accepted caption lands.

## Current architecture

- Frame router asks Qwen2-VL (circe-vl lane) for a walkway caption only on `triggered`-tier frames with a gate-label detection (7 triggered walkway frames vs 827 baseline in the hour after 22:40 UTC). 2 Qwen captions since the 22:51 rebuild: 1 accepted, 1 rejected `stationary_only`.
- `caption_gate.py` accepts or rejects; accepted text goes into `summary.captions` only. That is published on `orion:vision:windows` and kept in a 50-entry Redis/memory ring (minutes).
- Council (`orion-vision-council`) narrates only on label transitions or every 600 s (`refresh_ttl`); its narrative ("A vehicle is observed in the walkway area.") is the only `vision_events` text for the walkway. Even when it narrates a caption window, evidence grounding drops person/activity claims without `person` in the believed labels.

## Architecture touched

- orion-vision-window: new producer on `orion:vision:events` (`vision.event.bundle`).
- `orion/situational/street_material.py`, `perception_reader.py`, `orion/curiosity/study_material.py`: caption SQL selects `tags`; rendering labels `camera_caption` rows.

## Files changed

- `services/orion-vision-window/app/caption_events.py`: new; builds the `camera_caption` `VisionEventBundleItem` from an accepted-caption record.
- `services/orion-vision-window/app/caption_gate.py`: `apply(..., accepted_out=)` returns provenance of each accepted caption (moving boxes only, not stationary props).
- `services/orion-vision-window/app/main.py`: `caption_accepted` log line; `_publish_caption_events` after the window publish; metrics `vision_window_caption_events_{published,failed}_total`.
- `services/orion-vision-window/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: `WINDOW_CAPTION_EVENTS_ENABLED`, `CHANNEL_CAPTION_EVENTS_PUB`.
- `orion/bus/channels.yaml`: `orion-vision-window` added as a producer of `orion:vision:events`.
- `orion/situational/street_material.py`: `CAMERA_CAPTION_EVENT_TYPE` (single definition, imported by the window), `label_camera_caption` (names at most 3 detector labels, then "+N"), `detector_labels_from_tags`, `split_caption_sql`; `dedupe_street_captions(max_camera_captions=1)` labels and caps camera captions; `street_caption_line(tz=)` adds the local write time; `STREET_CAPTIONS_SQL` selects `tags` and splits caption/council limits.
- `orion/situational/perception_reader.py`: street caption SQL selects `tags`, split limits.
- `orion/curiosity/study_material.py`: asyncpg caption SQL selects `tags`, split limits.
- Tests: `services/orion-vision-window/tests/test_caption_events.py` (new), `orion/situational/tests/test_street_material.py`, `services/orion-thought/tests/test_reverie_street_context.py` (fixture gains the real `tags` column).

## Schema / bus / API changes

- Added: producer `orion-vision-window` on `orion:vision:events`; new `vision_events.event_type` value `camera_caption`.
- Removed: none.
- Renamed: none.
- Behavior changed: street captions from `camera_caption` rows render with a caption label in every consumer.
- Compatibility notes: no schema model change (`VisionEventBundleItem` as-is). `event_id` is a uuid5 of the caption artifact id.

## Env/config changes

- Added keys: `WINDOW_CAPTION_EVENTS_ENABLED=true`, `CHANNEL_CAPTION_EVENTS_PUB=orion:vision:events` (orion-vision-window).
- Removed keys: none. Renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys orion-vision-window`: yes (both keys added).
- skipped keys requiring operator action: none.

## Tests run

```text
services/orion-vision-window: pytest tests -q -> 168 passed
orion/situational/tests + orion/curiosity/tests + tests/test_curiosity_study_material.py + peer kickoff + dream hypotheses -> 436 passed, 20 skipped
services/orion-thought: full suite -> 707 passed, 32 skipped, 1 failed:
  (the failure, test_settings_mind_enrichment::test_mind_enrichment_defaults_off, also fails on main -- env-sensitive, unrelated)
services/orion-cortex-exec: tests/test_situation_perception_context.py -> 56 passed
services/orion-hub: tests/test_curiosity_street_captions_read.py tests/test_curiosity_investigation.py -> 155 passed
scripts/check_env_key_single_source.py OK; scripts/check_bus_reply_channels.py OK; check_service_env_compose_parity orion-vision-window N/A (env_file)
```

## Evals run

```text
No new eval. The street line has an eval (services/orion-cortex-exec/evals/run_street_who_line_eval.py) for the who-line, not captions.
Follow-up: a live smoke that counts camera_caption rows vs vision_window_caption_gate_accepted_total after deploy.
```

## Docker/build/smoke checks

```text
Not run (no deploy/restart in scope). Live read-only evidence before the fix:
- orion-vision-window /api/vision-window/metrics: caption_gate_accepted_total=1, rejected {host:null_answer:31, stationary_only:1}
- frame-router log 22:53:28 caption_ok corr=c293d785 text='A person is pushing a stroller on a sidewalk ...'
- council log 22:53:28.9 evidence_transition skip stream=walkway reason=stable_scene labels=vehicle
- vision_events walkway since 22:45: only council rows "A vehicle is observed in the walkway area/scene."
- fetch_street_summary('walkway') in orion-athena-cortex-exec-chat: no caption text
```

## Review findings fixed

- Finding: camera captions (up to one per ~8 s) could fill the SQL LIMIT (20) and push the council row out; reverie/curiosity had no one-caption cap.
  - Fix: `split_caption_sql` fetches 2 newest captions UNION ALL N newest other rows in all three SQL copies; `dedupe_street_captions(max_camera_captions=1)` caps every consumer.
  - Evidence: `test_caption_sql_fetches_captions_and_council_rows_as_separate_limits` (real SQL on SQLite, 25 captions + 1 council row), `test_many_captions_never_crowd_out_the_council_row_in_any_consumer`; the asyncpg SQL ran read-only on live Postgres and returned the walkway council rows.
- Finding: a caption tagged with all 7 gate labels pushed the brief to 430 chars (> 420 budget).
  - Fix: label names at most 3 detector labels, then "+N".
  - Evidence: `test_seven_detector_labels_still_fit_the_street_budget`.
- Finding (nit): a non-numeric `box_xyxy` in `source_objects` could raise inside `apply()` and fail the flush.
  - Fix: `_candidates` skips boxes that do not parse as floats.
  - Evidence: `test_malformed_source_box_never_fails_the_flush`.
- Finding (nit): "at HH:MM" claimed the frame time but is the row write time.
  - Fix: reworded to "logged HH:MM".
- Finding (nit): `CAMERA_CAPTION_EVENT_TYPE` defined twice.
  - Fix: the window imports it from `street_material`; `test_writer_and_reader_share_one_event_type`.
- Not fixed (pre-existing, follow-up): when labels DO change, the council still narrates from `summary.captions` and its row is not labelled, so Qwen's invented motion can still arrive as fact through a council row (evidence grounding drops person/activity claims without `person` in hard labels, which limits this).

## Restart required

```bash
scripts/safe_docker_build.sh orion-vision-window up -d --build
```

Rebuild also the consumers that bundle `orion/` (street_material change): orion-cortex-exec (chat), orion-thought, orion-hub, with the same wrapper from the clean primary main checkout after merge.

## Risks / concerns

- Severity: low. Concern: a caption of a stationary prop during the first 30 min after a window restart (prop runs are in memory) can now reach Orion directly. Mitigation: unchanged gate behavior; label says "one frame, motion not verified".
- Severity: low. Concern: captions are rare (triggered tier only), so the line will often still show only the council text. Mitigation: by design; not changed here.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2740

🤖 Generated with [Claude Code](https://claude.com/claude-code)
