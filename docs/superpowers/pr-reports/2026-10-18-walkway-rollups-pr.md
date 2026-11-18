## Summary

- Orion now remembers the street. A new reducer in orion-sql-writer counts the walkway camera's sightings into one row per **hour, day, week and month** (`vision_rollup`), with the numbers and one plain sentence. Days are built from their hours, weeks and months from their days, so every grain agrees. No model writes them.
- An hour the camera did not deliver frames says **"camera was down"**, never "nobody came". A period with no row says **"No record"**.
- The nightly walkway journal now writes a **day entry every night, a week entry on Sunday, a month entry on the last day**, from the rollup. Any composed entry that states a number the rollup does not hold is replaced by the plain template line. This fixes the invented "the walkway has been empty on at least five occasions" (10-17).
- The rollups reach four more places: the **day letter** (one cited line), **curiosity study material** (newest day, next to the 6 h captions), a **"Walkway history" panel** in the Hub's Walkway (live) view (Hours / Days / Weeks / Months tabs), and the **substrate**: each named person's entity node gets an `observed_in` link to a day node for each complete day they were seen (they had 0 edges).
- Backfill script (dry run by default, not run live) and a read-only replay eval.

## Outcome moved

Before: walkway content reached Orion only as "right now" (15 min street line, 6 h captions, last 2 captions in reveries); nothing summarized the street over time; the nightly journal could invent claims; named people were isolated graph nodes. After: deterministic hour/day/week/month memory of the street with camera-down honesty, five consumers wired, and a numeric guard on walkway journal prose.

## Current architecture

- Sightings: `vision_individual_sighting` + `vision_individual` (orion-sql-writer individuals reducer, 60 s).
- Frame heartbeat: `vision_scene_inventory`, one row per vision window (~400/h), written whether or not anything was seen.
- Captions: `vision_events` `camera_caption`.
- Motion stories: `vision_motion_story`, Hub only behind Juniper's spot-check gate (>= 50 checked, >= 80% right, 0 wrong names; 8 checked today).
- Nightly walkway journal: `services/orion-actions/app/walkway_forecast.py` (22:30 local): grade + forecast from `vision_percept_expectation` (0 rows).
- Day letter: `orion/orion_day/gather.py`, no vision source. Curiosity: 6 h captions only. Named-people entity nodes: written only by `ask_answered_listener.py`, 0 edges.

## Architecture touched

- orion-sql-writer: new clocked reducer (`app/vision_rollup.py`, `app/vision_rollup_loop.py`), wired in `app/main.py`; pure rules in `orion/vision/walkway_rollup.py`.
- orion-sql-db: new table `vision_rollup`.
- orion-actions: walkway day/week/month journal + `guard_walkway_draft` in `_run_journal`.
- orion/journaler: trigger kinds `walkway_day|week|month` (digest, no email), compose prompt rule.
- orion/orion_day: `walkway_day` source, cited `[walkway_day:<date>]`.
- orion-hub: curiosity study material card; `GET /api/vision/rollups`; Walkway history panel.
- orion-substrate-runtime: clocked "seen on <day>" linker (`app/walkway_seen_loop.py`) via the existing materializer path.
- Not touched: attention (stays fenced), episodes. Dreams: follow-up.

## Files changed

- `orion/vision/walkway_rollup.py`: periods (DST-aware), hour build, parent build, template line, journal number guard.
- `orion/vision/motion_story_gate.py`: the spot-check gate as a shared pure function (Hub copy pinned to it by a parity test).
- `services/orion-sql-writer/app/vision_rollup.py`, `app/vision_rollup_loop.py`, `app/main.py`, `app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: reducer, loop, config.
- `services/orion-sql-db/manual_migration_vision_rollup_v1.sql`: the table (auto-apply clean).
- `scripts/backfill_vision_rollups.py`: 7-day backfill, dry run by default, snapshot + progress log on `--apply`.
- `services/orion-sql-writer/evals/run_vision_rollup_replay_eval.py`: read-only replay with consistency checks.
- `services/orion-actions/app/walkway_forecast.py`, `app/main.py`: rollup journal jobs, guard, "days watched" from rollups, reworded "not enough days" line.
- `orion/journaler/schemas.py`, `worker.py`, `dispatch_registry.py`, `orion/cognition/prompts/journal_compose_prompt.j2`: new trigger kinds and rule.
- `orion/orion_day/gather.py`, `budget.py`, `letter_parts.py`, `orion/schemas/orion_day.py`: day letter source.
- `orion/curiosity/study_material.py`, `kickoff_prompt.py`, `services/orion-hub/scripts/curiosity_investigation.py`: curiosity card.
- `services/orion-hub/scripts/walkway_rollup_routes.py`, `scripts/main.py`, `static/js/walkway-history.js`, `templates/index.html`: Hub panel.
- `orion/substrate/adapters/vision_individual.py`, `services/orion-substrate-runtime/app/walkway_seen_loop.py`, `app/main.py`, `app/settings.py`, `.env_example`, `docker-compose.yml`: seen-on links.
- `scripts/sync_local_env_from_example.py`: `VISION_ROLLUP_`, `WALKWAY_SEEN_` prefixes (the default sync skipped them silently).
- `.github/workflows/orion-sql-writer-tests.yml`, `schedule-browser-smoke.yml`: new tests in CI (PG step must not skip).
- Tests: `services/orion-sql-writer/tests/test_vision_rollup.py`, `services/orion-actions/tests/test_walkway_forecast.py`, `orion/orion_day/tests/test_gather.py` (+fixtures), `services/orion-hub/tests/test_curiosity_street_day_read.py`, `test_walkway_rollup_routes.py`, `test_walkway_history_browser_smoke.py`, `static/js/walkway-history.test.js`, `services/orion-substrate-runtime/tests/test_walkway_seen_loop.py`, `tests/test_journal_compose_prompt.py`.
- `docs/superpowers/plans/2026-10-18-walkway-rollups-design.md`: design + metric quality gate.

## Schema / bus / API changes

- Added: table `vision_rollup`; journal trigger kinds `walkway_day`, `walkway_week`, `walkway_month`; `OrionDayMaterialV1.walkway_day` (optional); `GET /api/vision/rollups`.
- Removed: nothing.
- Renamed: nothing.
- Behavior changed: walkway forecast "days watched" now counts day rollups where the camera was up >= 60 min (was: days with any sighting). Walkway journal prose with unsupported numbers is replaced by the template.
- Compatibility notes: no bus channel or schema-registry change. `OrionDayMaterialV1` is a forbid model stored in the durable brief: rebuild **orion-durable-runs before or with orion-hub** (same rule as `corrections`). `JournalTriggerV1` gained Literal values: rebuild orion-actions with the journaler consumers (orion-sql-writer journal persist reads trigger_kind as text).

## Env/config changes

- Added keys: orion-sql-writer `VISION_ROLLUP_ENABLED=true`, `VISION_ROLLUP_INTERVAL_SEC=300`, `VISION_ROLLUP_STREAMS=walkway`, `VISION_ROLLUP_PARKED_MIN_HOURS=4`, `VISION_ROLLUP_CATCHUP_HOURS=48`; orion-substrate-runtime `WALKWAY_SEEN_ENABLED=true`, `WALKWAY_SEEN_INTERVAL_SEC=1800`, `WALKWAY_SEEN_STREAMS=walkway`, `WALKWAY_SEEN_LOOKBACK_DAYS=3`.
- Removed / renamed keys: none.
- `.env_example` updated: yes (both services; settings + compose in step).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (all 9 keys added; no `--force`).
- skipped keys requiring operator action: none.

## Tests run

```text
# throwaway Postgres: ORION_TEST_PG_URI=postgresql://postgres:test@127.0.0.1:55499/postgres (docker postgres:16-alpine)
services/orion-sql-writer  pytest tests/test_vision_rollup.py                         28 passed (incl. real-PG path, own schema)
services/orion-sql-writer  CI unit list (walkway + rollup + orion/vision/tests)      112 passed, 1 skipped
services/orion-actions     pytest tests                                              256 passed (full suite; walkway_forecast 21)
services/orion-hub         pytest tests -k "curiosity or walkway or motion or orion_day or vision"   752 passed, 13 skipped
services/orion-hub         node --test static/js/walkway-history.test.js            3 pass
services/orion-hub         browser smoke (playwright 1.64, chromium, 360 px): test_walkway_history_browser_smoke.py + test_walkway_stories_browser_smoke.py   2 passed
services/orion-substrate-runtime  pytest tests/test_walkway_seen_loop.py tests/test_ask_answered_listener.py   26 passed
services/orion-substrate-runtime  full suite: 14 failed / 9 errors, identical on clean main (env-sensitive baseline); +4 new passing
services/orion-durable-runs  test_orion_day_arc_citations.py test_orion_day_graph.py   28 passed
repo root  orion/curiosity orion/orion_day orion/journaler orion/vision orion/situational + journal prompt/registry tests   543 passed, 20 skipped
orion-static-gates.yml, every step run locally                                       34/34 OK
python scripts/check_journal_dispatch_registry.py                                    OK (14 trigger kinds)
auto_apply_refusals(manual_migration_vision_rollup_v1.sql)                            [] (auto-apply clean)
```

## Evals run

```text
PYTHONPATH=. python services/orion-sql-writer/evals/run_vision_rollup_replay_eval.py --days 3 --redact
  (READ ONLY transaction against the live DB; nothing written)
  {"hours": 69, "days": 4, "weeks": 1, "months": 1, "clamped_to_first_frame": "2026-10-15T04:56:46Z", "story_gate_passed": false}
  sightings in hours: 1818; excluded {stationary: 469, prop: 139, patio: 0, parked_repeat: 218}
  PASS: day = sum of hours, week/month = sum of days, down hours read "camera was down", story content only behind the gate.

Sample lines (names redacted here; the live rows carry Juniper's typed names):
  hour  Walkway 14:00–15:00 (Sat 17 Oct): 27 passing (15 people, 6 dogs, 6 vehicles; most at once 2 people).
  hour  Walkway 10:00–11:00 (Sat 17 Oct): camera was down (0 of 60 minutes seen), so no count.
  day   Walkway Thu 15 Oct 2026: 185 passings (105 people, 10 dogs, 24 bicycles, 8 strollers, 38 vehicles) by 93 distinct;
        named: <5 named people with times>; 3 unnamed regulars (seen on another day this week too);
        busiest 16:00 (36, most at once 5 people), quietest 00:00 (0). Camera down 13:00–14:00, 20:00–21:00; counts cover 20.2 h of 24 h.
  week  Walkway week of Mon 12 Oct 2026 (so far, through Sat 17 Oct): 580 passings (317 people, 30 dogs, 42 bicycles, 17 strollers,
        174 vehicles) by 392 distinct; named: <5 named people, days each>; 35 unnamed regulars (seen on 2+ days this week);
        busiest Fri 16 Oct (309, most at once 5 people), quietest Thu 15 Oct (185). No record Mon 12 Oct–Tue 13 Oct; counts cover 61.3 h of 139 h.
```

## Docker/build/smoke checks

```text
docker compose config (orion-sql-writer, orion-substrate-runtime) with the live .env: OK, new keys present.
Image builds not run (no new dependencies; both Dockerfiles copy orion/ whole).
```

## Review findings fixed

Code review ran in a subagent (orion-repo-agent) on `origin/main...HEAD`: 0 blockers, 7 should-fix, 4 nits. All should-fix and three nits fixed; one nit is a documented limit.

- Finding: a day/week with no rows at all read "camera was down".
  - Fix: parent camera status is computed only over children with a row; no rows = status `missing`, line "no record".
  - Evidence: `test_a_day_with_no_rows_is_no_record_never_camera_down`, `test_status_is_computed_only_over_children_with_rows`.
- Finding: the journal guard took digits from timestamps, dates and captions, so "17 dogs" passed on the 17th.
  - Fix: counts are allowed only from numeric values and the template lines, after dates/years/clock times are stripped from both sides; each HH:MM in prose must be a time the seed names.
  - Evidence: `test_guard_dates_and_minutes_cannot_vouch_for_counts_and_times_must_match`.
- Finding: an hour with sightings but < 15 frame minutes said "no count" while its passings were summed.
  - Fix: status `unknown` ("camera record incomplete ... may be low"); the backfill now never starts before the first frame record.
  - Evidence: `test_sightings_without_frame_record_are_unknown_not_down`, `test_backfill_never_starts_before_the_first_frame_record`; replay now shows `clamped_to_first_frame`.
- Finding: "busiest ... (0)" on empty days.
  - Fix: busiest/quietest only when something passed. Evidence: `test_no_busiest_or_quietest_when_nothing_passed`.
- Finding: compound detector labels ("mail truck", "vehicle truck") became "other" and could name a peak.
  - Fix: word-based kind mapping (bicycle first, then vehicle words); "other" and "vehicle" never name a peak. Evidence: `test_compound_detector_labels_map_to_kinds_and_other_never_names_a_peak`.
- Finding: forecast "days watched" from rollup rows would be false until a backfill.
  - Fix: counted straight from scene-inventory minutes per local day (live now: 4). Evidence: `test_days_watched_counts_frame_minutes_not_sightings_or_rollup_rows`.
- Finding: deploy order for the new journal trigger kinds was not documented.
  - Fix: restart section below orders the journal consumers before orion-actions.
- Nit: a day was marked linked even when nobody was linkable, so a person named later never got the link. Fix: re-link when the linkable set grows (test extended).
- Nit: curiosity card appended "(today, still running)" to incomplete past days. Fix: the line alone (it already says "so far" or "No record").
- Nit: regulars used the last 6 day rows without a calendar limit. Fix: only rows inside the trailing week. Evidence: `test_regulars_ignore_day_rows_older_than_the_trailing_week`.
- Nit (not changed, documented): the 22:30 journal covers up to 22:00.

Also changed during review prep: the Hub motion-story card keeps its own gate copy (no edit to `motion_story_routes.py`, which open PR #2753 rewrites); a parity test pins it to `orion/vision/motion_story_gate.py`. Real names were scrubbed from tests (the personal-terms gate caught them).

## Restart required

```bash
# On primary main after merge + git pull. The pull hook auto-applies the migration; confirm
# ~/.orion/applied_migrations.log says APPLIED for manual_migration_vision_rollup_v1.sql.
# Order matters: journal consumers (they validate the new trigger kinds) before orion-actions;
# orion-durable-runs before orion-hub (forbid model OrionDayMaterialV1 gained walkway_day).
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
scripts/safe_docker_build.sh orion-cortex-orch up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-actions up -d --build
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
# Then backfill the last 7 days (dry run prints lines; --apply snapshots to /tmp/vision-rollup-backfill/ first):
PYTHONPATH=. python scripts/backfill_vision_rollups.py --days 7
PYTHONPATH=. python scripts/backfill_vision_rollups.py --days 7 --apply
# Proof: docker logs orion-athena-sql-writer 2>&1 | grep VISION_ROLLUP ; the Hub "Walkway history" panel;
# docker logs orion-athena-substrate-runtime 2>&1 | grep walkway_seen
```

## Risks / concerns

- Severity: medium. Concern: passings over-count people when the individuals reducer splits one person into several clusters (10-16 12:00 MDT: 79 person clusters in an hour, never more than 5 people in one frame). Mitigation: every row also stores `peak_at_once` from a different producer (scene inventory) as a lower bound; lines show it for the busiest period; the design doc names it.
- Severity: low. Concern: the journal guard checks numbers only; the forecast seed legitimately holds "5" (support days), so a "five occasions" style misreading of the forecast would still pass. Mitigation: the forecast line is reworded so it no longer reads as a claim about the street; day/week/month entries carry real counts.
- Severity: low. Concern: retina decode corruption is not persisted; it shows only as missing camera minutes. Mitigation: documented.
- Severity: low (live context, not a code bug). The reviewer saw the newest walkway `vision_scene_inventory` row at 2026-10-17 19:24 MDT. If frames are still not arriving at deploy time, new hours will read "camera was down", which is correct only if the camera or window path is really down: check the retina before reading the panel. UNVERIFIED whether the camera recovered.
- Severity: low. The day/week/month journal entries and the "seen on" links are UNVERIFIED live until deployed (no deploy from this branch).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2755

🤖 Generated with [Claude Code](https://claude.com/claude-code)
