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
(see "Tests run" block filled at the end of the branch)
```

## Evals run

```text
(see below)
```

## Docker/build/smoke checks

```text
docker compose config (orion-sql-writer, orion-substrate-runtime) with the live .env: OK, new keys present.
Image builds not run (no new dependencies; both Dockerfiles copy orion/ whole).
```

## Review findings fixed

(filled after review)

## Restart required

```bash
# On primary main after merge + git pull (the pull hook auto-applies the migration; check ~/.orion/applied_migrations.log says APPLIED for manual_migration_vision_rollup_v1.sql):
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-actions up -d --build
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
# Optional, after the first [VISION_ROLLUP] tick: backfill the last 7 days (dry run first)
PYTHONPATH=. python scripts/backfill_vision_rollups.py --days 7
PYTHONPATH=. python scripts/backfill_vision_rollups.py --days 7 --apply
```

## Risks / concerns

- Severity: medium. Concern: passings over-count people when the individuals reducer splits one person into several clusters (10-16 12:00 MDT: 79 person clusters in an hour, never more than 5 people in one frame). Mitigation: every row also stores `peak_at_once` from a different producer (scene inventory) as a lower bound; lines show it for the busiest period; the design doc names it.
- Severity: low. Concern: the journal guard checks numbers only; the forecast seed legitimately holds "5" (support days), so a "five occasions" style misreading of the forecast would still pass. Mitigation: the forecast line is reworded so it no longer reads as a claim about the street; day/week/month entries carry real counts.
- Severity: low. Concern: retina decode corruption is not persisted; it shows only as missing camera minutes. Mitigation: documented.

## PR link

(filled after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
