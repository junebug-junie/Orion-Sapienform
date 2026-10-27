# Walkway rollup lists named people by the person record

## Summary

- The walkway rollup now lists a named person once, under the name from the person record. Before, it listed the raw cluster label. Several camera clusters of one person merge into one entry.
- A cluster linked to a `not_a_person` row (a delivery-driver label, a "two kids together" label) is no longer listed as a named person.
- A labelled thing that is not a person (a vehicle answered "no", "unsure") is no longer listed as a named person. A thing that the person record links to its owner (`vision_belonging`) shows as "<name>'s bicycle".
- An unlinked typed person label keeps the old behavior.
- When the person tables cannot be read, the rollup groups by label as before and logs once.
- `rules_version` goes from 2 to 3. Each hour stores a digest of the person links, so a rename or new link rebuilds the recent 48 hours once.

## Outcome moved

Orion's day letter reads only `walkway_day.line`. That line now names people, not cluster strings. The live replay below shows the junk names ("no", "unsure", the delivery label) gone and bicycles credited to their owner.

## Current architecture

`orion-sql-writer` builds `vision_rollup` hour rows from sightings. In those rows `named` was keyed by `vision_individual.label`, for every detector kind. Day, week and month rows merge their children's `named` by key. `walkway_seen_loop` (orion-substrate-runtime) reads `named[*].individual_ids` and links entity nodes to a day node. The person record (`person`, `person_link`, PR #2852) existed, but the rollup did not read it.

## Architecture touched

- `orion/vision/walkway_rollup.py`: `PersonRef`, `_name_bucket` (person record -> named / belongings / not named), `_merge_named` for parents, `belongings` + `person_record` fields, render of belongings, `RULES_VERSION = 3`.
- `services/orion-sql-writer/app/vision_rollup.py`: `DbReader.person_links()` (one read per tick, inside a savepoint, fail-open), the links digest in `_stale`, `persons` passed to every hour build.
- `services/orion-substrate-runtime/app/walkway_seen_loop.py`: also reads `rollup->'belongings'`, so a named bicycle's entity node keeps its day link.

## Files changed

- `orion/vision/walkway_rollup.py`: person-record grouping, belongings, version bump, docstring.
- `services/orion-sql-writer/app/vision_rollup.py`: reader for links, digest-driven rebuild, docstring.
- `services/orion-sql-writer/tests/test_vision_rollup.py`: 5 unit tests + 1 Postgres-lane test; two version asserts now use `wr.RULES_VERSION`.
- `services/orion-sql-writer/evals/run_vision_rollup_replay_eval.py`: new check (named individuals are person-kind when the record was read); redaction covers belongings.
- `services/orion-sql-writer/README.md`: rules_version 3 section.
- `services/orion-substrate-runtime/app/walkway_seen_loop.py` + its tests: belongings are linked.
- `services/orion-hub/scripts/walkway_rollup_routes.py` + test: the panel shows belongings (ids stripped).
- `services/orion-actions/app/walkway_forecast.py` + test: forecast facts carry belongings.

## Schema / bus / API changes

- Added: `vision_rollup.rollup.belongings` ({owner display_name: {times|days, sightings, individual_ids, person_ids, kinds}}), `rollup.person_record` (hour rows: {available, digest}), `named[*].person_ids` (linked entries only).
- Removed: none.
- Renamed: none.
- Behavior changed: `named` keys are person display names when linked; non-person clusters and `not_a_person` links are not in `named`.
- Compatibility notes: no SQL migration (JSON fields only). v2 rows keep their stored line and still render (`display_line`); readers treat missing `belongings`/`person_ids` as empty. No bus channel or registered schema changes.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- skipped keys requiring operator action: none.

## Tests run

```text
PYTHONPATH=.:services/orion-sql-writer pytest services/orion-sql-writer/tests/test_vision_rollup.py
  46 passed, 1 skipped (no PG)
ORION_TEST_PG_URI=<throwaway postgres:16 container> ... test_vision_rollup.py
  47 passed (Postgres lane: label grouping + one warning with no person tables, then person grouping with real tables)
PYTHONPATH=.:services/orion-sql-writer pytest services/orion-sql-writer/tests -k "rollup or person or walkway"
  136 passed, 35 skipped
PYTHONPATH=.:services/orion-substrate-runtime pytest services/orion-substrate-runtime/tests/test_walkway_seen_loop.py
  9 passed, 2 skipped (real FalkorDB/PG lanes)
Consumers of walkway_rollup: test_walkway_forecast 24, test_curiosity_street_day_read 5,
  test_walkway_rollup_routes 7, test_reverie_street_day 17, test_reverie_thin_import_boundary 6,
  test_street_people_range 14 -- all passed
PYTHONPATH=. pytest tests/test_no_personal_terms.py orion/orion_day  112 passed
```

## Evals run

```text
PYTHONPATH=. python services/orion-sql-writer/evals/run_vision_rollup_replay_eval.py --days 10 --hours 0
  (read-only, live Postgres) PASS, including the new "named people are person-kind individuals" check
```

Live A/B replay, read only, same data, same code, person record off vs on (run 2026-10-27 ~03:15Z, before review fixes; a re-run after the fixes gives the same AFTER lines, but by then the vehicle labels "no"/"unsure" had been cleared in the live table, so its BEFORE no longer shows them). Names are replaced with placeholders in this tracked file; the real lines are in the PR thread handoff. (Rows stored on those days differ more, because labels and clusters changed since they were written.)

```text
2026-10-23
  BEFORE: named: A (17:47, 18:10, 18:31), B (09:16, 17:48, 18:09, 18:11), C (18:25, 18:31), D (18:28), E (18:25, 18:31), <delivery label> (15:42), no (19:08, 21:15), unsure (11:48, 14:07, 15:12)
  AFTER:  named: A (17:47, 18:10, 18:31), B (09:16, 17:48, 18:09, 18:11), C (18:25, 18:31), D (18:28), E (18:25, 18:31), E's bicycle (18:25, 18:31)
2026-10-24
  BEFORE: named: A (10:56, 13:25, 13:28, 14:17), B (11:43, 11:44, 12:03, 14:17), C (13:26, 13:27), E (13:25, 13:27, 13:28, 14:14), no (07:18, 19:07, 22:03), unsure (11:38)
  AFTER:  named: A (10:56, 13:25, 13:28, 14:17), B (11:43, 11:44, 12:03, 14:17), C (13:26, 13:27), E (13:25, 13:28, 14:14), E's bicycle (13:25, 13:27, 14:14)
2026-10-25
  BEFORE: named: A (10:27, 11:47, 12:16, 13:04), B (10:27, 10:49, 11:00, 12:16), C (11:47, 14:21, 15:18, 15:20), D (13:57, 13:58), E (11:47, 12:45, 13:29, 13:31), no (19:02, 19:04)
  AFTER:  named: A (10:27, 11:47, 12:16, 13:04), B (10:27, 10:49, 11:00, 12:16), C (14:21, 15:18, 15:20, 15:48), D (13:57, 13:58), E (11:47, 12:45, 13:31, 14:15), C's bicycle (11:47, 15:57, 15:58), E's bicycle (13:29, 13:31, 14:21, 15:18)
2026-10-26 (camera down 02:00-21:00): no named people either way.
```

## Docker/build/smoke checks

```text
No Docker build: no dependency, compose, or env change. Not deployed (per brief).
```

## Review findings fixed

- Finding (MUST): a day at the edge of the 48 h rebuild window, and any week/month holding older days, merged old label keys with new person keys ("Rob", "Robin", "no" in one line).
  - Fix: parents get the person record and re-key every label-grouped child (rules_version < 3, or built while the record was unreadable) through it (`_label_grouped`, `_regroup` in `walkway_rollup.py`).
  - Evidence: `test_review_mixed_v2_and_v3_children_never_name_labels_next_to_people` (day and week).
- Finding (SHOULD): the hub panel did not show belongings.
  - Fix: `belongings` is in `_PANEL_KEYS`, with `individual_ids` stripped. Evidence: `test_panel_shows_belongings_without_individual_ids`.
- Finding (SHOULD): forecast/journal facts dropped belongings.
  - Fix: `compact_rollup_facts` carries `belongings` (times/days/sightings/kinds). Evidence: `test_compact_facts_carry_belongings_without_ids`.
- Finding (SHOULD): an unlinked cluster typed with a linked person's name shared that entry, and `person_ids` made it look confirmed.
  - Fix: such individuals are listed in `unlinked_individual_ids`. Evidence: `test_review_an_unlinked_label_sharing_a_persons_name_is_listed_as_unlinked`.
- Finding (SHOULD): the links digest was global, so any link change rebuilt 48 hours.
  - Fix: each hour's digest covers only the individuals that hour saw. Evidence: `test_review_a_link_change_for_someone_the_hour_never_saw_does_not_rebuild_it`.
- Finding (NIT): the log-once flag never reset. Fix: reset on success, with one `vision_rollup_person_record_available` info line (Postgres-lane test).
- Finding (NIT): the READ ONLY replay with missing person tables was not tested. Fix: covered in the Postgres-lane test.
- Finding (NIT): the belonging name was built in two places. Fix: `walkway_seen_loop` uses `walkway_rollup.belonging_name`.
- Finding (NIT): time de-duplication on the fall-back day. Checked: the repeated hour carries its zone name (`hhmm`), so the two times stay apart. A comment says so.
- Finding (NIT): the PR report was not committed. Fix: committed.

## Restart required

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
```

Run from the clean primary `main` checkout after merge. The first tick after restart rebuilds the last 48 hours once (version bump).

## Risks / concerns

- Severity: low. Concern: a link change rebuilds the hours (inside 48 h) that saw that individual. Mitigation: per-hour digest; the work is the same as the existing catch-up.
- Severity: low. Concern: day rows older than 48 hours keep their v2 line (label grouping). Mitigation: `scripts/backfill_vision_rollups.py --days N --apply` rebuilds them if wanted.
- Severity: low. Concern: an unlinked person cluster whose label equals a linked person's name shares that name's entry in the line (same string key, the old behavior). Mitigation: listed in `unlinked_individual_ids`; the person record links such clusters on its next tick; conflicts (two people, one name) stay unlinked, as the planner intends.
- Severity: low. Concern: when an old label-grouped entry is re-keyed and its individuals land on several people, each gets the entry's times (times were not stored per individual). Mitigation: only old rows; rare (one label, several people).

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
