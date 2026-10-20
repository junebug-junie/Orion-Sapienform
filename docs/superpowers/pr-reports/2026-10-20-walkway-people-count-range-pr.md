## Summary

Item 2, patch A of the walkway follow-ups spec (PR #2763): the honesty rule. Juniper approved it as decision 5.

- The walkway rollup now stores people as a range: `kinds.person.count_range = {low, high, method_low, method_high}`. Low is the most people in one camera window. High is the number of person clusters.
- Every rollup line says "people: at least 5, at most 79 (the camera splits one person into several)". It never says "79 people". Other totals read "at most N passings". Vehicle counts do not change.
- `rules_version` goes from 1 to 2. On the next tick, sql-writer rebuilds every v1 hour inside its 48 h catch-up window once.
- Older rows keep v1. Each reader re-renders them at read time (`display_line`). The readers are the day letter, the curiosity day card, the Hub walkway history, the walkway journal and the substrate "seen on <day>" links.
- The walkway journal guard now refuses composed prose that states one number of people ("137 people walked by"), even when that number is in the facts.

## Outcome moved

Before: 66 of 130 live rollup lines stated a bare people count, such as "92 passing (79 people, ...)" for 10-16 12:00 MDT. No frame in that hour held more than 5 people.

After: all 130 live rows pass through `display_line` with 0 bare people counts and 0 empty lines (read-only replay against the live table). The 10-16 12:00 row now reads "people: at least 5, at most 79 (the camera splits one person into several)".

## Current architecture

- `orion/vision/walkway_rollup.py` (pure) builds the hour, day, week and month rollups and the one-line text. `services/orion-sql-writer/app/vision_rollup.py` reads the raw tables and writes `vision_rollup` rows.
- `passings` counts person clusters. `peak_at_once` is the most of each kind in one vision window (`vision_scene_inventory.counts`, max per hour). A parent takes the max over its children. The docstring already called passings "an upper bound", but the line printed them as a head count.
- Readers took the stored `line` column as it was.

## Architecture touched

- Rollup JSON only. There is no new column, table, bus or schema-registry change, and no matcher change.
- Provenance of low (traced): `PgReader.peaks` (`services/orion-sql-writer/app/vision_rollup.py`) takes the max per label of `vision_scene_inventory.counts` per hour. `_peaks` maps the labels to kinds with max, not sum.
- Live finding: the patio skeleton prop (`fefb24d7...`, kind `person`) is counted in `peak_at_once.person`. On 10-15, 11 hours read "most at once 1 person" while nobody passed. So when a person-kind prop is sighted in an hour, the hour's low drops by the number of those props (`method_low = most_in_one_window_less_props`). The low never goes below 1 when a person was sighted.
- Low can be above clusters. This happens when a person in frame was excluded as stationary (live: 2 hours). High is then raised to low (`method_high = raised_to_low`), so `low <= high` holds at every grain.
- Day, week and month: low is the largest child low ("most at once in any one hour"). High is the sum of the child highs, so an hour whose high was raised keeps that raise. It is never below the summed clusters or the low.
- Glimpse stories do not count (decision 6 is still open).

## Files changed

- `orion/vision/walkway_rollup.py`: `count_range`, `person_count_range` (with a fallback for v1 rows), `_parent_range`, `range_phrase`, `display_line`, `bare_people_counts`; the line text; `RULES_VERSION = 2`.
- `services/orion-sql-writer/app/vision_rollup.py`: rebuild hours whose `rules_version` is old, inside the catch-up window.
- `services/orion-sql-writer/evals/run_vision_rollup_replay_eval.py`: range checks. Low <= high. An hour's low equals the window max. A parent's low equals the largest child low. No line has a bare count.
- `services/orion-sql-writer/README.md`: describes the range rule and the recompute.
- `orion/orion_day/gather.py`: the day letter reads `rollup` and uses `display_line`.
- `orion/curiosity/study_material.py`: the curiosity day card does the same.
- `services/orion-hub/scripts/walkway_rollup_routes.py`: the Hub panel line uses `display_line` and gains a `people` field. The JS renders `row.line` unchanged (`static/js/walkway-history.js`, textContent).
- `services/orion-actions/app/walkway_forecast.py`: the journal facts give people only as `{at_least, at_most}`, and person cluster and distinct counts are gone. Totals are `passings_at_most`. The guard refuses bare people counts for day, week and month entries.
- `services/orion-substrate-runtime/app/walkway_seen_loop.py`: the day node summary uses `display_line`.
- Tests: `services/orion-sql-writer/tests/test_vision_rollup.py`, `orion/orion_day/tests/test_gather.py`, `services/orion-hub/tests/test_walkway_rollup_routes.py`, `services/orion-hub/tests/test_curiosity_street_day_read.py`, `services/orion-actions/tests/test_walkway_forecast.py`, `services/orion-substrate-runtime/tests/test_walkway_seen_loop.py`.

## Schema / bus / API changes

- Added: `rollup.kinds.person.count_range`; `series[].people`; `busiest.people` / `quietest.people`; Hub `/api/vision/rollups` row field `people`.
- Removed: none (`passings`, `distinct` and `peak_at_once` stay in the JSON).
- Renamed: none.
- Behavior changed: the `line` text for people; walkway journal seed facts (`total.passings_at_most`, person `{at_least, at_most}`).
- Compatibility notes: v1 rows have no `count_range`. `person_count_range` derives it from their fields (`method_low = most_in_one_window_uncorrected`, because props cannot be taken out of an old row). `display_line` shows the stored line only for a row it knows is v2 or later. It re-renders an older row. It returns "" when the rollup is missing, is bad JSON or cannot be rendered. It never returns an old bare line.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change)
- skipped keys requiring operator action: none

## Tests run

```text
pytest services/orion-sql-writer/tests/test_vision_rollup.py              38 passed (incl. real-Postgres test on a throwaway postgres:16)
pytest services/orion-actions/tests/test_walkway_forecast.py              23 passed
pytest orion/orion_day/tests                                              105 passed
pytest services/orion-hub/tests/test_walkway_rollup_routes.py test_curiosity_street_day_read.py test_walkway_history_browser_smoke.py   11 passed, 1 skipped (playwright)
pytest services/orion-substrate-runtime/tests/test_walkway_seen_loop.py   7 passed, 1 skipped (Falkor), incl. real-Postgres SQL test
Workflow steps run locally: orion-static-gates (all 34 PASS), orion-sql-writer-tests unit, orion-reading-tests,
orion-day-letter-hub-tests, temporal-self-readers-tests, agency-ask-episodes -- all PASS
git diff --check                                                          clean
```

New tests cover these cases:
- For every hour, low <= high, and low equals the most people in one window. This runs over a 6 x 7 grid of clusters and window counts.
- Every grain from the engine: low <= high, and a day's low equals its largest hourly low.
- A 10-16 12:00 fixture (79 clusters, 5 at most in one frame) reads "at least 5, at most 79" and never "79 people".
- The live v1 row with its old stored line re-renders as a range.
- A prop comes off the floor. A stationary person raises the high.
- The version bump rebuilds v1 hours in the catch-up window, then returns to normal.
- Old parent rows re-render.
- One test per reader checks that an old row reaches it as a range.
- The journal guard refuses "137 people" and keeps "at least 5, at most 137 people".

## Evals run

```text
PYTHONPATH=. python services/orion-sql-writer/evals/run_vision_rollup_replay_eval.py --days 6 --redact
  121 hours, 6 days, 2 weeks, 1 month rebuilt from live tables (read-only)
  PASS: ... people always a range (low <= high, no bare count)
Read-only display_line pass over every live vision_rollup row:
  rows=130 stored_lines_with_bare_people=66 shown_bare=0 shown_empty=0
  NOON 10-16: "people: at least 5, at most 79 (the camera splits one person into several); 5 bicycles, 8 vehicles passing ..."
```

## Docker/build/smoke checks

```text
tzdata present in running containers (needed to re-render old rows):
  orion-athena-hub python3 ZoneInfo('America/Denver') OK
  orion-athena-substrate-runtime ZoneInfo OK
No image build run (no dependency or Dockerfile change).
```

## Review findings fixed

The code-review skill ran in a subagent on `origin/main...HEAD`. It found no must-fix items and four should-fix items.

- Finding: a day, week or month could print an "at most" that was too low. Its high was the summed clusters, so a child hour whose high had been raised to its low (people standing still) lost the raise.
  - Fix: a parent's high is now the sum of the child highs (`_parent_range`).
  - Evidence: `test_a_parent_keeps_a_child_high_that_was_raised_to_its_low` (4..4 + 1..1 now gives 4..5, not 4..4). The Postgres test's day now reads 2..3.
- Finding: `display_line` returned the stored line when the rollup was missing, was bad JSON or had no `period_start`. One forgotten `rollup` column in a reader would have leaked the old bare line.
  - Fix: it shows the stored line only for a known v2+ row. Every other case is a re-render or "".
  - Evidence: `test_display_line_never_shows_a_stored_line_it_cannot_vouch_for`. The day letter and curiosity card tests now check that a row with no rollup gives no line. Reader fixtures now carry `rules_version: 2`.
- Finding: the journal guard let a range word in another phrase excuse a count. "at most 3 dogs and 40 people came" and "Up to noon, 79 people walked by" both passed. It also refused "2 to 5 people" and "79 people at most".
  - Fix: a range word counts only directly before the number, as the low end of a pair (`5 and`, `5,`, `2 to`), or directly after it (`at most`, `or fewer`). The noun list did not grow.
  - Evidence: the new cases in `test_bare_people_counts_finds_one_number_of_people_and_passes_ranges`.
- Finding: an hour where every person was stationary no longer mentions people. Before this patch it said "most at once N people".
  - Decision, not changed: those people did not pass, and on 10-15 the same text came from the patio skeleton prop in 11 hours ("most at once 1 person" with nobody passing). Dropping it removes a false people claim. The window count stays in `peak_at_once` for the Hub and for later patches.
- Nits left as they are: the "unnamed regulars" count is clusters but is not called people. The Hub API JSON still carries `kinds.person.passings`, which the panel does not render and Orion does not read. A prop split into several ids over-corrects the low to 1, which is still a valid floor.

## Restart required

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-actions up -d --build
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
```

Optional: rebuild rows older than 48 h for good. Without this step the readers re-render those rows anyway:

```bash
PYTHONPATH=. python scripts/backfill_vision_rollups.py --days 7 --apply
```

## Risks / concerns

- Severity: low
  - Concern: low is "most people in one camera window". The detector can miss people, and the prop correction only knows the configured props, so low is a floor only as far as the detector is right.
  - Mitigation: `method_low` names the source on every range. Item 2's count eval (next patch) scores this against hand-checked hours.
- Severity: low
  - Concern: v1 rows older than 48 h are re-rendered on every read. Their low cannot have props taken out (`most_in_one_window_uncorrected`).
  - Mitigation: run the optional backfill above.
- Severity: low
  - Concern: substrate "seen on <day>" evidence nodes written before this deploy keep the old summary. The loop only re-applies a day after a restart or when someone new is named.
  - Mitigation: the restart above re-applies the last `lookback_days`.
- Severity: info
  - Concern: the live street line (`orion/situational/perception_reader.py`, "N unfamiliar people") counts individuals seen recently. It does not read rollups, so it is out of scope here, but it can over-count the same way.
  - Mitigation: follow-up in Item 2's next patch.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
