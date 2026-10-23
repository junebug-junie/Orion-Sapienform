## Summary

- Orion's reveries now get one more line about the walkway: how the day has gone so far, read from the hourly walkway rollup (spec PR #2763, item 1, "Reveries").
- The line goes through `orion.vision.walkway_rollup.display_line`, so people are always a range ("at least 1, at most 2"). An old row is re-rendered or dropped. The raw stored line is never shown.
- Today's row only. Yesterday's row and stray hour rows are never "the day so far". Just after midnight, before today's first hour is rolled up, there is no line.
- The line is cut at a clause boundary to a budget (480 characters). A cut never lands inside the people range or inside brackets.
- The trace log line gains a field: `reverie_street_context corr=... captions=N graded=N rollup=0|1`.

## Outcome moved

Before: every reverie saw only the last 2 walkway captions (`captions=2 graded=0`, 98 lines in 24 h per the spec). Now it also sees the day's shape: the people range, strollers, vehicles, named visitors, busiest and quietest hour, and the usual day. Graded walkway predictions were already wired. They now exist live (17 met, 15 missed in `vision_percept_expectation`), so `graded` should stop reading 0.

## Current architecture

`services/orion-thought/app/reverie.py` `run_reverie_once` gathered street captions and graded predictions through `app/vision_reader.py`. It put them under their own context keys and rendered them in `orion/cognition/prompts/reverie_narrate.j2` under "OUTSIDE, ON THE WALKWAY (your outdoor camera ...)". The rollup table `vision_rollup` (PR #2755, ranges in #2781) had readers in the day letter, curiosity and the Hub, but not in reverie.

## Architecture touched

- orion-thought reverie tick: one more parallel read, `_fetch_street_day`, fail-open and separate from the caption read.
- Prompt `reverie_narrate.j2`: the street section also opens on `street_day` alone, in both branches.
- No bus, schema or table changes. The read is a read-only SELECT on `vision_rollup`.

## Files changed

- `services/orion-thought/app/vision_reader.py`: `read_street_day_line`, `trim_rollup_line`, `STREET_DAY_ROLLUP_SQL` (additive section at the end of the file).
- `services/orion-thought/app/reverie.py`: thread `street_day` into context and plan request. Extend the log line.
- `services/orion-thought/app/settings.py`, `.env_example`, `README.md`: two new keys, documented.
- `orion/cognition/prompts/reverie_narrate.j2`: render "the day so far, counted from this camera: ...".
- `services/orion-thought/tests/test_reverie_street_day.py`: new.
- `services/orion-thought/tests/test_reverie_thin_import_boundary.py`: call-time import check for the new reader.

## Schema / bus / API changes

- Added: reverie context key `street_day` (string, absent when empty).
- Removed / Renamed: none.
- Behavior changed: the `reverie_street_context` log line has a trailing `rollup=0|1`. It also fires when only the day line is present.
- Compatibility notes: none. The key is optional and the template guards on it.

## Env/config changes

- Added keys: `ORION_REVERIE_STREET_DAY_ENABLED=true`, `ORION_REVERIE_STREET_DAY_MAX_CHARS=480` (shipped ON).
- Removed / Renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes. Both keys are now in the primary checkout's `services/orion-thought/.env`.
- skipped keys requiring operator action: none. Compose reads the keys through `env_file: .env`, the same as the other street keys.

## Tests run

```text
services/orion-thought: pytest tests evals -q  -> 787 passed, 32 skipped
tests/test_reverie_street_day.py + test_reverie_thin_import_boundary.py -> 23 passed
.github/workflows/orion-static-gates.yml, every run step locally -> 36 OK, 0 FAIL
```

Key properties tested:
- The material holds the range text and never a bare people count, at every budget from 1 to the full length.
- A real rules_version 1 row, built by the rollup code, is re-rendered as a range by the unpatched `display_line`.
- Unreadable rollup JSON, yesterday's day row, yesterday's 23:00 hour at 00:30, other streams, an empty stream list, and a dead DB all give no line.
- A failing day read costs only that line: captions still go in, and the log says `rollup=0`.
- The tick log line ends `captions=0 graded=0 rollup=1` when the line goes in.

## Evals run

```text
services/orion-thought/evals ran in the suite above (all pass).
```

There is no reverie street eval harness. The existing evals cover hollow-guard and chain continuity, not street content. Follow-up: an eval that replays real reverie outputs to check whether the walkway line changes the thoughts at all.

Live read (read-only, real Postgres, 2026-10-23 16:4x UTC): `read_street_day_line` returned today's 455-character line "Walkway Fri 23 Oct 2026 (so far, through 10:00): people: at least 1, at most 2 ...", and `bare_people_counts` found nothing in it. At a 200-character budget it was cut to 183 characters after "12 vehicles (8 distinct) passing".

## Docker/build/smoke checks

```text
Not run: no Docker build. No new dependencies; orion.vision.walkway_rollup is stdlib-only, and the subprocess import-boundary test proves it in a clean interpreter.
```

## Review findings fixed

- Finding (MUST): the hour fallback could only surface yesterday's late hours as "the day so far". The writer builds today's day row in the same tick as today's first hour, so the fallback had no legitimate case. Its test built a state the writer never produces.
  - Fix: removed the fallback. Today's day row only.
  - Evidence: `test_just_after_midnight_yesterdays_late_hours_are_not_today`.
- Finding (SHOULD): the boundary test did not cover the new call-time import.
  - Fix: `test_reading_the_walkway_day_line_does_not_load_substrate`.
- Finding (SHOULD): the range guard was tested at one budget only, and the real `display_line` was never run.
  - Fix: a budget sweep test, plus an unpatched real rules_version 1 row.
- Finding (NIT): a cut could leave an open bracket.
  - Fix: split on top-level "; " only. The sweep asserts balanced brackets.
- Finding (NIT): two cameras on one day gave an arbitrary pick. A bad timezone fell back to UTC silently.
  - Fix: the configured order breaks the tie, and a bad timezone now logs a warning.

## Restart required

```bash
scripts/safe_docker_build.sh orion-thought up -d --build
```

Run it from the clean primary main checkout after merge.

## Risks / concerns

- Severity: low. Concern: the line adds about 450 characters to every reverie prompt. Mitigation: `ORION_REVERIE_STREET_DAY_MAX_CHARS`, or `ORION_REVERIE_STREET_DAY_ENABLED=false`.
- Severity: low. Concern: UNVERIFIED that a live reverie logs `rollup=1` or that the thoughts change. Mitigation: after deploy, run `docker logs --since 1h orion-athena-thought | grep reverie_street_context` and expect `rollup=1`.

## PR link

TBD

🤖 Generated with [Claude Code](https://claude.com/claude-code)
