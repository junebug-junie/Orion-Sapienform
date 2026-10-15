## Summary

- The street line in Orion's chat situation ("On the walkway in the last 15 minutes: ...") now lists names Juniper gave first. Next come unnamed people and animals. Vehicles and other things come last.
- Unnamed individuals of one kind are grouped into one phrase. Two parked cars now read "2 vehicles I have each seen on 2 days but have no name for", not the same sentence twice. When the day counts differ, the phrase says "at least N days" (the smallest count).
- A name only falls into "and N more" when there are more names than the 4-phrase cap. "and N more" now counts the hidden individuals, not hidden phrases.
- Each kind stays together (its recurring phrase, then its new phrase). The last item gets an "and" when nothing is hidden. Two individuals with the same name show as one name.
- `fetch_street_summary(now=)` used to be ignored by the SQL, which windowed on the server's `now()`. It is now bound into every windowed street read, with an upper bound of `now` + 60 s.

## Outcome moved

Live, 2026-10-15 23:35 UTC, from `orion-athena-cortex-exec-chat`:

> a vehicle I have seen on 2 days but have no name for, `<name>`, a vehicle I have seen on 2 days but have no name for, 4 unfamiliar bicycles, and 2 more.

Recently seen parked cars took the first slots, and the same phrase appeared twice. A named person could be pushed into "and N more". With this patch, names lead the line. Chat stance clips the street summary at 420/520 chars, so with names first a clip now cuts things, not names.

Live read-only before/after (the same database, the same `now=2026-10-15T23:35Z`; `<name>` is a real label given by Juniper):

```text
before (main):   a vehicle I have seen on 2 days but have no name for, <name>, a vehicle I have seen on 2 days but have no name for, 4 unfamiliar bicycles, and 2 more.
after  (branch): <name>, 11 unfamiliar people, 2 vehicles I have each seen on 2 days but have no name for, 5 unfamiliar vehicles, and 4 more.
```

The after line has more people because the branch read windows on the bound `now` (23:35). It also counts sightings that started up to 60 s after that, so people seen from 23:20 to 23:36 are included.

Live at the current moment (23:45 UTC):

```text
before: a vehicle I have seen on 2 days but have no name for, an unfamiliar person, 2 unfamiliar vehicles.
after:  an unfamiliar person, a vehicle I have seen on 2 days but have no name for, and 2 unfamiliar vehicles.
```

`now=` was ignored before, and this run shows it. With `now=2026-10-15T20:00Z`, main printed today's 23:xx individuals: "... <name>, ... <name>, and 6 more". Its SQL windowed on the server clock, and its Python filter had only a lower bound. The branch prints nothing for 20:00. That is correct: the last walkway sightings before then ended at 18:53 (checked with psql).

## Current architecture

- `orion/situational/perception_reader.py` `summarize_street` built the who-phrases in sightings order (most recent first). Grouping was by kind, and only for brand-new unnamed individuals. The list was capped at 4 phrases, and the rest became "and N more" (a count of phrases).
- `fetch_street_summary(now=)` used `now` only for the Python 15-minute cut. All four windowed statements used `now()` on the server. The only production caller (`context.py:1807`) does not pass `now=`, so live behavior matched. Replays and evals did not get the window they asked for.

## Architecture touched

- One shared module: `orion/situational/perception_reader.py`. There is no service-internal change, no schema change, and no bus change.

## Files changed

- `orion/situational/perception_reader.py`: new `_who_line`, `_recurring_phrase`, `_new_phrase` and `_LIVING_KINDS`. Street SQL is bound to `:now`/`:until`. New `STREET_CLOCK_GRACE_SECONDS = 60`. The CLOCK comment documents what a replay does and does not bound.
- `orion/situational/tests/test_street_who_line.py` (new): tests for ordering, grouping, cap, the reproduced live line, name dedupe, and `now`/`until` binding.
- `orion/situational/tests/test_street_summary.py`: an existing expectation now ends with "and 2 unfamiliar people" (the final "and").
- `services/orion-cortex-exec/evals/run_street_who_line_eval.py` (new): replays the live defect from a fixture. With `--live`, it checks the real walkway read-only against the named individuals in the window.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: the wording and order of the street who-line. `fetch_street_summary(now=)` now shifts the SQL window.
- Compatibility notes: no consumer parses this line (confirmed in review: context.py joins the lines; chat_stance only clips them).

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
pytest -q orion/situational/tests services/orion-cortex-exec/tests/test_situation_perception_context.py
267 passed
```

## Evals run

```text
python services/orion-cortex-exec/evals/run_street_who_line_eval.py          -> PASS (fixture of the live defect)
python services/orion-cortex-exec/evals/run_street_who_line_eval.py --live   -> PASS (read-only, live walkway; no names in the window at run time)
check() against the old live line -> fails on "a phrase repeats" and "names not first" (the eval catches the defect)
```

## Docker/build/smoke checks

```text
No build run (shared Python module; no deploy per task). Live read-only DB smoke above.
python scripts/rebuild_affected_services.py --paths orion/situational/perception_reader.py
  -> orion-cortex-exec, orion-hub, orion-thought
```

## Review findings fixed

- Finding: real personal names were committed in tests and the eval.
  - Fix: replaced them with invented placeholders (Pip, Wren, Rowan Example). The live quote uses `<name>`.
  - Evidence: grep for the real labels in changed files is empty.
- Finding: the CLOCK comment claimed a replay is fully isolated.
  - Fix: the comment now says only the event window is bounded. `label`/`distinct_days` are current state.
  - Evidence: perception_reader.py CLOCK comment.
- Finding (nit): the last item had no "and".
  - Fix: the last item now gets an "and" when nothing is hidden.
  - Evidence: `test_one_kind_stays_together_and_the_last_item_gets_and`.
- Finding (nit): one kind was split across the line.
  - Fix: a kind's recurring phrase and its new phrase are now adjacent.
  - Evidence: same test.
- Finding (nit): the eval's boundary check used `<` instead of `<=`.
  - Fix: changed it to `<=`.
  - Evidence: eval source.
- Finding (nit): the eval crashed when its own name query failed.
  - Fix: it now uses try/except and exits 2.
  - Evidence: eval source.
- Finding (nit): there was no test for a named thing.
  - Fix: added `test_a_named_thing_sorts_among_the_names`.
  - Evidence: test passes.
- Not changed (nit): when there are more than 4 names, "and N more" mixes names and things. The count is accurate, and the case is rare.
- Not changed (nit): there is no worst-case line-length test. Names-first means a clip cuts things, not names.

## Restart required

Not deployed (Juniper merges). For the change to reach chat, after merge, from the clean primary `main` checkout:

```bash
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-thought up -d --build
```

## Risks / concerns

- Severity: low
  - Concern: the `:until` upper bound (`now` + 60 s) could drop a live row if a writer's clock ran more than 60 s ahead of the reader's clock.
  - Mitigation: the DB clock and the host clock agree to about 20 ms (checked in review). All windowed columns are `timestamptz`.
- Severity: low
  - Concern: in a replay, the patio head-count line still uses the current presence row and wall-clock age, not `now`.
  - Mitigation: this is unchanged from before and out of scope. Production never passes `now=`.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
