# fix(dream): reverie themes reach the dream as loop titles, not bare ids

## Summary

- Orion's sleep material named reverie themes by bare ids (`open-loop-968fda08910c`). In the
  10-09 06:33 sleep that was 4 of the 12 replayed items, and the carried dream had nothing to
  picture for them.
- The two reverie sources (compaction requests, resonance alerts) now also fetch each loop's
  human title, as it read when the window closed. The title comes from the newest
  `attention_salience_trace.description` for that theme, looked up by the indexed `theme_key`.
- Replay text uses the title: "Harness closure prediction error: reverie_chain_max_steps".
  With no title it falls back to the id.
- The id stays the novelty key, ref and tag, so **tiredness does not change**: 0 metric
  definitions changed.

## Outcome moved

- **Before:** about a third of real sleep material was opaque ids.
- **After:** all 11 reverie themes from the last week reach the sleep, its hypotheses, the story
  and the carried dream as readable titles. Examples: "Cabinet prediction error",
  "Juniper is talking (2 messages in the last 15 min)".

## Current architecture

`services/orion-dream/app/cycle_store.py` `SOURCE_QUERIES` → `app/replay.py` candidate builders →
replay text used by recombination, the story/carry material, and the Hub Dream tab.

## Architecture touched

orion-dream only. There is no schema, bus, env or table change.

## Files changed

- `services/orion-dream/app/cycle_store.py`: a `_title_join` LEFT JOIN LATERAL on both reverie
  queries, bounded by `:until`.
- `services/orion-dream/app/replay.py`: a `_loop_name` helper; the title is used in text only.
- `services/orion-dream/tests/test_dream_cycle_v2.py`: 4 tests.
- `services/orion-dream/evals/test_reverie_loop_titles_eval.py`, plus the fixture
  `reverie_loop_titles_2026-10-13.json` (the 11 real themes and their titles).
- `services/orion-dream/README.md`: "Reverie themes read as titles", including the scorecard
  cohort break.

## Schema / bus / API changes

None. Rows gain a `title` key, which is read only by replay.

## Env/config changes

None.

## Tests run

```text
cd services/orion-dream && pytest -q tests evals         233 passed, 2 skipped
Mutation check: replay ignoring the title -> the title test fails
check_definition_drift --gate PASS (0 changed) · check_metric_lineage --gate PASS · diff --check clean
```

## Evals run

```text
evals/test_reverie_loop_titles_eval.py
  A coverage check over the 11 real themes: no bare ids remain in the replay text.
SQL, live (from the review), old vs new query:
  identical rows and order for 7-day and lookback windows, at limit 3 and 5000;
  title lookup 0.02-0.05 ms per loop (0.23 ms with the :until bound) on
  idx_attention_salience_trace_theme;
  whole compaction query 1.8 ms.
```

## Docker/build/smoke checks

```text
Both new queries were run read-only against live Postgres over 48 h: titles returned for every compaction theme.
Live sleep with titles: UNVERIFIED until deploy and the next sleep.
```

## Review findings fixed

- Finding (low): the title was today's wording, not the wording at the time.
  - Fix: `AND a.created_at < :until`; still an index scan (0.23 ms).
  - Evidence: a test asserts the bound.
- Finding (low): the eval overstated what it proves.
  - Fix: its docstring scopes it to coverage and cites the live SQL comparison.
- Finding (low): this is a cohort break for the dream hypothesis scorecard (fragments become
  titled).
  - Fix: README note to split the scorecard at the deploy date. Arm vs arm stays fair.
- Finding (low, no action): a loop whose every trace had an empty description would scan all its
  traces. Live, 0 such loops exist (description is NOT NULL and filled).

## Restart required

```bash
git pull --ff-only && scripts/safe_docker_build.sh orion-dream up -d --build
```

## Risks / concerns

- Severity: low
  - Concern: the scorecard cohort break.
  - Mitigation: noted in the README.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2626

🤖 Generated with [Claude Code](https://claude.com/claude-code)
