## Summary

- July–September backfilled memories never create confirmation questions for Juniper. A high-stakes backfilled row is written with a new confirmation state, `not_asked`, instead of `pending_confirmation`.
- The confirmation asker only picks `pending_confirmation` rows, so a `not_asked` row can never become a question.
- `not_asked` rows are still hedged as unconfirmed everywhere they are shown: in chat recall ("Unconfirmed, check with Juniper if natural: …") and in page:juniper's sources ("not yet confirmed").
- The correction step may retire a `not_asked` row, because nobody will ever settle it. A `not_asked` row never retires anything itself, because it is unconfirmed.
- The backfill dry run now samples episodes spread across the whole July–September range, and reports the referent rate with a confidence interval.

Refs #3537. This PR does not close it: the larger dry run and the full backfill are still running.

## Outcome moved

- Before: a full backfill would have queued roughly a third of its memories as new confirmation questions. In the 2-episode dry run, 5 of 15 memories were high stakes. Over about 101 episodes that is likely 100 or more questions, oldest first, ahead of current memories, while the queue already grows faster than it is asked (#3661).
- After: backfilled memories add 0 questions. They stay labelled unconfirmed, so Orion still treats them with care.
- Before: unconfirmed rows were immune to correction for as long as they waited for an answer. A backfilled row would never get an answer, so it would have stayed immune forever.
- After: a later confirmed correction can retire it, under the same eval gate and per-pair review as any other correction.

## Current architecture

- `orion/memory/episode/confirmation.py` asks Juniper about high-stakes memories whose state is `pending_confirmation`, under a daily cap, oldest first.
- `orion/memory/episode/supersede.py` never lets a pending row be retired. Only settled rows may correct others.
- `scripts/backfill_episode_memory_jul_sep.py` (PR #3760) distills old Hub chat through the live distill path. The live path marked high-stakes rows `pending_confirmation`.

## Architecture touched

- Memory confirmation states: one new value, `not_asked`. It is free text in an existing column, so no DDL is needed.
- The backfill script, correction rules, recall rendering, the page:juniper source labels, the episode report, and the ungrounded-name repair script.
- No services, bus channels, schemas or env keys change.

## Files changed

- `scripts/backfill_episode_memory_jul_sep.py`: `without_asks()` turns high-stakes backfilled rows into `not_asked`. Adds a spread sample across the range, a Wilson interval, and a cluster interval for the referent rate.
- `orion/memory/episode/supersede.py`: documents that `not_asked` can be retired but never corrects.
- `orion/memory/voice_render.py`: recall hedges `not_asked` as "Unconfirmed".
- `orion/memory/pages/juniper.py`: page sources label `not_asked` as "not yet confirmed".
- `orion/memory/episode/report.py`: the episode report shows `not_asked` counts.
- `scripts/repair_episode_memory_ungrounded_names.py`: skips backfill rows, so the name repair cannot turn them back into questions.
- `orion/memory/episode/tests/test_backfill_jul_sep_script.py`: 5 new tests.

## Schema / bus / API changes

- Added: confirmation state value `not_asked` (an existing text column).
- Removed: none.
- Renamed: none.
- Behavior changed: backfilled high-stakes rows are never asked about; they are hedged in recall; the correction step can retire them.
- Compatibility notes: live distill is unchanged. Only rows written by the backfill (run id `backfill3537-*`) get `not_asked`.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- skipped keys requiring operator action: none.

## Tests run

```text
orion/memory/episode/tests/test_backfill_jul_sep_script.py, 5 new:
  test_backfill_rows_queue_no_confirmation_asks
  test_the_asker_selects_only_pending_rows
  test_not_asked_is_supersedable_but_never_corrects_and_is_hedged
  test_spread_sample_and_intervals
  test_name_repair_never_turns_backfill_rows_into_asks
CI: 16/16 checks pass.
```

## Evals run

```text
Larger backfill dry run (15 episodes spread over July–September, 66 turns, 40 of Juniper's exchanges,
writes nothing; 2026-11-24 03:25-03:46Z, run detached):
  memories 106, rejected by the validator 12, errors 0, dates after their episode 0
  any referent: 106/106 (100%)                       floor >= 97%   pass
  referent besides Juniper/Orion: 61/106 (57.5%)     floor >= 68%   FAIL
    95% range 48.0-66.5% (rows independent), 41.4-73.4% (whole episodes resampled)
  high-stakes rows that would be written not_asked: 26 of 106
Live October comparison (47 episodes, 239 distilled rows): besides Juniper/Orion 178/239 (74.5%),
  95% range 68.6-79.6% (rows), 65.6-82.6% (episodes); lowest live week 68.4%.
  Gap 16.9 points, 95% range 0.0-34 points.
Proposed backfill floor (NOT applied, waits on Juniper): 53% besides Juniper/Orion, any-referent stays >= 97%.
  Basis: lowest live week 68.4% x measured backfill/October ratio 0.773 = 52.9%.
Earlier 2-episode dry run: 15 memories, 0 rejected, 0 errors; any-referent 100%, other-than-Juniper/Orion 53% (floor 68%).
```

## Docker/build/smoke checks

```text
None needed. No service code changes. The backfill runs as a host script.
```

## Review findings fixed

- Finding: the episode report had no count for `not_asked` rows.
  - Fix: `report.py` labels and counts them.
  - Evidence: commit 3f3d93fd9.
- Finding: the referent rate interval was wrong for small clustered samples.
  - Fix: added a cluster interval next to the Wilson interval.
  - Evidence: `test_spread_sample_and_intervals`.
- Finding: the ungrounded-name repair script could flip backfill rows back to `pending_confirmation`.
  - Fix: it skips `backfill3537-*` rows.
  - Evidence: `test_name_repair_never_turns_backfill_rows_into_asks`.

## Restart required

```text
No restart required. The backfill script runs on the host; the library changes are picked up by the next rebuild of services that import orion/memory (post-pull rebuild handles it).
```

## Risks / concerns

- Severity: low.
- Concern: a backfilled high-stakes memory can be wrong, and Juniper is never asked about it.
- Mitigation: it is always labelled unconfirmed in recall and on pages, it never retires anything, and a later confirmed correction can retire it.
- Concern: the backfill itself is still gated on the referent floor; this PR does not run it.
- Issues opened/closed: none. Refs #3537.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3762

🤖 Generated with [Claude Code](https://claude.com/claude-code)
