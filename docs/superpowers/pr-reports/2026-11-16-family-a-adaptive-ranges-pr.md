# Family A per-reading ranges fit calm and busy hours (#3310)

## Summary

- Family A's per-reading 80 / 95 % ranges were one width for calm and busy hours, taken from validation residuals. On a busy day the busiest readings landed inside their 80 % range only about 60 % of the time. Quiet readings sat near 97 %.
- Each reading's range now comes from that reading's own recent live misses. It is scaled by how badly that reading was missed in the last hour. This is the same method the single-series forecasts already use (`app/adaptive.adaptive_offsets`, normalized split conformal), reused, not rewritten.
- A reading keeps its validation range when its pool is too thin (under a day of scored rows from the rung serving it now), when it has no recent misses to scale by, when the adaptive range would have zero width, or when anything here fails. The forecast is never lost.
- Same flag as the single-series ranges: `PREDICTION_ADAPTIVE_INTERVALS_ENABLED` (on).

## Outcome moved

The eval rebuilds, for every stored scored row, the range the live loop would now serve, from the pool before that row's origin (`evals/run_family_a_adaptive_ranges_eval.py`, 11-14 17:03Z .. 11-16 17:03Z, 64.7k rows):

| reading | served 80 % coverage | adaptive |
|---|---|---|
| athena.load_1m | 0.642 | **0.752** (11-15, the busy day: 0.585 -> 0.771) |
| athena.chassis_watts | 0.640 | **0.726** |
| athena.board_temp_c | 0.675 | **0.768** |
| circe.gpu3_temp_c | 0.647 | **0.726** |
| office.pm25_ug_m3 (too wide) | 0.971 | **0.792** |
| athena.gpu0_watts (too wide) | 0.962 | **0.864** |

All 41 readings with n >= 200 now land in 0.72-0.88; served, 12 were outside. That is the ticket's check. `athena.cpu_watts`, named in the ticket, isn't served right now (select refuses it as an input). Adaptive was used on 66 % of rows. The rest come early in the window, before a reading has a day of pool for its current rung, and they keep the served range, exactly as live does.

Live state right now: the 15-minute horizon goes adaptive at once. The 5-minute horizon switched rung (`ridge_var` -> `ar`) at 11-15 18:48Z and has about 22 h of `ar` rows against the 24 h minimum, so it goes adaptive on its own within hours. Pools are kept per rung on purpose: different rungs miss by different amounts.

## Current architecture

`FamilyALive.issue` built each reading's quantiles from the served rung's standardized validation residuals (`interval_source = 'validation'`).

## Architecture touched

orion-prediction only.

## Files changed

- `services/orion-prediction/app/family_a_live.py`:
  - `adaptive_ranges()` reads the pool once per issue and runs `adaptive_offsets` for each reading.
  - `issue()` uses adaptive quantiles where present and wide (`interval_source = 'adaptive'`).
  - `last_issue.n_adaptive`, `errors.adaptive`.
  - A `MemoryStore` twin of the new store read.
- `services/orion-prediction/app/store.py`: `family_champion_pool(series_ids, since, max_h)`, served by `prediction_forecast_scored_idx`.
- `services/orion-prediction/app/main.py`: passes the existing adaptive flag to family A.
- `services/orion-prediction/.env_example`: the flag's comment now says it covers family A too.
- `services/orion-prediction/evals/run_family_a_adaptive_ranges_eval.py`: new; served vs adaptive coverage per reading and per day.
- `services/orion-prediction/tests/test_family_a_live.py`: 4 new tests:
  - a busy hour widens the range and a calm one narrows it
  - the flag off, and a failing read, keep validation
  - another rung's pool doesn't size this rung's range
  - a zero-width adaptive range keeps validation

## Schema / bus / API changes

- None. Rows now carry `interval_source = 'adaptive'` (an existing value; joint scoring never reads the quantiles, and the self-monitor reads only `in_80`).

## Env/config changes

- No new keys. `PREDICTION_ADAPTIVE_INTERVALS_ENABLED`'s comment updated in `.env_example`; its meaning is extended to family A, and the value is unchanged.

## Tests run

```text
pytest services/orion-prediction/tests -q (prediction image) -> 525 passed
mutations: no adaptive wiring -> the new tests fail; owner filter removed -> other-rung test fails
```

## Evals run

```text
evals/run_family_a_adaptive_ranges_eval.py --days 2 --stride 2 (live DB, read-only) -> PASS (table above)
```

## Docker/build/smoke checks

```text
Read-only live smoke of adaptive_ranges() in the prediction image: 1.9 s per issue (41 reading/horizon pairs today),
runs in a worker thread (asyncio.to_thread), so /health and the loop are not blocked.
EXPLAIN ANALYZE pool read: Bitmap Index Scan on prediction_forecast_scored_idx, 327 ms for 57k rows. The old
starts_with form was a seq scan at 80 ms today, but it grows with the 35-day retention (~23k family rows/day); the
index read stays bounded by the 3-day pool.
```

## Review findings fixed

- Finding: the pool query could not use the scored index (`starts_with`, and no `scored_at IS NOT NULL`), so it would scan all retained family rows every 5 min.
  - Fix: `series_id = ANY(readings)` plus `scored_at IS NOT NULL`.
  - Evidence: EXPLAIN shows `prediction_forecast_scored_idx`.
- Finding: family C refuses a zero-width range, and adaptive offsets are equal when >= 80 % of scaled misses are identical.
  - Fix: adaptive only when the 0.9 offset > the 0.1 offset.
  - Evidence: `test_a_zero_width_adaptive_range_keeps_the_validation_range`.
- Finding: a reading with an adaptive range was dropped when its validation offsets were non-finite.
  - Fix: the validation offsets are checked only on the fallback branch.
- Finding: the eval differed from live (pool lower edge, horizon list from scored rows).
  - Fix: same edge and full served horizon list; dead code removed.
  - Evidence: eval re-run, still PASS.
- Finding: no test of the owner filter.
  - Fix: `test_a_pool_served_by_the_other_rung_does_not_size_this_rungs_range`.
  - Evidence: removing the filter fails it.
- Checked with no finding: no leakage (score runs before issue; the pool holds targets <= origin); units consistent; nothing reading family rows breaks on `adaptive`; the cost is off the event loop.

## Restart required

```text
No manual restart. After merge, the post-pull rebuild rebuilds orion-prediction.
Check: /health family_a.last_issue.n_adaptive > 0 and errors has no "adaptive".
```

## Risks / concerns

- Severity: low. Concern: the method and its parameters were chosen for the single series and are reused unchanged; the eval window (2 days) is short. Mitigation: re-run the eval after a week; the calm check (#3309) reads the joint score, which this does not touch.
- Severity: low. Concern: each select that switches a horizon's rung puts that horizon back on validation ranges for a day. Mitigation: intended (rungs miss differently); rung switches have been rare since 11-15.
- Issues: Fixes #3310. Opened: none.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
