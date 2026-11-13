## Summary

- Family A (#3318) deployed 2026-11-13 08:31Z issued forecasts but never scored any of them.
- Cause: the store found Family A rows with `series_id >= 'family_a:' AND series_id < 'family_a;'`. That range is only correct under byte ordering. The live database collation is `en_US.utf8`, which ignores punctuation at the first sort level, so the range matched **0 of 1,326** rows.
- Effect: every reading row stayed unscored; every joint row was marked `missing` / `too_few_readings` with `n_observed=0`; shadow rows were never scored; 35-day retention would have deleted nothing.
- Fix: all three queries use `starts_with(series_id, %s)`, which is collation-independent.
- The old unit test compared strings in Python (byte order), so it passed for the wrong reason. It is replaced by a test that pins the SQL shape of every prefix query.

## Outcome moved

Family A forecasts get scored, so the 7-day calm check (#3309) and the AC event check (#3311) can start.

## Current architecture

`app/store.py` `family_due`, `family_due_shadow`, `prune_family` selected rows by a byte-order prefix range `_prefix_range()`.

## Architecture touched

`services/orion-prediction/app/store.py` only. No schema, bus, env or API change.

## Files changed

- `services/orion-prediction/app/store.py`: `_prefix_range` removed; three queries use `starts_with`.
- `services/orion-prediction/tests/test_family_a_live.py`: byte-order test replaced by a SQL-shape regression test, plus a source scan of `app/` and `evals/` for any `series_id >=` range.
- `services/orion-prediction/evals/run_family_a_calm_live_eval.py`: same range bug in the calm check's reading query; now `starts_with`.
- this report.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none. Behavior changed: Family A rows are now found. Compatibility: none needed.

## Env/config changes

None.

## Tests run

```text
orion-prediction suite in the service image: 450 passed
```

Live read-only proof (2026-11-13 ~10:00Z, en_US.utf8):

```text
starts_with(series_id,'family_a:') unscored prediction_forecast rows: 1404
old range on prediction_shadow_forecast: 0   new: 1404
EXPLAIN: Index Only Scan using prediction_forecast_unscored_idx (index still used)
```

## Evals run

No eval change. The calm check (`run_family_a_calm_live_eval.py`) has nothing to read until this deploys.

## Docker/build/smoke checks

```text
Tests run in the deployed orion-prediction image. No dependency change.
```

## Review findings fixed

Code review ran in a subagent (read-only).

- Finding (material): the calm check (`evals/run_family_a_calm_live_eval.py`) used the same byte-order range, so #3309 would have reported 0 scored readings forever.
  - Fix: `starts_with`; test now scans `app/` and `evals/` for `series_id >=`.
  - Evidence: test suite below.
- Finding: store comment still described the range/left() match.
  - Fix: comment rewritten.
- Finding: the post-deploy reset UPDATE was unbounded and would re-take joints the new code legitimately marked missing (double count).
  - Fix: bounded by `scored_at < <container start>`.
- Checked, no change: `orion/self_monitor/collect.py:57` uses `left(series_id, 9) <> 'family_a:'` (equality, byte-exact under en_US.utf8); other evals use `= ANY`/`=`.

## Restart required

```bash
scripts/rebuild_services_from_git_diff.sh
```

(rebuilds orion-prediction). Post-deploy: joint rows the old code marked `missing` with `n_observed=0` (30 at ~10:00Z, more until deploy) can be reset to `issued` so the next pass scores them (their readings are still unscored). Bound it to rows scored before the new container started, so joints the new code legitimately marked missing are not re-taken. Check `calibration` counts first; include `no_run` rows if any.

```bash
docker exec orion-athena-sql-db psql -U postgres -d conjourney -c "UPDATE prediction_joint_surprise SET status='issued', scored_at=NULL, calibration=NULL, n_observed=NULL WHERE family='family_a' AND status='missing' AND calibration='too_few_readings' AND n_observed=0 AND scored_at < '<orion-prediction container start, UTC>'"
```

## Risks / concerns

- Severity: low. Concern: `starts_with` is not index-searchable on `series_id`. Mitigation: EXPLAIN shows the partial unscored index is still used via `target_ts`; the table is small.
- The #3309 7-day calm clock starts at this deploy, not 11-13 08:31Z.

## PR link

(this PR)
