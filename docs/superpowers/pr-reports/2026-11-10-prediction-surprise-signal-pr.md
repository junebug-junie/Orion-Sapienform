## Summary

- orion-prediction can now turn each scored cabinet/office forecast into **surprise**: how unlikely the real value was under the forecast, in one unit (`-ln p`), with a calm point set by math (0.95), not by tuning.
- A per-horizon gate says which forecast distances are fit to carry it (real forecaster, live 80 % coverage in band, not worse than the baseline live).
- `GET /series/{id}/surprise` and `evals/run_surprise_calm_live_eval.py` run the spec's calm check against live data.
- **The calm check fails today**, so this PR ships the measurement and its debug surface only. No bus event, no consumer.
- Second build patch of the prediction-error suite rebuild spec (PR #3168).

## Outcome moved

The first prediction-error measurement in Orion that is a real prediction: forecast stored before the target time, scored when the clock says, surprise in one unit, calm defined by theory.

Its first result: the forecaster's ranges fit busy hours and calm hours with one width, so calm days read far below the calm point. That is the next thing to fix, and it had no measurement before.

## Scope change during the PR

The first version also published `prediction.surprise.v1` on a new bus channel. CI's metric lineage gate refused it (`orphan ratchet: bus_channel grew 17 -> 18`): a channel with no consumer feeds nothing (CLAUDE.md 0A). The gate is right, and no consumer may read surprise until the calm check passes anyway. The event, its schema and the issue-time gating (judge the model that served when the forecast was issued, joined from `prediction_model_run.assignment`) ship with the first consumer, the substrate node patch.

## Metric quality gate

1. **Provenance.** `app/surprise.py::conformal_p_value` over `prediction_forecast.quantiles` (written by `Engine.build_forecast`, split-conformal from backtest residual paths) and `actual` (written by `Engine.score` from the 5-minute bucket mean).
2. **Independence.** Not wired into any model. Shares its input rows with `in_80`/`in_95` coverage: coverage is the drift reading, surprise the spike reading, by design.
3. **Theory anchor.** Probability integral transform: under a calibrated forecaster the CDF at the outcome is uniform, so `p = 2 min(u, 1-u)` is uniform and `E[-ln max(p, 0.05)] = 1 - 0.05`. Method bias measured: the 9-level linear CDF reads -0.033 for a perfectly calibrated normal (`tests/test_surprise.py`); the calm verdict allows for it.
4. **Live sanity (2026-11-10, champion rows, independent origins).** Not degenerate: varies, reaches the floor, returns to rest. But it rests **below** the calm point:

```text
7 days     cabinet 15 min excess -0.115 +-0.056   office 15 min -0.116 +-0.061   (80% coverage 0.84 / 0.85)
quiet 24 h cabinet 15 min -0.281 +-0.101, 60 min -0.372 +-0.145; office 15 min -0.355 +-0.085
```

5. **Existing mechanism.** orion-prediction already scores coverage; no surprise or p-value existed anywhere for these forecasts.
6. **Reversibility.** No table, migration, bus event or consumer. Remove = drop the module, the endpoint and the eval.

## Current architecture

orion-prediction forecasts `athena.cabinet_temp_c` and `office.temp_c` every 5 min at 5 min / 15 min / 1 h / 6 h / 24 h, stores 9 quantiles per horizon, scores each when its bucket lands (`actual`, `in_80`, `in_95`, or `missing`), and monitors live coverage per horizon. Nothing turned a scored miss into surprise.

## Architecture touched

- `services/orion-prediction/app/surprise.py` (new): p-value, surprisal, calm point, excess, per-horizon gate, calm summary (clock-anchored independent origins, null band, method-bias allowance).
- `app/engine.py`: the monitor records the gate per horizon; `surprise_summary` for the endpoint.
- `app/store.py`: new read `scored_with_quantiles`.
- `app/main.py`: `GET /series/{id}/surprise?hours=` (1-840).

## Files changed

- `services/orion-prediction/app/{surprise,engine,store,main}.py`: as above.
- `services/orion-prediction/tests/{test_surprise,test_engine}.py`: math, collapsed and non-monotone quantiles, calm point, gate reasons, clock-anchored thinning, long-window calm.
- `services/orion-prediction/evals/run_surprise_calm_live_eval.py`: live calm check (read-only).
- `services/orion-prediction/README.md`: Surprise section, eval command.

## Schema / bus / API changes

- Added: `GET /series/{id}/surprise` (HTTP, read-only).
- Removed / Renamed: none. Bus and schema registry: unchanged.
- Behavior changed: none for existing consumers.

## Env/config changes

- None. `.env_example` unchanged; local `.env` sync not needed; no skipped keys.

## Tests run

```text
docker run orion-prediction-prediction (worktree mounted :ro) pytest tests -> 300 passed
python scripts/check_metric_lineage.py --gate -> PASS (784 URNs)
```

## Evals run

```text
evals/run_surprise_calm_live_eval.py --hours 168 / --hours 24 (live, read-only): numbers above. Calm check FAILS today.
```

## Docker/build/smoke checks

```text
UNVERIFIED live until deploy. After deploy:
curl -fsS 'http://localhost:8132/series/athena.cabinet_temp_c/surprise?hours=24'
```

## Review findings fixed

- Finding (medium): the gate judged the model serving at scoring time, not the one that issued the forecast.
  - Fix: built (join to `prediction_model_run.assignment`, test), then removed with the bus event; it ships with the consumer patch. The remaining gate is display-only on the endpoint and drives nothing.
- Finding (low-medium): outcomes scored before the first monitor pass after a restart were lost to the event stream.
  - Fix: moot without the event; carried to the consumer patch.
- Finding (low): an exception building events could skip exceedance and shadow scoring.
  - Fix: the event path is gone; `score()` is unchanged from main.
- Finding (low-medium): `calm` read False for a calibrated forecaster over long windows; `mean_excess` and `calm` used different samples; thinning by row index broke on gaps.
  - Fix: thinned mean reported as `mean_excess`; thinning anchored to the clock; `calm` allows `METHOD_BIAS` (0.033). Evidence: `test_long_windows_do_not_call_a_calibrated_forecaster_uncalm`, gap case in `test_summary_thins_overlapping_origins_and_flags_calm`.
- Minor: non-monotone quantiles refused (`ValueError`); weak tests pinned (exact collapsed p = 0.075; summary gate reason computed from the served assignment).

## Restart required

Post-pull rebuild of `orion-prediction` from the clean primary main checkout after merge.

## Risks / concerns

- Severity: medium. Concern: the measurement rests below its calm point (ranges too wide on quiet days). Mitigation: nothing consumes it; next patch = volatility-adaptive ranges in the forecaster, then the substrate node with the bus event.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3170

🤖 Generated with [Claude Code](https://claude.com/claude-code)
