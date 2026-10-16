## Summary

- The model gate now requires a **significant** win over the better baseline: a one-sided paired test across backtest windows, p < 0.05. A lower average alone is no longer enough.
- That test accounts for overlapping, correlated backtest windows. Its variance is the larger of a Newey-West estimate and a bias-corrected AR(1) inflation. A plain t-test would let about 17% of noise "wins" through.
- The leave-one-out coverage check is removed from the gate. It depended only on how errors rank, so every model scored exactly 0.85. It is replaced by a time-ordered holdout coverage that is reported but not gated.
- Selections now record a gate version. After this deploy, restore refuses old selections and reselects at boot, so the bad picks don't keep serving for up to 6 h.
- The replay eval judges holdout coverage with a binomial test on independent windows instead of a fixed band.

## Outcome moved

**Before:** live for 4.5 h after PR #2716, the served forecast was 1.4x worse than the baseline. Two picks lost live:

| pick | live vs baseline | backtest paired p |
|---|---|---|
| `window_average_1h` @ 1 h | 1.57x worse | 0.21 |
| `mstl_arima` @ 15 min | 2.0x worse | 0.31 |

**After:** on real history (40 windows), the gate serves:

| horizon | served by | paired p |
|---|---|---|
| 5 min | `direct_ar` | 0.03 |
| 15 min | `direct_arx` | 0.03 |
| 1 h, 6 h, 24 h | `naive` | (no candidate had a significant win) |

Both live losers would have been refused.

## Current architecture

`select_champion` served the lowest backtest MAE ratio below 1, plus a leave-one-out coverage band. The live monitor raised `worse_than_baseline`.

## Architecture touched

`services/orion-prediction/app/backtest.py` (gate and metrics), `app/engine.py` (gate version, step size passed to the gate), the replay eval, and tests. No bus, schema or env changes.

## Files changed

- `app/backtest.py`: `paired_win_p`, `overlap_lag`, `GATE_ALPHA`, `GATE_VERSION`; time-ordered holdout coverage in `horizon_metrics`.
- `app/engine.py`: passes the backtest step to the gate; stores and checks `gate_version`.
- `evals/run_cabinet_replay_eval.py`: shows p in each cell; binomial coverage check; served picks must have a significant win.
- `tests/test_backtest.py`, `tests/test_engine.py`: noise-win rejection, false-win rate on correlated windows, holdout drift detection, proof that leave-one-out coverage can't see drift, gate-version restore.
- `services/orion-sql-db/manual_migration_prediction_v1.sql`: comment only (metric keys). No schema change.

## Schema / bus / API changes

- Added: per-model metrics `p_win`, `n_paired`, `coverage80_holdout`, `coverage95_holdout`, `n_holdout`; blend metrics `gate_version`.
- Removed: per-model backtest `coverage80` and `coverage95` (leave-one-out).
- Renamed: none.
- Behavior changed: fewer non-baseline picks are served. A restart reselects once after deploy, taking about 5 min.
- Compatibility notes: nothing reads the removed keys. Live-scoring `coverage80` is unchanged.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: n/a.
- skipped keys requiring operator action: none.

## Tests run

```text
services/orion-prediction: pytest tests -q -> 70 passed
```

## Evals run

```text
evals/run_cabinet_replay_eval.py (40 windows, real history) -> PASS
  assignment {5: direct_ar (p 0.03), 15: direct_arx (p 0.03), 60/360/1440: naive}
false-win rate (Monte Carlo, n=40, AR(1) phi=0.5): iid t-test > 0.12, new test <= 0.09, power >= 0.8 at a 1-sd edge
live cabinet loss-difference lag-1 autocorrelation: 0.1-0.45
```

## Docker/build/smoke checks

```text
No dependency change; not rebuilt locally.
```

## Review findings fixed

- Finding: the t-test assumed independent windows, so p came out too small at long horizons.
  - Fix: long-run variance = max(Newey-West, bias-corrected AR(1)).
  - Evidence: `test_paired_test_keeps_false_wins_near_alpha_on_correlated_windows`.
- Finding: restore kept the old picks for up to 6 h after deploy.
  - Fix: `gate_version` is stored with each selection and checked on restore.
  - Evidence: `test_restore_refuses_a_selection_from_another_gate_version`.
- Finding: the eval coverage check lost its upper bound.
  - Fix: replaced by a binomial test on independent windows. A fixed band on 13 overlapping windows failed by chance in practice.
- Finding: eval output could crash on a None value before the failure list printed.
  - Fix: the print is None-safe.
- Finding: comments and docs didn't match the code.
  - Fix: test comment (measured values), migration metrics comment, `width80` meaning.

## Restart required

Run from the clean primary `main` checkout after merge:

```bash
scripts/safe_docker_build.sh orion-prediction up -d --build
```

## Risks / concerns

- Severity: low.
  - Concern: with no correction across the 8 candidates, about one false win per few selections is still possible.
  - Mitigation: live `worse_than_baseline` alerts; the selection reruns every 6 h.
- Severity: low.
  - Concern: when window correlation is around 0.8, the false-win rate is still about 10%.
  - Mitigation: live cabinet correlation is 0.1–0.45.
- Severity: info.
  - Concern: on this history the honest answer at 1 h and beyond is "same as now".
  - Mitigation: none needed; the gate now says so instead of serving noise.

## PR link

<link>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
