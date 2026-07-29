# PR Report: execution_prediction_error self-calibrating EWMA baseline

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1434
Branch: `fix/execution-prediction-error-ewma-baseline`

## Summary

- `execution_prediction_error` (`orion/substrate/prediction_error.py`) normalized via a fixed module constant (`_THRESHOLD=0.30`), the same disease already fixed for `recent_perturbations` (fixed `/20.0` cap, in-flight in a sibling worktree) and `bus_synaptic_prediction_error` (calm-floor bug, PR merged 2026-07-26) elsewhere in this codebase.
- Live Postgres data (120 real `substrate.execution_trajectory` receipts, 2026-07-28) confirmed this instrument reads ~0 essentially always (mean 0.0001, max ever observed 0.0009) -- real deltas run ~1000x below the constant, the mirror-image failure of bus_synaptic's old bug (that one couldn't read "calm"; this one can't read "surprised").
- Replaced the fixed divisor with an EWMA mean/variance baseline (`orion.bus.ewma.compute_ewma_update`), persisted on `ExecutionTrajectoryProjectionV1` itself, scoring each tick's raw delta as a z-score against its own live baseline.
- Found and fixed a *second*, deeper instance of the same disease while verifying the first fix against real replayed data: `compute_ewma_update`'s shared `_MIN_VARIANCE` (1e-6, calibrated for orion-bus-mirror's real-time-gap domain) is five orders of magnitude larger than execution's real warmed-up variance (~4e-11), so it silently dominated every z-score and reintroduced a milder version of the same bug one layer down. Added an optional `min_variance` override to `compute_ewma_update` (default unchanged for existing callers) and gave `execution_prediction_error` its own calibrated floor (1e-10).
- Scope was narrowed by live-data investigation, not assumed from code shape: `biometrics_prediction_error` is healthy (real spread 0-0.6, untouched), `transport_prediction_error` is already fully retired with zero live callers, `chat_prediction_error` has an unrelated bug (its reducer emits zero receipts of any kind) and is explicitly out of scope -- all per user decision after the investigation surfaced these facts.

## Outcome moved

`execution_prediction_error` goes from "structurally incapable of ever reading surprised" (max error 0.0009 across 120 real receipts) to a genuinely discriminating 0-1 signal (replayed against the same real history: max 0.97, mean 0.12, no saturation).

## Current architecture

`orion/substrate/prediction_error.py` has 6 "0-1 surprise score" functions diffing successive reducer-projection snapshots, feeding `services/orion-substrate-runtime/app/worker.py`'s per-domain ticks. `execution_prediction_error`, `biometrics_prediction_error`, and `chat_prediction_error` all shared a fixed `_THRESHOLD=0.30` divisor; `route_prediction_error` is deliberately exempt (categorical mismatch rate, documented); `bus_synaptic_prediction_error` already got its own EWMA-derived (calm-floor) fix 2026-07-26; `transport_prediction_error` was fully retired the same day.

## Architecture touched

- `orion/schemas/execution_projection.py` (`ExecutionTrajectoryProjectionV1`, Postgres-persisted, `substrate_execution_trajectory_projection` table)
- `orion/substrate/prediction_error.py` (`execution_prediction_error`)
- `orion/bus/ewma.py` (`compute_ewma_update`, shared utility)
- `services/orion-substrate-runtime/app/worker.py` (`_execution_tick`)

## Files changed

- `orion/schemas/execution_projection.py`: added `prediction_error_baseline_ewma`/`_var`/`_n` fields to `ExecutionTrajectoryProjectionV1`, defaulted for backward-compat with existing persisted rows
- `orion/substrate/prediction_error.py`: `execution_prediction_error` now computes an EWMA/z-score baseline instead of dividing by the fixed `_THRESHOLD`; new domain-specific `_EXECUTION_PREDICTION_ERROR_MIN_VARIANCE`/`_EWMA_ALPHA`/`_ZSCORE_SATURATION` constants
- `orion/bus/ewma.py`: `compute_ewma_update` gained an optional `min_variance` parameter (default = existing `_MIN_VARIANCE`, no behavior change for existing callers)
- `services/orion-substrate-runtime/app/worker.py`: `_execution_tick` now unconditionally re-saves `curr_projection` after computing `error`, so the mutated baseline fields persist every tick, not only when `error > 0.0`
- `orion/substrate/tests/test_prediction_error.py`: rewrote the `execution_prediction_error` test block for the new cold-start/z-score semantics; added a regression test locking in the domain-specific variance floor
- `services/orion-substrate-runtime/tests/test_worker_execution_tick_baseline_persistence.py` (new): regression coverage for the worker-level persistence fix

## Schema / bus / API changes

- Added: `ExecutionTrajectoryProjectionV1.prediction_error_baseline_ewma` / `_ewma_var` / `_ewma_n` (defaults `0.0`/`0.0`/`0`)
- Removed: none
- Renamed: none
- Behavior changed: `execution_prediction_error`'s return semantics -- a cold-start tick (no baseline yet) now always returns `0.0` regardless of delta magnitude (previously returned `min(1, delta/0.30)` immediately); once a baseline exists, the return is a z-score-derived surprise score, not the old numeric scale
- Compatibility notes: Pydantic fills the new fields' defaults on any existing persisted row missing them (verified directly against a real live row, 2000 runs, before merge)

## Env/config changes

None.

## Tests run

```
PYTHONPATH=. pytest orion/substrate/tests/test_prediction_error.py orion/bus -q
80 passed

cd services/orion-substrate-runtime && PYTHONPATH=../..:. pytest tests -q --ignore=tests/test_grammar_consumer_integration.py
175 passed, 13 failed, 9 errors
```

The 13 failed + 9 errors are pre-existing, confirmed via `git stash` on this same worktree (re-running the two prediction_error-adjacent failures against unpatched `main` gives identical failures) -- unrelated to this patch (cursor-reset-auth env requirements, a stale poll-task-count assertion, a dynamics-engine fixture gap). No new failures; 175 passed vs 173 on main (the 2 new regression tests added here).

## Evals run

No dedicated eval harness exists for `orion/substrate/prediction_error.py`. Substituted with real-data replay:
- Backed out real raw deltas from 120 live `substrate.execution_trajectory` receipts (`error * 0.30`, none saturated) and replayed them through the new formula, confirming a genuine, non-degenerate 0-1 spread (max 0.97, mean 0.12, no saturation) vs. the old formula's effective ceiling of 0.015.
- Confirmed a real existing persisted `ExecutionTrajectoryProjectionV1` row (2000 runs) parses through the updated schema with new fields correctly defaulting.

## Docker/build/smoke checks

Not run -- pure Python/Postgres-projection change, no Docker-relevant config/dependency/port/compose changes. `scripts/check_substrate_projection_schema_drift.py` was attempted against live Postgres but errored on a missing `asyncpg` dependency in this environment (pre-existing gap); the manual real-row-parse check above substitutes for it.

## Review findings fixed

- Finding: shared `compute_ewma_update`'s `_MIN_VARIANCE=1e-6` floor (borrowed from orion-bus-mirror's real-time-gap domain) is ~5 orders of magnitude larger than execution's real warmed-up variance (~4e-11), silently dominating every z-score and reintroducing a milder version of the exact bug this patch fixes (max error only 0.015 under the shared floor, confirmed by replaying real historical data)
  - Fix: added an optional `min_variance` parameter to `compute_ewma_update` (default preserves existing behavior for all other callers); `execution_prediction_error` passes its own calibrated floor (1e-10, one order of magnitude below the smallest real warmed-up variance observed)
  - Evidence: replayed all 120 real historical receipts through both floors -- shared floor: max error 0.015, mean 0.0019; domain floor: max error 0.97, mean 0.12, no saturation. New regression test (`test_execution_prediction_error_uses_domain_specific_variance_floor`) locks this in.
- Finding: no test coverage for the worker-level persistence fix -- a regression there would silently revert the whole fix (baseline never survives a restart) with nothing catching it
  - Fix: added `test_worker_execution_tick_baseline_persistence.py`, asserting the save happens both when `error == 0.0` and when `error > 0.0`
  - Evidence: both new tests pass.
- Finding (noted, not changed): alpha=0.2's justification (reused from orion-bus-mirror's per-real-event cadence) doesn't fully transfer, since execution's "observations" are already batch-means over variable-sized batches, not individually homogeneous samples. Left as a documented starting-point default subject to retuning against more real data.

## Restart required

No restart required for this branch alone -- ships as part of the next `services/orion-substrate-runtime` deploy/restart cycle.

## Risks / concerns

- Severity: Low
- Concern: `alpha=0.2` and `_EXECUTION_PREDICTION_ERROR_MIN_VARIANCE=1e-10` are both calibrated against ~120 real receipts from one measurement window (2026-07-28), a reasonable starting point, not a long-term-verified constant. If execution's real delta scale drifts by orders of magnitude in the future, this floor would need revisiting.
- Mitigation: both constants are named, module-level, and commented with the exact real-data justification and numbers behind them, making a future recalibration a one-line, well-documented change.

## Post-merge live verification (2026-07-28 22:12 UTC -> 2026-07-29 04:21 UTC, ~6h)

- Confirmed the redeployed container actually ran the new code (`grep`'d `_EXECUTION_PREDICTION_ERROR_MIN_VARIANCE` inside the live container's `prediction_error.py`), not just "the PR merged."
- Background monitor polled every 10 min for container health, exceptions, and real `execution_prediction_error` receipt stats. 7 heartbeats across ~6h, all healthy: mean 0.33-0.52, 8-35% saturated at 1.0, never fully pinned to 0 or 1. Zero exceptions/tracebacks the whole window.
- The container restarted once mid-window (03:56 UTC, clean: `RestartCount: 0`, `ExitCode(prev): 0`, unrelated to this patch) -- an unplanned real-world test of the persistence fix. `prediction_error_baseline_ewma_n` read **5507** afterward (not reset to 0), confirming the baseline survives a process restart via the unconditional `save_execution_trajectory` call, exactly as designed.
- Cumulative post-restart stats (76 ticks, ~29 min -- older rows already pruned by the routine `substrate_receipt_safe_prune` retention job, not data loss): mean 0.4378, min 0.0024, max 1.0, 18.4% saturated, 0% stuck-at-zero.

Also documented in `services/orion-substrate-runtime/README.md`'s own dated entry for this fix.

## Related, deliberately out of scope

- `recent_perturbations`' EWMA fix: separate, in-flight worktree at the time of this patch (`../Orion-Sapienform-recent-perturbation-baseline`, branch `fix/recent-perturbation-baseline`) -- same root-cause pattern, independent files, no collision.
- `chat_prediction_error`: live-confirmed its reducer emits zero receipts of any kind right now (not just zero prediction-error receipts) -- a different bug class from normalization, needs its own investigation.
- `biometrics_prediction_error`: live-confirmed healthy (real spread 0-0.6), no action needed.
- `transport_prediction_error`: already fully retired 2026-07-26, zero callers, no action needed.
