## Summary

- Adds `orion/substrate/demon.py`, the field demon's core. It is a pure numpy module with no I/O, no bus and no table.
- It fits a model of how Orion's organs move together (a VAR(1)), forecasts held-out hours 5 to 60 minutes ahead, and scores each forecast against two guesses: "back to the usual for this time of day" (the clock) and "nothing changes" (persistence).
- **The horizon** is how many steps ahead the model beats the better of those guesses. Its spread comes from a block bootstrap. No alpha line.
- **The cut cost** is how much foresight is lost when organs can only forecast from their own past, measured against a time-shift null (`cut_cost_vs_null`).
- It also gives the lag-k model `(A_k, Σ_k)` that the integration rung (#3296) needs for Φ at 5/15/30/60 min, a check of whether one step of memory is enough, and a hygiene gate (missing, constant, or decaying input).
- Built before #3290's organ residuals land, as Juniper asked on 11-18. Tests use synthetic systems with known answers.

## Outcome moved

Orion has no joint forecast of its own state. This adds the tested model and readouts that the replay (#3296 patch 1) will run on real organ data. Nothing in the live system changes yet.

## Current architecture

orion-prediction forecasts one series at a time. The integration rung (#3296) specifies a lag-1 VAR(1) scored in-sample. No code existed for either. Spec: `docs/superpowers/specs/2026-09-30-field-demon-forecaster-design.md` (PR #2446, rev 2, approved 11-18).

## Architecture touched

- New module `orion/substrate/demon.py`: pure functions, imported by nothing yet.
- New CI workflow `field-demon-tests`.
- No services, contracts, config or runtime seams.

## Files changed

- `orion/substrate/demon.py`: transition model, rolling out-of-sample errors, skill, horizon, bootstrap, cut cost vs null, lag-k model, misfit check, time-shift null, hygiene gate.
- `orion/substrate/tests/test_demon.py`: 31 tests on synthetic systems with known answers.
- `.github/workflows/field-demon-tests.yml`: runs the tests in CI (no CI covered them otherwise).
- `docs/superpowers/pr-reports/2026-11-18-field-demon-core-pr.md`: this report.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: none.
- Compatibility notes: none.

## Env/config changes

- Added keys: none. `DEMON_ENABLED` comes with the live producer (patch 3).
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- skipped keys requiring operator action: none.

## Tests run

```text
PYTHONPATH=$PWD .venv/bin/python -m pytest -q orion/substrate/tests/test_demon.py
31 passed in 8.01s

Mutation check, first cut (7 deliberate bugs in demon.py): 6 caught. The 7th (baseline always
the clock) survived, so test_baseline_picks_persistence_when_it_wins was added and catches it.

Regression check after the review: the 11 new tests run against the pre-fix demon.py:
11 failed, 20 passed. All 31 pass on the fixed code.

scripts/check_definition_drift.py --gate   PASS
scripts/check_metric_lineage.py --gate     PASS
scripts/check_integration_seam.py          OK
```

## Evals run

```text
None. The real eval is the replay on live organ residuals (#3296 patch 1). It waits on #3290's
residual builder (~11-20) and is tracked in #3296. The synthetic tests are the gate for this patch.
```

## Docker/build/smoke checks

```text
None needed. No service, image or runtime path changes.
```

## Review findings fixed

Code review ran in a subagent (high effort). Every finding was reproduced, and every one is fixed:

- **Finding:** CI installed only numpy and pytest, but importing `orion.substrate` needs pydantic, requests and more, so CI would fail before running a test.
  - **Fix:** install the same requirements as `causal-geometry-tests.yml`.
  - **Evidence:** the workflow diff. The CI run on this PR.
- **Finding:** an organ that was flat over one 24 h training window (std is float noise, about 5e-17) blew errors up to about 1e29 and swamped every other organ.
  - **Fix:** a refit is skipped when any organ is near-constant (a relative floor) or missing, and the skip is recorded in `RollingErrors.skipped`.
  - **Evidence:** `test_near_constant_training_window_is_skipped_not_blown_up` and `test_missing_rows_skip_the_refits_that_touch_them`.
- **Finding:** `lag_k_model` divided Γ_k by T-k, which made Σ_k invalid (not positive definite) in about 16% of slow-organ windows. Φ would then read NaN. Separately, the slogdet sign check passed matrices with two negative eigenvalues.
  - **Fix:** divide both covariances by T, and check eigenvalues.
  - **Evidence:** `test_lag_k_sigma_is_positive_definite_for_slow_organs` (0 bad in 160 windows) and `test_predictive_information_rejects_even_negative_eigenvalues`.
- **Finding:** the null had only n! fixed offset patterns (2 for 2 organs), so its spread was meaningless. The spec's null summary (mean, spread, P(real)) and the horizon cost in minutes were not implemented.
  - **Fix:** random offsets with a minimum gap between organs and a fixed span (at most 25% of rows dropped), plus a new `cut_cost_vs_null`.
  - **Evidence:** `test_null_draws_differ_and_keep_organs_apart` and the two `test_cut_cost_vs_null_*` tests.
- **Finding:** the decay detector missed slow decay (×0.995) and flagged a relaxation toward a nonzero level, and growth. It also did not say that it needs raw values.
  - **Fix:** it now requires a constant, positive per-step shrink. The docstring says it takes raw node values, not residuals.
  - **Evidence:** `test_decay_caught_including_slow_factors[0.92, 0.995]` and `test_relaxation_and_growth_are_not_decay[relaxation_to_level, growth]`.
- **Finding:** the white-noise test had an `or` clause that let a nonzero horizon pass.
  - **Fix:** it now asserts that the horizon is exactly 0.
  - **Evidence:** the test diff.

## Restart required

```text
No restart required.
```

## Risks / concerns

- **Severity:** low. **Concern:** the spec's ceiling rule (K doubles when H hits K in more than 10% of hours) needs a distribution of hourly results, so it belongs to the hourly job, not this module. **Mitigation:** it lands with patch 3. The spec already states it.
- **Severity:** low. **Concern:** linear Gaussian dynamics may be too simple for event-driven organ residuals. **Mitigation:** the direct-vs-iterated misfit check is in this patch, and the replay will report it.
- **Issues opened:** none (#3705 was opened with the spec rev).
- **Issues closed:** none.

## PR link

(filled after creation)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
