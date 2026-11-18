## Summary

Learning progress (`GET /learning-progress`, the "are Orion's forecasts getting better?" reading that Prime turns into a hunger) called a move too easily, and missed some of our own deploys. This PR fixes both, as far as the stored data allows.

- **Weather spells.** Daily forecast skill is correlated day to day. The old test treated each day as independent, so on a null series it called a move on about 9-13% of looks, not 5%. The standard error is now the exact one for an AR(1) day-to-day correlation `rho`. `rho` is estimated from up to 56 days of history, then shrunk toward a 0.3 prior. At `rho` 0 it is the old Welch test exactly.
- **Same-name deploys.** A change of gate rule or covariates under the same model names now counts as a roster change. It already happened twice on cabinet without being seen (10-21 covariates, 10-23 gate v3).
- **Not fixed: model code changes.** No table records a code or image version. That half of #3691 needs a schema change and stays open.
- **New null eval.** `evals/run_learning_progress_null_eval.py`: shuffled labels, circular shifts, and simulated AR(1) at the live spread. It prints the false-move rate per look and per week.
- Response gains `rho`, `rho_pairs`, `rho_source`, `se_iid`. Nothing is renamed; Prime still reads `direction` and `p_value`.

## Outcome moved

False-move rate under a null (simulated AR(1), live spread sd 0.156; shipped / old Welch). The 14-day column is the live regime today: both series have 14 days since their last roster change.

| rho | per look, 14 days | per look, 56 days | per week (7 overlapping looks, 56 days) |
|---|---|---|---|
| 0 | 0.017 / 0.042 | 0.020 / 0.040 | 0.120 / 0.193 |
| 0.26 (live, used) | **0.038 / 0.104** | **0.041 / 0.106** | 0.193 / 0.417 |
| 0.3 | 0.037 / 0.122 | 0.051 / 0.133 | 0.213 / 0.473 |
| 0.5 | 0.094 / 0.209 | 0.069 / 0.201 | 0.303 / 0.557 |

Live data: shuffled labels give cabinet 0.000 / 0.010 and office 0.016 / 0.041. Circular shifts give cabinet 0/14 / 0/14 and office 0/14 / 8/14. Office's real 11-06 step is in that data, so the circular-shift number is an upper bound.

## Current architecture

`app/learning_progress.py` (PR #3692, live since 2026-11-18) is pure. It compared two 7-day windows of daily skill (`1 - MAE champion / MAE baseline`, 15 min horizon) with a Welch t test, Student's cut at the Welch-Satterthwaite df, and `MIN_EFFECT` 0.05. It dropped days up to the last change in the set of fitted model *names* (`prediction_model_run`, kind `model`). `orion/prime/hunger.py` uses `drive = 1 - p_value` for better/worse readings.

## Architecture touched

- orion-prediction only. No schema, bus, or env change. Computed on read, as before.
- `app/main.py` reads 56 days of history instead of 15, so it can estimate `rho`.

## Files changed

- `services/orion-prediction/app/learning_progress.py`: `moment_rho` + `estimate_rho` (shrunk toward the prior), AR(1) window moments and covariance, `roster_keys`, new fields, docstring.
- `services/orion-prediction/app/store.py`: `daily_rosters` adds the config of the day's last `champion_blend` run (`gate_version`, `covariates`, `overlap_covariates`; a missing list reads as `[]`).
- `services/orion-prediction/app/main.py`: history window = `RHO_HISTORY_DAYS`; response adds `rho`, `rho_pairs`, `rho_source`, `se_iid`.
- `services/orion-prediction/tests/test_learning_progress.py`: planted AR(1) test, seeded null-rate test (14 and 56 days), Welch equivalence at rho 0, `estimate_rho` (spells vs step vs trend vs prior), `roster_keys` same-name config change. One existing test now pins `rho=0.0`, because it tests the Student cut under independent days.
- `services/orion-prediction/evals/run_learning_progress_null_eval.py`: new null eval (baseline in docstring).
- `services/orion-prediction/evals/run_learning_progress_live_eval.py`: prints `rho` and `se_iid`; before/after baseline.
- `services/orion-prediction/README.md`: "Learning progress" section.

## Schema / bus / API changes

- Added: response fields `rho`, `rho_pairs`, `rho_source`, `se_iid` per series (`SeriesProgress` gains them with defaults).
- Removed: none.
- Renamed: none.
- Behavior changed: `se`, `t`, `t_crit`, `p_value` are the AR(1) ones. The roster key now includes the selection config, so more days can read `insufficient` after a gate or covariate change.
- Compatibility notes: Prime reads `direction` and `p_value` only; `orion/prime/hunger.py` is unchanged.

## Env/config changes

- Added keys: none. Removed keys: none. Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change).
- skipped keys requiring operator action: none.

## Metric quality gate (the changed standard error and roster key)

1. **Provenance.** Daily skill: `store.daily_skill_pairs` (paired champion/baseline `abs_error` per UTC day at 15 min). `rho`: `learning_progress.estimate_rho` over the same days. Roster config: `prediction_model_run.metrics` on `champion_blend` rows, written at `app/engine.py:359` (`gate_version`, `covariates`, `overlap_covariates`).
2. **Independence.** `rho` is not a new signal fed to Prime. It is a nuisance parameter of the existing reading and only widens or narrows its error bar. The roster config tokens come from the same selection run as the model names.
3. **Theory anchor.** For a stationary AR(1) series, the variance of a mean of n correlated days is `s^2 * 1'R1 / n^2`, not `s^2 / n`. The sample variance is biased by `tr(AR)/(n-1)`, and adjacent windows co-vary by `s^2 * 1'R_rp 1 / (n_r n_p)`. The df come from the Satterthwaite chi-square match `tr(AR)^2 / tr((AR)^2)`. A lag-1 estimate from short series is biased low (about -(1+4rho)/n). So `rho` is moment-matched against the exact expectation of the detrended-block statistic, not taken raw. Twelve pairs still swing the estimate between 0 and 0.4, so it is shrunk toward the prior (worth 24 pairs).
4. **Live sanity.** Measured lag-1 on the live daily skill (read-only SQL, 2026-11-18): cabinet +0.070 (n 14, 11-04..11-17), office +0.21 (n 12, 11-06..11-17); +0.377 over 14 days with office's pre-promotion 0.0 days, which are a step. Both are inside sampling noise (about +/-0.55 at these n), but the effect on the rule is material: in simulation at rho 0.2 Welch false-moves 9%. The moment-matched rho: cabinet 0.37, office 0.00 (12 pairs each, noisy); used after shrinkage: 0.32 and 0.20. Rest still works: today both series read `flat` (cabinet -0.073 t -0.67 cut 2.43; office +0.031 t 0.28 cut 2.22). The signal can still move: seeded gradual-improvement and decline tests read better and worse.
5. **Existing mechanism.** No AR(1) or block-bootstrap SE elsewhere in orion-prediction. `gate_version` already existed and is what `engine.py` uses to refuse restoring a stale selection; the roster key reuses it.
6. **Reversibility.** Pure function, computed on read, no stored state. `progress(..., rho=0.0)` gives the old Welch test exactly.

## Tests run

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider services/orion-prediction/tests/test_learning_progress.py -q
17 passed
Full services/orion-prediction/tests: branch 27 failed / 554 passed; clean main 27 failed / 549 passed.
The 27 failure names are identical (local package versions; CI is the real check).
```

## Evals run

```text
POSTGRES_URI=<127.0.0.1:55432> PYTHONPATH=../..:. python evals/run_learning_progress_null_eval.py
PASS: shipped per-look false-move at live rho 0.26: 0.041 (target 0.05)   (full table above)

POSTGRES_URI=... python evals/run_learning_progress_live_eval.py
cabinet: flat 3, insufficient 17 (was flat 8, insufficient 12): the 10-23 gate v2->v3 change now counts
office: flat 3 (was better once on 11-16, +0.228 t 2.86)
```

The office 11-16 `better` is gone because of the roster key, not `rho`. The gate rule changed on 11-03 (v4: live promotion also needs window coverage). The office champion started serving on 11-06. The prior window now loses 11-02..11-03: lp +0.183, Welch t 2.03 against a cut of 2.32, so it is flat even at rho 0. The old call was plausibly our gate deploy, not Orion learning (cause UNVERIFIED).

## Docker/build/smoke checks

```text
No Docker build: pure Python, no dependency or compose change (numpy and scipy already pinned).
Live read-only check of the new daily_rosters SQL against conjourney: detects 10-21 (covariates), 10-23 (gate v3), 11-03 for cabinet; 11-03 for office.
```

## Review findings fixed

Code review ran in a subagent. It found no blockers. It confirmed the SE, covariance, df, moment matching, SQL, and Prime back-compat.

- Finding (should): the null rate was only checked with 56 days of history. Live there are 14, and at 12 pairs the estimate clamped to 0 (plain Welch) about 30% of the time: 6.9% at rho 0.3.
  - Fix: shrink the estimate toward `RHO_PRIOR` with weight `RHO_PRIOR_WEIGHT` = 24 pairs. Add 14-day cases to the null eval and the null-rate test.
  - Evidence: 14 days, rho 0.3: 0.037 (was 0.069 per the review's simulation); the eval table above.
- Finding (should): `rho_pairs` 0 was documented as "prior used", but the code did not do that.
  - Fix: explicit `rho_source` field: `estimated` / `prior` / `given`. Docs updated.
  - Evidence: `test_estimate_rho_sees_spells_but_not_steps_or_trends`, `test_http_route_...`.
- Finding (nit): the `RHO_PRIOR` comment's live numbers did not match the eval.
  - Fix: the comment now quotes the eval's numbers.
- Finding (nit): Satterthwaite df used `(vr+vp)^2` in the numerator, not the tested variance `vr+vp-2cov`. That made it slightly liberal.
  - Fix: `se**4` in the numerator. Still exact Welch at rho 0 (`test_rho_zero_is_the_plain_welch_test`).
- Finding (nit): a missing `overlap_covariates` read as `None`, but an empty one read as `[]`. That was a latent false roster change.
  - Fix: `COALESCE(..., '[]'::jsonb)` for both lists.
- Finding (nit): the null eval's live section used list positions as days, so gaps looked consecutive.
  - Fix: it now passes real day ordinals.

## Restart required

```bash
scripts/safe_docker_build.sh orion-prediction up -d --build   # from clean primary main after merge
```

## Risks / concerns

- Severity: medium. Concern: limit (2) is half fixed. A change to a model's own code under the same name is still invisible, because no table records a code or image version. Mitigation: #3691 stays open (`Refs`), with a comment saying it needs a schema column (for example `prediction_model_run.code_version`).
- Severity: medium. Concern: re-checking daily on overlapping windows still gives about 19% false-move *per week* at the live rho (Welch 42%). Mitigation: Prime reads each look's `p_value` as a graded drive, not a one-shot alarm; the eval prints both rates so this stays visible.
- Severity: low. Concern: `rho` from 12-48 day pairs is noisy (live per series, raw: 0.37 vs 0.00). At true rho 0.5 the rule still false-moves 7-9% per look. Mitigation: shrinkage toward a 0.3 prior; the history grows to 56 days; the clamp, prior, and weight are documented constants.
- Severity: medium. Concern: power is about halved. A real 0.02/day gain in skill (0.14 between windows, independent noise sd 0.156) reads `better` 13% of looks at 14 days of history and 20% at 56 (Welch: 28% and 33%). The 0.3 prior holds `rho` at 0.2 or more even when days are truly independent. Mitigation: that is the price of a 5% false-move rate under the measured spells. If Prime's learning hunger turns out starved of `better` readings, the dial is `RHO_PRIOR_WEIGHT` (lower = trust the data sooner).
- Issues opened: none. Issues closed: none (#3691 referenced, left open for the code-version half).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3710

🤖 Generated with [Claude Code](https://claude.com/claude-code)
