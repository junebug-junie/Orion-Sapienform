# Family C slice 1: what do my own actions do? (effect table + efference copy, eval only)

## Summary

- Adds the **effect table**. It has one row per (actuator, raw reading, window after the act), with an
  ON-minus-OFF estimate, a 95 % interval, SE, blocks per arm and experiment ids. Rows come only from
  finished randomised switchback experiments. The table is recomputed read-only from the logged
  blocks and the raw readings. Nobody writes a row by hand. (`app/family_c.py`, `GET /effects/table`)
- Adds `choose()`, the only reader for acting or forecasting. It refuses any row whose method is
  not `randomised`, so observational and conditional results can never choose an action.
- Adds the **efference copy**. Before an act, it takes a Family A forecast whose bucket closed before
  the act started and moves its quantiles by the table's effect. After the act, it scores the real reading:
  "I did X, expected Y, got Z", in Family A's excess-surprise unit (calm = 0 by theory).
- Adds the **model gate**. Does knowing the effect beat the forecast alone? Gain is the log predictive
  density difference, a proper score. The gate reports the mean gain over acts, SE and prob_better.
  There is no alpha line.
- Eval/replay only: no bus event, no substrate node, no consumer, no new live actuator, no
  experiment scheduled, nothing written to the database.

## Outcome moved

Before this patch, the causal level 3 answered one question per experiment, about one target. Its
only reader (`best_effect`) would hand back an observational answer when no experiment existed.
Now every experiment is read against every raw reading Family A reads. A scorer can say how wrong
Orion's own expectation about its act was. The scorer rests at 0 on live data when there is no act.

## Current architecture

- `app/causal.py`: three labelled levels (conditional, observational, randomised switchback).
- `app/experiment_runner.py`: the only actuator is `vision.focus:<stream>`. During an ON block a
  Redis key with a TTL forces the stream's triggered tier. It has heat guards, a lock and boot recovery.
  It analyses ONE target series (`prediction_experiment.series_id`) and a watts manipulation check.
- `app/effects.py::best_effect`: newest experimental row, else pooled observational, else none.
- Family A (`app/family_a_live.py`) has been live since 11-13. It makes 5/15-min forecasts of ~42
  raw readings with 9 stored quantiles each (`prediction_forecast`, `series_id = family_a:<id>`).

## Inventory (live, 2026-11-15)

- **What the experiment runner can do today:** randomise ON/OFF blocks of ONE actuator, which only
  ADDS vision focus on one camera stream. It is guarded by the cabinet heat forecast. It cannot
  act on GPU load, power or cooling.
- **Experiments ever run:** one. `9e6f9ab7` was `vision.focus:walkway`, on 2026-10-29 12:05-16:05Z,
  with 12 x 20-min blocks, a 10-min burn-in, status `analyzed`. Stored result for the cabinet:
  +0.21 C (95 % -0.16..+0.58, p 0.23).
- **Is it usable?** Yes, as a randomised row with an honest wide interval. It is weak as a
  lever. The manipulation check shows +50 W on circe and +8 W on athena. But in 3 of the 6 OFF
  blocks the vision router's own `triggered:expectation` tier ran at the full ON rate (477 dispatches
  vs 476 for ON). So half the "off" arm was effectively on, and the contrast is diluted. Heat-per-watt
  predicts only +0.009 C from the watts it moved.
- **Actuators with a real on/off switch:** only vision focus. The cooling act exists, but the AC is
  off for the season (skipped by design). No GPU load-shaping actuator exists. `PowerIntentV1`
  (orion-biometrics) measures a declared draw; it does not change one.

## Live effect rows

No rows are stored (the table is computed, not written). Recomputed live from the one experiment
(`evals/run_family_c_live_eval.py`), 45 rows came out, and 31 have an estimate. The other 14 readings
(per-GPU watts, closet, some office sensors) had no data on 10-29. **1 of 31 rows has p <= 0.05**
(`athena.cabinet_vibration_g`, -0.0003 g). With 31 rows, chance alone gives about 1.5. Every other
interval contains 0. Cabinet temp: +0.29 C (95 % -0.47..+1.04). This differs from the stored
+0.21: the stored analysis reads the series registry's view (exclusion windows, gap fill), and this
table reads Family A's plain 5-min bucket means. I did not pin down which difference moves it. Plain
answer: Orion has measured one act, and it cannot yet tell what that act does to any reading.

## Architecture touched

- `services/orion-prediction` only, plus the instruments registry and the CI workflow.
- New read-only HTTP route. No bus, schema, channel, env or migration change.

## Files changed

- `services/orion-prediction/app/family_c.py`: effect table, pooling, choice guard, efference copy,
  quantile PIT scorer, gate, the `EffectTable` orchestrator (cached per experiment).
- `services/orion-prediction/app/main.py`: `GET /effects/table` (declared before `/effects/{action}`),
  built from Family A's input config and loader.
- `services/orion-prediction/app/store.py`: `experiments(status)` read.
- `services/orion-prediction/tests/test_family_c.py`: 21 tests (synthetic switchbacks, refusals,
  PIT, calm/spike/gain, widening cannot win, unaligned logged blocks, route).
- `services/orion-prediction/evals/run_family_c_synthetic_eval.py`: planted end-to-end eval (CI).
- `services/orion-prediction/evals/run_family_c_live_eval.py`: live read-only table + rest check +
  planted-act replay on real forecasts.
- `services/orion-prediction/README.md`: Family C section + eval commands.
- `orion/sentience_striving_program/instruments.yaml`: `action_effect_table` (O2) with two manual claims.
- `.github/workflows/orion-prediction-tests.yml`: runs the synthetic eval.

## Schema / bus / API changes

- Added: `GET /effects/table` (read-only JSON).
- Removed / Renamed: none.
- Behavior changed: none for existing routes. `best_effect` is untouched.
- Compatibility notes: no table, bus or schema change.

## Env/config changes

- Added / Removed / Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change).
- skipped keys requiring operator action: none.

## Metric quality gate: the efference-copy miss score

1. **Provenance.** `family_c.score()`. `pit()` reads the stored Family A quantiles
   (`prediction_forecast.quantiles`, written by `family_a_live.issue`) after `shifted()` moves them by
   `EffectRow.estimate` (from `causal.analyze_switchback` on the raw readings) and widens them by
   sqrt(1 + (se / sigma80)^2), where se is read off the row's own t interval. p = 2 min(u, 1-u),
   excess = -ln p - 1. The gate's gain is `log_density()` with the effect minus without it.
2. **Independence.** With no act, it is exactly Family A's per-reading surprise. That is the same
   forecasts and the same unit, so it is NOT new signal at rest. It is not the joint surprise
   (`prediction_joint_surprise` whitens across readings; this is per reading, per act). The new
   information is only the act term: the gain vs the forecast alone, and the miss when the table is
   wrong. It shares no input with the `node:prediction.*` nodes beyond Family A, and it feeds none.
   Any future node must use the gain/act-conditional part, never the at-rest score, or it would
   double-count Family A.
3. **Theory anchor.** Efference copy / forward model: an agent predicts the sensory result of its
   own motor command and registers the mismatch. The prediction is causal only because the effect
   comes from randomised blocks. Calm = 0 because E[-ln U] = 1 for a calibrated p.
4. **Live rest check** (48 h, 33,656 scored Family A forecasts, no act): mean excess +0.099
   (5 min, 95 % -0.067..+0.323) and -0.030 (15 min, -0.132..+0.087). Both contain 0. p <= 0.05 share
   is 0.045 / 0.044, and hours at rest 0.59 / 0.65. **A first version that read a single Gaussian off the
   80 % range did NOT rest: +0.375 (+0.17..+0.69) / +0.165 on the first run, and +0.614 / +0.257 on the
   re-run.** Family A's misses have heavier tails than a
   Gaussian, so the scorer now reads all 9 quantiles. Per-reading outliers: `athena.load_1m` +4.4,
   `athena.cabinet_humidity_pct` +1.9 (Family A's per-reading coverage, tracked in #3310). Planted act
   (+2 sigma) on real forecasts: if the table knows the act, excess is -0.18 / -0.19; if it is blind,
   +1.60 / +1.80. Gain +1.72 (SE 0.09) / +2.06 (SE 0.12), prob_better 1.000. **No act has been scored
   live** (no experiment since Family A started on 11-13). UNVERIFIED on a real act.
5. **Existing mechanism.** `best_effect` returns one experimental row per target and falls back to
   observational; `manipulation_watts` reads ON-OFF watts. Neither builds a multi-reading table or
   scores an expectation against an outcome. `action_bets` scores bets on acts (stated, not measured
   effects), and `turn_self_bets` scores turn stances. Neither uses randomised effects. Nothing was
   duplicated: the estimator is `analyze_switchback`, the loader is Family A's, and design matching is
   `Effects._match_design`.
6. **Reversibility.** Cheap. There is no table, schema, bus event, node or consumer. Removing it means
   deleting one module, one route, one store method, two evals and one instrument entry.

Model gate (efference copy vs forecast alone, log score): synthetic eval gap +6.49 (SE 0.107,
prob_better 1.000) with a right table. Null actuator: gap -0.088 (SE 0.012). That is the honest cost of
an uncertain table about an effect that is really 0, and it never reads as a gain. Wrong-sign table:
gap -13.98 (SE 0.141, prob_better 0.000). Live planted: above. Live real acts: none yet.

## Proposal: the first new switchback experiment (needs Juniper yes/no, #3482)

**GPU power-cap switchback on circe GPU3.**

- **Capability change.** Orion would get a second actuator, `gpu.power_cap:circe.gpu3`. In an ON block the
  card's power limit drops (`nvidia-smi -i 3 -pl 120`). In an OFF block it is restored to the card
  default. The plan is 16 x 20-min randomised blocks (~5.3 h), 5-min burn-in, run once by hand via
  the existing runner. A cap only ever REMOVES heat, so the direction is safe for the cabinet. GPU3
  averaged 105 W (68-186 W) over the last 48 h, so a 120 W cap binds often enough to be a real lever.
  GPU0 averages 39 W, so a cap there would almost never bind.
- **Data touched.** New `prediction_experiment` / `prediction_intervention` rows (existing tables).
  Reads: raw readings already in Family A. Nothing else is written.
- **Privacy boundary.** None crossed. The data is hardware telemetry only.
- **Trace that proves it worked.** The manipulation check shows `circe.gpu3_watts` ON-minus-OFF below 0
  with its interval excluding 0. `GET /effects/table` then shows rows for circe GPU3 temp, fan, chassis
  watts, closet temp and cabinet temp. Then score the efference copy on the next un-randomised uses.
- **Dangerous failure mode.** The cap gets stuck ON. Unlike the vision key, a GPU power limit has no
  TTL and survives a crashed runner. The lane on GPU3 then serves slower until someone notices. Also,
  `nvidia-smi -pl` needs root on circe, and orion-prediction runs on athena without it. This needs a
  small host helper on circe with a watchdog that restores the default after a TTL. That helper is
  the real new machinery and the reason this is a proposal.
- **Disable / rollback.** Do not start the run (nothing schedules it). If a cap is stuck:
  `ssh circe@circe sudo nvidia-smi -i 3 -pl <default>` (read the default from `nvidia-smi -q -d POWER`).
  The helper's watchdog does this automatically after its TTL.
- **Question for Juniper:** yes/no to building the circe power-cap helper (root, watchdog, TTL)
  and running one 16-block experiment on GPU3?

## Tests run

```text
docker run (orion-prediction image + pytest), services/orion-prediction:
python -m pytest tests -q            -> 504 passed (pre-review), 509 passed (post-review)
python -m pytest tests/test_family_c.py -q -> 21 passed
python scripts/check_sentience_instruments.py --static-only -> All claims hold.
```

## Evals run

```text
python evals/run_family_c_synthetic_eval.py
  table right : CI coverage 0.93  excess -0.056  p<=.05 0.046  gap +6.491 se 0.107 prob_better 1.000
  null act    : excess -0.098  gap -0.088 se 0.012 prob_better 0.000
  wrong sign  : excess +57.440  gap -13.978 se 0.141 prob_better 0.000
  PASS
python evals/run_family_c_live_eval.py --hours 48   (read-only, live DB)
  table: 1 analyzed experiment, 45 rows, 31 with an estimate (see "Live effect rows")
  rest:  5 min +0.099 (-0.067..+0.323); 15 min -0.030 (-0.132..+0.087) [Gaussian read-off +0.614 / +0.257]
  planted act: knows -0.184 / -0.192 (gain +1.72 / +2.06), blind +1.600 / +1.796
```

## Docker/build/smoke checks

```text
Tests and evals ran inside the orion-prediction image (deps baked in). No new dependency.
GET /effects/table is not live until orion-prediction is rebuilt from main. UNVERIFIED live.
```

## Review findings fixed

A code-review subagent read the diff. Findings and fixes:

- Finding (must-fix): the table loader asked for a grid starting at the logged block start. Logged
  blocks can be up to 2 min off the design, while the SQL buckets sit on 5-min boundaries, so every
  column would come back empty.
  - Fix: floor the start and ceil the end to the grain before loading.
  - Evidence: `test_effect_table_aligns_an_unaligned_logged_start_to_the_loader_grid` (blocks logged 97 s late).
- Finding: a forecast with origin == act start had already seen the act. Family A's origin is the
  START of the last complete bucket.
  - Fix: refuse unless origin + grain <= act start.
  - Evidence: test updated to refuse exactly that case.
- Finding: the effect's SE was the raw Welch SE (t(dof) ~ 2.2 at 6 blocks). Pooling and widening
  used it, so the efference copy came out too narrow.
  - Fix: `EffectRow.eff_se`, read off the row's own 95 % interval, used for pooling and widening.
  - Evidence: pooling and widening tests assert the interval-implied SE.
- Finding: `choose()` took the first covering row even if it had no estimate.
  - Fix: filter to measured rows first, then take the most precise one.
  - Evidence: `test_choose_skips_rows_without_estimate_and_prefers_the_most_precise`.
- Finding: an act shorter than the window got the full effect.
  - Fix: `efference_copy` takes `act_end` and refuses a target after it.
  - Evidence: `test_efference_copy_refuses_an_act_that_ended_before_the_target`.
- Finding: a naive datetime crashed with TypeError.
  - Fix: all times normalised to UTC.
  - Evidence: `test_efference_copy_accepts_naive_utc_times`.
- Finding: the profile spans change events (differs from the stored `/effects` analysis).
  - Fix: documented in `rows_from_experiment`. The profile is variance reduction only, and Family A's
    readings have no registry spec to read breaks from.
- Nits fixed: `experiments()` now orders newest first; tied quantiles are nudged apart for `np.interp`;
  `prob_better` = 0.5 on an exact tie; `expected` is the shifted median, the centre the PIT uses.
- Found while re-running the evals after these fixes (not by the reviewer): `gain` was the
  two-sided-p surprisal difference. Any widening lowers that score, so a table that only added
  uncertainty "won" (the null actuator read gap +0.079, SE 0.017).
  - Fix: `gain` is now the log predictive density difference (a proper score).
  - Evidence: `test_a_table_that_only_adds_uncertainty_cannot_win_the_gate`. Synthetic null now reads -0.088.
- Not fixed (nit): the table does not cache an experiment that fails. With one experiment this costs
  one reload per failing GET.

## Restart required

```bash
scripts/safe_docker_build.sh orion-prediction up -d --build   # from clean primary main after merge
curl -fsS http://localhost:8132/effects/table | head -c 600
```

## Risks / concerns

- Severity: low. Concern: `GET /effects/table` loads ~14 days of raw readings per experiment on the
  first call (about 10 s for the one experiment, then cached). Mitigation: the work runs in a thread
  and is cached per finished experiment.
- Severity: low. Concern: 45 readings per experiment means about 1 in 20 rows shows p <= 0.05 by
  chance. Mitigation: the table's own note says so. A slice that acts on rows must pre-register its
  target or correct for multiplicity.
- Severity: medium. Concern: the one experiment's OFF arm was half contaminated (router expectation
  tier). Mitigation: reported here; #3484.
- Issues opened: #3482 (first new experiment, needs-juniper), #3483 (verify-live: GET /effects/table), #3484 (vision-focus OFF arm contamination)
- Issues closed: none.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3485

🤖 Generated with [Claude Code](https://claude.com/claude-code)
