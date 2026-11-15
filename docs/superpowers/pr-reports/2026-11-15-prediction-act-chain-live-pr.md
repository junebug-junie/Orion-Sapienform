# Act chain goes live, and the night cells keep the night mix

## Summary

- Orion's act chain (Family B; PR #3345) now runs live in orion-prediction. Every 15 minutes it saves a forecast of which act Orion does next. 20 minutes after that bucket ends, it scores the forecast.
- New table `prediction_act_chain_forecast` (additive migration, auto-applied after pull). New endpoint `GET /act-chain`: the current forecast, how many buckets got a forecast, and the 14-day live grade in the same shape as the replay. Flag `PREDICTION_ACT_CHAIN_ENABLED`, on.
- Root cause of the night overconfidence found and partly fixed. The rarer acts (`ask`, `sense`, `infer`, `say`) happen more at night than forecast. The served chain backed off to "what follows the last act", which is learned mostly in daytime. Now a thin cell backs off to a blend of the last act and the clock block (parallel backoff). The blend weight is chosen by leave-one-out on training data only.
- Not here: no bus event, node or consumer. The night block is still overconfident (see Risks).

## Outcome moved

- Forecasts are stored before their bucket's outcome exists, so the first real live check of Family B starts accumulating now.
- On the same 17 replay days, the served model got better on every measure:

| same 1,536 buckets, 10-30..11-15 | before | after |
|---|---|---|
| served loss (nats/bucket) | 1.789 | **1.770** |
| vs clock hour: gap/day ± SE, prob_better | -0.044 ± 0.066, 0.74 | **-0.061 ± 0.062, 0.83** |
| vs clock block: prob_better | 0.89 | **0.96** |
| vs marginal: prob_better | 0.96 | **0.98** |
| excess surprise overall | +0.087 ± 0.073 | **+0.049 ± 0.071** |
| excess, asleep block | +0.182 | **+0.156 ± 0.051** |
| excess, day / evening / weekend morning | +0.14 / +0.09 / -0.09 | +0.08 / +0.06 / -0.10 |

- Rest-state check: the rolling 1-hour excess has a median of +0.04. It sits at or below 0 in 47 % of hours, and its 5-95 % range is -0.73..+1.00. It returns to calm.
- Calibration: when the top state has a forecast p of 0.2-0.4, it happens 31 % of the time. At 0.4-0.6 it happens 43 %. At 0.6-0.8 (18 buckets) it happens 61 %.
- The biggest surprise is still the first `predict` act Orion ever opened (10-30 20:45Z, 10.3 nats).

## Why the night was overconfident (evidence)

- Per night, the asleep-block excess was positive on 13 of 16 nights. It started in the first week, so it is not the 11-13 dream downscale I had guessed.
- Split by observed state, the excess comes from `ask` (+0.12/bucket), `sense` (+0.09), `infer` (+0.08) and `say` (+0.06). `look`, `make` and `idle` pull it down.
- Night cells of (clock block, previous act) are thin: about 20 buckets per night. The old backoff gave them the all-day mix.
- After the fix, nights are still +0.16 ± 0.05. Part of the cause remains. Still open in #3344.

## Current architecture

`app/act_chain.py` was replay-only (PR #3345). Its served cell backed off to P(state | previous act) only.

## Architecture touched

orion-prediction only, plus one additive migration in orion-sql-db. No bus or schema-registry changes.

## Files changed

- `services/orion-prediction/app/act_chain.py`:
  - `ActChain` issue/settle/live, `observe()`.
  - Parallel backoff, with ties kept at the old weight.
  - Per-block excess in `verdict()`.
  - A 30-s commit margin after each cutoff.
  - A row from another verb vocabulary is skipped, and no longer blocks other rows.
- `services/orion-prediction/app/store.py`: the act chain table in the boot schema check; issue/pending/settle/rows methods.
- `services/orion-prediction/app/main.py`: a 30-s loop, `GET /act-chain`, a health entry.
- `services/orion-prediction/app/settings.py`, `.env_example`, `docker-compose.yml`: `PREDICTION_ACT_CHAIN_ENABLED=true`.
- `services/orion-sql-db/manual_migration_prediction_act_chain_forecast_v1.sql`: the new table. `REQUIRED-BY: orion-prediction`, additive only.
- `services/orion-prediction/evals/run_act_chain_replay_eval.py`: reports the blend weight separately.
- `services/orion-prediction/tests/test_act_chain.py`: 17 tests (8 new). They cover:
  - issue timing (slack, commit margin, once only)
  - settle grace in and out, and settles that run hours late
  - live equals replay to 1e-12 on 3 origins
  - stale `current`, and the coverage count
  - vocabulary drift
  - thin night cells, compared with and without parallel backoff
- `services/orion-prediction/README.md`: the live section and the new replay numbers.

## Schema / bus / API changes

- Added:
  - table `prediction_act_chain_forecast`
  - `GET /act-chain`
  - an `act_chain` entry in `/health`
- Removed / renamed: none.
- Behavior changed: the served act-chain model's backoff (eval-only until now).
- Compatibility notes:
  - The table is now in the boot schema check, as for reachable and the bill forecast.
  - The `REQUIRED-BY` header makes the deploy gate refuse `up` until the migration is applied.
  - The post-pull rebuild applies it automatically. It is the same statement kinds as `manual_migration_prediction_reachable_forecast_v1.sql`, and the reviewer confirmed `auto_apply_refusals` returns `[]` while the table is absent.

## Env/config changes

- Added keys: `PREDICTION_ACT_CHAIN_ENABLED` (default `true`).
- `.env_example` updated: yes.
- Local `.env` synced: yes. The worktree has no `.env`, so the key was appended to the primary checkout's `services/orion-prediction/.env`. `scripts/check_env_template_parity.py` → PASS (89 services).
- Skipped keys requiring operator action: none.

## Tests run

```text
pytest services/orion-prediction/tests -q (prediction image)   -> 467 passed (before the last test rewrite); test_act_chain.py 17 passed after it
mutation checks (one-line bugs): settle without grace (2 forms), no parallel backoff, forecast ignoring the blend,
  condition on final view, train on final view, plain-argmax tie-break, replay without grace, t_hi - 1  -> each fails the suite
python scripts/check_migration_required_by.py   -> OK
python scripts/check_env_template_parity.py     -> PASS
```

## Evals run

```text
evals/run_act_chain_replay_eval.py --since 2026-10-23 (live DB, read-only)  -> numbers above
old model (main) over the same window, same script                          -> "before" column
```

## Docker/build/smoke checks

```text
Read-only live smoke in the orion-prediction image against the live DB:
  forecast() for 2026-11-15 00:00Z bucket: prev_state=say, block=day, n_train=2686, 0.13 s,
  top look 0.30 / idle 0.22 / make 0.16 / say 0.13
  observe() of 2026-11-14 23:15Z bucket: idle, 0 acts
Not deployed: the table is not applied yet (to_regclass null); deploy happens through the post-pull rebuild after merge.
```

## Review findings fixed

The first review was of the live loop (orion-repo-agent, by hand plus mutations). Its scratch test showed live equals replay for all 192 settled buckets, to rtol 1e-12.

- Finding: the settle grace was untested. Two mutations that drop the grace passed.
  - Fix: a late-settle test with one act inside the grace (counts) and one past it (never counts).
  - Evidence: both mutations now fail.
- Finding: the "rows after the origin change nothing" check was a tautology, and no test compared live to replay.
  - Fix: `test_live_issue_equals_replay_once_history_covers_the_lookback` (31-day stream, 3 origins, every model, rtol 1e-12).
  - Evidence: test passes.
- Finding: one bad pending row (for example, after a verb-vocabulary change) blocked every later settle. The same drift could also 500 `/act-chain`.
  - Fix: settles run first, one row at a time, with try/except. Rows from another vocabulary are skipped in grading and counted as `ungradable_rows`.
  - Evidence: `test_a_row_from_another_vocabulary_neither_blocks_settles_nor_breaks_the_grade`.
- Finding: a row stamped just before a cutoff could commit just after the live read, so live and replay could disagree.
  - Fix: `COMMIT_MARGIN` = 30 s. Issue no earlier than origin + 30 s; settle at window end + grace + 30 s. The cutoffs are unchanged.
  - Evidence: issue/settle timing tests updated.
- Finding: `current` showed the newest row even when it was stale.
  - Fix: `current` only when its origin is the current bucket; `newest_origin` is reported beside it.
  - Evidence: asserted in the live test.
- Finding (pre-existing, not in this diff): close-before-open acts keep an early `created_at`.
  - Fix: none needed. Live check 11-13: 0 rows with `created_at < opened_at`.

The second review was of the parallel-backoff commit.

- Finding: the night test passed without the feature (the probed cell held about 180 transitions, so it wasn't thin).
  - Fix: the test now builds a truly thin (asleep, look) cell and asserts parallel backoff > 0.35 > prev-only backoff, via monkeypatch.
  - Evidence: mutations removing the blend now fail.
- Finding: ties went to blend weight 0.
  - Fix: the grid is searched in reverse, so ties keep 1.0 (the old prev-only behaviour).
- Finding: the blend weight was reported under the K label.
  - Fix: `lambda_prev_median` is reported separately.
- The LOO math, the log(0)/NaN risk and the forecast-time parents were checked: no findings.

## Restart required

```text
No manual restart. After merge, the post-pull rebuild applies the migration, then rebuilds orion-prediction.
Check: curl -fsS http://localhost:8132/act-chain
```

## Risks / concerns

- Severity: medium. Concern: night (asleep block) is still overconfident, +0.16 ± 0.05 nats. Mitigation: no consumer; #3344 stays open for the rest of the cause and for the 7-day live check.
- Severity: low. Concern: the table is required at boot. If the auto-apply refuses it, orion-prediction stays on the old image. Mitigation: the `REQUIRED-BY` gate names the exact apply command; the statement kinds match an already auto-applied migration.
- Severity: low. Concern: the skill vs the clock-hour guess is still not settled (P 0.83 over 17 days). Mitigation: the live grade accumulates in `GET /act-chain`.
- Issues: Refs #3344 (the live half is done; the night calibration and the 7-day live check stay open). Opened: none. Closed: none.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
