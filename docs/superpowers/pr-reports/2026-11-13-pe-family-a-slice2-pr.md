# Family A slice 2: the whole body and room forecast live, with one honest joint surprise per bucket

## Summary

- Orion now forecasts every raw body and room reading (~40: machine watts, temperatures, fans, load,
  cabinet sensors, AC, office and closet air) 5 and 15 minutes ahead, every 5 minutes, inside
  orion-prediction's own loop. Each forecast is stored before its target bucket ends and graded when
  the real reading arrives. (`services/orion-prediction/app/family_a_live.py`)
- The hygiene gate runs on every issue. When an input is stale, stuck, missing or out of its trained
  range, nothing is forecast for that bucket: a `refused` row stores the reasons and a log line names
  them. It never forecasts on bad inputs.
- One joint surprise number per bucket and horizon is stored: "how surprised is the whole body right
  now", with coupled readings counted once. It is now calibrated against the last hour's difficulty
  (adaptive). On held-out data and on the last 24 h this fixes the old one-width score: calm days no
  longer read too calm, and busy days no longer read 25 % "rare event". (`prediction_joint_surprise`)
- The three slice 1 review items are fixed: missing readings are flagged at every lag, the input screen
  never sees the backtest's held-out rows, and bootstrap blocks never join two held-out stretches.
- Still no bus event, no substrate node, no consumer. That is the next patch, gated on a 7-day calm check.

## Outcome moved

Family A moved from "measured once, offline" to "issued, stored and graded live every 5 minutes", with
the gate enforced at run time. The joint score's calibration problem from slice 1 is fixed and measured:
on the last 24 h (a busy day, host load ~150) the old static score read +1.00 / +0.84 with 28-29 % of
buckets at p <= 0.05; the served adaptive score reads +0.04 / +0.04 with 7 %, both 95 % intervals
containing 0.

## Current architecture

orion-prediction ran select -> issue -> score for two single readings (cabinet and office temperature;
`app/engine.py`), with `prediction_model_run`, `prediction_forecast`, `prediction_shadow_forecast` and
adaptive ranges (`app/adaptive.py`). Family A (slice 1, PR #3267) existed only as pure functions plus an
offline eval (`app/family_a.py`, `evals/run_family_a_baselines_eval.py`): nothing stored, nothing served.

## Architecture touched

- orion-prediction: a new loop task (`_family_a_forever` in `app/main.py`) runs `FamilyALive.tick` in a
  thread every `PREDICTION_TICK_SEC`: score -> issue -> select (every `PREDICTION_SELECT_INTERVAL_SEC`)
  -> prune (daily). Independent of the single-series engine.
- Tables reused: `prediction_model_run` (`series_id = 'family_a'`: one `model` row per rung + the served
  `champion_blend`, whose `metrics` hold everything scoring needs), `prediction_forecast` (per reading,
  `series_id = 'family_a:<reading>'`, role champion), `prediction_shadow_forecast` (the unserved rung).
- New table: `prediction_joint_surprise` (one row per bucket and horizon; lifecycle issued -> scored /
  missing, or refused).
- `orion/self_monitor/collect.py`: the forecast source skips `family_a:` rows, so the self-monitor does
  not start grading ~23k new rows a day with no consumer decision behind them.

## Files changed

- `services/orion-prediction/app/family_a_live.py` (new): the live loop, the in-memory store for replay
  and tests, a frame loader for replays.
- `services/orion-prediction/app/family_a.py`: `fit_horizon` / `HorizonFit.predict` (serve what the
  backtest fits), masks at every lag, `backtest` (train-rows-only screen), segment-aware block bootstrap,
  adaptive joint calibration (`JointCal`, `score_joint`, `difficulty_rows`), shared Gram matrices.
- `services/orion-prediction/app/store.py`: family methods (issue, due, scores, recent scored, shadow,
  prune) and the new table in the boot schema check.
- `services/orion-prediction/app/main.py`, `app/settings.py`, `.env_example`, `docker-compose.yml`: the
  loop task, `/health` -> `family_a`, two env keys.
- `services/orion-prediction/requirements.txt`: `threadpoolctl==3.7.0` (already in the image as a
  transitive dependency; now imported directly).
- `services/orion-sql-db/manual_migration_prediction_joint_surprise_v1.sql` (new).
- `services/orion-prediction/evals/run_family_a_calm_live_eval.py` (new): the calm check on stored rows,
  and `--replay-hours` to dry-run the live loop in memory.
- `services/orion-prediction/evals/run_family_a_baselines_eval.py`: train-rows screen, static vs adaptive.
- `services/orion-prediction/tests/test_family_a.py`, `tests/test_family_a_live.py` (new).
- `orion/self_monitor/collect.py`: skip `family_a:` rows.
- `services/orion-prediction/README.md`, `config/prediction/family_a.v1.yaml` (comment only).

## Schema / bus / API changes

- Added: table `prediction_joint_surprise`; `/health` key `family_a`; `prediction_forecast` rows with
  `series_id` prefix `family_a:` and `interval_source = 'validation'`; `prediction_model_run` rows with
  `series_id = 'family_a'`.
- Removed: none.
- Renamed: none.
- Behavior changed: the self-monitor's forecast source excludes `family_a:` rows (it never saw them before).
- Compatibility notes: no bus channel or schema registry change. The migration is additive (one table,
  two indexes) and passes the post-pull auto-apply allowlist (`auto_apply_refusals` = []); header
  `ORION-MIGRATION-REQUIRED-BY: orion-prediction`. The service's boot schema check now needs the table,
  so the deploy wrapper refuses `up` until it is applied (the post-pull rebuild applies it first).

## Env/config changes

- Added keys: `PREDICTION_FAMILY_A_ENABLED=true` (kill switch, ships ON),
  `PREDICTION_FAMILY_A_CONFIG_PATH=/app/config/prediction/family_a.v1.yaml`.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes (`services/orion-prediction/.env_example`), plus `docker-compose.yml`.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes, both keys present in
  the primary checkout's `services/orion-prediction/.env`.
- skipped keys requiring operator action: none.

## Metric quality gate: family A joint surprise (served, adaptive)

1. **Provenance.** `FamilyALive.score`: residual per reading = (actual 5-min mean - stored point) / the
   issuing run's training std; d^2 = r' Sigma^-1 r on the readings that arrived (`family_a.joint_d2`,
   Sigma from the run's fit-row residuals, shrunk and eigen-clipped); adaptive tail = chi-square sf of
   d^2 / s on the observed count, s = mean d^2/dof of the scored rows whose target fell in the hour up
   to the origin (stored on the row at issue time as `difficulty`, floored); p = conformal against the
   run's validation rows scaled the same way (`JointCal.cal_tail_adaptive`); excess = -ln p - calm point.
2. **Independence.** Against `node:prediction.body` / `.room` (cabinet and office temperature, one
   reading each): those two readings carry 2 % of the joint score on the last 24 h; the largest shares
   are athena load (21-23 %), board temp, office eCO2, CPU temp. Shared chain: the same two sensors
   only. Against `node:substrate.biometrics` (the node the next patch replaces): that node is the mean change of each
   machine's hint values (`node_biometrics` v3, `orion/schemas/prediction_error_definitions.py`), not a
   forecast miss; family A refuses hint channels as inputs at config load and reads raw measurements
   only. Not redundant.
3. **Theory anchor.** Mahalanobis distance of a forecast-residual vector (chi-square under a Gaussian
   residual), made exactly uniform under exchangeability by split conformal; locally weighted
   (normalized) conformal (Lei et al. 2018) for the heteroscedastic hours. Calm = E[-ln p] of a uniform
   conformal p, so excess rests at 0 by theory.
4. **Live sanity.** Not live yet (nothing deployed). Replay of the live loop over real data
   2026-11-12 07:11 .. 11-13 07:11 UTC (in memory): returns to rest (hourly mean -0.92 .. +1.37 at 5 min,
   -0.99 .. +1.52 at 15 min; longest run above +0.5: 1-2 h), not pinned at 0 (p <= 0.05 7 %, spikes
   named by `top_shares`: athena load, office eCO2). The static score on the same rows stayed above +0.5
   for 8 h at a time. 7-day live check: #3309.
5. **Existing mechanism.** `app/adaptive.py` (same normalized conformal, per single reading); reused
   the idea, not the code (it calibrates intervals, not a joint p). No other joint score exists.
6. **Reversibility.** `PREDICTION_FAMILY_A_ENABLED=false` stops the loop; no consumer reads the table;
   drop `prediction_joint_surprise` and delete `family_a:` rows to remove it entirely.

## Calibration result (held-out data, `run_family_a_baselines_eval.py --days 14 --seeds 0 1 2`)

Window 2026-10-30 07:10 .. 11-13 07:10 UTC (final code, after review fixes), 5 purged blocks, model
trained without the held-out rows. Calibrated means: CI contains 0, p <= 0.05 ~ 0.05, p <= 0.2 ~ 0.20,
hours at rest ~ the null share.

| horizon | seed | calibration | mean excess | 95 % CI | p <= 0.05 | p <= 0.2 | hours at rest (null) | longest h > 0.5 |
|---|---|---|---|---|---|---|---|---|
| 5 min | 0 | static | -0.000 | [-0.201, +0.190] | 0.062 | 0.194 | 0.57 (0.54) | 4 |
| 5 min | 0 | adaptive | -0.004 | [-0.082, +0.070] | 0.055 | 0.203 | 0.56 (0.54) | 2 |
| 5 min | 1 | static | +0.217 | [-0.051, +0.526] | 0.082 | 0.223 | 0.45 (0.54) | 5 |
| 5 min | 1 | adaptive | +0.026 | [-0.084, +0.116] | 0.059 | 0.203 | 0.56 (0.54) | 3 |
| 5 min | 2 | static | +0.205 | [+0.004, +0.359] | 0.081 | 0.251 | 0.45 (0.54) | 4 |
| 5 min | 2 | adaptive | -0.006 | [-0.065, +0.082] | 0.060 | 0.195 | 0.57 (0.54) | 2 |
| 15 min | 0 | static | +0.066 | [-0.174, +0.316] | 0.097 | 0.229 | 0.56 (0.54) | 5 |
| 15 min | 0 | adaptive | -0.004 | [-0.098, +0.107] | 0.050 | 0.240 | 0.56 (0.54) | 2 |
| 15 min | 1 | static | +0.095 | [-0.186, +0.465] | 0.082 | 0.192 | 0.67 (0.54) | 5 |
| 15 min | 1 | adaptive | -0.039 | [-0.163, +0.057] | 0.052 | 0.175 | 0.58 (0.54) | 3 |
| 15 min | 2 | static | +0.158 | [-0.053, +0.335] | 0.057 | 0.226 | 0.52 (0.54) | 4 |
| 15 min | 2 | adaptive | -0.062 | [-0.148, +0.042] | 0.056 | 0.175 | 0.65 (0.54) | 1 |

Summary: adaptive CI contains 0 in 6/6 runs (static 5/6), p <= 0.05 0.050-0.060 (static 0.057-0.097),
intervals 2-3x narrower. Hours at rest 0.56-0.65 vs 0.54 (static 0.45-0.67): adaptive is slightly on the
calm side on held-out blocks, within the 0.10 tolerance the calm check uses.

Live rest check (trained before the last 24 h, scored on it):

| horizon | calibration | n | mean excess | 95 % CI | p <= 0.05 | hours at rest (null) | longest h > 0.5 |
|---|---|---|---|---|---|---|---|
| 5 min | static | 287 | +1.001 | [+0.011, +1.904] | 0.275 | 0.44 (0.55) | 8 |
| 5 min | adaptive | 287 | +0.036 | [-0.103, +0.244] | 0.070 | 0.48 (0.55) | 1 |
| 15 min | static | 285 | +0.838 | [-0.062, +1.646] | 0.288 | 0.50 (0.54) | 7 |
| 15 min | adaptive | 285 | +0.043 | [-0.135, +0.315] | 0.070 | 0.54 (0.54) | 1 |

Verdict: adaptive is served. It is tighter and centred on every split; static was too calm on slice 1's
quiet day and far too hot on this busy one. Trade-off, stated: a busy stretch reads as surprise for about
an hour, then as the new normal.

Ladder (same run, with masks at every lag and the train-rows screen): `ridge_var` still beats the best
baseline at every seed, 5 min x0.939-0.959, 15 min x0.968-0.981 (paired t p <= 0.021, bootstrap p <= 0.03).
On 3 seed-0 targets (athena GPU0 watts and temp, closet CO2) the train-rows screen now refuses what the
whole-window screen kept: review item 2 doing its job. The live selection serves a horizon's `ridge_var`
only when both seeds win; in the 24 h replay that held for 5 min in 3 of 4 selections and for 15 min in
1 of 4 (the rest served `ar`). 9 days hold every target; the neural rung waits (21 days, 2026-11-26
earliest).

## Tests run

```text
cd services/orion-prediction   (orion-prediction image + pytest/httpx/sqlalchemy, worktree mounted read-only)
OPENBLAS_NUM_THREADS=4 PYTHONPATH=../..:. python -m pytest tests/test_family_a.py tests/test_family_a_live.py -q
  -> 64 passed (family A + live loop, after review fixes)
OPENBLAS_NUM_THREADS=4 PYTHONPATH=../..:. python -m pytest tests -q   -> 449 passed, 2 warnings in 284 s (after review fixes)
python -m pytest -q orion/self_monitor/tests      -> 148 passed (2 new: family A exclusion pinned)
python3 scripts/check_migration_required_by.py   -> OK (1 new migration file)
auto_apply_refusals(manual_migration_prediction_joint_surprise_v1.sql) -> []
python3 scripts/check_env_template_parity.py      -> PASS
python3 scripts/check_env_key_single_source.py    -> OK
scripts/check_metric_lineage.py --gate            -> PASS
scripts/check_inner_state_registry.py             -> OK
```

Postgres smoke (throwaway `postgres:16-alpine` container, not the live DB; run before and after the
review fixes): applied
`manual_migration_prediction_v1.sql`, `_shadow_v1`, `_forecast_interval_source_v1`, `_joint_surprise_v1`
(re-apply is a no-op), then ran `PostgresStore` + `FamilyALive` for 30 ticks: 3 run rows, 108 reading rows
scored, 108 shadow rows scored, 54 joint rows scored (40 adaptive, 14 `static:no_difficulty` in the first
hour), 0 rows issued at or after their target bucket's end, a fresh instance scored the 4 rows the first
left pending using the stored run, refused rows stored `{"s.y": ["stale"]}` as jsonb, the self-monitor's
forecast query returned 0 family rows, prune removed every reading and shadow row (232) and kept the 82
joint rows.

## Evals run

```text
OPENBLAS_NUM_THREADS=4 POSTGRES_URI=... python evals/run_family_a_baselines_eval.py --days 14 --seeds 0 1 2
  -> tables above (107 s, read-only)
OPENBLAS_NUM_THREADS=4 POSTGRES_URI=... python evals/run_family_a_calm_live_eval.py --replay-hours 24
  -> dry run of the live loop over real data 2026-11-12 07:11 .. 11-13 07:11 UTC, in memory, nothing stored:
     4 selections (18-27 s each, 40 targets), 288 ticks, 287 issues, 0 refused, 0 late, 570 joint rows scored
     5 min  served +0.066 [-0.066, +0.269]  p<=.05 0.066  hours at rest 0.48 (0.55)  longest >0.5: 1 h  adaptive 98 %
     15 min served +0.067 [-0.079, +0.286]  p<=.05 0.074  hours at rest 0.52 (0.54)  longest >0.5: 2 h  adaptive 97 %
     (static on the same rows: +0.620 / +0.606, p<=.05 0.231 / 0.218, 8 h above +0.5)
     per-reading 80 % coverage: median 0.804 over 41 readings; athena load 0.41, CPU watts 0.43 (#3310)
     Verdict: NOT YET (0.99 days of scored rows < 7) -- the 7-day check is #3309
```

## Docker/build/smoke checks

```text
docker build -f services/orion-prediction/Dockerfile -t orion-prediction-fama-build:check .   -> ok
  (scratch tag, so the live image tag is untouched)
docker run ... orion-prediction-fama-build:check python -c "import app.main, app.family_a_live ..."
  -> import ok, threadpoolctl 3.7.0, 45 inputs parsed from /app/config/prediction/family_a.v1.yaml,
     family_a_enabled True
Not deployed (task scope). Live path after deploy: UNVERIFIED.
```

## Review findings fixed

Code review ran in a subagent on the full diff (no blockers; it reproduced the live difficulty against
the backtest's `difficulty_rows`: 66 rows compared, 0 mismatches).

- Finding: a scoring backlog past the 2000-joint-row limit scored readings without their joint row; those
  joints later found no readings and were written `missing` for good (reproduced with an in-memory store).
  - Fix: each pass scores only the readings of the joint rows it took (`FamilyALive.score`).
  - Evidence: `test_a_scoring_backlog_past_the_joint_limit_never_strands_a_joint_row` (limit 5, 15
    buckets of outage: every joint row scored with both readings).
- Finding: an empty calibration pool gave p = 1 and calm point 0, i.e. "calm" on no evidence; a tiny
  adaptive pool gave coarse p steps.
  - Fix: `conformal_p` returns NaN with no pool; a static pool under `MIN_CAL` (200) refuses the
    selection; an adaptive pool under 200 is dropped (rows fall back to static, labelled); a row with no
    p is `missing` (`no_calibration`).
  - Evidence: `test_no_calibration_rows_is_no_p_never_a_calm_p`, `test_a_thin_calibration_pool_refuses_the_selection`,
    `test_a_thin_adaptive_pool_is_dropped_not_served`.
- Finding: the BLAS thread limit was switched on and off every tick while the engine thread could be
  inside BLAS.
  - Fix: set once, process-wide, at start (`family_a_live.limit_blas`, called from `lifespan` and the replay).
  - Evidence: replay ran with it (selections 18-27 s on the loaded host).
- Finding: no test pinned the self-monitor exclusion.
  - Fix: `orion/self_monitor/tests/test_self_monitor_family_a_excluded.py` (prefix and its length).
  - Evidence: 148 passed.
- Nits fixed: the kill switch now removes the boot dependency on the new table and a bad config stops
  family A only; log chi-square tails (`chi2_logsf`, finite far into the tail) so extreme misses no longer
  tie at 0; in-memory calibration rounded exactly as stored (same scores before and after a restart);
  segments computed after the NaN filter; the `issued` counter counts only written issues; prefix filters
  are index-friendly ranges (`_prefix_range`, tested); the calm eval counts days OF scored rows and fails
  under 90 % adaptive rows; unused helper removed; README notes select delay and unpruned run rows.
- Nits left, stated: the calm eval's static column reuses the served row's calm point (differs by
  < 0.001 at ~900-row pools); no `GRANT SELECT ... TO orion_readonly` on the new table (a GRANT would fail
  the post-pull auto-apply allowlist; the shadow table set the same precedent).

## Restart required

```bash
git -C /mnt/scripts/Orion-Sapienform pull --ff-only   # post-pull rebuild applies the migration first, then rebuilds orion-prediction
```

If the post-pull rebuild does not run, by hand from the clean primary main checkout:

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_prediction_joint_surprise_v1.sql
scripts/safe_docker_build.sh orion-prediction up -d --build
curl -fsS http://localhost:8132/health | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin)['family_a'], indent=1))"
```

## What is written live after deploy

- Every ~6 h (and at start, ~15-20 s of CPU): 3 `prediction_model_run` rows (`series_id = 'family_a'`).
- Every 5 min: ~40 `prediction_forecast` rows per horizon (~80 per bucket, ~23k a day) and ~80
  `prediction_shadow_forecast` rows, each scored ~5-15 min later; 2 `prediction_joint_surprise` rows
  (one per horizon; 576 a day), scored when due, or 2 `refused` rows with reasons.
- Reading rows are deleted after 35 days; joint rows and run rows are kept.

## Next patch and its gate

**Next patch:** publish the joint surprise and give it its consumer. A `prediction.joint_surprise.v1`
event on a new channel (`orion/bus/channels.yaml`, `orion/schemas/registry.py`) per scored joint row that
passes an issue-time gate (served calibration `adaptive`, status scored, the run that issued it is not
serving a baseline at that horizon); orion-substrate-runtime folds it into a new body node that replaces
`node:substrate.biometrics`, and the old node's tick and producer are killed in the same patch (no partial
exclusion). Coordinate with the agent currently retiring substrate PE nodes in `orion/substrate` and the
field digester.

**Gate (all three before that patch merges):**
1. #3309: `run_family_a_calm_live_eval.py --hours 168` prints PASS on >= 7 days of live rows.
2. #3311: a live cabinet AC off/on event scores as one event, not one per coupled reading (spec rule 5).
3. The refusal rate over those 7 days is reported. If one flaky sensor refuses the whole body often,
   decide (with Juniper) whether that sensor leaves the config. Never loosen a gate threshold.

## Risks / concerns

- Severity: medium. Concern: the gate refuses the WHOLE bucket when any one of ~41 inputs is stale or
  stuck (spec: refuse, never degrade), so one flaky sensor can blank the joint score. Mitigation: refusals
  are stored with reasons; 0 refusals in the 24 h replay; #3309 reports the rate.
- Severity: low. Concern: per-reading 80 % ranges are one width (athena load covered 43 % on a busy day).
  Mitigation: no consumer reads them; #3310.
- Severity: low. Concern: OpenBLAS is held to 4 threads for the whole service from start, so the
  single-series engine's numpy calls run on 4 threads too. Mitigation: on this host that is faster, not
  slower (thread thrash at load ~150: > 120 s vs 1.5 s for one family A fit). Off with
  `PREDICTION_FAMILY_A_ENABLED=false`.
- Severity: low. Concern: after a restart the first hour of joint rows uses the static p
  (`calibration = 'static:no_difficulty'`). Mitigation: labelled on every row; the calm check reports it.
- Severity: low. Concern: volume, ~23k reading rows and ~23k shadow rows a day. Mitigation: pruned after
  35 days (~0.8M rows each at steady state).
- `office.sound_dba` passes presence around 11-14 and will become a target at the next selection. Its level
  (75-81 dBA) was flagged in slice 1 and is not checked here.

Issues opened: #3309 (verify-live, 7-day calm check), #3310 (follow-up, adaptive per-reading ranges),
#3311 (verify-live, AC event counted once). Issues closed: none.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3318

🤖 Generated with [Claude Code](https://claude.com/claude-code)
