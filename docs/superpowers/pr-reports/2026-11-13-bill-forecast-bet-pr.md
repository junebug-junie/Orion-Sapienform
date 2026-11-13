# Power-bill bet: Orion makes a daily, gradeable guess at the power bill

## Summary

- Once a day, orion-prediction saves Orion's own guess at the current Rocky Mountain Power (RMP) bill:
  total kWh and pre-tax dollars, each with an 80 % range. The guess is the run-rate projection Orion
  already computed for the stakes snapshot (`orion.energy.reconcile.project_period`), which was never
  stored before.
- When RMP's bill for that period lands (`energy_bill_actual`), every guess for the period is settled
  against it: kWh billed, and the pre-tax total on the same basis as `reconcile._utility_pretax`.
- Past days are backfilled from stored meter history as `origin = 'replay'`, knowing only what the
  meter feed had delivered by then. Replays are reported, never graded.
- The self-monitor grades live guesses as two faculties, `forecast.energy.bill_kwh` and
  `forecast.energy.bill_usd`. The ~30 daily guesses at one bill share one answer, so they count as
  ONE sample (`SettledBetV1.answer_key`). Status stays `unknown` until 10 bills have settled: about
  10 months.
- Found while checking live data, NOT fixed here: one missing hour at the 11-02 DST change has frozen
  Orion's whole energy ledger since 2026-11-02 08:00 UTC (see Risks).

## Outcome moved

Orion had no money bet: the self-monitor listed `energy.bill` as ungradeable ("Orion makes no forward
bill bet"). Now there is one, with a stated confidence, settled by the world. The first live bill
(period from 2026-10-19) should land ~11-19..11-22.

## Current architecture

- `orion/energy/reconcile.py:project_period` projected period-end kWh and pre-tax $ from metered hours.
  Used live by `orion/energy/stakes.py:72`, never stored.
- `energy_reconcile` had 2 rows, both `actual` (Orion priced a completed period vs RMP's bill). The
  `forecast` kind only reconciles RMP's own forecast (`energy_bill_forecast`, 0 rows ever).
- Self-monitor counted every settled bet as an independent sample and used one 7-day window for every
  source, so a monthly answer could never reach `MIN_GRADED` (10).

## Architecture touched

- orion-prediction: new loop `app/bill_forecast.py` (issue daily, replay once per boot, settle),
  store section, `GET /bill-forecast`, `/health.bill_forecast`, tariff file baked into the image.
- New table `prediction_bill_forecast` (additive migration).
- `orion/self_monitor`: two adapters + SOURCES, units for shared answers, per-source window for slow
  sources, catalog entries, plain names.
- `orion/schemas/self_monitor.py`: `SettledBetV1.answer_key` (optional).

## Files changed

- `services/orion-prediction/app/bill_forecast.py`: the bet, the range, replay, settlement, grading view, in-memory store for evals/tests.
- `services/orion-prediction/app/store.py`: bill section; read-only reads of `energy_usage_interval` / `energy_bill_actual`; boot schema check.
- `services/orion-prediction/app/main.py`: loop, `/bill-forecast`, health.
- `services/orion-prediction/app/settings.py`, `.env_example`, `docker-compose.yml`: flag, tick, the four `ENERGY_*` keys (same values as orion-energy).
- `services/orion-prediction/Dockerfile`: `COPY config/energy`.
- `services/orion-prediction/evals/run_bill_forecast_replay_eval.py`: live dry-run replay (writes nothing).
- `services/orion-prediction/tests/test_bill_forecast.py`: range, DST, replay leakage, settlement, lifecycle, config parity.
- `services/orion-prediction/README.md`: section "Power-bill bet".
- `services/orion-sql-db/manual_migration_prediction_bill_forecast_v1.sql`: the table.
- `orion/schemas/self_monitor.py`: `answer_key`.
- `orion/self_monitor/record.py`: units, slow-source window.
- `orion/self_monitor/adapters.py`, `collect.py`, `catalog.py`, `names.py`: adapters, sources, catalog, names.
- `orion/self_monitor/tests/test_self_monitor_bill.py` (+ `test_cue.py`, `test_self_monitor_host.py`): effective-n, window, adapter, cue ranking.
- `.github/workflows/orion-prediction-tests.yml`: run on energy / self-monitor changes too.
- `docs/superpowers/handoffs/2026-11-04-handoff-B-self-monitor-followups.md`: Step 3 item marked built.

## Schema / bus / API changes

- Added: table `prediction_bill_forecast`; `GET /bill-forecast`; `SettledBetV1.answer_key: Optional[str]`.
- Removed: catalog entry `energy.bill` (replaced by `forecast.energy.bill_kwh` / `_usd`).
- Renamed: none.
- Behavior changed: the self-monitor collapses bets sharing an `answer_key` into one unit. Bets with no
  key reduce exactly as before (each bet is its own unit; all 139 existing self-monitor tests pass
  unchanged except two that list sources/names). For keyed faculties `n_graded` counts answers; `n_bets` keeps bets.
- Compatibility notes: `SettledBetV1` is built and reduced in-process (durable-runs host, report); it is
  not published on the bus, so the optional field needs no consumer-first migration. No bus channel
  added.

## Env/config changes

- Added keys (orion-prediction): `PREDICTION_BILL_FORECAST_ENABLED=true`,
  `PREDICTION_BILL_FORECAST_TICK_SEC=900`, `ENERGY_TARIFF_PATH`, `ENERGY_TIMEZONE`,
  `ENERGY_BILLING_CYCLE_START_DAY`, `ENERGY_USAGE_POINT_ID` (same values as orion-energy; a test checks).
- Removed / renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (primary checkout's
  `services/orion-prediction/.env` has the five non-empty keys).
- skipped keys requiring operator action: `ENERGY_USAGE_POINT_ID` (empty in the template, so the sync
  leaves it out; empty is the default and means "the only usage point").

## Metric quality gate

1. **Provenance.** Point: `project_period` (`orion/energy/reconcile.py:84`) over
   `energy_usage_interval` (RMP Green Button hours, orion-energy portal -> sql-writer). Range:
   `app/bill_forecast.py:kwh_range` from `spread()` over the same hours. Outcome: `energy_bill_actual`
   (RMP bill JSON, file drop). Settle basis: `reconcile._utility_pretax`.
2. **Independence.** Not independent of the stakes snapshot: `orion_projected_total_usd` IS this point
   estimate. This patch adds no new signal; it makes an existing one gradeable. The $ faculty is
   mostly a monotone transform of the kWh faculty (`Tariff.itemize`); it adds only tariff error. Kept
   because a tariff mistake should show up somewhere. vs GPU / chassis power (`body.gpu_power`,
   iLO/PDU watts): Orion's machines are about half the house's kWh, so they share a physical cause,
   but a different sensor (utility meter) and a different claim (a period total).
3. **Theory anchor.** Prediction interval for a future total from an estimated mean:
   var = s^2 (R + R^2/n), inflated by the AR(1) long-run variance factor (1+rho)/(1-rho).
4. **Live-data sanity** (replay over real rows, 2026-11-13; numbers below). Ranges are not degenerate:
   +-2,600 kWh on day 1, +-85 kWh on day 26. Rest point: a calibrated range is upheld 80 % of the
   time; the self-monitor checks that. A frozen-input artifact WAS found: the ledger has not moved since
   11-02 (Risks), so every bet in the current period repeats the 14.1-day projection.
5. **Existing mechanism.** `energy_reconcile` `forecast` kind (dormant; RMP's forecast, not Orion's);
   stakes snapshot (computes, never stores); catalog `energy.bill`. Reused `project_period`,
   `_utility_pretax`, `UsageLedger`, `Tariff.itemize`, the reachable loop pattern.
6. **Reversibility.** `PREDICTION_BILL_FORECAST_ENABLED=false` stops the loop; the table drops cleanly;
   `answer_key` is optional and only these adapters set it.

## Backfill result (replay, dry run, real data)

Only ONE completed bill period has meter data: 2026-09-18 -> 10-19 (3,106 kWh, $409.38 pre-tax). The
08-12 -> 09-18 period has none (the meter feed has a hole 08-07 -> 09-17). Replay feed lag measured at
69.7 h. 26 replay bets on the 09-18 bill:

| meter days in cycle | projected kWh | 80 % range | miss vs 3,106 | projected $ | miss vs $409.38 |
|---|---|---|---|---|---|
| 1.1 | 2,880 | 279 - 5,481 | -7.3 % | 380.20 | -$29 |
| 5.1 | 2,786 | 1,671 - 3,900 | -10.3 % | 367.74 | -$42 |
| 10.1 | 3,026 | 2,232 - 3,820 | -2.6 % | 399.42 | -$10 |
| 15.1 | 2,908 | 2,407 - 3,409 | -6.4 % | 383.89 | -$25 |
| 20.1 | 3,000 | 2,680 - 3,319 | -3.4 % | 395.95 | -$13 |
| 24.1 | 2,971 | 2,784 - 3,158 | -4.3 % | 392.20 | -$17 |
| 26.1 | 2,965 | 2,880 - 3,050 | -4.5 % | 391.42 | -$18 |

24 of 26 ranges held (kWh and $); the last two missed. Mean miss 195 kWh / $25. Most of the late miss
is the period length: Orion's schedule ended the cycle 10-18 (30 days), RMP read 10-19 (31 days); 2,965
x 31/30 = 3,064 (-1.4 %). The range does not model the read date moving.

Metered 30-day windows (truth = metered kWh, 16 windows inside the 55 continuous days, heavily
overlapping): 80 % ranges held 100 % of the time on days 2-13, 94 % on days 14-21, 75-88 % on days
22-29. Early ranges are too wide; late ranges are about right. This is a sanity check, not an
independent sample (about 2 independent periods).

Current period (10-19 ->): 20 replay bets, all frozen at 14.1 meter days (2,779 kWh, $349.79,
80 % 2,666 - 2,893) because of the DST hole.

## How the range was derived

Not from past bills' errors by days-into-cycle: one period cannot give a range. Instead from a model
of the house's own daily kWh: the projection = used so far + R days x the mean of n days seen. Its
error has variance s^2 x (1+rho)/(1-rho) x (R + R^2/n), with s and rho (lag-1 day-to-day persistence,
clamped 0..0.8) from the last 28 complete meter days before the bet. 80 % = +-1.2816 sd, floored at the
kWh already used; $ = those two kWh priced by `Tariff.itemize`. Fewer than 7 complete days: the bet is
stored with no range and is not graded. The bills will say whether it is calibrated.

## Effective sample size

`SettledBetV1.answer_key` = `energy_bill:<usage point>:<bill period start>`. In `record.py` every key
is one unit: its bets' upheld share, stated confidence and weight are averaged. `MIN_GRADED`, the
Wilson intervals (Kish over unit weights) and the calibration check count units. Brier skill is not
computed over shared answers. The two bill sources get their own window (400 days, half-life 180
days) so 10 bills can be in the window together. Status: `unknown` until 10 bills with live bets
have settled (~10 months, first live bill ~11-20).

## Tests run

```text
pytest orion/self_monitor/tests -q                                   -> 146 passed
pytest services/orion-prediction/tests/test_bill_forecast.py -q      -> 18 passed
pytest services/orion-prediction/tests -q                            -> 345 passed, 20 failed; main fails the
                                                                        same 20 in this venv (331 passed): pins
                                                                        differ from requirements.txt; CI is the gate
pytest services/orion-durable-runs/tests -k self_monitor -q          -> 14 passed, 1 skipped
pytest tests/test_sql_migration_drift_gate.py tests/test_metric_definition_drift.py
       tests/test_metric_lineage_gate.py tests/test_metric_liveness.py
       tests/test_deploy_migration_gate.py -q                        -> 282 passed
scripts/check_env_template_parity.py                                 -> PASS
orion.sql_migration_drift.auto_apply_refusals(migration, existing_tables=set()) -> []
```

## Evals run

```text
evals/run_bill_forecast_replay_eval.py (live DB, read-only, in the built image) -> numbers above
```

## Docker/build/smoke checks

```text
docker build -f services/orion-prediction/Dockerfile .   -> OK; /app/config/energy present
docker run ... python evals/run_bill_forecast_replay_eval.py -> replay inserted 47 (in memory), 58 gap days,
  live bet for 11-12 made, 26 settled
```
Live service: UNVERIFIED (not deployed).

## Review findings fixed

Review: code-review subagent on `git diff origin/main...HEAD`. No must-fix; all 3 should-fix and the
material nits fixed.

- Finding: the daily spread, replay lag and replay days mixed every meter (usage point) in the table.
  - Fix: `tick` keeps only the chosen meter's hours after `pick_usage_point`.
  - Evidence: `test_a_second_meter_never_leaks_into_the_spread_or_the_replay` fails with the filter removed, passes with it.
- Finding: one invalid energy row in 400 days would raise on every tick (no bets, no settlement).
  - Fix: rows validated one by one; bad ones skipped and counted in `last_pass.skipped_rows` + a log line.
  - Evidence: `test_a_bad_row_is_skipped_and_counted_not_fatal`.
- Finding: the stance/reverie cue ranked faculties by bet count, so 300 guesses on 10 bills would rank as "hundreds".
  - Fix: for bets sharing an `answer_key`, `FacultyRecordV1.n_graded` now counts answers (bills); `n_bets` keeps the raw count. No bus-schema change. Field descriptions updated.
  - Evidence: `test_cue_ranks_a_bill_faculty_by_its_answers_not_its_guesses` (tier 2, not 3).
- Finding (nit): day counts used plain datetime subtraction. Fix: `reconcile._duration_seconds`, same as `project_period`. Evidence: `test_days_into_cycle_are_real_hours_across_the_fall_back_change`.
- Finding (nit): identical days give a zero-width range. Fix: no range, `range_reason = zero daily spread`. Evidence: `test_identical_days_give_no_range_rather_than_a_zero_width_one`.
- Finding (nit): mean $ miss mixed pre-tax and with-tax bills. Fix: pre-tax bills only.
- Finding (nit): replay sees sql-writer's overwritten (corrected) values. Fix: stated in the module docstring (replays are never graded).
- Not changed: the per-bet streak (~30x long for one bill) only drives `broken`, and bill bets never fail to execute; commented in `record.py`.
- UNVERIFIED: that orion-prediction's DB role can SELECT `energy_usage_interval` / `energy_bill_actual`. The dry run read both tables with the same `postgres` URI the service uses.
- Found during my own testing: `daily_totals` measured a local day with same-tzinfo subtraction, so a 25-hour DST day read as 24 h and was dropped. Fixed (UTC bounds); `test_daily_totals_count_a_25_hour_dst_day_as_complete`.

## Restart required

Deploy from the clean primary `main` checkout, synced with `origin/main`, with no chat turn running:

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only
```

The post-pull rebuild applies `manual_migration_prediction_bill_forecast_v1.sql` (additive, header
`REQUIRED-BY: orion-prediction`) before rebuilding. Because `orion/schemas/self_monitor.py` changed,
its list (`rebuild_services_from_git_diff.sh --base origin/main --list-only`) is: orion-actions,
orion-cortex-exec, orion-cortex-orch, orion-dream, orion-durable-runs, orion-embodiment, orion-hub,
orion-mind, orion-prediction, orion-sql-writer, orion-thought. The hub and cortex rebuilds are why no
chat turn should be running. orion-prediction and orion-durable-runs (the self-monitor host) are the
two that matter. Check after: `curl -fsS http://localhost:8132/bill-forecast`.

## Risks / concerns

- Severity: high (pre-existing, not this patch).
  Concern: Orion's energy ledger is frozen since 2026-11-02 08:00 UTC. RMP's day files skipped the
  hour 08:00-09:00 UTC on 11-02 (the DST fall-back), the ledger treats a hole as "coverage stops here",
  so accrual, the stakes snapshot (`pressure=unknown, coverage_lag_hours=257`) and this bet all read
  14.1 days into the cycle. It will recur at every DST change.
  Mitigation: separate fix in orion-energy (refetch with overlap, or accept a one-hour hole at a DST
  edge). Until then, the current period's bets repeat one projection.
- Severity: medium. Concern: the range ignores the meter-read date moving (the 09-18 bill ran a day
  past schedule) and tariff error. Mitigation: graded; if bills miss late in the cycle, add read-date
  spread from the bills' own read offsets.
- Severity: medium. Concern: a status needs ~10 months. Follow-up (not built): a weekly kWh bet
  settled against the daily meter would give an answer every week.
- Severity: low. Concern: a re-issued bill after settlement does not re-settle. Mitigation: rare; the
  row keeps the bill period it settled on.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3280

🤖 Generated with [Claude Code](https://claude.com/claude-code)
