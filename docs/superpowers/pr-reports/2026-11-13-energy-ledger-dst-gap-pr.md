## Summary

Fixes #3284. One missing meter hour at the 11-01 clock change stopped Orion's house-power numbers at 11-02. Cost tracking and every bill bet then used only about 14 days of data.

- **Root cause (ingest, from the source file):** RMP's 2026-11-01 file is a 25-hour day. It stamps two of its 25 readings with the same hour (10:00 UTC twice). Every later reading that day sits one hour early, so 2026-11-02 08:00 UTC is left empty. The reading for that hour is in the file, but it carries the wrong timestamp. All three refetches of that day (the portal already fetches 3 overlapping days) have the same bad stamps, so fetching again could not fix it.
- **Ingest fix:** the Green Button parser now checks each daily block. If a block has a doubled stamp and its own length matches its readings (25 readings = 90,000 s), the parser lays the readings end to end. Otherwise it keeps the file exactly as delivered and logs a warning.
- **Ledger fix:** a short hole no longer stops coverage. Each hole can be up to 3 h, with at most 6 h in total per window. A hole's kWh stays unknown and is never counted as zero. Metered sums leave the hole out. The run-rate projection fills it at the run rate, and the projection reports `bridged_hours`. A longer hole still stops coverage.
- A billed-total reconcile (`reconcile_actual`) and the replay eval's "truth" windows stay strictly gap-free (`bridge=False`).
- The bill bet records `bridged_gap_hours` in `range_inputs`. Bets already stored are not rewritten, because the insert is `ON CONFLICT DO NOTHING`.

## Outcome moved

- orion-energy rebuilds its ledger from the processed files at boot. On deploy, with the parser fix alone, its coverage goes from 2026-11-02 08:00 UTC to the latest delivered hour, with no gap. Live check against the real processed directory: `covered 2026-11-12 09:00:00+00:00 strict 2026-11-12 09:00:00+00:00 bridged_s 0.0`.
- orion-prediction reads Postgres, which still has the hole. Bridging carries it past the hole: covered_through moves to the last stored hour, and the hour is imputed at the run rate. The optional re-ingest (below) fills the hole with the real reading.

## Current architecture

`parse_espi` turned each `IntervalReading.timePeriod.start` into an interval and did no consistency check. `UsageLedger._contiguous_from_cycle_start` stopped at the first hole, and every coverage consumer used it: accrual, `cycle_kwh_before` (run cost), `window_prefix` → `project_period` (stakes, bill bet, replay), and `reconcile_actual`.

## Architecture touched

`orion/energy` (shared package used by orion-energy and orion-prediction). No bus, schema, or env changes.

## Files changed

- `orion/energy/espi.py`: `_restamp_fold_collision` and `_block_seconds`.
- `orion/energy/ledger.py`: hole bridging (`MAX_BRIDGED_GAP` 3 h, `MAX_BRIDGED_TOTAL` 6 h), `bridged_seconds`, and `window_prefix(bridge=...)`.
- `orion/energy/reconcile.py`: `project_period` imputes bridged hours at the run rate and adds `PeriodProjection.bridged_hours`. `reconcile_actual` is strict.
- `orion/energy/stakes.py`: docstring update. `cycle_accumulated_kwh` is metered kWh only, so it is a lower bound.
- `services/orion-prediction/app/bill_forecast.py`: `bridged_gap_hours` in `range_inputs`.
- `services/orion-prediction/evals/run_bill_forecast_replay_eval.py`: truth windows are strict.
- `services/orion-energy/README.md`: the hole and clock-change rules, and how to re-ingest a day.
- `services/orion-energy/app/pipeline.py`: `energy_cycle_bridged` warning when accrual steps over a hole.
- Tests: new `orion/energy/tests/test_energy_dst_gap.py`. Old hole tests now use a 4 h hole, because a 1 h hole is now bridged by design. New bill-bet test with the live shape.

## Schema / bus / API changes

- Added: none on the bus. `range_inputs.bridged_gap_hours` is a new key in an existing JSONB column.
- Removed / renamed: none.
- Behavior changed: a hole of 3 h or less no longer stops coverage, accrual, run cost, stakes, or projection.
- Compatibility notes: `PeriodProjection.bridged_hours` has a default value, so existing constructors still work.

## Env/config changes

- None. `.env_example` is unchanged, so no sync was needed.

## Tests run

```text
PYTHONPATH=. pytest -q orion/energy/tests                                    -> 204 passed
PYTHONPATH=.:services/orion-energy pytest -q services/orion-energy/tests services/orion-energy/evals -> 144 passed
cd services/orion-prediction && PYTHONPATH=../..:. pytest -q tests/test_bill_forecast.py -> 19 passed
cd services/orion-prediction && PYTHONPATH=../..:. pytest -q tests          -> 20 failed / 409 passed;
   the same 20 fail on main in the same local env (diff of failure lists is empty), so this branch adds no failure
Regression check: the new #3284 tests run against main's espi/ledger/reconcile/bill_forecast:
   5 of 7 in test_energy_dst_gap.py fail and the bill-bet live-shape test fails; all pass on the branch.
```

## Evals run

- `services/orion-energy/evals`: passes as part of the service test run.
- `run_bill_forecast_replay_eval.py`: needs live Postgres and was not run. Its truth windows now use strict coverage.

## Docker/build/smoke checks

- No image or compose changes. Live-data check: I rebuilt the ledger from `/mnt/storage-warm/orion-energy/processed` with this branch. Result: 10-19 cycle covered to 2026-11-12 09:00 UTC, no bridged hours, projected 2,686 kWh / $338.05 pre-tax.

## Review findings fixed

- Finding: the parser fix never reaches Postgres. Prediction and the Hub read SQL, where 11-02 08:00 UTC is still empty and the later 11-01 hours sit one hour early.
  - Fix: I checked the claim that a re-dropped file would not publish. It was wrong: `UsageLedger.upsert` refuses a row only when the stored row is strictly newer, and a re-drop carries a fresh `retrieved_at`. I added `test_redropped_fall_back_file_republishes_the_restamped_hours`, which proves all 49 hours publish, including 08:00. The re-ingest command is under Restart required. It writes to production, so it is left for Juniper. Snapshot: `/tmp/energy-dst-reingest-3284/before.csv` (29 rows).
  - Evidence: test passes. The SQL upsert keeps the row whose `retrieved_at` is newer or equal.
- Finding: the restamp guard was looser than the RMP shape. A doubled stamp plus a hole elsewhere would be re-laid on a guess.
  - Fix: restamp only when there is exactly one zero step and every other step equals the previous reading's length.
  - Evidence: `test_doubled_stamp_plus_a_hole_elsewhere_is_left_as_delivered`.
- Finding: bridged holes became silent in orion-energy logs (`energy_cycle_incomplete` no longer fires).
  - Fix: `_accrue_cycle` logs `energy_cycle_bridged ... bridged_hours=`.
- Finding: a short hole at the cycle start is now bridged with no test.
  - Fix: kept on purpose, because a dropped first hour would freeze a whole cycle the same way. Pinned by `test_a_short_hole_at_the_cycle_start_is_bridged_on_purpose`.
- Finding: metered stakes and accrual numbers carry no bridged-hole flag. Deferred to #3302 (schema change).

## Restart required

Deploy from the clean primary `main` after merge (the post-pull rebuild covers both):

```bash
scripts/safe_docker_build.sh orion-energy up -d --build
scripts/safe_docker_build.sh orion-prediction up -d --build
```

Re-ingest of 2026-11-01 into Postgres, for Juniper to run after the orion-energy deploy. It republishes the day's 25 hours plus the next day's and fills 11-02 08:00 UTC. The snapshot of the current rows is at `/tmp/energy-dst-reingest-3284/before.csv`:

```bash
cp /mnt/storage-warm/orion-energy/processed/20261105T054654Z__rmp-portal-20261105T054610Z-2026-11-01.xml /mnt/storage-warm/orion-energy/inbox/rmp-portal-reingest-3284-2026-11-01.xml
```

Post-deploy check:

```bash
curl -fsS http://localhost:8132/bill-forecast   # covered_through past 2026-11-02, days_into_cycle > 14.1
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select as_of, pressure, pressure_reason, covered_through from energy_stakes_snapshot order by as_of desc limit 1"   # no coverage_lag_hours > ~72
```

## Risks / concerns

- Severity: low. Concern: the restamp trusts reading order inside a block. Mitigation: it runs only when a block has a doubled stamp, sorted stamps, and the block length equals the reading total. Any other case is kept as delivered and logged.
- Severity: low. Concern: bridged hours are imputed at the period's run rate. A hole at a peak hour is under-filled, and a hole at a quiet hour is over-filled. Mitigation: the error is at most 6 h out of about 720 h, and `bridged_hours` / `bridged_gap_hours` make the imputation visible.
- Severity: low (follow-up #3302). Concern: the metered stakes and accrual numbers are a lower bound when a hole is bridged, and they carry no flag. Mitigation: the `energy_cycle_bridged` log line.
- Severity: medium (follow-up #3301). Concern: RMP blocks start at Pacific midnight and readings look hour-ending, so Orion's hour stamps may be 1-2 h late. Totals are not affected. Mitigation: filed as #3301, UNVERIFIED.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3303
