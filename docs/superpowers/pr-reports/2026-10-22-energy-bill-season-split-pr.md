## Summary

- The second real Rocky Mountain Power bill (billing date 2026-10-20, Sep 18 - Oct 19, 31 days, 3,106 kWh, $427.39) showed three tariff rules Orion did not know. Tariff r3 adds them and reproduces **both** real bills to the cent.
  - A bill that spans the Oct 1 season change splits its kWh and its blocks by days in each season (12 summer + 19 winter days).
  - A 27-33 day bill is a plain month (no 31/30 stretch of the 400 kWh block or the $12 charge).
  - Schedule 92 changed rate mid-period (1.17% -> 1.78% on 2026-09-25); riders can now carry dated `rates:`.
- Projection (stakes, forecast reconcile) and actual-bill reconcile now price the period exactly like the bill (`Tariff.itemize`), not as a sum of hourly approximations.
- Billing cycles follow real meter-read dates from actual bills: a read within 10 days of `ENERGY_BILLING_CYCLE_START_DAY` replaces it. The current cycle becomes Oct 19 -> Nov 18 (was Oct 18 -> Nov 18, which dropped the bill's last day into the next cycle).
- Hub's projection tile now says "(pre-tax)": the number never included the $18 sales tax, which is part of why "$400" looked far from "$427".

## Outcome moved

On the real bill's own inputs (3,106 kWh, Sep 18 -> Oct 19):

| | total |
|---|---|
| RMP bill | $427.39 |
| r2 tariff | $405.81 (all priced as winter, prorated 31/30) |
| r3 tariff | $427.39, every line exact |

Live reconcile against Orion's own metered hours (3,079 kWh): Orion $405.83 vs RMP $409.38 pre-tax. The -$3.55 is the 27 kWh between midnight boundaries and RMP's mid-day meter read; every per-line bucket delta is now $0.00 except energy (-$3.13, that same kWh gap).

## Current architecture

orion-energy priced each metered hour by its own month, prorated every period by days/30, and used one flat rate per rider. Cycles started on a fixed calendar day. Reconcile and projection summed the hourly prices.

## Architecture touched

orion-energy only (plus the Hub label). No bus, schema, or SQL change. `config/energy/` is baked into the image, so orion-energy needs a rebuild.

## Files changed

- `config/energy/tariff.rmp_ut_sch1.2026-08-10.r3.yaml`: new tariff version (normal_days, dated Schedule 92, sources, known gaps).
- `orion/energy/tariff.py`: `proration.normal_days`, `Rider.rates`/`rate_on`, `itemize(kwh, start=, end=)` with season split by days and per-rate rider lines; whole-kWh split clamped so near-zero usage cannot overdraw.
- `orion/energy/reconcile.py`: projection and actual reconcile priced by `itemize`.
- `orion/energy/ledger.py`: `set_read_dates`; cycle starts snap to meter reads; moved cycles marked dirty.
- `services/orion-energy/app/pipeline.py`: bills feed read dates to the ledger and re-accrue moved cycles; a re-issued bill with the same start replaces the older one.
- `services/orion-energy/app/settings.py`, `.env_example`: r3 path; cycle day 18 (the account's read day).
- `services/orion-hub/templates/index.html`: "Orion projects (pre-tax)".
- `orion/energy/testing.py`: flat test tariff carries its fixed charge as a line item (itemize reads line items).
- Tests/evals/README.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: `energy.reconcile.v1` and `energy.stakes.snapshot.v1` totals are bill-exact (cents-rounded per line); `energy.cost.accrued.v1` `cycle_start` follows meter reads once an actual bill is known. Accrual rows are keyed by tariff_version, so r3 rows sit beside r2 rows.
- `Tariff.itemize` signature: `(kwh, *, month, period_days)` -> `(kwh, *, start, end)`; line names are `energy_<season>_block_<n>`. Only tests called it.

## Env/config changes

- Changed: `ENERGY_TARIFF_PATH` -> r3; `ENERGY_BILLING_CYCLE_START_DAY` example/default 1 -> 18.
- `.env_example` updated: yes.
- local `.env` synced: the sync script reports changed values without overwriting; `ENERGY_TARIFF_PATH` was updated by hand (its meaning changed), `ENERGY_BILLING_CYCLE_START_DAY` was already 18. `ENERGY_PORTAL_ENABLED=true` locally is an intentional override and was left alone.
- skipped keys requiring operator action: none.

## Tests run

```text
PYTHONPATH=. python -m pytest -q orion/energy/tests                                   168 passed
PYTHONPATH=.:services/orion-energy python -m pytest -q services/orion-energy/tests services/orion-energy/evals   143 passed
python scripts/check_env_template_parity.py                                            PASS
```

## Evals run

```text
services/orion-energy/evals/test_energy_real_bill_eval.py: both real bills replayed hour by hour reconcile within cents (bill 2 < $0.01)
Live check (scratch script, real energy_usage_interval rows): r3 reconcile delta -$3.55 (= 27 kWh read-time gap), current cycle Oct 19 -> Nov 18, projected $402.70 pre-tax
```

## Docker/build/smoke checks

```text
Not run: deploy happens from main after merge.
```

## Review findings fixed

- Finding: near-zero kWh across a season change rounded the first season up past the total -> TariffError, aborting a bill batch.
  - Fix: clamp to the remaining kWh. Evidence: `test_r3_near_zero_kwh_across_a_season_change_does_not_overdraw`.
- Finding: a re-issued bill (same start, corrected end) left its stale end date acting as a meter read.
  - Fix: the newer bill replaces the older one; the ledger read set is replaced, not unioned. Evidence: `test_reissued_bill_replaces_the_older_read`.
- Finding: `add_read_dates` left moved cycles' cached costs serving old prices until re-accrued.
  - Fix: moved cycles are marked dirty. Evidence: ledger tests.
- Finding: bills replayed at boot move cycles in memory but publish nothing.
  - Fix: documented (boot never publishes, by design); drop bills through the inbox. The 2026-10-20 bill is not in `processed/` yet, so it will publish when dropped.

## Restart required

```bash
# from the clean primary main checkout after merge
scripts/safe_docker_build.sh orion-energy up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
# then drop the 2026-10-20 bill (write .part, then rename)
```

## Risks / concerns

- Severity: low. Concern: Schedule 92's 1.78% from 2026-09-25 is read off the bill; the approved revised sheet was not found. Mitigation: known_gaps entry; the next bill confirms or corrects it.
- Severity: low. Concern: normal_days [27, 33] is fitted to two bills. Mitigation: known_gaps entry.
- Severity: low. Concern: hourly accrual and run cost still price each hour by its own month and use the latest rider rate. Mitigation: they serve the next-kWh price; the bill-facing numbers use itemize.

## PR link

(this PR)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
