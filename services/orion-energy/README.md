# orion-energy

Gives Orion the household electricity bill as a stake. Whole-house Green Button
usage from Rocky Mountain Power is priced with a versioned tariff, and every
settled GPU power intent gets a dollar estimate.

Spec: `docs/superpowers/specs/2026-09-26-orion-energy-watcher-design.md`

## What it publishes

| Channel | Kind | Meaning |
|---|---|---|
| `orion:energy:usage:observed` | `energy.usage.observed.v1` | Metered house kWh per interval |
| `orion:energy:cost:accrued` | `energy.cost.accrued.v1` | Interval priced at its billing-cycle block position |
| `orion:energy:run_cost:estimated` | `energy.run_cost.estimated.v1` | Cost of one settled power intent |

It consumes `orion:power:intent:settled`.

Two run costs are published and never merged: `estimated_run_cost_usd` (Orion's
own meter, incremental over baseline, at the marginal tariff rate — the only
number autonomy may read) and `house_share_cost_usd` (share of the whole-house
interval; context only). A null cost always has a `*_gap` reason. Null is
unknown, not free. All costs are pre-tax (`cost_basis: pre_tax`).

## Feeding it (Plan 1: file drop)

1. On rockymountainpower.net: *Energy usage → Green Button → Download my data* (XML).
2. Copy the file into `${ENERGY_HOST_DATA_DIR}/inbox/`.
3. Within `ENERGY_SCAN_INTERVAL_SEC` the file moves to `processed/` (or `inbox/failed/` if unparseable).

Re-dropping an overlapping export is safe: the newer retrieval wins per interval.
Set `ENERGY_BILLING_CYCLE_START_DAY` to your bill's meter-read day.

## Debug queries

```sql
-- Cycle to date
SELECT cycle_start, max(cycle_accumulated_kwh) kwh, max(cycle_to_date_total_usd) usd
FROM energy_cost_accrued GROUP BY cycle_start ORDER BY cycle_start DESC LIMIT 3;

-- Recent run costs (null = unknown; read the gap column)
SELECT intent_id, workload_kind, energy_kwh, energy_basis, estimated_run_cost_usd,
       run_cost_gap, house_share_cost_usd, house_share_gap
FROM energy_run_cost ORDER BY window_start DESC LIMIT 20;
```

## Run

```bash
scripts/safe_docker_build.sh orion-energy up -d --build
docker logs --tail=100 orion-athena-orion-energy
```

## Tests

```bash
PYTHONPATH=. python -m pytest orion/energy/tests -q
cd services/orion-energy && PYTHONPATH=../..:. python -m pytest tests evals -q
```
