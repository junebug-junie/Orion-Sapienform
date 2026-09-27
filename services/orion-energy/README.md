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
| `orion:energy:bill:actual` | `energy.bill.actual.v1` | A closed RMP bill (dropped JSON or portal) |
| `orion:energy:bill:forecast` | `energy.bill.forecast.v1` | RMP's in-cycle bill estimate |
| `orion:energy:reconcile` | `energy.reconcile.v1` | Orion's estimate minus RMP's bill or forecast |
| `orion:energy:stakes:snapshot` | `energy.stakes.snapshot.v1` | Cycle-to-date cost and pressure vs RMP forecast |
| `orion:energy:importer:status` | `energy.importer.status.v1` | Whether house usage is actually arriving |

It consumes `orion:power:intent:settled`.

Two run costs are published and never merged: `estimated_run_cost_usd` (Orion's
own meter, incremental over baseline, at the marginal tariff rate — the only
number autonomy may read) and `house_share_cost_usd` (share of the whole-house
interval; context only). A null cost always has a `*_gap` reason. Null is
unknown, not free. All costs are pre-tax (`cost_basis: pre_tax`).

## Feeding it (Plan 1: file drop)

1. On rockymountainpower.net: *Energy usage → Green Button → Download my data* (XML).
2. Copy the file into `${ENERGY_HOST_DATA_DIR}/inbox/`. Only `*.xml` files are scanned — copy large exports under a temporary name (e.g. `.xml.part`) and rename into `inbox/` when the copy finishes.
3. Within `ENERGY_SCAN_INTERVAL_SEC` the file moves to `processed/` (or `inbox/failed/` if unparseable).

The first drop must cover the **current billing cycle from day one** (`ENERGY_BILLING_CYCLE_START_DAY`, your meter-read day in `ENERGY_TIMEZONE`). Mid-cycle exports publish usage but accrual and run-cost stay unknown until a contiguous prefix from cycle start exists. Look for `energy_cycle_incomplete` in logs when coverage is partial.

Re-dropping an overlapping export is safe when the new file is **newer**: `retrieved_at` is the drop time, so dropping an **old** export after a newer one overwrites fresher data. Pending run-cost re-pricing state is in-memory and lost on restart. Bus publishing after a file moves to `processed/` is best-effort — re-drop the export to republish.

Set `ENERGY_BILLING_CYCLE_START_DAY` to your bill's meter-read day.

## Bills (file drop)

Drop one JSON file per bill into `${ENERGY_HOST_DATA_DIR}/bills/inbox/`. Lines the bill
does not show are simply omitted (unknown, not $0):

```json
{"kind": "energy.bill.actual.v1", "billing_period_start": "2026-08-12",
 "billing_period_end": "2026-09-11", "kwh_billed": 712, "energy_charge": 85.10,
 "customer_charge": 12.00, "taxes": 4.10, "current_charges": 101.23}
```

Only `*.json` files are scanned, so write the file as `name.json.part` and rename it to
`name.json` once it is complete — a half-written file would otherwise be read, fail to
parse, and land in `bills/inbox/failed/`. A file that fails to parse is kept there under a
`<timestamp>__<name>` prefix, so repeated failures never overwrite each other.

RMP's in-cycle estimate uses `"kind": "energy.bill.forecast.v1"` with `billing_period_start`,
`as_of`, and `projected_total_usd` and/or `projected_kwh`. The period is
`[billing_period_start, billing_period_end)` at local midnight. Each bill publishes a
`energy.reconcile.v1` row (Orion minus RMP); late usage re-reconciles automatically.

## Status and stakes

Every `ENERGY_STATUS_INTERVAL_SEC` the service publishes `energy.importer.status.v1`
(`healthy` / `stale` / `reauth_required` / `degraded`) and `energy.stakes.snapshot.v1`
(cycle-to-date cost, next-kWh price, projected total vs RMP forecast). Pressure is
`unknown` whenever the importer isn't healthy or a forecast is missing.

## Debug queries

```sql
-- Cycle to date
SELECT cycle_start, max(cycle_accumulated_kwh) kwh, max(cycle_to_date_total_usd) usd
FROM energy_cost_accrued GROUP BY cycle_start ORDER BY cycle_start DESC LIMIT 3;

-- Recent run costs (null = unknown; read the gap column)
SELECT intent_id, workload_kind, energy_kwh, energy_basis, estimated_run_cost_usd,
       run_cost_gap, house_share_cost_usd, house_share_gap
FROM energy_run_cost ORDER BY window_start DESC LIMIT 20;

SELECT reconcile_kind, billing_period_start, orion_total_usd, utility_total_usd, utility_basis, delta_usd, reconcile_gap
FROM energy_reconcile ORDER BY computed_at DESC LIMIT 5;
SELECT as_of, state, reason, usage_lag_hours FROM energy_importer_status ORDER BY as_of DESC LIMIT 3;
SELECT as_of, pressure, pressure_reason, cycle_to_date_total_usd, orion_projected_total_usd, forecast_total_usd
FROM energy_stakes_snapshot ORDER BY as_of DESC LIMIT 3;
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
