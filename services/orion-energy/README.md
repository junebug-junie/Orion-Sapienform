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

## Tariff

`ENERGY_TARIFF_PATH` defaults to `config/energy/tariff.rmp_ut_sch1.2026-08-10.r2.yaml`,
which reproduces the first real bill (billing date 2026-09-21, 37 days, 3,772 kWh,
$554.99) line for line. What that bill showed, and the old file got wrong:

- The 400 kWh first block and the monthly charges prorate by days / 30 (493 kWh,
  $14.80 and $0.20 for 37 days). Every price uses the length of its own period: the
  bill's dates for reconcile, the `ENERGY_BILLING_CYCLE_START_DAY` cycle for accrual.
- Each rider has its own base: Schedule 92 is on energy + customer charge; efficiency and
  EV riders are on energy + EBA + renewable adjustment.
- Utah sales tax (4.40%) skips the lifeline charge. Tax is only in `Tariff.itemize`, a
  bill-shaped estimate; accruals, stakes, and reconcile stay pre-tax.

The old file's total missed that bill by only ~$0.67 because a ~$2.50 energy overcharge
and a $2.64 fixed-charge undercharge (flat $12.16) cancel; its per-bucket deltas show both.
Evidence: `orion/energy/tests/test_energy_tariff.py` (line-exact) and
`evals/test_energy_real_bill_eval.py` (hour-by-hour ledger + reconcile, within cents).

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

| Key | Default | What it does |
|---|---|---|
| `ENERGY_STALE_AFTER_HOURS` | `48` | Newest metered interval older than this makes the importer `stale` (usage is then unknown, never $0). RMP data normally lags ~24h. |
| `ENERGY_STAKES_NEAR_RATIO` | `1.0` | Orion's projected cycle total / RMP's forecast total at or above this reads `near_forecast`. |
| `ENERGY_STAKES_OVER_RATIO` | `1.10` | Same ratio at or above this reads `over_forecast`. Hub curiosity holds on either (only when `ORION_ENERGY_STAKES_ENABLED=true` on Hub). |

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

## Portal (optional, compose profile `portal`)

`orion-energy-portal` logs into rockymountainpower.net once a day, downloads Green Button
XML into the usage inbox, and (optionally) scrapes bills into the bill inbox.

RMP keeps its login only for the life of the browser (checked live 2026-09-28: after
closing the browser, the saved profile held only analytics cookies and reopened on the
sign-in page). A saved session alone therefore dies every day, so the fetcher logs in
itself from a **credentials file** on the host, kept in an owner-only dir outside
`ENERGY_HOST_DATA_DIR` so the main `orion-energy` container never sees it:

```bash
install -d -m 700 ~/.orion/secrets ~/.orion/secrets/rmp
install -m 600 /dev/null ~/.orion/secrets/rmp/credentials.env
nano ~/.orion/secrets/rmp/credentials.env   # RMP_USERNAME=... / RMP_PASSWORD=...
```

Only the portal service bind-mounts that dir, read-only, at `/run/secrets/rmp`
(`ENERGY_PORTAL_CREDENTIALS_HOST_DIR` picks it). The dir must exist or compose refuses to
start the portal; the file inside it is optional (no file = manual reauth mode below).
Mounting the dir rather than the file means an edit is picked up at the next attempt
even when the editor replaces the file, with no restart. One `KEY=VALUE` per line; an
optional leading `export ` and one pair of surrounding quotes are stripped, nothing else --
a password with leading/trailing spaces must be quoted.

- Read at every attempt, never from the environment, never logged. A file that group or
  other can read is refused (`error`/`credentials_file_too_open`) before a browser starts;
  a file missing either key is `credentials_incomplete`, an unreadable one
  `credentials_unreadable:<error>`.
- One login submit per attempt, never a retry. If it does not get past the sign-in page
  (wrong password, MFA prompt, captcha) the status is `reauth_required`/`login_failed` and
  the loop waits a full interval, so a bad password cannot lock the account.
- If the sign-in form itself breaks (fields not found), the status is
  `error`/`login_form_failed:<error>` -- a selector problem, not a password problem.
- No file: a login redirect is `reauth_required`/`session_expired` (manual reauth below).

Usage comes one day at a time: RMP's Green Button download follows the usage page's
period dropdown, and only "One Day" is hourly (One Week/Month are daily, Two Year monthly).
Each attempt reads the date picker's allowed range and downloads the newest
`ENERGY_PORTAL_BACKFILL_DAYS` days as `rmp-portal-<stamp>-<day>.xml`. A download that is not
all one-hour readings is refused (`non_hourly_download`) -- a daily reading would overwrite
that day's first hour in the ledger; a file whose readings belong to a different day is
refused as `wrong_day_download`. A refused file is kept in the raw dir and the run moves on
to the next day, ending `error`/`usage_days_bad:<bad>/<total>:<first bad day>`. Three bad
days in a row mean the page itself is broken, so the run stops there
(`usage_days_bad:<bad>/<total>:stopped:...`) instead of spending a long backfill on it. A lost
session or browser error stops the run (`usage_day_failed:<day>:...`); days already
downloaded stay delivered. Each day gets one page reload and retry first. RMP's day files run 02:00-02:00 local, not midnight-midnight (seen live, not
explained). Don't hand-drop One Week/One Month exports for the same reason.

Login and the usage download are verified live (2026-09-28); billing selectors are still
UNVERIFIED (`portal/selectors.py`). Bill scraping is off by default
(`ENERGY_PORTAL_SCRAPE_BILLS=false`), so a good run reads `ok`/`fetched_usage_only`.

| Key | Default | What it does |
|---|---|---|
| `ENERGY_PORTAL_CREDENTIALS_HOST_DIR` | `/home/athena/.orion/secrets/rmp` | Host dir holding the login file; compose-only, mounted read-only into the portal container. |
| `ENERGY_PORTAL_CREDENTIALS_PATH` | `/run/secrets/rmp/credentials.env` | Where the portal reads that file inside the container (see above); absent = manual reauth only. |
| `ENERGY_PORTAL_SCRAPE_BILLS` | `false` | Also scrape billing history / forecast after the usage download. |
| `ENERGY_PORTAL_TIMEOUT_SEC` | `300` | Base cap on one fetch attempt, plus 45s per requested day; hitting it records `error`/`timeout` in `status.json`. |
| `ENERGY_PORTAL_RAW_DIR` | `/data/energy/portal/raw` | Where a failed download/scrape keeps its raw artifact (see below). |
| `ENERGY_PORTAL_BACKFILL_DAYS` | `3` | Days of usage each daily fetch requests (1-730); `--days` overrides it for a one-off backfill. |

**Stop the running portal service before a headed reauth or a `run --rm ... --once`.**
Both use the same persistent Chromium profile, and two browsers on one profile can
corrupt the saved session:

```bash
scripts/safe_docker_build.sh orion-energy --profile portal stop orion-energy-portal
```

Bring it back with the `up -d` line in step 2 once the reauth or one-off fetch is done.

1. Create the credentials file above. (Without one: a one-time login on a host with a
   display, same profile dir the container mounts, portal service stopped. This only
   helps while RMP keeps that session alive.)
   ```bash
   pip install playwright==1.49.0 pydantic-settings==2.7.1 && python -m playwright install chromium
   cd services/orion-energy && PYTHONPATH=../..:. python -m portal.reauth \
     --profile /mnt/storage-warm/orion-energy/portal/profile \
     --status /mnt/storage-warm/orion-energy/portal/status.json
   ```
2. Backfill in small chunks (portal service stopped), then the daily loop. Live 2026-09-28,
   RMP ended the session after ~8 day-downloads (and sooner after many logins in one
   half hour), so fetch older days a few at a time with `--through`, hours apart:
   ```bash
   scripts/safe_docker_build.sh orion-energy --profile portal run --rm orion-energy-portal python -m portal.main --once --days 6 --through 2026-09-23
   scripts/safe_docker_build.sh orion-energy --profile portal up -d --build orion-energy-portal
   ```
3. Set `ENERGY_PORTAL_ENABLED=true` for `orion-energy` and restart it.

When the importer reads `reauth_required`: with a credentials file, `login_failed` means
the password or RMP's sign-in flow changed -- fix the file (or the login selectors), then
run the `--once` fetch. Without one, stop the portal service, repeat the headed login in
step 1, then the
`--once` fetch from step 2 (the loop otherwise waits a full `ENERGY_PORTAL_INTERVAL_HOURS`
after any recorded attempt, including across container restarts). Reauth clears
`reauth_required` but does not count as a successful fetch.

The profile dir holds live session cookies and is forced to mode `0700`. UNVERIFIED:
the container runs as root, so after it has used the profile, files in it may be
root-owned and a host-user reauth can fail with permission errors. Fix ownership first:
`sudo chown -R "$(id -u):$(id -g)" /mnt/storage-warm/orion-energy/portal/profile`.

Failed downloads/scrapes keep the raw artifact in `${ENERGY_HOST_DATA_DIR}/portal/raw/`
(dir `0700`, files `0600`). Inline `<script>` bodies and hidden-input values are
stripped before writing, but visible page text may still contain account details —
local disk only. Unchanged bills are not re-sent: `bills_seen.json` next to
`status.json` remembers a content hash per billing period; delete it to force a resend.

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
