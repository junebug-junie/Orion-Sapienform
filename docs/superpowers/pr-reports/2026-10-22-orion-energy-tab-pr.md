## Summary

- New Hub tab **Orion Energy** shows what Orion's machines (athena, circe, hecate) and its cabinet cooling (the Shelly-plugged AC) draw. Bars are stacked per source, and the Rocky Mountain Power (RMP) house meter is drawn on the same kWh axis.
- Six grains: hour, day, week, month, quarter, year. Hovering a period shows each source's kWh and how much of the period was measured, the house kWh and how many RMP hours have arrived, Orion's share, and the dollars.
- A second chart shows Orion's share of the house. There is also a table view, and a notes block listing what is not measured.
- sql-writer now keeps **kWh per hour per source** in a new table, `orion_energy_hourly`. The raw per-machine watts are deleted after 30 days. Without this table, the month, quarter and year views could never fill.
- New API: `GET /api/energy/orion?grain=…&periods=…`.

## Outcome moved

Juniper can now see Orion's electricity against the whole house. Live numbers over the last ~29 days:

- **Orion with its cooling is about 50% of the house.** Machines alone are ~33% and the AC ~17%.
- **Orion costs about $47–50 a week before tax,** out of the house's $83–93.

Before this, the Hub showed only the house. Per-machine watts were deleted after 30 days, so no long view of Orion's own usage was possible.

## Current architecture

- Per-machine wall watts: `orion_biometrics_cluster.measurements_by_node[*].chassis_watts`, every ~15 s. athena reads iLO. circe and hecate read the PDU, through athena's proxy. Kept for 30 days.
- AC watts: `home_cooling_sample.cooling_watts`, every ~5 s (orion-zwave → sql-writer).
- House: `energy_usage_interval`. RMP hourly data in whole kWh, ~24 h late.
- Pricing: `energy_cost_accrued`, one row per interval per tariff version (orion-energy → sql-writer).
- Hub: an Energy strip in Biometrics → Cabinet, house only.

## Architecture touched

- **orion-sql-writer:** a new rollup loop (`app/orion_energy_rollup.py`) and a new table (`orion_energy_hourly`, created by `create_all`). The loop integrates watts into kWh straight from Postgres. No bus contract.
- **orion-hub:** a new API route, a new iframe tab page, and tab wiring in `app.js` / `index.html`.
- **CI:** a must-not-skip Postgres step for the rollup, run in its own database.

## Files changed

- `services/orion-sql-writer/app/orion_energy_rollup.py`: the rollup SQL and its loop.
- `services/orion-sql-writer/app/models/orion_energy_hourly.py`, `app/models/__init__.py`: the table.
- `services/orion-sql-writer/app/settings.py`, `.env_example`: `ORION_ENERGY_ROLLUP_INTERVAL_SEC`.
- `services/orion-sql-writer/app/main.py`: starts and stops the loop.
- `services/orion-sql-writer/tests/test_orion_energy_rollup_pg.py`: 6 tests against real Postgres.
- `services/orion-sql-writer/README.md`: the rollup section.
- `services/orion-hub/scripts/energy_routes.py`: `/api/energy/orion` and the pure `aggregate_orion_energy`.
- `services/orion-hub/static/orion-energy.html`, `static/js/orion-energy.js`: the tab page and its charts.
- `services/orion-hub/static/js/app.js`, `templates/index.html`: the tab button, section and hash route.
- `services/orion-hub/tests/test_orion_energy_routes.py`, `test_orion_energy_hub_tab.py`, `static/js/orion-energy.test.js`: tests.
- `services/orion-hub/README.md`: the tab section.
- `.github/workflows/orion-sql-writer-tests.yml`: the rollup Postgres step.

## Schema / bus / API changes

- **Added:**
  - Postgres table `orion_energy_hourly (hour_start, source)`, with `kwh`, `covered_sec`, `sample_count` and `updated_at`.
  - HTTP `GET /api/energy/orion`.
- **Removed / Renamed:** none.
- **Behavior changed:** none to existing routes or tables.
- **Compatibility notes:**
  - The table is new and created at sql-writer boot.
  - The Hub returns `ok:false` until the table exists. It never returns zeros.
  - No bus channel or registry change.

## Env/config changes

- **Added keys:** `ORION_ENERGY_ROLLUP_INTERVAL_SEC=600` (orion-sql-writer). 0 disables the rollup.
- **`.env_example` updated:** yes.
- **Local `.env` synced:** `python scripts/sync_local_env_from_example.py` ran in the worktree. The worktree has no sql-writer `.env`, so the main checkout's `.env` picks up the key on the post-merge sync. The settings default is also 600, so nothing breaks before that.
- **Skipped keys:** none.

## Tests run

```text
ORION_TEST_PG_URI=… pytest services/orion-sql-writer/tests/test_orion_energy_rollup_pg.py      6 passed
  mutation check: removing the clock-skew clamp makes the new test fail (1 failed, 5 passed)
pytest services/orion-hub/tests/test_orion_energy_routes.py test_orion_energy_hub_tab.py test_energy_routes.py   34 passed
node --test services/orion-hub/static/js/orion-energy.test.js                                 8 pass
pytest services/orion-sql-writer/tests (full)   14 failed, 1035 passed. The same 14 fail on a clean
  origin/main detached worktree (notify ack/escalate and subscription-matches-consumers tests, which
  depend on this host's environment). This branch adds none.
static gates from orion-static-gates.yml: all rc=0. check_migration_required_by --base origin/main OK.
check_env_template_parity PASS.
```

## Evals run

```text
Live-data replay (read-only): the rollup SELECT ran over 40 days of live conjourney data, and the
output went through aggregate_orion_energy.
  week of Oct 12: Orion 397 kWh / house 700 kWh = 56.7%, Orion $49.77 / house $87.82
  coverage: athena 99.1%, circe 99.7%, cooling 99.5%, hecate 99.8% per measured hour
  sanity: in 0 of 685 hours did Orion's meters read more than the house meter
Full 30-day backfill query: 3.1 s (statement timeout 30 s).
There is no eval harness for the Hub energy tab. This replay is the eval. Follow-up: turn it into
a periodic eval that fails if Orion > house in any hour.
```

## Docker/build/smoke checks

```text
docker build -f services/orion-sql-writer/Dockerfile .   OK. The new modules import inside the image.
  (safe_docker_build.sh build exits 0 without building in a worktree that has no root .env.)
UI: rendered with headless Chromium against a stand-in API serving live-derived data, in the
  hour, day, week and month views, with the hover tooltip. Screenshots checked for label collisions,
  clipping and the tooltip text.
Live deploy: UNVERIFIED. It needs the merge plus a rebuild of sql-writer and hub from main.
```

## Review findings fixed

- **Finding:** The share was biased low when a source's feed dropped mid-hour.
  - **Fix:** An hour joins the share only when every reporting source is at least 90% measured. The tooltip and API report `partial_hours`.
  - **Evidence:** `test_partly_measured_hour_is_left_out_of_the_share_and_counted`. On live data, 13 of 685 hours are affected.
- **Finding:** The Orion and house tiles covered different hours, because of RMP's lag.
  - **Fix:** Both tiles show the share's hours. The Orion tile notes its total including hours RMP hasn't sent.
  - **Evidence:** the JS test asserts `orion_kwh_overlap`.
- **Finding:** A failed load left the previous grain's charts on screen.
  - **Fix:** `clearView` blanks the charts, tiles and table.
  - **Evidence:** `a failed load blanks the previous numbers` (node test).
- **Finding:** A future-stamped sample could freeze the rollup window.
  - **Fix:** The window is clamped to `date_trunc('hour', now())`.
  - **Evidence:** `test_future_stamped_sample_does_not_freeze_the_rollup`, mutation-checked.
- **Finding:** A priced bucket with 0 kWh showed its cost as unknown.
  - **Fix:** The check now counts priced hours.
  - **Evidence:** `test_priced_hours_with_zero_kwh_cost_zero_not_unknown`.
- **Finding:** The hover highlight stuck on the other chart, and the stale-response guard checked the grain instead of the request.
  - **Fix:** Leaving a chart clears both. Each request gets a sequence number.
- **Finding:** The CI test narrowed shared tables.
  - **Fix:** The step creates and uses its own `energy_rollup_test` database.
- **Finding (own render check):** An empty month's tooltip blamed RMP.
  - **Fix:** It now says "no Orion data".
  - **Evidence:** node test.
- **Not changed:**
  - The iframe reloads when the tab is re-entered. Every iframe tab does this.
  - Hour-edge attribution is under 0.5%. It is documented in the module docstring.

## Restart required

Run from the clean primary main checkout after merge (the post-pull rebuild does this):

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build && scripts/safe_docker_build.sh orion-hub up -d --build
```

Then check that the rollup ran: `docker logs orion-athena-sql-writer 2>&1 | grep ORION_ENERGY_ROLLUP`.

## Risks / concerns

- **Severity: low.**
  - **Concern:** Per-machine history starts 2026-09-22, and cooling starts 2026-09-26. Hecate's power reading starts in early October. Older periods show no Orion data.
  - **Mitigation:** The empty periods are shown as unknown, never 0, and the notes block says so.
- **Severity: low.**
  - **Concern:** RMP reports whole kWh per hour, so the hourly share is rough.
  - **Mitigation:** The hour view shows a warning. Over a day the rounding mostly cancels out (about ±1.5%).
- **Severity: low.**
  - **Concern:** prometheus, the network switch and UPS losses are not measured, so the real share is somewhat higher.
  - **Mitigation:** The tab says so.
- **Severity: low.**
  - **Concern:** Rows replayed from a bus backlog more than 6 h late are not rolled up.
  - **Mitigation:** This is documented. Clearing the table re-backfills from the 30 days of raw data.

## PR link

(filled in after creation)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
