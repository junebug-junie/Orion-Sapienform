# Office air live-path smoke

## Summary

Adds one read-only operator smoke that proves the deployed PAN-1 office-air path instead
of treating a successful container restart as proof.

## Current architecture

```text
PAN-1 CCS811 + SHT40
  -> office.env.sample.v1
  -> orion-sql-writer
  -> office_env_sample
  -> Hub /api/office/env/latest + rendered Cabinet / Office panel
```

The existing service tests cover each component. Before this patch there was no single
post-deploy check for the materialized row, fallback debt, rendered DOM, and frontend
asset together.

## Change

- `scripts/smoke_office_air_live.py` checks:
  - Hub's latest sample is current and online;
  - eCO2/TVOC are present and inside the schema contract ranges;
  - the rendered Hub page contains the Office DOM and links `office-env.js`;
  - the served JavaScript contains the CCS811 Office bindings;
  - `office_env_sample` has all three air columns and a fresh, non-stale row;
  - no `office.env.sample.v1` rows entered `bus_fallback_log` in the bounded window.
- The script never reads or prints fallback payloads.
- `--allow-warming` is an explicit operator override for the CCS811's 20-minute
  conditioning state; it never substitutes zeros or claims a usable air reading.
- Hub's README documents the Athena command.

## Tests

```text
pytest -q tests/scripts/test_smoke_office_air_live.py  -> 12 passed
python3 -m py_compile scripts/smoke_office_air_live.py -> passed
git diff --check                                      -> passed
```

Coverage includes stale/offline Hub data, missing DOM/asset linkage, missing asset
bindings, schema drift, stale destination rows, fallback debt, warm-up behavior,
bounded fallback windows, and clean handling of HTTP connection failures.

## Live evidence

Run read-only against Athena on 2026-11-03 after the `main` deployment:

```text
OFFICE AIR LIVE SMOKE PASS
Hub:      age ~23 s, eCO2 1255 ppm, TVOC 130 ppb
Postgres: age ~23 s, eCO2 1255 ppm, TVOC 130 ppb
required columns: air_warming, office_eco2_ppm, office_tvoc_ppb
office fallback rows in 24 h: 0
```

## Review findings fixed

- HTTP transport failures originally escaped as raw tracebacks. They are now wrapped in
  an operator-readable `OFFICE AIR LIVE SMOKE FAIL` path with a regression test.
- The first version fetched the JavaScript directly but did not prove the rendered page
  linked it. The smoke now checks the Office DOM and script link before checking the
  asset itself.

## Env/config/schema

No env, configuration, bus, or schema changes. No `.env` sync is required.

## Deployment

No service rebuild is required. After merge, run from Athena's clean primary `main`:

```bash
cd /mnt/scripts/Orion-Sapienform
python3 scripts/smoke_office_air_live.py
```
