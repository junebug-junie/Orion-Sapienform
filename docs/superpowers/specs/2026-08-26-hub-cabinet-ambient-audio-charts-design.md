# Hub Cabinet ambient audio charts (multi-day) — design

Date: 2026-08-26  
Status: implemented on `feat/hub-cabinet-ambient-charts` — awaiting merge
Worktree: `/mnt/scripts/Orion-Sapienform-hub-cabinet-ambient-charts` on `feat/hub-cabinet-ambient-charts`

## Arsonist summary

Add multi-day ambient-audio trend charts to the existing Hub **Cabinet** tab. History comes from Postgres `orion_biometrics_summary` (already written every ~30s by biometrics → sql-writer with `cabinet_ambient_rms` / `cabinet_ambient_audio_activity`). Live chips come from the host `/run/orion-audio` snapshot. No new timeseries writer. No Chart.js. No Nano sparklines in this patch.

## Decisions locked

| Topic | Choice |
|---|---|
| Scope | Ambient audio only (Nano tiles already exist) |
| Retention | Days — query existing `orion_biometrics_summary` |
| Windows | `24h` (default), `3d`, `7d` |
| Grain | ~30s biometrics summary (not 1 Hz host reader) |
| Charts | RMS (hero) + activity 0–1 (secondary); SVG polylines |
| Live chips | Host `/run/orion-audio/latest.json` via Hub API |
| Downsample | Server-side cap ≈ 800 points |
| Index | Add `(node, timestamp)` on `orion_biometrics_summary` |
| Field / Hub pressures | Label Hub charts as biometrics-summary history; do not claim identity with live field lattice |

## Current architecture (grounded)

```text
CMTECK → orion-ambient-audio-reader (~1 Hz) → /run/orion-audio/latest.json
  → orion-biometrics (~30s) → BiometricsSummaryV1
       measurements.cabinet_ambient_rms / cabinet_ambient_peak
       pressures.cabinet_ambient_audio_activity
  → bus → orion-sql-writer → orion_biometrics_summary (JSONB)

Hub Cabinet tab today:
  GET /api/cabinet/sensors/latest → Nano tiles + Hub-local activity chips
  Poll ~1s only while #cabinet visible
  No ambient section; no history charts
```

Live evidence (2026-08-26): athena rows already contain `cabinet_ambient_rms` and `cabinet_ambient_audio_activity` in `orion_biometrics_summary`. Table has ~60k+ rows and **only a primary-key index** — multi-day filters need `(node, timestamp)`.

Hub already has `DATABASE_URL` / asyncpg patterns and SVG polyline helpers (`cocreation-signals.js`, `attention-organ.js`).

## End-to-end data flow (new)

```text
Cabinet tab visible
  → GET /api/cabinet/ambient/latest     (host snapshot → live chips)
  → GET /api/cabinet/ambient/history?window=24h|3d|7d
       SQL read orion_biometrics_summary for node=athena
       where measurements ? 'cabinet_ambient_rms'
       downsample → ≤ ~800 points
  → SVG RMS + activity charts
```

Hard rules:

1. No new ambient history producer or table in v1.
2. Pre-mic history has no ambient keys → empty series, never zero-filled.
3. Caption must state ~30s biometrics grain.
4. History/latest fetch only while tab visible (or on window toggle / Refresh).
5. Index ships in the same changeset as the history API.

## API

### `GET /api/cabinet/ambient/latest`

Mirror Nano cabinet snapshot style. Read `AMBIENT_AUDIO_PATH` (default `/run/orion-audio/latest.json`).

```json
{
  "ok": true,
  "age_sec": 0.4,
  "snapshot": {
    "status": "ok",
    "received_at": "2026-08-26T02:54:09Z",
    "device": "plughw:CARD=CMTECK,DEV=0",
    "rms": 5055.4,
    "peak": 19725,
    "window_sec": 0.5
  }
}
```

Missing / unreadable → `ok=false`, `snapshot=null`. Stale → `ok=false` with snapshot still returned for last-seen chips.

### `GET /api/cabinet/ambient/history?window=24h|3d|7d`

Default `window=24h`.

```json
{
  "ok": true,
  "node": "athena",
  "window": "24h",
  "grain_sec": 30,
  "points": [
    {"t": "2026-08-26T02:54:09Z", "rms": 7457.8, "peak": 19148, "activity": 0.30}
  ],
  "stats": {
    "n_raw": 2880,
    "n": 800,
    "rms_min": 1200.0,
    "rms_max": 9000.0,
    "activity_max": 0.85
  }
}
```

Query sketch (exact SQL at implementation):

- Filter `node = 'athena'`
- Time lower bound from window
- Require `measurements ? 'cabinet_ambient_rms'`
- Select `timestamp`, `measurements->>'cabinet_ambient_rms'`, `measurements->>'cabinet_ambient_peak'`, `pressures->>'cabinet_ambient_audio_activity'`
- Cast `timestamp` carefully (column is `varchar` storing timestamptz-ish strings)
- If row count > cap, bucket-average downsample

DB down / undefined table → `ok=false` + error string for UI.

Node id: fixed `athena` for v1 (cabinet is Athena-only), overridable via env if needed later.

## UI

Extend `#cabinet` panel (do not invent a new tab):

1. **Ambient audio** section (above or beside Nano grid)
   - Live chips: rms, peak, age, status
   - Window toggles: 24h / 3d / 7d
   - Chart: RMS vs time (hero)
   - Chart: activity vs time (0–1 secondary)
   - Caption: biometrics summary ~30s grain
2. Keep existing Nano sensor grid + pressure strip unchanged
3. Poll latest ~1s while tab visible; refetch history on window change / Refresh / activate (not every second)
4. Preserve last good charts on transient errors
5. SVG only — reuse Hub dark polyline style; no new chart library

## Config / Docker

| Key | Default | Notes |
|---|---|---|
| `AMBIENT_AUDIO_PATH` | `/run/orion-audio/latest.json` | Host snapshot for latest |
| `AMBIENT_AUDIO_STALE_AFTER_SEC` | `5` | Match biometrics |
| `CABINET_AMBIENT_HISTORY_NODE` | `athena` | Optional |
| `CABINET_AMBIENT_HISTORY_MAX_POINTS` | `800` | Downsample cap |
| `DATABASE_URL` | (existing) | History source |

Compose: bind `/run/orion-audio:/run/orion-audio:ro` on Hub if missing.

## Postgres index

Same changeset (writer boot ALTER and/or `scripts/sql/`):

```sql
CREATE INDEX CONCURRENTLY IF NOT EXISTS orion_biometrics_summary_node_ts_idx
  ON orion_biometrics_summary (node, timestamp);
```

(If `CONCURRENTLY` cannot run inside a transactioned boot path, use plain `CREATE INDEX IF NOT EXISTS` with a documented one-shot script — pick the path that matches existing sql-writer ALTER style.)

## Files likely to touch

- `docs/superpowers/specs/2026-08-26-hub-cabinet-ambient-audio-charts-design.md`
- `services/orion-hub/scripts/cabinet_ambient_routes.py` (new)
- `services/orion-hub/scripts/api_routes.py`
- `services/orion-hub/static/js/cabinet-sensors.js` (ambient section)
- `services/orion-hub/templates/index.html`
- `services/orion-hub/app/settings.py`, `.env_example`, `docker-compose.yml`
- `services/orion-hub/tests/test_cabinet_ambient_api.py`, panel wiring tests
- `services/orion-sql-writer/` and/or `scripts/sql/` for index
- Hub / biometrics README cross-links

## Tests

- Latest: missing / stale / fresh
- History: empty (no ambient keys), window parse, downsample under cap, stats
- Index migration/script present
- Panel: mounts, window toggles, script URLs; history not polled every 1s
- Absent ≠ zero for pre-ambient eras

## Acceptance checklist

1. Cabinet tab shows ambient live chips when `/run/orion-audio` is fresh.
2. 24h/3d/7d charts render from Postgres for athena ambient keys.
3. Pre-mic window shows empty chart state, not a flat zero line.
4. Response point count ≤ configured max.
5. Index exists; history query is acceptable on current table size.
6. Leaving the tab stops latest polling; history is not a 1 Hz DB hammer.
7. Nano cabinet UI unchanged.
8. Focused API + UI wiring tests pass; Hub restart listed after mount/env.

## Non-goals

- 1 Hz multi-day retention
- Dedicated ambient timeseries table
- Whisper / STT
- Nano channel sparklines
- Chart.js / heavy chart libs
- Claiming Hub activity series equals field-digester channel values
- Multi-node cabinet charts

## Risks

- **Index missing** → slow 7d scans — ship index with API
- **varchar timestamps** → fragile casts — fixture-cover parse path
- **Hub DB credentials** — reuse existing Hub `DATABASE_URL`; fail closed with explicit UI error
- **Cold ambient history** — only hours of ambient keys until days accumulate; UI must tolerate short series

## Recommended next patch

1. Merge `feat/hub-cabinet-ambient-charts` after operator smoke on Athena Hub (`#cabinet` → ambient chips + 24h/3d/7d charts).
2. Optional v2: Nano channel sparklines (explicitly out of scope here).

## Related

- Ambient levels design: `docs/superpowers/specs/2026-08-24-athena-ambient-audio-levels-design.md`
- Cabinet Nano Hub tab: `docs/superpowers/specs/2026-08-24-hub-cabinet-sensors-tab-design.md` (explicitly deferred historical charts; this patch adds them for ambient only)
