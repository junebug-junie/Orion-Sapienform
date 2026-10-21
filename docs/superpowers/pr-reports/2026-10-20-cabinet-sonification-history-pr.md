# Cabinet sonification: historical playback

## Summary

- Extend PR #2799's instrument with mutually exclusive live/history modes and
  24h/3d/7d playback in 60 seconds using the audio clock.
- Add Pause, paused Seek, Replay, Stop/reset, source time, compression and aggregation
  disclosure. Reuse existing history requests; no competing poller or server change.
- Interpolate independently by timestamp only within an explicit gap policy;
  unavailable ranges stay silent. Omit aggregated vibration and preserve calibration
  gates. Keep physical/audio scaling fixed throughout the selected window.

## Outcome moved

An operator can listen to recorded magnetic-field levels with inspectable source
time and missingness, without treating downsampled vibration means as impacts.
Real 24h history has 94–188-second spacing, so the default 90-second gap policy
correctly leaves it silent until the operator explicitly allows wider interpolation.

## Current architecture

Parent PR #2799 provides the browser instrument, source adapter, fixed mappings,
separate audio context, Cabinet poll integration and local traces. Hub's existing
`/api/cabinet/sensors/history` returns per-series `{t,v}` points, 30-second grain
metadata, per-series raw counts and min/max. Its point budget downsamples each series.
The Cabinet subview already fetches this API for its charts on activation/window change.

## Architecture touched

`orion-hub` browser only: new pure history adapter; instrument mode/clock lifecycle;
existing history-fetch forwarding; template controls; tests/evals/docs. No new service,
producer, cognition loop, environment variable or persistent storage.

## Files changed

- `static/js/cabinet-sonification-history.js`: validation, immutable snapshots,
  spacing metadata, time compression, independent timestamp interpolation.
- `static/js/cabinet-sonification-history.test.js`: seven deterministic history tests.
- `static/js/cabinet-sonification.js`: history clock, lifecycle/controls/trace metadata;
  avoid repeatedly extending unchanged gain ramps on the 25-ms history scheduler.
- `static/js/cabinet-sensors.js`: feed the existing history request lifecycle and expose
  window selection through the same chart fetch path.
- `templates/index.html`: mode/window/gap selectors, timeline, pause/replay/seek and asset.
- `tests/test_cabinet_sensors_panel.py`: rendered controls and asset dependency order.
- `evals/cabinet_sonification_browser.cjs`: native-browser history interaction checks.
- `evals/fixtures/cabinet-sonification-history-live.json`: sanitized real 24h response.
- `README.md`: playback/gap semantics, limitations and verification.

Paths above are relative to `services/orion-hub`. This report is under
`docs/superpowers/pr-reports/`. Existing static CI automatically discovers the new
Node test file; the parent's source-provenance parity gate remains intact.

## Schema / bus / API changes

- Added/removed/renamed: no server, shared schema or bus changes.
- Browser-local frames add mode/source time/playback time/compression/gap policy and
  actual interpolation brackets; downloaded traces include history metadata.
- Compatibility: existing history API and chart controls remain the fetch owners.
  Sensor live polling updates its cache while history plays but cannot replace its
  frame; live network failure does not invalidate already downloaded historical data.

## Env/config changes

- Added/removed/renamed keys: none.
- `.env_example` updated: no; sync script not needed because templates did not change.
- Root/Hub ignored `.env` copies from primary used solely for the required build wrapper.
- Skipped keys requiring operator action: none. No Redis client or URL was introduced.

## Six-part display-signal gate

1. Provenance: existing `query_sensor_history_rows()` selects stored
   `measurements->>'cabinet_magnetic_ut'` from `orion_biometrics_summary`;
   `rows_to_sensor_series()` and `_downsample_series()` produce the existing HTTP
   series. Original physical producer/extraction is documented in parent PR #2799.
   The new browser `prepare()`/`frame()` consume only those series and timestamp
   brackets. No new independent sensor measurement is produced.
2. Independence: these are aggregated/downsampled versions of the same existing
   magnetic/LiDAR measurements, not new evidence. Audio pitch is a deterministic
   transform. IMU-derived vibration is deliberately omitted from historical audio.
3. Theory: parameter-mapping sonification maps data levels into pitch, not affect,
   safety, physical causes or reconstructed transients. Temporal interpolation is
   an explicit display choice limited by the operator-selected maximum gap.
4. Live sanity: read the running Hub's 24h history on 2026-10-20. Captured 800 points
   per series / 2758 raw rows. Returned timestamps run 2026-10-19T18:17:31.853380Z to
   2026-10-20T18:15:20.801161Z. Magnetic raw min/max 21.96–53.36 µT, within existing
   fixed 20–60 µT bounds. Actual gaps 93.7188–187.6418 seconds (median 93.8577).
   The 90-second default cannot bridge these points; the browser eval verifies
   silence then sound only after explicitly choosing 300 seconds. Captured LiDAR
   88–101 mm and vibration 0.03828–0.04416 g do not establish controlled movement
   calibration. Their parent approval gates remain; vibration is omitted regardless.
   Missingness remains missing; a steady field maps to steady pitch, not an anomaly
   expected to decay to zero. Human listening usefulness is **UNVERIFIED**.
5. Existing mechanism: reuse chart window selection and fetch, API aggregation,
   existing pure mapping and audio graph. No duplicated backend/reducer or new poller.
6. Reversibility: remove history adapter/UI hooks and restore the parent's instrument.
   No Orion state, schema, metric channel or migration needs undoing.

## Tests run

```text
node --test services/orion-hub/static/js/cabinet-sonification-history.test.js services/orion-hub/static/js/cabinet-sonification.test.js services/orion-hub/static/js/cabinet-sensors.test.js
19 passed.
python -m pytest services/orion-hub/tests/test_cabinet_sensors_panel.py services/orion-hub/tests/test_cabinet_sonification_provenance.py services/orion-hub/tests/test_cabinet_sensors_api.py tests/test_cabinet_sensors.py -q
72 passed; 22 existing dependency warnings.
git diff --check: passed.
```

## Evals run

```text
python services/orion-hub/evals/cabinet_sonification_server.py --port 18098
PUPPETEER_MODULE=<primary Hub>/node_modules/puppeteer node services/orion-hub/evals/cabinet_sonification_browser.cjs http://127.0.0.1:18098
PASS: default history gaps silent while clock runs, explicit wider gap audibility,
pause freezes cursor, seek pauses, replay, Stop resets, 3d/7d window fetch reuse,
history hide/reopen, failed window clears history, mode switch disarms.
All parent live controls/audio/replay checks passed too. Native offline mix finite,
peak 0.1097215935587883; 120 original live frames replayed.
```

The eval uses the actual rendered Cabinet controls and native Web Audio in an isolated
loopback server. The captured 24h response drives the historical tests; 3d/7d use
partial fixture windows. It does not access production writes or claim human hearing.

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-hub build
PASS: Hub app and retro gateway images built; no deployment.
```

Running Hub history GET verified; isolated worktree template/static/browser path
verified. Deployed historical data-to-audio path is **UNVERIFIED**; no merge or deployment.

## Review findings fixed

- Finding: ending a window at the latest supported sensor observation would hide a
  trailing outage and shift the requested window backwards.
  - Fix: freeze the browser request-start time as the approximate query endpoint;
    disclose the approximation in UI/trace/docs. Never derive window bounds from
    sensor timestamps. Reject points outside the selected window.
  - Evidence: regression with a six-hour trailing outage and an independently fresh
    second series; both produce the same 24h bounds and preserve trailing silence.
- Follow-up review using the required requesting-code-review skill/subagent found no
  remaining material issues and independently passed all 13 sonification Node tests
  plus diff checks. Human hearing and deployed runtime are excluded explicitly.

## Restart required

After parent and this patch are merged and a separate deployment is authorized,
from clean primary main synchronized with origin/main:

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Medium: human headphone/blinded listening acceptance remains outstanding. Existing
  LiDAR/vibration calibration gates are unchanged; no additional voice enabled.
- Low: existing API omits query bounds; browser request-start approximates them.
  Clock skew can shift the estimated window; UI/trace disclose the approximation.
  Observation timestamps always remain real stored source timestamps.
- Low: wide interpolation policies can smooth long unobserved intervals. They require
  explicit operator selection; actual spacing and policy remain visible.
- Scope: magnetic historical levels only audible by default; optional light/climate/
  particulate/ambient voices remain later gated patches. No production deployment.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2833 (follow-up to #2799)
