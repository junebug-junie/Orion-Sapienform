# Cabinet sonification: history playback was silent on every real window

## Summary

- Juniper reported that history playback in Hub did not work. Live check: it was silent at the default
  setting on 24h, 3d and 7d.
- Cause: the history API thins each window to ~800 points, so points are ~94 s (24h),
  ~313 s (3d) and ~750 s (7d) apart. The fixed 90-second default gap rule bridged none of them.
  PR #2833 saw this and called it intended. It was not.
- Gap policy is now a multiple of each series' own median spacing: 1.5×, **2× (default)**, 4×, 10×.
  Normal jitter plays. A real outage stays silent.
- Review fix: a series with fewer than 3 gaps has no known rhythm, so it stays silent
  instead of gliding across its single gap.

## Outcome moved

Live hub data, real page, headless browser, 2026-10-21:

```text
before (main):   24h@90s silent · 3d@90s/300s silent · 7d@90s/300s silent
after (branch):  24h/3d/7d × {1.5, 2, 4} all audible (gain 0.12, status ok)
```

## Current architecture

PR #2833's browser-only history adapter (`cabinet-sonification-history.js`) interpolated between
points only when they were within a fixed number of source seconds (90/300/900/1800/3600).

## Architecture touched

`orion-hub` browser only. No server, schema, bus, env or storage change.

## Files changed

- `static/js/cabinet-sonification-history.js`: `GAP_MULTIPLES`, `gapSecFor()`, a per-reading `gapSec`
  in frames, and median spacing only with ≥3 gaps.
- `static/js/cabinet-sonification.js`: `historyGapMultiple: 2`, `gapMultiple` in setGap, the trace and
  the info line (the info line shows the resulting seconds).
- `templates/index.html`: gap selector options are now multiples.
- `static/js/cabinet-sonification-history.test.js`: relative-outage test, sparse-series test, and a
  real captured 24h fixture that must be audible over >95% of playback at the default.
- `evals/cabinet_sonification_browser.cjs`: now asserts the default is audible on real history
  (it used to assert silence).
- `README.md`: gap-policy semantics with live spacing numbers.

## Schema / bus / API changes

None. Browser-local frame field `gapSec` is replaced by `gapMultiple`. Each reading carries its own
resolved `gapSec`. Trace `history.gapSec` → `history.gapMultiple`.

## Env/config changes

None. `.env_example` unchanged, so no sync was needed.

## Tests run

```text
node --test services/orion-hub/static/js/*.test.js                 340 pass, 0 fail
pytest test_cabinet_sensors_panel / _sonification_provenance / _sensors_api, tests/test_cabinet_sensors.py   72 passed
git diff --check                                                   clean
```

## Evals run

```text
cabinet_sonification_browser.cjs (isolated server, native Web Audio)   passed: all 21 checks
live probe: worktree page + real /api/cabinet history from running hub   all windows audible (table above)
```

## Docker/build/smoke checks

No image change. Hub bind-mounts `static/` and `templates/` from the primary checkout, so the change
goes live after merge and `git pull` on main, plus a browser reload.

## Review findings fixed

- Finding: a two-point series always bridged its whole span (one gap = its own median), so a
  22 h outage would play as a glide.
  - Fix: `medianSpacingSec` is null below 3 gaps, so the series is silent.
  - Evidence: new test `a series too sparse to show its normal spacing never bridges its one gap`.
- Not fixed (low): a sensor that flaps across more than half the window inflates the median ~2×. A single
  contiguous outage cannot do this (the reviewer probed a 7d half-window outage: still silent).

## Restart required

```text
No restart required. After merge: git pull on primary main, then reload the Hub page.
```

## Risks / concerns

- Low: fragmented outages over >50% of a window can widen bridging (see above). Shown honestly in the
  info line's resulting seconds.
- Human listening usefulness remains UNVERIFIED.

## PR link

Added after push.
