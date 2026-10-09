## Summary

- The Hub biometrics card no longer blanks on every 10 s poll. New tiles are built off-screen and swapped in once all readings are back; the last view stays up meanwhile.
- A poll that overlaps one still in flight is skipped, so a slow older response can't overwrite a newer one.
- Card fetches abort after one poll interval, so a hung request shows as "partial"/unreachable instead of freezing stale data under "live".
- GPU subview: same swap; a response for a node no longer selected is dropped; switching node shows "Loading <node>…".

## Outcome moved

Card flicker gone (Juniper report, 2026-10-09). Polling cadence unchanged.

## Current architecture

`loadCardPreview` called `clear(grid)` and then awaited 6 fetches (snapshot + induction for 3 nodes), so the grid sat empty for the whole round trip every poll.

## Architecture touched

Hub frontend only: `services/orion-hub/static/js/biometrics-view.js`.

## Files changed

- `services/orion-hub/static/js/biometrics-view.js`: off-DOM build + `replaceChildren`, in-flight guard, `fetchJson` timeout, GPU node guard.
- `services/orion-hub/static/js/biometrics-view.test.js`: tests with a fake DOM.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
node --test services/orion-hub/static/js/biometrics-view.test.js   18 pass
  new: tiles stay during fetch then swap once; guard releases; hung fetch aborts -> "partial"
  mutation check: re-adding clear(grid) fails "card blanked while fetching"
pytest services/orion-hub/tests/test_biometrics_view_ui.py tests/test_biometrics_preview_api.py
  72 passed, 1 failed (test_app_js_deactivates_biometrics_view_when_leaving_the_hub_tab -- fails identically on main)
```

## Evals run

None: UI rendering fix, no behavior change in cognition.

## Docker/build/smoke checks

UNVERIFIED live until Hub is redeployed.

## Review findings fixed

- Finding: in-flight guard + no fetch timeout → one hung request freezes the card on stale data showing "live".
  - Fix: card fetches abort at `CARD_POLL_MS`; aborted reads render as unreachable/"partial".
  - Evidence: test "a hung card request is aborted".
- Finding: GPU subview could draw the previous node's cards under the newly active button.
  - Fix: `loadGpu` returns early if `node !== gpuNode`; `setGpuNode` shows "Loading <node>…".
- Finding: test didn't cover the guard releasing.
  - Fix: test asserts a new round starts after the first completes.

## Related finding (not in this PR)

Athena's power/mobo tiles show "—" about half the time because hecate's biometrics service runs with `NODE_NAME=athena` (its own `/health` reports `"node":"athena"`), so its 12-key summary overwrites athena's real 30-key one. Fix is config on hecate: `NODE_NAME=hecate` in `services/orion-biometrics/.env`, then restart.

## Restart required

```bash
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-hub up -d --build   # athena, primary checkout on main
```

## Risks / concerns

- Severity: low. Concern: on browsers without `replaceChildren`/`AbortController` falls back to clear+append / no timeout. Mitigation: both fallbacks are the old behavior.
