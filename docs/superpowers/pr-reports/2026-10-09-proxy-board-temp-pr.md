## Summary

- hecate's "mobo" tile on the Hub showed "—" even after PR #2566 restored its BMC readings.
- Cause: athena polls hecate's BMC, so hecate's board sensors never reach hecate's own summary. The proxy only passed watts along, and the hub tile only looked at the node's own summary.
- The aggregator now also passes along the motherboard temperature (`board_temp_c_max`) from proxied BMCs. The hub tile falls back to athena's per-node cluster data, the same way the wattage tile already does.
- The chipset/VR picker moved into a shared helper (`board_temp_c_max()`), so the node's own summary and the proxy pick the same sensors.

## Outcome moved

hecate's mobo tile shows a real number. Live input today gives 40°C (`PCH_Temp`; `CPU0_VR_Temp` and `CPU1_VR_Temp` also match).

## Architecture touched

- `orion/telemetry/biometrics_pipeline.py`: `board_temp_c_max()` helper, used by `extract_measurements`
- `services/orion-biometrics/app/main.py`: `_proxy_measurements` forwards `board_temp_c_max`
- `services/orion-hub/static/js/biometrics-view.js`: `boardTempFor(node, snapshot, athenaSnapshot)` falls back to `cluster_measurements_by_node`
- tests in `services/orion-biometrics/tests/test_ilo_proxy.py` and `services/orion-hub/static/js/biometrics-view.test.js`

## Schema / bus / API changes

- No schema change. `measurements_by_node` is `Dict[str, Dict[str, float]]`; a new float key fits.
- A node's own value wins. The merge fills only keys the node didn't report, so circe keeps its own reading.
- Fleet totals are unchanged: `board_temp_c_max` is in neither `FLEET_SUM_KEYS` nor `FLEET_MAX_KEYS`.

## Env/config changes

None.

## Tests run

```text
orion-biometrics tests: 168 passed, 2 failed (pre-existing: circe expected_offline x2, catalog says online)
tests/test_biometrics_measurements.py + hub preview API: 84 passed, 1 failed
  (pre-existing on main: test_review_disk_and_net_rates_are_not_promoted_to_physical_quantities)
node --test biometrics-view.test.js: 19 pass, 0 fail
live: hecate's real proxied thermal map -> board_temp_c_max = 40.0
```

## Evals run

No eval harness for the hub biometrics card. The live check is visual: the hecate mobo tile shows °C.

## Review findings fixed

The review subagent found no material issues. Non-blocking: hecate's 24-hour mobo history chart in the modal reads hecate's own summary rows, so it stays empty. This is the same limitation the proxied wattage chart already has.

## Restart required

From the primary checkout on main, after merge:

```bash
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-biometrics up -d --build && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low. hecate's memory voltage regulators (`CPU0_DIMMVR_Temp`) don't match the VR pattern. They currently read below the chipset, so the tile value is unaffected. Not patched by growing the name list.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
