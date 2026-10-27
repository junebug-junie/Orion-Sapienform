## Summary

- Juniper will play through the Hub from the carbon-x1 with an 8BitDo Pro 2, plus the keyboard.
- Fixed the face buttons. The standard gamepad mapping is positional, and the page sent the bottom button as NES A (the Xbox layout). Now the right face button is A and the bottom one is B, as on an NES pad and as labelled on the 8BitDo.
- The page picks this player's connected standard pad (Player 2 takes the second one), else the first. It skips empty slots, which browsers report as `null`, and non-standard pads. Before, it took `pads[port] || pads[0]`, which could be an empty slot.
- New controller status line (`#padInfo`):
  - with a standard pad: it names the controller;
  - with a pad in another mapping: it says to set the 8BitDo mode switch to X;
  - with no pad: it says none was found.
- In fullscreen, pad input works without clicking the game first. The help text, the runbook and the asset version (v=11) are updated.

## Outcome moved

The pad's right face button sends NES A (256) and the bottom one sends NES B (1). Before, they were swapped.

## Current architecture

`mask()` combines held keys with `navigator.getGamepads()` every 50 ms (the `send` interval), only while the game canvas has focus.

## Architecture touched

Hub front end only (`retro-panel.js`, `retro.html`), plus tests, the eval and the runbook.

## Files changed

- `services/orion-hub/static/js/retro-panel.js`: `gamepad()`, `padMask()`, `padStatus()`, and the connect/disconnect status.
- `services/orion-hub/templates/retro.html`: the controller help, `#padInfo`, and `?v=11`.
- `services/orion-hub/tests/retro/test_play_panel_static.py`: the asset version, and a Node-run test of the real mapping functions.
- `services/orion-hub/evals/retro_play_panel.mjs`: the asset version.
- `services/orion-hub/evals/retro_browser.mjs`: the synthetic pad uses index 1 for A, behind an empty slot.
- `docs/runbooks/retro-browser.md`: the controller layout and the 8BitDo mode.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
cd services/orion-hub && pytest tests/retro -q  -> 192 passed
node --check static/js/retro-panel.js           -> ok
Mutation: old A/B mapping -> new mapping test fails
```

## Evals run

```text
evals/retro_browser.mjs (CI service-container) asserts right+A (mask 384) through the authority's applied journal
```

## Docker/build/smoke checks

```text
No build. The post-pull rebuild covers orion-hub.
```

## Review findings fixed

- Finding (low): Player 2 in a second window now picked Player 1's pad.
  - Fix: the page takes the port's standard pad, else the first one.
- Finding (nit): a pad that was disconnecting could show the "not standard mode" message.
  - Fix: pads that are not connected are filtered out.
- Finding (info): over Bluetooth, X mode may read as non-standard in Firefox.
  - Fix: the status line and the runbook say to use a USB cable then.
- Checked, no change needed: `stage` has no temporal-dead-zone risk, `#padInfo` exists before load, and the P2 eval still passes.

## Restart required

```bash
git pull   # rebuilds orion-hub; then reload /retro (v=11)
```

## Risks / concerns

- Severity: medium
  - Concern: no physical 8BitDo Pro 2 has been tested (UNVERIFIED). The Linux browser on the carbon-x1 must report it in standard mapping.
  - Mitigation: the status line says so when it does not. X-input mode is the standard-mapping mode on Linux Chrome and Firefox.
- Severity: low
  - Concern: anyone used to the old swapped layout will notice the change.
  - Mitigation: none needed. The new layout matches the NES.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2935

🤖 Generated with [Claude Code](https://claude.com/claude-code)
