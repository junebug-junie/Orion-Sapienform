## Summary

- New **Tribute64 layout** checkbox in the Retro panel (under More). With it on, a Retro-Bit Tribute64's four yellow buttons work as N64 C buttons, and its B button works as N64 B.
- The choice is kept per browser. The controller status line says when it is on.
- It only changes N64 games. NES and SNES are unchanged.

## Outcome moved

Juniper could not strafe in GoldenEye with the yellow C buttons, only with the D-pad. The controller's own layouts (Default, Legacy, All Star) changed nothing. Cause: the Tribute64's USB receiver sends its C buttons as plain Xbox buttons (Y, X, left-stick click, right-stick click). The panel only read N64 C from a right stick. So C-down pressed N64 B, C-left pressed an ignored button, C-right and C-up did nothing, and the B button did nothing.

Measured from the pad's raw input on carbon-x1 (`/dev/input/js0`, receiver reports as `045e:028e` "Microsoft X-Box 360 pad"): C-left = button 3, C-right = 9 (L3), C-up = 10 (R3), C-down = 2, A = 0, B = 1, Z = left trigger. In the browser's standard mapping these become buttons 3, 10, 11, 2, 0, 1, 6.

## Current architecture

`retro-panel.js` builds a libretro button mask (`padMask`) and, for N64, four axes (`sticks`). Mupen64Plus-Next (pinned 6752836) reads N64 A from libretro B, N64 B from libretro Y, Z from L2, and the C buttons from the right analog stick (`emulate_game_controller_via_libretro.c`).

## Architecture touched

Hub Retro page only (static JS + template). No service, bus, schema, or env change.

## Files changed

- `services/orion-hub/static/js/retro-panel.js`: `TRIBUTE_C` button-to-C-stick table, `tributeMask()`, the stick override in `sticks()`, the checkbox wiring, and the status line.
- `services/orion-hub/templates/retro.html`: the checkbox, a help-text line, and the cache-buster bumped to v=22.
- `services/orion-hub/tests/retro/test_play_panel_static.py`: a new test that runs the shipped functions under node; one older probe now declares the setting.
- `services/orion-hub/tests/retro/test_chat_box_static.py`: cache-buster v=22.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: N64 pad input, only when the Tribute64 layout is on.
- Compatibility notes: it's opt-in, because an 8BitDo Pro 2 in X mode names itself the same Xbox 360 pad.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
pytest services/orion-hub/tests/retro/ -q   -> 274 passed
node --check services/orion-hub/static/js/retro-panel.js -> ok
```

## Evals run

```text
None. This is a static input mapping; the new node test pins each measured button to its N64 output.
```

## Docker/build/smoke checks

```text
Not run. Static file change; served by orion-hub after rebuild.
Live check after deploy: Retro page -> More -> Tribute64 layout, GoldenEye: yellow buttons strafe and look, B reloads. UNVERIFIED until Juniper plays.
```

## Review findings fixed

- Finding: no correctness bugs (subagent review). Minor: Z mapping not documented as measured; nothing tells the player the layout is on when they use a different pad.
  - Fix: comment notes Z = left trigger (measured); the controller status line shows "(Tribute64 layout on for N64)".
  - Evidence: 274 retro tests pass.

## Restart required

```bash
scripts/rebuild_services_from_git_diff.sh   # after merge, from clean main (rebuilds orion-hub)
```

## Risks / concerns

- Severity: low
- Concern: with the layout left on, a different pad in N64 gets C-down on its left face button.
- Mitigation: opt-in, and the status line shows when it is on.
- Issues opened: #3810 (on standard pads, the N64 right face button presses nothing, while the help text says it is A).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3812
