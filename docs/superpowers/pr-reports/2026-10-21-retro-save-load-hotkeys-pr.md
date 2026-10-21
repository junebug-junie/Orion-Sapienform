## Summary

- F7 saves the game, the same as pressing Save.
- F10 loads the newest save into the game that is running, even when an older save is selected.
- F10 never ends one game to start another. With no matching game running or paused, it refuses and says to use the Load button.
- Both keys work anywhere on the retro page, including fullscreen, except while typing in a text field. With Shift, Ctrl, Alt or Meta held, the keys keep their normal browser meaning (Shift+F10 opens the context menu).
- In fullscreen, save and load messages now show in a small toast at the top of the screen. Before, the status line was outside the fullscreen area, so nothing was visible.

## Outcome moved

Juniper asked for F7 save and F10 load. Quick save and quick load now work from the keyboard without leaving fullscreen, and the result is visible.

## Current architecture

Save and Load were buttons only (PR #2825). Status messages went to `#status`, which sits outside the fullscreen element `#stage`.

## Architecture touched

Hub retro page only: `services/orion-hub/static/js/retro-panel.js` and `services/orion-hub/templates/retro.html`. No retro service, API, schema, bus or env change.

## Files changed

- `services/orion-hub/static/js/retro-panel.js`: F7/F10 handler; Load action takes an in-place-only path for F10; `report()` mirrors into the fullscreen toast.
- `services/orion-hub/templates/retro.html`: `#toast` inside `#stage` with fullscreen-only CSS; help text; cache-bust `retro-panel.js?v=7`.
- `services/orion-hub/evals/retro_browser.mjs`: F7 in fullscreen (new save, toast visible, still fullscreen); F10 loads newest with an older save selected; F10 refuses with no running game and the game stays stopped.
- `services/orion-hub/tests/retro/test_boundary.py`: static check that the hotkeys are wired.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: browser keyboard only.
- Compatibility notes: none.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
pytest services/orion-hub/tests/retro -q          -> 5 passed
node --check retro-panel.js retro_browser.mjs     -> ok
```

## Evals run

```text
services/orion-hub/evals/retro_browser.py --image orion-retro-retro                     -> PASS
services/orion-hub/evals/retro_browser.py --image orion-retro-retro --ack-delay-ms 350  -> PASS
Mutation: hotkey click disabled                                                          -> FAIL at the F7 step (TimeoutError, retro_browser.mjs:78)
```

Real Chromium through the real Hub retro router against the deployed retro image. The eval checks server state (`loaded_save`, session state, save list), not only page text.

## Docker/build/smoke checks

```text
No image build needed for retro. The hub image needs a rebuild to serve the new JS.
```

## Review findings fixed

- Finding (SHOULD): the hotkeys were skipped for every `<input>`, including the Mute checkbox, which also let the browser defaults run.
  - Fix: skip only text-like inputs, textareas and contenteditable.
  - Evidence: `typing()` in retro-panel.js.
- Finding (SHOULD): in fullscreen, F7/F10 results were invisible because `#status` is outside `#stage`.
  - Fix: `#toast` inside `#stage`, shown only in fullscreen, fed by `report()`, cleared after 3 s.
  - Evidence: the eval asserts the toast text, its display and that it is inside the fullscreen element.
- Finding (SHOULD): F10 could end the running game and start another when the game dropdown showed a different game.
  - Fix: F10 loads in place only and refuses otherwise.
  - Evidence: the eval presses F10 with the game stopped and asserts the refusal message and that the session stays stopped.
- Finding (NIT): F10 changed the dropdown even when a load was already in flight.
  - Fix: nothing runs while Save or Load is busy.
- Finding (NIT): F10 right after F7 could load the previous newest save.
  - Fix: same busy check covers Save.
- Finding (NIT): Shift+F10 (context menu) was swallowed.
  - Fix: keys with a modifier are left to the browser.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && scripts/safe_docker_build.sh orion-hub up -d --build
```

The hub restart pauses a running game; it does not end it (the game lives in orion-retro).

## Risks / concerns

- Severity: low
- Concern: F7/F10 are swallowed on the retro page, so any browser feature on those keys (caret browsing, menu bar) is unavailable there.
- Mitigation: only on the retro page and only without modifiers.

## PR link

<link>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
