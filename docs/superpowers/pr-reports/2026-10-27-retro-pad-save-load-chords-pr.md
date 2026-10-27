## Summary

- Juniper asked for pad shortcuts: Select + L2 saves and Select + R2 loads the newest save. They do the same as F7 and F10, behind the same guards (save bar shown, no save or load in flight).
- Select is a real game button (in Zelda it pauses and opens the map), so the pad's Select is held back until release:
  - used in a chord, the game never sees it;
  - released alone, the game gets a 150 ms tap.
  Keyboard Shift-as-Select is unchanged.
- Each chord fires once per Select hold.
- The F7/F10 guard moved into `runHotkey`, shared by the keyboard and the pad. The help text, the runbook and the asset version (`?v=12`) are updated.

## Outcome moved

Save and load work from the 8BitDo Pro 2 without the keyboard, and a chord never pauses the game.

## Current architecture

Save and load ran from F7 and F10 or the buttons. Pad Select went straight to the game.

## Architecture touched

Hub front end only.

## Files changed

- `services/orion-hub/static/js/retro-panel.js`: `selectChord`, `runHotkey`, and the call from `mask()`.
- `services/orion-hub/templates/retro.html`: the help text and `?v=12`.
- `services/orion-hub/tests/retro/test_play_panel_static.py`: the version, and a Node-run test of the chord logic.
- `services/orion-hub/evals/retro_play_panel.mjs`: the version.
- `docs/runbooks/retro-browser.md`: the pad save and load section.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
cd services/orion-hub && pytest tests/retro -q -> 193 passed
node --check (Node 20 and 22)                  -> ok
Mutations: Select not held back -> fails; chord fires every frame -> fails
```

## Evals run

```text
The browser evals are unchanged apart from the asset version (CI). The pad chord is not exercised in the browser eval.
```

## Docker/build/smoke checks

```text
No build. The post-pull rebuild covers orion-hub.
```

## Review findings fixed

(filled in after review)

## Restart required

```bash
git pull   # rebuilds orion-hub; then reload /retro (v=12)
```

## Risks / concerns

- Severity: low
  - Concern: a pad Select tap now reaches the game on release instead of on press, so the pause or map opens about one press later.
  - Mitigation: needed so that chords never pause. Keyboard Shift is unchanged.
- Severity: low
  - Concern: the chord on the physical pad is UNVERIFIED.
  - Mitigation: Juniper tests it on the carbon-x1 after deploy; Retro's evidence shows the save and load events.

## PR link

(added after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
