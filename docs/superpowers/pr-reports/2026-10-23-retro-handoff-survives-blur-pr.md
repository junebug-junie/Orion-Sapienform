## Summary

- Clicking into another window ended Orion's co-play turn within about a second. This fixes that.
- Cause: when the Retro page lost focus, it closed its connection to the game. Retro ends any handoff whose sponsor connection closes, and pauses the game.
- Now, during a handoff, clicking into another window keeps the connection open, so Orion keeps playing while you watch.
- A hidden tab or a closed page still ends the handoff, as before. Nobody can watch a hidden tab.
- A human's controller still releases on blur, as before, so keys never stick.
- A gamepad disconnect during a handoff no longer ends Orion's turn. Closing the page (`pagehide`) still does.
- When the handed-off turn ends (time limit, pause, or leisure takes the port), the page leaves handoff mode. Clicking the game takes it back again.

## Outcome moved

Juniper can hand Zelda to Orion ("Try Orion goals") and watch from another window. Live evidence of the failure, 2026-10-23 05:13 UTC, gateway log: `POST .../scripted 200` → both sockets closed → one `POST .../plan 409` → `resume`. Atlas: script controlled 162 frames, 0 decisions.

## Current architecture

`retro-panel.js` called `neutral()` on `blur`, `visibilitychange` (hidden) and `gamepaddisconnected`. `neutral()` closes the input and media sockets. Retro's worker handles the resulting `disconnect` by cancelling grants with `sponsor_disconnected` (`services/orion-retro/app/worker.py:143-148`).

## Architecture touched

Hub static asset only. No Retro, contract, or env change. Design change: the runbook said blur retains priority over a handoff. It now says a hidden tab does. Solo leisure already plays with no browser at all, so blur ending co-play protected nothing extra.

## Files changed

- `services/orion-hub/static/js/retro-panel.js`: a `stepAway` handler skips closing sockets while a handoff is active. `handoffGrant` tracks the handed-off grant's port and epoch, so the page leaves handoff mode when that grant ends.
- `services/orion-hub/templates/retro.html`: asset `v=9` (cache bust).
- `services/orion-hub/tests/retro/test_play_panel_static.py`: version bump plus a regression check on the blur handler.
- `services/orion-hub/evals/retro_play_panel.py`: the fixture logs socket closes and can end the grant (`/__fixture/end_grant`).
- `services/orion-hub/evals/retro_browser.mjs` (CI, real native core): blur during a handoff keeps it running; a hidden tab pauses and ends it. This replaces the old assertion that blur ends it.
- `docs/runbooks/retro-browser.md`: blur no longer ends a handoff.
- `services/orion-hub/evals/retro_play_panel.mjs`: hands off, blurs, and asserts the sponsor socket stays open. Then ends the grant while the game runs, blurs, and asserts the socket closes.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: browser handoff survives blur; hidden tab still ends it
- Compatibility notes: none

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
pytest services/orion-hub/tests/retro -q  -> 90 passed
```

## Evals run

```text
RETRO_PUPPETEER_PACKAGE=<main>/services/orion-hub/package.json python services/orion-hub/evals/retro_play_panel.py --output <tmp>
  fixed code      -> PASS
PYTHONPATH=. python services/orion-hub/evals/retro_browser.py --output /tmp/orb7                     -> PASS
PYTHONPATH=. python services/orion-hub/evals/retro_browser.py --ack-delay-ms 350 --output /tmp/orb8  -> PASS
  (output dir must be short: the Unix socket path limit is ~107 bytes)
  origin/main JS  -> AssertionError: a blur during a handoff closed the sponsor socket and would end Orion's turn
  first-commit JS -> AssertionError: after the handoff ended, a blur must release the socket (page stuck in handoff mode)
```

## Docker/build/smoke checks

```text
Static asset change. Hub bind-mounts services/orion-hub/{static,templates} from the primary checkout (docker-compose.yml:61-62), so git pull on main makes it live. Live check after deploy: hand off, click another window, confirm decisions > 0 in the atlas session.
```

## Review findings fixed

Review ran in a subagent. No blockers. Core fix confirmed correct: no path leaves a human holding the controller in handoff mode.

- Finding (medium): a hidden tab kept Orion's grant alive, but the status poll skipped hidden tabs, so Orion stopped planning.
  - Fix: a hidden tab now ends the handoff, as it did before this PR. Blur alone keeps it.
  - Evidence: retro_browser.mjs asserts blur keeps the grant running, then hidden pauses and ends it.
- Finding (medium): the page left handoff mode only when Retro was paused with nothing scripted. Before, blur also reset it by accident. If leisure took the port after the grant expired, the page stayed in handoff mode, and clicking the game did nothing.
  - Fix: the page latches the handed-off grant's epoch and leaves handoff mode when that grant is gone.
  - Evidence: new eval step. The first-commit JS fails it, and the fix passes it.
- Finding (low): sockets stay open forever in a hidden tab after the grant ends.
  - Fix: a hidden tab closes the sockets again.
  - Evidence: same retro_browser step.
- Finding (CI, service-container): the Nova goal test timed out. Retro's status lags a new handoff, so the first poll showed no grant and cleared handoff mode. The page then never planned.
  - Fix: the grant counts as ended only after a status showed it, and only for a status requested after the handoff.
  - Evidence: reproduced locally (timeout at retro_browser.mjs:145). Both retro_browser modes now pass.
- Finding (low): the second takeover check in the eval could flake.
  - Fix: step replaced by the grant-end step, which waits for the page's own status text.
  - Evidence: 3 PASS runs.

## Restart required

```text
No restart required (static and templates are directory bind mounts). Reload the Retro page.
```

## Risks / concerns

- Severity: low
- Concern: Chrome reports a fully covered window as hidden, which ends the handoff. To watch, keep part of the game window visible.
- Mitigation: by design, matching the old hidden-tab rule. Each handoff is still capped at 60 s.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2865
