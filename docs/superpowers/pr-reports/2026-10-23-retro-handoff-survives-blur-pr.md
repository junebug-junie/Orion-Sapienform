## Summary

- Clicking into another window ended Orion's co-play turn within about a second. This fixes that.
- Cause: when the Retro page lost focus, it closed its connection to the game. Retro ends any handoff whose sponsor connection closes, and pauses the game.
- Now, during a handoff, losing focus or hiding the tab only clears held keys. The connection stays open, so Orion keeps playing.
- A human's controller still releases on blur, as before, so keys never stick.
- A gamepad disconnect during a handoff no longer ends Orion's turn. Closing the page (`pagehide`) still does.
- A hidden tab keeps asking Orion for goals during a handoff, so Orion keeps playing.
- When the handed-off turn ends (time limit, pause, or leisure takes the port), the page leaves handoff mode. Clicking the game takes it back again.

## Outcome moved

Juniper can hand Zelda to Orion ("Try Orion goals") and watch from another window. Live evidence of the failure, 2026-10-23 05:13 UTC, gateway log: `POST .../scripted 200` → both sockets closed → one `POST .../plan 409` → `resume`. Atlas: script controlled 162 frames, 0 decisions.

## Current architecture

`retro-panel.js` called `neutral()` on `blur`, `visibilitychange` (hidden) and `gamepaddisconnected`. `neutral()` closes the input and media sockets. Retro's worker handles the resulting `disconnect` by cancelling grants with `sponsor_disconnected` (`services/orion-retro/app/worker.py:143-148`).

## Architecture touched

Hub static asset only. No Retro, contract, or env change.

## Files changed

- `services/orion-hub/static/js/retro-panel.js`: a `stepAway` handler skips closing sockets while a handoff is active. `handoffGrant` tracks the handed-off grant's port and epoch, and the status poll keeps running in a hidden tab during a handoff.
- `services/orion-hub/templates/retro.html`: asset `v=9` (cache bust).
- `services/orion-hub/tests/retro/test_play_panel_static.py`: version bump plus a regression check on the blur handler.
- `services/orion-hub/evals/retro_play_panel.py`: the fixture logs socket closes and can end the grant (`/__fixture/end_grant`).
- `services/orion-hub/evals/retro_play_panel.mjs`: hands off, blurs, and asserts the sponsor socket stays open. Then ends the grant while the game runs, blurs, and asserts the socket closes.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: browser handoff survives blur/hidden
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
  fixed code      -> PASS (3 runs in a row)
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
  - Fix: the poll keeps running while a handoff is active.
  - Evidence: static test asserts the guard.
- Finding (medium): the page left handoff mode only when Retro was paused with nothing scripted. Before, blur also reset it by accident. If leisure took the port after the grant expired, the page stayed in handoff mode, and clicking the game did nothing.
  - Fix: the page latches the handed-off grant's epoch and leaves handoff mode when that grant is gone.
  - Evidence: new eval step. The first-commit JS fails it, and the fix passes it.
- Finding (low): sockets stay open forever in a hidden tab after the grant ends.
  - Fix: covered by the two fixes above. The poll runs and clears handoff mode.
  - Evidence: same eval step.
- Finding (low): the second takeover check in the eval could flake.
  - Fix: step replaced by the grant-end step, which waits for the page's own status text.
  - Evidence: 3 PASS runs.

## Restart required

```text
No restart required (static and templates are directory bind mounts). Reload the Retro page.
```

## Risks / concerns

- Severity: low
- Concern: in a hidden tab, browsers throttle timers to about once per second, so planning can lag slightly.
- Mitigation: the grant is capped at 60 s, and Retro's motor holds still between goals.

## PR link

PR_LINK
