## Summary

- A tap that is replaced before the next frame still reaches the game for one frame (`latch` in `Arbiter`). A newer latched direction wins, and a latched direction yields to the current opposite one.
- A command (the browser's resume right after New game) waits for the worker to be ready (≤4.5 s) before its 0.75 s timeout starts, so core loading can no longer fail the session.
- The wait re-checks the pending cap and the session id. `stop()` wakes waiters.
- `retro_play.mjs` taps Start 4 times and reports `start_presses_applied` from the server journal.

## Outcome moved

Found by running `retro_play.mjs` (2026-10-20):
- **Crystalis, live:** 1 of 4 quick Start taps reached the game, and it never left the title. On this branch: 4 of 4, and the intro started.
- **Nova:** failed at frame 0 with `worker_unavailable`. Its resume came ~0.48 s after start, inside a 0.75 s budget, while the core was loading.
- **On this branch:** all 6 catalog games (nova, crystalis, spacegulls, tetris, super-mario-bros, pad-lab) give 4/4 Start taps, ack p95 25-27 ms, no closes, no drops. Runs used a private isolated stack with the operator catalog.

## Current architecture

- A newer input on a port replaced its pending input before the frame. Only the last mask reached the core.
- `Sessions.command` started its 0.75 s timeout immediately, even while the worker was still starting. A timeout stops the session.

## Architecture touched

`services/orion-retro/app/control.py`, `services/orion-retro/app/session.py`. Harness: `services/orion-hub/evals/retro_play.mjs`. No API, schema, bus or env change.

## Files changed

- `services/orion-retro/app/control.py`: latch, newest-direction and yield rules.
- `services/orion-retro/app/session.py`: ready event, bounded wait, re-checks after the wait, `stop()` sets ready.
- `services/orion-retro/tests/test_authority.py`: tap, flick, yield, takeover, starting-worker and waiter-cap tests.
- `services/orion-hub/evals/retro_play.mjs`: Start taps and `start_presses_applied`.
- `docs/runbooks/retro-browser.md`: Quick taps, Start readiness.

## Schema / bus / API changes

- Behavior changed: an `applied` journal row can show a combined mask (latched plus current). A `superseded` reply no longer means "never reached a frame".
- Compatibility: no consumer compares the submitted and applied masks (checked the smokes and the browser eval).

## Env/config changes

None.

## Tests run

```text
services/orion-retro/tests: 42 passed (new tap, starting-worker and waiter-cap tests fail on old code)
services/orion-hub/tests/retro: 3 passed
```

## Evals run

```text
retro_browser.py (branch image): delay 0 PASS, delay 350 PASS
container_smoke.py (branch image): PASS
retro_play.mjs, 6 games on an isolated stack: all 4/4 Start, 0 drops
```

## Docker/build/smoke checks

```text
docker build orion-retro (branch): OK
```

## Review findings fixed

- Finding: left, right, release inside one frame cleared the axis.
  - Fix: a newer latched direction replaces the older one.
  - Evidence: `test_flick_across_both_directions_keeps_the_newest`.
- Finding: waiters were not counted in the pending cap, so they could overflow the 32-slot queue on ready and fail the session.
  - Fix: re-check the cap after the wait.
  - Evidence: `test_waiters_released_together_still_respect_the_pending_cap` fails without the fix.
- Finding: a stop/restart left waiters on a dead event, and a waiter could reach a newer session.
  - Fix: `stop()` sets ready; session-id re-check.
- Finding: the 10 s wait outlasted Hub's 5 s proxy timeout.
  - Fix: the wait is 4.5 s.
- Noted only: a latch-then-takeover is correct by construction, now pinned by a test.

## Restart required

Post-pull rebuild of orion-retro (automatic). Manual equivalent:

```bash
scripts/safe_docker_build.sh orion-retro up -d --build
```

## Risks / concerns

- Severity: low. Concern: a one-frame press can still be missed by a game that skips its controller read on a lag frame. This is game-side and unchanged.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
