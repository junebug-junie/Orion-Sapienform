## Summary

- New eval `services/orion-hub/evals/retro_coplay.mjs`. A real browser loads a Zelda save, clicks the game, and hands the controller to Orion for every 60 s grant. It then reads the session's action bets and records every socket close.
- **Planner:** a stray `value` on a non-`reach_room` prediction is now dropped (and logged) instead of refusing the whole move. Plan refusals are logged with their reason.
- **Gateway:** `orion-hub-retro-gateway` now exports co-play sessions. It registers the session-end sink and opens its memory pool in the background, so it never waits for the database.
- **Leisure:** a session that never played a goal skips its end save.
- **Menu:** counts below one unit are hidden, and the line falls back to the skill count.

## Outcome moved

First live harness run (2026-10-23, 5 min, PRG1 save d4d73abd…):

- Orion took 5 grants and placed 15 action bets. 11 succeeded, 3 failed (`interrupted:hit`) and 1 was voided at session end.
- The chosen option was rank 1 eleven times.
- The atlas grew from 2 rooms to 7.

The run found four real bugs:

1. **Orion could never fight.** In room 118, with 4 enemies, Orion answered "fight nearest, enemies_cleared, value 1" 13 times, and every answer was refused (`planner_prediction_value_mismatch`). The same decision replayed on the live GPU now gives 3 of 3 "fight nearest" (menu rank 2), with nothing refused.
2. **Co-play memory has never worked.** `watch_coplay` runs in the gateway process, but only Hub `main.py` registered the exporter. From 2026-10-20 every co-play handoff wrote no `retro_play_sessions` row and no memory card: 7 handoffs, 0 rows. The only row today is a leisure session from the main Hub.
3. **Solo leisure trapped itself on the title screen.** It started fresh on a game with no playable save. The planner never steers through menus, so it planned 0 goals, then saved frame 10. Every later session loaded that title save.
4. **Unreadable menu lines.** Counts like "0.5 of 0.5 in this situation" no longer appear.

## Current architecture

- **Two processes mount the same Retro router:** Hub `main.py` and `scripts/retro_gateway.py`. Tailscale sends `/api/retro` to the gateway.
- The session-end sink and the memory pool existed only in Hub `main.py`.

## Architecture touched

- `scripts/retro_gateway.py`: a lifespan that registers the sink and opens the pool in the background.
- `retro_planner.validate_zelda_choice`: normalizes the stray value.
- `retro_leisure._finish`: the end-save rule.

There is no contract change.

## Files changed

- `services/orion-hub/evals/retro_coplay.mjs`: new co-play harness.
- `services/orion-hub/scripts/retro_gateway.py`: the exporter and the background pool.
- `services/orion-hub/scripts/retro_planner.py`: drops and logs a stray value; plain refusal note.
- `services/orion-hub/scripts/retro_routes.py`: the `retro_plan_rejected` log line.
- `services/orion-hub/scripts/retro_experience.py`: counts below one unit hidden; skill fallback.
- `services/orion-hub/scripts/retro_leisure.py`: `skipped:no_goal_played`.
- `services/orion-hub/tests/retro/test_planner_v2.py`, `test_experience.py`, `test_leisure.py`: regression tests.
- `docs/runbooks/retro-browser.md`: the harness section.

## Schema / bus / API changes

None. Retro already accepted `value: null` for every non-`reach_room` kind (`services/orion-retro/app/contracts.py`). Those predictions were posted that way before this patch.

## Env/config changes

- No template keys changed.
- Operator-local only: `RETRO_LEISURE_GAME_IDS` changed from `the-legend-of-zelda` to `the-legend-of-zelda-prg1` in `services/orion-hub/.env`. PRG1 is the copy with the operator's progress; the other copy has only a title-screen save. The `.env_example` default stays generic, because the ROM choice belongs to the operator.

## Tests run

```text
PYTHONPATH=. pytest services/orion-hub/tests/retro -q      173 passed
node --check services/orion-hub/evals/retro_coplay.mjs     ok
```

## Evals run

```text
retro_coplay.mjs --minutes 5 (live)        15 bets (11 success, 3 failure, 1 void), 5 grants, 7 rooms
room 118 decision replay (live GPU, fixed planner)   3/3 fight nearest, 0 refused (before: 3/3 refused)
retro_coplay.mjs --minutes 0.3 --no-click (paused and stopped session)   handoff holds, bets placed
```

## Docker/build/smoke checks

```text
gateway app with RECALL_PG_DSN unreachable: /api/retro-gateway/health 200 after 0.19 s; sink registered;
shutdown 0.19 s with the pool retry still running (cancelled)
```

## Review findings fixed

- **Finding (medium):** the relay awaited the memory pool before serving. That is about 58 s of refused game sockets while sql-db restarts, and up to about 31 min if connects hang.
  - Fix: the sink is registered at once and the pool opens in a background task. Shutdown cancels the task, and the pool close is bounded at 5 s, with `terminate()` as the fallback.
  - Evidence: the smoke check above, and `test_gateway_registers_the_coplay_exporter`, which uses a slow pool.
- **Finding (low):** dropping the value could hide a mislabelled prediction kind.
  - Fix: `retro_planner_prediction_value_dropped kind=… value=…` is now logged, so the rate is visible.
- **Finding (low):** the harness missed `closes` and the session on an error.
  - Fix: both are collected in the error path.
- **Not changed:**
  - Skipping the save does not repair a title save that already exists. The PRG1 setting avoids it here; choosing a save is a follow-up.
  - The harness can over-count handoffs. It is diagnostic only.

## Restart required

```bash
git pull   # post-pull rebuild of orion-hub (Hub + retro gateway); also picks up the local RETRO_LEISURE_GAME_IDS change
```

## Risks / concerns

- **Severity Low:** a stray value is now accepted silently, apart from the log line.
  - Concern: an `enter_cave` that was meant as `reach_room` gets graded as the wrong claim.
  - Mitigation: the drop is logged, so the rate can be checked.
- **Severity Low:** the 3 s sponsor drop from the first harness run (handoff right after Load, with no click) did not reproduce.
  - Mitigation: the harness now records every socket close with its stack, to catch it next time.
- **UNVERIFIED:** that a co-play memory card is written. That needs this deployed and one more harness run.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2882

🤖 Generated with [Claude Code](https://claude.com/claude-code)
