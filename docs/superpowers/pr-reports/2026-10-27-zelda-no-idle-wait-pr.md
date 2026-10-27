## Summary

- In live co-play after #2905, Orion chose "wait" 41 times out of 41. Link never left the first room.
- Cause: the room was empty and its open sides were unknown. Every move option said "Predicts unknown", and "wait" said "Predicts survive". The model took the safe-sounding pick. Each wait was met, so its chance went up (37 of 37), which made the next wait look better still.
- Fix: the menu offers "wait" only when an enemy is close to Link, when Link cannot be controlled, or when no other move is possible (`build_options`, `services/orion-hub/scripts/retro_experience.py`). A far enemy is not enough: Link survives every wait, so the wait counts as met and the loop never ends.

## Outcome moved

The same live decision (session 6c359e66, room 124), replayed through the real planner with the fixed menu, gave "explore N" 5 times out of 5 (before: "wait" 41 of 41 live).

## Current architecture

The Hub builds a menu of scored options (action bets from orion-prediction) and the planner picks one by number. "wait" was always on the menu.

## Architecture touched

Only the Hub's Zelda option builder. There are no contract, schema, bus or env changes.

## Files changed

- `services/orion-hub/scripts/retro_experience.py`: offer wait only with a close enemy, an uncontrollable Link, or as the only option. The cap keeps the last option only when it is wait.
- `services/orion-hub/tests/retro/test_experience.py`: added `watched_decision` (room 119 with an enemy next to Link) for tests that need wait. Added a regression test: an empty room, a far enemy, an uncontrollable Link, and a boxed room.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: the wait option is offered only with a close enemy, an uncontrollable Link, or as the only move.
- Compatibility notes: existing hold counts in orion-prediction stay. They are no longer offered in calm rooms.

## Env/config changes

None.

## Tests run

```text
cd services/orion-hub && pytest tests/retro -q   -> 187 passed
Mutation check: gate back to "any enemy on screen" -> regression test fails; restored -> 187 passed
zelda_experience_replay.py (reviewer run) -> PASS 5/5
```

## Evals run

```text
Live co-play before fix: retro_coplay.mjs --minutes 5 -> 41 bets, all "hold" (rank 5), room 124 only
Replay of that session's first decision through the real planner (fixed menu, live prediction quotes):
  menu = explore N/S/E/W (no wait); 5/5 runs -> explore N, rank 1
zelda_planner_replay.py --with-menu (room 119 fixture) -> PASS 5/5 (before this patch, unchanged)
```

## Docker/build/smoke checks

```text
No build run. Python-only Hub change; the post-pull rebuild covers orion-hub and the retro gateway.
```

## Review findings fixed

- Finding (medium): one far wandering enemy put wait back on the menu, and the loop could return, because each wait still counts as met.
  - Fix: wait needs a close enemy (the same "near" band that offers retreat).
  - Evidence: far-enemy case in `test_wait_is_offered_only_when_something_on_screen_can_change`; mutation check fails without it.
- Finding (medium): the "an unchanged wait cannot repeat" rule would end solo play in a room where wait is the only move (refused twice, 3 failures, `planner_failed_x3`). It also ignored changes after the wait ended (spawns, shutters).
  - Fix: dropped that rule. The menu gate alone fixes the live loop (replay 5/5 explore).
  - Evidence: `retro_planner.py` unchanged versus main; boxed-room case keeps wait as the only option.
- Finding (low): the menu dropped wait when Link cannot be controlled, and that is the only goal Retro accepts then.
  - Fix: wait is offered when `link.controllable` is false.
  - Evidence: frozen case in the same test.
- Not fixed (low): drops are not in Retro's digest. This no longer matters, because no digest rule was added.

## Restart required

```bash
git pull   # rebuilds orion-hub and the retro gateway
```

## Risks / concerns

- Severity: medium
  - Concern: Retro reported "sides unknown" in room 124, so the menu still offers exits into walls. In the replay, "explore N" walks toward a wall (the only exit is south).
  - Mitigation: the existing no-effect refusal stops repeats. Follow-up: work out open sides from the visible terrain when Retro does not report them.
- Severity: low
  - Concern: a wait with no close enemy can be the right move (for example, a timed door, or letting a far enemy come closer).
  - Mitigation: none observed in play so far. Wait stays when it is the only option.
- Severity: low
  - Concern: room 124 reported level "unknown", which is why its sides were unknown (Retro works out sides only on the overworld). The save likely starts in a dungeon the game memory reader cannot decode.
  - Mitigation: follow-up.

## PR link

(added after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
