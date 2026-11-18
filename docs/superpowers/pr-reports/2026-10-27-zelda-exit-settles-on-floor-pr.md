## Summary

- After #2925, live co-play (session 88a0a869) still looped: Link crossed rooms 69 <-> 85 twenty-nine times.
- Cause: room 85, entered from the north, puts Link at tile row 1 of a passage. The RAM wall grid marks that passage solid down to row 8, though the game lets Link walk it. The exit ended on the room change. The next decision searched from a "solid" tile, so it could reach nothing and offered only `open_sides=['N']`, the way back.
- Fix: once an exit or explore goal is in the new room, it keeps walking the goal's direction until Link stands on open floor. This lasts at most `SETTLE_FRAMES` (90 frames) and stops at the goal deadline. Only then does the goal finish and the next decision get offered (`on_floor`, `_settling` in `services/orion-retro/app/zelda/navigate.py`).
- During the walk-in, no bumps are learned. A blind push is no evidence about a planned tile, and the planned path still belongs to the room Link left.

## Outcome moved

Regression test: in an eastward entry band that is solid on the grid but walkable in play, Link ends on open floor (column 4 or more). The next offer lists more than the way back. Without the fix Link stays at column 0, so the test fails.

## Current architecture

An exit finishes on the first play frame in a new room. The decision offer computes `open_sides` by motor reachability from Link's tile.

## Architecture touched

The orion-retro navigator only. No contract changes.

## Files changed

- `services/orion-retro/app/zelda/navigate.py`: `SETTLE_FRAMES`, `on_floor`, `_settling`, the settle push in `_drive`, and the bump skip while settling.
- `services/orion-retro/tests/test_zelda.py`: the entry-passage regression test.

## Schema / bus / API changes

- Behavior changed: an exit or explore goal can last up to 1.5 s longer, walking Link onto open floor. It never runs past its deadline and never delays a death (settling is play-mode only).

## Env/config changes

None.

## Tests run

```text
PYTHONPATH=. pytest tests/retro services/orion-retro/tests -q -> 163 passed
services/orion-hub pytest tests/retro -q                  -> 191 passed
Mutation: settling off -> regression test fails (Link at column 0)
```

## Evals run

```text
evals/zelda_play.py on the real core (retro image) -> 13/13 PASS
Live co-play before fix (88a0a869): 69<->85 x29; room-85 offers at tile [16, 1] had open_sides ['N']
```

## Docker/build/smoke checks

```text
zelda_play.py ran in orion-retro-retro with the real fceumm core
```

## Review findings fixed

- Finding (should): during settling, the old room's planned path could be learned as a blocked tile in the new room.
  - Fix: `_bump` is skipped while `settle_ticks > 0`.
- Finding (should): settling could overrun the goal deadline by 90 frames.
  - Fix: settling only runs while `now < deadline`.
- Finding (behaviour): moving the death check ahead of the verdict changed death outcomes from `prediction_refuted` to `death`, which changes the Hub's bet data.
  - Fix: reverted. The death order is as before. Settling is play-mode only, so it never delays a death.

## Restart required

```bash
git pull   # rebuilds orion-retro; a running game ends
```

## Risks / concerns

- Severity: medium
  - Concern: the 70 <-> 71 loop in the same session is not this bug. Both rooms have another open side, but the neighbours' visit counts tie (9 each), so "chance it works" (5 of 5 for the way back) decides. It self-corrects slowly as visits grow.
  - Mitigation: the next PR ranks moves with the trained world model, served first as a shadow model in orion-prediction. No more hand rules.
- Severity: low
  - Concern: arriving from the north can read as row 0. That adds a few frames of push before the goal finishes.
  - Mitigation: harmless (review checked).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2934

🤖 Generated with [Claude Code](https://claude.com/claude-code)
