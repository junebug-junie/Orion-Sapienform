## Summary

- After #2921, Orion chose the right doors in dungeon room 124 (exit S and exit W), and Link walked to each doorway. But he never went through. He shook 1 pixel back and forth for the whole 15 s budget, 19 times.
- Cause: on the last floor tile, the motor lines Link up to the exact pixel on both axes before it pushes. Each push moved him 1 px past the tile (y 205 to 206), and the next step pulled him back to "line up". On the overworld this never showed, because one push at the screen edge starts the scroll at once.
- Fix: a push lines Link up only across its own direction. Pushing N or S aligns x only; pushing W or E aligns y only. Collect (no push) still aligns both (`_follow`, `services/orion-retro/app/zelda/navigate.py`).

## Outcome moved

Stand-in run of "exit S" on the live room-124 grid, with the door frame passable as in the game: Link walks past the frame with x still 120 (before: the push stalled at y 205/206).

## Current architecture

`_follow` walks the A* path. On a goal tile it aligns exactly, then returns the push direction.

## Architecture touched

orion-retro motor only. No contract, schema or env changes.

## Files changed

- `services/orion-retro/app/zelda/navigate.py`: alignment skips the push axis.
- `services/orion-retro/tests/test_zelda.py`:
  - the `World(passable=...)` stand-in: tiles that are solid on the grid but open in play;
  - a regression test that pushes through the room-124 south door.

## Schema / bus / API changes

- Behavior changed: exit and enter pushes keep pushing along their direction once on the goal tile.

## Env/config changes

None.

## Tests run

```text
PYTHONPATH=. pytest tests/retro services/orion-retro/tests -q  -> 160 passed
Mutation: restore y-alignment during an N/S push -> new test fails; restored -> pass
```

## Evals run

```text
Live co-play (session 0b1d8ce8, after #2921): 19 goals, exit S x14 / exit W x5, all budget_expired,
Link at (120, 205<->206) and (15<->16, 141); 0 room changes
```

## Docker/build/smoke checks

```text
No build run. The post-pull rebuild covers orion-retro.
```

## Review findings fixed

- Review found no material issues. It confirmed the following:
  - cave `enter` still aligns x before pushing N;
  - collect still aligns both axes;
  - bump and stuck detection are unchanged;
  - the new test fails on the old code.
- The fix also helps caves. A push that moved Link 1-3 px without changing his tile was pulled back the same way.
- Not added (nit): a cave-entry test with Link offset vertically. Coverage gap, not a bug.

## Restart required

```bash
git pull   # rebuilds orion-retro; a running game ends
```

## Risks / concerns

- Severity: medium
  - Concern: the real game scrolling Link through the door is still UNVERIFIED. The stand-in only shows that the push is no longer undone.
  - Mitigation: rerun the live co-play harness after deploy, and look for a room change out of 124.
- Severity: low
  - Concern: in an earlier rebuild after `git pull`, the run stopped while building orion-harness-governor. Retro was never rebuilt, so I rebuilt it by hand.
  - Mitigation: check `docker ps` after the pull.

## PR link

(added after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
