## Summary

- Orion stood still in a dungeon room during co-play. The planner said "wait" at every decision, because Retro said the room had no exits.
- The room's exits were bombed holes on the west and east walls. The floor of a hole stops 3 tiles short of the screen edge. Retro counted a dungeon side as open only when the floor got within 2 tiles, which is right for doorways.
- West and east now use 3 tiles. North and south stay at 2.

## Outcome moved

- Live 2026-10-30, session `e17c02c5`, dungeon room 85: `open_sides` was `[]`. The planner chose `wait` at about 15 decisions. Link stood in the west hole he had just walked through.
- With this change, the same live grid gives `['E', 'W']`.

## Current architecture

`open_sides` in `services/orion-retro/app/zelda/navigate.py` runs the motor's own reachability search. It lists the sides whose edge Link can reach. In a dungeon it allows `DOOR_DEPTH` tiles of slack, because the opening's outer frame reads as collision. The planner can only choose an exit on a listed side.

## Architecture touched

orion-retro only. No bus, schema, env or API change. The meaning of `open_sides` is unchanged; it now lists more of the real exits.

## Files changed

- `services/orion-retro/app/zelda/navigate.py`: `DOOR_DEPTH` is now per side, `{N: 2, S: 2, W: 3, E: 3}`.
- `services/orion-retro/tests/test_zelda.py`: `ROOM_85` live grid. The bombed holes count as exits. The same room with the holes filled stays closed.
- `services/orion-retro/README.md`: the `open_sides` description gives the per-side rule.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: in dungeons, a W/E opening 3 tiles short of the edge is now an open side.
- Compatibility notes: none.

## Env/config changes

None.

## Tests run

```text
services/orion-retro: pytest tests -q -> 138 passed
Mutation: W/E back to 2 -> the new test fails.
Mutation: N to 3 -> 2 existing tests fail (a plain N wall stops 3 short), so N stays 2.
```

## Evals run

No new eval. `evals/zelda_dojo.py` uses the same navigator, and the dojo keeps running.

## Docker/build/smoke checks

Live check after the deploy: hand off to Orion in a dungeon room with W/E holes or doorways. The decision's `attempt.open_sides` should list them, and Orion should exit instead of waiting.

## Review findings fixed

See below (filled in after review).

## Restart required

The Retro service rebuilds on `git pull`.

## Risks / concerns

- Severity: low. A W/E feature that is not an exit could make a 1-tile notch 3 tiles short of the edge. If an exit there fails, the motor learns the bump, and the side then reads as closed.
- Severity: low. Bombed holes on the N/S walls are not measured yet. If they also stop 3 short, N cannot tell them from a plain wall with this rule.
- Not in scope: the hidden stairs in the block square. The motor has no "push block" goal.

## PR link

TBD
