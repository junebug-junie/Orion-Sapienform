## Summary

- After #2918, live co-play moved, but Orion chose "explore N" 32 times in a row, and Link hugged the top wall of dungeon room 124.
- Cause: Retro reported no open sides in dungeons. So all four directions looked the same, and the model took the first one each time. Each run moved Link along the wall, so the "did nothing" check never fired.
- In a dungeon, the outer two tiles of each doorway read as collision in the game's wall map, yet the game walks Link through an open door.
- Fix: `open_sides(..., dungeon=True)` counts a side when the floor reaches within `DOOR_DEPTH = 2` tiles of that edge (`services/orion-retro/app/zelda/navigate.py`). Room 124 now reports S and W, its real doors. The overworld rule is unchanged.

## Outcome moved

Replay of the live room-124 decision through the real planner with the new sides: "exit S" 5/5 (before: "explore N" 32/32 live).

## Current architecture

Retro's decision offer carries `open_sides` from the motor's own reachability search. It was null outside the overworld, so the Hub offered `explore` on all four sides, with no side refusals.

## Architecture touched

orion-retro only (the navigator and README), plus one Hub comment. The contract field already exists; it is now a list in dungeons too.

## Files changed

- `services/orion-retro/app/zelda/navigate.py`: `DOOR_DEPTH`; `on_screen_edge` takes a slack; `open_sides` takes `dungeon`; the offer uses `dungeon = level != 0`.
- `services/orion-retro/tests/test_zelda.py`:
  - the live ROOM_124 grid, which gives S and W;
  - synthetic N and E doors;
  - the motor walks "exit S" to the doorway and pushes DOWN with no bump;
  - the W door target tiles;
  - the old "dungeons give null" test now expects sides.
- `services/orion-retro/README.md`: documents the dungeon rule.
- `services/orion-hub/scripts/retro_experience.py`: updates the stale comment (dungeon places stay overworld-only).

## Schema / bus / API changes

- Behavior changed: `attempt.open_sides` is a list in dungeon play mode. It was `null` there before.
- Compatibility notes: the Hub already handles a list. It offers `exit` per listed side and refuses exits through unlisted sides.

## Env/config changes

None.

## Tests run

```text
PYTHONPATH=. pytest tests/retro services/orion-retro/tests -q   -> 159 passed
cd services/orion-hub && pytest tests/retro -q                   -> 187 passed
Mutation: DOOR_DEPTH = 0 -> 2 dungeon tests fail; restored -> pass
```

## Evals run

```text
Live co-play (after #2918): 32 goals, all explore N, room 124, Link bouncing between (4,5) and (26,5)
Replay, open_sides=[S, W]: menu exit S / exit W; 5/5 exit S, rank 1
```

## Docker/build/smoke checks

```text
No build run. The post-pull rebuild covers orion-retro.
```

## Review findings fixed

- Finding (medium): no test drives the motor through a doorway.
  - Fix: a motor test runs "exit S" on ROOM_124. Link reaches (15,19), holds DOWN, and no bump is recorded. A second check confirms the W door target is column 2.
  - Evidence: `test_dungeon_exit_walks_to_the_doorway_and_pushes_through_without_learning_a_wall`. The real scroll through the door is UNVERIFIED until the live run.
- Finding (low): the N and E doors were inferred, not observed.
  - Fix: synthetic N and E doors, mirrored from the live S and W ones, test all four sides.
  - Evidence: `test_dungeon_sides_find_north_and_east_doors_mirrored_from_the_live_ones`.
- Finding (nit): stale comment in `retro_experience.py`.
  - Fix: updated.
- Not fixed (medium, by choice): locked doors and bombable walls are no longer offered in dungeons. Keys read as "unknown", so Orion cannot tell when a locked door could open. Offering every wall again would bring back the wall-hugging loop. Follow-up: decode keys, then offer a locked side when a key is held.

## Restart required

```bash
git pull   # rebuilds orion-retro; a running game ends
```

## Risks / concerns

- Severity: medium
  - Concern: Link walking through a real dungeon door is UNVERIFIED. If the game needs exact alignment, the push ends as `interrupted:stuck` after 2 s and nothing is learned. The side stays on offer.
  - Mitigation: the live co-play harness right after deploy. Look for a `goal_finished` with a room change out of 124.
- Severity: low
  - Concern: dungeon exits predict "unknown". The atlas skips dungeon rooms while the level reads "unknown", so verdicts stay unknown and no target room is shown.
  - Mitigation: follow-up with the dungeon level decode.
- Severity: low
  - Concern: the entrance room's S door leaves the dungeon.
  - Mitigation: none needed. It is a real exit.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2921

🤖 Generated with [Claude Code](https://claude.com/claude-code)
