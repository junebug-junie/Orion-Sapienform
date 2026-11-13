## Summary

- The town was frozen. Both NPC pairs were stuck walking toward each other across a wall, and no lines were spoken. The meeting spot (their midpoint) was the wall tile itself.
- Now, when the midpoint is a wall or off the map, one NPC walks to a free tile next to its partner, and the partner waits. An NPC never waits for a human.
- An accepted walk-over that hasn't met in 3 minutes is abandoned. `orion-conversation-proximity.patch` had removed the only time limit.
- This also ends Orion's loop: its invite to p:0 was refused about once a second, because p:0 was stuck in the frozen walk-over.

## Outcome moved

- Failure mode: after the 2026-10-23 resume, the engine ran, but no conversation ever reached `participating`. Live evidence:
  - World state: `c:63910` and `c:580262`, both members `walkingOver`, 0 messages, positions unchanged over 75 s.
  - `convex logs`, 25 s: 32× `Failed to route to {"x":25,"y":26}`, 28× `Failed to route to {"x":4,"y":23}`, 18× `Player p:29 is already in a conversation`.
- I replayed the rule on the live map with the live positions. Both walkers get reachable routes: p:0 → (28, 26) in 45 steps, p:2 → (5, 21) in 12 steps.

## Current architecture

`agent.ts`, walking-over branch: below 4 tiles apart, an agent walks to the partner's tile; otherwise it walks to the floored midpoint. The route search (A*) can't reach a wall tile, and its fallback keeps the start tile when the start is already closest. So it returns null, the player stops, and the next tick repeats.

## Architecture touched

New patch `services/orion-ai-town/patches/orion-walkover-meet.patch` (`convex/aiTown/agent.ts`, `convex/constants.ts`), applied last.

## Files changed

- `services/orion-ai-town/patches/orion-walkover-meet.patch` (new): blocked-midpoint rule, `freeTileNextTo`, `WALK_OVER_TIMEOUT`.
- `services/orion-ai-town/scripts/apply_upstream_patches.sh`: registers it.
- `services/orion-ai-town/tests/test_walkover_meet_patch.py` (new): patch contract.
- `services/orion-ai-town/tests/test_live_turn_roles_patch.py`: that patch is no longer last.
- `services/orion-ai-town/README.md`: section for the new patch.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: walk-over destination when the midpoint is blocked; walk-overs end after 3 minutes.
- Compatibility notes: after deploy, the two currently stuck pairs (accepted weeks ago) leave on the first tick.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- skipped keys requiring operator action: none.

## Tests run

```text
pytest services/orion-ai-town/tests (AITOWN_REQUIRE_TOOLS=1, node 24, local pinned clone; includes full-chain apply) -> 144 passed, 1 skipped (pre-existing)
tsc --noEmit -p convex on the fully patched upstream -> 0 errors
live-map replay of the rule (Python, same blocking rules as movement.ts) -> both walkers reach their tile (45 and 12 steps)
git diff --check -> ok
```

## Evals run

None. The live check after deploy is that conversations reach `participating`, and town lines appear in `social_room_turns` again. The last one was on 2026-09-17.

## Docker/build/smoke checks

Not deployed. It needs a merge, then a pull on circe. Once #2880 is merged, that pull deploys it automatically; without #2880, use the manual recipe. Live result UNVERIFIED.

## Review findings fixed

- Finding (medium): an NPC with a larger id waited for a human partner, who only moves on a click.
  - Fix: `!otherPlayer.human &&` on the wait branch.
  - Evidence: test asserts the guard.
- Finding (low): the comment said the route search can't go around walls. It's A*; the real cause is the unreachable-target fallback.
  - Fix: corrected the comment.
  - Evidence: patch.
- Noted, not fixed (low; the timeout bounds them): no path between the pair, all four tiles next to the partner taken, and a pair that re-invites after the cooldown. All are documented in the README.

## Restart required

Merge #2880 (deploy on pull) first, then this PR. Then on circe, from the clean primary checkout: `git pull --ff-only`. The post-merge rebuild runs `pre_rebuild.sh`, builds, then runs `post_rebuild.sh`.

## Risks / concerns

- Severity: low. Concern: a pair with no path between them freezes for 3 minutes at a time. Mitigation: timeout and cooldown.
- Severity: low. Concern: the 3-minute limit also ends a slow but real walk across the map. Mitigation: 3 minutes covers about 2 map widths at walking speed.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2881

🤖 Generated with [Claude Code](https://claude.com/claude-code)
