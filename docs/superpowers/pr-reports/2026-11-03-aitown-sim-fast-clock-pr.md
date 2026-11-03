## Summary

Juniper asked (2026-11-02): speed up the town's game clock "to like 20x or something", and drop to real time whenever a GPU needs to run, so the world fits inside Orion's 2-hour daily town budget.

- The simulation clock now runs fast while the town needs no model call. It drops to real time while any conversation is open or any NPC has a model call in flight.
- The fast speed is 45x, not 20x. On 10-31, chats held the GPU about 75% of each session, so 20x would average about 6x: about 12 game hours per 2-hour day. 45x averages about 12x: about one game day per 2-hour day, which is the goal Juniper named. It is one constant, `FAST_CLOCK_X`.
- The cron now ticks every 60 s instead of every 5 min. The clock moves in whole 5-minute steps, and the remainder carries to the next tick.
- `simState.clock` counts busy and fast ticks, so the real split shows in `sim/state:snapshot`.

## Outcome moved

- Before: game time moved only at real speed, and only while the town was open (at most 2 h a day). The first breakfast came after 3 real days (live, 10-30 to 11-02: game minute 560 after about 9.3 open hours).
- After: about one game day per 2-hour daily budget (modelled on the 10-31 chat load; live rate UNVERIFIED until a session runs).

## Current architecture

`crons.ts` (through `orion-sim-core-hooks.patch`) ran `sim/tick:run` every 5 minutes, which advanced the simulation by 5 game minutes while the world status was `running`. The town runs only during Orion's chosen sessions or Juniper's manual sessions.

## Architecture touched

AI Town overlay and one patch line. No bus, schema-registry, env or service-code changes.

## Files changed

- `services/orion-ai-town/overlay/convex/sim/core.ts`: `FAST_CLOCK_X`, `CLOCK_TICK_SECONDS`, `townNeedsGpu`, `advanceClock` (pure).
- `services/orion-ai-town/overlay/convex/sim/tick.ts`: reads the world doc, advances the clock, ticks the whole steps.
- `services/orion-ai-town/overlay/convex/sim/schema.ts`: optional `clock` on `simState`.
- `services/orion-ai-town/overlay/convex/sim/state.ts`: the snapshot returns `clock`.
- `services/orion-ai-town/patches/orion-sim-core-hooks.patch`: the cron runs every 60 s.
- `services/orion-ai-town/tests/test_sim_core.py`, `tests/fixtures/sim_core_driver.ts`: clock tests.
- `services/orion-ai-town/README.md`: the game clock.

## Schema / bus / API changes

- Added: optional `simState.clock` `{pending, busy_ticks, fast_ticks}` (Convex table, not the Orion registry).
- Removed: `SIM_TICK_MINUTES`.
- Renamed: none.
- Behavior changed: the cron cadence and the game minutes per tick.
- Compatibility notes: the field is optional, so the live `simState` row validates. The first tick creates it.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- skipped keys requiring operator action: none.

## Tests run

```text
pytest services/orion-ai-town/tests/test_sim_core.py (node 24, AITOWN_REQUIRE_TOOLS=1) -> 24 passed
  idle hour = 2700 game min; busy hour = 60 (steps 0/5 only, carry 0); 2 h at 75% busy = 20-28 game h
tsc --noEmit -p convex on the fully patched tree (circe node_modules, circe _generated/api.d.ts) -> 0 errors
  planted error in tick.ts -> TS2551 reported
git diff --check -> clean
```

## Evals run

None. The live check is the clock split in `sim/state:snapshot` after the first session.

## Docker/build/smoke checks

Not run by hand. The merge redeploys the town on circe through `pre_rebuild.sh` (codegen) and `post_rebuild.sh` (push).

## Review findings fixed

Review: no material findings. It confirmed four things: ended conversations leave `world.conversations`, stuck operations time out after 120 s, the engine is never the side that retries on a conflict, and old `simState` rows validate.

- Finding (minor): `bootstrap` with `replace` dropped `clock`, so a reseed reset the busy/fast counts.
  - Fix: the replace keeps the existing `clock`.
  - Evidence: `tsc` and the 24 tests pass.
- Noted, not fixed (low): after both players accept, a walk-over never times out (`orion-walkover-meet.patch`). If two players can never meet, the clock stays at real time until the conversation ends. The simulation still advances, slowly.
- Noted: `simEvents` gets up to 45 times more rows per idle hour. The snapshot reads at most 200 rows.

## Restart required

None by hand. After the merge, check on circe: `tail -6 ~/.orion/post_pull_rebuild.log` shows `POST-END rc=0 orion-ai-town`. During the next session, `npx convex run sim/state:snapshot` shows the `clock` counts moving.

## Risks / concerns

- Severity: low. Concern: 45x is fitted to one day's chat load. If sessions run quieter, a 2-hour day can reach about 3.75 game days. Mitigation: one constant; the busy/fast counts show the real split.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3036

🤖 Generated with [Claude Code](https://claude.com/claude-code)
