# AI Town living world, PR 2: the simulation core

## Summary

- The town now has a world that changes on its own, in plain code with no model calls: places, stock, cash, each resident's needs, trust between characters, facts and quests. A Convex cron moves it forward 5 game-minutes every 5 minutes, and pauses while the town is stopped.
- Every change goes through one function (`core.applyEvent`) with a closed set of five effect operations, each clamped (needs 0..1, trust -1..1, stock never below 0). A free-text event with no effects becomes a rumor fact and moves no numbers.
- Quests are events with a goal (an AND of comparisons on stock, facts, needs or trust), a deadline and participants. The tick resolves them: success when the goal holds, failure past the deadline. Either way the reward or penalty applies.
- Every random draw comes from a generator stored in the world, so the same world plus the same seed replays exactly.
- The cards carry the numbers: each resident card has a `sim:` block, and `generate_descriptions.py` renders the seed from them.
- Juniper can add events from the CLI (`npx convex run sim/events:inject`). The Hub form and the bus path arrive with PR 4.

Design: `docs/superpowers/specs/2026-10-14-aitown-living-world-design.md` (PR #2666, not merged), sections 1, 2 (sources `sim` and `juniper` only), 3 (goal and deadline checks in the tick) and 8.

## Outcome moved

- **A1 (state moves), passed.** With no model calls at all, a seeded 24 game-hour run changes stock, needs and trust. Replaying the same seed gives exactly the same world. Running the day as 288 five-minute ticks, each followed by a Convex-style save (a JSON round trip with sorted keys), gives the same world and the same events as one 24-hour tick.
- **A7, Juniper half, passed on a scratch Convex backend.** A Juniper event writes one `simEvents` row with `source: "juniper"`. A refused one writes a row with `status: "rejected"` and the reason. An `orion` source is refused without a row.
- The world stored in Convex after a 24 game-hour forced tick equals a local Node replay of the same seed and the same event, byte for byte (after key sorting).

## Current architecture

- AI Town runs on circe as self-hosted Convex plus 18 Orion patches (`services/orion-ai-town/patches/`). They are applied to the gitignored, pinned `upstream/` by `scripts/apply_upstream_patches.sh`. `scripts/pre_rebuild.sh` resets `upstream/` to the pin and re-applies the chain before every post-pull rebuild. `scripts/post_rebuild.sh` pushes the functions.
- No table held stock, money, needs, events or facts. NPC activity is picked at random from three hardcoded options.
- The cards (`cards/town_cards.yaml`, real ones in the gitignored `.local.yaml`) held prose only.

## Architecture touched

- **New overlay mechanism.** `services/orion-ai-town/overlay/` holds whole new files, copied into `upstream/` after the patches. `pre_rebuild.sh` deletes them on reset, along with any overlay directory the pin does not have (`convex/sim/`) as a whole, so a file later dropped from `overlay/` cannot linger and deploy.
- **Convex (overlay, `convex/sim/`).** `core.ts` (pure), `schema.ts`, `store.ts`, `tick.ts` (`run`), `events.ts` (`inject`), `state.ts` (`snapshot`, `bootstrap`), `seed.ts` (generated).
- **Hook patch.** `patches/orion-sim-core-hooks.patch` adds `...simTables` to `convex/schema.ts` and the `sim tick` cron to `convex/crons.ts`, and nothing else. It is registered last in the chain.
- **Cards.** A `sim:` block on each resident, plus a top-level `sim_world:` block.

**Design choice: two tables instead of the spec's ten.** `simState` holds one versioned document per world with the whole world in it (places, stock, needs, trust, facts, quests, the random generator). `simEvents` is the append-only log (kind, source, title, place, effects, seed, status, reason, game-minute). The reasons:
1. One event's effects often span several of the spec's tables (stock, trust and a fact together). With one document they commit in a single write, so a half-applied event cannot exist.
2. A replay starts from one value. Splitting the world across tables would need a consistent multi-table snapshot to replay from.
3. The world is small (six characters, five places, about 50 numbers), far below Convex's 1 MB document and 1024-fields-per-object limits. Rumor facts, the only part that can grow without limit, are capped at the 20 newest.
4. Scenes and predictions (PRs 3 and 4) are logs, not world state, so they get their own tables when they arrive. Beliefs (the spec's `simBeliefs`) go into the same document when PR 3 needs them.

## Files changed

- `services/orion-ai-town/overlay/convex/sim/core.ts`: the pure simulation. It has types, the generator, `createState`, the effect set, `applyEvent`, conditions, the template table, `tick` and `checkEntities`. It has no imports and uses only erasable TypeScript.
- `services/orion-ai-town/overlay/convex/sim/schema.ts`: the `simState` and `simEvents` tables.
- `services/orion-ai-town/overlay/convex/sim/store.ts`: shared helpers (default world, load, create from the seed, save, log).
- `services/orion-ai-town/overlay/convex/sim/tick.ts`: the cron mutation `run`. It does nothing unless the world is running, or `force: true` is passed.
- `services/orion-ai-town/overlay/convex/sim/events.ts`: the `inject` mutation, Juniper's way in.
- `services/orion-ai-town/overlay/convex/sim/state.ts`: `snapshot` (query) and `bootstrap` (create or reset from the seed).
- `services/orion-ai-town/overlay/convex/sim/seed.ts`: generated from the redacted example cards.
- `services/orion-ai-town/patches/orion-sim-core-hooks.patch`: registers the tables and the cron.
- `services/orion-ai-town/scripts/apply_upstream_patches.sh`: registers the hook patch last, copies `overlay/`, and renders the seed from the cards in use.
- `services/orion-ai-town/scripts/pre_rebuild.sh`: deletes overlay files and overlay-only directories on reset.
- `services/orion-ai-town/scripts/generate_descriptions.py`: `render_sim_seed`, `--render-sim-seed`, and an example-only refresh of the tracked seed. Local cards without `sim:` blocks borrow the example's.
- `services/orion-ai-town/cards/town_cards.yaml`: `sim:` blocks and `sim_world:`.
- `services/orion-ai-town/tests/test_sim_core.py` and `tests/fixtures/sim_core_driver.ts`: new core tests (Node with type stripping) plus seed and generator tests.
- `services/orion-ai-town/tests/test_live_turn_roles_patch.py`, `tests/test_pre_rebuild.py`: the chain tests copy `overlay/` and check that it lands and resets.
- `.github/workflows/aitown-patch-chain.yml`: runs `test_sim_core.py`, and triggers on `overlay/**` and `pre_rebuild.sh`.
- `services/orion-ai-town/README.md`: a new "Living world: simulation core" section, and the `pre_rebuild.sh` steps.

## Schema / bus / API changes

- Added: Convex tables `simState`, `simEvents`. Internal Convex functions `sim/tick:run`, `sim/events:inject`, `sim/state:snapshot`, `sim/state:bootstrap`. The cron `sim tick` (every 5 minutes).
- Removed: none.
- Renamed: none.
- Behavior changed: none for existing town behavior. Nothing reads the sim state yet (scenes arrive in PR 3).
- Compatibility notes: no bus channels, no `orion/schemas` models, no Postgres. Those arrive in PR 4 per the design. The tables are additive, and Convex creates them on push.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no env template changed).
- skipped keys requiring operator action: none.

## Tests run

```text
$ AITOWN_UPSTREAM_GIT=/mnt/scripts/Orion-Sapienform/services/orion-ai-town/upstream AITOWN_REQUIRE_TOOLS=1 \
  AITOWN_TEST_NODE=<node 24 wrapper> python -m pytest -q services/orion-ai-town/tests/test_live_turn_roles_patch.py \
  services/orion-ai-town/tests/test_pre_rebuild.py services/orion-ai-town/tests/test_sim_core.py
32 passed in 14.19s     (after review fixes; was 23 before)

$ python -m pytest -q services/orion-ai-town/tests            # no node / upstream: tool tests skip
148 passed, 24 skipped in 19.40s

$ python scripts/check_personal_terms.py
no personal terms in tracked files
$ python -m pytest -q tests/test_no_personal_terms.py tests/test_gitignore_no_bare_catchall.py
10 passed
```

Mutation check (each mutant run against the same driver): if the generator is replaced by `Math.random`, replay and stepped equality both fail. If the sorted-key iteration is replaced by insertion order, stepped equality fails, which is the Convex save hazard. If trust clamping is removed, the clamp assertion fails (trust reads 2 and -2).

## Evals run

```text
None. The service has no eval harness for the simulation yet. A1 is a deterministic acceptance test (above).
A4 (a live 24h run: results applied, a quest resolves, a false belief spreads) needs PR 3's scenes and beliefs.
```

## Docker/build/smoke checks

```text
Typecheck against a fully patched upstream (scratch clone at the pin, all 19 patches + overlay applied by
pre_rebuild.sh, node_modules from the implementation worktree, TypeScript 5.1.3):
$ node_modules/.bin/convex dev --once      # against a throwaway local convex-backend container
  ... [+] simEvents.worldMinute, [+] simEvents.worldSource, [+] simState.worldId
  ✔ Convex functions ready!                # real codegen + Convex's own typecheck + push
$ node_modules/.bin/tsc --noEmit -p convex  -> exit 0
Planted error in crons.ts ({ minutes: 'five' }) -> TS2322, exit 2; reverted -> exit 0
Planted error in events.ts (status: 5)          -> TS2322, exit 2
simTables dropped from convex/schema.ts         -> TS2339 on simState reads, exit 2

After the review fixes, the same patched tree (pre_rebuild.sh from scratch, sim modules added to
api.d.ts the way the earlier real codegen listed them): tsc --noEmit -p convex -> exit 0;
planted checkInjectSource(input.source, 1) in events.ts -> TS2554, exit 2.
The scratch-backend smoke below ran before the review fixes and was not re-run (no live Convex calls).

Runtime smoke on that scratch backend (ephemeral container, stopped afterwards; not the live town):
init, then testing:stop
sim/events:inject juniper stock_delta   -> {"accepted": true, "eventId": "e1", "status": "applied"}
sim/events:inject unknown op            -> {"accepted": false, "reason": "unknown effect op teleport"}
sim/events:inject source orion          -> Uncaught Error: unknown source orion (allowed: sim, juniper)
sim/tick:run {} while stopped           -> {"ticked": false, "reason": "world is stoppedByDeveloper"}
sim/tick:run {"minutes":1440,"force":true} -> {"ticked": true, "minute": 1440, "events": 31}
sim/state:snapshot                      -> 33 rows: 31 sim, 2 juniper (e1 applied, one rejected with reason)
stored state == local Node replay (createState + same event + tick 1440): true
quest with 60-minute deadline + forced 60-minute tick -> status failed, penalty trust -0.1,
  quest_resolved event logged
sim/state:bootstrap {"replace": true}  -> clock and event numbering carried on (minute 1500, next e35)
```

## Review findings fixed

Code review by the orchestrating agent. Every finding is fixed in `2add77bce`.

- Finding (HIGH): a quest with 16 reward or penalty effects resolves as 17 effects (the tick adds `quest.<id>`). `applyEvent` throws inside the tick, so the cron would fail forever.
  - Fix: reward and penalty are capped at `MAX_EFFECTS - 1` when the quest is created.
  - Evidence: `test_biggest_quest_resolves_both_ways`. A 15-effect quest succeeds (16 effects logged) and another fails. 16 is refused ("at most 15").
- Finding (HIGH): unknown keys on effects and conditions survived validation. They would break the strict `simEffect` insert, which wedges `tick.run` during a quest resolution.
  - Fix: `normEffect` and `normCondition` rebuild each one as exactly `{op, target, value}` and `{kind, target, cmp, value}` at the entry. Only the rebuilt values are applied, stored and logged.
  - Evidence: `test_stored_shapes_match_the_simevents_validator`. It reads the field list from `schema.ts` and checks the event, goal, reward and resolved-event shapes against it.
- Finding (HIGH): defence in depth. One bad quest must never wedge the tick.
  - Fix: each quest resolution is wrapped. On any error, the quest becomes `abandoned` with `error` set to the reason, a `rejected` row (`status`, `reason`) is logged, and the tick continues. A non-object quest entry is dropped.
  - Evidence: `test_a_corrupted_quest_does_not_stop_the_tick`. A quest with a corrupted reward, plus a `null` quest, sits next to a good quest in the same tick. The bad one is abandoned with "unknown effect op teleport", the null one is removed, the good one succeeds with its reward, and a later 24-hour tick runs.
- Finding (MEDIUM): state growth.
  - Fix: past each limit, new keys are refused with a SimError, never evicted. Limits: 100 plain facts ("fact limit"), 16 flags per place, 10 open quests. The newest 20 resolved quests are kept, and their `quest.<id>` facts are pruned with them. `rumor.*` and `quest.*` are reserved for the sim.
  - Evidence: `test_growth_stays_bounded`. Over 30 game-days, 360 quests are tried (300 opened, the rest refused by the open quest limit) and 300 facts are tried (100 kept, the rest refused). Ending state: 20 quests, 100 plain facts, 20 rumors, 20 quest facts, all matching kept quests. State size 9.7 KB.
- Finding (MEDIUM): `inject` accepted source `sim`.
  - Fix: `core.checkInjectSource` allows only `juniper`. `events.ts` calls it before anything else, so a refused source writes no row.
  - Evidence: `test_only_juniper_can_be_injected`.
- Finding (MEDIUM): the pause and drift wording was wrong, and `minutes` was uncapped.
  - Fix: the `tick.ts` comment and the README now say that game time pauses with the world, does not catch up, and drifts from wall-clock hours after any pause. `core.tick` refuses more than 1440 minutes.
  - Evidence: `test_tick_takes_whole_steps_up_to_a_day`.
- Finding (LOW): malformed quest input raised TypeError, not SimError.
  - Fix: type checks on the quest object, the goal items, condition targets, and the reward and penalty lists.
  - Evidence: `test_malformed_quest_input_is_a_sim_error` (8 cases, all SimError).
- Finding (LOW): targets like `constructor`, `__proto__` and `toString` could reach the prototype, and a stock target without an item part was accepted.
  - Fix: own-property checks in `readTarget` and on need names. Stock conditions need `<owner>.<item>`.
  - Evidence: `proto_*` cases in the rejection and malformed tests, plus `stock_no_item`.
- Finding (LOW): `stock_delta` could create any key, for example `diner.cash`.
  - Fix: a new key is allowed only for an item that owner kind already holds. Places hold coffee and parts. Residents hold cash.
  - Evidence: `diner.cash` and `cam_lin.parts` are refused, `workshop.coffee` is allowed.
- Finding (LOW): the comment claiming `tick(s,60) == 12 x tick(s,5)` was wrong, because `version` differs.
  - Fix: the comment now names the version difference. The test already excluded `version`.
- Finding (LOW): `work_end: 24` could make a resident unpaid.
  - Fix: schedule hours must be 0..23. The seed's `sleep: 24` values became 23.
  - Evidence: `test_schedules_are_hours_0_to_23_and_every_wage_pays`. Hour 24 is refused, and Mara, Nico and Cam are all paid in a day.
- Finding (LOW): a seed render failure left the copied example `seed.ts` behind.
  - Fix: `apply_upstream_patches.sh` removes it before exiting.
  - Evidence: `test_seed_render_failure_leaves_no_example_seed` (a broken local card makes the script exit 1, with no `seed.ts` left).
- Finding (keyword cathedral): `purpose_weights.persuade`, `home` and the `shipment.coffee_late` fact were never read. The `maps`, `flyers` and resident `parts` stock and `square.flyers` were never moved.
  - Fix: all of these are removed from the types, the cards, the seed and the templates. `test_seed_carries_only_fields_the_core_reads` checks that every card sim field and every weight has a reader in `core.ts`, and that place stock is only `coffee` and `parts`.
  - Kept: `checkEntities`. It is tested (`test_entity_rule`), and the PR 3 scene writer needs it to refuse invented names. It has no runtime caller in this PR.

## Restart required

```bash
# No restart. On the next post-pull rebuild, pre_rebuild.sh applies the hook patch and copies the
# overlay, and post_rebuild.sh pushes the functions (Convex creates the two tables).
# The town is stoppedByDeveloper, so the sim cron does nothing until it resumes:
#   (from services/orion-ai-town/upstream) npx convex run testing:resume
```

## Risks / concerns

- Severity: low. Concern: `apply_upstream_patches.sh` now needs python3 + PyYAML for the seed render. The human-join patch already needed both. Mitigation: the script fails loudly, the same way the human patch render does.
- Severity: low. Concern: the template rates and drift constants are hand-set (for example, a coffee delivery every 16 game-hours on average). They were tuned only so that a 72-hour run keeps stock bounded and every number moves. Mitigation: they are one table in `core.ts`. PR 3's live A4 run is the first real check.
- Severity: low. Concern: the seed is rendered into `upstream/` at apply time. A card edit takes effect on the next rebuild, and only `sim/state:bootstrap {"replace": true}` applies it to an existing world. Mitigation: documented in the README.
- Severity: low. Concern: CI does not run the Convex typecheck. It needs `node_modules` (`npm ci` of upstream). The deploy path's `convex dev --once` typechecks before it pushes. Mitigation: the evidence above. Follow-up: a cached `npm ci` typecheck job if overlay churn grows.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2978

🤖 Generated with [Claude Code](https://claude.com/claude-code)
