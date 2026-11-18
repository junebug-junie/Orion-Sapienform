## Summary

Lane A (Retro core) of [Orion plays Zelda](../specs/2026-10-22-orion-plays-zelda-design.md) (#2846).

- Orion can now read The Legend of Zelda's state from game memory: the room, Link, hearts, the sword and rupees, enemies, item drops, the room's walls, pause and death. It only reads; it never writes. (`app/zelda/observe.py`, profile `zelda-1-ram-v1`)
- Orion can be given a goal like "leave east", "enter that cave", "pick that up" or "fight that Octorok", and a fast local motor turns it into button presses frame by frame. It finds a path round walls, learns from bumping into things, and records whether its prediction came true. (`app/zelda/navigate.py`, `GoalV2`)
- Orion keeps its own map of Hyrule across sessions, built only from what it saw. (`app/zelda/atlas.py`, `zelda-atlas-v1`, `GET /sessions/{id}/atlas`)
- A server-side "leisure" sponsor can now hold the controller with no browser, but any human who grabs the controller takes it back at once. (Contract and arbiter hook only; Lane B builds the runner.)
- If Orion dies during its own play, it now picks Continue on the death menu by itself and keeps playing. It finds Continue by reading where the menu cursor is, not by counting button presses. Human play is never touched. (`death_recovered` journal row)
- Nova and every other game keep the old path unchanged.

## Outcome moved

Before: Orion could choose one of five moves every five seconds in one platformer, with no map, no rooms and no notion of a goal finishing. After: from power-on, a fixed goal sequence takes the sword from the first cave and walks two screens, by RAM evidence, in about 4 seconds of emulated play, with a `met`/`refuted` verdict per goal. This is the motor and map that the Lane B planner will drive.

## Current architecture

`services/orion-retro` ran one native libretro worker with frame authority, a human/script arbiter, the `Scripted` baseline and the v1 `Goals` mailbox (five Nova goals, five-second windows, 15 Hz), the Nova-only fingerprint-gated `Observer`, and save states. Scripts needed a live browser connection as sponsor.

## Architecture touched

- Retro service only. No bus, schema-registry, memory or cognition-pipeline changes. No new env keys.
- Worker: a Zelda session (fingerprint match **and** `RETRO_COGNITION_ENABLED=true`) builds a `ZeldaRuntime` that reads RAM every frame, updates the atlas, and drives the GoalV2 motor at one decision per frame. Status gains `zelda.observation` (no wall grid; `walls_hash` instead) and `zelda.atlas`; ready/session gain `game_profile`.
- Arbiter: `control(..., leisure=True)` marks a preemptible grant. Human acquire/takeover replaces it; a leisure sponsor can never take a port from a human. `Scripted.start` carries the flag onto the script's grant.
- Saves: a save made in a Zelda session carries `lineage_id`; loading it continues that atlas lineage.

## Files changed

- `services/orion-retro/app/zelda/observe.py`: reader, live verdict table, tile convention, controllable gate.
- `services/orion-retro/app/zelda/navigate.py`: GoalV2 mailbox + motor (A*, bump, combat reflex, end conditions, `goal_finished`).
- `services/orion-retro/app/zelda/atlas.py`: per-lineage atlas, bounded, observed-only, neighbourhood view.
- `services/orion-retro/app/zelda/runtime.py`: per-frame hook shared by worker and eval.
- `services/orion-retro/app/contracts.py`: `GoalV2`, `Prediction`, goal kinds, `LeisureControl`, `LeisureScriptControl`.
- `services/orion-retro/app/control.py`: leisure preemption hook.
- `services/orion-retro/app/scripted.py`: goals factory, per-policy interval, abort hook.
- `services/orion-retro/app/worker.py`, `session.py`, `saves.py`, `main.py`: dispatch, status, lineage, `/goal` (GoalV2 for Zelda), leisure `/control`+`/scripted`, `GET /atlas`.
- `services/orion-retro/tests/test_zelda.py`: 35 deterministic tests (synthetic RAM, no ROM).
- `services/orion-retro/evals/zelda_play.py`: native eval (acceptance checks 1-3).
- `services/orion-retro/evals/zelda_http_smoke.py`: real HTTP app + spawned worker smoke at real time.
- `services/orion-retro/README.md`: Lane A section.

## Schema / bus / API changes

- Added: `POST /sessions/{id}/goal` accepts `GoalV2` on Zelda sessions:
  `{connection_id | leisure_id (exactly one), generation, port, expected_epoch, decision_id, goal, prediction, rationale (<=512), planner_id, budget_seconds (3..30)}`.
  `goal` is one of `{kind:'exit'|'explore', dir:'N'|'S'|'E'|'W'}`, `{kind:'enter'|'collect', tile:{x:0..31, y:0..21}}`, `{kind:'fight', target:'nearest'|1..11}`, `{kind:'retreat'}`, `{kind:'wait'}`.
  `prediction` is `{kind: reach_room|enter_cave|item_gained|enemies_cleared|survive|unknown, value}`; `value` is the room id for `reach_room` and must be null otherwise.
  `decision_id` echoes `status.scripted[].cognition.attempt.decision_id`; the worker offers one when Link is controllable and no goal is active, and it expires after 30 s.
- Added: `status.scripted[].cognition = {attempt, goal, last_finished, policy_id}` for Zelda; `last_finished` is the `goal_finished` record: `decision_id, goal, prediction, planner_id, outcome, verdict (met|refuted|unknown), start_digest, end_digest, start_frame, end_frame, applied_frames, human_intervention, bumps`. Outcomes: `prediction_met | prediction_refuted | completed | death (always refuted) | interrupted:hit | interrupted:stuck | budget_expired | stopped:<reason>`.
- Added: `status.zelda = {observation, atlas: {lineage_id, rooms_known, session, disabled}}`, `status.game_profile`.
- Added: `attempt.observation.terrain`: a list of 22 strings, 32 characters each. Row 0 is the top of the play area, and column/row use the same tile grid as `tile`. The characters are `#` blocked, `.` walkable, `x` a learned-blocked tile (a bump recorded on this exact room layout, matched by `walls_hash`), and `C` a cave this lineage has entered (overworld only). It is about 790 bytes as JSON. Only the offered decision carries it; `status.zelda.observation` stays grid-free.
- Added: Each `applied` journal row carries `controller: human|script|leisure`. It names the owner of that input's grant epoch: `script` is a browser-sponsored policy, and `leisure` is a leisure sponsor's policy or lease. `zelda.atlas.session.frames_by_controller = {human, script, leisure, none}` counts native frames by who held port 0 on that frame. `none` means no live lease. The atlas session row stores the same counts, next to `rooms_new` and `deaths`. Lane C derives the control share from these counts ("played together" when human/total > 0.5).
- Added: journal row `death_recovered {decision_id | null, room, level, death_frame, room_after, frame}` and `observation.menu = {kind: 'continue', cursor: continue|save|retry, activated}` (mode 8 only).
- Added: `GET /sessions/{id}/atlas` -> `{lineage_id, level, room, current, adjacent{N,S,E,W:{room, known, seen_exit}}, rooms_known, rooms_with_unexplored_exits, caves, deaths, map}`; 404 `no_atlas` / `atlas_not_written_yet`.
- Added: `POST /control` and `/scripted` accept `leisure_id` instead of `connection_id` (Zelda sessions only; `leisure_requires_zelda` otherwise). Leisure control actions: `acquire`, `release`.
- Added: save metadata `lineage_id` (optional; only Zelda saves).
- Removed / renamed: none.
- Behavior changed: none for Nova or other games. Worker `runtime_sha256` now hashes `app/**/*.py` (it includes the new subpackage), so its value changes for every game, Nova included. Evidence fingerprints from before and after this PR will not match.
- Compatibility notes: additive. The Hub does not yet call any of this; Lane B builds against these shapes.

## Env/config changes

- Added keys: none. Removed: none. Renamed: none.
- `.env_example` updated: not needed.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template changed).
- skipped keys requiring operator action: none.
- The atlas lives under the existing `RETRO_SAVES_DIR` (`/saves/atlas/`).

## Metric quality gate (observation fields)

1. **Provenance.** Each field is a direct RAM address from aldonunez/zelda1-disassembly (`Variables.inc`, `Z_01/Z_05/Z_07.asm`), read in `observe.decode`. Live probes with real controller input: the earlier probe (mode, room, Link, hearts ladder, invincibility, sword, enemies, death, walls), plus this patch's eval/settlement runs: `$E0` pause 0->1->0 on SELECT; `$66D` rupees 0->1 on a type-`$60` drop pickup in room `$78`, and it stayed; cave submodes 2..8 and person state `$AD` 0/1/3/7 freeze Link; `$506` item lift 127->0 freezes Link.
2. **Independence.** Every field comes from one RAM snapshot per frame. Verdicts are comparisons of those fields. Nothing is summed into a cognition metric.
3. **Theory.** Direct readings of program state (the game's own variables and its own collision table), not inferred reward.
4. **Rest state.** Standing still reads stable x/y, mode play, no heart change, `controllable: true`; the motor's mask is 0 with no accepted goal (tested). Death has its own mode. Bombs, keys, candle, raft and dungeon level 1-9 had no live probe (no bomb/key drop in a 15-room farming probe; no dungeon entered), so they read `unknown`, never 0.
5. **Existing mechanism.** The Nova `Observer` is the only precedent; Zelda uses the same fingerprint gate and the same two pinned core hashes.
6. **Reversibility.** Service-local. The atlas is a deletable file per lineage. Turning `RETRO_COGNITION_ENABLED` off restores the v1 path for Zelda too.

## Tests run

```text
/tmp/orion-retro-service-venv/bin/python -m pytest services/orion-retro/tests tests/retro -q
152 passed (38 new tests in tests/test_zelda.py, some parametrized)
git diff --check                                   clean
python -m pytest tests/test_no_personal_terms.py tests/test_no_private_eval_artifacts.py tests/test_gitignore_no_bare_catchall.py -q
13 passed
python scripts/check_metric_lineage.py --gate      PASS
python scripts/check_definition_drift.py --gate    PASS
```

Mutation spot-checks: removing the fight/retreat hit exemption, the leisure preemption rule, the bump detector, or cave state 0 from the freeze set each fails at least one test.

## Evals run

```text
python services/orion-retro/evals/zelda_play.py --core <fceumm ac0d00a7> --rom <PRG0>   OK 13/13 (~5 s)
python services/orion-retro/evals/zelda_play.py --core <fceumm ac0d00a7> --rom <PRG1>   OK 13/13
  control share: every applied row tagged script; frames_by_controller
  {human 0, script 5732, leisure 0, none 490} sums to the 6222 frames played (none = boot, before hand-off)
  live terrain: start room shows the entered cave as C at (8,2); in the cave the bumped barrier row is x
  boot reaches play at room $77; SELECT pause reads $E0 and freezes control;
  enter cave (met); bump at the cave barrier marks learned-blocked tiles;
  collect sword ($657 0->1, met); exit S to $77 (met); exit E to $78 (met);
  fight kills an Octorok (slot gone);
  forced death: Link stands among Octoroks, dies in $78 at frame 5715 (wait goal: outcome death,
  verdict refuted); menu cursor ($13) reads continue -> START -> activated; play resumes in $77 at
  frame 6222 with 3 hearts; death_recovered names the dead goal's decision_id; a new decision is offered;
  atlas session 1 records $77, $78, the cave and its sword;
  save-state continuity (sword=1, room $77); session 2 on the same lineage: $77/$78 not new, $68 new.
python services/orion-retro/evals/zelda_http_smoke.py --core <so> --catalog <catalog.json>  OK
  real HTTP app + spawned worker at real time: enter/sword/exit S all met, idempotent retry 200,
  invalid goal 422, GET /atlas 200, save carries the atlas lineage, leisure acquire + script start,
  human acquire preempts it (preempted_leisure: true, policy stopped human_takeover).
Evidence (rerun after review fixes): /tmp/zelda-play/{report.json, atlas.json, journal_session_1.json, journal_session_2.json,
          http_smoke.json, settlement_probe.json, settlement_probe.py, prg1/}
```

No LLM ran. Acceptance check 4 (real planner) is Lane B and waits for the cabinet heat line.

## Docker/build/smoke checks

```text
docker build -f services/orion-retro/Dockerfile -t orion-retro:zelda-lane-a-check .   exit 0
docker run --rm --network none orion-retro:zelda-lane-a-check python -c "..."
  core in CORES: True   (the container's own FCEUmm build matches the Zelda fingerprint set)
scripts/safe_docker_build.sh orion-retro config   not run: the worktree has no local .env (gitignored)
```

## Review findings fixed

A review-only subagent (orion-repo-agent) reviewed `origin/main...HEAD`. It found 1 blocker, 9 should-fix items and 9 nits. All are fixed except the items listed at the end.

- Finding (blocker): the motor sent input only every other frame. Its interval was 1/60 s, and the NES frame is shorter (60.0988 fps). About half the sword swings never reached the game, and `applied_frames` counted masks that were never sent.
  - Fix: GoalV2 policies send on every worker tick. `applied_frames` now counts only the inputs the arbiter really applied, through `Scripted.annotate` -> `ZeldaGoals.note_applied`. The eval harness now uses the same order as the worker.
  - Evidence: `test_motor_sends_every_native_frame_and_counts_only_applied_inputs` drives `Scripted.tick` at 60.0988 fps and gets one send per frame. Eval applied counts now drop on scroll frames (for example exit 55 of 153 frames).
- Finding: the atlas endpoint swapped in level 0 or room -1 when either was unknown.
  - Fix: `neighbourhood` returns `level: 'unknown'`, `current: null`, `adjacent: {}`.
  - Evidence: `test_atlas_never_invents_exits_lineages_or_places`.
- Finding: `reach_room` could be met by a dungeon room with the same id.
  - Fix: a level change refutes the prediction.
  - Evidence: `test_reach_room_across_a_level_change_is_refuted_not_met`.
- Finding: `fight` and `enemies_cleared` could be met at once in a room with no enemies yet.
  - Fix: `accept` refuses them with `no_enemies_present`.
  - Evidence: a regression test.
- Finding: a goal could hang when observations stayed unknown.
  - Fix: the deadline check now runs before the wait for the first good frame.
  - Evidence: a regression test.
- Finding: a bad `lineage_id` in save metadata, or a malformed atlas file, could kill the worker. A failed load could also leave a goal driving the replaced game.
  - Fix: the lineage is validated in `Atlas.open` and in `on_load`. `load()` checks the document shape. `flush` catches path errors. The script is cancelled before the load.
  - Evidence: regression tests.
- Finding: an exit was recorded from any change to an adjacent room (a respawn, for example).
  - Fix: an exit is recorded only after a seen scroll whose `$EC` names the new room.
  - Evidence: a respawn test. The atlas tests now use real scroll frames.
- Finding: learned-blocked tiles were permanent across layout changes.
  - Fix: each tile is keyed to the room's `walls_hash`, so a changed layout never inherits it.
  - Evidence: the bump test asserts that another layout gets no learned tiles.
- Finding: an idle policy wrote an input row every frame.
  - Fix: while no goal is active and the mask is neutral, the motor refreshes the hold without a new input.
  - Evidence: `test_idle_motor_sends_no_rows_and_leisure_without_goals_stops`.
- Finding: a lost leisure sponsor was never detected, and a leisure caller could stop another sponsor's script.
  - Fix: a leisure-sponsored policy stops with `leisure_sponsor_lost` after two offer windows (60 s) with no goal. A leisure stop of another sponsor's script returns `not_script_sponsor`.
  - Evidence: a regression test (sponsor loss). The stop check has no test of its own.
- Nits fixed:
  - Leisure routing now decides on the value of `leisure_id`, not on whether the key is present, and parses the body once (also catching RecursionError).
  - Decisions are counted in the right lineage on a load.
  - `start_frame` is reset when a goal is accepted.
  - Port 1 is refused for Zelda.
  - The atlas file is read off the event loop.
  - `caves` and `deaths` are filtered to the current level. `rooms_with_unexplored_exits` is renamed `rooms_with_unvisited_sides`, because the atlas cannot tell an unexplored exit from a wall.
  - The size trim is bounded.
  - An identity mismatch starts a new lineage and never overwrites the old file.
  - The `runtime_sha256` change is called out above.
- Not fixed:
  - Retrying a commit after its goal finished returns `stale_decision`, not the earlier result. This is a nit.
  - Native eval session 2 restores the core directly. It does not go through the worker's `load()`. That path is covered by the worker unit test (the save carries `lineage_id`) and by the HTTP smoke (save lineage equals atlas lineage).

## Restart required

```bash
scripts/safe_docker_build.sh orion-retro up -d --build
```

Run from the clean primary `main` checkout after merge. Not deployed by this PR.

## Risks / concerns

- Severity: medium. Concern: combat is crude. In the eval Link sometimes loses hearts or dies fighting several Octoroks with three hearts. Mitigation: the eval targets one Octorok slot; `fight` ends on death with `refuted`; Lane B's planner chooses `retreat`. Tuning is follow-up work, not a contract change.
- Severity: low. Concern: death recovery always chooses Continue. Save and Retry are never chosen. The live run went straight to Continue; the SELECT path (cursor on Save or Retry) is covered only by a deterministic test. Mitigation: the cursor is read from `$13` every frame, so a wrong count cannot misselect.
- Severity: low. Concern: the coordinator's follow-up named `$16` as the cursor. The disassembly shows `$16` is the file-select slot; the Continue menu uses `GameSubmode $13` (`Z_05.asm` `UpdateMode8ContinueQuestion_Full`), and the live run confirmed it.
- Severity: low. Concern: the observation has no list of cave items (the cave sword is not an object). The planner needs the tile for `collect`. Mitigation: `drops` covers enemy drops; cave contents stay UNSETTLED.
- Severity: low. Concern: bombs, keys, candle, raft and dungeon level read `unknown` until a live probe settles them.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2851

🤖 Generated with [Claude Code](https://claude.com/claude-code)
