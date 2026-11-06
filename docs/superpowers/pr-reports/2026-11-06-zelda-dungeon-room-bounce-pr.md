# fix(hub/retro): solo Zelda stops bouncing between two dungeon rooms

## Summary

Orion's first two real solo Zelda sessions (2026-11-06) each spent 30 minutes walking between the same two rooms. The cause is that inside a dungeon nothing remembered which rooms Orion had already seen.

- Inside a dungeon, Retro cannot tell which dungeon Link is in. That number in game memory has not been checked by a live probe yet, so Retro reports `level: 'unknown'` (`services/orion-retro/app/zelda/observe.py:40`, `:150-151`).
- Retro's atlas (Orion's map of rooms seen) skips every observation with an unknown level (`services/orion-retro/app/zelda/atlas.py:133`). For such a room it serves no neighbours, no visits and no crossed sides (`atlas.py:73-76`).
- Hub's move menu (`services/orion-hub/scripts/retro_experience.py`) then sees every side as "never visited". A never-visited side is worth the full 1.0 (`option_for_goal`, `:212-213`), so both sides of room 36 score the same. The tie is broken by N-S-E-W order (`build_menu`, `:331`), so north always wins. Room 20 has only a south door. The result is 36 -> N -> 20 -> S -> 36, forever. The planner model was told to take option 1 and did so every time (`chosen_rank=1` on every logged bet).
- With no neighbour room, the move cannot predict a room either, so every prediction was `unknown` (`Option.default_plan`, `:86-91`). That is why 0 of 676 predictions were graded.
- Fix: Hub now remembers, for the current session and the current dungeon only, which rooms Link arrived in and which sides the last goal walked through (`retro_planner.RoomMemory`). It fills the atlas view with that memory only while Retro reports an unknown level. A visited room is then worth less (1/visits), so Orion leaves by the side it has not tried. Each exit now predicts `reach_room <id>`, which Retro can grade.
- Leisure counts every decision Retro offers. Co-play counts only the decisions it plans.

## Outcome moved

- Replaying the recorded decisions: back in room 36 at decision 2:3, the menu used to rank "exit N -> room 20" first. It now ranks "exit S -> room 52 (never visited)" first, with prediction `reach_room 52`. "exit N -> room 20 (visited 1 time)" drops from score 0.973 to 0.290.
- Closed loop in a synthetic 2 x 3 dungeon, taking option 1 each time as the live model did: before, 24 decisions visit 2 rooms; after, all 6 rooms.
- UNVERIFIED live: no solo session has run on this code yet. The next solo session in a dungeon should show `rooms` other than 36/20 and some `reach_room` predictions graded `met`/`refuted`.

## Live evidence (root cause)

- `retro_play_sessions`: `3bf6973a` (00:48-01:18Z) and `0d7e21de` (03:48-04:18Z), both `end_reason=session_cap`, `rooms_new=0`, decisions 337 and 339, predictions unknown 337 and 339.
- Retro evidence `~/.local/share/orion/retro/service/evidence/3bf6973a-....jsonl`: all 337 `decision_offered` rows have `level: 'unknown'`, `mode: 'play'`. 169 are in room 36 (`open_sides` N, S) with goal exit N. 168 are in room 20 (`open_sides` S only) with goal exit S.
- Hub log for `0d7e21de`: every bet is `action_bet_placed ... skill=traverse place=None chosen_rank=1 p=0.959..`, with the same p for N and S. The menu existed, and its top pick was the bounce.
- The live atlas file (`~/.local/share/orion/retro/service/saves/atlas/6bab9d0f69c8414d820493617d11bc61.json`) holds 149 rooms, all level 0. `neighbourhood(doc, 'unknown', 36)` returns `adjacent: {}` and `current: null`. The prompt therefore said "this room unknown:36: first visit (not in the atlas yet)" every time, and "rooms known on this level: 125" (an overworld count).
- Candidates checked: "no memory of visited rooms" is the cause. "Novelty resets on room change" does not apply: there was no novelty to reset. "Tie-break picks the reverse door" is a side effect: the tie-break is a fixed N-S-E-W order, not a reverse-door rule. "Stale room id" was ruled out: room ids were correct (36 and 20, a real N/S pair on the grid). "Predictions all unknown" is a symptom of the same missing neighbour, not a separate cause.
- Co-play comparison: the 10-30 co-play `e17c02c5` was also inside a dungeon (37 decisions, all `level: 'unknown'`). It made no exits (33 waits in room 85, 4 fights in room 1), so it never hit this tie.

## Current architecture

Retro offers a decision on each room entry and after each finished goal. Hub reads Retro's atlas view (`GET /sessions/{id}/atlas`), builds and quotes a move menu, and asks the planner model. In leisure (`scripts/retro_leisure.py` `_plan`) and co-play (`scripts/retro_routes.py` `plan_v2`) the atlas went straight from Retro into the menu and the prompt.

## Architecture touched

Hub only (`services/orion-hub`). No change to Retro, the bus, schemas, env or Docker.

## Files changed

- `services/orion-hub/scripts/retro_planner.py`: `RoomMemory` (`observe`, `fill`, `atlas_for`) and the grid helpers. `atlas_neighbourhood` says when the view is this session's dungeon memory.
- `services/orion-hub/scripts/retro_leisure.py`: `_Run.rooms`. The poll loop counts every offered decision. `_plan` fills the atlas before the menu and the planner.
- `services/orion-hub/scripts/retro_routes.py`: co-play `plan_v2` fills the atlas the same way.
- `services/orion-hub/tests/retro/test_room_memory.py`: a regression test from the recorded decisions, reset rules, the walked-side rule, a closed loop, and leisure and co-play wiring.
- `services/orion-hub/evals/zelda_dungeon_bounce_replay.py` + `evals/fixtures/zelda_dungeon_bounce_3bf6973a.json`: a replay eval on the recorded session with orion-prediction's real model. The fixture is a read-only export of decisions 2:1-2:6 and the real atlas view.
- `services/orion-hub/README.md`: the dungeon room memory paragraph.
- `docs/superpowers/pr-reports/2026-11-06-zelda-dungeon-room-bounce-pr.md`: this report.

## Schema / bus / API changes

- Added: none. The atlas view dict Hub passes internally gains a `session_memory: true` key when filled. That dict is not published anywhere.
- Removed / Renamed: none.
- Behavior changed: inside a dungeon, traverse options carry the neighbour room, so `reach_room` predictions are sent to Retro. Retro already grades `reach_room` within the same level, `'unknown' == 'unknown'` included (`services/orion-retro/app/zelda/navigate.py` `judge`).
- Compatibility notes: overworld (level 0) is untouched. `fill` returns Retro's atlas object unchanged whenever Retro sends an integer level.

## Env/config changes

- Added keys / Removed keys / Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced: not needed.
- skipped keys requiring operator action: none.

## Tests run

```text
python -m pytest services/orion-hub/tests/retro -q          -> 231 passed
python -m pytest services/orion-hub/tests/retro/test_room_memory.py -q (fill() forced to return Retro's atlas) -> 5 failed: the tests catch the bug
```

## Evals run

```text
PYTHONPATH=. .venv/bin/python services/orion-hub/evals/zelda_dungeon_bounce_replay.py
ok   bug reproduces: without memory 2:3 ranks exit N first: first=exit:N
ok   with memory 2:3 ranks exit S first, predicting reach_room 52: first=exit:S
ok   closed loop without memory bounces between 2 rooms: rooms=[20, 36]
ok   closed loop with memory visits all 6 rooms: rooms=[20, 21, 22, 36, 37, 38]
PASS: 4/4 checks (model action_beta_backoff from services/orion-prediction/app/action_bets.py)
```

## Docker/build/smoke checks

```text
Not run: no Dockerfile, dependency, or env change. Hub code only, and Hub imports these scripts at run time.
```

## Review findings fixed

REVIEW_PLACEHOLDER

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build orion-hub
```

Run it from the clean primary `main` checkout after merge. The post-pull rebuild does this on its own.

## Risks / concerns

- Severity: low. Concern: the real fix is to settle `level_dungeon` with a live RAM probe, so that Retro's atlas records dungeons across sessions. Until then, this memory is per session: a new session starts each dungeon knowing nothing. Mitigation: once Retro sends an integer level, `RoomMemory.fill` returns Retro's atlas untouched. The class is then dead code, to delete.
- Severity: low. Concern: co-play sees only the decisions it plans, so rooms a human walks through in between are not counted. Mitigation: a side counts as crossed only when the goal that just ended walked that way, so a human walk never invents a door.
- Severity: low. Concern: a death respawns Link at the dungeon entrance. That counts as a visit, with no crossed side. Mitigation: none needed. The visit is real.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
