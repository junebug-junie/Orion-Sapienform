# Orion plays The Legend of Zelda

Status: design, approved direction (Juniper, 2026-10-22). Builds on the Retro
stack (#2796 scripted baseline, #2801 goal loop, #2825 save/load) and on the
original roadmap in #2751 (open). This is one arc delivered as a stacked set of
PRs, not a series of single-signal slices.

## Arsonist summary

Today Orion can pick one of five moves every five seconds in one platformer.
That shape cannot grow into play. Zelda needs four layers:

1. **Eyes**: a Zelda reader that turns game memory into a room-level picture
   (Link, enemies, hearts, items, the room's walls, transitions, death).
2. **Hands**: a fast local navigator that turns a goal ("leave north", "fight
   the nearest enemy", "enter the cave") into frame-by-frame input. No GPU.
3. **Map**: Orion's own atlas of Hyrule. It persists across sessions and is
   built only from what Orion observed.
4. **Mind**: a planner that runs about once per room. It reads the room and the
   atlas, then chooses a goal with a reason and a checkable prediction.

Plus: a Hub view of the map, goal and reason; solo leisure sessions when Orion
is idle and the cabinet is cool; and one memory per session so Orion can talk
about playing.

Juniper's decisions (2026-10-22):

- The planner uses the local GPU only. When the heat shed refuses, Orion does
  not choose. No Claude or CPU fallback.
- Session summaries go into Orion's real memory.
- Solo leisure is allowed.
- Deliver as a spec, then build the whole arc.

## Current architecture

- `services/orion-retro`: native libretro worker. It holds frame authority,
  the human/script arbiter, `Scripted` (#2796), the `Goals` mailbox (#2801,
  five Nova goals, five-second windows), `Observer` (Nova only,
  fingerprint-gated), and save states (#2825).
- `services/orion-hub/scripts/retro_*`: same-origin relay, `retro_planner.py`
  (one `llm.chat.request` on `metacog_background`, 20 s timeout), and the
  browser client `static/js/retro-panel.js`.
- A browser connection must sponsor every script grant. Nothing runs without
  it.
- Orion's memory gets nothing from Retro, and the allocator schedules no
  Retro act.

## Verified Zelda facts (live probe, 2026-10-22)

The probe used the pinned FCEUmm `ac0d00a7` (`ff0aa3f2…fab2`). Both operator
ROMs were tested: PRG0 `8f72dc2e98572eb4…` and PRG1 `89232edf4f9b52e3…`. Input
was real controller presses only; nothing wrote to RAM. Source:
aldonunez/zelda1-disassembly (`src/Variables.inc`, `Z_02/05/07.asm`). PRG0 and
PRG1 produced byte-identical RAM timelines over 2,857 samples.

| Field | Address | Verdict |
|---|---|---|
| game mode | `$12` (5 play, 7 scroll, `$B` cave, `$10` stairs, `$11` death, 4/6 room enter/leave, 1/`$E` menus) | CONFIRMED |
| level | `$10` (0 overworld, 1–9 dungeon) | PARTIAL (0 only) |
| Link x/y/facing | `$70`, `$84` (y includes 64 px HUD), `$98` (1 R, 2 L, 4 D, 8 U) | CONFIRMED |
| room / next room | `$EB`, `$EC` (start `$77`; east exit → `$78`) | CONFIRMED |
| hearts | `$66F` high nibble = containers−1, low nibble = full; `$670` partial | CONFIRMED |
| invincibility | `$4F0` (24 on hit, −1 per 2 frames) | CONFIRMED |
| sword | `$657` | CONFIRMED (0→1) |
| enemies | type `$34F+i`, x `$70+i`, y `$84+i`, dir `$98+i`, hp `$485+i`, slots 1–11 | CONFIRMED |
| death counter | `$E5` during mode `$11` | CONFIRMED |
| rupees/keys/bombs/candle/raft | `$66D`, `$66E`, `$658`, `$65B`, `$660` | UNTESTED |
| pause, world flags, cave source | `$E0`, `$67F…`, `$526` | UNTESTED |
| walkable tiles | save RAM (libretro region 0) offset `$530`: 32×22 column-major; tile < `ram[$34A]` walkable | CONFIRMED on start screen |

The boot sequence from power-on is in the probe script, along with its frame
counts. It must read `$16` (the menu cursor) rather than count SELECT presses,
because the cursor skips empty slots.

Lane A must settle every UNTESTED and PARTIAL row live before the reader
exposes that field. Until then, an unsettled field is reported as `unknown`,
never as 0.

## Proposed schema / API changes

All of these are Retro-service-local strict models (`extra='forbid'`) unless
marked as global. Nova keeps its v1 path unchanged.

### A. Observation `zelda-1-ram-v1` (`app/zelda/observe.py`)

```text
status: observed | unknown (+reason)
mode: play | scroll | cave | stairs | death | menu | transition | other
level, room, next_room
link: x, y, facing, invincible_frames
hearts: current (0.5 steps), containers
items: sword, (rupees, keys, bombs, candle, raft once CONFIRMED)
enemies: [ {slot, type, x, y, hp} ]  (alive slots only)
walls: 22x32 bool grid (play/cave mode only), hash
frame
```

The reader is gated on the ROM+core fingerprint, like Nova. Any other build
reads `unknown`. It reads RAM and save RAM through `retro_get_memory_data`
only and never writes.

### B. Navigator goals v2 (`app/zelda/navigate.py`, `Goals` generalised)

`GoalV2` (a new contract next to `Goal`; `Goal` stays for Nova):

```text
connection_id | leisure_id   (exactly one sponsor)
generation, port, expected_epoch, decision_id
goal: exit(dir N|S|E|W) | enter(tile x,y) | fight(target slot|nearest)
      | retreat | collect(tile) | explore(frontier dir) | wait
prediction: {kind: reach_room|enter_cave|item_gained|enemies_cleared|survive|unknown,
             value?: int}
rationale (≤512), planner_id
budget_seconds: 3..30
```

- Navigation is A* on the wall grid, at 8-px tile resolution, on a 16×16 Link
  hitbox. A move that leaves Link's position unchanged for 8+ frames marks the
  target tile as learned-blocked in the atlas. That is the "bump" signal.
- The combat reflex is part of `fight` and `retreat`: face the target, swing on
  alignment, back off on invincibility. No reflex runs outside an accepted
  goal.
- A goal ends at the first of these: prediction met, prediction refuted (room
  changed elsewhere, death), Link damaged (`interrupted:hit`), stuck 2 s
  (`interrupted:stuck`), or budget expired.
- The `goal_finished` record holds: decision_id, start/end observation digest,
  outcome, prediction verdict (`met | refuted | unknown`), applied frames, and
  human-intervention flag.
- Takeover, pause, sponsor loss and deadline-miss semantics are unchanged from
  #2796/#2801.

### C. Atlas `zelda-atlas-v1` (`app/zelda/atlas.py`)

The atlas is one JSON document per save lineage. A lineage starts with a
`save_id` or a fresh game. The file lives under `RETRO_SAVES_DIR/atlas/`,
bounded at 1 MiB.

```text
lineage_id, game fingerprint, updated_at
rooms[level:room]: first_seen, visits, exits_seen{N,S,E,W: room|null},
  caves[{tile, entered: bool, contents_seen}], learned_blocked[tile],
  enemy_types_seen[], deaths, damage_taken, last_visit
items_gained[{item, room, frame, session}]
sessions[{session_id, start, end, rooms_new, deaths, decisions, met, refuted}]
```

- The atlas is written only from observed transitions (`$EB` change with
  `$EC`), observed item deltas, and observed deaths. Nothing comes from a
  walkthrough. It is never seeded.
- The atlas stays out of the memory store. Section F exports only a summary.

### D. Planner v2 (`services/orion-hub/scripts/retro_planner.py`)

- **Trigger.** The planner runs on a room entry, when a goal finishes, after a
  death, or when the player gets stuck. Never more than one request is in
  flight. Expected rate: about 1 per 10–40 s.
- **Input.** The current observation without the wall grid: a compact text map
  of the room with walls, Link, enemies and caves. Also the atlas neighbourhood
  (this room, adjacent rooms, unexplored exits, known caves, death sites), the
  last 5 decisions with their verdicts, and the supported goals.
- **Output.** `GoalV2` JSON with a prediction. Parse failures, unsupported
  goals and goals on an `unknown` observation are all rejected. None of them
  falls back to anything.
- **Model.** Local GPU, route `metacog_background`.
- **Heat shed.** `gpu_pool_unavailable` is surfaced with `details.reason` (for
  example `shed:cabinet_hot`) and `retry_after_sec`. The motor holds `wait`.
  The planner retries with `orion/gpu_pool/shed_backoff.py`. A co-play session
  shows "Orion is waiting: cabinet hot". A leisure session ends after 3
  consecutive sheds.
- **Trace.** `planner_id` keeps the existing correlation id.

### E. Solo leisure (act-backed, no new scheduler)

- **Global contract change.** Add `play` to `ActVerb`
  (`orion/schemas/grammar.py`), with a dispatch route `retro_leisure_session`
  in `config/execution_dispatch/execution_dispatch_policy.v1.yaml` (`act:`
  block). The producer is the existing act-candidate path. The consumer is the
  Hub leisure runner. The allocator admits or refuses it like any other act. A
  refusal is stamped `allocator:<reason>` on the dispatch frame.
- **Doer-side gates.** All of these are checked at start and every 30 s.
  Failing any one refuses or ends the session, with a named reason in the
  journal:
  - arousal not `strained`, and `cabinet_hot` absent;
  - no human Retro session in the last 10 minutes and no browser connected;
  - at most one leisure session per 3 h and at most 90 minutes per day;
  - `RETRO_LEISURE_ENABLED=true` (ships on; this is the rollback lever).
- **Leisure runner** (`services/orion-hub/scripts/retro_leisure.py`).
  - It is a server-side sponsor that holds a leisure grant instead of a
    browser connection. Arbiter rule: any human acquire or takeover preempts
    leisure at once. Opening the page shows the session live.
  - It loads the lineage's latest save, or starts fresh. Then it runs the
    planner loop.
  - It saves a state at the end of every session, so the lineage carries over.
  - Maximum 30 minutes per session.
- **Grant.** The operator grant is env-configured: game_id allowlist (Zelda
  PRG0 by default), daily budget, session cap. The grant ships on.

### F. Memory export

- At session end (co-play or leisure), Hub builds one deterministic summary
  from the journal and atlas deltas:

  > Played The Legend of Zelda for 22 min, 9 new rooms (3 caves), got the
  > wooden sword, died twice near the lake east of the start, 14 of 31
  > predictions met.

  The summary is facts and counts only. It has no LLM-written narrative, and
  it is never "fun" or "learned" language without evidence.
- **Cross-day.** Write one `memory_cards` row through the existing writer,
  `orion/core/storage/memory_cards.py:insert_card`, with source
  `retro_play`. The card links to session_id and lineage_id. Recall already
  reads memory_cards.
- **Same-day.** Add a Temporal Self source, `retro_play`. It follows the
  strain-arcs pattern (commit `9508aeab4`): schema kind, adapter in
  `orion/temporal_self/sources.py`, a durable-runs SQL source reading a new
  table `retro_play_sessions` (additive migration with a REQUIRED-BY header),
  and a stance-cue render. The rollout is consumer-first (#2751 §7.3).
- Each summary records who controlled each port, and for what share of the
  frames (#2751 §7.4). A session where the human drove more than 50% of frames
  is recorded as "played together", never as Orion's achievement.
- Not exported: frames, RAM, ROM, raw journals. The summaries do not feed φ,
  arousal, reward or training.

### Proposal-mode record (memory and autonomy)

- **Capability.** Orion can choose to play when idle. Orion can recall having
  played.
- **Data.** Game observations stay local. One summary card plus one Temporal
  Self row go out per session.
- **Privacy.** Game facts only. No chat or private recall goes into the
  planner prompt.
- **Proof trace.** act admitted → leisure session → decisions with verdicts →
  atlas delta → card id → chat recall cites it.
- **Dangerous failures.** Leisure steals the cabinet from a human or heats the
  GPU. Mitigation: human preemption, heat/arousal gates, caps. A fabricated
  achievement becomes memory. Mitigation: deterministic summary from the
  journal, plus the control-share rule.
- **Disable.** Each of these switches off its own part:
  `RETRO_LEISURE_ENABLED=false`, `HUB_RETRO_MEMORY_EXPORT_ENABLED=false`, and
  the allocator route `enabled: false`. Human play keeps working.

### Metric quality gate (observation fields)

1. **Provenance.** The disassembly lines plus a live probe for each CONFIRMED
   row.
2. **Independence.** Every field comes from one RAM snapshot. Outcome verdicts
   are comparisons of those fields. None of them is summed into a cognition
   metric.
3. **Theory.** These are direct readings of program state, not inferred
   reward.
4. **Rest state.** Standing still reads as stable x/y, no hp change and mode 5.
   Death has its own mode. Lane A re-checks this per field.
5. **Existing mechanism.** The Nova Observer is the only precedent. Zelda gets
   its own module behind the same fingerprint gate.
6. **Reversibility.** All of it is service-local. The atlas is a deletable
   file, and the memory cards carry the `retro_play` source so they can be
   removed.

The arc metrics (rooms per minute, deaths per room, prediction met-rate,
time-to-known-room across sessions) are Retro-local reports. None of them is
wired into any cognition pipeline in this arc.

## Files likely to touch

- Lane A (Retro core): `services/orion-retro/app/zelda/{observe,navigate,atlas}.py`,
  `contracts.py` (`GoalV2`), `goals.py`/`scripted.py`/`worker.py`/`main.py`
  (game dispatch), `tests/`, `evals/zelda_play.py` (native: boot → sword →
  exits), README.
- Lane B (Hub play): `scripts/retro_planner.py` (v2 + shed handling),
  `scripts/retro_routes.py`, `scripts/retro_leisure.py`, `templates/retro.html`,
  `static/js/retro-panel.js` (atlas map, goal, reason, verdicts), settings and
  `.env_example`, `tests/retro`, `evals/retro_browser.mjs`.
- Lane C (Orion integration): `orion/schemas/grammar.py` (`play`),
  `config/execution_dispatch/execution_dispatch_policy.v1.yaml`,
  `orion/temporal_self/{sources,stance_cue}.py`, schema `temporal_self.py`,
  durable-runs SQL source, migration `manual_migration_*_retro_play_sessions.sql`,
  memory card export, tests and eval fixture.

Lanes B and C build against the contracts above while A builds them.
Collisions happen only at `GoalV2` and the session summary. A merges first.

## Non-goals

- Pixel perception, learned or trained policies, and speedrun competence.
- Zelda II, and any game beyond The Legend of Zelda.
- A walkthrough or a seeded map.
- Claude or CPU planner fallback.
- RAM writes or cheats.
- Rewards fed into φ, arousal or training.
- Replacing human play.
- WebRTC.

## Acceptance checks

1. **Reader (deterministic).** Fixture RAM snapshots decode to the expected
   observation. An unknown fingerprint gives `unknown`. Every field marked
   CONFIRMED has a live probe artifact.
2. **Navigator (native eval, no LLM).** From power-on, a scripted goal sequence
   (`enter` cave → `collect` sword → `exit` S → `exit` E) completes by RAM
   evidence: sword=1, room `$78`. `fight` kills an Octorok by enemy hp. A bump
   marks a learned-blocked tile.
3. **Atlas.** Two sessions on one lineage. In session 2, the rooms already
   known are not "new", and the save-state continuity holds.
4. **Planner (real, local GPU).** In one ≥10-minute co-play or leisure session,
   from a fresh start:
   - sword gained;
   - at least 8 new overworld rooms;
   - every goal comes from a real planner response;
   - prediction verdicts are recorded;
   - BLOCKED while `cabinet_hot`, recorded with the shed reason. Never faked.
5. **Leisure.** Each refusal, by arousal/heat, human-present, cap and switch,
   is tested deterministically and appears in the journal. A live human
   acquire preempts a leisure session within one frame of the arbiter.
6. **Memory.** A session card exists with the `retro_play` source and a
   correct control share. Same-day, the Temporal Self cue renders it. A chat
   question ("did you play anything today?") cites it. The live check is
   UNVERIFIED until shown.
7. **Learning signal (arc-level, reported, not gated).** Session 2 on the same
   lineage needs fewer decisions to reach a known room than session 1.

## Recommended next patch

Lane A first: the reader with live settlement of the UNTESTED fields, plus the
navigator and atlas with a native eval that needs no LLM. Lanes B and C run in
parallel against the contracts above. The real planner acceptance (check 4)
waits for the cabinet to drop below the `cabinet_hot` clear line.
