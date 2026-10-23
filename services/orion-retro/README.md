# Retro headless authority (P1)

A persistent, operator-only NES service. One spawned child owns the libretro core;
HTTP/WebSocket handling never calls the native core. Browser access is provided by the separate Hub adapter below. The native service has no bus connection or global event schemas. Optional
Hub goal planning runs outside the native process; see the bounded goal experiment below.

The fixed catalog is imported from operator-owned local kits, including Pad Lab,
Nova and Spacegulls when installed. ROMs, credentials, saves and evidence are never
committed. The API cannot download a ROM, select a host path or upload native code.
Save states are enabled when the `RETRO_SAVES_DIR` host directory is mounted
writable at `/saves` (see "Save states" below). Nova's strict serialized-state
replay fails with the pinned core (a load resumes play, but not bit-identical
replay); Spacegulls passed the existing short replay check. Neither claim
establishes game competence.

## Local setup and deployment

After merging, from clean primary main synchronized with origin/main:

```sh
python3 scripts/sync_local_env_from_example.py
python3 services/orion-retro/evals/install_catalog.py --env services/orion-retro/.env
scripts/safe_docker_build.sh orion-retro up -d --build
```

The installer writes operator-only local catalog/evidence directories and a random
credential below `~/.local/share/orion/retro/service`; preserves existing credentials
and nonempty env overrides; enables this explicitly requested installation. Default
`.env_example` is disabled. No host emulator or system package is installed. Core
compiles in a Docker build stage pinned to FCEUmm
`ac0d00a7fc01166f84c7bac01d1084ebfd2b865e`. GPL license is retained in the image;
matching source is https://github.com/libretro/libretro-fceumm/tree/ac0d00a7fc01166f84c7bac01d1084ebfd2b865e.

The operator client uses the private Unix socket at `service/ipc/retro.sock`
(override `--socket` for isolated tests) and needs the service requirements (install in a venv). Run:

```sh
python3 services/orion-retro/evals/operator_smoke.py \
  --token-file ~/.local/share/orion/retro/service/operator.token \
  --game spacegulls --output /tmp/spacegulls-authority-smoke.json
```

This starts a fresh game, opens an authenticated WebSocket, acquires P1, sends
Start then right/fly inputs, verifies acknowledgments with actual frame numbers,
checks disconnect pauses, and stops the session. It is scripted actuation, not a
successful gameplay policy. Use `--game pad-lab` for the original test ROM.

## API and authority

All routes require `Authorization: Bearer <operator credential>`; browser Origin
requests are rejected. P1 maps that credential to a fixed `operator` principal;
request fields cannot select identities. No Hub authentication is claimed.

- GET `/health/live`: API process responds. Disabled service can still be live.
- GET `/health/ready`: catalog/core ready, plus current session/frame status.
- GET `/games`: hashes, supported ports, save capability; no filesystem paths.
- POST `/sessions`: `game_id`, `idempotency_key`; one active session, conflicts
  rejected. At most 256 keys per process; retrying a key returns its original ID.
- GET `/sessions/{id}`: lifecycle, actual frame, callback counts, core identity.
- WS `/sessions/{id}/input`: returns connection UUID and boot generation.
- POST `/sessions/{id}/control`: live connection UUID, port 0/1, action
  acquire/release/takeover. Returns connection-bound token and epoch.
- WS input: version 1, session_id, generation, lease, epoch, seq, port, mask.
  Ack contains actual applied frame. Invalid/replaced inputs receive rejection.
  If a reply is lost during timeout/shutdown, the result is `unknown`; inspect
  the retained evidence rather than assuming the action never happened.
- POST pause/resume/stop endpoints: explicit lifecycle; resume requires new leases.
- POST `/sessions/{id}/save`: worker serializes the core and writes a save state.
- POST `/sessions/{id}/load` `{save_id}`: worker restores it in place (running or paused).
- GET `/games/{game_id}/saves`: saves newest first, with frame, played frames,
  UTC time and `compatible` (ROM and core SHA-256 both match).
- POST `/sessions` with optional `save_id`: start a fresh session from a save.
- GET `/sessions/{id}/evidence`: last 256 applied inputs and local artifact ID;
  full journal remains operator-local. Tokens are never journaled.

Only NES button bits are accepted, including A=256, B=1, Start=8, Right=128.
Opposing directions are rejected. Takeover releases the old mask for one frame
before the new owner's input; old epochs and connections cannot regain authority.
Start/resume without acquiring any controller pauses after a five-second grace.
Send button changes at once and repeat the state every 50 ms. A lease expires
after 2 s without input (browsers can throttle the repeat to about 1 Hz); expiry of
the last lease or last controller disconnect pauses. UI blur handlers belong to
P2; this server requires no browser message to expire a lease.

Native worker queues and HTTP/WS ingress are bounded. Three consecutive late
frames pause, a silent/hung/crashed worker fails the session, and stop escalates
from cooperative shutdown to termination/kill of only its child. Restart marks
active persisted state failed (`service_restart`), never resumes play by itself;
start a new session from a save instead. No raw A/V recording, model invocation or memory export occurs.

## Save states

`RETRO_SAVES_DIR` (host) is bind-mounted at `/saves`. It must exist and be
writable by `RETRO_UID`; otherwise `save_enabled` is false and save/load refuse
with `saves_unavailable`. Layout: `<game_id>/<save_id>.state` (raw
`retro_serialize` bytes) and `<save_id>.json` (game, ROM/core SHA-256, frame,
played frames, UTC, state SHA-256). The JSON is written last, so a crash never
lists a half-written save.

- Only the worker process serializes, restores, writes and reads state files.
  The API process lists metadata and pre-checks identity before a load.
- A save made with a different ROM or core is refused (`save_identity_mismatch`);
  a damaged state file is refused (`save_corrupt`). Both are checked twice:
  in the API and again in the worker against the core it actually loaded.
- Disk bound: at most 4 MiB per state and 20 saves per game; each new save
  deletes the oldest beyond 20 (and any orphaned state file). An NES state is
  normally under 100 KiB, so a game uses about 2 MiB at most
  (catalog cap 64 games, so about 128 MiB worst case in practice).
- No fsync: a blocked disk must not stall the frame loop (the API fails a worker
  silent for 1 s). A save survives container restarts and rebuilds; a host power
  loss within the kernel's writeback window can lose the newest save.
- `frame` is the session's own frame clock; `played_frames` carries over through
  loads, so it counts total play time across sessions.
- Evidence records `save` (save_id, frame, played_frames, bytes) and `load`
  (save_id, frame, save_frame, played_frames) events.

## Isolation and evidence

Container: non-root UID/GID, read-only root, dropped capabilities, no-new-privileges,
512 MiB RAM, 2 CPUs, 64 PIDs, journald, bounded tmpfs. Only catalog and credential
are read-only mounts; dedicated evidence and save-state directories are writable. No Docker socket, host
credentials, production database or repo mount. Networking disabled; private Unix socket in a dedicated operator-only directory.
No published TCP ports. Deployment validation must test actual egress denial.
The container is the security boundary; a child process is fault isolation, not
protection of the API credential against a compromised native sibling process.

Evidence JSONL records lifecycle, control changes, applied actions and neutral
transitions with frame/UTC/session/core/ROM identities. Quotas: 256 MiB per session,
2 GiB across top-level evidence files. Quota/I/O failure terminates the native
session and marks failure; never evicts previous evidence automatically. Mount a
dedicated directory, not a shared data tree. No replay-completeness claim is made
for a crashed/truncated journal. Retained capability results are not cognition
metrics and are not fed into any model.

## Checks and rollback

```sh
python3 -m pytest tests/retro services/orion-retro/tests -q
python3 scripts/check_env_template_parity.py orion-retro
```

Native tests require the image/core: run the authored-ROM pad eval and preflight
inside the built image, then the operator API smoke. A production claim also
requires actual container inspect, callback/frame movement and applied-input
records; process health alone is insufficient. The runbook records outstanding
acceptance gates rather than pretending P2 is complete.

Rollback: operator stop, set local `RETRO_ENABLED=false`, recreate from main
through the safe wrapper; keep catalog and evidence. No automatic resume or
schema migrations. A stopped service does not delete game data.

## Browser adapter (P2)

Hub now supplies a browser control/media proxy using Hub’s network access boundary;
see [Retro browser runbook](../../docs/runbooks/retro-browser.md). The original CLI
credential remains separate. `/sessions/{id}/media` carries bounded transient
video/PCM packets, authenticated and owner-checked like input. Direct browser Origin
requests to this service remain forbidden; Hub proxies over the private socket.
All session reads/mutations require the credential that created the session.

## Scripted baseline (first P3 slice)

Owner-only `POST /sessions/{id}/scripted` accepts `connection_id`, `generation`,
`port`, `expected_epoch`, `action` (`start`/`stop`), and `duration_seconds`
(integer 5–1800, default 60). Start transfers the caller's current live human
lease only; stop rejects a different active script epoch. The fixed
`scripted-right-jump-v1` runs inside the native worker at at most 15 Hz with
a 100 ms hold deadline. It is an open-loop diagnostic, not learning.

Human takeover, pause, sponsor disconnect, missed deadline and budget expiry
cancel it permanently. No implicit reacquisition on resume. Status exposes
active `scripted` summaries; applied journal/history rows add `policy_id` for
script actions. Start/stop records include policy, epoch and reason; secrets
remain excluded. No bus, metrics, memory, dependencies or env keys are added.

Validated-input rejection/unknown responses now also include `port`, `epoch` and
`seq`. This is additive: existing CLI clients can still inspect `rejected` or
`unknown`. Hub ignores replies from an earlier lease. A connection that no longer
owns any port cannot pause a fresh resume merely by closing.


## Bounded goal experiment (P3/P4)

`RETRO_COGNITION_ENABLED` and Hub's `HUB_RETRO_COGNITION_ENABLED` default on;
set either to `false` to disable the experiment. Nothing runs until you choose
**Try Orion goals · 60s** in More for a live session. Start the game normally and reach the mechanic before handoff.
The verified Nova v1.0.6a/core pair supplies RAM-assisted position/health facts;
all other fingerprints are unknown. The motor supports wait, walk left/right,
and pulsed jump left/right; it does not support general goals like reach_exit.

Each acknowledged goal lasts five seconds. Planner unavailability returns the
motor to neutral/wait, and human takeover remains immediate. The UI shows the
last accepted goal/rationale and the observed outcome; no success claim is
inferred from a position change. Local evidence links observations, acknowledgments
and applied controls. Nothing is written to cognition memory or training.

The native `/goal` endpoint and Hub `/plan` endpoint are documented in the
[implementation and observation gate](../../docs/superpowers/specs/2026-10-20-retro-goal-loop-implementation.md).
The real-planner eval is `evals/cognitive_loop.py --core <pinned-core> --rom <nova>
--bus-url <tailscale-Redis-URL> --output <private-report.json>`; use
`--mechanic-only` only for a separately labeled fixture test. GPU admission
refusals fail the real-planner gate; they are never replaced with scripted replies.


## The Legend of Zelda: reader, navigator, atlas (Lane A)

Design: [Orion plays Zelda](../../docs/superpowers/specs/2026-10-22-orion-plays-zelda-design.md).
A Zelda session uses its own reader, motor and atlas only when the ROM+core
fingerprint matches (PRG0 `8f72dc2e…`, PRG1 `89232edf…`, pinned FCEUmm `ac0d00a7`)
**and** `RETRO_COGNITION_ENABLED=true`. Every other game, and Zelda with the gate
off, keeps the Nova v1 path unchanged. No new env keys.

**Reader (`app/zelda/observe.py`, `zelda-1-ram-v1`).** One RAM + save-RAM read per
frame through `retro_get_memory_data`; nothing is written. Fields: `mode`
(play/scroll/cave/stairs/death/menu/transition/other), `level`, `room`, `next_room`,
`paused`, `link` (x, y, `tile`, facing, invincible_frames, `controllable`), `hearts`
(current in 0.5 steps, containers), `items` (sword, rupees; bombs/keys/candle/raft
read `unknown` until a live probe settles them), `enemies` (alive slots with type,
x, y, tile, hp), `drops` (type-$60 item objects), `walls` (22x32, True = blocked; play
and cave only) and `walls_hash`. `level` 1-9 also reads `unknown` (no dungeon probe
yet). `VERDICTS` in the module is the live-settlement table. `controllable` is false
during transitions, cave text (`$AD` 0/1/3/7), an item lift (`$506`) and pause.
Tile convention (shared by motor, atlas and the planner's text map): column
`(x+4)//8`, row `(y+11-64+4)//8`, i.e. the game's own collision hotspot.

**Motor (`app/zelda/navigate.py`, `zelda-navigator-v1`).** Runs inside the worker at
one decision per frame. When a script starts with `cognition: true` on a Zelda
session, the worker offers a decision slot in status
(`scripted[].cognition.attempt.decision_id`) whenever Link is controllable and no
goal is active. The offered decision's observation carries the room map as
`attempt.observation.terrain`: a list of 22 strings, 32 characters each. Row 0 is the top of the play area, and column/row use the same tile grid as `tile`. The characters are `#` blocked, `.` walkable, `x` a learned-blocked tile (a bump recorded on this exact room layout, matched by `walls_hash`), and `C` a cave this lineage has entered (overworld only). It is about 790 bytes as JSON. Only the offered decision carries it; `status.zelda.observation` stays grid-free. `attempt.open_sides` lists the sides (`N`, `S`, `W`, `E`) whose screen edge Link can walk to from where he stands, by the motor's own search, counting only the atlas's learned blocks for this room (overworld play only; `null` in dungeons and other modes, where doorways are not verified against the grid). An exit can only use one of these (2026-10-23: the planner asked "explore S" eight times against the start room's solid south wall). `POST /sessions/{id}/goal` takes a `GoalV2` (below) that echoes it.
Goals: `exit{dir}`, `explore{dir}`, `enter{tile}`, `collect{tile}`,
`fight{target: slot|nearest}`, `retreat`, `wait`. A* runs on the wall grid plus
learned-blocked tiles. A press that leaves Link's position unchanged for 8 frames is
a bump: the tile is marked learned-blocked in the atlas and the path is replanned.
The combat reflex (face, swing when aligned, back off after a hit) runs only inside
`fight`/`retreat`. A goal ends at the first of: prediction met, prediction refuted
(another room), `death` (always refuted), `interrupted:hit` (not for fight/retreat), `interrupted:stuck`
(2 s), `budget_expired`, goal completed, or `stopped:<reason>` (takeover, pause,
sponsor loss, load). Each end writes one `goal_finished` journal row: decision_id,
start/end digests, outcome, verdict (`met|refuted|unknown`), applied frames, bumps,
`human_intervention`. Status shows it as `cognition.last_finished`.

**Death recovery.** This happens only while a script grant is active; human play is never touched. A death ends any active goal with outcome `death` and verdict `refuted`. On the Continue/Save/Retry screen (mode 8), the motor reads the cursor from `GameSubmode $13` (low two bits: 0 Continue, 1 Save, 2 Retry; `$80` = START accepted). It never counts presses. It taps SELECT until the cursor is on Continue, then taps START. When play returns it writes a `death_recovered` row (`decision_id` or null, death `room`/`level`, `death_frame`, `room_after`; the journal stamps `frame`) and offers a new decision. The reader exposes the screen as `observation.menu = {kind: 'continue', cursor, activated}`.

**GoalV2 body.** `connection_id` **or** `leisure_id` (exactly one), `generation`,
`port`, `expected_epoch`, `decision_id`, `goal` (discriminated by `kind`),
`prediction{kind: reach_room|enter_cave|item_gained|enemies_cleared|survive|unknown,
value: room id for reach_room only}`, `rationale` (≤512), `planner_id`,
`budget_seconds` (3-30). A retry with the same decision_id is idempotent and never
extends the budget; any change is `goal_idempotency_conflict`.

**Leisure sponsor (contract + arbiter hook only; Hub builds the runner).**
`POST /control` and `/scripted` also accept `leisure_id` instead of `connection_id`,
on Zelda sessions only. A leisure grant is preemptible: any human `acquire` or
`takeover` replaces it at once (the reply carries `preempted_leisure: true` and the
script stops with `human_takeover`). A leisure sponsor can never take a port from a
human. A session paused for `no_controller` must be resumed before a leisure acquire.

**Controller attribution.** Each `applied` journal row carries `controller: human|script|leisure`. It names the owner of that input's grant epoch: `script` is a browser-sponsored policy, and `leisure` is a leisure sponsor's policy or lease. `zelda.atlas.session.frames_by_controller = {human, script, leisure, none}` counts native frames by who held port 0 on that frame. `none` means no live lease. The atlas session row stores the same counts, next to `rooms_new` and `deaths`. Lane C derives the control share from these counts ("played together" when human/total > 0.5).

**Atlas (`app/zelda/atlas.py`, `zelda-atlas-v1`).** One JSON file per save lineage at
`RETRO_SAVES_DIR/atlas/<lineage_id>.json`, bounded at 1 MiB, written at most every 2 s
and on save/stop. It records only observed room changes (exits seen both ways),
caves entered and what was gained inside, item deltas, deaths, heart loss, enemy
types, bumps, and one session row (rooms_new, deaths, decisions, met, refuted). It is
never seeded. A save made in a Zelda session carries `lineage_id`; loading it
continues that lineage. A fresh game, or a pre-atlas save loaded mid-session, starts
a new one. A ROM/core mismatch never reuses another build's atlas. Atlas I/O faults
disable the atlas for the session; the game keeps running. `GET /sessions/{id}/atlas`
returns the current room's neighbourhood (adjacent rooms, unexplored exits, caves,
deaths, and a compact map of the level). Delete the file to forget it.

**Evals.** `evals/zelda_play.py --core <so> --rom <zelda.nes>` (no LLM, ~4 s,
simulated clock): boot from power-on with real input, pause, enter the cave, bump the
cave barrier, take the sword, exit S, exit E, kill an Octorok, die standing among the
Octoroks and recover through Continue, then a second session
on the same lineage from the save state. `evals/zelda_http_smoke.py --core <so>
--catalog <catalog.json>` drives the real HTTP app and a spawned worker at real time,
including the leisure preemption check. Both write evidence under `/tmp/zelda-play/`.
