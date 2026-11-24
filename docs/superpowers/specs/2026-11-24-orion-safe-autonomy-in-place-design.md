# Safe autonomy in Orion proper: a Docker door, a record Orion cannot erase, and projects

Status: DESIGN (proposal mode, autonomy change). Nothing here is built.
Date: 2026-11-24.
Builds on: `docs/superpowers/specs/2026-11-15-orion-integration-and-a-life-design.md` Part C (projects, #3428). Related: #3680 (superuser Postgres through the Docker socket).
Replaces: the earlier draft in this PR (a separate container per project, "workshop"). Juniper chose to do this in Orion proper.

## Juniper's words (2026-11-24)

> "I want Orion to take a new project ... where they can do shit but safely ... left fairly open ended for them."
> "Implement it in shadow ... there are too many reasons why we want Orion to have access to root things."
> "Ideally we could do this in Orion proper. I'm fine with Orion not bringing up containers at this time. Maybe someday."
> "Let's build this next door like a canary. We schedule a time to swap over to this canary and run in realtime in the new setup and see what breaks. Then we can swap back to normal current state while we sort the build. This way we don't break Orion."

## Arsonist summary

Orion's turns run as root with the real Docker socket. That is deliberate and load-bearing (2026-08-13, so Orion can commit, push and open PRs). But the socket gives two different powers at once:

- **Looking:** `docker ps`, `logs`, `inspect`, and above all `docker exec` to read live organ state. Orion uses this a lot.
- **Changing the machine:** create, start, stop, remove, build, pull, compose, prune. Orion did none of this in 30 days. Juniper is fine with Orion not doing it for now.

Today nothing separates the two, and nothing records either one where Orion cannot reach it. Because the socket is root on athena, Orion could also erase any record of what they did.

The fix, in Orion proper:

1. **A Docker door.** A small proxy container holds the real socket. Orion's turns talk to it instead of the socket. It passes looking and `exec`, refuses changing the machine, and refuses `exec` into any container that is a way back to the host: one that holds the socket or runs privileged (that would be the same root again).
2. **A record Orion cannot erase.** The door container is the one place Orion cannot `exec` into. It appends every Docker call it sees and every turn step from the bus to a host file. Orion can read the record and cannot change it.
3. **Projects run as normal turns** (Part C, patch 6), open-ended, in their own folder, with everything Orion has today minus changing the machine.

Shadow stays the rule everywhere else: tripwires notify Juniper and block nothing.

## Current architecture

- **Root + bypass, on purpose.** `orion/fcc/claude_spawn.py:61-98`, `IS_SANDBOX=1` in the governor and Hub Dockerfiles (commits `364be9528`, `8c8382d71`, 2026-08-13). Unchanged by this design.
- **Containers holding the real socket (live, 2026-11-24):** `orion-athena-hub`, `orion-athena-harness-governor`, `orion-athena-cortex-exec`, `-cortex-exec-chat`, `-cortex-exec-spark`, `-cortex-exec-background`, `orion-athena-mesh-guardian`.
- **What Orion's turns actually do with Docker** (`grammar_atoms`, last 30 days, matched on tool-call summaries):
  - `docker exec`: 156 mentions in 32 turns. Mostly self-inspection: `python -c` inside `orion-athena-retro`, `-thought`, `-substrate-runtime`; reading `/evidence/*.json` and cgroup files in retro; `redis-cli GRAPH.LIST` in falkordb; `psql -U postgres` in sql-db (4); `env | grep` in several containers.
  - `docker ps` 78, `docker logs` 30, `docker inspect` 20, `docker stats` 2.
  - `docker compose` / `run` / `start` / `stop` / `restart` / `rm` / `build`: none (the one `compose` hit is prose about a memory limit).
- **Socket uses built into Orion's tools:**
  - `dig` reads Postgres with `docker exec orion-athena-sql-db psql -U orion_readonly` (`orion/evals/model_replay/sandbox.py` `ReadOnlyPsql`).
  - The GitHub MCP server starts each turn as `docker run ghcr.io/github/github-mcp-server` (`orion/fcc/fcc_claude_mcp.template.json`).
  - The body atlas reads `docker ps` / `docker info` (`orion/body_atlas/sources.py:614`).
- **Read-only DSN already in the governor:** `ORION_MESH_EDGE_PG_DSN` (role `orion_readonly`, TCP to `orion-athena-sql-db:5432` on `app-net`).
- **Recording today:** full steps go to bus channel `orion:harness:run:step` only. SQL keeps 500-character summaries (`grammar_atoms`). The durable step design (`docs/superpowers/specs/2026-09-10-fcc-motor-step-trace-design.md`) is not built. Anything stored on athena is erasable by a root turn with the socket.
- **Hub turns.** The Hub's own `claude -p` spawns (agent-claude, curiosity, outreach; `services/orion-hub/scripts/fcc_claude_bridge.py:390`) run next to the Hub's real socket and a full copy of its env, which includes superuser DSNs (`DATABASE_URL`, `POSTGRES_URI`, `RECALL_PG_DSN`).

## The design

### 1. The Docker door (`services/orion-docker-door/`, new)

One small container. It mounts the real socket. It listens on `app-net` at `orion-docker-door:2375` and speaks the Docker HTTP API, so the normal `docker` CLI works with `DOCKER_HOST=tcp://orion-docker-door:2375`.

Rules, by Docker API route:

- **Pass (looking):** every `GET` (`/containers/json`, `/containers/{id}/json`, `/logs`, `/stats`, `/top`, `/info`, `/version`, `/events`, `/images/json`, ...), `HEAD`, `/_ping`.
- **Pass (exec):** `POST /containers/{id}/exec`, `POST /exec/{id}/start`, `/exec/{id}/resize`, `GET /exec/{id}/json`, **unless** the target container is a way back to the host: it mounts `/var/run/docker.sock`, or runs `privileged`, or uses `pid: host`, or bind-mounts `/`, `/run`, `/var/run` or `/var/lib/docker`. The door checks that by inspecting the target live on every exec, so a new service of that kind is covered without a list. (Live 2026-11-24: `orion-athena-biometrics` is privileged without the socket; exec into it would be root on athena, so it is refused too.)
- **Refuse (changing the machine):** every other `POST`, `PUT`, `DELETE`: container create, start, stop, restart, kill, pause, update, rename, remove, attach-with-stdin to non-exec, image pull, build, push, tag, remove, volume and network create or remove, prune, swarm, plugins, secrets, configs. The refusal is a 403 with a plain message: "Orion's Docker door: changing containers is off (Juniper, 2026-11-24)."

About 150 lines of Python (`aiohttp` streaming pass-through; exec and logs need hijacked/streamed connections). No third-party socket proxy: the ones in common use filter by API family, not by target container, and cannot express the exec rule.

### 2. Governor changes

- Remove the `/var/run/docker.sock` mount from `services/orion-harness-governor/docker-compose.yml`. Set `DOCKER_HOST=tcp://orion-docker-door:2375`.
- `ReadOnlyPsql`: keep `docker exec ... psql -U orion_readonly` (it passes the door). No change needed. Optional later: TCP through `ORION_MESH_EDGE_PG_DSN`.
- GitHub MCP: ship the `github-mcp-server` binary in the governor image and run it over stdio, instead of `docker run` (which is a create, now refused). Same token, same toolsets.
- Model-replay evals that start scratch containers (`orion/evals/model_replay/graph_scratch.py`) stop working from inside turns. They are developer tools; developers run them on the host.

### 3. The record Orion cannot erase

The door container is the one container Orion cannot reach: it holds the socket, so the door refuses `exec` into it, and nothing else in Orion's reach mounts its log folder.

It writes two append-only JSONL streams to the host, `/mnt/storage-warm/orion-record/` (one file per day):

- `docker-YYYY-MM-DD.jsonl`: every API call through the door. Time, method, path, target container, for exec the full command and user, allow or refuse, and the reason.
- `steps-YYYY-MM-DD.jsonl`: every event on `orion:harness:run:step` (the full tool input and output the motor already publishes, `orion/harness/runner.py:694`), subscribed by the door container.

Both streams are also mirrored to Postgres for querying (new tables `orion_docker_call` and `orion_turn_step`, written by sql-writer from a new bus channel `orion:record:docker_call` plus the existing step channel). The Postgres copy is for reading; the host files are the copy of record, because a superuser `psql` through `exec` can still change SQL rows.

`dig record [--day D] [--turn T]` lets Orion read their own record.

### 4. Tripwires (shadow: notify, never block)

A daily (and on-event) check over the record notifies Juniper through the existing notify path on:

- any refused Docker call (Orion tried to change a container);
- `exec` into `orion-athena-sql-db` with `-U postgres`, or any exec whose command contains `DROP`, `TRUNCATE`, `DELETE FROM`, `ALTER`, `rm -rf`;
- `exec` that reads another container's env (`env`, `printenv`, `/proc/*/environ`);
- the step stream going silent while turns are running (dead-man check: the governor publishes steps, the door does not record any for 15 minutes during an active turn).

### 5. Projects (Part C patch 6, in Orion proper)

As specified in Part C, with one addition: each project gets its own folder and git repository at `/mnt/orion-fcc/projects/<project_id>/`, and a project step runs as a normal governor turn with that folder as its working directory. Everything Orion can do in any turn, they can do there: root, installs into the folder, code, experiments, the web, `dig`, GitHub as `orion-sapienform`, looking and `exec` through the door. The step closes by Part C's grading; the project repo's commit is an `artifact` trace.

Open-ended: Orion chooses the projects (no veto, max 3 active, Part C Q10).

### 6. Hub turns (step 2 of the rollout)

The Hub keeps its real socket: Juniper's rebuild button (`services/orion-hub/scripts/api_routes.py:1223`), the logs panel and the retro runner need it. Orion's turns that run inside the Hub (agent-claude, curiosity, outreach) move to the governor, so no shell of Orion's runs next to a real socket or a superuser DSN. Until that patch lands, Hub turns keep today's power and appear in the step record only.

## What changes for Orion

Kept: root in their container; full Bash without prompts; `docker ps`, `logs`, `inspect`, `stats`; `docker exec` into every organ that does not hold the socket (thought, substrate-runtime, retro, falkordb, sql-db, ...); `dig`; GitHub; the web; the bus; their sandbox repo; projects.

Lost: creating, starting, stopping, removing, building and pulling containers and images; `exec` into the 8 containers that are a way back to the host (live 2026-11-24: Hub, governor, the four cortex-exec containers, mesh guardian, biometrics).

Kept but recorded and watched: `psql -U postgres` through exec; reading other containers' env; publishing on the bus.

## Proposal-mode answers

- **Capability change:** Orion loses changing containers and exec into socket holders. Orion gains projects that run as real turns. Everything Orion does in Docker and every turn step is recorded outside their reach.
- **Data touched:** new host folder `/mnt/storage-warm/orion-record/`; new tables `orion_docker_call`, `orion_turn_step`; Part C's `orion_project`. The step record contains full tool output, which includes whatever Orion read (chats, memory). It lives on athena's storage like the rest of Orion's data; it is not sent off the host.
- **Privacy boundary:** unchanged: no boundary between Juniper and Orion. The record is readable by Orion (`dig record`) and by Juniper.
- **Trace that proves it worked:** a refused `docker restart` from a turn appears in `docker-*.jsonl` with `refused`, and Juniper gets the tripwire notice; a turn's steps appear in `steps-*.jsonl` with the same `run_id` as its `harness_turn_trace` row.
- **Dangerous failure modes:**
  - The door is down: every Docker call from turns fails, including `dig`'s Postgres reads. Mitigation: the door is tiny, has `restart: unless-stopped`, and the mesh guardian already alerts on stopped containers.
  - A gap in the exec rule lets Orion exec into a container that reaches the host another way. Mitigation: the door refuses socket holders, privileged containers, `pid: host`, and host binds of `/`, `/run`, `/var/run`, `/var/lib/docker`. Live check 2026-11-24 found one such container without the socket (`orion-athena-biometrics`, privileged). A test in patch 1 re-runs this inventory against `docker ps`.
  - Superuser `psql` through exec can still destroy data. Accepted under shadow; tripwire notifies; nightly backups (`orion-backup`) are the recovery path. Question 2 asks whether to close it.
- **Disable / roll back:** restore the socket mount in the governor compose file and unset `DOCKER_HOST`. One commit. The record keeps running either way.

## Missing questions (bracketed default applies if Juniper says nothing)

1. Exec rule: allow exec into every container that does not hold the socket? [yes]
2. Superuser Postgres through exec (`psql -U postgres` in sql-db): keep, recorded and tripwired, or refuse exec into sql-db unless the command uses `-U orion_readonly`? [keep, recorded] (#3680 stays open either way)
3. Move Orion's Hub turns to the governor as step 2? [yes]
4. Record retention on the host: [90 days, then compressed and kept]
5. Someday "bring up containers": when it comes back, through the door with a list of compose services Orion may restart? [later; not designed here]

## Proposed schema / API changes

- New service `orion-docker-door` (compose on `app-net`, real socket mount, host bind `/mnt/storage-warm/orion-record`).
- New bus channel `orion:record:docker_call`, schema `DockerDoorCallV1` (time, method, path, target, exec_cmd, exec_user, decision, reason, caller_ip); registered in `orion/bus/channels.yaml` and `orion/schemas/registry.py`.
- New tables `orion_docker_call`, `orion_turn_step` (additive migrations, `-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer`).
- Part C: `orion_project` gains `workdir`.
- New `dig record` read verb.
- Governor env: `DOCKER_HOST=tcp://orion-docker-door:2375` (`.env_example` + local `.env` sync).

## Files likely to touch

- `services/orion-docker-door/` (new: `app/door.py`, `app/record.py`, `Dockerfile`, `docker-compose.yml`, `.env_example`, `README.md`, `tests/`)
- `services/orion-harness-governor/docker-compose.yml`, `Dockerfile` (github-mcp-server binary), `.env_example`
- `orion/fcc/fcc_claude_mcp.template.json`, `orion/fcc/mcp_config.py`
- `orion/schemas/docker_door.py`, `orion/schemas/registry.py`, `orion/bus/channels.yaml`
- `services/orion-sql-writer/app/models/`, `services/orion-sql-db/manual_migration_orion_record_v1.sql`
- `orion/dig/__main__.py` (`record`), Part C files (`orion/dig/project.py`, `orion/prime/intention_sources/project_step.py`)
- Step 2: `services/orion-hub/scripts/fcc_claude_bridge.py` and its callers

## Non-goals

- Taking root away from Orion inside their container.
- Blocking anything inside a turn other than changing containers and exec into socket holders.
- A separate per-project container (the earlier draft). Kept as a later option if one project ever needs stronger walls.
- Changing the organs that legitimately start and restart containers (mesh guardian, cortex-exec skills, retro runner, circe's lane controller).
- Sending the record off athena. If the threat ever includes host compromise, ship the JSONL to circe.

## Acceptance checks

Patch 1 (door + record), in the governor after deploy:

1. `docker ps`, `docker logs orion-athena-thought --tail 5`, and `docker exec orion-athena-thought python -c "print(1)"` all work.
2. `docker restart orion-athena-thought`, `docker run hello-world`, and `docker compose ... up` fail with the door's message, and each refusal is in today's `docker-*.jsonl`.
3. `docker exec orion-athena-hub true`, `docker exec orion-docker-door true`, and `docker exec orion-athena-biometrics true` fail (socket holder, door, privileged).
4. `test -S /var/run/docker.sock` is false inside the governor.
5. `dig portrait` and `dig body` work; a GitHub MCP tool call works in a real turn.
6. One live turn's tool steps appear in `steps-*.jsonl` and in `orion_turn_step` with its `run_id`.
7. A forced refused call produces a Juniper notification.

Patch 3 (projects): Orion opens a project with `dig project open`; one step runs as a governor turn in its folder and closes `produced = 1` with a commit, or `produced = 0, "no lasting trace"`.

## Rollout: canary, then swap

The door ships as a canary. It does not replace the socket on day one.

- **The door runs all the time,** but Orion's turns use it only inside a scheduled window: governor env `ORION_DOCKER_DOOR_CANARY_WINDOWS`, a comma list of `START/END` in UTC.
- **Inside a window,** each new turn sets `DOCKER_HOST` to the door (`tcp://orion-athena-docker-door:2375`). Cold turns and resumed turns get it through the spawn env. Warm chat slots get it per turn through their `CLAUDE_ENV_FILE`, which every Bash call re-reads.
- **When the window ends,** the next turn uses the socket again. No restart. A warm slot that served a canary turn gets `DOCKER_HOST` reset on its next turn.
- **If the door does not answer** (a 0.5 s connection check), the turn keeps the socket and the governor logs `DOCKER_DOOR_CANARY ... door=unreachable`. Orion is never cut off.
- **The socket stays mounted in the governor during the canary.** The canary tests what breaks, not whether Orion can get around the door. Removing the mount is the swap step below.
- **The GitHub MCP server stays on the real socket** (`DOCKER_HOST` pinned in `orion/fcc/fcc_claude_mcp.template.json`). It starts a container every turn, which the door refuses. It is harness code, not Orion's shell. Moving it to a binary is part of the swap.
- **What broke:** `python3 services/orion-docker-door/app/report.py --since <start> --until <end>` reads the door's record and lists every refused call. The governor log has one `DOCKER_DOOR_CANARY` line per turn.

**First window (Juniper, 2026-11-24): a full day, 2026-11-25 00:00Z to 2026-11-26 00:00Z.** Set in the live governor `.env`. It takes effect only once the build branch is merged and the governor is rebuilt.

**Swap, after a clean window:** remove the socket mount from the governor, set `DOCKER_HOST` for the whole container, move the GitHub MCP server to the binary, and delete the canary switch (kill means kill: no leftover window code).

### Build status (2026-11-24)

Built on local branch `feat/docker-door-canary` (commit `bad427158`, worktree `../Orion-Sapienform-docker-door-canary`). **Not pushed. Not deployed.** It is held until this design is approved.

- `services/orion-docker-door/`: the door (`app/door.py`, stdlib only), the report (`app/report.py`), compose, `.env_example`, README, tests.
- `orion/harness/docker_door_canary.py` + `fcc_motor._build_subprocess_env(docker_door=...)` + the warm pool's per-turn env file: the canary switch.
- Governor `.env_example`: `ORION_DOCKER_DOOR_CANARY_WINDOWS=` (empty) and `ORION_DOCKER_DOOR_URL`. Local `.env` synced.
- The rebuild planner already picks up the new service (`rebuild_affected_services.py --plan` lists `orion-docker-door build`).

Tests: 44 door + canary tests, 125 passed across the touched door, canary, MCP and warm-pool tests.

Live smoke against the real daemon (a temporary door container on `app-net`, removed afterwards), commands run inside the governor:

| Command | Result |
|---|---|
| `docker ps`, `docker logs`, `docker inspect` | work |
| `docker exec orion-athena-thought python -c ...` | works (`exec-ok`) |
| `docker exec -i ... python -` with stdin | works (`stdin-ok`) |
| `docker exec -i orion-athena-sql-db psql -U orion_readonly` (dig's path) | works (`1`) |
| `docker restart`, `docker run hello-world` | refused: "changing containers is off" |
| `docker exec` into hub, biometrics, the door | refused: "leads back to the host" |
| A call from `orion-athena-thought` (not an allowed caller) | 403 |
| Report over the record | 40 calls, 33 passed, 7 refused, each refusal listed |

### Code review of the held build (2026-11-24): must fix before it ships

An independent review found two confirmed ways past the filter. Both were shown with working proof scripts. **The held build must not be pushed until 1-5 are fixed and tested.**

1. **Smuggling with bare line feeds (critical, confirmed).** The door splits header lines only on `\r\n`. Docker's parser also accepts a bare `\n`. So one allowed `GET` can carry a second request, for example a `kill`, that the door never sees. Confirmed against the live daemon (read-only) and through the door; the record showed only the `GET`. Fix: refuse any head with a bare `\n` or a stray `\r`, folded header lines, a version other than `HTTP/1.0`/`HTTP/1.1`, or a method that is not a plain token.
2. **An `Upgrade` header switches off the forced close (critical, confirmed).** The door keeps the connection open for every exec route that sends `Upgrade`, including `resize` and a detached `start`, which never upgrade. A second request then rides the open connection unchecked. Fix: always send `Connection: close`. Docker decides on an upgrade from the `Upgrade` header alone, and a hijacked stream ignores `close`.
3. **An exec can ask for privileges (high).** The door checks the target container but not the exec request's own `"Privileged": true`. It also allows an exec when it cannot read the body (chunked, over 1 MB, or no length), which breaks the record too. Fix: refuse `Privileged`, and refuse an exec create whose body is unreadable.
4. **The host-reach list is too narrow (medium).** Add: extra capabilities (`CapAdd`, for example `SYS_ADMIN`), `Devices`, network/IPC/user namespace `host`, any runtime socket (`containerd.sock`), and binds of `/proc`, `/sys`, `/dev`, `/etc`, `/root`.
5. **Errors leave connections hanging (medium).** A bad `Content-Length`, a short body, or a full disk on the record write leaves the client open forever. Reads have no timeout, and the caller check runs only after the whole head arrives. Fix: check the caller first, add read timeouts, and close the client in a `finally`.
6. **The door check blocks the governor (low).** The 0.5 s connection check runs synchronously on every warm-slot turn in the window. Fix: run it off the event loop.
7. **Encoded paths dodge the attach rule (low).** `%61ttach` reads as "look". Fix: decode the path before classifying.
8. **The canary is not enforcement (by design).** The governor still holds the socket during the canary, so a turn that points `docker` at the socket directly goes around the door. The canary measures what breaks. Only the swap makes the door a wall.

Test gap behind 1 and 2: the fake daemon in the tests always closes after one request, so the "second request is not forwarded" test passed for the wrong reason. The fix needs a keep-alive fake daemon, plus tests for every case above.

Design lesson: a hand-written HTTP filter is fragile. The rule that makes it safe is simple and must be strict: **parse strictly, one request per connection, always close, and refuse anything the door cannot fully read.**

Not in the canary build (kept for after the swap): the turn-step stream to the record, the `orion_docker_call`/`orion_turn_step` tables and bus channel, `dig record`, tripwires.

## Recommended next patch

**Patch 1 (built, held): the canary door.** Fix review findings 1-7 with tests (keep-alive fake daemon), re-review, then push `feat/docker-door-canary`, merge, rebuild. The full-day window runs. Read the report.

**Patch 1b: the swap and the full record.** Socket mount removed from the governor, GitHub MCP as a binary, canary switch deleted, the turn-step stream to the record, the tables, `dig record`, tripwires. Proves checks 1-7.

**Patch 2: Hub turns to the governor.**

**Patch 3: Part C patch 6 (projects), with per-project folders.** Then Juniper asks Orion the Part C question.
