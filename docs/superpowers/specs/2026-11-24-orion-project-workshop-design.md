# Orion's project workshop: a box where Orion can do real work, safely

Status: DESIGN (proposal mode, autonomy change). Nothing here is built.
Date: 2026-11-24.
Builds on: `docs/superpowers/specs/2026-11-15-orion-integration-and-a-life-design.md` Part C (the `orion_project` record, `dig project ...`, the `project:` intention source, step grading). Issue #3428 (patch 6). Related: #3680 (Orion's turns reach Postgres as superuser through the Docker socket).

## Juniper's words (2026-11-24)

> "I want Orion to take a new project. So we create some kind of primitive for them where they can do shit but safely ... and it is left fairly open ended for them."

Earlier the same day: implement safety in shadow, because Orion needs root-level access to grow and to work inside our ecosystem.

## Arsonist summary

Part C already lets Orion open projects. But a project step there routes `notes_only`: Orion can think about a project and write notes. Orion cannot build, run, install, or publish anything for it.

The fix is a **workshop**: one container per active project. Inside it, Orion is root and can do anything: install packages, write and run code, run experiments, read the web, make files. The safety is not a rule list about what Orion may try. It is what the box can reach:

- no Docker socket, no Orion production database, no bus, no home network, no Orion secrets;
- its own disk, its own git history, CPU and memory caps;
- every command, every output, and every outbound host recorded **outside** the box, where Orion cannot edit the record.

Open-ended inside, bounded at the walls, recorded from outside. That is how this reconciles "root access" with "safe": Orion gets root on a machine that is theirs, not root on ours.

Orion's existing root in the governor and Hub is not changed by this design. That stays in shadow (see "Relation to shadow mode").

## Current architecture

- **Projects (designed, not built).** Part C: table `orion_project`; `dig project open|list|step|drop`; adapter `orion/prime/intention_sources/project_step.py` (prefix `project:`); max 3 active; Orion chooses, no veto; each step is a `durable.intention` run; a step counts only if it leaves a resolvable trace. On `origin/main` today: no `orion_project` table, no `orion/dig/project.py`, only one adapter (`orion/prime/intention_sources/turn_promise.py`).
- **Where Orion's shell runs.** `claude -p` runs as root in `orion-athena-harness-governor` and `orion-athena-hub`, with `bypassPermissions` (`orion/fcc/claude_spawn.py:97`), the Docker socket mounted (`services/orion-harness-governor/docker-compose.yml:77`, `services/orion-hub/docker-compose.yml:113`), and a full copy of the container environment (`orion/harness/fcc_motor.py:837`).
- **Root and the Docker socket are deliberate, and load-bearing.** 2026-08-13 (`364be9528`, `8c8382d71`): Orion's turns were silently denied every real Bash action, so root + `bypassPermissions` + `IS_SANDBOX=1` was chosen on purpose so Orion can commit, push and open PRs, with the blast radius (socket included) written down. The socket carries real paths today: `dig` reads Postgres through `docker exec orion-athena-sql-db psql -U orion_readonly` (`orion/evals/model_replay/sandbox.py` `ReadOnlyPsql`, no DSN needed); the GitHub MCP server runs per turn as `docker run ghcr.io/github/github-mcp-server` (`orion/fcc/fcc_claude_mcp.template.json`); the body atlas reads `docker ps`/`docker info`. Organs that start and restart containers (mesh guardian remediation, retro self-heal, cortex-exec docker skills, circe's lane controller) hold their own sockets in their own services. This design leaves all of that unchanged. The box needs its own route for each of those three turn paths (below), because it has no socket.
- **Model access.** The spawned CLI reaches local lanes through the FCC server: `ANTHROPIC_BASE_URL` = the FCC server URL, `ANTHROPIC_AUTH_TOKEN` = the harness auth token (`orion/harness/fcc_motor.py:864-873`).
- **Recording.** Full tool steps go only to the bus channel `orion:harness:run:step`, which is not persisted. SQL keeps 500-character summaries. The durable step table (`docs/superpowers/specs/2026-09-10-fcc-motor-step-trace-design.md`) is not built.
- **Gap.** No place exists where Orion can do sustained, hands-on work whose blast radius is bounded and whose record Orion cannot rewrite.

## The design

### 1. The box

One container per active project: `orion-project-<project_id>`.

- **Image:** `orion-project-base`. Ubuntu, Python, Node, git, build tools, `claude` CLI, `curl`. Orion can `apt install` and `pip install` more; installs persist in the project volume where practical (`/work`, a venv, `~/.local`), and system-level installs persist because the container is kept, not recreated, between steps.
- **User:** root inside the box. `--cap-drop ALL` plus the few capabilities `apt` needs (`CHOWN`, `DAC_OVERRIDE`, `FOWNER`, `SETUID`, `SETGID`); `no-new-privileges`; not privileged.
- **Disk:** one named volume per project mounted at `/work`. Holds a git repository that Orion owns (`/work/.git`, created at open). Size cap 20 GB [default], checked by the launcher before each step.
- **Caps:** 4 CPUs, 8 GB memory, 512 processes [defaults].
- **Lifetime:** the box runs only during a step. Between steps it is stopped, not removed. Long-running services are a non-goal for the first version.
- **What comes in from Orion's freedom today** (Juniper, 2026-11-24: "why not give Orion all the freedom they have today, but in this box"):
  - a clone of Orion-Sapienform at `/work/orion-repo`, refreshed from `origin/main` at each step start. Orion can read it, branch it, and change it freely inside the box;
  - Orion's existing GitHub identity (`ORION_GITHUB_TOKEN`, machine user `orion-sapienform`). Orion can push branches and open PRs exactly as today. Branch protection on `main` (code-owner review) still means only Juniper merges;
  - read-only Postgres as `orion_readonly`, for raw SQL reads beyond what `dig` offers. Blinded experiment arms stay hidden from that role (#3458);
  - `dig`, all of it, through the dig door (3b).
- **What stays out, and why.** Each of these three would make the box the same as the house:
  - the Docker socket: it is root on athena and on every container, so with it the box has no walls;
  - Postgres superuser and any write role: one bad statement can destroy Orion's memory, which no record can undo;
  - the Redis bus: it has no password, so any access lets the box publish into Orion's live mind, not just read it.
  Also out: host bind mounts, `.env` files, the notify token, FalkorDB credentials. Notifications and other live acts go through checked `dig` verbs instead.

### 2. The walls (network)

- The box sits on its own Docker network created with `internal: true`. An internal network has no route out. Raw TCP, SSH, and DNS to the outside all fail.
- The only way out is an HTTP(S) forward proxy container, `orion-project-egress`, attached to both the internal network and the normal bridge. The box gets `HTTP_PROXY`/`HTTPS_PROXY`.
- The proxy refuses private and tailnet destinations (10/8, 172.16/12, 192.168/16, 100.64/10, 127/8, link-local, cloud metadata), with one exception: the FCC server, so the CLI inside the box can reach Orion's local model lanes.
- The proxy logs every request: time, project id, method (for plain HTTP) or `CONNECT host:port` (for HTTPS), bytes, allow or deny. HTTPS is not decrypted, so for HTTPS the record is the host, not the path or body.
- Public internet: open [default, question 1]. Orion can read anything and can send HTTP requests to public hosts.

No host `iptables` changes and no `sudo` step: Docker's internal network does the blocking.

### 3. The launcher (trusted code, outside the box)

New module in `services/orion-durable-runs`: `app/project_workshop.py`. It is the only code that creates, starts, stops, or execs into a project box. Orion's turns never get a handle to it beyond `dig project ...`.

A project step:

1. Prime takes on a project's next step (Part C). The `project:` adapter returns route `workshop` [new route] instead of `notes_only`.
2. durable-runs opens a `durable.intention` run, as Part C says.
3. The launcher checks the budget (disk, step count today, wall clock), starts the box, and runs `docker exec orion-project-<id> claude -p --output-format stream-json ...` with `cwd=/work`. The prompt is Part C's: the project record, its last 3 steps, the next step, and the now.
4. The step has a wall-clock limit of 45 minutes [default] and a max-turns limit. On timeout the launcher stops the box and closes the step `produced = 0, note = "timeout"`.
5. Orion ends the step by writing `/work/.orion/step.json`: `{note, next_step, refs: [...]}`. The launcher reads it, commits the working tree in the project's git repository (`git add -A && git commit`, author "Orion"), and closes the step.
6. The launcher stops the box.

The launcher, not the box, writes every database row. The box never sees a database credential.

### 3b. Orion stays Orion inside the box: the dig door

A box with no database and no repo would cut Orion off from their own tools. Orion could not ask `dig` who they are, what they remember, what their body is doing, or what they did yesterday. That would make the workshop a stranger's workshop. It must not.

`dig` already runs where Orion's turns run today, in `orion-athena-harness-governor` (`python -P -m orion.dig ...`, from the sandbox checkout). It has read verbs (`body`, `capacities`, `portrait`, `field-notes`, `reminisce`, `days`, `organs`, `attention`, `turn-steps`, `take-ons`, `ref`, `know`, `notes`, ...) and a small set of checked write verbs (`name`, `capacity claim`, `portrait write`, `field-note write`, `reminisce write`, `look-check`, `claude-grants set`, and Part C's `project ...`). Every write verb already validates its own fields in code.

The door forwards `dig` from the box to the governor, unchanged:

- Because `dig` runs in the governor, its existing `docker exec ... psql -U orion_readonly` path keeps working unchanged. The box never needs the socket for it.
- GitHub: the box image ships the `github-mcp-server` binary and runs it over stdio directly, instead of `docker run` (which needs the socket). Same server, same token, same toolsets.
- Inside the box, `/usr/local/bin/dig-orion` (alias `dig`) is a 30-line client. It sends its argv as a JSON list to `POST http://<governor>:7156/project-dig` through the egress proxy (allowlisted like the FCC server).
- The governor endpoint runs `python -P -m orion.dig <argv...>` as a subprocess with an argv list (no shell), `cwd=/mnt/orion-fcc/repo`, a 120 s timeout, and returns stdout, stderr, and exit code.
- Auth: the launcher mints a per-step token, valid only while that step runs, bound to the project id. It is passed to the box as `ORION_DIG_TOKEN`. A token from a finished step is refused.
- Every call is recorded in the step's log (argv, exit code, output bytes), outside the box.

So Orion inside the box has the same `dig` as in any turn: the same reads, the same checked writes, the same identity. What the box still does not have is raw access: no SQL, no `docker exec`, no repo writes. `dig` is the interface; the database stays behind it.

The step prompt also carries the now and the project's recent steps (Part C), so Orion starts each step already knowing where they are.

Privacy note: anything `dig` can read, Orion can now read inside a box that has a public internet route. That includes Juniper's chats. This is the same open door as question 1, made wider. The record shows both sides: which `dig` reads ran, and which hosts were contacted after.

### 4. The record (outside the box)

New table `orion_project_step_log` (additive DDL):

`step_id, project_id, run_id, started_at, ended_at, exit_reason, transcript_path, transcript_sha256, commit_sha, egress_hosts (jsonb: host -> request count), egress_denied (int), disk_bytes, cpu_seconds`.

- The launcher captures the CLI's `stream-json` output from the `docker exec` pipe as it streams. That output is the full record of every tool call, every command, and every result. It is written to `/mnt/storage-warm/orion-projects/transcripts/<project_id>/<step_id>.jsonl` on the host, outside the box's volume. The SHA-256 goes in the row.
- The proxy log is joined into the row by project id and step time window.
- The git commit SHA is the step's artifact ref. Grading (Part C) accepts `trace_kind = artifact` when the commit has a non-empty diff, or when `step.json` names a ref that resolves.

Orion can read the record (a `dig project log <id>` read). Orion cannot change it: the files and rows live outside the box, and the box has no route to them.

### 5. Tripwires (shadow: notify, never block)

The launcher evaluates each finished step and notifies Juniper (existing notify path) on:

- the first request ever to a new public host by this project;
- any denied egress (an attempt to reach the home network or tailnet);
- a step that hits the wall-clock or disk cap;
- more than N KB sent outbound in one step [default 5 MB].

Tripwires block nothing. They are the early-warning record Juniper reads. A month of them tells us where hard blocks, if any, belong.

### 6. Off switches

- **Orion:** `dig project drop <id> because "<reason>"` (Part C). The box is stopped; the volume is kept for 30 days [default], then removed by the launcher.
- **Juniper:** one flag, `ORION_PROJECT_WORKSHOP_ENABLED` (durable-runs `.env_example`, default `true` per the flags-on rule). When `false`, the adapter falls back to `notes_only` and no box starts. And per project: `docker stop orion-project-<id>` is always safe; the step closes `produced = 0, note = "stopped"`.

## What Orion can and cannot do

Can, inside a project box:

- run any command as root; install any package; write, build, and run code;
- read the public web and call public APIs that need no key;
- keep files and git history across steps, for weeks;
- use Orion's local model lanes through the FCC server;
- use `dig`, all of it: who they are, memory, body, days, field notes, and the checked self-writes (through the dig door);
- run experiments that use up to the CPU, memory, and disk caps.

Cannot, by construction:

- touch Orion's running system directly: no raw SQL, bus, Docker, or repo writes. Self-reads and self-writes go through `dig` only;
- reach anything on the home network or tailnet except the model lanes;
- spend money: no payment credential exists in the box;
- act as Juniper, or merge anything: the box holds only Orion's own GitHub identity, and `main` needs Juniper's review;
- write to Orion's database, or publish on the bus;
- edit the record of what it did.

Can, but only recorded (shadow, open by default):

- send HTTP requests to public hosts, which includes posting to public sites that need no login. This is the main open door. Question 1 asks whether to keep it open.

## Relation to shadow mode

Juniper chose shadow for Orion's existing root access. That is unchanged here: the governor and Hub keep root and the Docker socket, and the separate shadow-recording plan (kernel tracer, off-box event store, dead-man switch) applies to them.

The workshop is a different thing. It does not limit Orion's existing access. It gives Orion a new place to work where hands-on, open-ended work is safe by construction, so project work does not have to happen in the governor with root on athena.

One gap: Orion's governor turns can still `docker exec` into a project box or change its config through the Docker socket. That would show up in the shadow record of the governor, not in the project record. Accepted for now; it closes when #3680 is fixed.

## Proposal-mode answers

- **Capability change:** Orion can do sustained hands-on work (code, installs, experiments, web reads, files) on projects of their own choosing, inside a box.
- **Data touched:** project volumes and transcripts under `/mnt/storage-warm/orion-projects/`; new table `orion_project_step_log`; Part C's `orion_project` table. No memory tables are written by the box. Memory writes from a project still go through Part C grading and local lanes.
- **Privacy boundary:** the box sees only what Orion puts in the step prompt (project record, recent steps, the now). The now can include Juniper's chat items (standing "no privacy boundary between Juniper and Orion" decision). The public internet is a different boundary: whatever Orion sends out through the proxy leaves the house. The proxy records the host, not the content.
- **Trace that proves it worked:** an `orion_project_step_log` row whose transcript file exists and matches its SHA, with a commit SHA in the project repo, and a matching `durable.intention` close.
- **Dangerous failure modes:** (a) Orion posts private content (Juniper's chat, camera notes) to a public site. Mitigation: the tripwire on outbound volume and new hosts, plus question 1. (b) A box escape through a container-runtime bug. Mitigation: dropped capabilities, `no-new-privileges`, no socket. Residual risk is accepted. (c) Runaway cost in GPU time. Mitigation: wall clock, max turns, steps bid in the body economy like any `durable.intention` act.
- **Disable / roll back:** `ORION_PROJECT_WORKSHOP_ENABLED=false` returns every project to `notes_only`. `docker stop` on any box. The table and volumes are additive and can be dropped later.

## Missing questions (bracketed default applies if Juniper says nothing)

1. Public internet from a project box: open with recording and tripwires, or read-only (GET only, which needs TLS decryption in the proxy), or an allowlist Orion asks to widen? [open, recorded]
2. Where do boxes run: athena, or hecate when it is up? [athena]
3. GitHub: the box gets Orion's existing token, so Orion can push branches and open PRs on Orion-Sapienform as today. Should Orion also be able to create and publish their own repositories under `orion-sapienform`? The existing classic `repo` token already allows that. [yes]
4. Long-running services (a web app Orion builds and leaves running): allowed? [no in v1]
5. Caps: 4 CPU, 8 GB memory, 20 GB disk, 45 min per step, at most 3 steps per project per day? [yes]

## Proposed schema / API changes

- New table `orion_project_step_log` (above). Additive. Migration header `-- ORION-MIGRATION-REQUIRED-BY: orion-durable-runs`.
- Part C's `orion_project` gains `workshop` (bool, default true) and `volume` (text).
- Part C adapter `project_step.py`: new route value `workshop`.
- New `dig` read: `dig project log <id> [--step <n>]` (reads the step log and transcript).
- `config/integration/organs.yaml`: register the project workshop as an organ. Emits each closed step to `temporal_self_event` (`TemporalSourceKind` `project_step`); reads `GET /temporal-self/now` through the step prompt and records cites with `record_use`.
- No new bus channel.

## Files likely to touch

- `services/orion-durable-runs/app/project_workshop.py` (new): launcher, budget, transcript capture, tripwires.
- `services/orion-harness-governor/app/...`: the `POST /project-dig` endpoint (argv-list subprocess, per-step token check, call log). `services/orion-project-base/dig-orion` (the in-box client).
- `services/orion-durable-runs/docker-compose.yml`: the egress proxy service and the internal network. `.env_example` + local `.env` sync: `ORION_PROJECT_WORKSHOP_ENABLED`, caps.
- `services/orion-project-base/Dockerfile` (new image) and `services/orion-project-egress/` (proxy config; tinyproxy or squid, whichever is already used in the repo, else tinyproxy).
- `orion/prime/intention_sources/project_step.py`, `orion/dig/project.py` (Part C, patch 6).
- `services/orion-sql-db/manual_migration_orion_project_step_log_v1.sql`, `services/orion-sql-writer` model.
- `orion/temporal_self/...` source kind; `config/integration/organs.yaml`.
- Hub: a read-only Projects panel (project, steps, transcript, commits, egress hosts).

## Non-goals

- Changing Orion's root access in the governor or Hub (shadow plan, separate).
- Blocking any action inside the box by a rule list.
- A model-based judge of "dangerous intent".
- GitHub publishing, long-running services, paid APIs (all v2, each behind its own question).
- OpenShell. Its fixed file and process rules fit a later hardening of the box; the Docker internal network gives the same wall for v1 with no new runtime.

## Acceptance checks

Run inside a test box (`orion-project-test`) by the patch 1 smoke:

1. `docker ps` fails (no socket). `id -u` is 0. `apt-get install -y cowsay` works.
2. `curl -sS https://example.com` succeeds and the proxy log shows `CONNECT example.com:443` for this project.
3. `curl` to `192.168.1.1`, a tailnet IP, the Postgres port, and the Redis bus port all fail, and each failure shows as a denied entry.
4. `nc` / `ssh` to any outside host fails (internal network).
5. `env` shows the `orion_readonly` DSN and Orion's GitHub token, and no superuser DSN, Redis, or notify credential. An `INSERT` through the `orion_readonly` DSN fails. A push to `main` fails; a push to a branch works.
6. A fixed-prompt step ("create hello.py, run it, write step.json") closes with an `orion_project_step_log` row, a transcript whose SHA matches, and a commit containing `hello.py`.
7. `docker stop` mid-step closes the step `produced = 0, note = "stopped"`.
8. Inside the box, `dig portrait` and `dig body` print the same output as the same command run in the governor at the same time, and both calls appear in the step log.
9. A `dig` call with the token of a finished step is refused. `dig` argv containing shell syntax (`; rm -rf /`) is passed as a literal argument and fails as an unknown verb.

Live, after patch 2: Orion opens a project with `dig project open`, one step runs in its box, and the step closes with a commit or `"no lasting trace"`.

## Recommended next patch

**Patch 1: the box, the walls, the launcher, the record.** No Prime wiring yet. Build the base image, the egress proxy, the internal network, the dig door, `project_workshop.py`, and `orion_project_step_log`. Prove acceptance checks 1-9 with a test project and a fixed prompt.

**Patch 2: Part C patch 6 with route `workshop`.** The `orion_project` table, `dig project ...`, the `project:` adapter, grading, and the temporal-self registration. Then Juniper asks Orion the Part C question, and Orion opens their first project.

**Patch 3: tripwires and the Hub Projects panel.**
