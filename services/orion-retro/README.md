# Retro headless authority (P1)

A persistent, operator-only NES service. One spawned child owns the libretro core;
HTTP/WebSocket handling never calls the native core. Browser access is provided by the separate Hub adapter below. This service has
no Orion policy or cognition integration. No bus connection or global event schemas.

The fixed catalog is imported from operator-owned local kits, including Pad Lab,
Nova and Spacegulls when installed. ROMs, credentials, saves and evidence are never
committed. The API cannot download a ROM, select a host path or upload native code.
All games currently have save/load disabled. Nova's strict serialized-state replay
fails with the pinned core; Spacegulls passed the existing short replay check.
Neither claim establishes game competence or browser play.

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
active persisted state failed (`service_restart`), never resumes play. Save/load
is not offered. No raw A/V recording, model invocation or memory export occurs.

## Isolation and evidence

Container: non-root UID/GID, read-only root, dropped capabilities, no-new-privileges,
512 MiB RAM, 2 CPUs, 64 PIDs, journald, bounded tmpfs. Only catalog and credential
are read-only mounts; dedicated evidence is writable. No Docker socket, host
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

Validated-input rejection/unknown responses now also include `port`, `epoch` and
`seq`. This is additive: existing CLI clients can still inspect `rejected` or
`unknown`. Hub ignores replies from an earlier lease. A connection that no longer
owns any port cannot pause a fresh resume merely by closing.
