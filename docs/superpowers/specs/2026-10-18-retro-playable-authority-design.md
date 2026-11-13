# Retro Play: from installed diagnostics to a playable service

Status: proposal, 2026-10-18. No runtime or deployment change in this document.
Continues [design PR #2751](https://github.com/junebug-junie/Orion-Sapienform/pull/2751),
which remains open. The merged foundation is #2754 and #2757. This follow-up
narrows their next implementation step; it does not replace the longer-term
cooperative play, learning, or leisure proposal.

## Arsonist summary

The installed kit runs games in disposable Docker containers. It is not a
deployed game service. Pulling #2757 correctly creates no persistent container.
Build one persistent `orion-retro` service next, with one authoritative NES
session and inspectable applied inputs. Then add human viewing/control in Hub.
Only then connect a clearly labeled scripted Orion controller. Do not promise
game competence, multiplayer Nova, or cognition from an emulator benchmark.

Nova is the first real game, explicitly supplied by the operator. Keep the
original Pad Lab ROM for independent controller tests. Nova's current core fails
strict serialized-state replay equality; preserve that failure and disable its
save/resume and replay-dependent training until resolved. Fresh-start human
play can ship independently, with this limitation visible before starting.

## Current architecture

Inspected on main `64683e26f2280e3d2de7ea5ce5b5ab6a1a10efc1`:

| Surface | Current evidence / gap |
|---|---|
| Emulator | `orion/retro/probe.py`: ctypes libretro ABI host, frame hash, audio sample counts, controller masks, snapshot/restore. Callback buffers are not yet a browser media transport. |
| Entry points | `setup_local.py` installs; `local_runner.py` creates an isolated container, runs pads/smoke/baseline, records inspect/output, removes it. |
| Config | Local manifest pins ROM/core hashes and runner image. No Retro `.env_example`, service settings, Dockerfile, compose, or health endpoint exists. |
| Bus / registry | No Retro bus producer, consumer, channel or registered schema. None required for the next headless slice. |
| Tests / evals | `tests/retro/` (27 tests), authored-ROM pad eval, preflight CLI, native ABI fixture, real-core CI. No interactive browser acceptance test. |
| Hub | `services/orion-hub/scripts/main.py`, `api_routes.py`, `websocket_handler.py`, `scripts/settings.py`, and `docker-compose.yml` exist. Chat WebSocket is not the game input loop. Authenticated game-session ownership is not verified. |
| Deployment | Pull hook selects affected compose services; #2757 adds none. Installed runtime is a versioned copy, not an automatic live view of git. |

Measured facts, not targets:

- Pad Lab: 110,000 frames / 1,830.29 seconds, nominal 60.0998 FPS,
  step p95 5.42 ms, zero measured deadline misses; 1,000-frame video and
  serialized-state replay equality passed. This is one host-load sample.
- Actual host inventory: dual Xeon Platinum 8168, 48 physical / 96 logical
  cores. This corrects the original design's unmeasured Xeon 6138 hypothesis;
  it does not establish available CPU capacity under all loads.
- Nova official v1.0.6a ROM SHA-256:
  `e4780e90b9d1587489bfb797d2ca395be21371ea9262fa9f87f99324ec6960ab`.
  Core FCEUmm revision `ac0d00a7fc01166f84c7bac01d1084ebfd2b865e`.
  Short smoke: ~60.09 FPS, video/audio callbacks, 1,000 identical replayed
  video frames. Final serialized states differ in one byte of MMC1 `LRST`:
  34 versus 0. Core `src/boards/mmc1.c` resets `lreset` on restore.
  Do not mask the byte to manufacture a pass. Nova audible PCM content,
  30-minute stability, actual level entry and browser latency remain UNVERIFIED.
- Evidence lives under the operator's `~/.local/share/orion/retro/evidence/`
  and `games/nova-the-squirrel-v1.0.6a/evidence/`; `deployment-verification.json`
  and the game's `validation.json` point to the individual reports. These are
  local artifacts, not tracked ROMs, saves or public performance guarantees.

## Missing questions

These are implementation gates, not requests to make the operator supply setup:

1. Inspect Hub middleware/proxy configuration and identify the actual trusted
   principal, origin checks and mutation authorization. If absent, implement a
   scoped authenticated operator boundary before exposing game controls. Never
   treat a client-supplied player name, session ID or Tailscale reachability as
   proof of ownership.
2. Measure browser transport on the actual client: propose WebRTC A/V with
   authenticated same-origin signaling and a separate input WebSocket. Verify
   ICE reachability, codec/CPU cost, audio startup and input-to-photon latency
   before selecting final media dependencies/ports. Do not add TURN by default.
3. Determine whether Nova's mapper restore behavior can meet a defensible
   reproducibility contract. First reproduce on a pinned upstream core in an
   offline regression fixture; any core change needs new hashes and the entire
   diagnostic and Nova gates. Existing failure stays visible meanwhile.
4. Inspect Nova source and exercise level entry before naming game outcomes or
   RAM features. Do not reuse Pad Lab counters as a Nova decoder. Treat Nova as
   single-player for this milestone; a second port in the emulator proves no
   cooperative gameplay capability.

## Proposed schema / API changes

### One service and one session

`services/orion-retro/` owns lifecycle, catalog validation, controller authority
and bounded evidence. A separate child process inside its container owns the
native core and advances frames on monotonic deadlines. The API process never
loads the core. Use bounded local IPC, not Redis, for commands and callbacks.
One active session maximum; competing starts return conflict. No queue of
future game starts and no unattended restart after a crash or deployment.

Persistent states: `idle -> starting -> running <-> paused -> stopped`, with
`failed` reachable from any active state. Each transition has a timestamp,
session ID and reason. Restart marks any previously active session `failed` with
reason `service_restart`, never resumes buttons or loads a save implicitly.
Child exit, IPC loss or three missed frame
deadlines pauses/fails the session; do not fast-forward a backlog to catch up.

The container is the native-code security boundary; splitting processes is
fault isolation, not a claim that an exploited core cannot affect its sibling.
Non-root, read-only root, no capabilities, no-new-privileges, bounded CPU/RAM/
PIDs, journald, no Docker socket, no host credentials, no production data mounts.
Build the pinned core in a Docker build stage so deployment no longer depends on
host compiler ABI. Re-run compatibility checks for that resulting binary.
Only an allowlisted local ROM catalog is mounted read-only; one writable Retro
artifact directory has a hard disk budget. Never accept arbitrary filesystem
paths, core uploads, shell commands or download URLs in the runtime API.

P1 uses a private Docker network with API published on host loopback only, for
the existing host-networked Hub to reach later. This intentionally differs from
the offline runner's `--network none`. Deny external egress and test the actual
network configuration; publishing a loopback port alone does not deny egress.
No bus credentials in P1. Backend requests use a scoped service credential kept
out of browser responses/logs; user authorization remains Hub's responsibility.

Before Hub exists, P1's smoke CLI authenticates with a separate operator
credential mapped by server configuration to one fixed operator principal.
Untrusted request bodies cannot choose that principal. P2 uses a distinct Hub
credential and an authenticated principal assertion from Hub; only the trusted
Hub caller may assert a principal, after validating the user's session. Neither
credential is mounted into the ROM catalog or included in evidence. The native
container boundary still applies to these service-held secrets.

### Concrete contracts

Service-local strict models in `app/contracts.py` until there are real cross-
service event consumers. Unknown fields rejected; UUID identifiers, bounded
strings, explicit protocol version. No new global schema/channel registrations
for proposed future cognition. Register a shared event with producer, consumer
and fixtures in the later patch that actually publishes it.

| Contract | Fields / observable behavior |
|---|---|
| Catalog entry | `game_id`, title, ROM SHA, core/build SHA, allowed ports, verified capabilities, provenance/license references. Server resolves fixed mounted files; startup verifies hashes. |
| Session | `session_id`, boot generation, state, game/core fingerprints, current frame, owner principal, active control epochs, save capability verdict, failure reason. |
| Controller intent | version, session/generation, connection-bound lease token, port, epoch, increasing `seq`, legal button mask. Server stamps monotonic receive time; client clocks do not extend expiry. |
| Applied action | session, port, epoch, seq, requested mask, applied mask, actual frame and rejection/neutralization reason. Receipt is not actuation. |
| Local evidence | transition/action records with monotonic/frame ordering and UTC for persisted lifecycle records. Include core/ROM/runtime identity; no tokens or per-frame screenshots. |

Proposed internal API, mirrored under Hub `/api/retro` only after authorization:

| Route | Contract |
|---|---|
| `GET /health/live`, `/health/ready` | API alive versus catalog/core dependencies ready; distinguish idle from healthy frame advancement. |
| `GET /games` | Public catalog fields and measured capability verdicts, no host paths. |
| `POST /sessions` | `game_id` and idempotency key bound to principal; return new session or same prior result, conflict on different payload/active session. |
| `GET /sessions/{id}` | Owner-authorized state and latest applied frame, not just process uptime. |
| `POST /sessions/{id}/pause`, `/resume`, `/stop` | Owner-only, idempotent; stop revokes leases and neutralizes before child teardown. Resume requires new leases. |
| `POST /sessions/{id}/control` | Explicit acquire/release/takeover; return epoch and short-lived connection-bound lease. Human takeover wins at next frame. |
| `WS /sessions/{id}/input` | Authenticated input + applied/rejected acknowledgments; no chat or LLM forwarding. |
| `GET /sessions/{id}/evidence` | Owner-authorized bounded action/lifecycle history or artifact IDs, never arbitrary paths. |

Open and authenticate the input WebSocket before acquiring control. The server
returns an unpredictable `connection_id`, bound to that principal and session.
`POST /control` references this ID; acquisition requires a live connection owned
by the same principal. Return its lease token/epoch over the authenticated
response, never a URL. Input tokens are valid only on that connection. A closed
connection revokes its leases and cannot be revived by retrying the HTTP call.
P1's CLI exercises this exact sequence; P2's Hub proxy preserves a distinct
upstream connection for each browser connection rather than multiplexing owners.

No save/load endpoint in P1. Add it only with compatibility checks, atomic writes,
size/quota enforcement and passing replay tests; Nova remains disabled until its
gate passes. Browser capabilities must derive from the catalog verdict.

### Authority and failure defaults

- Input lease expires after 250 ms without a valid heartbeat/input; heartbeat
  interval 100 ms. These are proposed fail-safe budgets, not measured latency.
  Expiry releases the port to neutral on the next frame. Paused time cannot keep
  a stale lease alive. Disconnect invalidates the connection's leases immediately.
- On takeover, increment epoch, reject all old epoch commands and apply neutral
  before the new owner's first mask. Never OR two owners' inputs. An authorized
  takeover can bypass an overloaded ordinary input queue through a bounded
  priority control slot. Reject duplicate/out-of-order sequence numbers.
- Browser blur, hidden tab and gamepad disconnect send neutral/release. Server
  expiry is authoritative when those messages never arrive. Loss of the last
  controlling connection (CLI or browser) pauses the session; unattended
  continuation is out of scope.
- Retain only latest pending mask per port; acknowledgments must explicitly
  report superseded commands. Bound ingress at 120 messages/second/connection,
  4 KiB/message, two controller connections and one session. Slow observers lose
  obsolete frames, not emulator time. Disconnect slow acknowledgment consumers
  rather than growing queues. No unbounded per-frame log writes.
- Write action ranges and lifecycle records with a 256 MiB/session and 2 GiB
  installation budget. Stop and mark evidence incomplete if storage cannot
  accept a required record; never label truncated evidence replay-complete.
  Preserve existing saves/evidence; require explicit operator deletion rather
  than automatic eviction to make room.

### Media, cognition and privacy boundaries

P2 copies video and PCM out of callbacks before native buffers expire; bounded
media queues decouple encoding from frame advancement. Use frame IDs and audio
sample counts for synchronization. Start audio on a user gesture. A black canvas,
silent callback count or merely connected WebRTC peer is not acceptance evidence.
Measure native-rate play, actual audible output, A/V drift and browser controls.
Codec dependencies and exposed UDP ports belong to that reviewed media patch.

Do not export images, chat, inputs, or game events into memory/cognition in P1/P2.
Raw A/V is transient by default; demonstrations need an explicit recording UI.
The later scripted-controller patch must display its identity and allow immediate
human takeover. A later cognition proposal must name the exact existing producer/
consumer, stored data, privacy boundary and disable path, and prove mistake ->
revised goal -> changed action -> checked outcome. No leisure admission, world-
model revival, RAM-derived reward or persistent identity changes in this patch.

## Files likely to touch

P1 creates the currently absent `services/orion-retro/README.md`, `.env_example`,
`docker-compose.yml`, `Dockerfile`, `requirements.txt`, `settings.py`,
`app/{main,contracts,session,worker,control,evidence}.py`, `tests/`, and `evals/`.
Reuse/refactor `orion/retro/probe.py` only with existing ABI tests preserved; leave
the standalone installed diagnostic kit functional. Add `docs/runbooks/retro-service.md`.
Extend `.github/workflows/retro-preflight.yml` and deployment-classifier tests.

P2 touches Hub `scripts/{main,api_routes,settings}.py`, adds
`scripts/retro_routes.py`, `static/js/retro-panel.js`, its actual rendered template,
and focused browser tests. Inspect the template/router loading path before edits.
Do not route game traffic through `scripts/websocket_handler.py`'s chat flow.

Proposed config: `RETRO_ENABLED=false`, private bind port, fixed catalog/artifact
paths, service credential, quotas, and CPU/memory limits. Add keys only alongside
their consumers; synchronize local `.env` when templates change. P1 needs no bus
URL. A later bus consumer must use the verified Tailscale-node Redis address per
repo policy, never a compose hostname. No empty future cognition flags now.

## Non-goals

SNES, native-console support, multiplayer Nova, universal game competence,
training, automatic ROM discovery/download, public streaming, new hardware,
per-frame LLM calls, a general container supervisor, automatic migration of
diagnostic artifacts, and production cognition/metric wiring are excluded.
Nova's license/provenance stays with local artifacts; no ROMs or game assets are
committed. Operator-authorized installation already happened; runtime does not
gain permission to download other games.

## Acceptance checks

| Stage | Required proof |
|---|---|
| P1 contracts | Invalid masks/ports, stale sessions, duplicate sequence, wrong owner, expired token, takeover races and IPC overload tested with deterministic clocks. |
| P1 operator | Authenticated CLI starts session, opens WS, acquires a connection-bound lease, observes input applied, then disconnects. Assert neutral/pause, rejection of old lease after reconnect, and no client-selected principal. |
| P1 native | Pad Lab verifies independent P1/P2 effects, nonzero PCM, video, replay and 30-minute cadence under service packaging. Real core required; stub-only tests cannot pass this gate. |
| P1 failures | Crash child, block IPC, fill evidence quota, expire lease, terminate service, restart container: neutral/paused or failed states, bounded cleanup, no automatic continuation. |
| P1 deployment | Clean synchronized main; build/up via `safe_docker_build.sh`; actual inspect confirms restrictions. Health plus advancing frame and applied-input evidence proves live path. A subsequent service edit is selected by pull-hook classifier tests. |
| Nova capability | Fresh-start boot and visible level entry from known inputs, nonzero audio samples, sustained cadence. Strict save/replay remains failed until independently fixed and re-evaluated. |
| P2 playable | Rendered Hub template loads the correct asset; authenticated owner sees/hears game, presses Start, moves/jumps, pauses and reconnects. Test keyboard plus gamepad, blur/disconnect, takeover, unauthorized HTTP/WS and slow-client behavior. |
| P2 performance | 30-minute actual-client session; proposed input-to-photon p95 <=100 ms and A/V drift <=80 ms, measured with a documented capture method. Server receipt-to-frame latency is a separate measurement, never a proxy for input-to-photon. |

These are acceptance measurements, not new cognition metrics. Record provenance,
calibration method, host load and raw evidence with each run; the repo's metric
quality gate applies before any value enters a model or cognition pipeline.

Rollback: disable the feature, reject new sessions, neutralize/revoke, stop the
worker and service, remove the Hub entry when present. Preserve artifacts. Image
rollback does not authorize loading incompatible snapshots. A service restart
interrupts a game visibly; it is not seamless resume. No sudo restart commands.

## Recommended next patch

1. **Headless persistent authority (P1):** service packaging, fixed catalog,
   session lifecycle, leases, applied actions and local evidence. Validate with
   Pad Lab and fresh-start Nova; no Hub media dependency. Ship tests/evals,
   runbook, independent review and CI. After merge, deploy from main and show
   the named persistent container, readiness and actual input/frame evidence.
2. **Playable Hub (P2):** resolve auth first, then media and human control with
   failure/latency acceptance. Save UI appears only for games with passing gates.
3. **Scripted Orion control (P3):** explicit policy identity, bounded control,
   human takeover and one separately reviewed cognition experiment. Infrastructure
   ready and learning demonstrated remain separate verdicts.

This design PR requires only documentation/privacy checks and independent review.
It does not create a service or require deployment. The next implementation PR,
unlike the diagnostic kit, must leave a verifiably running service after deployment.
