# orion-mesh-guardian

Detects half-dead bus consumers on the chat critical path, publishes Hub Pending Attention cards via notify, and optionally auto-remediates rostered services through docker compose.

Also publishes a bus-native `SystemHealthV1` heartbeat to `orion:system:health` every
`HEARTBEAT_INTERVAL_SEC` (default 10s), on its own independent bus connection, separate from
the equilibrium-snapshot watcher above -- process liveness for the guardian itself, not the
services it watches.

## Deploy

```bash
cd services/orion-mesh-guardian
cp .env_example .env   # set PROJECT, ORION_BUS_URL, NOTIFY_* , MESH_GUARDIAN_*
docker compose --env-file .env -f docker-compose.yml up -d --build
```

Requirements:

- Join `app-net`
- Mount repo read-only at `/repo` (roster + compose files)
- Mount `/var/run/docker.sock` when auto-remediation is enabled
- Equilibrium snapshots on `CHANNEL_EQUILIBRIUM_SNAPSHOT` (default `orion:equilibrium:snapshot`)

## Rollout

1. Ship with `MESH_GUARDIAN_AUTO_REMEDIATE=false` (observe-only).
2. Confirm Pending Attention cards on induced failures.
3. Run live-stack acceptance (`scripts/smoke_mesh_guardian.sh` + Task 22 checklist in implementation plan).
4. Enable `MESH_GUARDIAN_AUTO_REMEDIATE=true` only after evidence. **Enabled 2026-10-02** after the docker CLI/compose/buildx fix and the worktree-deploy guard (PR #2475); compose `--dry-run` verified for every `auto_remediate` roster service.

## Probes

- HTTP: roster `ready_url` (must return 200 and JSON with `ok: true` where applicable)
- Redis: PING + `PUBSUB NUMSUB` on intake channels
- Equilibrium: snapshot layer for degraded/down beyond grace

## GPU actuation watch

Added after the 2026-10-09/10 incident: circe's `orion-gpu-lane-controller` ran an image older than
`config/gpu_pool.yaml`, could not parse its own config, and refused every GPU swap
(`config_unloadable:ValidationError`, 155 times in 25.7 h). Its `/health` stayed ok and nothing alerted.

`app/gpu_watch.py` (pure) + `_gpu_loop` / `_gpu_event_loop` in `app/service.py`:

- **Active**, every `MESH_GUARDIAN_GPU_PROBE_INTERVAL_SEC` (300): for each role with a `launch` block in
  the live `<ORION_REPO_ROOT>/config/gpu_pool.yaml`, sends its controller the `digest` probe
  (`orion/gpu_pool/actuator_probe.py`, shared with `scripts/gpu_pool_actuator_probe.py`; waits up to
  `MESH_GUARDIAN_GPU_PROBE_WAIT_SEC`, 90). Never `status`: it makes the controller replay its last result
  under the pool's action_id, which can flip the pool's belief about a card (operator CLI only).
  Own loop, so it never delays the 60 s stability loop.
  Cards: config unloadable -> critical; launch digest mismatch -> error (may be the guardian's own image);
  no answer, or a non-transient refusal (anything but busy / deadline_passed / stale_generation), on
  2 cycles in a row -> error.
- **Passive**: listens on `orion:gpu_pool:event`. A pool `actuate_refused` whose reason is
  `config_unloadable:*` or `launch_digest_mismatch` -> critical at once; any 3 refusals for one role
  within 60 min -> error.
- Both go through the same `AlertGate` as the stability checks (key per role + kind, shared by both views):
  one card per 6 h while the condition lasts, except that a higher severity on the same key goes out at once. If the guardian itself cannot parse the live YAML (its own
  image is older), it raises a card saying so instead of going blind.
- Probes use their own `probe-*` action_ids: the controller's `status` writes no fence state and the digest
  `load` is refused before any docker call or generation; the pool drops results for action_ids it did not
  issue (see `orion/gpu_pool/actuator_probe.py` docstring for the one replay side effect).
- The event watch re-subscribes 5 s after an error; events in that gap are lost (the probes cover the
  same failures).
- Disable with `MESH_GUARDIAN_GPU_WATCH_ENABLED=false`.

Eval: `evals/test_gpu_incident_replay.py` replays the real 155 refusals (exported from `gpu_pool_events`):
first card within 0 s of the first refused swap, worst case 295 s from probes alone; 5 cards over 25.7 h.

## Crash loops on other hosts (circe)

Added after 2026-10-14..23: `orion-circe-affectgpt-worker` restarted 601 times on circe (CUDA OOM at
model load, ~144 W for 9 days) and nothing alerted, because the crash-loop check read only athena's
docker socket.

- **Producer:** circe's `orion-gpu-lane-controller` (already holds circe's docker socket, already
  heartbeats) reads every container's `RestartCount` every 30 s (read-only `docker ps -aq` +
  `docker inspect`) and puts it on its heartbeat as `details["container_restarts"]`
  (`ContainerRestartsV1`, `orion/schemas/telemetry/system_health.py`).
- **Carrier:** orion-equilibrium-service already copies each heartbeat's `details` into the
  equilibrium snapshot this guardian already consumes. No new channel, no SSH, no remote socket.
- **Consumer:** `remote_restart_counts` (`app/stability.py`) folds every non-local host's fresh
  census into the same `CrashLoopTracker` as athena's own. Every key is host-qualified
  (`<host>/<container>@<id12>`), so the card is `crash_loop:circe/<container>`.
- **Absence is not calm.** A host in `MESH_GUARDIAN_REMOTE_CONTAINER_HOSTS` (default `circe`) with no
  census, a failed one, or one older than `MESH_GUARDIAN_REMOTE_CONTAINER_STALE_SEC` (300) raises a
  `container_watch_blind:<host>` warning card (after a 300 s boot grace) and feeds nothing. A host
  that cannot be read keeps its restart history. hecate is on-demand and has no reporter yet, so it
  is not listed; a hecate census would still be watched if one appeared.
- Disable with `MESH_GUARDIAN_REMOTE_CONTAINER_WATCH_ENABLED=false`.

Eval: `evals/test_remote_crash_loop_incident_replay.py` replays the circe incident through the
real hops (30 s census, 15 s equilibrium lag, 60 s cycle). At the conservative cadence (601 restarts
over 235 h, one per 23.5 min) the first card lands at 72 min, inside the 2 h window; at the reported
~22 s cadence, at cycle 2. The whole incident would have raised 40 cards (one per 6 h) instead of 0.
A reporter that stops heartbeating reads blind 5 min later.

## Tests

```bash
PYTHONPATH=.:services/orion-mesh-guardian ../../venv/bin/python -m pytest services/orion-mesh-guardian/tests/ -q
./scripts/smoke_mesh_guardian.sh
```
