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

## Tests

```bash
PYTHONPATH=.:services/orion-mesh-guardian ../../venv/bin/python -m pytest services/orion-mesh-guardian/tests/ -q
./scripts/smoke_mesh_guardian.sh
```
