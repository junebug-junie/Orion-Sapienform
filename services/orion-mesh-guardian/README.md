# orion-mesh-guardian

Detects half-dead bus consumers on the chat critical path, publishes Hub Pending Attention cards via notify, and optionally auto-remediates rostered services through docker compose.

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
4. Enable `MESH_GUARDIAN_AUTO_REMEDIATE=true` only after evidence.

## Probes

- HTTP: roster `ready_url` (must return 200 and JSON with `ok: true` where applicable)
- Redis: PING + `PUBSUB NUMSUB` on intake channels
- Equilibrium: snapshot layer for degraded/down beyond grace

## Tests

```bash
PYTHONPATH=.:services/orion-mesh-guardian ../../venv/bin/python -m pytest services/orion-mesh-guardian/tests/ -q
./scripts/smoke_mesh_guardian.sh
```
