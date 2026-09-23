# orion-kev

Local [Kev](https://github.com/jaredpalmer/kev) server that speaks TypeSafe's
`POST /v1/systemone` API. Orion's substrate shadow appraisal calls this as
`http://orion-athena-kev:8009` (`SUBSTRATE_SYSTEM_ONE_BASE_URL`).

## What it does

Runs a small decision model (default `jaredpalmer/kev-0.8b`) on a pinned GPU
and answers typed score/choice/noul questions. Orion uses it for
**behavior-inert** System One shadow frames only — no consumer reads those
frames yet.

## Bring-up (Athena)

```bash
# from a worktree
cp services/orion-kev/.env_example services/orion-kev/.env   # first time
python scripts/sync_local_env_from_example.py orion-kev      # if keys exist

scripts/safe_docker_build.sh orion-kev up -d --build

docker ps --filter name=kev
curl -fsS http://127.0.0.1:8009/docs >/dev/null && echo ok
```

Restart policy is `unless-stopped` — Docker brings the container back after
reboot. No systemd linger required.

## Wiring

| Surface | Value |
|---------|--------|
| Container name | `${PROJECT}-kev` → `orion-athena-kev` |
| Network | `app-net` (external) |
| Port | `8009` |
| GPU | `KEV_GPU_DEVICE_ID` (host nvidia-smi index) |
| Weights cache | `KEV_HF_CACHE_HOST_DIR` |

Substrate must have:

```bash
SUBSTRATE_SYSTEM_ONE_APPRAISAL_ENABLED=true
SUBSTRATE_SYSTEM_ONE_BASE_URL=http://orion-athena-kev:8009
```

Post-deploy smoke:

```bash
python scripts/smoke_system_one_appraisal.py
```

## Privacy

The request body is a bounded attention summary (no raw chat). Still derived
from private context — keep Kev local on Athena; pointing the URL at a hosted
provider is an explicit privacy-boundary change.
