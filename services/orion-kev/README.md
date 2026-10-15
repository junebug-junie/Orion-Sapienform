# orion-kev

Local [Kev](https://github.com/jaredpalmer/kev) server that speaks TypeSafe's
`POST /v1/systemone` API. It runs on **circe**, GPU 3, beside the fast and
metacog llama.cpp lanes. Orion's substrate appraisal on athena calls it as
`http://100.112.254.99:8009` (`SUBSTRATE_SYSTEM_ONE_BASE_URL`).

## What it does

Runs a small decision model (default `jaredpalmer/kev-0.8b`, ~4GB VRAM) on a
pinned GPU and answers typed score/choice/noul questions. Orion uses it for
System One appraisal frames. Per-question consumers are promoted independently:
`curiosity_pull` is a live endogenous-curiosity admission gate; the other
initial questions remain observational. If Kev is down, the gate fails open.

## Bring-up (circe)

Deploy from circe's clean primary `main` checkout:

```bash
cp services/orion-kev/.env_example services/orion-kev/.env   # first time
python scripts/sync_local_env_from_example.py orion-kev      # if keys exist

scripts/safe_docker_build.sh orion-kev up -d --build

docker ps --filter name=kev
curl -fsS http://127.0.0.1:8009/docs >/dev/null && echo ok
```

Then from athena: `curl -fsS http://100.112.254.99:8009/docs >/dev/null && echo ok`.

Restart policy is `unless-stopped` — Docker brings the container back after
reboot. No systemd linger required.

## Wiring

| Surface | Value |
|---------|--------|
| Host | circe (`100.112.254.99`) |
| Container name | `${PROJECT}-kev` → `orion-circe-kev` |
| Network | `app-net` (external) |
| Port | `8009` |
| GPU | `KEV_GPU_DEVICE_ID=3` (host nvidia-smi index) |
| GPU pool | role `kev` on `gpu3` in `config/gpu_pool.yaml` (VRAM accounting + probe) |
| Weights cache | `KEV_HF_CACHE_HOST_DIR` |

If Kev moves to another card, change `KEV_GPU_DEVICE_ID` and the `kev` role's
`cards` together.

Substrate (athena) must have:

```bash
SUBSTRATE_SYSTEM_ONE_APPRAISAL_ENABLED=true
SUBSTRATE_SYSTEM_ONE_BASE_URL=http://100.112.254.99:8009
```

Post-deploy smoke:

```bash
python scripts/smoke_system_one_appraisal.py
```

## Privacy

The request body is a bounded attention summary (no raw chat). It is still
derived from private context, so Kev stays on Orion's own hosts (athena or
circe, over Tailscale). Pointing the URL at a hosted provider is an explicit
privacy-boundary change.
