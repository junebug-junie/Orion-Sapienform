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

### Moving from athena (order matters)

1. Deploy on circe (above).
2. From athena: `curl -fsS http://100.112.254.99:8009/docs >/dev/null && echo ok`.
3. Restart orion-substrate-runtime on athena so it reads the circe URL.
   Check its log for `substrate_system_one_appraisal_completed`.
4. Stop the old athena container so it frees ~4GB on the T10:
   `scripts/safe_docker_build.sh orion-kev down` from athena's primary checkout.

If substrate restarts before step 1, appraisals fail and the curiosity gate
fails open (curiosity runs ungated) until Kev answers on circe.

Restart policy is `unless-stopped` — Docker brings the container back after
reboot. No systemd linger required.

## Wiring

| Surface | Value |
|---------|--------|
| Host | circe (`100.112.254.99`) |
| Container name | `${PROJECT}-kev` → `orion-circe-kev` |
| Network | `app-net` (external) |
| Port | `8009` |
| GPU | `KEV_GPU_DEVICE_ID=GPU-599d16a7-8c7c-1e3a-ac94-a01e5cb5cf7a` (circe GPU UUID; the V100-SXM2, nvidia-smi index 3 on 2026-10-23) |
| GPU pool | role `kev` on `gpu3` in `config/gpu_pool.yaml` (listed + probed; never scheduled) |
| Weights cache | `KEV_HF_CACHE_HOST_DIR` |

The pin is a UUID, never a bare index. circe's indices renumbered when cards
moved, and orion-affectgpt-worker's index pin landed it on a full card
(601 restarts, 2026-10-14..23). Find UUIDs with `nvidia-smi -L` on circe.

If Kev moves to another card, change `KEV_GPU_DEVICE_ID`, the `kev` role's
`cards`, and `CIRCE_CARD_UUIDS` in `tests/test_orion_kev_contract.py` together
(the pool config has no UUID field yet).

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
