# PR report: orion-kev Docker service

## Summary

- Add `services/orion-kev`: GPU Docker image for Jared Palmer's Kev
  (`POST /v1/systemone`), named `orion-athena-kev` on `app-net`.
- Restart `unless-stopped` so reboot does not depend on systemd linger.
- Wire env sync (`KEV_` prefix + DEFAULT_SERVICES) and point substrate README
  at the new bring-up path.

## Outcome moved

System One shadow appraisal has an in-repo, Docker-owned Kev peer instead of a
host/systemd process + socat shim.

## Architecture touched

- New service: `services/orion-kev`
- Consumer docs: `services/orion-substrate-runtime/README.md`
- Env sync: `scripts/sync_local_env_from_example.py`

## Env/config changes

- Added keys (orion-kev): `PROJECT`, `KEV_HOST_PORT`, `KEV_GPU_DEVICE_ID`,
  `KEV_HF_CACHE_HOST_DIR`, `KEV_MODEL`
- `.env_example` updated: yes
- local `.env` synced: yes (primary checkout)

## Tests run

```text
pytest services/orion-kev/tests/test_orion_kev_contract.py -q
# 3 passed
```

## Restart required

```bash
scripts/safe_docker_build.sh orion-kev up -d --build
```

Live Athena already runs an equivalent container cut over from `/mnt/scripts/kev`;
rebuild from this service when convenient so the image matches the repo Dockerfile.
