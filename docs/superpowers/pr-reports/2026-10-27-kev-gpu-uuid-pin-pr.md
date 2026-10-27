## Summary

- Kev now picks its circe graphics card by the card's fixed ID (UUID), not by its position number (index). Positions can change; UUIDs cannot.
- `KEV_GPU_DEVICE_ID` in `services/orion-kev/.env_example` is now `GPU-599d16a7-8c7c-1e3a-ac94-a01e5cb5cf7a`, the card Kev runs on today. It was `3`.
- New gate `services/orion-kev/tests/test_gpu_pin.py` reads the compose file as YAML. It requires every `device_ids` entry to be the required env key, and the checked-in value to be a UUID. It mirrors affectgpt's gate from #2894.
- The contract test joins the pool's `kev` card (`gpu3`) to the UUID through a map of the four circe card UUIDs, all checked live.

## Outcome moved

Before: if circe numbered its cards differently, Kev could start on the wrong card. That is the failure that crash-looped orion-affectgpt-worker 601 times (2026-10-14..23). After: a renumbering cannot move Kev. The change takes effect only after circe's own `.env` is edited (see Restart required).

## Current architecture

`services/orion-kev/docker-compose.yml` reserves one card through `deploy.resources.reservations.devices[].device_ids: ["${KEV_GPU_DEVICE_ID:?...}"]` and sets `CUDA_VISIBLE_DEVICES=0` inside the container. The value was the bare index `3`. `config/gpu_pool.yaml` lists Kev as role `kev` on card `gpu3` and names cards by index only (`CardSpec` is `extra="forbid"`, with no UUID field).

## Architecture touched

orion-kev only: env template, compose comment and error text, README, tests. No code reads `KEV_GPU_DEVICE_ID` as an int. Only compose consumes the key. The reviewer searched services/, scripts/, mesh-utilities/ and workflows to confirm this.

## Files changed

- `services/orion-kev/.env_example`: the pin value is now Kev's live card UUID, with the reason.
- `services/orion-kev/docker-compose.yml`: comment and fail-loud message now say "UUID".
- `services/orion-kev/README.md`: wiring table, the reason for the pin, and what to move together.
- `services/orion-kev/tests/test_gpu_pin.py`: new YAML-parsed gate.
- `services/orion-kev/tests/test_orion_kev_contract.py`: pool join by UUID map; the map must cover exactly the pool's circe cards; inline-comment stripping.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: the value of `KEV_GPU_DEVICE_ID` is now a GPU UUID (Docker's NVIDIA runtime accepts an index, a PCI bus ID, or a UUID in device_ids).
- Compatibility notes: an old `.env` with `3` still works today, because index 3 is still the same card. It just stays exposed to renumbering.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: yes, the value of `KEV_GPU_DEVICE_ID` changed.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: ran it. The script does not overwrite an existing diverged value without `--force`. It reported athena's `services/orion-kev/.env` as `KEV_GPU_DEVICE_ID=1`, a stale value from before Kev moved to circe. Kev never runs on athena, so that value is inert.
- Skipped keys requiring operator action: `KEV_GPU_DEVICE_ID` on circe holds `3`, read live on 2026-10-27. `sync_local_env_from_example.py --all-keys orion-kev` will **not** change it (it reports the key as diverged). Use the targeted edit below.

## Tests run

```text
.venv/bin/python -m pytest services/orion-kev/tests tests/scripts/test_host_allowlist_buildable.py -q   -> 11 passed
mutation: KEV_GPU_DEVICE_ID=3 in .env_example -> test_env_example_key_is_a_uuid and test_gpu_pin_matches_gpu_pool_kev_role FAIL; restored -> pass
python scripts/check_env_template_parity.py   -> PASS (95 services)
git diff --check   -> clean
```

## Evals run

```text
None. orion-kev has no eval harness, and this is a deploy-pin change with no behavior to score.
```

## Docker/build/smoke checks

```text
Live check (read-only), 2026-10-23 and re-checked 2026-10-27:
  docker inspect orion-circe-kev DeviceRequests -> DeviceIDs ["3"]
  Kev pid 3211620 (python -m kev.serve) -> nvidia-smi gpu_uuid GPU-599d16a7-8c7c-1e3a-ac94-a01e5cb5cf7a
  nvidia-smi -L -> GPU 3: Tesla V100-SXM2-32GB (UUID: GPU-599d16a7-8c7c-1e3a-ac94-a01e5cb5cf7a)
Reviewer: docker compose config with this .env_example resolves device_ids [GPU-599d16a7-...]; an empty key refuses with the new message.
No deploy done (as instructed).
```

## Review findings fixed

- Finding (must): the live pin does not change after merge, because the sync script does not overwrite circe's existing `3`.
  - Fix: Restart required now gives a targeted one-key edit on circe and a `docker inspect` check. The bug counts as fixed on circe only when DeviceIDs shows the UUID.
  - Evidence: circe `.env` line 17 `KEV_GPU_DEVICE_ID=3`; the script's docstring says "diverged" keys need `--force`.
- Finding (should): `CIRCE_CARD_UUIDS` could go stale without any test failing.
  - Fix: assert the map's keys equal the pool's circe card set; the comment states this is a consistency check, not proof of placement.
  - Evidence: 11 passed.
- Finding (nit): `_env_value` did not strip inline comments, unlike `test_gpu_pin.py`.
  - Fix: now strips them.
- Finding (should, not fixed, follow-up): no gate checks the deployed `.env` value. A bare index in the operator `.env` still passes CI and `safe_docker_build.sh`. Affectgpt has the same gap. Proposed follow-up: a pre-`up` check in `safe_docker_build.sh` that refuses a non-UUID `*_GPU_DEVICE_ID`.
- Finding (nit, not fixed): `test_gpu_pin.py` has the same file name in four services. Running pytest on two of them together fails at collection. CI runs each service directory on its own, and the name follows the existing pattern.

## Other bare-index pins found (not changed: not trivial)

- Already stable (PCI bus ID, not index): `VISION_EDGE_GPU_DEVICE_ID`, `WHISPER_TTS_GPU_DEVICE_ID`, `VISION_HOST_GPU_DEVICE_ID`.
- Pinned by index through `CUDA_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES_OVERRIDE`: the atlas llama.cpp lanes (`ATLAS_{CHAT,METACOG,FAST,AGENT,AGENT_BURST}_CUDA_VISIBLE_DEVICES`), `HECATE_AGENT_DEEP_CUDA_VISIBLE_DEVICES`, `DSV41_CUDA_VISIBLE_DEVICES`, `BONSAI_CUDA_VISIBLE_DEVICES`, `CIRCE_QWEN_CUDA_VISIBLE_DEVICES`, and the diffusion-host, world-model, vision-host, llama-cola, llamacpp-host and neural-host `CUDA_VISIBLE_DEVICES` values. These are tied to `config/gpu_pool.yaml` card `index` values and the lane controller's actuation. Moving them to UUIDs needs a `uuid` field on the pool's `CardSpec`. That model is `extra="forbid"` and many services load it, so the change needs a consumer-first migration. Follow-up.

## Restart required

Run each command on circe, one at a time, after this PR merges and circe has pulled main:

```bash
ssh circe@circe 'sed -i "s/^KEV_GPU_DEVICE_ID=.*/KEV_GPU_DEVICE_ID=GPU-599d16a7-8c7c-1e3a-ac94-a01e5cb5cf7a/" /mnt/scripts/Orion-Sapienform/services/orion-kev/.env'
```
```bash
ssh circe@circe 'cd /mnt/scripts/Orion-Sapienform && scripts/safe_docker_build.sh orion-kev up -d --force-recreate kev'
```
```bash
ssh circe@circe 'docker inspect -f "{{json .HostConfig.DeviceRequests}}" orion-circe-kev'
```

The last command must show `"DeviceIDs":["GPU-599d16a7-8c7c-1e3a-ac94-a01e5cb5cf7a"]`. Optional: run `sync_local_env_from_example.py --all-keys orion-kev` on circe afterwards to sync other keys. It does not change this key.

## Risks / concerns

- Severity: medium. Concern: until the circe `.env` edit runs, Kev is still pinned by index. Mitigation: the targeted command above.
- Severity: low. Concern: if Kev's card is physically replaced, the UUID no longer exists and compose fails to start Kev. Mitigation: that is a loud failure (the intended behavior), not a silent move to the wrong card.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2904
