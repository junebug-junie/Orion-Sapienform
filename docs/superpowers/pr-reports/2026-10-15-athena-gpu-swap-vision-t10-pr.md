## Summary

- The vision host moves from athena's small card (Tesla P4, 7.68 GB) to the big one (Tesla T10, 16 GB). It shares the T10 with the qwen captioner.
- Speech (whisper-tts) and the cam0 YOLO edge detector (vision-edge) move to the P4. Speech runs a few times a month, so it takes the small card.
- All three containers now get exactly one card from compose `device_ids`, named by host PCI bus id. No more `gpus: all`, no "index means something different inside and outside" guessing.
- The vision host's memory gate now checks the card the work really runs on. Before, `cuda:N` was used as both a torch index and an NVML index. That only agreed by luck. The qwen lane ran on the T10 and checked the P4's free memory.

## Outcome moved

- The P4 was at 100 % utilization serving the vision host alone. The T10 is a Turing card with fp16 tensor cores, and most vision profiles run fp16 (`dtype: auto`), which Pascal runs at a small fraction of fp32 speed. Expected: faster vision tasks. UNVERIFIED until deploy.
- vision-edge no longer lands on the T10 by accident through CUDA's fastest-first device order.
- A wrong-card admission check (qwen lane) is fixed in code; it takes effect when that lane is next rebuilt.

## Current architecture

- vision-host: `gpus: all`, `CUDA_VISIBLE_DEVICES=0`, `VISION_DEVICES=cuda:0`. NVML saw both cards, torch saw only the P4. Admission (`app/gpu.py`, `pick_best_gpu`) used the `cuda:N` number as an NVML index.
- whisper-tts: `device_ids: ["${WHISPER_TTS_GPU_DEVICE_ID}"]` = `1` (T10).
- vision-edge: `runtime: nvidia` plus a device request with no ids (all cards), `YOLO_DEVICE=0` -> landed on the T10.
- qwen lane (`docker-compose.circe-qwen.yml`): `gpus: all`, `CUDA_VISIBLE_DEVICES=1`, `VISION_DEVICES` unset -> `cuda:0` -> admission read NVML 0 = the P4.
- Live 20:45 UTC: P4 = vision host 5632 MiB. T10 = kev 3892 + whisper 2830 + edge 288 + qwen 6556 = 13566 MiB.

## Architecture touched

- `services/orion-vision-host`: compose GPU pin, scheduler index mapping, env, README.
- `services/orion-whisper-tts`: env value (P4 PCI bus id), comments, README.
- `services/orion-vision-edge`: compose GPU pin, `runtime: nvidia` removed, env, README.
- `config/vision_profiles.yaml`: one stale comment.
- `config/gpu_pool.yaml`: checked, no change. It lists only circe and hecate cards; athena's vision host, speech and edge are not pool roles. No contract test pins athena GPU roles.

## Files changed

- `services/orion-vision-host/docker-compose.yml`: `gpus: all` -> `device_ids` from required `VISION_HOST_GPU_DEVICE_ID`.
- `services/orion-vision-host/app/gpu.py`: new `torch_to_nvml_index` (inverse of `nvml_to_torch_index`).
- `services/orion-vision-host/app/scheduler.py`: map `cuda:N` -> NVML before the gate, map the pick back to the torch index.
- `services/orion-vision-host/.env_example`: `VISION_HOST_GPU_DEVICE_ID=00000000:D8:00.0`; notes that `CUDA_VISIBLE_DEVICES`/`VISION_DEVICES` are container-relative; expected T10 numbers.
- `services/orion-vision-host/README.md`: GPU placement section.
- `services/orion-vision-host/tests/test_gpu_index_mapping.py`: mapping, both container layouts, own-cache lookup, expected T10 numbers, compose/env pins.
- `services/orion-whisper-tts/.env_example`, `docker-compose.yml`, `README.md`: P4 pin, Pascal check results.
- `services/orion-whisper-tts/tests/test_gpu_pin.py`: pin + `fp16=False` gate.
- `services/orion-vision-edge/docker-compose.yml`, `.env_example`, `README.md`: P4 pin.
- `services/orion-vision-edge/tests/test_gpu_pin.py`: pin gate.
- `config/vision_profiles.yaml`: `tile_batch` comment.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: vision-host admission reads the NVML card mapped from `CUDA_VISIBLE_DEVICES`. Identical for the shared host (one card, identity). Fixes the qwen lane on next rebuild.
- Compatibility notes: `VISION_HOST_GPU_DEVICE_ID` and `VISION_EDGE_GPU_DEVICE_ID` have no default; a bring-up without them fails loudly.

## Env/config changes

- Added keys: `VISION_HOST_GPU_DEVICE_ID` (orion-vision-host), `VISION_EDGE_GPU_DEVICE_ID` (orion-vision-edge)
- Removed keys: `GPU_INDEX` (orion-whisper-tts; read only by commented-out compose lines; removed by hand from local `.env` too)
- Renamed keys: none
- Changed value: `WHISPER_TTS_GPU_DEVICE_ID` `1` -> `00000000:86:00.0`
- `.env_example` updated: yes, all three services
- local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys orion-vision-host orion-whisper-tts orion-vision-edge`: yes, two keys added.
- skipped keys requiring operator action: `WHISPER_TTS_GPU_DEVICE_ID` was reported diverged (local `1`). Changed by hand in the primary checkout's `services/orion-whisper-tts/.env` to `00000000:86:00.0`.

## Tests run

```text
orion-whisper-tts (host venv):   85 passed
orion-vision-host (in image):    4 failed, 327 passed, 1 skipped
  main baseline (same image):    4 failed, 310 passed, 1 skipped  -- same 4, pre-existing
  (test_heartbeat_chassis x2, test_identity_face_broadcast_suppression x2)
orion-vision-edge (in image):    1 failed, 13 passed, 1 collection error
  main baseline:                 1 failed, 11 passed, 1 collection error -- pre-existing
check_env_template_parity, check_env_key_single_source, check_gpu_pool_config,
check_compose_no_relative_mounts, check_compose_no_host_claude_json_mount,
check_service_hostname_refs, check_metric_lineage --gate,
check_definition_drift --gate, check_service_env_compose_parity x3: PASS
git diff --check: clean
```

## Evals run

```text
No eval harness for these services. The live checks below stand in for one.
```

## Docker/build/smoke checks

All run as throwaway `docker run --rm` probes. No service container was started, stopped or recreated.

```text
docker device id forms (edge image, nvidia-smi -L):
  device=00000000:86:00.0                    -> GPU 0: Tesla P4 only
  device=GPU-49b63420-...                    -> GPU 0: Tesla P4 only
  device=0                                   -> GPU 0: Tesla P4 only
  --runtime nvidia + device=00000000:86:00.0 -> "unresolvable CDI devices" (why runtime: nvidia is removed)

vision-host image, device=00000000:D8:00.0, CUDA_VISIBLE_DEVICES=0, branch code:
  NVML [(0, 'Tesla T10', 16384)]
  torch 1 Tesla T10
  torch_to_nvml_index(0)=0, nvml_to_torch_index(0)=0
  scheduler: gpu map torch->nvml {0: 0}; picked cuda:0 (nvml=0 Tesla T10); own_reusable_mb 512 counted

vision-edge image, device=00000000:86:00.0, YOLO_DEVICE=0:
  torch 2.4.0+cu121 device_count 1, Tesla P4, arch list includes sm_60
  YOLO yolov8n predict -> device cuda:0 = Tesla P4

whisper-tts image, device=00000000:86:00.0:
  torch 2.8.0+cu126, Tesla P4 (6, 1), arch ['sm_50','sm_60','sm_70',...]
  fp32 + fp16 matmul OK
  whisper "base", fp16=False, one 30 s window: P4 0.373 s, T10 0.099 s (encoder 0.079 vs 0.015 s), peak 595 MiB

docker compose config (worktree compose, primary .env):
  vision-host device_ids ['00000000:D8:00.0'], CUDA_VISIBLE_DEVICES=0, VISION_DEVICES=cuda:0
  whisper-tts device_ids ['00000000:86:00.0']
  vision-edge device_ids ['00000000:86:00.0'], no runtime, YOLO_DEVICE=0
```

## Pascal (P4) verdict for speech

Compatible. The image's torch 2.8.0+cu126 ships sm_60 kernels, which run on the P4's sm_61. XTTS loads through coqui `TTS(gpu=True)` with no half/bf16 cast, so fp32. Whisper is called with `fp16=False` (pinned by a test now). Nothing needs sm_70+. XTTS speed was measured GPU-independent before. Whisper STT is about 3.7x slower on the P4 (0.37 s vs 0.10 s per 30 s window) -- still far under the 90 s STT timeout.

## Expected memory after the move

| Card | Holds | Used | Free |
|---|---|---|---|
| P4 (7680 MiB) | whisper-tts ~2830 + vision-edge ~288 | ~3120 MiB | ~4560 MiB |
| T10 (16384 MiB) | qwen ~6556 + vision host ~5632 | ~12190 MiB | ~4190 MiB |

Vision-host gate: admits when NVML free + own idle cache - 1200 >= 800, so it needs 2000 MiB. At ~4190 free it admits even with zero own cache. Qwen counts as another process (used), so qwen can grow ~2.2 GB before the vision host refuses. Qwen lane gate (after its next rebuild): 4190 - 512 >= 512, admits. Until then it still reads the P4 (~4560 free), which also admits.

## Review findings fixed

- Finding (should): an unmappable `cuda:N` fell back to "same NVML number", which could gate a non-existent device on another card's memory (`CUDA_VISIBLE_DEVICES=1`, `cuda:0,cuda:1`).
  - Fix: identity fallback only for non-numeric `CUDA_VISIBLE_DEVICES` (UUID/MIG, old behaviour); a numeric list that does not cover `cuda:N`, or a duplicate mapping, drops that device with an error log.
  - Evidence: `test_numeric_cvd_drops_a_device_it_does_not_cover`.
- Finding (should): stale qwen-lane comments (`docker-compose.circe-qwen.yml`) said retina/detection stay on the P4 and justified the low 512 floors with whisper-tts as the neighbour.
  - Fix: comments updated; the floor asymmetry (qwen admits at ~1 GB free, shared host needs 2 GB) is now named there and in Risks below.
  - Evidence: diff.
- Finding (nit): refusal text mixed NVML and torch index spaces.
  - Fix: rows render `gpu=cuda:<torch> (nvml=<n>)` when they differ; "no candidate" prints the torch->nvml map.
  - Evidence: `test_refusal_text_names_both_index_spaces`.
- Finding (nit): test reserve/floor were hard-coded.
  - Fix: read from `.env_example`.
- Finding (nit): `.env_example` "qwen can grow ~2.2GB" assumed zero own cache.
  - Fix: comment says so.
- Finding (nit): stale `GPU_INDEX=2` in whisper-tts `.env_example`, read only by commented-out compose lines.
  - Fix: removed from `.env_example`, compose comments, and by hand from the primary checkout's local `.env`.
- Finding (nit): whisper-tts pin test did not ignore comment lines.
  - Fix: strips comments like the other two.

Re-run: vision-host in image 329 passed / same 4 pre-existing failures; whisper-tts 85 passed; env parity PASS.

## Restart required

Deploy order matters. Running whisper-tts first would put the vision host (5.6 GB) and speech (2.8 GB) on the 7.68 GB P4 together. From clean primary `main` synced with `origin/main`, after this merges:

```bash
docker stop orion-athena-kev
./scripts/safe_docker_build.sh orion-vision-edge up -d --build
docker stop orion-athena-whisper-tts
./scripts/safe_docker_build.sh orion-vision-host up -d --build
./scripts/safe_docker_build.sh orion-whisper-tts up -d --build
```

Post-deploy checks:

```bash
nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv
for c in orion-athena-vision-host orion-athena-whisper-tts orion-athena-vision-edge; do docker inspect $c --format "$c {{json .HostConfig.DeviceRequests}}"; done
journalctl CONTAINER_NAME=orion-athena-vision-host --since "-5 min" --no-pager | grep -c '"ok": true'
journalctl CONTAINER_NAME=orion-athena-vision-host --since "-5 min" --no-pager | grep -c '"ok": false'
curl -s localhost:32797/health | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['ok'],d['scheduler']['gpu_hard_floor_refusal_streak'])"
curl -s localhost:7800/health
docker exec orion-athena-whisper-tts python3 scripts/smoke_xtts.py
```

Expect: P4 bus 86:00.0 holds whisper-tts + vision-edge, T10 bus D8:00.0 holds qwen + vision host. Baseline before the move was 95 `"ok": true` and 0 `"ok": false` in 5 minutes; refusal streak 0; whisper `cuda_available: true`.

## Risks / concerns

- Severity: medium
  - Concern: the primary checkout's `services/orion-whisper-tts/.env` already says the P4. A whisper-tts recreate from main before the vision host moves would put both on the P4 (~8.4 GB on 7.68 GB).
  - Mitigation: follow the order above. A plain `docker restart` keeps the old pin; only compose `up` re-reads it.
- Severity: low
  - Concern: the T10 is now compute-shared by qwen (walkway captions, ~2-4 s each) and the vision host. Both run fp16 on tensor cores, but they time-slice.
  - Mitigation: check task latency in the `[VISION_TASK]` log lines after deploy.
- Severity: low
  - Concern: whisper-tts reported `Failed to initialize NVML: Unknown Error` from `nvidia-smi` inside the current container (torch CUDA still available). Pre-existing; a recreate should clear it.
  - Mitigation: the post-deploy `/health` and smoke call.
- Severity: low
  - Concern: qwen lane admits at ~1 GB free (512/512 floors), the shared vision host needs 2 GB. Under memory pressure on the T10, qwen keeps admitting while the shared host refuses.
  - Mitigation: qwen runs one task at a time with bounded pixels; watch the shared host's `/health` refusal streak after deploy.
- Severity: low
  - Concern: the qwen lane's compose still uses `gpus: all` + `CUDA_VISIBLE_DEVICES=1`. Correct for it after its next rebuild, but not pinned by `device_ids`.
  - Mitigation: follow-up to pin it the same way.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2697

🤖 Generated with [Claude Code](https://claude.com/claude-code)
