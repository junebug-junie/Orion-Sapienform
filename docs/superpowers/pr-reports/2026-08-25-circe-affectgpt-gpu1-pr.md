## Summary

- Moved `orion-affectgpt-worker`'s GPU pin on circe from GPU2 (Tesla PG500-216) to GPU1 (Tesla V100-SXM2-32GB), per Juniper's request.
- Pure config change: `docker-compose.yml`'s `deploy.resources.reservations.devices.device_ids` `["2"]` → `["1"]`. `AFFECTGPT_DEVICE` stays `cuda:0` (Docker always remaps a pinned physical device to container-local index 0, confirmed live 2026-08-22 and re-confirmed here).
- Found and corrected a stale claim in this session's own reference memory while verifying the target GPU was actually free: GPU1 was documented as "reserved — Orion's chat (:8011, the 35B)". Live evidence shows the real 35B chat process actually runs tensor-split across **GPU0+GPU3**, not GPU1. GPU1 was genuinely idle (0MiB, no compute-apps entry).

## Outcome moved

`orion-affectgpt-worker` now runs on GPU1 instead of GPU2, freeing GPU2 (PG500) back up. Confirmed live: `docker inspect` shows `DeviceIDs=["1"]`, `nvidia-smi` shows GPU1 memory went 0 → 18447MiB after model load while GPU2 stayed at 0MiB, and a full functional request against the real running container succeeded end to end.

## Current architecture

`orion-affectgpt-worker` is a single-GPU, single-model FastAPI service on circe running AffectGPT (Qwen2.5-7B-Instruct + CLIP ViT-L + HuBERT-L, LoRA fine-tuned) for Juniper's facial+vocal affect assessment, consumed via `orion:exec:request:AffectGptWorkerService` on the bus and directly via `POST /v1/affect/assess`. GPU pinning is done via Docker Compose's `deploy.resources.reservations.devices` block (Compose's `gpus:` top-level key only accepts `'all'` — confirmed live 2026-08-22).

## Architecture touched

`services/orion-affectgpt-worker/docker-compose.yml`, `.env_example`, `README.md`. No code, schema, or bus changes.

## Files changed

- `services/orion-affectgpt-worker/docker-compose.yml`: `device_ids: ["2"]` → `["1"]`; updated the explanatory comment above it to record the move, why GPU1 was picked, and the correction to the stale reference doc.
- `services/orion-affectgpt-worker/.env_example`: header comment "circe (GPU2)" → "circe (GPU1, ... as of 2026-08-25 ...)".
- `services/orion-affectgpt-worker/README.md`: top-line GPU description and the "Provenance" section note both updated to reflect the move without erasing the original 2026-08-22 live-verification record (that evidence is still valid — the model/pipeline didn't change, only which physical card it runs on).
- `services/orion-affectgpt-worker/.env` (circe's primary checkout and this deploy worktree, both gitignored, not part of this commit): hand-synced comment to match `.env_example`, per CLAUDE.md env parity rule. No env *keys* changed.

## Schema / bus / API changes

None.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes (comment only, no key/value change).
- local `.env` synced: yes, by hand (this change doesn't touch any key `sync_local_env_from_example.py` would add/remove — it's a comment-only diff — so the script wasn't the right tool here; edited directly on circe's primary checkout and the new deploy worktree).
- skipped keys requiring operator action: none.

## Tests run

None applicable — no code changed, this is a deploy-config-only change (GPU device pin + comments). No unit test exists or should exist for "which physical GPU index a compose file names."

## Evals run

No eval harness exists for this service (noted already in its own PR history). Live functional verification below is the real evidence.

## Docker/build/smoke checks

All on the real circe host, from a fresh worktree (`chore/circe-affectgpt-gpu1`, not the shared checkout):

```text
$ bash scripts/safe_docker_build.sh orion-affectgpt-worker up -d --build
  Image orion-affectgpt-worker-affectgpt-worker Built
  Container orion-circe-affectgpt-worker Recreated / Started

$ docker inspect orion-circe-affectgpt-worker --format '{{json .HostConfig.DeviceRequests}}'
  [{"Driver":"nvidia","Count":0,"DeviceIDs":["1"],"Capabilities":[["gpu"]],"Options":null}]

$ nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
  (after model warm-load)
  1, Tesla V100-SXM2-32GB, 18447 MiB, 0 %
  2, Tesla PG500-216, 0 MiB, 0 %          <- confirms it left GPU2

$ curl -s -m 90 -X POST http://localhost:32798/v1/affect/assess \
    -d '{"video_path": ".../demo/sample_00000000.mp4", "audio_path": ".../demo/sample_00000000.wav"}'
  {
    "ok": true,
    "raw_response": "... real grounded description of the demo clip ...",
    "face_detection": {"detection_rate": 1.0, "frames_total": 88, "frames_detected": 88},
    "subtitle_source": "transcribed",
    "transcript": "I don't know. I don't know how to explain this.",
    "timings": {"transcribe_s": 1.149, "data_load_s": 0.084, "encode_s": 0.62, "generate_s": 13.535, "total_s": 15.387}
  }
```

Full pipeline (Whisper transcription, face detection, AffectGPT inference) confirmed working end to end on the new card.

## Review findings fixed

Code review dispatched via the code-review skill in a subagent (medium effort) against this diff. [Fill in from review results before merge if not already reflected here.]

## Restart required

Already done as part of live verification — `orion-circe-affectgpt-worker` is live on this branch's commit right now on circe. No further action needed once this merges to `main`, unless `main` diverges before another deploy.

## Risks / concerns

- Severity: low. Concern: this session's own `reference_circe_gpu_inventory_and_lane_map` memory had a stale, incorrect GPU1 assignment ("reserved for chat") that could mislead a future agent into avoiding GPU1 unnecessarily, or worse, assuming GPU0/GPU3 have headroom they don't. Mitigation: memory corrected in this session with the live evidence that superseded it; the docker-compose.yml comment also records the correction inline so it survives even if the memory file is lost.
- Severity: low. Concern: GPU1 has no compose-level guard preventing another service from later being pinned there too. Mitigation: none needed beyond what already exists elsewhere in this repo (manual lane-map convention) — out of scope for this patch.

## PR link

(filled in after creation)
