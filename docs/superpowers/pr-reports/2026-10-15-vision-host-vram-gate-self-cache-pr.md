## Summary

- The vision host was refusing all of its own work. Its GPU check counted the host's own idle PyTorch memory as "taken". The check now treats that memory as free for this process.
- Memory held by any other process still counts as taken. The protection against neighbours growing stays in place.
- Every `gpu_hard_floor` refusal now says why, with the numbers: NVML free, own reusable cache, reserve, effective free, floor.
- `GET /health` now reports a refusal count, the current refusal streak, the last reason, and the last admit time. A full refusal streak is no longer invisible on `/health`.

## Outcome moved

Live incident, 2026-10-15 from 18:54 UTC: `orion-athena-vision-host` refused every task (707 refusals between 18:54 and 19:23 UTC, last success 18:54:04). Orion's eye was blind. After deploy, the host can reuse its own cached GPU memory, so it stops starving itself.

## Current architecture

- `services/orion-vision-host/app/gpu.py` `pick_best_gpu`: admit when `nvml_free - VISION_VRAM_RESERVE_MB (1200) >= VISION_VRAM_HARD_FLOOR_MB (800)`. So it needs at least 2000 MiB free per NVML.
- NVML reports device-wide memory. PyTorch's caching allocator keeps freed tensors' memory reserved for reuse. NVML shows that memory as used.
- `scheduler.py` `submit`: when no GPU is picked, it calls the handler with `device="cpu"`. `main.py` `_make_execute_handler` then returns `ok=false, device=null, error_code=gpu_hard_floor`. There is no CPU fallback. It refuses outright.
- `app/liveness.py` (from the 2026-08-20 incident) does alert on a sustained refusal rate. It fired here at 19:00:37 UTC (`[LIVENESS] 100% of vision tasks failing (gpu_hard_floor) -> attention request`). `/health` showed nothing.

## Architecture touched

Only `orion-vision-host`. No bus, schema, or env changes.

## Files changed

- `services/orion-vision-host/app/gpu.py`: adds `own_reusable_cache_mb(idx)` (`torch.cuda.memory_reserved - memory_allocated`). It returns 0 when torch is missing, CUDA is unavailable or not initialised, or the index is not visible to this process, and it never initialises CUDA. `pick_best_gpu` adds that figure to NVML free. It also stores a per-candidate breakdown in `last_pick_detail` and logs it at DEBUG.
- `services/orion-vision-host/app/scheduler.py`: counts refusals (total, streak, last reason, last refusal time, last admit time) and exposes them through `refusal_stats()`. It logs a WARNING on the first refusal of a streak, again every 100th refusal, and an INFO line when the streak ends. Readiness probes (`can_pick_gpu`) do not count as refusals.
- `services/orion-vision-host/app/main.py`: the refusal error text and `meta.refusal_reason` carry the breakdown. `/health` → `scheduler` includes `refusal_stats()`.
- `services/orion-vision-host/README.md`: documents the admission rule and the new `/health` fields.
- `services/orion-vision-host/tests/test_gpu_admission_own_cache.py`: new tests.

## Schema / bus / API changes

- Added: `/health` `scheduler.{gpu_hard_floor_refusals_total, gpu_hard_floor_refusal_streak, last_refusal_reason, last_refusal_at, last_admit_at}`; `VisionResult.meta.refusal_reason` on `gpu_hard_floor` results.
- Removed / Renamed: none.
- Behavior changed: GPU admission counts this process's reusable cache as free.
- Compatibility notes: additive only. `error_code` is unchanged, and the `error` string keeps its old prefix.

## Env/config changes

- Added / Removed / Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced: not needed.
- skipped keys requiring operator action: none.

## Live evidence (read-only, 2026-10-15 19:23 UTC)

```text
$ nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
0, Tesla P4, 7680 MiB, 5613 MiB, 1988 MiB
$ nvidia-smi --query-compute-apps=pid,used_memory --format=csv   (GPU 0 tenant)
3104887, 5610 MiB      <- orion-athena-vision-host, the only process on GPU 0
```

Gate arithmetic now: `1988 - 1200 = 788 < 800`. It refuses by **12 MiB**. Note that `used + free = 7601`, not 7680, because the driver keeps about 79 MiB. The configured budget leaves almost no margin on this card even when nothing else is running.

Container started 18:47:55 UTC (the PR #2691 deploy). Log timeline:

- 13 `[DETECT_TILES] stream=walkway tiles=2/2` lines from 18:48:38 to 18:53:53. That is about one every 20 to 25 s, and each run spends about 2.4 s on tiles plus about 1.0 s on the full frame.
- 105 successful `retina_fast` tasks, the last at 18:54:04.
- The first refusal came at 18:54:01. Every task after 18:54:04 was refused (707 refusal lines by 19:23:51).
- LIVENESS alert fired at 19:00:37.

**Did tiling cause the growth? UNVERIFIED.** The timing fits: refusals start about 6 min after the deploy that added tiling, after 13 tiled runs. But no log ever recorded `memory_reserved`/`memory_allocated` or per-PID NVML usage before 18:54. There is no per-process baseline from before the deploy either. So "tiling grew the cache" cannot be told apart from "normal cache growth on a card with a 12 MiB margin". The own-reusable figure for the live PID can't be read from outside the process, so how much the fix will free on this exact process is also UNVERIFIED. Once deployed, the new refusal reason prints it.

## Tests run

```text
docker run --rm -v <worktree>:/src:ro ... orion-vision-host-vision-host:latest \
  (pytest + pytest-asyncio installed to /tmp in the throwaway container)
python3 -m pytest -q tests
314 passed, 1 skipped   (after review fixes)
```

New tests (`tests/test_gpu_admission_own_cache.py`):

- The live incident numbers plus own reusable cache are admitted. The breakdown is recorded.
- The same numbers with no own cache are refused. This is the old behaviour, so the test fails if the fix is reverted.
- Another process eating memory is still refused, with a small own cache or none.
- If the cache probe raises, the check falls back to NVML free only.
- `own_reusable_cache_mb` returns reserved minus allocated. It returns 0 for: no CUDA, CUDA not initialised, index not visible, negative difference, torch not importable.
- The scheduler counts the refusal streak, the reason names the numbers, and an admit resets the streak.
- A readiness probe does not count as a refusal.
- `/health` exposes the stats.

## Evals run

No eval harness for scheduler admission exists in this service. The behaviour is deterministic, so the gate tests cover it. Follow-up: a live smoke after deploy (below).

## Docker/build/smoke checks

Tests ran inside the existing `orion-vision-host-vision-host:latest` image with a read-only source mount. No image rebuild and no deploy happened from this branch.

## Review findings fixed

The code review ran in a subagent and found no must-fix bugs.

- Finding (should): the own-cache probe passed the NVML index straight to `torch.cuda.memory_reserved`. Torch indices are relative to `CUDA_VISIBLE_DEVICES`. With `CUDA_VISIBLE_DEVICES=N` (N not 0), one card's cache could be added to another card's free figure.
  - Fix: `nvml_to_torch_index()` maps the NVML index through `CUDA_VISIBLE_DEVICES`. `CUDA_DEVICE_ORDER=PCI_BUS_ID` is already pinned, so this mapping is exact. An index outside the list, or UUID entries, gives 0 own cache (old behaviour).
  - Evidence: `test_nvml_to_torch_index` (9 cases), `test_own_cache_is_read_from_the_mapped_torch_device`.
- Finding (should): a test comment presented "~1000+ MiB idle cache" as a live number. It was never measured.
  - Fix: the comment now says it is a fixture value. The report states the live own-cache figure as UNVERIFIED and notes that the gate misses by only 12 MiB.
  - Evidence: `tests/test_gpu_admission_own_cache.py`, this report's Live evidence section.
- Not fixed, pre-existing, recorded here: the admission snapshot is taken before the per-GPU semaphore. The new term errs safe there, because a running task's activations count as allocated. When no CUDA devices are configured, the result is still labelled `gpu_hard_floor` with no reason. Nits not taken: discounting fragmented split blocks, and putting raw reserved/allocated in the breakdown.

## Restart required

From the clean primary `main` checkout synced with `origin/main`, after merge:

```bash
scripts/safe_docker_build.sh orion-vision-host up -d --build
```

Post-deploy check. This proves tasks succeed again and shows the refusal streak:

```bash
sleep 300; curl -fsS http://localhost:32797/health | python3 -m json.tool | grep -E 'refusal|admit'; docker logs --since 5m orion-athena-vision-host 2>&1 | grep -c '"ok": true'; docker logs --since 5m orion-athena-vision-host 2>&1 | grep -c 'gpu_hard_floor'
```

Pass means: `gpu_hard_floor_refusal_streak` is 0, `last_admit_at` is recent, and the `"ok": true` count is above 0. The restart alone frees the cache, so a pass in the first minutes proves little. The real proof is the same check about 30+ minutes later, once NVML free has dropped below 2000 MiB again (`nvidia-smi --query-gpu=memory.free --format=csv -i 0`) and tasks still succeed.

## Risks / concerns

- Severity: medium. Concern: reusable cache can be fragmented, so a large single allocation might still not fit, and over-admission could OOM. Mitigation: the 1200 MiB reserve still applies on top. An OOM raises inside the task and fails that one task, the same as before.
- Severity: medium. Concern: the 12 MiB margin shows the budget is close to unsatisfiable on the P4 even without the cache issue. Mitigation: out of scope here. Re-derive `VISION_VRAM_RESERVE_MB` for the P4 if refusals recur, using the new breakdown.
- Severity: low. Concern: the tiling-caused-growth claim is UNVERIFIED (see above).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2694

🤖 Generated with [Claude Code](https://claude.com/claude-code)
