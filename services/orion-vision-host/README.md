# Orion Vision Host

GPU-backed vision inference service (FastAPI + Redis bus). Tasks arrive on `CHANNEL_VISIONHOST_INTAKE` with payload schema **`VisionTaskRequestPayload`** (`orion/schemas/registry.py`, `orion/bus/channels.yaml`). Replies are published to `reply_to` (channel pattern `orion:vision:reply:*`) as envelope kind **`vision.task.result`** with payload **`VisionTaskResultPayload`**, including **`error_code`**, **`timings`**, and optional **`meta`** (e.g. `warnings`) on failures. Optional **`vision.artifact`** broadcast on `CHANNEL_VISIONHOST_PUB`.

## Probes

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | **Liveness:** process up; static scheduler limits and bus flag. |
| `GET /ready` | **Readiness:** HTTP **503** when not ready — profiles loaded, bus connected if enabled, at least one CUDA device passes VRAM hard floor, and **warm-on-start** profiles loaded without error. Body lists `degraded_reasons` and `warm_failed_profiles`. |
| `GET /profiles` | Loaded YAML version, enabled profiles, routing. |

## Operator checklist

1. **Caches:** Set `MODEL_CACHE_DIR`, `HF_HOME`, and `TRANSFORMERS_CACHE` to fast, persistent disk (see `.env_example` / deployment env).
2. **GPU visibility:** `CUDA_VISIBLE_DEVICES` and `VISION_DEVICES` must list indices the container can see. NVML (`nvidia-ml-py`) must work for VRAM-aware scheduling.
3. **Bus:** When `ORION_BUS_ENABLED=true`, Redis must be reachable before `/ready` goes green.
4. **VRAM floors:** Tune `VISION_VRAM_RESERVE_MB`, `VISION_VRAM_SOFT_FLOOR_MB`, `VISION_VRAM_HARD_FLOOR_MB` to match co-hosted workloads.
5. **Concurrency:** `VISION_MAX_INFLIGHT`, `VISION_MAX_INFLIGHT_PER_GPU`, `VISION_QUEUE_WHEN_BUSY`, `VISION_MAX_QUEUE` — queue full returns `error_code=queue_full` on the bus reply and in structured logs.
6. **Timeouts:** `VISION_TIMEOUT_S` wraps the threaded `VisionRunner.execute` path (wall-clock); logs include `scheduler_total_s`, estimated `queue_wait_est_s`, and `inference_s` when available.
7. **Models:** Override `VISION_VLM_MODEL_ID` for your VRAM budget (default is smaller than 8B-class; profile YAML `REPLACE_ME` falls back to runner defaults). Enable only profiles you need via `VISION_ENABLED_PROFILES`.

## Observability (logs-first)

Each finished task emits one structured line prefix `[VISION_TASK]` with JSON containing at least: `correlation_id`, `task_type`, `ok`, `device`, `error`, `error_code`, `queue_depth_at_submit`, `scheduler_total_s`, `inference_s`, `queue_wait_est_s`.

**Dashboard hints:** aggregate `error_code`, p95 `scheduler_total_s` / `inference_s` by `task_type`, and `queue_depth_at_submit` at submit time.

## Config drift: `adaptive_degrade`

The repository root `config/vision_profiles.yaml` may describe `runtime.adaptive_degrade`. **That block is not implemented in `orion-vision-host`** — scheduling uses env-backed floors and `VisionScheduler` only. Do not assume resolution drops or profile disables happen automatically.

## Smoke scripts

- `scripts/publish_test_task.py` — publish a task over the bus.
- `scripts/tap_artifacts.py` — subscribe to artifact channel.

## Tests

```bash
cd services/orion-vision-host
PYTHONPATH=. python3 -m pytest tests/ -q --tb=short
```

Scheduler tests do not require CUDA or model weights.
