# Orion Vision Host — production readiness (GPU inference) — design

**Date:** 2026-05-02  
**Status:** Draft for operator review (scope: **orion-vision-host** only; not window/scribe)

---

## 1. Purpose

Make **`services/orion-vision-host`** trustworthy to run in production: predictable behavior under load, visible queue and GPU health, clear failure semantics on the bus, and operator defaults that match real hardware—without redesigning the vision pipeline or replacing Window/Scribe in this iteration.

---

## 2. Current state (baseline)

- **Intake:** Redis bus (`CHANNEL_VISIONHOST_INTAKE`) and optional HTTP `POST /v1/vision/task`. Results go to `reply_to` and successful runs may broadcast `vision.artifact` on `CHANNEL_VISIONHOST_PUB`.
- **Inference:** `VisionRunner` executes profile-defined pipelines (embedding, open-vocab detection, captioning, etc.) with lazy-loaded models via `ModelManager` (including **SigLIP2 → SigLIP** load fallback for embedders).
- **Scheduling:** `VisionScheduler` enforces global and per-GPU inflight limits, optional queuing when busy, and VRAM-aware GPU selection (`GpuInspector` / NVML).
- **Health:** `GET /health` returns static config snapshot and bus enabled flag; it does **not** yet distinguish “process up” from “models warm / GPU usable / queue healthy.”
- **Profile YAML:** `config/vision_profiles.yaml` documents **`adaptive_degrade`** and VRAM budgets; **application code under `services/orion-vision-host/` does not reference `adaptive_degrade` by name today.** Closing or explicitly deferring that gap is part of production clarity (either implement consumption of those steps or mark them documentation-only until implemented).

---

## 3. Goals

1. **Observability:** Operators can answer: backlog depth, accept vs refuse rate, per-`task_type` latency and failure reasons, GPU free VRAM (or refusal cause), bus connectivity—without attaching a debugger.
2. **Readiness:** A probe (or extended health contract) distinguishes **ready** (GPU policy allows work, profiles loaded, bus connected if enabled) from **degraded** (e.g. all GPUs below hard floor; warm failed for critical profiles).
3. **Failure semantics:** Errors map consistently to `VisionTaskResultPayload` (`ok`, `error`, optional timings): timeouts, missing `image_path`, OOM, scheduler refusal (“no GPU above hard floor”), and model load failures—each identifiable in logs and in the reply envelope.
4. **Operator defaults:** Document and align env defaults with deployment reality (e.g. **VLM model size** via `VISION_VLM_MODEL_ID` and enabled profiles); avoid implying an 8B-class default where nodes cannot fit it.

---

## 4. Non-goals (this spec)

- **orion-vision-window** and **orion-vision-scribe** behavior (persistence, multi-instance windowing).
- **Micro-batching** for embed/detect in v1 (optional follow-up if metrics show GPU idle under burst).
- **Splitting** into separate “fast” vs “slow” deployments (optional scale-out pattern after single-host SLOs are visible).

---

## 5. Design

### 5.1 Observability

- **Structured logs** on each task completion (success or failure), including at minimum: `correlation_id`, `task_type`, `ok`, `device` (or refusal reason), `latency_s` or scheduler wait + execute breakdown where available, `error` class or short code (not multi-line stacks in the primary line), `queue_depth` or inflight snapshot at submit time if cheap.
- **Metrics:** Prefer **alignment with existing Orion services** (e.g. simple counters/histograms if Prometheus is already used nearby). **Do not** introduce a new metrics stack solely for this service; if the repo pattern is logs-first, ship structured logs first and add `/metrics` only where consistent with sibling services.
- **Optional dashboard hints:** Document which fields to graph (p95 latency by `task_type`, refusal rate, queue depth).

### 5.2 Health and readiness

- Extend **`/health`** or add **`/ready`** (choose one clear contract and document it in `services/orion-vision-host/README.md`):
  - **Liveness:** process serving HTTP.
  - **Readiness:** bus connected when `ORION_BUS_ENABLED`; profiles loaded; at least one device policy allows scheduling (or explicitly report **degraded** with reason when every GPU is below hard floor).
- Warm-up: retain existing `warm_profiles()` behavior; if warm fails for an enabled profile, **readiness** should not claim full “ready” unless policy says optional profiles may fail (make this explicit in the spec implementation).

### 5.3 Timeouts and backpressure

- **`VISION_TIMEOUT_S`:** Apply consistently to the **user-visible** execution path (runner execute + any blocking load on critical path where feasible). Document interaction with `asyncio.to_thread` and scheduler queue wait: distinguish **queue wait** vs **inference** time in logs when practical.
- **Queue:** When `VISION_MAX_QUEUE` is hit, behavior must be **explicit** (reject with clear error in result payload and structured log), not silent stall.

### 5.4 Degradation and profile YAML

- **Short term:** Ensure runtime behavior matches **documented** operator levers: VRAM floors, `VISION_ENABLED_PROFILES`, device list, dtype.
- **Profile file:** Either **(a)** implement reading **`adaptive_degrade`** (and related steps) in the runner/scheduler path, or **(b)** add a prominent comment in `vision_profiles.yaml` and README that those sections are **not yet enforced by code**—pick one in implementation to avoid silent drift.

### 5.5 Testing and verification

- **Unit-level:** Scheduler refusal paths, mapping of exceptions to `VisionResult` / payload fields where logic is isolated.
- **Integration:** Existing scripts under `services/orion-vision-host/scripts/` (`publish_test_task.py`, `tap_artifacts.py`) remain the smoke path; extend or add a minimal test that uses a **tiny local image** and asserts a successful **`vision.task.result`** on the reply channel when GPU is available (skip or mark xfail in CI without GPU as appropriate).
- **Documentation:** `README.md` — operator checklist: cache dirs (`MODEL_CACHE_DIR`, `HF_HOME`), GPU visibility, env tuning table for inflight/queue/VRAM.

---

## 6. Success criteria

- Under synthetic load, operators can see **queue depth and failure reasons** without reading Python tracebacks in production logs for the common cases.
- **`/health` or `/ready`** reflects **GPU/scheduling reality**, not only “FastAPI started.”
- **Adaptive degrade** is either **implemented** or **explicitly labeled non-functional** in config/docs—no ambiguous middle state.

---

## 7. Follow-ups (after this spec)

- **Optional batching** for embed (and possibly detect) if profiling shows low utilization.
- **Replica / tier split** if a single host cannot meet latency SLOs after observability is in place.

---

## 8. Implementation handoff

After this document is approved, create an implementation plan (writing-plans flow) scoped to `services/orion-vision-host/` and shared `config/vision_profiles.yaml` / `orion/schemas/vision.py` only if envelope fields require clarification—avoid scope creep into Window/Scribe.
