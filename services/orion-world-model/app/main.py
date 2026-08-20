from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from loguru import logger

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly
from orion.schemas.world_model import (
    WorldModelPredictionPayload,
    WorldModelTaskRequestPayload,
    WorldModelTrajectoryStepV1,
)

from .gpu import GpuInspector
from .model import FEATURE_GROUP_NAMES, FeatureGroupDims, WorldModel
from .settings import Settings

settings = Settings()


def _build_dims(s: Settings) -> FeatureGroupDims:
    return FeatureGroupDims(
        biometrics=s.WM_DIM_BIOMETRICS,
        affect=s.WM_DIM_AFFECT,
        execution_context=s.WM_DIM_EXECUTION_CONTEXT,
        memory_pointers=s.WM_DIM_MEMORY_POINTERS,
        temporal=s.WM_DIM_TEMPORAL,
        vision_embedding=s.WM_DIM_VISION_EMBEDDING,
    )


def _select_device(gpu: GpuInspector, s: Settings) -> Optional[str]:
    """Returns e.g. 'cuda:0', or None if no candidate GPU clears the hard
    floor (or no GPU / pynvml at all -> caller falls back to CPU).

    `WM_DEVICE_STRATEGY`: `"fixed"` restricts the candidate pool to
    `WM_DEFAULT_DEVICE` only (still subject to the same reserve/hard-floor
    check via `pick_best_gpu` -- "fixed" means "don't scan other GPUs",
    not "skip the floor check"). Anything else (default `"best_free_vram"`)
    scans every `cuda:*` entry in `WM_DEVICES` and picks the best.
    """
    if s.WM_DEVICE_STRATEGY == "fixed":
        if not s.WM_DEFAULT_DEVICE.startswith("cuda:"):
            return None
        try:
            candidates: List[int] = [int(s.WM_DEFAULT_DEVICE.split(":", 1)[1])]
        except ValueError:
            return None
    else:
        candidates = []
        for d in s.devices:
            if d.startswith("cuda:"):
                try:
                    candidates.append(int(d.split(":", 1)[1]))
                except ValueError:
                    continue
        if not candidates:
            return None
    picked = gpu.pick_best_gpu(
        candidates,
        reserve_mb=s.WM_VRAM_RESERVE_MB,
        hard_floor_mb=s.WM_VRAM_HARD_FLOOR_MB,
        metric=s.WM_PICK_GPU_METRIC,
    )
    if picked is None:
        return None
    idx, _info = picked
    return f"cuda:{idx}"


_DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def resolve_dtype(wm_dtype: str) -> torch.dtype:
    """`"auto"` resolves to fp32 -- deliberately, not a device-conditional
    guess. This is an untrained scaffold with no accuracy target yet
    (README "Status"); silently dropping precision on `"auto"` would change
    numerics for a model nothing is measuring against. Set `WM_DTYPE`
    explicitly (`fp16`/`bf16`/`fp32`) to opt into lower precision."""
    key = (wm_dtype or "auto").strip().lower()
    if key == "auto":
        return torch.float32
    if key not in _DTYPE_MAP:
        raise ValueError(f"unknown WM_DTYPE={wm_dtype!r}, expected one of auto|fp16|bf16|fp32")
    return _DTYPE_MAP[key]


def trajectory_steps_to_tensors(
    steps: List[WorldModelTrajectoryStepV1],
    dims: FeatureGroupDims,
    device: str,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Validates each step's declared `dim` against the configured encoder
    input dims and builds `[1, window, group_dim]` tensors (batch=1 -- one
    request is one trajectory in this patch)."""
    out: Dict[str, List[List[float]]] = {name: [] for name in FEATURE_GROUP_NAMES}
    expected = dims.as_dict()
    for step in steps:
        for name in FEATURE_GROUP_NAMES:
            group = getattr(step, name)
            if group.dim != expected[name]:
                raise ValueError(
                    f"feature group '{name}' dim={group.dim} does not match configured "
                    f"WM_DIM_{name.upper()}={expected[name]}"
                )
            if len(group.vector) != group.dim:
                raise ValueError(
                    f"feature group '{name}' vector length {len(group.vector)} != declared dim {group.dim}"
                )
            out[name].append(group.vector)
    return {name: torch.tensor([out[name]], dtype=dtype, device=device) for name in FEATURE_GROUP_NAMES}


class WorldModelService:
    def __init__(self):
        self.bus: Optional[OrionBusAsync] = None
        self.gpu = GpuInspector()
        self.model: Optional[WorldModel] = None
        self.device: str = "cpu"
        self.dtype: torch.dtype = torch.float32
        self.dims: Optional[FeatureGroupDims] = None
        self._inflight_sem: Optional[asyncio.Semaphore] = None
        self._consumer_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        logger.remove()
        logger.add(lambda m: print(m, end=""), level=settings.LOG_LEVEL)

        self.dims = _build_dims(settings)
        picked_device = _select_device(self.gpu, settings)
        self.device = picked_device or "cpu"
        if picked_device is None:
            logger.warning(
                "[GPU] no CUDA device cleared hard floor (or pynvml unavailable) -- "
                "falling back to cpu. Untrained-scaffolding forward passes still work on CPU."
            )

        self.dtype = resolve_dtype(settings.WM_DTYPE)

        self.model = WorldModel(
            self.dims,
            fusion_dim=settings.WM_FUSION_DIM,
            d_model=settings.WM_DYNAMICS_D_MODEL,
            nhead=settings.WM_DYNAMICS_NHEAD,
            num_layers=settings.WM_DYNAMICS_NUM_LAYERS,
            dim_feedforward=settings.WM_DYNAMICS_DIM_FEEDFORWARD,
            max_window=settings.WM_DYNAMICS_MAX_WINDOW,
            state_dim=settings.state_dim,
            dropout=settings.WM_DYNAMICS_DROPOUT,
        ).to(device=self.device, dtype=self.dtype)
        self.model.eval()
        logger.info(
            f"[MODEL] loaded (untrained) device={self.device} dtype={self.dtype} "
            f"params={self.model.num_parameters()} ({self.model.num_parameters() / 1e6:.1f}M)"
        )

        self._inflight_sem = asyncio.Semaphore(settings.WM_MAX_INFLIGHT)

        if settings.ORION_BUS_ENABLED:
            self.bus = OrionBusAsync(url=settings.ORION_BUS_URL)
            await self.bus.connect()
            self._consumer_task = asyncio.create_task(self._consume_loop())
            logger.info(f"[READY] bus-first intake={settings.CHANNEL_WORLDMODEL_INTAKE}")
        else:
            logger.warning("[READY] bus disabled (HTTP only)")

    async def stop(self) -> None:
        self._shutdown_event.set()
        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
        if self.bus:
            await self.bus.close()
        self.gpu.shutdown()

    def can_pick_gpu(self) -> bool:
        return _select_device(self.gpu, settings) is not None

    def readiness_payload(self) -> Dict[str, Any]:
        model_ok = self.model is not None
        bus_ok = True
        if settings.ORION_BUS_ENABLED:
            bus_ok = bool(self.bus and self.bus.enabled)

        degraded: List[str] = []
        gpu_ok = True
        if settings.devices and any(d.startswith("cuda:") for d in settings.devices):
            gpu_ok = self.can_pick_gpu()
            if not gpu_ok:
                degraded.append("no_gpu_above_hard_floor")
        if not model_ok:
            degraded.append("model_not_loaded")
        if settings.ORION_BUS_ENABLED and not bus_ok:
            degraded.append("bus_not_connected")

        ready = model_ok and bus_ok and gpu_ok

        return {
            "ready": ready,
            "model_loaded": model_ok,
            "model_untrained": True,
            "device": self.device,
            "bus_required": bool(settings.ORION_BUS_ENABLED),
            "bus_connected": bus_ok,
            "gpu_schedulable": gpu_ok,
            "degraded_reasons": degraded,
        }

    def _forward_no_grad(self, tensors: Dict[str, torch.Tensor]):
        """Runs the model forward pass with grad disabled.

        Deliberately NOT `with torch.no_grad(): asyncio.to_thread(self.model,
        tensors)` in the caller -- PyTorch's grad-enabled flag is thread-local
        (verified live: entering `torch.no_grad()` on the calling coroutine's
        thread has no effect inside a worker thread spawned by
        `asyncio.to_thread`, so the forward pass would silently build an
        autograd graph anyway). Entering `torch.no_grad()` here, inside the
        function that actually runs *on* the worker thread, is required.
        """
        with torch.no_grad():
            return self.model(tensors)

    async def _run_forward(self, payload: WorldModelTaskRequestPayload):
        """Tensor build + forward pass, wrapped by `asyncio.wait_for(...,
        timeout=WM_TIMEOUT_S)` in the caller -- WM_TIMEOUT_S was previously
        declared in Settings/.env_example but never actually read anywhere
        (code review finding)."""
        tensors = await asyncio.to_thread(
            trajectory_steps_to_tensors, payload.trajectory, self.dims, self.device, self.dtype
        )
        return await asyncio.to_thread(self._forward_no_grad, tensors)

    async def run_prediction_task(self, payload: WorldModelTaskRequestPayload) -> WorldModelPredictionPayload:
        if self.model is None or self.dims is None:
            return WorldModelPredictionPayload(
                ok=False,
                task_type=payload.task_type,
                error="service not ready",
                error_code="service_not_ready",
            )
        t0 = time.monotonic()
        async with self._inflight_sem:  # type: ignore[union-attr]
            try:
                mean, log_var = await asyncio.wait_for(
                    self._run_forward(payload), timeout=float(settings.WM_TIMEOUT_S)
                )
            except asyncio.TimeoutError:
                return WorldModelPredictionPayload(
                    ok=False,
                    task_type=payload.task_type,
                    device=self.device,
                    error=f"forward pass timed out after {settings.WM_TIMEOUT_S}s",
                    error_code="timeout",
                )
            except ValueError as exc:
                return WorldModelPredictionPayload(
                    ok=False,
                    task_type=payload.task_type,
                    device=self.device,
                    error=str(exc),
                    error_code="bad_trajectory",
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("world_model_forward_failed")
                return WorldModelPredictionPayload(
                    ok=False,
                    task_type=payload.task_type,
                    device=self.device,
                    error=str(exc),
                    error_code="forward_failed",
                )

        elapsed = time.monotonic() - t0
        return WorldModelPredictionPayload(
            ok=True,
            task_type=payload.task_type,
            device=self.device,
            state_dim=mean.shape[-1],
            window_len=len(payload.trajectory),
            predicted_mean=mean.squeeze(0).tolist(),
            predicted_log_var=log_var.squeeze(0).tolist(),
            model_untrained=True,
            model_param_count=self.model.num_parameters(),
            timings={"inference_s": round(elapsed, 4)},
        )

    async def _consume_loop(self) -> None:
        if not self.bus:
            return
        async with self.bus.subscribe(settings.CHANNEL_WORLDMODEL_INTAKE) as pubsub:
            while not self._shutdown_event.is_set():
                try:
                    async for msg in self.bus.iter_messages(pubsub):
                        if self._shutdown_event.is_set():
                            break
                        data = msg.get("data")
                        if not data:
                            continue
                        decoded = self.bus.codec.decode(data)
                        if not decoded.ok or not decoded.envelope:
                            logger.error(f"[BUS] decode failed: {decoded.error}")
                            continue
                        asyncio.create_task(self._handle_envelope(decoded.envelope))
                except Exception as exc:
                    logger.error(f"[BUS] consumer error: {exc}")
                    await asyncio.sleep(1)

    async def _handle_envelope(self, envelope: BaseEnvelope) -> None:
        """Runs standalone via `asyncio.create_task` from `_consume_loop`
        (fire-and-forget, nothing awaits this task) -- an exception anywhere
        in this method, including inside `_publish_result`'s bus.publish
        calls (e.g. ORION_BUS_ENFORCE_CATALOG rejecting an unregistered
        `reply_to`, or a transient Redis error), previously propagated out of
        that unawaited task with nothing to catch it: the request would be
        silently dropped with no correlation-id-tagged log line anywhere
        (code review finding). The whole body is wrapped for that reason --
        this is a minimal fix (log with correlation id), not a full
        system.error bus publish like BaseChassis's handler wrapping."""
        try:
            if isinstance(envelope.payload, dict):
                payload = WorldModelTaskRequestPayload(**envelope.payload)
            elif isinstance(envelope.payload, WorldModelTaskRequestPayload):
                payload = envelope.payload
            else:
                logger.error(f"[BUS] unexpected payload type: {type(envelope.payload)}")
                return

            result = await self.run_prediction_task(payload)
            await self._publish_result(result, envelope)
        except Exception as exc:
            logger.exception(
                f"[BUS] handle_envelope failed correlation_id={envelope.correlation_id} error={exc}"
            )

    async def _publish_result(self, result: WorldModelPredictionPayload, source_envelope: BaseEnvelope) -> None:
        if not self.bus:
            return
        host_ref = ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION)

        reply_envelope = source_envelope.derive_child(
            kind="world_model.prediction",
            source=host_ref,
            payload=result,
        )
        reply_channel = (
            source_envelope.reply_to
            or f"{settings.CHANNEL_WORLDMODEL_REPLY_PREFIX}:{source_envelope.correlation_id}"
        )
        await self.bus.publish(reply_channel, reply_envelope)

        if result.ok:
            broadcast_envelope = source_envelope.derive_child(
                kind="world_model.prediction",
                source=host_ref,
                payload=result,
            )
            await self.bus.publish(settings.CHANNEL_WORLDMODEL_PUB, broadcast_envelope)


service = WorldModelService()
heartbeat_chassis: Optional[HeartbeatOnly] = None


def build_heartbeat_chassis() -> HeartbeatOnly:
    """Own, independent bus connection publishing SystemHealthV1 to
    orion:system:health every heartbeat_interval_sec (see
    services/orion-vision-host/app/main.py::build_heartbeat_chassis, same
    pattern, deliberately separate bus connection from `service.bus`)."""
    return HeartbeatOnly(
        ChassisConfig(
            service_name=settings.SERVICE_NAME,
            service_version=settings.SERVICE_VERSION,
            node_name=settings.NODE_NAME,
            bus_url=settings.ORION_BUS_URL,
            bus_enabled=settings.ORION_BUS_ENABLED,
            heartbeat_interval_sec=settings.HEARTBEAT_INTERVAL_SEC,
        )
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global heartbeat_chassis
    await service.start()
    try:
        heartbeat_chassis = build_heartbeat_chassis()
        await heartbeat_chassis.start_background()
        logger.info(
            f"[HOST] system_health_heartbeat_started service={settings.SERVICE_NAME} "
            f"interval_sec={settings.HEARTBEAT_INTERVAL_SEC}"
        )
    except Exception as exc:
        logger.warning(f"[HOST] system_health_heartbeat_start_failed error={exc}")
        heartbeat_chassis = None
    yield
    if heartbeat_chassis is not None:
        try:
            await heartbeat_chassis.stop()
        except Exception as exc:
            logger.warning(f"[HOST] system_health_heartbeat_stop_error error={exc}")
        heartbeat_chassis = None
    await service.stop()


app = FastAPI(title="Orion World Model", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "ok": True,
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "model_untrained": True,
        "device": service.device,
        "bus_enabled": bool(service.bus and service.bus.enabled),
        "intake": settings.CHANNEL_WORLDMODEL_INTAKE,
        "pub": settings.CHANNEL_WORLDMODEL_PUB,
    }


@app.get("/ready")
async def ready():
    body = service.readiness_payload()
    code = 200 if body.get("ready") else 503
    return JSONResponse(body, status_code=code)


@app.post("/v1/world-model/predict")
async def http_predict(payload: Dict[str, Any]):
    """Optional HTTP entrypoint mirroring vision-host's `/v1/vision/task` --
    mainly useful for local smoke without a bus. Real integration is expected
    to go over the bus intake channel."""
    if service.model is None:
        return JSONResponse({"ok": False, "error": "service not ready"}, status_code=503)
    try:
        task_payload = WorldModelTaskRequestPayload(**payload)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": f"invalid payload: {exc}"}, status_code=400)

    result = await service.run_prediction_task(task_payload)
    return result.model_dump()
