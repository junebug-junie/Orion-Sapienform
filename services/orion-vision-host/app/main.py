import asyncio
import json
import os
import uuid
import time
from typing import Any, Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from loguru import logger

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly
from orion.schemas.vision import (
    VisionTaskRequestPayload,
    VisionTaskResultPayload,
    VisionArtifactPayload,
)

from .artifacts import build_artifact_payload
from .liveness import (
    build_attention_request,
    build_watcher_or_default,
    post_attention_request,
)
from .models import VisionResult, VisionTask
from .profiles import VisionProfiles
from .runner import VisionRunner
from .scheduler import ScheduledPick, VisionQueueFullError, VisionScheduler
from .settings import Settings

settings = Settings()

# Task types excluded from the general CHANNEL_VISIONHOST_PUB broadcast
# (review finding, 2026-08-25/26). That channel has more real consumers
# than any one file enumerates -- this suppression works by never
# publishing in the first place, so it protects every one of them, known
# or not, rather than requiring an accurate consumer inventory. See the
# design doc's own section 6.5: identity-bearing data needs a retention
# policy that ships WITH this feature, not after.
_TASK_TYPES_EXCLUDED_FROM_BROADCAST = frozenset({"identity_face"})


def should_broadcast_artifact(task_type: str, payload: Optional[VisionArtifactPayload] = None) -> bool:
    """False suppresses the broadcast on EITHER of two independent checks:

    1. ``task_type`` is directly excluded (the identity_face RPC/HTTP call
       itself).
    2. The artifact's own ``outputs`` carries an ``identities`` field
       (review finding, 2026-08-26, second pass: a task_type-only check is
       bypassable by a config-only change -- adding `- use: identity_face`
       as a step in ANY pipeline in config/vision_profiles.yaml. runner.py's
       `_run_pipeline` merges every step's dict output with zero content
       filtering, and `artifacts.py`'s generic passthrough
       (`setattr(outputs, k, v)` for any unreserved key) attaches
       `identities` onto the merged artifact regardless of the outer
       pipeline's own task_type name. Checking the artifact's real content,
       not just the caller-supplied task_type string, closes that gap
       structurally instead of hoping nobody ever adds that pipeline step.
    """
    if task_type in _TASK_TYPES_EXCLUDED_FROM_BROADCAST:
        return False
    if payload is not None and getattr(payload.outputs, "identities", None) is not None:
        return False
    return True


class VisionHostService:
    def __init__(self):
        self.bus: Optional[OrionBusAsync] = None
        self.profiles: Optional[VisionProfiles] = None
        self.runner: Optional[VisionRunner] = None
        self.sched: Optional[VisionScheduler] = None
        self._consumer_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        # Review finding (MEDIUM): the event loop keeps only a WEAK reference to
        # tasks, so a bare create_task() for the alert POST could be collected
        # mid-flight -- silently dropping the one notification this whole
        # feature exists to deliver. Held here the same way _consumer_task is.
        self._alert_tasks: set = set()
        # Liveness watcher: notices a healthy-but-serving-nothing host. See
        # app/liveness.py. Constructed unconditionally so /profiles can always
        # report the current failure rate even when alerting is disabled.
        #
        # Review finding (HIGH): this construction runs at module import
        # (`service = VisionHostService()` at the bottom of this file), and the
        # watcher raises ValueError on non-hysteretic config. A typo'd
        # VISION_LIVENESS_* value in a .env therefore crashlooped the entire
        # vision service -- the alerting subsystem becoming a brand-new way to
        # cause the exact 21-hour blackout it was written to detect, and
        # VISION_LIVENESS_ALERT_ENABLED=false did not bypass it.
        #
        # The ValueError stays in the watcher: refusing a flapping config is
        # correct for a library, and the tests still assert it. But nothing in
        # the alerting path is allowed to stop this service from seeing, so a
        # bad value here degrades loudly to the known-good defaults instead.
        self.liveness = build_watcher_or_default(settings)

    async def start(self):
        logger.remove()
        logger.add(lambda m: print(m, end=""), level=settings.LOG_LEVEL)

        self._apply_env_runtime()

        # Profiles + runner
        self.profiles = VisionProfiles(settings.VISION_PROFILES_PATH)
        self.profiles.load()

        self.runner = VisionRunner(
            profiles=self.profiles,
            enabled_names=settings.enabled_profiles,
            cache_dir=settings.MODEL_CACHE_DIR,
        )

        # Scheduler
        self.sched = VisionScheduler(
            devices=settings.devices,
            pick_metric=settings.VISION_PICK_GPU_METRIC,
            max_inflight=settings.VISION_MAX_INFLIGHT,
            max_inflight_per_gpu=settings.VISION_MAX_INFLIGHT_PER_GPU,
            queue_when_busy=settings.VISION_QUEUE_WHEN_BUSY,
            max_queue=settings.VISION_MAX_QUEUE,
            reserve_mb=settings.VISION_VRAM_RESERVE_MB,
            soft_floor_mb=settings.VISION_VRAM_SOFT_FLOOR_MB,
            hard_floor_mb=settings.VISION_VRAM_HARD_FLOOR_MB,
        )
        await self.sched.start()

        warmed = self.runner.warm_profiles()
        logger.info(f"[WARM] warmed={warmed}")

        # Bus
        if settings.ORION_BUS_ENABLED:
            self.bus = OrionBusAsync(url=settings.ORION_BUS_URL)
            await self.bus.connect()
            self._consumer_task = asyncio.create_task(self._consume_loop())
            logger.info(f"[READY] bus-first intake={settings.CHANNEL_VISIONHOST_INTAKE}")
        else:
            logger.warning("[READY] bus disabled (HTTP only)")

    async def stop(self):
        self._shutdown_event.set()
        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass

        if self.sched:
            await self.sched.stop()

        if self.bus:
            await self.bus.close()

    def _apply_env_runtime(self) -> None:
        os.environ.setdefault("HF_HOME", settings.HF_HOME)
        os.environ.setdefault("TRANSFORMERS_CACHE", settings.TRANSFORMERS_CACHE)

    def _make_execute_handler(self, task: VisionTask):
        async def handler(pick: ScheduledPick):
            if pick.device == "cpu":
                return VisionResult(
                    corr_id=task.corr_id,
                    ok=False,
                    task_type=task.task_type,
                    device=None,
                    error="No GPU available above hard floor (VRAM pressure).",
                    meta={"error_code": "gpu_hard_floor"},
                )
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(self.runner.execute, task, pick.device),
                    timeout=float(settings.VISION_TIMEOUT_S),
                )
            except asyncio.TimeoutError:
                return VisionResult(
                    corr_id=task.corr_id,
                    ok=False,
                    task_type=task.task_type,
                    device=pick.device,
                    error=f"Vision inference timed out after {settings.VISION_TIMEOUT_S}s",
                    meta={
                        "error_code": "timeout",
                        "timings": {"timeout_s": float(settings.VISION_TIMEOUT_S)},
                    },
                )

        return handler

    def _attach_scheduler_timings(self, res: VisionResult, t0: float) -> VisionResult:
        sched_total = time.monotonic() - t0
        meta = dict(res.meta or {})
        timings = dict(meta.get("timings") or {})
        timings["scheduler_total_s"] = round(sched_total, 4)
        inf = meta.get("latency_s")
        if inf is not None:
            timings["inference_s"] = inf
            timings["queue_wait_est_s"] = round(max(0.0, sched_total - float(inf)), 4)
        meta["timings"] = timings
        return res.model_copy(update={"meta": meta})

    def _log_task_completion(
        self,
        *,
        correlation_id: str,
        task_type: str,
        res: VisionResult,
        queue_depth_at_submit: int,
    ) -> None:
        meta = res.meta or {}
        line = {
            "event": "vision_task_complete",
            "correlation_id": correlation_id,
            "task_type": task_type,
            "ok": res.ok,
            "device": res.device,
            "error": res.error,
            "error_code": meta.get("error_code"),
            "queue_depth_at_submit": queue_depth_at_submit,
            "scheduler_total_s": (meta.get("timings") or {}).get("scheduler_total_s"),
            "inference_s": meta.get("latency_s"),
            "queue_wait_est_s": (meta.get("timings") or {}).get("queue_wait_est_s"),
        }
        logger.info("[VISION_TASK] {}", json.dumps(line, default=str))
        self._note_liveness(ok=bool(res.ok), error_code=meta.get("error_code"))

    def _note_liveness(self, *, ok: bool, error_code: Optional[str]) -> None:
        """Feed one task outcome to the watcher; fire an alert if it asks.

        Wrapped whole: a liveness bug must never break task completion, which
        is the path that logs and replies to the caller.
        """
        try:
            decision = self.liveness.record(ok=ok, error_code=error_code)
            if not (decision.alert or decision.recovered):
                return
            if not settings.VISION_LIVENESS_ALERT_ENABLED:
                # Not delivered -- roll the provisional state back, or the
                # watcher believes it alerted and suppresses forever.
                self.liveness.note_alert_delivered(False)
                logger.warning(
                    "[LIVENESS] {} (alerting disabled) fail_rate={:.2f} n={}",
                    decision.reason, decision.fail_rate, decision.sample_count,
                )
                return
            if not settings.NOTIFY_BASE_URL:
                self.liveness.note_alert_delivered(False)
                logger.warning(
                    "[LIVENESS] {} but NOTIFY_BASE_URL is unset -- no alert sent",
                    decision.reason,
                )
                return
            body = build_attention_request(decision, node_name=settings.NODE_NAME)
            logger.warning("[LIVENESS] {} -> attention request", decision.reason)
            # to_thread: urllib is blocking and this runs on the task-completion
            # path. A hung notify service must not stall the vision event loop.
            task = asyncio.create_task(
                asyncio.to_thread(
                    post_attention_request,
                    body,
                    base_url=settings.NOTIFY_BASE_URL,
                    token=settings.NOTIFY_API_TOKEN or None,
                )
            )
            self._alert_tasks.add(task)
            task.add_done_callback(self._alert_tasks.discard)
            task.add_done_callback(self._on_alert_sent)
        except Exception as exc:
            logger.warning("[LIVENESS] watcher error (ignored): {}", exc)

    def _on_alert_sent(self, task: "asyncio.Task") -> None:
        """Confirm or roll back the watcher's provisional alert state.

        Without this a failed POST would still leave the watcher believing it
        had alerted, suppressing every retry for the rest of the incident.
        """
        try:
            delivered = bool(task.result())
        except Exception as exc:
            logger.warning("[LIVENESS] alert send raised: {}", exc)
            delivered = False
        try:
            self.liveness.note_alert_delivered(delivered)
            if not delivered:
                logger.warning("[LIVENESS] alert NOT delivered -- will retry")
        except Exception as exc:
            logger.warning("[LIVENESS] confirm error (ignored): {}", exc)

    async def run_vision_task(self, task: VisionTask) -> VisionResult:
        if not self.runner or not self.sched:
            return VisionResult(
                corr_id=task.corr_id,
                ok=False,
                task_type=task.task_type,
                device=None,
                error="service not ready",
                meta={"error_code": "service_not_ready"},
            )
        t0 = time.monotonic()
        q_depth = self.sched.queue_depth()
        handler = self._make_execute_handler(task)
        try:
            res: VisionResult = await self.sched.submit(handler)
        except VisionQueueFullError:
            res = VisionResult(
                corr_id=task.corr_id,
                ok=False,
                task_type=task.task_type,
                device=None,
                error="Scheduler queue full (VISION_MAX_QUEUE)",
                meta={"error_code": "queue_full"},
            )
        except Exception as e:
            res = VisionResult(
                corr_id=task.corr_id,
                ok=False,
                task_type=task.task_type,
                device=None,
                error=str(e),
                meta={"error_code": "scheduler_error"},
            )
        res = self._attach_scheduler_timings(res, t0)
        self._log_task_completion(
            correlation_id=task.corr_id,
            task_type=task.task_type,
            res=res,
            queue_depth_at_submit=q_depth,
        )
        return res

    def readiness_payload(self) -> Dict[str, Any]:
        profiles_ok = self.profiles is not None
        bus_ok = True
        if settings.ORION_BUS_ENABLED:
            bus_ok = bool(self.bus and self.bus.enabled)

        degraded: list[str] = []
        gpu_schedulable = False
        if self.sched:
            gpu_schedulable = self.sched.can_pick_gpu()
            if not gpu_schedulable:
                degraded.append("no_gpu_above_hard_floor")
        else:
            degraded.append("scheduler_not_started")

        warm_failed = list(self.runner.warm_errors.keys()) if self.runner else []
        if warm_failed:
            degraded.extend([f"warm_failed:{n}" for n in warm_failed])

        ready = profiles_ok and bus_ok and gpu_schedulable and not warm_failed

        return {
            "ready": ready,
            "profiles_loaded": profiles_ok,
            "bus_required": bool(settings.ORION_BUS_ENABLED),
            "bus_connected": bus_ok,
            "gpu_schedulable": gpu_schedulable,
            "warm_failed_profiles": warm_failed,
            "degraded_reasons": degraded,
            "queue_depth": self.sched.queue_depth() if self.sched else 0,
        }

    async def _consume_loop(self):
        if not self.bus:
            return

        async with self.bus.subscribe(settings.CHANNEL_VISIONHOST_INTAKE) as pubsub:
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

                        # Process in background to not block consumer
                        asyncio.create_task(self._handle_envelope(decoded.envelope))

                except Exception as e:
                    logger.error(f"[BUS] consumer error: {e}")
                    await asyncio.sleep(1)

    async def _handle_envelope(self, envelope: BaseEnvelope):
        # Validate payload
        try:
            if isinstance(envelope.payload, dict):
                 # Try to convert dict to model if codec returned dict
                payload = VisionTaskRequestPayload(**envelope.payload)
            elif isinstance(envelope.payload, VisionTaskRequestPayload):
                payload = envelope.payload
            else:
                logger.error(f"[BUS] unexpected payload type: {type(envelope.payload)}")
                return
        except Exception as e:
            logger.error(f"[BUS] payload validation failed: {e}")
            return

        corr_id = str(envelope.correlation_id) if envelope.correlation_id else str(uuid.uuid4())
        reply_to = envelope.reply_to or f"{settings.CHANNEL_VISIONHOST_REPLY_PREFIX}:{corr_id}"

        task = VisionTask(
            corr_id=corr_id,
            reply_channel=reply_to,
            task_type=payload.task_type,
            request=payload.request,
            meta=payload.meta or {},
        )

        res = await self.run_vision_task(task)
        await self._publish_result(res, envelope)

    async def _publish_result(self, res: VisionResult, source_envelope: BaseEnvelope):
        if not self.bus:
            return

        # Prepare artifacts (consolidated)
        artifact_payload = self._create_artifact_payload(res, source_envelope)

        # Prepare result payload (bus contract: error_code + meta for failures)
        meta_src = dict(res.meta or {})
        error_code = meta_src.pop("error_code", None)
        timings = meta_src.pop("timings", None)
        meta_src.pop("latency_s", None)
        if res.warnings:
            meta_src["warnings"] = list(res.warnings)
        result_meta = meta_src if meta_src else None

        result_payload = VisionTaskResultPayload(
            ok=res.ok,
            task_type=res.task_type,
            device=res.device,
            error=res.error,
            error_code=error_code,
            artifact=artifact_payload if res.ok else None,
            timings=timings,
            meta=result_meta,
        )

        # Publish reply (orion.envelope wrapper; kind carries vision.task.result)
        host_ref = ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION)
        reply_envelope = source_envelope.derive_child(
            kind="vision.task.result",
            source=host_ref,
            payload=result_payload,
        )
        reply_channel = source_envelope.reply_to or f"{settings.CHANNEL_VISIONHOST_REPLY_PREFIX}:{res.corr_id}"
        await self.bus.publish(reply_channel, reply_envelope)

        if (
            res.ok
            and artifact_payload
            and settings.VISION_ARTIFACT_BROADCAST_ENABLED
            and should_broadcast_artifact(res.task_type, artifact_payload)
        ):
             await self._publish_artifact_broadcast(artifact_payload, source_envelope)

    def _create_artifact_payload(self, res: VisionResult, source_envelope: BaseEnvelope) -> Optional[VisionArtifactPayload]:
        return build_artifact_payload(res)

    async def _publish_artifact_broadcast(self, payload: VisionArtifactPayload, source_envelope: BaseEnvelope):
        host_ref = ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION)
        envelope = source_envelope.derive_child(
            kind="vision.artifact",
            source=host_ref,
            payload=payload,
        )

        await self.bus.publish(settings.CHANNEL_VISIONHOST_PUB, envelope)

service = VisionHostService()
heartbeat_chassis: HeartbeatOnly | None = None


def build_heartbeat_chassis() -> HeartbeatOnly:
    """Own, independent bus connection publishing SystemHealthV1 to orion:system:health
    every heartbeat_interval_sec. Deliberately separate from `service.bus` above (see
    docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md)."""
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

app = FastAPI(title="Orion Vision Host", version="0.1.0", lifespan=lifespan)

@app.get("/health")
async def health():
    return {
        "ok": True,
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "bus_enabled": bool(service.bus and service.bus.enabled),
        "intake": settings.CHANNEL_VISIONHOST_INTAKE,
        "pub": settings.CHANNEL_VISIONHOST_PUB,
        "scheduler": {
            "max_inflight": settings.VISION_MAX_INFLIGHT,
            "max_inflight_per_gpu": settings.VISION_MAX_INFLIGHT_PER_GPU,
            "queue_when_busy": settings.VISION_QUEUE_WHEN_BUSY,
            "max_queue": settings.VISION_MAX_QUEUE,
        },
    }


@app.get("/ready")
async def ready():
    body = service.readiness_payload()
    code = 200 if body.get("ready") else 503
    return JSONResponse(body, status_code=code)


@app.get("/profiles")
async def profiles_summary():
    if not service.profiles:
        return JSONResponse({"ok": False, "error": "profiles not loaded"}, status_code=503)
    return {
        "ok": True,
        "version": service.profiles.version,
        "liveness": service.liveness.snapshot(),
        "enabled": settings.enabled_profiles,
        "pipelines": list(service.profiles.pipelines.keys()),
        "profiles": list(service.profiles.profiles.keys()),
        "task_routing": service.profiles.task_routing,
    }

@app.post("/v1/vision/task")
async def http_task(payload: Dict[str, Any]):
    """
    Optional HTTP entrypoint.
    Minimal request:
      { "task_type": "...", "request": {...} }
    """
    if not service.runner or not service.sched:
        return JSONResponse({"ok": False, "error": "service not ready"}, status_code=503)

    corr_id = payload.get("corr_id") or str(uuid.uuid4())
    task_type = payload.get("task_type") or "retina_fast"
    request = payload.get("request") or {}

    # We fake a VisionTaskRequestPayload
    task_payload = VisionTaskRequestPayload(
        task_type=task_type,
        request=request,
        meta=payload.get("meta")
    )

    # We mimic the bus handling logic but return result directly
    task = VisionTask(
        corr_id=corr_id,
        reply_channel="http-direct",
        task_type=task_payload.task_type,
        request=task_payload.request,
        meta=task_payload.meta or {},
    )

    try:
        res: VisionResult = await service.run_vision_task(task)

        # Also broadcast artifact if success -- same should_broadcast_artifact
        # guard as the bus-first path in _publish_result (found live while
        # verifying that fix, 2026-08-26: this HTTP entrypoint has its own,
        # separate _publish_artifact_broadcast call that bypassed it entirely).
        if res.ok and res.artifacts and service.bus:
             # Create dummy source envelope
             dummy_env = BaseEnvelope(
                 kind="http.direct",
                 source=ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION),
                 correlation_id=uuid.UUID(corr_id),
                 payload={},
             )
             # Reuse creation logic
             art_payload = service._create_artifact_payload(res, dummy_env)
             # Built BEFORE the broadcast decision (not after, as this used
             # to be structured) so should_broadcast_artifact can inspect
             # its real content, not just the caller-supplied task_type.
             if art_payload and should_broadcast_artifact(res.task_type, art_payload):
                 await service._publish_artifact_broadcast(art_payload, dummy_env)

        return res.model_dump()
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e), "corr_id": corr_id}, status_code=500)
