import asyncio
import logging
import cv2
import time
import os
import uuid
import numpy as np
from datetime import datetime, timezone

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.vision import (
    VisionEdgeArtifact, VisionObject, VisionEdgeHealth,
    VisionFramePointerPayload, VisionEdgeError
)

from .context import settings, bus, detectors
from .utils import draw_boxes

logger = logging.getLogger("orion-vision-edge.detector")

def _load_image(path: str) -> np.ndarray:
    if not os.path.exists(path):
        return None
    return cv2.imread(path)

def _save_debug_frame(frame, detections, name_suffix=""):
    try:
        if not settings.EDGE_DEBUG_SAVE_FRAMES:
            return

        # Simple sampling
        if int(time.time() * 100) % settings.EDGE_DEBUG_SAMPLE_RATE != 0:
            return

        annotated = frame.copy()
        # Draw boxes
        boxes_xywh = []
        for d in detections:
            x1, y1, x2, y2 = d.box_xyxy
            boxes_xywh.append((x1, y1, x2-x1, y2-y1))

        draw_boxes(annotated, boxes_xywh, name_suffix)

        os.makedirs(settings.EDGE_DEBUG_DIR, exist_ok=True)
        ts = int(time.time() * 1000)
        path = os.path.join(settings.EDGE_DEBUG_DIR, f"debug_{ts}_{name_suffix}.jpg")
        cv2.imwrite(path, annotated)
    except Exception as e:
        logger.warning(f"Debug save failed: {e}")

async def run_detector_loop():
    logger.info(f"[DETECTOR] Starting detector loop. Subscribing to {settings.CHANNEL_VISION_FRAMES}")

    if not bus.enabled:
        logger.warning("Bus disabled, detector loop aborting.")
        return

    # Subscribe
    async with bus.subscribe(settings.CHANNEL_VISION_FRAMES) as pubsub:
        async for msg in bus.iter_messages(pubsub):
            # Decode
            decoded = bus.codec.decode(msg.get("data"))
            if not decoded.ok or not decoded.envelope:
                continue

            env = decoded.envelope

            # Validate payload
            try:
                pointer = VisionFramePointerPayload.model_validate(env.payload)
            except Exception:
                logger.debug("Ignored non-pointer message")
                continue

            # Load Frame
            if not pointer.image_path:
                continue

            frame = await asyncio.get_event_loop().run_in_executor(
                None, _load_image, pointer.image_path
            )

            if frame is None:
                logger.warning(f"Could not load image: {pointer.image_path}")
                continue

            # Health Check
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mean_brightness = np.mean(gray)

                health_payload = VisionEdgeHealth(
                    camera_id=settings.SOURCE,
                    ts=time.time(),
                    ok=True,
                    fps=settings.FPS,
                    mean_brightness=float(mean_brightness),
                    resolution=f"{frame.shape[1]}x{frame.shape[0]}"
                )

                # Publish Health
                # TODO: Throttle this if needed. For now, emit every frame as per implicit prompt requirement.
                health_env = env.derive_child(
                    kind="vision.edge.health",
                    source=ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION),
                    payload=health_payload
                )
                await bus.publish("vision.edge.health", health_env)
            except Exception as e:
                logger.warning(f"Health check failed: {e}")

            # Run Detectors
            vision_objects = []

            try:
                # 1. Frame Detectors
                for name, mode, det in detectors:
                    if mode != "frame":
                        continue

                    try:
                        raw_results = det.detect(frame)
                    except Exception as det_err:
                        logger.error(f"Detector {name} failed: {det_err}")
                        # Emit Error
                        err_payload = VisionEdgeError(
                            camera_id=settings.SOURCE,
                            ts=time.time(),
                            error_type="detector_failure",
                            message=str(det_err),
                            meta={"detector": name}
                        )
                        err_env = env.derive_child(
                            kind="vision.edge.error",
                            source=ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION),
                            payload=err_payload
                        )
                        await bus.publish("vision.edge.error", err_env)
                        continue

                    for res in raw_results:
                        if len(res) == 6: # YOLO
                            x, y, w, h, label, score = res
                            vision_objects.append(VisionObject(
                                label=label or name,
                                score=float(score),
                                box_xyxy=[float(x), float(y), float(x+w), float(y+h)],
                                class_id=None
                            ))
                        elif len(res) == 4: # Face/Motion
                            x, y, w, h = res
                            vision_objects.append(VisionObject(
                                label=name,
                                score=1.0,
                                box_xyxy=[float(x), float(y), float(x+w), float(y+h)]
                            ))

                # 2. Event Detectors
                det_dicts = []
                for vo in vision_objects:
                    w = vo.box_xyxy[2] - vo.box_xyxy[0]
                    h = vo.box_xyxy[3] - vo.box_xyxy[1]
                    det_dicts.append({
                        "kind": "yolo" if vo.label in ["person", "cat", "dog"] else vo.label,
                        "label": vo.label,
                        "score": vo.score,
                        "bbox": (vo.box_xyxy[0], vo.box_xyxy[1], w, h)
                    })

                for name, mode, det in detectors:
                    if mode != "event":
                        continue
                    try:
                        events = det.update(det_dicts)
                        for ev in events:
                            vision_objects.append(VisionObject(
                                label=ev.get("label", name),
                                score=ev.get("score", 1.0),
                                box_xyxy=[0.0, 0.0, 0.0, 0.0]
                            ))
                    except Exception as e:
                        logger.error(f"Event detector {name} failed: {e}")

                # Debug Save
                if settings.EDGE_DEBUG_SAVE_FRAMES and vision_objects:
                    asyncio.get_event_loop().run_in_executor(
                        None, _save_debug_frame, frame, vision_objects, "all"
                    )

                # Publish Artifact
                artifact = VisionEdgeArtifact(
                    artifact_id=str(uuid.uuid4()),
                    correlation_id=str(env.correlation_id),
                    task_type="edge_detection",
                    device=settings.YOLO_DEVICE,
                    inputs={"image_path": pointer.image_path, "pointer_id": str(env.id)},
                    outputs={"objects": vision_objects},
                    timing={"ts": time.time(), "latency": time.time() - pointer.frame_ts} if pointer.frame_ts else {},
                    model_fingerprints={"yolo": settings.YOLO_MODEL},
                    debug_refs=None
                )

                out_env = env.derive_child(
                    kind=settings.CHANNEL_VISION_ARTIFACTS,
                    source=ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION),
                    payload=artifact
                )

                await bus.publish(settings.CHANNEL_VISION_ARTIFACTS, out_env)

            except Exception as e:
                logger.error(f"Detection loop error: {e}")
                # Generic Loop Error
                err_payload = VisionEdgeError(
                    camera_id=settings.SOURCE,
                    ts=time.time(),
                    error_type="loop_error",
                    message=str(e)
                )
                err_env = env.derive_child(
                    kind="vision.edge.error",
                    source=ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION),
                    payload=err_payload
                )
                await bus.publish("vision.edge.error", err_env)
