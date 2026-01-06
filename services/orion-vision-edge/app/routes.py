# app/routes.py
import asyncio
import base64
import json
from datetime import datetime
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)

from .context import settings, bus, camera
from .detector_worker import run_detectors_on_frame
from .utils import encode_jpeg

router = APIRouter()

# Path to static index.html
BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML_PATH = BASE_DIR / "static" / "index.html"


@router.get("/health")
async def health():
    return {
        "ok": True,
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "stream_id": settings.STREAM_ID,
        "source": settings.SOURCE,
        "detectors": settings.detector_names,
        "bus_enabled": settings.ORION_BUS_ENABLED,
        "events_publish_raw": settings.VISION_EVENTS_PUBLISH_RAW,
        "events_subscribe_raw": settings.VISION_EVENTS_SUBSCRIBE_RAW,
    }


@router.get("/", response_class=HTMLResponse)
async def index():
    """
    Serve the static HTML dashboard for the vision edge service.
    """
    if not settings.ENABLE_UI:
        return HTMLResponse("<h3>UI disabled</h3>", status_code=200)

    if not INDEX_HTML_PATH.exists():
        # Fail loudly but clearly if the static file is missing
        return HTMLResponse(
            "<h3>UI error: static index.html not found.</h3>",
            status_code=500,
        )

    html = INDEX_HTML_PATH.read_text(encoding="utf-8")
    return HTMLResponse(html)


@router.get("/snapshot.jpg")
async def snapshot():
    frame = camera.get_frame()
    if frame is None:
        return Response(status_code=503)
    data = encode_jpeg(frame)
    if data is None:
        return Response(status_code=500)
    return Response(content=data, media_type="image/jpeg")


@router.get("/stream.mjpg")
async def mjpeg_stream():
    async def gen():
        while True:
            frame = camera.get_frame()
            if frame is None:
                await asyncio.sleep(0.05)
                continue
            data = encode_jpeg(frame)
            if data is None:
                await asyncio.sleep(0.01)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(data)}\r\n\r\n".encode("ascii")
                + data
                + b"\r\n"
            )
            await asyncio.sleep(1.0 / max(settings.FPS, 1))

    headers = {"Content-Type": "multipart/x-mixed-replace; boundary=frame"}
    return StreamingResponse(gen(), headers=headers)


@router.get("/events")
def sse_events():
    """
    Developer/debug SSE endpoint.

    Streams messages from the configured raw vision channel.
    """
    if not settings.ORION_BUS_ENABLED:
        def disabled_gen():
            yield "data: {\"error\": \"bus disabled\"}\n\n"
        return StreamingResponse(disabled_gen(), media_type="text/event-stream")

    def event_gen():
        for msg in bus.subscribe(settings.VISION_EVENTS_SUBSCRIBE_RAW):
            data = msg.get("data", {})
            try:
                payload = json.dumps(data)
            except TypeError:
                payload = json.dumps({"raw": str(data)})
            yield f"data: {payload}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@router.post("/detect")
async def detect_upload(file: UploadFile = File(...)):
    """
    On-demand detection for uploaded images.

    - Runs the same detector pipeline (frame + event detectors)
    - Returns detections + annotated image (base64)
    - Also publishes a payload to the bus for downstream services
      on a secondary channel ("vision.detect.upload")
    """
    raw_bytes = await file.read()
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse({"error": "invalid image"}, status_code=400)

    annotated, detections = run_detectors_on_frame(frame)

    jpg_bytes = encode_jpeg(annotated)
    jpg_b64 = base64.b64encode(jpg_bytes).decode("utf-8") if jpg_bytes else ""

    payload = {
        "stream_id": settings.STREAM_ID,
        "ts": datetime.utcnow().isoformat(),
        "detections": [d.model_dump() for d in detections],
        "image_b64": jpg_b64,
    }

    if detections:
        bus.publish("vision.detect.upload", payload)

    return JSONResponse(payload)
