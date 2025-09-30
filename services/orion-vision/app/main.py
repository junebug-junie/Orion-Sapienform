import base64
import asyncio, time, json
from fastapi import FastAPI, Response, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse
from typing import List
import cv2

from .settings import settings
from .capture import CameraSource
from .utils import draw_boxes, encode_jpeg
from .schemas import Detection, Event
from orionbus import OrionBus

from .detectors.motion import MotionDetector
from .detectors.face import FaceDetector
from .detectors.yolo import YoloDetector
from .detectors.presence import PresenceDetector

app = FastAPI(title="Vision Service", version="0.1.0")

#loop = asyncio.get_event_loop()
#bus = EventBus(loop) #local
bus = OrionBus(url=settings.ORION_BUS_URL, enabled=settings.ORION_BUS_ENABLED)

cam = CameraSource(settings.SOURCE)
cam.start()

# Build detector pipeline
detectors = []
if "motion" in settings.DETECTORS.split(","):
    detectors.append(("motion", MotionDetector(min_area=settings.MOTION_MIN_AREA)))
if "face" in settings.DETECTORS.split(","):
    detectors.append(("face", FaceDetector(
        cascade_path=settings.FACE_CASCADE_PATH,
        scale_factor=settings.FACE_SCALE_FACTOR,
        min_neighbors=settings.FACE_MIN_NEIGHBORS,
        min_size=settings.FACE_MIN_SIZE,
    )))
# Optional YOLO (GPU-capable). Enable via ENABLE_YOLO=1 (and run container with --gpus all).
if settings.ENABLE_YOLO:
    classes = {c.strip() for c in settings.YOLO_CLASSES.split(",") if c.strip()} or None
    device = f"cuda:{settings.YOLO_DEVICE}" if settings.YOLO_DEVICE.isdigit() else settings.YOLO_DEVICE
    detectors.append(("yolo", YoloDetector(
        model_path=settings.YOLO_MODEL,
        classes=classes,
        conf=settings.YOLO_CONF,
        device=device
    )))

# Prescense detctors
if settings.ENABLE_PRESENCE:
    detectors.append(("presence", PresenceDetector(
        timeout=settings.PRESENCE_TIMEOUT,
        label=settings.PRESENCE_LABEL
    )))

last_event = None
_frame_counter = 0

async def detector_loop():
    global last_event, _frame_counter
    while True:
        frame = cam.get_frame()
        await asyncio.sleep(0.001)
        if frame is None:
            await asyncio.sleep(0.05)
            continue

        _frame_counter += 1
        if _frame_counter % max(settings.DETECT_EVERY_N_FRAMES, 1) != 0:
            continue

        detections: List[Detection] = []
        annotated = frame.copy()

        for name, det in detectors:
            try:
                boxes = det.detect(frame)
            except Exception:
                boxes = []
            # YOLO returns tuples (x,y,w,h,label,conf); others return (x,y,w,h)
            if boxes and settings.ANNOTATE:
                from .utils import draw_boxes
                if boxes and isinstance(boxes[0], tuple) and len(boxes[0]) == 6:
                    draw_boxes(annotated, [(x,y,w,h) for (x,y,w,h,_,_) in boxes], name)
                else:
                    draw_boxes(annotated, boxes, name)
            for b in boxes:
                if isinstance(b, tuple) and len(b) == 6:
                    x,y,w,h,label,conf = b
                    detections.append(Detection(kind=name, bbox=(x,y,w,h), score=conf, label=label))
                else:
                    x,y,w,h = b
                    detections.append(Detection(kind=name, bbox=(x,y,w,h), score=1.0))

        if detections:
            ev = Event(
                ts = __import__("datetime").datetime.utcnow(),
                stream_id = settings.STREAM_ID,
                detections = detections
            )
            last_event = ev
            # Ensure datetime is serializable
            ev_dict = ev.dict()
            ev_dict["ts"] = ev.ts.isoformat()

            bus.publish("vision.events", ev_dict)

            # Update last annotated frame
            cam.last_frame = annotated

@app.on_event("startup")
async def on_start():
    asyncio.create_task(detector_loop())

@app.get("/health")
async def health():
    return {"ok": True, "detectors": settings.DETECTORS}

@app.get("/", response_class=HTMLResponse)
async def index():
    if not settings.ENABLE_UI:
        return HTMLResponse("<h3>UI disabled</h3>", status_code=200)
    html = """
    <html>
    <head>
        <title>Vision Service</title>
        <style> body { font-family: ui-sans-serif, system-ui, Arial; margin: 1rem; } 
                 #wrap { display: grid; grid-template-columns: 640px 1fr; gap: 1rem; } 
                 #events { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space: pre; background: #111; color:#0f0; padding: 0.75rem; height: 480px; overflow:auto; border-radius: 8px; }
                 img { max-width: 100%; border-radius: 8px; }
                 a { color: #06f; }
        </style>
    </head>
    <body>
        <h1>Vision Service</h1>
        <div id="wrap">
            <div>
               <img src="/stream.mjpg" />
               <p><a href="/snapshot.jpg" target="_blank">Open snapshot</a></p>
            </div>
        </div>
        <script>
            const box = document.getElementById("events");
            const evt = new EventSource("/events");
            evt.onmessage = (e) => {
                box.textContent += e.data + "\n";
                box.scrollTop = box.scrollHeight;
            };
            evt.onerror = () => {
                box.textContent += "[SSE disconnected]\n";
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(html)

@app.get("/snapshot.jpg")
async def snapshot():
    frame = cam.get_frame()
    if frame is None:
        return Response(status_code=503)
    data = encode_jpeg(frame)
    if data is None:
        return Response(status_code=500)
    return Response(content=data, media_type="image/jpeg")

@app.get("/stream.mjpg")
async def mjpeg_stream():
    async def gen():
        boundary = "--frame"
        while True:
            frame = cam.get_frame()
            if frame is None:
                await asyncio.sleep(0.05)
                continue
            data = encode_jpeg(frame)
            if data is None:
                await asyncio.sleep(0.01)
                continue
            yield (b"%s\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % (b"--frame", len(data))) + data + b"\r\n"
            await asyncio.sleep(1.0 / max(settings.FPS, 1))
    headers = {"Content-Type": "multipart/x-mixed-replace; boundary=frame"}
    return StreamingResponse(gen(), headers=headers)

@app.get("/events")
async def sse_events():
    async def event_gen():
        async for item in bus.subscribe():
            yield f"data: {json.dumps(item)}\n\n"
    return StreamingResponse(event_gen(), media_type="text/event-stream")

@app.post("/detect")
async def detect_upload(file: UploadFile = File(...)):
    import numpy as np
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse({"error": "invalid image"}, status_code=400)

    detections = []
    annotated = frame.copy()

    for name, det in detectors:
        if hasattr(det, "detect"):  # Motion, Face, YOLO
            boxes = det.detect(frame)
            if boxes and settings.ANNOTATE:
                from .utils import draw_boxes
                draw_boxes(annotated, boxes, name)
            for (x, y, w, h) in boxes:
                detections.append(
                    Detection(kind=name, bbox=(x, y, w, h), score=1.0)
                )

        elif hasattr(det, "update"):  # PresenceDetector
            events = det.update([d.dict() for d in detections])
            for ev in events:
                detections.append(
                    Detection(kind="presence", bbox=(0,0,0,0), score=1.0, label=ev.get("label", "presence"))
                )

    # Encode annotated image
    jpg_bytes = encode_jpeg(annotated)
    jpg_b64 = base64.b64encode(jpg_bytes).decode("utf-8")

    # Return JSON with image + detections
    payload = {
        "detections": [d.dict() if hasattr(d, "dict") else d for d in detections],
        "image": jpg_b64
    }
    # also push to Orion bus
    if detections:
        await bus.publish("vision.detect.upload", payload)

    return JSONResponse(payload)
