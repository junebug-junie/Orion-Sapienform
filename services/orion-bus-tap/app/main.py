import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from redis.asyncio import Redis

from app.settings import settings

app = FastAPI(title="Orion Bus Tap")

static_dir = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


def _parse_envelope(payload: str | bytes) -> dict:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8", errors="replace")
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return {"raw": payload}


@app.websocket("/ws/tap")
async def tap_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    redis = Redis.from_url(settings.ORION_BUS_URL, decode_responses=True)
    pubsub = redis.pubsub()

    await pubsub.psubscribe(settings.TAP_PATTERN)
    logger.info("Bus tap connected: {} pattern={}", settings.ORION_BUS_URL, settings.TAP_PATTERN)

    try:
        async for message in pubsub.listen():
            if message is None:
                continue
            if message.get("type") != "pmessage":
                continue
            channel = message.get("channel")
            envelope = _parse_envelope(message.get("data", ""))
            await websocket.send_json({"channel": channel, "envelope": envelope})
    except WebSocketDisconnect:
        logger.info("Bus tap websocket disconnected")
    finally:
        await pubsub.close()
        await redis.close()
