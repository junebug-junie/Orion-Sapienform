from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from orion.core.bus.async_service import OrionBusAsync

from app.settings import settings

app = FastAPI(title="Orion Bus Tap")

static_dir = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


def _decode_envelope(bus: OrionBusAsync, payload: bytes | str) -> dict:
    decoded = bus.codec.decode(payload)
    if decoded.ok:
        return decoded.envelope.model_dump(by_alias=True, mode="json")
    return {"error": decoded.error, "raw": decoded.raw}


@app.websocket("/ws/tap")
async def tap_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    bus = OrionBusAsync(settings.ORION_BUS_URL)
    await bus.connect()

    logger.info("Bus tap connected: {} pattern={}", settings.ORION_BUS_URL, settings.TAP_PATTERN)

    try:
        async with bus.subscribe(settings.TAP_PATTERN, patterns=True) as pubsub:
            async for message in bus.iter_messages(pubsub):
                channel = message.get("channel")
                if isinstance(channel, bytes):
                    channel = channel.decode("utf-8", errors="replace")
                envelope = _decode_envelope(bus, message.get("data", b""))
                await websocket.send_json({"channel": channel, "envelope": envelope})
    except WebSocketDisconnect:
        logger.info("Bus tap websocket disconnected")
    finally:
        await bus.close()
