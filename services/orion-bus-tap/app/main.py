from pathlib import Path

import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.telemetry.system_health import SystemHealthV1

from app.settings import settings
import uuid
from datetime import datetime, timezone

# SystemHealthV1 requires boot_id and last_seen_ts. Constructing it without them
# raises inside the heartbeat loop's own try/except, which logs a warning and
# sleeps -- so the service looks alive while publishing no heartbeat at all.
# Confirmed live 2026-08-29 on orion-gpu-cluster-power: one failure per 30s tick,
# indefinitely, and nothing downstream noticed the silence.
# BOOT_ID identifies THIS process run, so a consumer can tell a restart from a
# continuous uptime (same convention as services/orion-whisper-tts/app/main.py).
BOOT_ID = str(uuid.uuid4())


app = FastAPI(title="Orion Bus Tap")

static_dir = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# Global vars for heartbeat
system_bus: OrionBusAsync = None
heartbeat_task: asyncio.Task = None

async def run_heartbeat():
    """Publishes a heartbeat every 30 seconds."""
    logger.info("Heartbeat loop started.")
    try:
        while True:
            if system_bus:
                try:
                    payload = SystemHealthV1(
                        service="orion-bus-tap",
                        version="1.0.0", # Hardcoded or needs settings?
                        node="tap-node",
                        status="ok",
                        boot_id=BOOT_ID,
                        last_seen_ts=datetime.now(timezone.utc),
                        # heartbeat_interval_sec must match this loop's real period. Left at the
                        # schema default of 10.0, orion-equilibrium-service computes
                        # grace = interval * EQUILIBRIUM_GRACE_MULTIPLIER (3.0) = 30.0s and marks the
                        # service "down" once delta > grace (service.py's status check). Publishing
                        # every 30s leaves ZERO margin, so any event-loop delay or bus latency flips
                        # it to down, emits a spurious transition and pushes distress_score.
                        heartbeat_interval_sec=30.0,
                    ).model_dump(mode="json")

                    await system_bus.publish("orion:system:health", BaseEnvelope(
                        kind="system.health.v1",
                        source=ServiceRef(name="orion-bus-tap", version="1.0.0"),
                        payload=payload
                    ))
                except Exception as e:
                    logger.warning(f"Heartbeat failed: {e}")

            await asyncio.sleep(30)
    except asyncio.CancelledError:
        logger.info("Heartbeat loop stopping...")

@app.on_event("startup")
async def startup_event():
    global system_bus, heartbeat_task
    system_bus = OrionBusAsync(settings.ORION_BUS_URL)
    await system_bus.connect()
    heartbeat_task = asyncio.create_task(run_heartbeat())

@app.on_event("shutdown")
async def shutdown_event():
    if heartbeat_task:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
    if system_bus:
        await system_bus.close()

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
