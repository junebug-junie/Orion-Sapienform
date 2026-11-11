"""Bounded Retro-only WebSocket relay; no planner, bus or Hub lifecycle state."""
import asyncio
import logging

import aiohttp
from starlette.websockets import WebSocketDisconnect

# The dedicated gateway only configures Uvicorn logging; a generic INFO logger
# would silently lose these diagnostics there (Hub main configures root separately).
logger = logging.getLogger('uvicorn.error.retro_transport')
SEND_TIMEOUT_SEC = .75


async def relay_messages(ws, upstream, channel, sid):
    # A slow viewer must not accumulate old pictures/PCM. Never drop input replies.
    latest = asyncio.Queue(maxsize=1)

    async def browser_to_native():
        import time
        count = 0
        start = time.monotonic()
        while True:
            msg = await ws.receive_text()
            if time.monotonic() - start >= 1:
                start = time.monotonic(); count = 0
            count += 1
            if channel == 'media' or len(msg.encode()) > 4096 or count > 120:
                return 'invalid_browser_message'
            await asyncio.wait_for(upstream.send_str(msg), .25)

    async def native_to_browser():
        while True:
            msg = await upstream.receive()
            if msg.type not in (aiohttp.WSMsgType.BINARY, aiohttp.WSMsgType.TEXT):
                return 'upstream_closed'
            if channel == 'media':
                if latest.full(): latest.get_nowait()
                latest.put_nowait(msg)
            else:
                await asyncio.wait_for(ws.send_text(msg.data), SEND_TIMEOUT_SEC)

    async def media_to_browser():
        while True:
            msg = await latest.get()
            send = ws.send_bytes if msg.type == aiohttp.WSMsgType.BINARY else ws.send_text
            # On timeout the connection is closed by the caller. A partially sent
            # WebSocket message must never be retried on the same connection.
            await asyncio.wait_for(send(msg.data), SEND_TIMEOUT_SEC)

    tasks = [asyncio.create_task(browser_to_native()), asyncio.create_task(native_to_browser())]
    if channel == 'media': tasks.append(asyncio.create_task(media_to_browser()))
    reason = 'cancelled'
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            try:
                reason = task.result()
            except WebSocketDisconnect:
                reason = 'browser_disconnected'
            except asyncio.TimeoutError:
                reason = 'send_timeout'
            except (aiohttp.ClientError, OSError, RuntimeError):
                reason = 'transport_error'
    finally:
        for task in tasks: task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info('retro_relay_closed sid=%s channel=%s reason=%s', sid, channel, reason)
    return reason
