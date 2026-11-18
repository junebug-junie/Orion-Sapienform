"""Bounded Retro-only WebSocket relay; no planner, bus or Hub lifecycle state of its own.

Two observer callbacks, and nothing else leaves the relay:

* ``on_input``: each browser input message, after it was forwarded (the sitting record,
  scripts/retro_sitting.py: who plays, from the button changes);
* ``on_media``: each binary video packet the engine sends, as it arrives, still compressed (Orion's
  look, scripts/retro_watch.py: one packet in 2 s is kept in memory; a look decodes up to 4 of them
  and a model reads them as stills, game company patch 2). The browser's stream is not delayed or
  changed by it.

Both are sync, must not raise and must not wait; the relay holds no state for them. The relay itself
records nothing and calls no model."""
import asyncio
import logging
import json

import aiohttp
from starlette.websockets import WebSocketDisconnect

# The dedicated gateway only configures Uvicorn logging; a generic INFO logger
# would silently lose these diagnostics there (Hub main configures root separately).
logger = logging.getLogger('uvicorn.error.retro_transport')
SEND_TIMEOUT_SEC = .75


async def relay_messages(ws, upstream, channel, sid, *, media_ack=False, on_input=None, on_media=None):
    # A slow viewer must not accumulate old pictures/PCM. Never drop input replies.
    latest = asyncio.Queue(maxsize=1)
    sent = acknowledged = 0
    credit = asyncio.Event()

    async def browser_to_native():
        nonlocal acknowledged
        import time
        count = 0
        start = time.monotonic()
        while True:
            msg = await ws.receive_text()
            if time.monotonic() - start >= 1:
                start = time.monotonic(); count = 0
            count += 1
            if len(msg.encode()) > 4096 or count > 120:
                return 'invalid_browser_message'
            if channel == 'media':
                if not media_ack: return 'invalid_browser_message'
                try: ack = json.loads(msg)
                except ValueError: return 'invalid_media_ack'
                if (not isinstance(ack, dict) or set(ack) != {'ack'} or type(ack['ack']) is not int
                        or not acknowledged < ack['ack'] <= sent):
                    return 'invalid_media_ack'
                acknowledged = ack['ack']; credit.set()
                continue  # presentation acknowledgments never reach native input
            await asyncio.wait_for(upstream.send_str(msg), .25)
            if on_input is not None:
                try: on_input(msg)
                except Exception: logger.warning('retro_relay_on_input_failed sid=%s', sid, exc_info=True)

    async def native_to_browser():
        while True:
            msg = await upstream.receive()
            if msg.type not in (aiohttp.WSMsgType.BINARY, aiohttp.WSMsgType.TEXT):
                return 'upstream_closed'
            if channel == 'media':
                if latest.full(): latest.get_nowait()
                latest.put_nowait(msg)
                if on_media is not None and msg.type == aiohttp.WSMsgType.BINARY:
                    try: on_media(msg.data)
                    except Exception: logger.warning('retro_relay_on_media_failed sid=%s', sid, exc_info=True)
            else:
                await asyncio.wait_for(ws.send_text(msg.data), SEND_TIMEOUT_SEC)

    async def media_to_browser():
        nonlocal sent
        while True:
            # send() can complete after buffering in TCP/the proxy, long before
            # presentation. Bound that whole path, not just our latest queue.
            while media_ack and sent - acknowledged >= 3:
                credit.clear()
                await credit.wait()
            msg = await latest.get()
            sent += 1
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
