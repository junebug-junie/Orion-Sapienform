
import json, asyncio, aiohttp, time
from typing import Optional
from .settings import settings

try:
    import redis
except Exception:
    redis = None

class EventBus:
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self._queue: asyncio.Queue = asyncio.Queue()
        self._redis = None
        if settings.REDIS_URL and redis is not None:
            try:
                self._redis = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
            except Exception:
                self._redis = None

    def put_nowait(self, event_dict: dict):
        # Make available to in-process subscribers (SSE)
        self.loop.call_soon_threadsafe(self._queue.put_nowait, event_dict)

        # Fire-and-forget to Redis (if configured)
        if self._redis is not None:
            try:
                self._redis.publish("vision.events", json.dumps(event_dict))
            except Exception:
                pass

        # Optionally, async-post webhook
        if settings.EVENT_WEBHOOK_URL:
            self.loop.create_task(self._post_webhook(event_dict))

    async def _post_webhook(self, payload: dict):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(str(settings.EVENT_WEBHOOK_URL), json=payload, timeout=5):
                    pass
        except Exception:
            pass

    async def subscribe(self):
        # Async generator for SSE endpoint
        while True:
            item = await self._queue.get()
            yield item
