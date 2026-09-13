"""Isolated typed transport: real envelopes/handlers, no production Redis calls."""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

from orion.core.bus.codec import OrionCodec


class TypedBus:
    def __init__(self):
        self.codec = OrionCodec()
        self.handlers = {}
        self.events = []
        self.inflight_rpc = {}
        self.timeout_calls = []
        self.tasks = set()
        self.enabled = True
        self.subscriptions = {}
        self.redis = SimpleNamespace(pubsub_numsub=self._numsub)

    async def _numsub(self, channel):
        return [(channel, int(channel in self.handlers) + len(self.subscriptions.get(channel, [])))]

    @asynccontextmanager
    async def subscribe(self, channel):
        queue = asyncio.Queue()
        self.subscriptions.setdefault(channel, []).append(queue)
        async def get_message(*, ignore_subscribe_messages=True, timeout=0):
            try:
                return await asyncio.wait_for(queue.get(), timeout) if timeout else queue.get_nowait()
            except (TimeoutError, asyncio.QueueEmpty):
                return None
        try:
            yield SimpleNamespace(get_message=get_message)
        finally:
            self.subscriptions[channel].remove(queue)
            if not self.subscriptions[channel]:
                del self.subscriptions[channel]

    async def publish(self, channel, envelope):
        decoded = self.codec.decode(self.codec.encode(envelope))
        assert decoded.ok, decoded.error
        envelope = decoded.envelope
        self.events.append((channel, envelope))
        raw = {"type": "message", "channel": channel, "data": self.codec.encode(envelope)}
        for queue in self.subscriptions.get(channel, []):
            queue.put_nowait(raw)
        future = self.inflight_rpc.get(channel)
        if future is not None and not future.done():
            future.set_result(raw)
        handler = self.handlers.get(channel)
        if handler is not None:
            async def serve():
                try:
                    await handler(envelope)
                except BaseException as exc:
                    response = self.inflight_rpc.get(envelope.reply_to)
                    if response is not None and not response.done():
                        response.set_exception(exc)
                    else:
                        raise
            task = asyncio.create_task(serve())
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)
        return 1

    async def rpc_request(self, channel, envelope, *, reply_channel, timeout_sec):
        assert reply_channel not in self.inflight_rpc, "RPC reply channels must be unique"
        future = asyncio.get_running_loop().create_future()
        self.inflight_rpc[reply_channel] = future
        self.timeout_calls.append((channel, timeout_sec))
        try:
            await self.publish(channel, envelope)
            return await asyncio.wait_for(future, timeout_sec)
        finally:
            self.inflight_rpc.pop(reply_channel, None)

    async def drain(self):
        while self.tasks:
            pending = list(self.tasks)
            await asyncio.gather(*pending)
            # gather() on already-finished tasks need not yield to their done
            # callbacks. Remove the awaited batch ourselves so drain cannot spin.
            self.tasks.difference_update(pending)

    async def close(self):
        for task in list(self.tasks):
            task.cancel()
        await asyncio.gather(*list(self.tasks), return_exceptions=True)

    async def reconnect(self):
        return None
