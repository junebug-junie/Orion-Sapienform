from __future__ import annotations

from typing import List, Optional

import orjson
from redis.asyncio import Redis

from orion.schemas.pad import PadEventV1, StateFrameV1, TensorBlobV1


class PadStore:
    def __init__(
        self,
        *,
        redis: Redis,
        events_stream_key: str,
        frames_stream_key: str,
        stream_maxlen: int,
        event_ttl: int,
        frame_ttl: int,
    ):
        self.redis = redis
        self.events_stream_key = events_stream_key
        self.frames_stream_key = frames_stream_key
        self.stream_maxlen = stream_maxlen
        self.event_ttl = event_ttl
        self.frame_ttl = frame_ttl
        self.latest_event_key = f"{events_stream_key}:latest"
        self.latest_frame_key = f"{frames_stream_key}:latest"

    async def store_event(self, event: PadEventV1) -> None:
        payload = event.model_dump(mode="json")
        data = orjson.dumps(payload)
        await self.redis.set(self.latest_event_key, data, ex=self.event_ttl)
        await self.redis.xadd(self.events_stream_key, {"data": data}, maxlen=self.stream_maxlen, approximate=True)

    async def store_frame(self, frame: StateFrameV1) -> None:
        payload = frame.model_dump(mode="json")
        data = orjson.dumps(payload)
        await self.redis.set(self.latest_frame_key, data, ex=self.frame_ttl)
        await self.redis.xadd(self.frames_stream_key, {"data": data}, maxlen=self.stream_maxlen, approximate=True)

    async def get_latest_frame(self) -> Optional[StateFrameV1]:
        raw = await self.redis.get(self.latest_frame_key)
        if raw is None:
            return None
        return StateFrameV1.model_validate(orjson.loads(raw))

    async def get_latest_tensor(self) -> Optional[TensorBlobV1]:
        frame = await self.get_latest_frame()
        if frame is None:
            return None
        return frame.tensor

    async def _parse_stream(self, key: str, limit: int) -> List[dict]:
        entries = await self.redis.xrevrange(key, max="+", min="-", count=limit)
        payloads: List[dict] = []
        for _, fields in entries:
            data = fields.get(b"data") or fields.get("data")
            if data:
                try:
                    payloads.append(orjson.loads(data))
                except Exception:
                    continue
        return list(reversed(payloads))

    async def get_frames(self, limit: int = 10) -> List[StateFrameV1]:
        payloads = await self._parse_stream(self.frames_stream_key, limit)
        frames: List[StateFrameV1] = []
        for payload in payloads:
            try:
                frames.append(StateFrameV1.model_validate(payload))
            except Exception:
                continue
        return frames

    async def get_salient_events(self, limit: int = 20) -> List[PadEventV1]:
        payloads = await self._parse_stream(self.events_stream_key, limit)
        events: List[PadEventV1] = []
        for payload in payloads:
            try:
                events.append(PadEventV1.model_validate(payload))
            except Exception:
                continue
        events = sorted(events, key=lambda e: e.salience, reverse=True)
        return events[:limit]
