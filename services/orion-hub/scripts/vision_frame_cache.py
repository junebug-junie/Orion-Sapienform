"""Latest-frame-per-stream cache for the Vision panel's "Carbon (live)"
option (2026-08-22).

Why this exists: confirmed live (investigation, 2026-08-22) that NO
mechanism anywhere in this repo lets a caller ask "what's the most recent
frame for stream_id=X" -- `vision_scene_inventory` (Postgres) has no sha256
column, `orion-percept-store` is purely content-addressed (you must already
know the hash), and the frame's sha256 otherwise only ever exists
transiently on the bus (`orion:vision:frames`, `VisionFramePointerPayload`).
This module is a Hub-side subscriber that catches those pointers as they
fly by and remembers the latest one per stream_id -- same pattern as
biometrics_cache.py (a background asyncio.create_task loop, an
asyncio.Lock-guarded dict, start()/stop() around a real OrionBusAsync).

Scoped to a small, explicit allowlist of stream_ids (default: just
"carbon") rather than caching every stream in the mesh -- cam0 alone
publishes far more traffic than the Vision panel has any use for, and an
unbounded per-stream dict would be a real, if slow, resource leak whenever
a new stream_id ever appears on the bus.

Bytes are fetched from orion-percept-store server-side (by Hub, not by the
browser) -- see api_routes.py's /api/vision/carbon/latest-frame/image. The
browser never talks to percept-store directly.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

from orion.core.bus.async_service import OrionBusAsync
from orion.schemas.vision import VisionFramePointerPayload

logger = logging.getLogger("orion-hub.vision_frame_cache")


@dataclass
class FramePointer:
    sha256: Optional[str]
    frame_ts: Optional[float]
    width: Optional[int]
    height: Optional[int]
    cached_at: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sha256": self.sha256,
            "frame_ts": self.frame_ts,
            "width": self.width,
            "height": self.height,
            "cached_at": self.cached_at,
        }


class VisionFrameCache:
    def __init__(self, *, enabled: bool, stream_ids: Set[str], channel: str) -> None:
        self.enabled = enabled
        self.stream_ids = stream_ids
        self.channel = channel
        self._latest_by_stream: Dict[str, FramePointer] = {}
        self._lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None
        self._bus: Optional[OrionBusAsync] = None

    async def start(self, bus: OrionBusAsync) -> None:
        if not self.enabled or not self.stream_ids:
            logger.info("Vision frame cache disabled or no stream_ids configured.")
            return
        if not bus or not bus.enabled:
            logger.warning("Vision frame cache not started (bus unavailable).")
            return
        if self._task and not self._task.done():
            return
        self._bus = bus
        self._task = asyncio.create_task(self._run(), name="hub-vision-frame-cache")

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def _run(self) -> None:
        if not self._bus:
            return
        logger.info(
            "Subscribing to %s for stream_ids=%s", self.channel, sorted(self.stream_ids)
        )
        try:
            async with self._bus.subscribe(self.channel) as pubsub:
                async for msg in self._bus.iter_messages(pubsub):
                    await self._handle_message(msg)
        except asyncio.CancelledError:
            logger.info("Vision frame cache task cancelled.")
        except Exception as exc:
            logger.error("Vision frame cache loop failed: %s", exc, exc_info=True)

    async def _handle_message(self, msg: Dict[str, Any]) -> None:
        if not self._bus:
            return
        decoded = self._bus.codec.decode(msg.get("data"))
        if not decoded.ok or not decoded.envelope:
            return
        env = decoded.envelope
        if env.kind != "vision.frame.pointer":
            return
        payload = env.payload
        try:
            frame = (
                payload
                if isinstance(payload, VisionFramePointerPayload)
                else VisionFramePointerPayload(**payload)
            )
        except Exception:
            return
        stream_id = frame.stream_id
        if not stream_id or stream_id not in self.stream_ids:
            return
        if not frame.sha256:
            # This stream isn't using percept_store addressing (e.g. a
            # shared-filesystem node using image_path only) -- nothing this
            # cache can serve back out over HTTP to a browser.
            return
        async with self._lock:
            self._latest_by_stream[stream_id] = FramePointer(
                sha256=frame.sha256,
                frame_ts=frame.frame_ts,
                width=frame.width,
                height=frame.height,
                cached_at=time.time(),
            )

    async def get_latest(self, stream_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            entry = self._latest_by_stream.get(stream_id)
            return entry.to_dict() if entry else None


cache: Optional[VisionFrameCache] = None
