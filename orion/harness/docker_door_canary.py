"""Docker-door canary: inside a scheduled window, Orion's turns use the Docker door, not the socket.

Juniper, 2026-11-24: "build this next door like a canary. We schedule a time to swap over ... run
in realtime in the new setup and see what breaks. Then we can swap back to normal current state."

The window lives in the governor env, ``ORION_DOCKER_DOOR_CANARY_WINDOWS``: a comma list of
``START/END`` UTC ISO times, for example ``2026-11-26T15:00Z/2026-11-26T19:00Z``. Inside a
window, a turn's ``DOCKER_HOST`` points at the door (``ORION_DOCKER_DOOR_URL``); outside it, the
turn keeps the container's own setting (the real socket). The swap back needs no restart: every
turn asks again. If the door does not answer, the turn keeps the socket (Orion is never cut off)
and the miss is logged. The door's own record (host JSONL) is the list of what broke.
Design: docs/superpowers/specs/2026-11-24-orion-safe-autonomy-in-place-design.md.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Awaitable, Callable, Mapping, Optional
from urllib.parse import urlparse

logger = logging.getLogger("orion.harness.docker_door_canary")

WINDOWS_ENV = "ORION_DOCKER_DOOR_CANARY_WINDOWS"
URL_ENV = "ORION_DOCKER_DOOR_URL"
DEFAULT_URL = "tcp://orion-athena-docker-door:2375"
# A turn waits at most this long on the door check, so a broken door never stalls a turn or
# keeps a warm slot busy. The answer is kept for PROBE_TTL_SEC, so a window of busy turns costs
# one check per half minute, not one per turn.
PROBE_WAIT_SEC = 2.0
PROBE_TTL_SEC = 30.0

Window = tuple[datetime, datetime]


def _ts(text: str) -> datetime:
    ts = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    if ts.tzinfo is None:
        raise ValueError(f"window time needs a zone (use Z for UTC): {text!r}")
    return ts


def parse_windows(text: str) -> list[Window]:
    """``START/END[,START/END...]`` -> windows. Raises ValueError on any bad part."""
    windows: list[Window] = []
    for part in (p.strip() for p in (text or "").split(",")):
        if not part:
            continue
        start_s, sep, end_s = part.partition("/")
        if not sep:
            raise ValueError(f"window needs START/END: {part!r}")
        start, end = _ts(start_s), _ts(end_s)
        if end <= start:
            raise ValueError(f"window ends before it starts: {part!r}")
        windows.append((start, end))
    return windows


def in_window(now: datetime, windows: list[Window]) -> bool:
    return any(start <= now < end for start, end in windows)


async def door_answers(url: str, timeout: float = PROBE_WAIT_SEC) -> bool:
    """True when the door answers ``GET /_ping`` with 200, which also proves it reaches the daemon.

    Native asyncio: no thread is left behind when the check times out.
    """
    parsed = urlparse(url)
    if parsed.scheme != "tcp" or not parsed.hostname:
        return False

    async def ping() -> bool:
        reader, writer = await asyncio.open_connection(parsed.hostname, parsed.port or 2375)
        try:
            writer.write(b"GET /_ping HTTP/1.0\r\nHost: docker\r\n\r\n")
            await writer.drain()
            status = await reader.readline()
        finally:
            writer.close()
        return status.split(b" ")[1:2] == [b"200"]

    try:
        return await asyncio.wait_for(ping(), timeout)
    except (OSError, asyncio.TimeoutError, ValueError):
        return False


_probe_cache: dict[str, tuple[float, bool]] = {}


async def _cached_door_answers(url: str) -> bool:
    hit = _probe_cache.get(url)
    if hit is not None and time.monotonic() - hit[0] < PROBE_TTL_SEC:
        return hit[1]
    ok = await door_answers(url)
    _probe_cache[url] = (time.monotonic(), ok)
    return ok


def window_door_url(environ: Mapping[str, str] = os.environ, now: Optional[datetime] = None) -> Optional[str]:
    """The door URL when ``now`` is inside a canary window, else None. Never touches the network."""
    raw = str(environ.get(WINDOWS_ENV) or "").strip()
    if not raw:
        return None
    try:
        windows = parse_windows(raw)
    except ValueError as exc:
        logger.warning("DOCKER_DOOR_CANARY bad_window value=%r error=%s (socket kept)", raw, exc)
        return None
    if not in_window(now or datetime.now(timezone.utc), windows):
        return None
    return str(environ.get(URL_ENV) or DEFAULT_URL).strip()


async def door_docker_host(
    *,
    environ: Mapping[str, str] = os.environ,
    now: Optional[datetime] = None,
    answers: Optional[Callable[[str], Awaitable[bool]]] = None,
    correlation_id: str = "",
) -> Optional[str]:
    """The door URL when this turn should use the door, else None (keep the socket)."""
    url = window_door_url(environ, now)
    if url is None:
        return None
    if not await (answers or _cached_door_answers)(url):
        logger.warning("DOCKER_DOOR_CANARY corr=%s door=unreachable url=%s (socket kept)", correlation_id, url)
        return None
    logger.info("DOCKER_DOOR_CANARY corr=%s door=on url=%s", correlation_id, url)
    return url


async def turn_docker_host(environ: Mapping[str, str] = os.environ, **kwargs) -> str:
    """Value for a warm slot's per-turn env file: the door, or the container's own DOCKER_HOST.

    An empty value makes the docker CLI use its default socket, the same as unset.
    """
    return await door_docker_host(environ=environ, **kwargs) or str(environ.get("DOCKER_HOST") or "")
