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
import socket
from datetime import datetime, timezone
from typing import Callable, Mapping, Optional
from urllib.parse import urlparse

logger = logging.getLogger("orion.harness.docker_door_canary")

WINDOWS_ENV = "ORION_DOCKER_DOOR_CANARY_WINDOWS"
URL_ENV = "ORION_DOCKER_DOOR_URL"
DEFAULT_URL = "tcp://orion-athena-docker-door:2375"
# The probe runs in a thread; this caps how long a turn waits on it (DNS is not under the
# socket timeout), so a broken door can never stall a turn or keep a warm slot busy.
PROBE_WAIT_SEC = 2.0

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


def door_answers(url: str, timeout: float = 0.5) -> bool:
    parsed = urlparse(url)
    if parsed.scheme != "tcp" or not parsed.hostname:
        return False
    try:
        with socket.create_connection((parsed.hostname, parsed.port or 2375), timeout=timeout):
            return True
    except OSError:
        return False


def door_docker_host(
    *,
    environ: Mapping[str, str] = os.environ,
    now: Optional[datetime] = None,
    answers: Optional[Callable[[str], bool]] = None,
    correlation_id: str = "",
) -> Optional[str]:
    """The door URL when this turn should use the door, else None (keep the socket)."""
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
    url = str(environ.get(URL_ENV) or DEFAULT_URL).strip()
    if not (answers or door_answers)(url):
        logger.warning("DOCKER_DOOR_CANARY corr=%s door=unreachable url=%s (socket kept)", correlation_id, url)
        return None
    logger.info("DOCKER_DOOR_CANARY corr=%s door=on url=%s", correlation_id, url)
    return url


def turn_docker_host(environ: Mapping[str, str] = os.environ, **kwargs) -> str:
    """Value for a warm slot's per-turn env file: the door, or the container's own DOCKER_HOST.

    An empty value makes the docker CLI use its default socket, the same as unset.
    """
    return door_docker_host(environ=environ, **kwargs) or str(environ.get("DOCKER_HOST") or "")


async def door_docker_host_async(**kwargs) -> Optional[str]:
    """``door_docker_host`` off the event loop: the probe is a blocking connect."""
    try:
        return await asyncio.wait_for(asyncio.to_thread(door_docker_host, **kwargs), PROBE_WAIT_SEC)
    except asyncio.TimeoutError:
        logger.warning("DOCKER_DOOR_CANARY corr=%s door=probe_timeout (socket kept)", kwargs.get("correlation_id", ""))
        return None


async def turn_docker_host_async(environ: Mapping[str, str] = os.environ, **kwargs) -> str:
    """``turn_docker_host`` off the event loop."""
    return await door_docker_host_async(environ=environ, **kwargs) or str(environ.get("DOCKER_HOST") or "")
