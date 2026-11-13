"""Producers' one call into the body economy: ``bid(want) -> WantDecisionV1 | None`` (Patch C1).

Design: docs/superpowers/specs/2026-10-23-organism-body-economy-design.md, "Client library".

POSTs a ``WantV1`` to orion-durable-runs ``POST /physiology/bid`` with a short timeout.

**Before Patch D (now): never fail open to "funded".** Any failure -- unreachable, timeout,
non-200, malformed answer, the flag off, no URL -- returns ``None``. A caller that gets ``None``
keeps its current gate, exactly as if this call did not exist. A decision that came back is
SHADOW (``shadow=True``): log it next to the real gate's decision, do not obey it.

Stdlib only (urllib), so any producer can import it without a dependency. ``abid`` runs the same
call in a worker thread for async callers.

``release(want_id)`` (Patch C3a): the want's act closed, so its reservation ends now instead of at
expiry (``POST /physiology/bid/{want_id}/release``, derived from the bid URL). Returns the number of
reservations released, or None on any failure; ``arelease`` is the async form.

Env (read at call time, so a test or a service can set it without import order games):

* ``ORION_PHYSIOLOGY_BID_URL``: default ``http://orion-athena-durable-runs:8121/physiology/bid``.
* ``ORION_PHYSIOLOGY_BID_TIMEOUT_SEC``: default 0.5.
* ``ORION_PHYSIOLOGY_BID_CLIENT_ENABLED``: default true. false: ``bid`` returns None without a call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

from orion.schemas.physiology_want import WantDecisionV1, WantV1

logger = logging.getLogger("orion.physiology.client")

DEFAULT_URL = "http://orion-athena-durable-runs:8121/physiology/bid"
DEFAULT_TIMEOUT_SEC = 0.5


def _enabled() -> bool:
    return os.environ.get("ORION_PHYSIOLOGY_BID_CLIENT_ENABLED", "true").strip().lower() not in ("0", "false", "no", "off")


def bid(want: WantV1, *, url: Optional[str] = None, timeout_sec: Optional[float] = None) -> Optional[WantDecisionV1]:
    """The shadow decision, or None on ANY failure (the caller keeps its own gate)."""
    if not _enabled():
        return None
    target = url or os.environ.get("ORION_PHYSIOLOGY_BID_URL") or DEFAULT_URL
    try:
        timeout = float(timeout_sec if timeout_sec is not None else os.environ.get("ORION_PHYSIOLOGY_BID_TIMEOUT_SEC", DEFAULT_TIMEOUT_SEC))
    except ValueError:
        timeout = DEFAULT_TIMEOUT_SEC
    body = want.model_dump_json().encode("utf-8")
    req = urllib.request.Request(target, data=body, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:   # noqa: S310 -- internal service URL
            if resp.status != 200:
                logger.info("physiology_bid_non_200 status=%s family=%s", resp.status, want.family)
                return None
            raw = resp.read()
    except Exception as exc:  # noqa: BLE001 -- ANY failure (incl. http.client errors on read) is None
        logger.info("physiology_bid_unavailable family=%s err=%s", want.family, type(exc).__name__)
        return None
    try:
        decision = WantDecisionV1.model_validate(json.loads(raw))
    except Exception as exc:  # noqa: BLE001 -- a malformed answer is no answer
        logger.info("physiology_bid_malformed family=%s err=%s", want.family, type(exc).__name__)
        return None
    if decision.want_id != want.want_id:
        logger.info("physiology_bid_wrong_want family=%s", want.family)
        return None
    return decision


def _release_url(want_id: str, url: Optional[str]) -> str:
    base = (url or os.environ.get("ORION_PHYSIOLOGY_BID_URL") or DEFAULT_URL).rstrip("/")
    return f"{base}/{urllib.parse.quote(want_id, safe='')}/release"


def release(want_id: str, *, url: Optional[str] = None, timeout_sec: Optional[float] = None) -> Optional[int]:
    """End the want's reservation now (its act closed). The count released, or None on ANY failure."""
    if not _enabled() or not want_id:
        return None
    try:
        timeout = float(timeout_sec if timeout_sec is not None else os.environ.get("ORION_PHYSIOLOGY_BID_TIMEOUT_SEC", DEFAULT_TIMEOUT_SEC))
    except ValueError:
        timeout = DEFAULT_TIMEOUT_SEC
    req = urllib.request.Request(_release_url(want_id, url), data=b"", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:   # noqa: S310 -- internal service URL
            if resp.status != 200:
                return None
            body = json.loads(resp.read())
        n = body.get("released") if isinstance(body, dict) else None
        return int(n) if isinstance(n, int) and not isinstance(n, bool) else None
    except Exception as exc:  # noqa: BLE001
        logger.info("physiology_release_unavailable want=%s err=%s", want_id, type(exc).__name__)
        return None


async def arelease(want_id: str, *, url: Optional[str] = None, timeout_sec: Optional[float] = None) -> Optional[int]:
    """``release`` for async callers (worker thread; same None-on-failure rule)."""
    try:
        return await asyncio.to_thread(release, want_id, url=url, timeout_sec=timeout_sec)
    except Exception:  # noqa: BLE001
        return None


async def abid(want: WantV1, *, url: Optional[str] = None, timeout_sec: Optional[float] = None) -> Optional[WantDecisionV1]:
    """``bid`` for async callers (runs in a worker thread; same None-on-failure rule)."""
    try:
        return await asyncio.to_thread(bid, want, url=url, timeout_sec=timeout_sec)
    except Exception:  # noqa: BLE001 -- never fail a producer because the economy could not answer
        return None


__all__ = ["DEFAULT_URL", "abid", "arelease", "bid", "release"]
