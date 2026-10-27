"""Producers' one call into the body economy: ``bid(want) -> WantDecisionV1 | None`` (Patch C1).

Design: docs/superpowers/specs/2026-10-23-organism-body-economy-design.md, "Client library".

POSTs a ``WantV1`` to orion-durable-runs ``POST /physiology/bid`` with a short timeout.

**Before Patch D (now): never fail open to "funded".** Any failure -- unreachable, timeout,
non-200, malformed answer, the flag off, no URL -- returns ``None``. A caller that gets ``None``
keeps its current gate, exactly as if this call did not exist. A decision that came back is
SHADOW (``shadow=True``): log it next to the real gate's decision, do not obey it.

Stdlib only (urllib), so any producer can import it without a dependency. ``abid`` runs the same
call in a worker thread for async callers.

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
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
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


async def abid(want: WantV1, *, url: Optional[str] = None, timeout_sec: Optional[float] = None) -> Optional[WantDecisionV1]:
    """``bid`` for async callers (runs in a worker thread; same None-on-failure rule)."""
    try:
        return await asyncio.to_thread(bid, want, url=url, timeout_sec=timeout_sec)
    except Exception:  # noqa: BLE001 -- never fail a producer because the economy could not answer
        return None


__all__ = ["DEFAULT_URL", "abid", "bid"]
