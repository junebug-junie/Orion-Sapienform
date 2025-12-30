# services/orion-hub/scripts/utils_bus_debug.py
from __future__ import annotations

import inspect
import json
import logging
import os
import uuid
from typing import Any, Dict, Optional

log = logging.getLogger("orion-hub.busdebug")


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)


def new_corr_id() -> str:
    return str(uuid.uuid4())


def debug_enabled() -> bool:
    return os.getenv("ORION_HUB_DEBUG", "0").lower() in ("1", "true", "yes", "on")


async def publish_debug(
    bus: Any,
    channel: str,
    payload: Dict[str, Any],
    *,
    corr_id: Optional[str] = None,
) -> None:
    """
    Publish a payload onto the Orion bus, correctly awaiting if bus.publish is async.

    - If ORION_HUB_DEBUG=1, logs every publish with payload keys + JSON.
    - Works with both sync and async bus implementations.
    """
    if debug_enabled():
        log.info(
            "[PUB] channel=%s corr_id=%s keys=%s payload=%s",
            channel,
            corr_id,
            sorted(list(payload.keys())),
            _safe_json(payload),
        )

    res = bus.publish(channel, payload)
    if inspect.isawaitable(res):
        await res


async def publish_or_warn(
    bus: Any,
    channel: str,
    payload: Dict[str, Any],
    *,
    corr_id: Optional[str] = None,
) -> bool:
    """
    Publish with guardrails:
    - Correctly awaits async publishes (prevents "coroutine was never awaited")
    - Logs a WARNING on failure (does not silently swallow)
    Returns True on success, False on failure.
    """
    try:
        await publish_debug(bus, channel, payload, corr_id=corr_id)
        return True
    except Exception:
        log.exception(
            "[PUB-FAIL] channel=%s corr_id=%s keys=%s payload=%s",
            channel,
            corr_id,
            sorted(list(payload.keys())),
            _safe_json(payload),
        )
        return False
