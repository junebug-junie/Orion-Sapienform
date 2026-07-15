"""Cross-turn continuity, within one service, for "what tool fetched content
last turn of this session".

The FCC motor (HarnessRunner.run(), orion-harness-governor) knows, per turn,
which tools were used to fetch static content (see
tool_provenance_audit.fetch_shaped_tool_names). That fact is useful on the
*next* turn of the same conversation -- but each turn is an independent
HarnessRunner.run() call with no shared process memory, only the Redis bus
already available via OrionBusAsync.

NOTE: this is deliberately NOT cross-service. An earlier version of this
module tried to surface this into orion-cortex-exec's chat pipeline on the
theory that a cortex-exec turn is "what comes after" a harness turn in the
same conversation -- that's wrong. orion/hub/turn_orchestrator.py's
run_unified_turn (harness path) and cortex-exec's chat() RPC are mutually
exclusive per-message branches in services/orion-hub/scripts/websocket_handler.py
(client_mode == "orion" hits `continue` right after the harness call and
never reaches the cortex-exec branch) -- they're two different modes, not
sequential turns of one thing. The only turn that legitimately comes "next"
after a harness turn is another harness turn of the same session_id, so both
sides of this cache live here, in orion/harness/.

Write: publish_last_tool_fetch(), called at the end of HarnessRunner.run().
Read: read_last_tool_fetch(), called at the start of the next run() for the
same session_id.

Key: ``orion:harness:last_tool_fetch:{session_id}``
Payload: ``{"tool_names": [...], "correlation_id": "...", "at": "<ISO8601 UTC>"}``
TTL: 600s (single SETEX call, not SET+EXPIRE).

KNOWN LIMITATION, disclosed not silently shipped: session_id is loaded once
from browser localStorage and reused across every tab on the same origin
(see the comment at services/orion-hub/scripts/websocket_handler.py's
session_id handling, which already flags this same value as cross-tab-shared
for a different reason -- its own turn-cancel registry deliberately does NOT
key on session_id for exactly this hazard). This cache does key on it, so
two tabs open to the same origin/session will cross-contaminate: tab A's
fetch can surface as "last turn you fetched X" on tab B's next turn, which
never happened in tab B. Low severity (a continuity-confusion nit, not a
content/security leak) and the multi-tab-same-session case may not be a
supported usage pattern at all -- but worth fixing properly (a per-tab or
per-turn-chain identifier) if that assumption turns out to be wrong.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("orion.harness.last_tool_fetch_cache")

_KEY_PREFIX = "orion:harness:last_tool_fetch"


async def publish_last_tool_fetch(
    bus: Any,
    *,
    session_id: str | None,
    correlation_id: str,
    tool_names: list[str],
    ttl_seconds: int = 600,
) -> None:
    """Best-effort write of this turn's fetch-shaped tool names to Redis.

    No-op (no Redis call at all) when there's no session to key on or no
    fetch-shaped tool was actually used this turn -- an empty/absent signal
    isn't worth a cache write and a stale entry lingering past this turn's
    relevance. Never raises: this runs on the hot chat-turn path inline with
    HarnessRunner.run(), and a Redis hiccup here must not break a turn that
    otherwise completed successfully.
    """
    if not session_id or not tool_names:
        if tool_names and not session_id:
            # Distinguishable from "no fetch tool used this turn" (the
            # common, silent no-op) -- a session-less turn using a fetch
            # tool is unusual enough to be worth a trace, since it means
            # continuity can never kick in for that turn's conversation.
            logger.info("last_tool_fetch_write_skipped reason=no_session_id tool_names=%s", tool_names)
        return

    key = f"{_KEY_PREFIX}:{session_id}"
    payload = json.dumps(
        {
            "tool_names": tool_names,
            "correlation_id": correlation_id,
            "at": datetime.now(timezone.utc).isoformat(),
        }
    )
    try:
        await bus.redis.setex(key, ttl_seconds, payload)
        logger.info("last_tool_fetch_write key=%s tool_names=%s", key, tool_names)
    except Exception:
        logger.warning(
            "last_tool_fetch_write_failed key=%s tool_names=%s",
            key,
            tool_names,
            exc_info=True,
        )


_EXPECTED_KEYS = ("tool_names", "correlation_id", "at")


async def read_last_tool_fetch(bus: Any, *, session_id: str | None) -> dict[str, Any] | None:
    """Read the immediately-prior turn's tool-fetch record for this session.

    Fail-open: returns None (never raises) when session_id is falsy, the key
    is absent, Redis errors, or the payload is malformed/incomplete. Called
    at the start of HarnessRunner.run() so the same turn that reads it can
    render it into that turn's own compiled prompt.
    """
    if not session_id:
        return None

    key = f"{_KEY_PREFIX}:{session_id}"
    try:
        raw = await bus.redis.get(key)
    except Exception:
        logger.warning("last_tool_fetch_read key=%s found=False redis_error", key, exc_info=True)
        return None

    found = raw is not None
    logger.info("last_tool_fetch_read key=%s found=%s", key, found)
    if not found:
        return None

    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        parsed = json.loads(raw)
    except Exception:
        logger.warning("last_tool_fetch_read key=%s found=%s malformed_json", key, found, exc_info=True)
        return None

    if not isinstance(parsed, dict) or not all(k in parsed for k in _EXPECTED_KEYS):
        logger.warning("last_tool_fetch_read key=%s found=%s missing_expected_keys", key, found)
        return None

    tool_names = parsed.get("tool_names")
    if not isinstance(tool_names, list) or not all(isinstance(name, str) for name in tool_names):
        # Guards prefix.py's `", ".join(prior_tool_fetch_names)` -- a
        # non-string element would raise there uncaught, breaking the whole
        # next turn over one malformed cache row. Validation belongs here,
        # at the one place responsible for what this function hands back.
        logger.warning("last_tool_fetch_read key=%s found=%s tool_names_invalid_shape", key, found)
        return None

    return parsed
