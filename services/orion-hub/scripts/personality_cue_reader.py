"""Hub's read of Orion's personality frame, rendered as the read-back cue (personality graph, spec rev 2).

What Orion keeps coming back to, in Orion's own words: a RECORD, not an instruction. The text is
the shared fixed template ``orion.personality.cue.project_personality_cue`` (no model, no tone or
style directives), so the curiosity kickoff and endogenous outreach see the same self as the chat
stance step and reverie.

One GET on orion-durable-runs ``{base}/personality/frame`` -- the same base URL Hub already uses for
durable-runs (``HUB_CURIOSITY_DURABLE_RUNS_URL``). Fail-open and bounded: off, no URL, unreachable,
slow, non-200, bad JSON, invalid/stale/empty frame -- every case returns a cue with ``text == ""``
and a ``skip_reason``. Never raises, never takes longer than ``timeout_sec``.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from orion.personality.cue import project_personality_cue
from orion.schemas.personality import PersonalityCueV1

logger = logging.getLogger("orion-hub.personality_cue")

# Same deadline as the kickoff's Temporal Self read (curiosity_investigation.TEMPORAL_SELF_FACT_TIMEOUT_SEC):
# the curiosity kickoff runs both reads concurrently, so the slower one bounds the wait.
DEFAULT_TIMEOUT_SEC = 1.5
FRAME_PATH = "/personality/frame"


def _skip(now: datetime, reason: str) -> PersonalityCueV1:
    return PersonalityCueV1(rendered_at=now, skip_reason=reason)  # type: ignore[arg-type]


async def _fetch_frame(root: str, timeout_sec: float) -> Optional[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=timeout_sec) as client:
        resp = await client.get(f"{root}{FRAME_PATH}")
    if resp.status_code != 200:
        return None
    body = resp.json()
    return body if isinstance(body, dict) else None


async def read_personality_cue(
    base_url: Optional[str],
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    *,
    now: Optional[datetime] = None,
    enabled: bool = True,
) -> PersonalityCueV1:
    """Always returns a cue; ``text`` is "" when nothing may be shown. Never raises."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    if not enabled:
        return _skip(now, "disabled")
    root = str(base_url or "").strip().rstrip("/")
    if not root:
        return _skip(now, "fetch_failed")
    try:
        # httpx's timeout is per phase; bound the whole fetch.
        frame = await asyncio.wait_for(_fetch_frame(root, timeout_sec), timeout=timeout_sec)
    except Exception as exc:  # noqa: BLE001 -- a reader must never fail its caller
        logger.info("personality_frame_fetch_failed err=%s: %s", type(exc).__name__, str(exc)[:200])
        return _skip(now, "fetch_failed")
    try:
        return project_personality_cue(frame, now=now)
    except Exception as exc:  # noqa: BLE001 -- the renderer promises never to raise; named if it does
        logger.warning("personality_cue_render_failed err=%s: %s", type(exc).__name__, exc)
        return _skip(now, "render_failed")


def cue_status(cue: PersonalityCueV1) -> str:
    """One log fragment: ``rendered=<bool> ids=<a,b|-> skip=<reason|->``."""
    return (
        f"rendered={bool(cue.text)} ids={','.join(cue.disposition_ids) or '-'} "
        f"skip={cue.skip_reason or '-'}"
    )
