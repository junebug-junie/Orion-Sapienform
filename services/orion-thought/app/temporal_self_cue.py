"""Fetch Orion's current day frame and render the stance cue (Temporal Self patch 4).

The unified turn's stance step (``stance_react``) is the only chat path that gets this cue
(Juniper, 2026-10-11: the production chat path). Read over HTTP from orion-durable-runs'
existing ``GET /temporal-self/frame`` -- the chronicle's own read route, the same seam the Hub
already uses for durable runs -- rather than a direct DB read: orion-thought holds no DSN for the
chronicle tables, and the route returns exactly the validated ``TemporalSelfFrameV1``.

Fail-open and bounded: a short timeout (``TEMPORAL_SELF_CUE_FETCH_TIMEOUT_SEC``); any failure,
a frame for another day, or one older than ``TEMPORAL_SELF_CUE_MAX_AGE_SEC`` renders nothing
(``text == ""``) and logs ``temporal_self_cue_skipped reason=...`` with the turn's correlation id.
The fetch runs concurrently with Mind enrichment, so it adds no wait to the turn unless Mind is off.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

import httpx
from pydantic import ValidationError

from orion.schemas.temporal_self import NOW_MAX_CITES, TemporalNowV1, TemporalSelfStanceCueV1
from orion.temporal_self.stance_cue import project_stance_cue

from .settings import settings

logger = logging.getLogger("orion-thought.temporal_self_cue")


def _transport() -> httpx.AsyncBaseTransport | None:
    """Seam for tests to inject an httpx.MockTransport. None in production."""
    return None


async def fetch_frame(base_url: str, timeout_sec: float) -> dict[str, Any] | None:
    try:
        async with httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=httpx.Timeout(timeout_sec), transport=_transport(),
        ) as client:
            resp = await client.get("/temporal-self/frame")
    except httpx.HTTPError as exc:
        logger.debug("temporal_self_frame_fetch_failed err=%s: %s", type(exc).__name__, exc)
        return None
    if resp.status_code != 200:
        return None
    try:
        body = resp.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


async def fetch_frame_bounded(timeout_sec: float, correlation_id: str = "") -> dict[str, Any] | None:
    """The current frame, or None. Never raises, never takes longer than ``timeout_sec``.

    Used by the stance cue."""
    try:
        # httpx's timeout is per phase (connect, each read; DNS not covered): bound the whole fetch.
        return await asyncio.wait_for(fetch_frame(settings.temporal_self_frame_url, timeout_sec), timeout_sec)
    except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001 -- a reader must never fail its caller
        logger.debug("temporal_self_frame_fetch_failed corr=%s err=%s", correlation_id, type(exc).__name__)
        return None


async def build_stance_cue(correlation_id: str, *, now: datetime | None = None) -> TemporalSelfStanceCueV1:
    """Always returns a cue; ``text`` is "" when nothing may be shown. Never raises."""
    now = now or datetime.now(timezone.utc)
    if not settings.temporal_self_stance_cue_enabled:
        return TemporalSelfStanceCueV1(rendered_at=now, skip_reason="disabled")
    started = time.perf_counter()
    frame = await fetch_frame_bounded(settings.temporal_self_cue_fetch_timeout_sec, correlation_id)
    try:
        cue = project_stance_cue(frame, now, max_age_sec=settings.temporal_self_cue_max_age_sec)
    except Exception as exc:  # noqa: BLE001 -- a render bug is named as one, never as a network problem
        logger.warning("temporal_self_cue_render_failed corr=%s err=%s: %s", correlation_id, type(exc).__name__, exc)
        cue = TemporalSelfStanceCueV1(rendered_at=now, skip_reason="render_failed")
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
    if cue.text:
        logger.info("temporal_self_cue corr=%s day=%s chars=%s arcs=%s dropped=%s fetch_ms=%s",
                    correlation_id, cue.day_id, len(cue.text), len(cue.arc_ids), cue.clauses_dropped, elapsed_ms)
    else:
        logger.info("temporal_self_cue_skipped corr=%s reason=%s day=%s fetch_ms=%s",
                    correlation_id, cue.skip_reason, cue.day_id, elapsed_ms)
    return cue


# --- Orion's now for reverie (integration patch 3) ----------------------------------------------
# Spec: docs/superpowers/specs/2026-11-15-orion-integration-and-a-life-design.md, Part A (A3, A4).
# Reverie reads GET /temporal-self/now?organ=reverie (the few most salient experiences OTHER organs
# recorded since reverie last looked; never its own) instead of the day frame's closed arcs, cites
# what it used, and posts the cites to POST /temporal-self/now/use at the question's close. The
# server checks every cite against the experience_shown row named by shown_id. The now off (404) or
# unreachable -> the reverie runs on its seed alone, logged reverie_now_unavailable.

NOW_ROUTE = "/temporal-self/now"
NOW_USE_ROUTE = "/temporal-self/now/use"
REVERIE_ORGAN = "reverie"
NOW_LINE_MAX = 260  # one rendered line: id + label (<= 200) + organ and age


@dataclass(frozen=True)
class NowRead:
    """One read of the now. ``now`` is set only when ``reason == "ok"``; any other reason means the
    caller falls back to its old input (``not_found`` = TEMPORAL_SELF_NOW_ENABLED=false on the server)."""

    reason: str
    now: Optional[TemporalNowV1] = None

    @property
    def ok(self) -> bool:
        return self.now is not None


async def _get_now(base_url: str, timeout_sec: float, organ: str, run_ref: Optional[str]) -> NowRead:
    params = {"organ": organ}
    if run_ref:
        params["run_ref"] = run_ref[:300]
    try:
        async with httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=httpx.Timeout(timeout_sec), transport=_transport(),
        ) as client:
            # No since=: the server then starts from when this organ last looked (its cursor) and moves
            # the cursor. An explicit since= is an inspection read that never moves it (durable-runs).
            resp = await client.get(NOW_ROUTE, params=params)
    except httpx.HTTPError as exc:
        return NowRead(reason=f"unreachable:{type(exc).__name__}")
    if resp.status_code == 404:
        return NowRead(reason="not_found")
    if resp.status_code != 200:
        return NowRead(reason=f"http_{resp.status_code}")
    try:
        return NowRead(reason="ok", now=TemporalNowV1.model_validate(resp.json()))
    except (ValueError, ValidationError):
        return NowRead(reason="invalid")


async def fetch_now_bounded(
    timeout_sec: float, *, organ: str = REVERIE_ORGAN, run_ref: Optional[str] = None, correlation_id: str = "",
) -> NowRead:
    """Orion's now for ``organ``, or a reason it is not available. Never raises, never takes longer
    than ``timeout_sec``."""
    try:
        return await asyncio.wait_for(
            _get_now(settings.temporal_self_frame_url, timeout_sec, organ, run_ref), timeout_sec)
    except asyncio.TimeoutError:
        return NowRead(reason="timeout")
    except Exception as exc:  # noqa: BLE001 -- a reader must never fail its caller
        logger.debug("temporal_self_now_fetch_failed corr=%s err=%s", correlation_id, type(exc).__name__)
        return NowRead(reason=f"error:{type(exc).__name__}")


def _age(occurred_at: datetime, as_of: datetime) -> str:
    minutes = max(0, int((as_of - occurred_at).total_seconds() // 60))
    if minutes < 60:
        return f"{minutes} min ago"
    hours = minutes // 60
    return f"{hours} h ago" if hours < 48 else f"{hours // 24} days ago"


def render_now_block(now: TemporalNowV1, *, as_of: Optional[datetime] = None) -> str:
    """One line per experience, in the server's order (highest salience first): the cite token, the
    code-written label, the organ, how long ago. "" when the now holds nothing. No LLM."""
    as_of = as_of or datetime.now(timezone.utc)
    lines = []
    for item in now.items:
        line = f"- [experience:{item.event_id}] {item.label} ({item.organ}, {_age(item.occurred_at, as_of)})"
        lines.append(line if len(line) <= NOW_LINE_MAX else line[: NOW_LINE_MAX - 1] + "…")
    return "\n".join(lines)


def shown_cites(raw_cites: Iterable[Any], shown_event_ids: Iterable[str], *, cap: int = NOW_MAX_CITES) -> list[str]:
    """The reader's cites, normalized (bare id or ``experience:<id>``, optionally bracketed), kept only
    when the now showed them, deduplicated in order, at most ``cap``. The server checks again."""
    from orion.temporal_self.now import normalize_cite

    shown = set(shown_event_ids)
    out: list[str] = []
    if cap <= 0:
        return out
    for raw in raw_cites:
        eid = normalize_cite(raw)
        if eid and eid in shown and eid not in out:
            out.append(eid)
            if len(out) >= cap:
                break
    return out


async def post_now_use(
    shown_id: str, cites: list[str], citing_ref: str, timeout_sec: float, *, correlation_id: str = "",
) -> Optional[dict[str, Any]]:
    """POST the cites for one shown now. Returns the server's ``ExperienceUseResultV1`` as a dict, or
    None on any failure (logged). Never raises, never takes longer than ``timeout_sec``."""

    async def _post() -> Optional[dict[str, Any]]:
        async with httpx.AsyncClient(
            base_url=settings.temporal_self_frame_url.rstrip("/"), timeout=httpx.Timeout(timeout_sec),
            transport=_transport(),
        ) as client:
            resp = await client.post(NOW_USE_ROUTE, json={
                "shown_id": shown_id, "citing_ref": citing_ref[:300], "cites": list(cites)})
        if resp.status_code != 200:
            logger.warning("temporal_self_now_use_failed corr=%s shown_id=%s status=%s",
                           correlation_id, shown_id, resp.status_code)
            return None
        body = resp.json()
        return body if isinstance(body, dict) else None

    try:
        return await asyncio.wait_for(_post(), timeout_sec)
    except Exception as exc:  # noqa: BLE001 -- recording a use must never fail the reader
        logger.warning("temporal_self_now_use_failed corr=%s shown_id=%s err=%s",
                       correlation_id, shown_id, type(exc).__name__)
        return None
