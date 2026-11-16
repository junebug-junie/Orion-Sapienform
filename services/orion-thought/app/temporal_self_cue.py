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

Orion's now for reveries (integration patch 3, docs/superpowers/specs/2026-11-15-orion-integration-and-a-life-
design.md A4; this module is reverie's registered ``reads_now`` in config/integration/organs.yaml). ``reverie_now``
reads ``GET /temporal-self/now?organ=reverie`` ONCE per reverie chain, at the first step that reaches the prompt
(later steps of the same chain reuse it; a standalone tick reads its own), with no ``since``, so it moves reverie's
cursor. ``record_reverie_cites`` posts the shown items a PUBLISHED thought named in its ``cites``. Both never raise,
are bounded by ``ORION_REVERIE_NOW_TIMEOUT_SEC``, and a failure means a reverie without the section.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from orion.integration import now_client
from orion.schemas.temporal_self import NOW_MAX_CITES, TemporalSelfStanceCueV1
from orion.temporal_self.now import EXPERIENCE_PREFIX, cites_in_text, normalize_cite, now_lines
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

    Shared by the stance cue and reverie's recent-arcs block (``app/reverie.py``)."""
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


# --- Orion's now (integration patch 3) -----------------------------------------------------------------------

REVERIE_ORGAN = "reverie"
NOW_LABEL_SHOWN = 140


def _now_http() -> dict[str, Any]:
    """Seam for tests: ``{"get": fn}`` / ``{"post": fn}`` to replace the client's own HTTP. Empty in production."""
    return {}


# The one cached read: (chain_id, NowRead). A reverie chain is one run of the organ: its steps share one now, so the
# cursor moves once per chain and every step can cite what the chain was shown. One entry: chains run one at a time.
_chain_now: tuple[str, now_client.NowRead] | None = None


async def reverie_now(chain_id: str | None, correlation_id: str) -> now_client.NowRead | None:
    """The now for this reverie step, or None when the read is switched off. Never raises."""
    global _chain_now
    if not settings.reverie_now_enabled:
        return None
    if chain_id and _chain_now is not None and _chain_now[0] == chain_id:
        return _chain_now[1]
    run_ref = f"reverie_chain:{chain_id}" if chain_id else f"reverie_tick:{correlation_id}"
    try:
        got = await now_client.aread_now(settings.temporal_self_frame_url, REVERIE_ORGAN, run_ref=run_ref,
                                         timeout_sec=settings.reverie_now_timeout_sec,
                                         get=_now_http().get("get"))
    except Exception as exc:  # noqa: BLE001 -- it runs inside the tick's gather: a raise here would drop the tick
        got = now_client.NowRead(None, f"{type(exc).__name__}: {str(exc)[:120]}")
    if got.now is None:
        logger.info("reverie_now_unread corr=%s ref=%s reason=%s", correlation_id, run_ref, got.error)
    else:
        logger.info("reverie_now_read corr=%s ref=%s shown_id=%s items=%d", correlation_id, run_ref,
                    got.now.shown_id, len(got.now.items))
    if chain_id:
        _chain_now = (chain_id, got)
    return got


def render_reverie_now(got: now_client.NowRead | None) -> str:
    """The prompt's lines (most unusual first, each with its [experience:<id>]); "" when there is nothing."""
    if got is None or got.now is None:
        return ""
    return "\n".join(now_lines(got.now, label_cap=NOW_LABEL_SHOWN))


def reverie_cites(raw_cites: Any, evidence_refs: list[str], shown_ids: list[str]) -> tuple[list[str], list[str]]:
    """(cites, evidence_refs without experience ids). An ``experience:`` ref is never grounding: one the model put
    in evidence_refs is moved to the cites. Cites are only ids the step was shown, in order, at most 3."""
    named: list[str] = []
    if isinstance(raw_cites, str):
        raw_cites = [raw_cites]
    if isinstance(raw_cites, list):
        named += [str(c) for c in raw_cites if isinstance(c, (str, int))]
    kept_refs = []
    for ref in evidence_refs:
        bare = ref.strip().strip("[]")
        if bare.startswith(EXPERIENCE_PREFIX):
            named.append(bare)
        else:
            kept_refs.append(ref)
    shown = set(shown_ids)
    cites: list[str] = []
    for c in named:
        eid = normalize_cite(c)
        if eid in shown and eid not in cites:
            cites.append(eid)
    # Also accept a cite written as text ("... [experience:<id>] ...") inside one cites entry.
    for eid in cites_in_text("\n".join(named), shown_ids):
        if eid not in cites:
            cites.append(eid)
    return cites[:NOW_MAX_CITES], kept_refs


async def record_reverie_cites(got: now_client.NowRead | None, cites: list[str], thought_id: str,
                               correlation_id: str) -> None:
    """Post a published thought's cites against the shown row its chain read. Never raises."""
    try:
        if got is None or got.now is None or not cites:
            return
        result, err = await now_client.arecord_use(
            settings.temporal_self_frame_url, got.now.shown_id, cites, f"reverie_thought:{thought_id}",
            timeout_sec=settings.reverie_now_timeout_sec, post=_now_http().get("post"))
        logger.info("reverie_now_cites corr=%s thought=%s shown_id=%s cited=%d stored=%d rejected=%d err=%s",
                    correlation_id, thought_id, got.now.shown_id, len(cites), len(result.uses) if result else 0,
                    (len(result.rejected_not_shown) + len(result.rejected_over_cap)) if result else 0, err or "-")
    except Exception as exc:  # noqa: BLE001 -- a reverie never fails on its cites
        logger.warning("reverie_now_cites_failed corr=%s err=%s", correlation_id, type(exc).__name__)
