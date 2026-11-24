"""Read-only self-observability summary API for the Hub Self panel.

Aggregates the four self-observability surfaces — self-state attention schema,
attention broadcast dwell, endogenous curiosity candidates, and hub presence —
into one payload. Every section degrades to null independently (missing table,
unset POSTGRES_URI, bad JSON): the route always answers 200 so the panel can
render partial truth instead of an error page.
"""

from __future__ import annotations

import asyncio

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter

from scripts.hub_presence import presence_snapshot

logger = logging.getLogger("orion-hub.substrate_observability")

router = APIRouter(prefix="/api/substrate/observability", tags=["substrate-observability"])

_CURIOSITY_SIGNALS_LIMIT = 5
# Same staleness gate the self-state runtime applies to substrate_hub_presence:
# past this age the persisted connection_health is a lie (computed at write time).
_PRESENCE_MAX_AGE_SEC = 600.0


def _engine():
    uri = os.getenv("POSTGRES_URI", "").strip()
    if not uri:
        return None
    from .pg_engine import module_engine

    return module_engine(__name__, uri)


def _latest_row(engine, sql: str) -> dict[str, Any] | None:
    from sqlalchemy import text

    with engine.connect() as conn:
        row = conn.execute(text(sql)).mappings().first()
    return dict(row) if row else None


def _parse_json(payload: Any) -> Any:
    if isinstance(payload, str):
        return json.loads(payload)
    return payload


def _iso(value: Any) -> str | None:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return str(value) if value else None


# _self_state_section removed 2026-07-22, SelfStateV1 burn
# (docs/superpowers/specs/2026-07-22-self-state-phi-endo-origination-burn-
# spec.md): substrate_self_state has no producer -- left in place this panel
# would show permanently stale data (not authorized to drop the table itself
# per CLAUDE.md sec 13, so old rows persist unpruned rather than going empty).


def _attention_broadcast_section(engine) -> dict[str, Any] | None:
    row = _latest_row(
        engine,
        """
        SELECT projection_json, generated_at FROM substrate_attention_broadcast_projection
        ORDER BY generated_at DESC LIMIT 1
        """,
    )
    if not row:
        return None
    projection = _parse_json(row["projection_json"])
    if not isinstance(projection, dict):
        return None
    return {
        "selected_description": projection.get("selected_description"),
        "attended_node_ids": projection.get("attended_node_ids", []),
        "dwell_ticks": projection.get("dwell_ticks", 0),
        "coalition_stability_score": projection.get("coalition_stability_score", 0.0),
        "generated_at": projection.get("generated_at") or _iso(row.get("generated_at")),
    }


_REVERIE_LIMIT = 5


def _reverie_section(engine) -> dict[str, Any] | None:
    """Recent reverie questions (reveries that ask, 2026-11-24): what each reverie started from and the
    question it left, or why none, or why code refused it. Read-only; None when there are none yet."""
    from sqlalchemy import text

    from orion.reverie.questions import is_accepted

    sql = (
        "SELECT question_id, seed_kind, question, none_reason, refused_reason, routed_to, created_at "
        "FROM reverie_question ORDER BY created_at DESC LIMIT :limit"
    )
    with engine.connect() as conn:
        rows = conn.execute(text(sql), {"limit": _REVERIE_LIMIT}).mappings().all()
    if not rows:
        return None
    recent = []
    for row in rows:
        r = dict(row)
        recent.append(
            {
                "question_id": r.get("question_id"),
                "seed_kind": r.get("seed_kind"),
                "question": r.get("question") if is_accepted(r) else None,
                "none_reason": r.get("none_reason"),
                "refused_reason": r.get("refused_reason"),
                "routed_to": r.get("routed_to"),
                "created_at": _iso(r.get("created_at")),
            }
        )
    return {"count": len(recent), "recent": recent}


def _curiosity_section(engine) -> dict[str, Any] | None:
    row = _latest_row(
        engine,
        """
        SELECT candidates_json, generated_at FROM substrate_endogenous_curiosity_candidates
        ORDER BY generated_at DESC LIMIT 1
        """,
    )
    if not row:
        return None
    candidates = _parse_json(row["candidates_json"])
    if not isinstance(candidates, list):
        return None
    gaps = [c for c in candidates if isinstance(c, dict)]
    signals = sorted(
        gaps,
        key=lambda c: float(c.get("signal_strength") or 0.0),
        reverse=True,
    )[:_CURIOSITY_SIGNALS_LIMIT]
    return {
        "gap_count": len(gaps),
        "signals": [
            {
                "signal_type": sig.get("signal_type"),
                "signal_strength": sig.get("signal_strength"),
                "evidence_summary": sig.get("evidence_summary"),
            }
            for sig in signals
        ],
        "generated_at": _iso(row.get("generated_at")),
    }


def _presence_row_fresh(generated_at: Any) -> bool:
    if not isinstance(generated_at, datetime):
        return False
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - generated_at).total_seconds()
    return age <= _PRESENCE_MAX_AGE_SEC


def _hub_presence_section(engine) -> dict[str, Any] | None:
    # Prefer the persisted row (matches what self-state consumed); fall back
    # to the in-process snapshot when the table is absent, empty, or stale.
    if engine is not None:
        try:
            row = _latest_row(
                engine,
                """
                SELECT presence_json, generated_at FROM substrate_hub_presence
                WHERE presence_id = 'hub' LIMIT 1
                """,
            )
            if row and _presence_row_fresh(row.get("generated_at")):
                presence = _parse_json(row["presence_json"])
                if isinstance(presence, dict) and presence:
                    presence["generated_at"] = _iso(row.get("generated_at"))
                    return presence
        except Exception:
            logger.debug("hub_presence_row_load_failed", exc_info=True)
    snapshot = presence_snapshot()
    if snapshot is None:
        return None
    snapshot["generated_at"] = datetime.now(timezone.utc).isoformat()
    return snapshot


def _load_sections(engine) -> dict[str, Any]:
    """Every observability section, on one worker thread.

    Each loader failing independently is deliberate and preserved: a section
    that errors reports None rather than taking the whole panel down.
    """
    out: dict[str, Any] = {}
    for name, loader in (
        ("attention_broadcast", _attention_broadcast_section),
        ("curiosity", _curiosity_section),
        ("reverie", _reverie_section),
    ):
        try:
            out[name] = loader(engine)
        except Exception:
            logger.debug("observability_section_failed section=%s", name, exc_info=True)
            out[name] = None
    return out


@router.get("/summary")
async def observability_summary() -> dict[str, Any]:
    engine = None
    try:
        engine = await asyncio.to_thread(_engine)
    except Exception:
        logger.debug("observability_engine_init_failed", exc_info=True)

    sections: dict[str, Any] = {
        "attention_broadcast": None,
        "curiosity": None,
        "reverie": None,
    }
    if engine is not None:
        # All sections in ONE dispatch, not one each: each opens a real Postgres
        # connection, and running them inline held the event loop for the sum
        # of every round trip. The previous patch threaded `_engine()` (which
        # is lazy and never blocks) and left these -- caught in review, and
        # then reproduced by the checker only after it learned to follow a
        # helper reached through a loop variable rather than by name.
        sections.update(await asyncio.to_thread(_load_sections, engine))

    hub_presence = None
    try:
        hub_presence = await asyncio.to_thread(_hub_presence_section, engine)
    except Exception:
        logger.debug("observability_section_failed section=hub_presence", exc_info=True)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **sections,
        "hub_presence": hub_presence,
    }
