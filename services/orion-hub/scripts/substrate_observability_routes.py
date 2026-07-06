"""Read-only self-observability summary API for the Hub Self panel.

Aggregates the four self-observability surfaces — self-state attention schema,
attention broadcast dwell, endogenous curiosity candidates, and hub presence —
into one payload. Every section degrades to null independently (missing table,
unset POSTGRES_URI, bad JSON): the route always answers 200 so the panel can
render partial truth instead of an error page.
"""

from __future__ import annotations

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
    from sqlalchemy import create_engine

    return create_engine(uri, pool_pre_ping=True)


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


def _self_state_section(engine) -> dict[str, Any] | None:
    row = _latest_row(
        engine,
        """
        SELECT self_state_json, generated_at FROM substrate_self_state
        ORDER BY generated_at DESC LIMIT 1
        """,
    )
    if not row:
        return None
    state = _parse_json(row["self_state_json"])
    if not isinstance(state, dict):
        return None
    return {
        "attention_schema_type": state.get("attention_schema_type"),
        "attention_dwell_ticks": state.get("attention_dwell_ticks", 0),
        "attention_node_count": state.get("attention_node_count", 0),
        "hub_presence": state.get("hub_presence"),
        "overall_condition": state.get("overall_condition", "unknown"),
        "summary_labels": state.get("summary_labels", []),
        "generated_at": state.get("generated_at") or _iso(row.get("generated_at")),
    }


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
    """Recent spontaneous thoughts (reverie Phase A). Read-only; empty until the
    default-off producer runs."""
    from sqlalchemy import text

    sql = (
        "SELECT thought_id, correlation_id, created_at, salience, interpretation, "
        "thought_json FROM substrate_reverie_thought "
        "ORDER BY created_at DESC LIMIT :limit"
    )
    with engine.connect() as conn:
        rows = conn.execute(text(sql), {"limit": _REVERIE_LIMIT}).mappings().all()
    if not rows:
        return None
    thoughts = []
    for row in rows:
        payload = _parse_json(row.get("thought_json")) or {}
        coalition = payload.get("coalition") or {}
        thoughts.append(
            {
                "thought_id": row.get("thought_id"),
                "salience": row.get("salience"),
                "interpretation": row.get("interpretation"),
                "attended_node_ids": coalition.get("attended_node_ids", []),
                "selected_open_loop_id": coalition.get("selected_open_loop_id"),
                "created_at": _iso(row.get("created_at")),
            }
        )
    return {"count": len(thoughts), "recent": thoughts}


def _compaction_queue_section(engine) -> dict[str, Any] | None:
    """Pending compaction requests (reverie Phase E). Queue only — applied by
    nothing. Empty until the default-off producer runs."""
    from sqlalchemy import text

    sql = (
        "SELECT request_id, theme, op_hint, reason, origin_chain_id, created_at, "
        "consumed_at FROM dream_compaction_request_queue "
        "WHERE consumed_at IS NULL ORDER BY created_at DESC LIMIT :limit"
    )
    with engine.connect() as conn:
        rows = conn.execute(text(sql), {"limit": _REVERIE_LIMIT}).mappings().all()
    if not rows:
        return None
    pending = [
        {
            "request_id": r.get("request_id"),
            "theme": r.get("theme"),
            "op_hint": r.get("op_hint"),
            "reason": r.get("reason"),
            "origin_chain_id": r.get("origin_chain_id"),
            "created_at": _iso(r.get("created_at")),
        }
        for r in rows
    ]
    return {"pending_count": len(pending), "pending": pending}


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


@router.get("/summary")
async def observability_summary() -> dict[str, Any]:
    engine = None
    try:
        engine = _engine()
    except Exception:
        logger.debug("observability_engine_init_failed", exc_info=True)

    sections: dict[str, Any] = {
        "self_state": None,
        "attention_broadcast": None,
        "curiosity": None,
        "reverie": None,
        "compaction_queue": None,
    }
    if engine is not None:
        for name, loader in (
            ("self_state", _self_state_section),
            ("attention_broadcast", _attention_broadcast_section),
            ("curiosity", _curiosity_section),
            ("reverie", _reverie_section),
            ("compaction_queue", _compaction_queue_section),
        ):
            try:
                sections[name] = loader(engine)
            except Exception:
                logger.debug("observability_section_failed section=%s", name, exc_info=True)

    hub_presence = None
    try:
        hub_presence = _hub_presence_section(engine)
    except Exception:
        logger.debug("observability_section_failed section=hub_presence", exc_info=True)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **sections,
        "hub_presence": hub_presence,
    }
