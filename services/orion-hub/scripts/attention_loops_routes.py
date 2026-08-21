"""Operator Pending Attention API — cognitive-loop rows + Resolve/Dismiss.

Flag-gated on ORION_ATTENTION_PENDING_CARDS_ENABLED. Human-only closure: Juniper
Resolves/Dismisses; each close emits AttentionLoopOutcomeV1 (the label), persists
it, and suppresses the loop so it exits the coalition (resolves, not pauses).
"""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from scripts.attention_loops_store import (
    build_loop_outcome,
    build_pending_card,
    card_kind_for_scope,
    latest_salience_for_theme,
    latest_scope_for_theme,
    load_pending_loops,
    persist_loop_outcome,
    suppress_loop,
)

logger = logging.getLogger("orion-hub.attention_loops_api")

router = APIRouter(prefix="/api/attention/loops", tags=["attention-loops"])

_TRUTHY = {"1", "true", "yes", "on"}


def _cards_enabled() -> bool:
    return str(os.getenv("ORION_ATTENTION_PENDING_CARDS_ENABLED", "false")).strip().lower() in _TRUTHY


class CloseRequest(BaseModel):
    note: str = ""


def publish_loop_outcome(outcome) -> None:
    """Best-effort publish of the label event; swallow any bus failure."""
    try:
        from scripts.bus_publish import publish_attention_loop_outcome  # thin helper (Task 15)

        publish_attention_loop_outcome(outcome)
    except Exception as exc:
        logger.warning("publish loop outcome failed id=%s err=%s", outcome.outcome_id, exc)


@router.get("")
def list_loops(limit: int = 50):
    if not _cards_enabled():
        return []
    cards = []
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    for loop, first_seen, recurrence, narrative, scope in load_pending_loops()[:limit]:
        card = build_pending_card(
            loop, first_seen=first_seen, recurrence_count=recurrence,
            narrative=narrative, now=now, scope=scope,
        )
        cards.append(card.model_dump(mode="json"))
    return cards


def _close(loop_id: str, verdict: str, note: str):
    if not _cards_enabled():
        raise HTTPException(status_code=404, detail="pending attention cards disabled")
    if card_kind_for_scope(latest_scope_for_theme(loop_id)) == "chronic_pressure":
        # Reverie/substrate-broadcast loops are re-selected every tick by design --
        # a human Resolve/Dismiss here would falsely mark still-live system
        # pressure as closed. The Hub UI no longer offers these buttons for a
        # chronic_pressure card; this is defense in depth against a stale client
        # or a direct API call.
        raise HTTPException(
            status_code=409,
            detail="this loop is sustained system pressure, not a resolvable decision -- it cannot be resolved/dismissed",
        )
    salience, features = latest_salience_for_theme(loop_id)
    outcome = build_loop_outcome(
        loop_id=loop_id, theme_key=loop_id, verdict=verdict, actor="juniper",
        note=note, salience_at_close=salience, features_at_close=features,
    )
    persist_loop_outcome(outcome)
    suppress_loop(loop_id)
    publish_loop_outcome(outcome)
    return {"status": "ok", "outcome_id": outcome.outcome_id, "verdict": verdict}


@router.post("/{loop_id}/resolve")
def resolve_loop(loop_id: str, payload: CloseRequest):
    return _close(loop_id, "resolved", payload.note)


@router.post("/{loop_id}/dismiss")
def dismiss_loop(loop_id: str, payload: CloseRequest):
    return _close(loop_id, "dismissed", payload.note)
