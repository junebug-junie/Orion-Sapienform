"""Best-effort writer for substrate_turn_referent (reverie semantic lift v1)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from sqlalchemy import text

if TYPE_CHECKING:
    from orion.schemas.harness_finalize import HarnessPostTurnClosureV1

logger = logging.getLogger("orion.substrate.runtime.turn_referent")


def _coalition_ref(correlation_id: str) -> str:
    return f"harness_closure:{correlation_id}"


def persist_turn_referent(
    closure: "HarnessPostTurnClosureV1",
    *,
    engine: Any,
    now: datetime | None = None,
) -> bool:
    if not closure.surprise_unresolved:
        return False
    excerpt_user = (closure.user_message_excerpt or "").strip()
    excerpt_stance = (closure.stance_imperative or "").strip()
    if not excerpt_user and not excerpt_stance:
        logger.info(
            "turn_referent_skip corr=%s reason=empty_excerpts",
            closure.correlation_id,
        )
        return False
    ts = now or datetime.now(timezone.utc)
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_turn_referent
                        (correlation_id, coalition_ref, user_message_excerpt,
                         stance_imperative, surprise_unresolved, thought_event_id, created_at)
                    VALUES
                        (:correlation_id, :coalition_ref, :user_message_excerpt,
                         :stance_imperative, :surprise_unresolved, :thought_event_id, :created_at)
                    ON CONFLICT (correlation_id) DO UPDATE SET
                        coalition_ref = EXCLUDED.coalition_ref,
                        user_message_excerpt = EXCLUDED.user_message_excerpt,
                        stance_imperative = EXCLUDED.stance_imperative,
                        surprise_unresolved = EXCLUDED.surprise_unresolved,
                        thought_event_id = EXCLUDED.thought_event_id,
                        created_at = EXCLUDED.created_at,
                        stored_at = now()
                    """
                ),
                {
                    "correlation_id": closure.correlation_id,
                    "coalition_ref": _coalition_ref(closure.correlation_id),
                    "user_message_excerpt": excerpt_user,
                    "stance_imperative": excerpt_stance,
                    "surprise_unresolved": True,
                    "thought_event_id": closure.thought_event_id,
                    "created_at": ts,
                },
            )
        logger.info(
            "turn_referent_upserted corr=%s coalition_ref=%s",
            closure.correlation_id,
            _coalition_ref(closure.correlation_id),
        )
        return True
    except Exception as exc:
        logger.warning("turn_referent_upsert_failed corr=%s err=%s", closure.correlation_id, exc)
        return False
