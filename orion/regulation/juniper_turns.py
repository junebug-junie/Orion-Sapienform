"""What counts as "Juniper spoke", in one place, for every reader of Juniper's quiet.

Live 2026-10-10 (chat_history_log, every row ever): each row with a prompt was Juniper-initiated
(her typed turns under hub_orion / hub_ws / hub_http / hub, her "Run your dream cycle." button,
her collapse-mirror entries), and each row WITHOUT a prompt (236 rows, NULL source) was Orion's
own outreach, stamped ``client_meta.unsolicited = true``. Before this module the dream's idle
gate read ``max(created_at)`` over all rows, so Orion speaking reset Orion's own idle clock.

The rule is structural, not a list of source labels: a turn is Juniper's when it carries a
prompt and is not marked unsolicited. A new Hub surface needs no edit here.

Used by orion-dream's idle gate (repair 1, spec R2) and by arousal E1 (spec R3).
"""

from __future__ import annotations

from typing import Any, Mapping

# SQL fragment over chat_history_log (prompt text, client_meta jsonb).
JUNIPER_TURN_PREDICATE = (
    "NULLIF(btrim(prompt), '') IS NOT NULL "
    "AND COALESCE(client_meta->>'unsolicited', 'false') <> 'true'"
)

# chat_history_log.created_at is `timestamp without time zone` defaulted by the server's now(),
# so compare against LOCALTIMESTAMP on the same server clock. NULL: no Juniper turn ever.
JUNIPER_IDLE_MINUTES_SQL = (
    "SELECT EXTRACT(EPOCH FROM (LOCALTIMESTAMP - max(created_at))) / 60.0 AS idle "
    f"FROM chat_history_log WHERE {JUNIPER_TURN_PREDICATE}"
)


def is_juniper_turn(payload: Mapping[str, Any] | None) -> bool:
    """The same rule for a ``chat.history.turn`` bus payload (``ChatHistoryTurnV1`` dict)."""
    if not payload:
        return False
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return False
    meta = payload.get("client_meta") or {}
    unsolicited = meta.get("unsolicited") if isinstance(meta, Mapping) else None
    return not (unsolicited is True or str(unsolicited).lower() == "true")


__all__ = ["JUNIPER_IDLE_MINUTES_SQL", "JUNIPER_TURN_PREDICATE", "is_juniper_turn"]
