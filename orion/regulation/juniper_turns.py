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


# When a Juniper turn STARTED, for readers that ask "did she speak by time T". chat_history_log
# is written only after Orion finishes answering (live 2026-10-16: 1 to 41 minutes after she sent
# the turn), and never for a turn that errored, so its created_at is the wrong clock for that
# question. The hub's unified turn records its start first, before any work that can fail: the
# cockpit ``ingress`` hop (orion/hub/turn_orchestrator.py -> begin_cockpit_timeline, seq 0),
# persisted by orion-sql-writer into cockpit_turn_sighting (ts = the hop's own UTC timestamp).
# Every unified turn has one, Orion's own internal turns too (outreach, reading, curiosity), so
# Juniper's are the ones a hub client surface sent: only the browser client sets
# ``surface_context`` on a turn (static/js/app.js typed + spoken, thought-process.js), and the
# turn orchestrator copies it into the ingress raw only when present. Live 2026-10-16, 10 days:
# 37 ingress hops carried it and all 37 were Juniper's logged turns; none of the 1,100+
# internal-turn hops did. scripts/trace_hub_skill_runner_e2e.py (an operator smoke over the hub
# WebSocket) marks itself ``surface: "operator_review"`` and is excluded (its logged row still counts under
# JUNIPER_TURN_PREDICATE). SQL fragment over cockpit_turn_sighting
# (unqualified stage/seq/raw columns).
# 2026-11-18 (#3602 review): an operator smoke marks itself ``surface: "operator_review"`` and is not Juniper; an empty
# ``{}`` is not a surface (the Python rule below already said so; the SQL now agrees). Every live browser hop
# carries ``surface: "hub_desktop"`` (42 of 42 logged turns, 11-08..11-18).
SMOKE_SURFACE = "operator_review"  # a SurfaceContextV1.surface value (orion/schemas/situation.py)
JUNIPER_INGRESS_PREDICATE = (
    "stage = 'ingress' AND seq = 0 AND raw->'surface_context' IS NOT NULL "
    "AND raw->'surface_context' <> '{}'::jsonb "
    f"AND COALESCE(raw->'surface_context'->>'surface', '') <> '{SMOKE_SURFACE}'"
)


def is_browser_surface_turn(payload: Mapping[str, Any] | None) -> bool:
    """``JUNIPER_INGRESS_PREDICATE`` for a Hub turn payload, before the ingress hop is written.

    True when the payload carries a non-empty ``surface_context`` dict: the same field, under the same
    test, that ``turn_orchestrator._ingress_hop`` copies into the ingress raw. A coding session's message
    posted to ``/api/chat`` (live 2026-11-15: two of them, persisted as ``hub_orion`` in Juniper's own
    session) and a ``no_write`` eval turn carry none."""
    if not payload:
        return False
    surface = payload.get("surface_context")
    return isinstance(surface, Mapping) and bool(surface) and surface.get("surface") != SMOKE_SURFACE


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


__all__ = ["JUNIPER_IDLE_MINUTES_SQL", "JUNIPER_INGRESS_PREDICATE", "SMOKE_SURFACE", "JUNIPER_TURN_PREDICATE", "is_browser_surface_turn",
           "is_juniper_turn"]
