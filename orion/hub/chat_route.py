from __future__ import annotations

"""Canonical ``chat_route`` tags for the branches inside
``services/orion-hub/scripts/api_routes.py:handle_chat_request``.

The four branches are today distinguished only by which response fields
happen to be populated. This tag is stamped once at the point each branch is
taken and threaded into the response payload and, for the unified-turn
route, into the persisted chat_turn envelope's ``spark_meta`` -- so a later
correlation-keyed trace lookup can tell "no stance row because this route
never produces one" from "no stance row because persistence broke."
"""

CHAT_ROUTE_CLASSIC_PLANRUNNER = "classic_planrunner"
CHAT_ROUTE_UNIFIED_TURN_HARNESS = "unified_turn_harness"
CHAT_ROUTE_AGENT_CLAUDE = "agent_claude"
CHAT_ROUTE_CONTEXT_EXEC_AGENT = "context_exec_agent"

ALL_CHAT_ROUTES = (
    CHAT_ROUTE_CLASSIC_PLANRUNNER,
    CHAT_ROUTE_UNIFIED_TURN_HARNESS,
    CHAT_ROUTE_AGENT_CLAUDE,
    CHAT_ROUTE_CONTEXT_EXEC_AGENT,
)
