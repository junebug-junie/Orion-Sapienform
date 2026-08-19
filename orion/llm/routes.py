"""One definition of what an LLM route name is, shared by every service that sends one.

WHY THIS EXISTS
---------------
Two services normalized route overrides independently, and they drifted:

- `orion-cortex-exec` (`_resolve_llm_route_override`) accepted
  {chat, quick, metacog, quick_background, agent}.
- `orion-actions` (`_normalized_llm_route`) accepted only {chat, quick, metacog} and silently
  rewrote **everything else to "chat"**.

`services/orion-actions/.env` has said `ACTIONS_JOURNAL_LLM_ROUTE=quick_background` since the
key was introduced. `quick_background` is not in the older allow-list, so the operator's stated
intent was rewritten -- with no log line and no error -- into `chat`, which is circe's single-slot
131,072-token lane. Orion's journaling ran there for a 1,749-token median prompt because a stale
`if route in {...}` two services away disagreed about what routes exist.

`agent` had the same fate.

WHAT THIS MODULE DOES AND DOES NOT COVER
----------------------------------------
Imported by the three services that set or accept an `llm_route` override:

    services/orion-actions/app/main.py                     (_normalized_llm_route)
    services/orion-cortex-exec/app/executor.py             (_resolve_llm_route_override)
    services/orion-hub/scripts/cortex_request_builder.py   (POST /api/chat -- found by review
                                                            2026-08-18 carrying a THIRD copy,
                                                            also missing quick_background)

The follow-up slice named here on 2026-08-18 -- the *catalog* half, four more copies of
`("chat", "quick", "agent", "metacog")` that made `quick_background` invisible to every operator
surface -- was closed on 2026-08-19. Those now derive from this module:

    services/orion-llm-gateway/app/route_catalog.py        CATALOG_ROUTE_IDS
    services/orion-hub/scripts/llm_gateway_client.py       VALID_ROUTE_IDS
    scripts/smoke_llm_gateway_routes.py                    asserts every route in the order below

The Hub UI's picker (`services/orion-hub/static/js/app.js`) deliberately does NOT show every
route. It now derives from `GET /routes` and filters on the route's own `priority` field rather
than on a hardcoded name list, so a background lane is visible to operators in the catalog while
staying unpickable by a human who would only get a yielding lane's latency. That is a policy
difference expressed as a property, not another list to drift.

Two neighbouring vocabularies are NOT this axis and must not be folded in here:

    orion/schemas/context_exec.py:13   ALLOWED_CONTEXT_EXEC_LLM_PROFILES -- a narrower
        investigation-profile allow-list that RAISES rather than degrading. Deliberately
        separate; widening it is a schema decision, not a routing one.
    "brain"  -- MEMORY_GRAPH_SUGGEST_ESCALATION_ROUTE's vocabulary
        (services/orion-hub/scripts/memory_graph_suggest.py:47) is mode, not route.
        `normalize_llm_route("brain")` returns None BY DESIGN; never route that path
        through this function without translating first.

THE FALLBACK IS `None`, NOT A ROUTE
-----------------------------------
The old fallback picked `chat` -- the largest, slowest, most contended lane in the fleet -- for
any value it did not recognise. A typo therefore cost the most expensive upstream available.

An unrecognised route now returns `None`, meaning *"no override"*: the caller falls through to
its own verb-based default mapping, which is what `orion-cortex-exec` already documented as the
correct behaviour for a rejected override. Absent is not a guess.
"""
from __future__ import annotations

from typing import FrozenSet, Optional

#: Route names the executor will actually dispatch on. Anything else is not an override.
#: Keep in sync with `LLM_GATEWAY_ROUTE_TABLE_JSON` in `services/orion-llm-gateway/.env_example`.
ACCEPTED_LLM_ROUTES: FrozenSet[str] = frozenset(
    {"chat", "quick", "metacog", "quick_background", "agent"}
)

#: Historical spellings of `quick`, kept working because live config still carries them.
LLM_ROUTE_ALIASES = {
    "chat_quick": "quick",
    "quick_chat": "quick",
    "chat_kids_story": "quick",
}


# Display/catalog order. Every accepted route appears exactly once, and the assertion below
# makes that a load-bearing guarantee rather than a convention: adding a name to
# ACCEPTED_LLM_ROUTES without placing it here fails at import, in every service that imports
# this module, rather than silently omitting the new route from `GET /routes`, the Hub, and the
# smoke. That is the failure mode this whole module exists to stop, one layer up.
#
# Order is deliberate and is what operators read: interactive lanes first, then the yielding
# lane beside the lane it yields on, then the specialist.
LLM_ROUTE_DISPLAY_ORDER: tuple[str, ...] = (
    "chat",
    "quick",
    "quick_background",
    "metacog",
    "agent",
)

if set(LLM_ROUTE_DISPLAY_ORDER) != set(ACCEPTED_LLM_ROUTES) or len(
    LLM_ROUTE_DISPLAY_ORDER
) != len(ACCEPTED_LLM_ROUTES):
    raise RuntimeError(
        "LLM_ROUTE_DISPLAY_ORDER must list every accepted route exactly once; "
        f"missing={sorted(ACCEPTED_LLM_ROUTES - set(LLM_ROUTE_DISPLAY_ORDER))} "
        f"extra={sorted(set(LLM_ROUTE_DISPLAY_ORDER) - ACCEPTED_LLM_ROUTES)}"
    )


def normalize_llm_route(raw: object) -> Optional[str]:
    """Canonical route name for `raw`, or None if it is absent or unrecognised.

    None is deliberately the same answer for "nothing was asked for" and "what was asked for is
    not a route": both mean *do not override*. Callers that need to tell those apart -- to log a
    rejected override rather than silently ignoring it -- should compare against
    `ACCEPTED_LLM_ROUTES` themselves, the way `orion-cortex-exec` keeps `attempted` distinct from
    `accepted`.
    """
    route = str(raw or "").strip().lower()
    if not route:
        return None
    route = LLM_ROUTE_ALIASES.get(route, route)
    return route if route in ACCEPTED_LLM_ROUTES else None
