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
    services/orion-hub/scripts/llm_gateway_client.py       VALID_ROUTE_IDS + backfill + ordering
    scripts/smoke_llm_gateway_routes.py                    both the GET /routes catalog check
                                                           and the RPC dispatch loop

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
#:
#: `harness` (added 2026-08-20): the Anthropic Messages passthrough route the FCC/Claude Code
#: CLI harness resolves `MODEL=llamacpp/harness` to (see anthropic_passthrough.py). Split off
#: `chat` for the same reason `agent` was split off `chat` on 2026-08-14: `chat` carries live
#: Hub chat traffic, has zero admission/concurrency throttling (priority_admission.py only
#: gates routes tagged "background"), and its worker is `n_parallel: 1` -- a single FCC harness
#: turn (up to HARNESS_FCC_TIMEOUT_SEC=900s) can occupy the only slot for the whole turn. `37f4fab9c`
#: (2026-08-16) already fixed this exact class of problem for one lighter call (the "5b
#: reflection" background LLM call) by moving it off `chat`; this closes the same gap for the
#: much heavier FCC harness. As of this patch `harness` is configured as an interim alias of
#: `chat`'s own worker (same pattern `agent` used before Muse Glimmer existed) -- see
#: services/orion-llm-gateway/README.md's "Route table example" sections. That means today this
#: is a labeling/observability seam, not yet physical isolation: a live FCC turn and live chat
#: traffic still share circe-worker-1's one slot until `harness` is pointed at a distinct
#: worker or gets its own admission policy. Both `chat` and `harness` remain Juniper's own
#: reserved capacity per the 2026-07-30 note below (AI Town is the thing kept off circe, not
#: FCC) -- this split exists so the gateway can tell the two apart, not to gate one against the
#: other. `harness` is also in SYSTEM_LLM_ROUTES below: it is never meant to be a human's
#: interactive Compute choice, and that is enforced, not just documented -- see that set for why.
ACCEPTED_LLM_ROUTES: FrozenSet[str] = frozenset(
    {"chat", "quick", "metacog", "quick_background", "agent", "harness"}
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
# lane beside the lane it yields on, then the specialist, then the system-only harness lane
# (see SYSTEM_LLM_ROUTES below for what makes it non-interactive, and why that is enforced).
LLM_ROUTE_DISPLAY_ORDER: tuple[str, ...] = (
    "chat",
    "quick",
    "quick_background",
    "metacog",
    "agent",
    "harness",
)

if set(LLM_ROUTE_DISPLAY_ORDER) != set(ACCEPTED_LLM_ROUTES) or len(
    LLM_ROUTE_DISPLAY_ORDER
) != len(ACCEPTED_LLM_ROUTES):
    raise RuntimeError(
        "LLM_ROUTE_DISPLAY_ORDER must list every accepted route exactly once; "
        f"missing={sorted(ACCEPTED_LLM_ROUTES - set(LLM_ROUTE_DISPLAY_ORDER))} "
        f"extra={sorted(set(LLM_ROUTE_DISPLAY_ORDER) - ACCEPTED_LLM_ROUTES)}"
    )


# Routes that YIELD: they wait for upstream slot slack before dispatching, so a human choosing
# one interactively buys nothing but latency. This is a property of what the route IS, not of
# how it happens to be configured right now, and that distinction is load-bearing.
#
# The gateway reports a route's `priority` from its route-table entry, so a consumer filtering
# on "priority == background" silently fails open in two real situations: a rolling deploy where
# an older gateway does not send the field at all, and a route absent from
# LLM_GATEWAY_ROUTE_TABLE_JSON (reported `not_configured`, priority null). In both, a background
# lane would be offered to a human as an ordinary one. Consumers therefore treat a route as
# background if EITHER the payload says so or it appears here -- fail-safe, never fail-open.
BACKGROUND_LLM_ROUTES: FrozenSet[str] = frozenset({"quick_background"})

if not BACKGROUND_LLM_ROUTES <= ACCEPTED_LLM_ROUTES:
    raise RuntimeError(
        "BACKGROUND_LLM_ROUTES names routes that are not accepted: "
        f"{sorted(BACKGROUND_LLM_ROUTES - ACCEPTED_LLM_ROUTES)}"
    )


# Routes that exist for an automated/system caller, never for a human's interactive Compute
# picker -- distinct from BACKGROUND_LLM_ROUTES on purpose. `harness` (FCC/Claude Code CLI
# turns) must dispatch immediately, not wait for slot slack the way a background lane does, so
# it cannot carry `priority: "background"` without also picking up that admission-wait
# behaviour in llm_backend.py/openai_passthrough.py (both gate on `== "background"` exactly).
# It still needs to be invisible to a human choosing from Hub's Compute selector, the same way a
# background lane is -- caught live 2026-08-20: an earlier version of this module only left a
# comment claiming `harness` was "not human-interactive" with nothing that actually enforced it,
# so Hub's picker (which filters on `priority`, not on this module) would have offered it as an
# ordinary chooseable lane. `priority: "system"` is the route-table value that signals this; the
# fail-safe/fail-open reasoning for keeping a *definitional* copy here, not just relying on the
# route table, mirrors BACKGROUND_LLM_ROUTES above.
SYSTEM_LLM_ROUTES: FrozenSet[str] = frozenset({"harness"})

if not SYSTEM_LLM_ROUTES <= ACCEPTED_LLM_ROUTES:
    raise RuntimeError(
        "SYSTEM_LLM_ROUTES names routes that are not accepted: "
        f"{sorted(SYSTEM_LLM_ROUTES - ACCEPTED_LLM_ROUTES)}"
    )

if SYSTEM_LLM_ROUTES & BACKGROUND_LLM_ROUTES:
    raise RuntimeError(
        "A route cannot be both background (yields for slot slack) and system "
        "(dispatches immediately, just hidden from the human picker): "
        f"{sorted(SYSTEM_LLM_ROUTES & BACKGROUND_LLM_ROUTES)}"
    )


def normalize_llm_route(raw: object) -> Optional[str]:
    """Canonical route name for `raw`, or None if it is absent, unrecognised, or system-only.

    None is deliberately the same answer for "nothing was asked for" and "what was asked for is
    not a route": both mean *do not override*. Callers that need to tell those apart -- to log a
    rejected override rather than silently ignoring it -- should compare against
    `ACCEPTED_LLM_ROUTES` themselves, the way `orion-cortex-exec` keeps `attempted` distinct from
    `accepted`.

    A `SYSTEM_LLM_ROUTES` member (`harness`) is also rejected here, not just accepted-and-hidden
    from a UI. Review caught this live 2026-08-20: widening `ACCEPTED_LLM_ROUTES` to include
    `harness` made it a valid override at all three of this function's real call sites --
    `orion-actions` (`ACTIONS_*_LLM_ROUTE=harness`), `orion-cortex-exec`
    (`_resolve_llm_route_override`, any caller-supplied `ctx["llm_route"]`), and Hub's
    `cortex_request_builder.py` (a raw `POST /api/chat` body, not just its own dropdown) -- none
    of which route through Hub's picker, so `SYSTEM_LLM_ROUTES`' UI-side exclusion never reached
    them. `harness` is resolved through a completely separate path in production
    (`anthropic_passthrough.resolve_anthropic_route`, which never calls this function -- it looks
    the route table up directly), so rejecting it here costs that real usage nothing.
    """
    route = str(raw or "").strip().lower()
    if not route:
        return None
    route = LLM_ROUTE_ALIASES.get(route, route)
    if route not in ACCEPTED_LLM_ROUTES or route in SYSTEM_LLM_ROUTES:
        return None
    return route
