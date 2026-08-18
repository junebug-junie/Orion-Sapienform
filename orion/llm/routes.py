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

`agent` had the same fate. The set is now defined once, here, and imported by both.

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
