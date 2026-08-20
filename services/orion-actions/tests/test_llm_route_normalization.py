"""The configured journal route survives instead of being rewritten to `chat`.

`services/orion-actions/.env` has said `ACTIONS_JOURNAL_LLM_ROUTE=quick_background` since the key
existed. orion-actions' own allow-list predated that route and silently rewrote it -- and `agent`
-- to `chat`, which is circe's single-slot 131,072-token lane. Orion's journaling ran there for a
1,749-token median prompt because two services disagreed about what routes exist, with no log line
on the disagreement.

Live route table, `services/orion-llm-gateway/.env_example` (2026-08-18):
    quick_background -> atlas 100.121.214.30:8013, background priority, 2 reserved slots
    chat             -> circe 100.112.254.99:8011, single slot, 131,072 ctx
"""
from __future__ import annotations

import pytest

from orion.llm.routes import ACCEPTED_LLM_ROUTES, SYSTEM_LLM_ROUTES, normalize_llm_route


# ------------------------------------------------------------ the regression

def test_quick_background_is_not_rewritten_to_chat():
    """THE BUG. This exact value is in the live .env and was becoming `chat`."""
    assert normalize_llm_route("quick_background") == "quick_background"


def test_agent_is_not_rewritten_to_chat():
    """Same silent rewrite, same cause -- a second route the older allow-list never learned."""
    assert normalize_llm_route("agent") == "agent"


def test_every_route_the_executor_dispatches_survives_normalization():
    """A route the executor accepts must round-trip -- except SYSTEM_LLM_ROUTES. This is the
    drift guard: adding a lane to ACCEPTED_LLM_ROUTES without teaching the normalizer would
    still fail here for every ordinary route.

    `harness` (2026-08-20) is the deliberate exception, not a gap in this guard: it is a real,
    accepted route (for the catalog, `GET /routes`, the Anthropic passthrough), but is never a
    valid general-caller override -- see normalize_llm_route's docstring. Before this carve-out
    existed, widening ACCEPTED_LLM_ROUTES to include `harness` silently made it a valid
    ACTIONS_*_LLM_ROUTE value too, which is exactly the class of bug this file exists to catch.
    """
    for route in ACCEPTED_LLM_ROUTES - SYSTEM_LLM_ROUTES:
        assert normalize_llm_route(route) == route
    for route in SYSTEM_LLM_ROUTES:
        assert normalize_llm_route(route) is None


def test_every_shared_route_survives_this_services_wrapper():
    """BEHAVIOURAL drift guard, not an identity assertion.

    `assert app.main.ACCEPTED_LLM_ROUTES is ACCEPTED_LLM_ROUTES` would be tautological -- it
    only checks a re-export, so someone reintroducing a private set and using THAT inside
    `_normalized_llm_route` passes it cleanly. That is precisely the bug this file exists for.
    Drive the real wrapper instead.

    Same SYSTEM_LLM_ROUTES carve-out as the test above: `ACTIONS_*_LLM_ROUTE=harness` must be
    rejected exactly like a typo, not silently accepted just because `harness` is a real route.
    """
    from app.main import _normalized_llm_route

    for route in sorted(ACCEPTED_LLM_ROUTES - SYSTEM_LLM_ROUTES):
        assert _normalized_llm_route(route, "metacog") == route, route
    for route in sorted(SYSTEM_LLM_ROUTES):
        assert _normalized_llm_route(route, "metacog") is None, route


# ------------------------------------------------------------ the fallback

def test_an_unrecognized_route_yields_no_override_rather_than_chat():
    """The old fallback picked the largest, slowest, most contended lane in the fleet for any
    value it did not recognise. A typo should not cost circe's 131k lane."""
    assert normalize_llm_route("qiuck") is None
    assert normalize_llm_route("gpt-4") is None
    assert normalize_llm_route("chat_background") is None


def test_absent_and_unrecognized_are_both_no_override():
    for raw in (None, "", "   ", 0, [], {}):
        assert normalize_llm_route(raw) is None


# ------------------------------------------------------------ aliases still work

@pytest.mark.parametrize("alias", ["chat_quick", "quick_chat", "chat_kids_story"])
def test_legacy_aliases_still_resolve_to_quick(alias):
    """Live config still carries these spellings; this patch must not break them."""
    assert normalize_llm_route(alias) == "quick"


def test_normalization_is_case_and_whitespace_insensitive():
    assert normalize_llm_route("  QUICK_Background ") == "quick_background"


# ------------------------------------------------------------ the actions wrapper

def test_the_wrapper_prefers_the_specific_route_over_the_fallback():
    from app.main import _normalized_llm_route
    assert _normalized_llm_route("quick_background", "metacog") == "quick_background"


def test_the_wrapper_uses_the_fallback_when_nothing_specific_is_set():
    from app.main import _normalized_llm_route
    assert _normalized_llm_route(None, "metacog") == "metacog"


def test_the_wrapper_returns_none_and_warns_on_a_bad_value(caplog):
    """Silence is how this bug survived. A rejected override must say so."""
    import logging
    from app.main import _normalized_llm_route
    with caplog.at_level(logging.WARNING):
        assert _normalized_llm_route("nonsense", "alsononsense") is None
    assert any("actions_llm_route_unrecognized" in r.getMessage() for r in caplog.records)


def test_the_wrapper_is_silent_when_nothing_was_configured_at_all():
    """No config is not a misconfiguration -- warning on it would train the reader to ignore
    the warning that matters."""
    import logging
    from app.main import _normalized_llm_route
    import logging as _l
    recs = []
    h = _l.Handler()
    h.emit = recs.append
    logger = _l.getLogger()
    logger.addHandler(h)
    try:
        assert _normalized_llm_route(None, "") is None
    finally:
        logger.removeHandler(h)
    assert not [r for r in recs if "actions_llm_route_unrecognized" in r.getMessage()]
