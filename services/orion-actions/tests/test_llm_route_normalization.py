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

from orion.llm.routes import ACCEPTED_LLM_ROUTES, normalize_llm_route


# ------------------------------------------------------------ the regression

def test_quick_background_is_not_rewritten_to_chat():
    """THE BUG. This exact value is in the live .env and was becoming `chat`."""
    assert normalize_llm_route("quick_background") == "quick_background"


def test_agent_is_not_rewritten_to_chat():
    """Same silent rewrite, same cause -- a second route the older allow-list never learned."""
    assert normalize_llm_route("agent") == "agent"


def test_every_route_the_executor_dispatches_survives_normalization():
    """A route the executor accepts must round-trip. This is the drift guard: adding a lane to
    ACCEPTED_LLM_ROUTES without teaching the normalizer would fail here."""
    for route in ACCEPTED_LLM_ROUTES:
        assert normalize_llm_route(route) == route


def test_actions_uses_the_shared_set_rather_than_its_own_copy():
    """The drift was two private copies. orion-cortex-exec's half of this pairing is asserted in
    that service's own suite (test_llm_route_override.py)."""
    from app.main import ACCEPTED_LLM_ROUTES as actions_set
    assert actions_set is ACCEPTED_LLM_ROUTES


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
