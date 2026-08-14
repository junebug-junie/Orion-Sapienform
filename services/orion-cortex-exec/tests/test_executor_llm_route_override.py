from app.executor import _resolve_llm_route_override


def test_agent_override_from_options_llm_route():
    """Hub's Compute selector sends options.llm_route -- this was previously
    silently rejected for "agent" specifically (bug fixed 2026-08-14): a Hub
    turn with Mode: Quick + Compute: Agent produced a response, but nothing
    reached the dedicated agent-lane worker."""
    ctx = {"options": {"llm_route": "agent"}}
    assert _resolve_llm_route_override(ctx) == "agent"


def test_agent_override_from_top_level_llm_route():
    ctx = {"llm_route": "agent"}
    assert _resolve_llm_route_override(ctx) == "agent"


def test_existing_override_values_still_accepted():
    for value in ("chat", "quick", "metacog", "quick_background"):
        assert _resolve_llm_route_override({"options": {"llm_route": value}}) == value


def test_legacy_quick_aliases_still_normalize():
    for alias in ("chat_quick", "quick_chat", "chat_kids_story"):
        assert _resolve_llm_route_override({"options": {"llm_route": alias}}) == "quick"


def test_case_and_whitespace_normalized():
    assert _resolve_llm_route_override({"options": {"llm_route": "  AGENT  "}}) == "agent"


def test_unrecognized_value_falls_through_as_none():
    """An invalid override must return None (not the raw value) so the caller
    falls through to its own verb-based default mapping, not an unrecognized
    route key forwarded straight to the gateway."""
    assert _resolve_llm_route_override({"options": {"llm_route": "bogus"}}) is None


def test_absent_override_returns_none():
    assert _resolve_llm_route_override({}) is None
    assert _resolve_llm_route_override({"options": {}}) is None
    assert _resolve_llm_route_override({"options": None}) is None


def test_top_level_llm_route_takes_precedence_over_options():
    ctx = {"llm_route": "agent", "options": {"llm_route": "quick"}}
    assert _resolve_llm_route_override(ctx) == "agent"
