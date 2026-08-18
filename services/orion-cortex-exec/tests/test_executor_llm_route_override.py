from app.executor import _resolve_llm_route_override


def test_agent_override_from_options_llm_route():
    """Hub's Compute selector sends options.llm_route -- this was previously
    silently rejected for "agent" specifically (bug fixed 2026-08-14): a Hub
    turn with Mode: Quick + Compute: Agent produced a response, but nothing
    reached the dedicated agent-lane worker."""
    ctx = {"options": {"llm_route": "agent"}}
    assert _resolve_llm_route_override(ctx) == ("agent", "agent")


def test_agent_override_from_top_level_llm_route():
    ctx = {"llm_route": "agent"}
    assert _resolve_llm_route_override(ctx) == ("agent", "agent")


def test_existing_override_values_still_accepted():
    for value in ("chat", "quick", "metacog", "quick_background"):
        assert _resolve_llm_route_override({"options": {"llm_route": value}}) == (value, value)


def test_legacy_quick_aliases_still_normalize():
    for alias in ("chat_quick", "quick_chat", "chat_kids_story"):
        assert _resolve_llm_route_override({"options": {"llm_route": alias}}) == ("quick", "quick")


def test_case_and_whitespace_normalized():
    assert _resolve_llm_route_override({"options": {"llm_route": "  AGENT  "}}) == ("agent", "agent")


def test_unrecognized_value_rejected_but_still_visible_as_attempted():
    """An invalid override must not be used for routing (accepted=None, so the
    caller falls through to its own verb-based default mapping), but the raw
    attempted value must stay visible for logging -- collapsing "no override
    supplied" and "override supplied but rejected" into the same None is
    exactly the diagnostic loss that let the original "agent" bug hide in
    plain sight (the llm_route_selected log line was how it was eventually
    traced live)."""
    assert _resolve_llm_route_override({"options": {"llm_route": "bogus"}}) == (None, "bogus")


def test_absent_override_returns_none_none():
    assert _resolve_llm_route_override({}) == (None, None)
    assert _resolve_llm_route_override({"options": {}}) == (None, None)
    assert _resolve_llm_route_override({"options": None}) == (None, None)


def test_top_level_llm_route_takes_precedence_over_options():
    ctx = {"llm_route": "agent", "options": {"llm_route": "quick"}}
    assert _resolve_llm_route_override(ctx) == ("agent", "agent")


def test_every_shared_route_survives_this_services_resolver():
    """BEHAVIOURAL drift guard, not an identity assertion.

    orion-actions kept its own copy of the accepted set and it drifted: still {chat, quick,
    metacog} long after `quick_background` and `agent` existed here, so its configured
    `ACTIONS_JOURNAL_LLM_ROUTE=quick_background` was silently rewritten to `chat` -- circe's
    single-slot 131k lane -- with no log line.

    Asserting `_SOME_LOCAL_NAME is ACCEPTED_LLM_ROUTES` would be tautological: a future private
    copy used by the resolver passes that cleanly. This drives the ACTUAL resolver with every
    route in the shared set instead, so a resolver that stops honouring one fails here.
    """
    from orion.llm.routes import ACCEPTED_LLM_ROUTES

    for route in sorted(ACCEPTED_LLM_ROUTES):
        assert _resolve_llm_route_override({"options": {"llm_route": route}}) == (route, route), route
