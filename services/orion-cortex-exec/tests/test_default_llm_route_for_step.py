from __future__ import annotations

from app.executor import _default_llm_route_for_step


def test_stance_react_routes_chat_not_none():
    """Regression for the 2026-08-20 production incident: stance_react was
    missing from the default-route mapping entirely, silently falling to
    `None`. With route=None, the gateway's own default-route fallback is
    "quick" (atlas-worker-fast-1, a 512-token payload budget) -- but a real
    stance_react prompt is ~16k chars (~4-5k tokens), so every real call
    overflowed and the turn was deferred (corr=8d2d2600-8aec-49d3-a42f-
    71510a5b86de, "[LLM-GW ctx] overflow on route=quick and no larger lane
    exists -- returning error"). Must resolve to the DEEP lane ("chat" /
    Circe), same as the other fat-prompt verbs below."""
    assert _default_llm_route_for_step(verb_name="stance_react", step_name="llm_stance_react", mode="brain") == "chat"


def test_harness_finalize_reflect_routes_chat():
    assert _default_llm_route_for_step(verb_name="harness_finalize_reflect", step_name="x", mode=None) == "chat"


def test_orion_voice_finalize_routes_chat():
    assert _default_llm_route_for_step(verb_name="orion_voice_finalize", step_name="x", mode=None) == "chat"


def test_chat_general_stance_brief_routes_quick():
    assert (
        _default_llm_route_for_step(
            verb_name="chat_general", step_name="synthesize_chat_stance_brief", mode=None
        )
        == "quick"
    )


def test_chat_general_final_response_routes_chat():
    assert _default_llm_route_for_step(verb_name="chat_general", step_name="llm_chat_general", mode=None) == "chat"


def test_chat_general_other_step_falls_through_to_none():
    """Only the two named steps of chat_general get a default -- any other
    step name for this verb is not one of this mapping's known cases."""
    assert _default_llm_route_for_step(verb_name="chat_general", step_name="some_other_step", mode=None) is None


def test_fast_single_pass_chat_verbs_route_quick():
    assert _default_llm_route_for_step(verb_name="chat_quick", step_name="x", mode=None) == "quick"
    assert _default_llm_route_for_step(verb_name="chat_kids_story", step_name="x", mode=None) == "quick"


def test_introspect_spark_routes_quick():
    assert _default_llm_route_for_step(verb_name="introspect_spark", step_name="x", mode=None) == "quick"


def test_memory_graph_suggest_routes_quick():
    assert _default_llm_route_for_step(verb_name="memory_graph_suggest", step_name="x", mode=None) == "quick"


def test_metacog_mode_routes_metacog_regardless_of_verb():
    assert _default_llm_route_for_step(verb_name="some_unrelated_verb", step_name="x", mode="metacog") == "metacog"


def test_unknown_verb_and_mode_falls_through_to_none():
    assert _default_llm_route_for_step(verb_name="totally_unmapped_verb", step_name="x", mode="brain") is None
    assert _default_llm_route_for_step(verb_name=None, step_name=None, mode=None) is None
