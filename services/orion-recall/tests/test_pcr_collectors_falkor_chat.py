"""apply_collector_plan must suppress falkor_chat the same way it suppresses
rdf_chat -- "rdf_chat" in _COLLECTOR_PLANS means "chat-turn text content",
not "RDF specifically". Regression for a real bug: PCR intents like
"procedural"/"contradiction" deliberately exclude chat-turn history, and
without this, RECALL_FALKOR_IN_CHAT=true would leak it back in via a
different backend name."""

from __future__ import annotations

from app.pcr_collectors import apply_collector_plan, collectors_for_intent


def test_procedural_intent_suppresses_falkor_chat():
    plan = collectors_for_intent("procedural")
    assert "rdf_chat" not in plan  # confirms the premise: this intent excludes chat-turn content

    narrowed = apply_collector_plan({}, plan)
    assert narrowed["enable_falkor_chat"] is False
    assert narrowed["relevance"]["backend_weights"]["falkor_chat"] == 0.0
    assert narrowed["relevance"]["backend_weights"]["rdf_chat"] == 0.0


def test_contradiction_intent_suppresses_falkor_chat():
    plan = collectors_for_intent("contradiction")
    assert "rdf_chat" not in plan

    narrowed = apply_collector_plan({}, plan)
    assert narrowed["enable_falkor_chat"] is False
    assert narrowed["relevance"]["backend_weights"]["falkor_chat"] == 0.0


def test_relational_intent_does_not_suppress_falkor_chat():
    """Intents that DO want chat-turn content (rdf_chat: True in the plan)
    must not have enable_falkor_chat forced False."""
    plan = collectors_for_intent("relational")
    assert plan.get("rdf_chat") is True

    narrowed = apply_collector_plan({}, plan)
    assert "enable_falkor_chat" not in narrowed
    assert "falkor_chat" not in narrowed.get("relevance", {}).get("backend_weights", {})
