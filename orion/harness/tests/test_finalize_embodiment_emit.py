from __future__ import annotations

from orion.harness.finalize import build_finalize_embodiment_intent


def test_relational_turn_builds_deliberate_approach():
    intent = build_finalize_embodiment_intent(correlation_id="turn-7", interlocutor_ref="Juniper", relational=True)
    assert intent is not None
    assert intent.source == "deliberate"
    assert intent.correlation_id == "turn-7"
    assert intent.kind == "approach_player"
    assert intent.ref == "Juniper"


def test_non_relational_turn_no_intent():
    assert build_finalize_embodiment_intent(correlation_id="t", interlocutor_ref=None, relational=False) is None
