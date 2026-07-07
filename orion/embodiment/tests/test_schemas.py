from __future__ import annotations

import pytest

from orion.schemas.embodiment import (
    EMBODIMENT_INTENT_KIND,
    EMBODIMENT_OUTCOME_KIND,
    EMBODIMENT_PERCEPTION_KIND,
    EMBODIMENT_PERSONA_KIND,
    EmbodimentIntentV1,
    EmbodimentOutcomeV1,
    OrionTownPersonaV1,
    WorldPerceptionV1,
)


def test_intent_round_trip():
    intent = EmbodimentIntentV1(
        kind="approach_player",
        source="involuntary",
        reason="social pressure 0.81 dominant",
        correlation_id="tick-1",
    )
    dumped = intent.model_dump(mode="json")
    assert EmbodimentIntentV1.model_validate(dumped) == intent
    assert dumped["urgency"] == 0.0


def test_intent_reason_must_be_non_empty():
    with pytest.raises(ValueError):
        EmbodimentIntentV1(kind="idle", source="deliberate", reason="  ", correlation_id="c")


def test_intent_rejects_unknown_kind():
    with pytest.raises(ValueError):
        EmbodimentIntentV1(kind="teleport", source="deliberate", reason="x", correlation_id="c")


def test_outcome_round_trip():
    out = EmbodimentOutcomeV1(
        intent_correlation_id="c",
        source="deliberate",
        status="actuated",
        reason="moved",
        send_input_ok=True,
        resolved_destination={"x": 1.0, "y": 2.0},
    )
    assert EmbodimentOutcomeV1.model_validate(out.model_dump(mode="json")) == out


def test_perception_and_persona_round_trip():
    perc = WorldPerceptionV1(player_id="p1", position={"x": 0.0, "y": 0.0})
    assert WorldPerceptionV1.model_validate(perc.model_dump(mode="json")) == perc
    persona = OrionTownPersonaV1(
        identity_blurb="Orion is a curious digital mind.",
        plan="explore and connect",
        spritesheet="f1",
        persona_source="projection",
    )
    assert OrionTownPersonaV1.model_validate(persona.model_dump(mode="json")) == persona


def test_kind_constants():
    assert EMBODIMENT_INTENT_KIND == "embodiment.intent.v1"
    assert EMBODIMENT_OUTCOME_KIND == "embodiment.outcome.v1"
    assert EMBODIMENT_PERCEPTION_KIND == "embodiment.perception.v1"
    assert EMBODIMENT_PERSONA_KIND == "embodiment.persona.v1"
