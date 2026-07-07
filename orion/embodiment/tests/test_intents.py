from __future__ import annotations

import pytest

from orion.embodiment.intents import build_intent
from orion.schemas.embodiment import EmbodimentIntentV1


def test_build_intent_sets_fields():
    intent = build_intent(
        kind="approach_player",
        source="deliberate",
        reason="walk toward Juniper",
        correlation_id="turn-9",
        ref="Juniper",
        urgency=0.4,
    )
    assert isinstance(intent, EmbodimentIntentV1)
    assert intent.kind == "approach_player"
    assert intent.ref == "Juniper"
    assert intent.correlation_id == "turn-9"


def test_build_intent_rejects_empty_reason():
    with pytest.raises(ValueError):
        build_intent(kind="idle", source="involuntary", reason="", correlation_id="c")
