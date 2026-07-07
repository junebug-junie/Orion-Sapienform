from orion.schemas.attention_frame import OpenLoopV1, SalienceFeaturesV1
from orion.schemas.registry import resolve


def test_salience_features_defaults_are_bounded():
    f = SalienceFeaturesV1()
    assert f.evidence_strength == 0.0
    assert f.habituation == 0.0
    dumped = f.model_dump(mode="json")
    assert set(dumped) >= {
        "evidence_strength", "evidence_breadth", "recurrence",
        "recency", "novelty_vs_known", "dwell", "habituation",
    }


def test_open_loop_carries_salience_fields():
    loop = OpenLoopV1(id="open-loop-x", description="thing")
    assert loop.salience == 0.0
    assert loop.salience_features == {}


def test_salience_features_registered():
    assert resolve("SalienceFeaturesV1") is SalienceFeaturesV1
