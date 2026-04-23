from __future__ import annotations

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.spark.concept_induction.tensions import derive_pressure_competition_tensions


def _envelope() -> BaseEnvelope:
    return BaseEnvelope(
        kind="test.kind",
        source=ServiceRef(name="test", version="0.0.0"),
        payload={},
    )


def test_pressure_competition_empty_when_spread_low() -> None:
    env = _envelope()
    out = derive_pressure_competition_tensions(
        envelope=env,
        intake_channel="orion:chat:intake",
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        pressures={k: 0.5 for k in ("coherence", "continuity", "capability", "relational", "predictive", "autonomy")},
    )
    assert out == []


def test_pressure_competition_emits_when_relational_only_under_stability_alias() -> None:
    env = _envelope()
    out = derive_pressure_competition_tensions(
        envelope=env,
        intake_channel="orion:chat:intake",
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        pressures={
            "relational_stability": 0.8,
            "predictive": 0.65,
            "continuity": 0.6,
        },
    )
    assert len(out) == 1
    assert out[0].kind == "tension.drive_competition.v1"


def test_pressure_competition_emits_when_spread_high() -> None:
    env = _envelope()
    pressures = {
        "coherence": 0.2,
        "continuity": 0.2,
        "capability": 0.2,
        "relational": 0.2,
        "predictive": 0.2,
        "autonomy": 0.95,
    }
    out = derive_pressure_competition_tensions(
        envelope=env,
        intake_channel="orion:chat:intake",
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        pressures=pressures,
    )
    assert len(out) == 1
    assert out[0].kind == "tension.drive_competition.v1"
    assert out[0].magnitude >= 0.12
    assert "autonomy" in out[0].drive_impacts
