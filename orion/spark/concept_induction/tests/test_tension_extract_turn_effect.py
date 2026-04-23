from __future__ import annotations

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.spark.concept_induction.tensions import extract_tensions


def _env(payload: dict) -> BaseEnvelope:
    return BaseEnvelope(
        kind="chat.history.turn.v1",
        source=ServiceRef(name="test", version="0.0.0"),
        payload=payload,
    )


def test_extract_tensions_uses_spark_meta_nested_metadata_turn_effect() -> None:
    env = _env(
        {
            "spark_meta": {
                "metadata": {
                    "turn_effect": {
                        "turn": {"coherence": -0.2, "novelty": 0.3, "valence": 0.0, "energy": 0.0}
                    }
                }
            }
        }
    )
    tensions = extract_tensions(
        envelope=env,
        intake_channel="orion:chat:history:turn",
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
    )
    kinds = {t.kind for t in tensions}
    assert "tension.contradiction.v1" in kinds
    assert "tension.identity_drift.v1" in kinds

