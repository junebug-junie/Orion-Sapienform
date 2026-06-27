from __future__ import annotations

from datetime import datetime, timezone

from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1
from orion.substrate.relational.adapters.self_state_ctx import map_self_state_ctx_to_substrate


def _state() -> SelfStateV1:
    now = datetime.now(timezone.utc)
    return SelfStateV1(
        self_state_id="ss1",
        generated_at=now,
        source_field_tick_id="ft1",
        source_field_generated_at=now,
        source_attention_frame_id="af1",
        source_attention_generated_at=now,
        overall_condition="loaded",
        overall_intensity=0.6,
        overall_confidence=0.7,
        dimensions={
            "coherence": SelfStateDimensionV1(dimension_id="coherence", score=0.4, confidence=0.8),
            "uncertainty": SelfStateDimensionV1(dimension_id="uncertainty", score=0.7, confidence=0.6),
        },
        dimension_trajectory={"coherence": -0.1},
        prediction_error_scores={"coherence": 0.55, "uncertainty": 0.2},
    )


def test_self_state_adapter_emits_self_model_nodes_carrying_prediction_error() -> None:
    record = map_self_state_ctx_to_substrate({"self_state": _state()})
    assert record is not None
    by_label = {n.label: n for n in record.nodes}

    # one node per dimension + an overall-condition anchor
    assert "self:coherence" in by_label
    assert "self:uncertainty" in by_label
    assert "self:overall_condition" in by_label

    # the self-model nodes carry standing prediction error (the rung-1 feedback signal)
    assert by_label["self:coherence"].metadata["prediction_error"] == 0.55
    assert by_label["self:uncertainty"].metadata["prediction_error"] == 0.2
    # dimension score is preserved as a belief
    assert by_label["self:coherence"].metadata["score"] == 0.4
    # overall node carries the worst-case surprise so the whole self gains pressure
    assert by_label["self:overall_condition"].metadata["prediction_error"] == 0.55
    # all anchored to orion
    assert all(n.anchor_scope == "orion" for n in record.nodes)


def test_self_state_adapter_accepts_dict_and_json_and_skips_when_absent() -> None:
    state = _state()
    assert map_self_state_ctx_to_substrate({"self_state": state.model_dump(mode="json")}) is not None
    assert map_self_state_ctx_to_substrate({"self_state": state.model_dump_json()}) is not None
    # no self_state in ctx -> no record (lane simply absent, never raises)
    assert map_self_state_ctx_to_substrate({}) is None
    assert map_self_state_ctx_to_substrate({"self_state": "not json"}) is None
