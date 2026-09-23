from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.schemas.attention_frame import (
    AttentionBroadcastProjectionV1,
    AttentionFrameV1,
    OpenLoopV1,
)
from orion.schemas.field_attention_frame import (
    FieldAttentionFrameV1,
    FieldAttentionTargetV1,
)
from orion.substrate.system_one_appraisal import (
    QUESTION_SET_ID,
    build_system_one_grammar_events,
    build_system_one_input_state,
    run_system_one_appraisal,
)


NOW = datetime(2026, 9, 23, 5, 0, tzinfo=timezone.utc)


def _broadcast() -> AttentionBroadcastProjectionV1:
    frame = AttentionFrameV1(
        generated_at=NOW,
        open_loops=[
            OpenLoopV1(
                id="loop-1",
                target_type="concept",
                description="unresolved graph contradiction",
                why_it_matters="two active claims conflict",
                salience=0.82,
                combined_salience=0.88,
                confidence=0.77,
                source_refs=["node:substrate.a", "node:substrate.b"],
            ),
            OpenLoopV1(
                id="loop-2",
                target_type="future_event",
                description="scheduled reading queue item",
                salience=0.42,
                combined_salience=0.42,
                confidence=0.65,
                source_refs=["reading:req-1"],
            ),
        ],
        live_unknowns=["which claim survives"],
        deferred_items=["reading:req-1"],
        effort_budget_used=0.2,
    )
    return AttentionBroadcastProjectionV1(
        generated_at=NOW,
        frame=frame,
        selected_action_type="reflect",
        selected_open_loop_id="loop-1",
        selected_description="unresolved graph contradiction",
        attended_node_ids=["node:substrate.a", "node:substrate.b"],
        dwell_ticks=3,
        coalition_stability_score=0.71,
    )


def _field_frame() -> FieldAttentionFrameV1:
    target = FieldAttentionTargetV1(
        target_id="node:substrate.execution",
        target_kind="node",
        salience_score=0.73,
        pressure_score=0.68,
        novelty_score=0.31,
        urgency_score=0.52,
        confidence_score=0.84,
        dominant_channels={"prediction_error": 0.68},
        reasons=["execution pressure elevated"],
        evidence_refs=["grammar:event-1"],
    )
    return FieldAttentionFrameV1(
        frame_id="field-frame-1",
        generated_at=NOW,
        source_field_tick_id="field-tick-1",
        source_field_generated_at=NOW,
        overall_salience=0.73,
        dominant_targets=[target],
        node_targets=[target],
    )


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload
        self.headers = {"x-typesafe-request-id": "kev-request-1"}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def _response_payload() -> dict:
    answers = {}
    for key in (
        "reverie_fit",
        "curiosity_pull",
        "deliberation_need",
        "attention_interrupt",
    ):
        answers[key] = {
            "type": "score",
            "score": 1.4,
            "confidence": 0.72,
            "legend": {"0": "low", "1": "middle", "2": "high"},
            "probabilities": {"0": 0.10, "1": 0.40, "2": 0.50},
        }
    return {
        "model": "kev-latest",
        "answers": answers,
        "usage": {"input_tokens": 123, "output_tokens": 45},
        "latency_ms": 87,
    }


def test_build_input_state_is_bounded_and_preserves_source_lineage() -> None:
    state = build_system_one_input_state(
        broadcast=_broadcast(),
        field_frame=_field_frame(),
        max_open_loops=1,
        max_targets=1,
    )

    assert state.source_broadcast_projection_id == "substrate.attention.broadcast.v1"
    assert state.source_field_attention_frame_id == "field-frame-1"
    assert [loop.loop_id for loop in state.open_loops] == ["loop-1"]
    assert state.open_loops[0].source_refs == [
        "node:substrate.a",
        "node:substrate.b",
    ]
    assert [target.target_id for target in state.field_dominant_targets] == [
        "node:substrate.execution"
    ]


def test_run_system_one_appraisal_preserves_probability_surface() -> None:
    calls = []

    def fake_post(url, *, json, headers, timeout):
        calls.append((url, json, headers, timeout))
        return _FakeResponse(_response_payload())

    frame = run_system_one_appraisal(
        broadcast=_broadcast(),
        field_frame=_field_frame(),
        base_url="http://kev:8009",
        model="kev-latest",
        post=fake_post,
        now=NOW,
    )

    assert frame.schema_version == "system_one.appraisal.frame.v1"
    assert frame.question_set_id == QUESTION_SET_ID
    assert frame.request_id == "kev-request-1"
    assert frame.answers["reverie_fit"].score == pytest.approx(1.4)
    assert frame.answers["reverie_fit"].probabilities == {
        "0": 0.10,
        "1": 0.40,
        "2": 0.50,
    }
    assert frame.usage.input_tokens == 123
    assert calls[0][0] == "http://kev:8009/v1/systemone"
    assert set(calls[0][1]["questions"]) == set(frame.answers)
    assert calls[0][1]["state"]["selected_open_loop_id"] == "loop-1"


def test_v1_base_url_is_not_duplicated() -> None:
    seen = {}

    def fake_post(url, *, json, headers, timeout):
        seen["url"] = url
        return _FakeResponse(_response_payload())

    run_system_one_appraisal(
        broadcast=_broadcast(),
        field_frame=None,
        base_url="http://kev:8009/v1",
        model="kev-latest",
        post=fake_post,
        now=NOW,
    )
    assert seen["url"] == "http://kev:8009/v1/systemone"


def test_partial_provider_response_is_rejected() -> None:
    payload = _response_payload()
    del payload["answers"]["curiosity_pull"]

    def fake_post(url, *, json, headers, timeout):
        return _FakeResponse(payload)

    with pytest.raises(ValueError, match="curiosity_pull"):
        run_system_one_appraisal(
            broadcast=_broadcast(),
            field_frame=_field_frame(),
            base_url="http://kev:8009",
            model="kev-latest",
            post=fake_post,
            now=NOW,
        )


def test_out_of_range_probability_is_rejected() -> None:
    payload = _response_payload()
    payload["answers"]["reverie_fit"]["probabilities"]["2"] = 1.2

    def fake_post(url, *, json, headers, timeout):
        return _FakeResponse(payload)

    with pytest.raises(ValueError, match="probability"):
        run_system_one_appraisal(
            broadcast=_broadcast(),
            field_frame=_field_frame(),
            base_url="http://kev:8009",
            model="kev-latest",
            post=fake_post,
            now=NOW,
        )


def test_grammar_trace_records_projection_not_full_frame_blob() -> None:
    def fake_post(url, *, json, headers, timeout):
        return _FakeResponse(_response_payload())

    frame = run_system_one_appraisal(
        broadcast=_broadcast(),
        field_frame=_field_frame(),
        base_url="http://kev:8009",
        model="kev-latest",
        post=fake_post,
        now=NOW,
    )
    events = build_system_one_grammar_events(frame)

    assert [event.event_kind for event in events] == [
        "trace_started",
        "atom_emitted",
        "projection_emitted",
        "trace_ended",
    ]
    projection = events[2].projection
    assert projection is not None
    assert projection.projection_type == "system_one_shadow_appraisal"
    assert projection.projection_id == frame.frame_id
    assert projection.expires_at == frame.expires_at
    assert events[1].atom is not None
    assert events[1].atom.payload_ref == f"substrate_system_one_appraisal:{frame.frame_id}"
    assert "input_state" not in projection.summary
