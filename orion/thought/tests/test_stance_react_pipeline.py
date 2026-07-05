from __future__ import annotations

import json
from datetime import datetime, timezone

from orion.schemas.attention_frame import (
    AttentionBroadcastProjectionV1,
    AttentionFrameV1,
    OpenLoopV1,
)
from orion.schemas.thought import (
    HubAssociationBundleV1,
    StanceHarnessSliceV1,
    StanceReactRequestV1,
    ThoughtEventV1,
)
from orion.thought.coalition import coalition_ids_from_association
from orion.thought.policy_refusal import TRUST_RUPTURE_DEFER_THRESHOLD
from orion.thought.stance_react import apply_stance_react_pipeline, parse_stance_react_payload


def _broadcast(
    *,
    attended: list[str] | None = None,
    open_loops: list[OpenLoopV1] | None = None,
) -> AttentionBroadcastProjectionV1:
    return AttentionBroadcastProjectionV1(
        frame=AttentionFrameV1(open_loops=open_loops or []),
        attended_node_ids=attended or [],
    )


def _request(
    *,
    broadcast_stale: bool = False,
    broadcast: AttentionBroadcastProjectionV1 | None = None,
) -> StanceReactRequestV1:
    return StanceReactRequestV1(
        correlation_id="c-1",
        session_id="sess-1",
        user_message="hello",
        association=HubAssociationBundleV1(
            correlation_id="c-1",
            broadcast=broadcast,
            broadcast_stale=broadcast_stale,
            read_source="felt_state_reader",
        ),
        repair_bundle=None,
        stance_inputs={"user_message": "hello"},
    )


def _thought(**overrides: object) -> ThoughtEventV1:
    base = {
        "event_id": "t-1",
        "correlation_id": "c-1",
        "session_id": "sess-1",
        "created_at": datetime.now(timezone.utc),
        "imperative": "Answer directly.",
        "tone": "neutral",
        "strain_refs": ["node-a"],
        "evidence_refs": ["node-a"],
        "stance_harness_slice": StanceHarnessSliceV1(
            task_mode="direct_response",
            conversation_frame="mixed",
            answer_strategy="direct",
        ),
    }
    base.update(overrides)
    return ThoughtEventV1.model_validate(base)


def test_coalition_ids_from_association() -> None:
    loop = OpenLoopV1(id="loop-1", description="unfinished thread")
    assoc = HubAssociationBundleV1(
        correlation_id="c-1",
        broadcast=_broadcast(attended=["node-a"], open_loops=[loop]),
        broadcast_stale=False,
        read_source="felt_state_reader",
    )
    assert coalition_ids_from_association(assoc) == {"node-a", "loop-1"}


def test_parse_stance_react_payload_from_dict() -> None:
    raw = _thought().model_dump(mode="json")
    parsed = parse_stance_react_payload(raw)
    assert parsed.event_id == "t-1"
    assert parsed.profile == "stance_react"


def test_parse_stance_react_payload_from_json_string() -> None:
    raw = json.dumps(_thought().model_dump(mode="json"))
    parsed = parse_stance_react_payload(raw)
    assert parsed.imperative == "Answer directly."


def test_apply_stance_react_pipeline_proceed() -> None:
    broadcast = _broadcast(attended=["node-a"])
    req = _request(broadcast=broadcast)
    thought = _thought()
    result = apply_stance_react_pipeline(thought, req)
    assert result.disposition == "proceed"
    assert result.boundary_register is False
    assert result.disposition_reasons == []


def test_apply_stance_react_pipeline_defer_missing_evidence() -> None:
    req = _request(broadcast_stale=True)
    thought = _thought(evidence_refs=[])
    result = apply_stance_react_pipeline(thought, req)
    assert result.disposition == "defer"
    assert "missing_evidence_refs" in result.disposition_reasons


def test_apply_stance_react_pipeline_defer_evidence_not_in_coalition() -> None:
    broadcast = _broadcast(attended=["node-a"])
    req = _request(broadcast=broadcast)
    thought = _thought(evidence_refs=["node-bad"], strain_refs=["node-a"])
    result = apply_stance_react_pipeline(thought, req)
    assert result.disposition == "defer"
    assert "evidence_refs_not_in_coalition" in result.disposition_reasons


def test_apply_stance_react_pipeline_refuse_trust_rupture() -> None:
    broadcast = _broadcast(attended=["node-a"])
    req = _request(broadcast=broadcast)
    thought = _thought(trust_rupture_score=TRUST_RUPTURE_DEFER_THRESHOLD + 0.01)
    result = apply_stance_react_pipeline(thought, req)
    assert result.disposition == "refuse"
    assert result.boundary_register is True
    assert "trust_rupture" in result.disposition_reasons
