from __future__ import annotations

from uuid import uuid4

from orion.mind.validation import hash_snapshot_inputs, validate_merged_stance_brief
from orion.mind.v1 import (
    MindControlDecisionV1,
    MindHandoffBriefV1,
    MindRunPolicyV1,
    MindRunRequestV1,
    MindRunResultV1,
    MindSnapshotFacetV1,
    MindStancePatchV1,
    MindStanceTrajectoryV1,
    MindUniverseSnapshotV1,
)


def test_mind_run_request_roundtrip() -> None:
    req = MindRunRequestV1(
        correlation_id="corr-1",
        session_id="sess-1",
        snapshot_inputs={"user_preview": "hello"},
        policy=MindRunPolicyV1(n_loops_max=2, router_profile_id="default"),
    )
    raw = req.model_dump(mode="json")
    again = MindRunRequestV1.model_validate(raw)
    assert again.correlation_id == "corr-1"
    assert again.policy.n_loops_max == 2


def test_mind_run_result_roundtrip() -> None:
    mid = uuid4()
    brief_dict = {
        "conversation_frame": "technical",
        "user_intent": "test",
        "self_relevance": "sr",
        "juniper_relevance": "jr",
        "answer_strategy": "DirectAnswer",
        "stance_summary": "summary",
    }
    res = MindRunResultV1(
        mind_run_id=mid,
        ok=True,
        trajectory=MindStanceTrajectoryV1(
            patches=[MindStancePatchV1(loop_index=0, structured={"stance_summary": "x"})],
            merged_stance_brief=brief_dict,
        ),
        decision=MindControlDecisionV1(allowed_verbs=["chat_general"]),
        brief=MindHandoffBriefV1(
            stance_payload=brief_dict,
            machine_contract={"mind.route_kind": "brain"},
        ),
    )
    raw = res.model_dump(mode="json")
    again = MindRunResultV1.model_validate(raw)
    assert again.mind_run_id == mid
    assert again.trajectory.patches[0].loop_index == 0


def test_universe_snapshot_facets() -> None:
    snap = MindUniverseSnapshotV1(
        facets={
            "autonomy": MindSnapshotFacetV1(trust="high", source="state", compact_json={"x": 1}, bytes_approx=12),
        },
        total_bytes_approx=12,
    )
    assert "autonomy" in snap.model_dump(mode="json")["facets"]


def test_validate_merged_stance_brief_accepts_minimal() -> None:
    brief = {
        "conversation_frame": "technical",
        "user_intent": "u",
        "self_relevance": "s",
        "juniper_relevance": "j",
        "answer_strategy": "a",
        "stance_summary": "st",
    }
    obj = validate_merged_stance_brief(brief)
    assert obj.conversation_frame == "technical"


def test_hash_snapshot_inputs_stable() -> None:
    h1 = hash_snapshot_inputs({"a": 1, "b": 2})
    h2 = hash_snapshot_inputs({"b": 2, "a": 1})
    assert h1 == h2
