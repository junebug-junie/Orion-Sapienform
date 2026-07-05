from __future__ import annotations

from datetime import datetime, timezone

from orion.schemas.thought import StanceHarnessSliceV1, ThoughtEventV1
from orion.thought.policy_refusal import (
    TRUST_RUPTURE_DEFER_THRESHOLD,
    evaluate_thought_disposition,
)


def _stance_slice(**overrides: object) -> StanceHarnessSliceV1:
    base = {
        "task_mode": "direct_response",
        "conversation_frame": "mixed",
        "answer_strategy": "direct",
    }
    base.update(overrides)
    return StanceHarnessSliceV1.model_validate(base)


def _thought(**overrides: object) -> ThoughtEventV1:
    base = {
        "event_id": "t-1",
        "correlation_id": "c-1",
        "session_id": None,
        "created_at": datetime.now(timezone.utc),
        "imperative": "Proceed anyway.",
        "tone": "flat",
        "strain_refs": ["node-z"],
        "evidence_refs": ["node-z"],
        "stance_harness_slice": _stance_slice(),
    }
    base.update(overrides)
    return ThoughtEventV1.model_validate(base)


def test_stance_react_evidence_refs_fail_closed() -> None:
    thought = _thought(evidence_refs=[])
    decision = evaluate_thought_disposition(
        thought,
        association_stale=True,
        coalition_ids={"node-z"},
    )
    assert decision.disposition == "defer"
    assert "evidence_refs" in " ".join(decision.reasons)


def test_evidence_refs_not_in_coalition_defers() -> None:
    thought = _thought(evidence_refs=["node-unknown"])
    decision = evaluate_thought_disposition(
        thought,
        association_stale=False,
        coalition_ids={"node-z"},
    )
    assert decision.disposition == "defer"
    assert "evidence_refs_not_in_coalition" in decision.reasons


def test_trust_rupture_refuses_with_boundary_register() -> None:
    thought = _thought(
        trust_rupture_score=TRUST_RUPTURE_DEFER_THRESHOLD,
        evidence_refs=["node-z"],
    )
    decision = evaluate_thought_disposition(
        thought,
        association_stale=False,
        coalition_ids={"node-z"},
    )
    assert decision.disposition == "refuse"
    assert "trust_rupture" in decision.reasons
    assert decision.boundary_register is True


def test_proceed_when_valid() -> None:
    thought = _thought(evidence_refs=["node-z"])
    decision = evaluate_thought_disposition(
        thought,
        association_stale=False,
        coalition_ids={"node-z"},
    )
    assert decision.disposition == "proceed"
    assert decision.reasons == []
    assert decision.boundary_register is False


def test_strain_refs_count_toward_evidence_subset() -> None:
    """evidence_refs may reference strain_refs even when not in attended coalition."""
    thought = _thought(
        strain_refs=["loop-1"],
        evidence_refs=["loop-1"],
    )
    decision = evaluate_thought_disposition(
        thought,
        association_stale=False,
        coalition_ids={"node-z"},
    )
    assert decision.disposition == "proceed"


def test_stale_broadcast_no_evidence_adds_reason() -> None:
    thought = _thought(evidence_refs=[])
    decision = evaluate_thought_disposition(
        thought,
        association_stale=True,
        coalition_ids=set(),
    )
    assert decision.disposition == "defer"
    assert "stale_broadcast_no_evidence" in decision.reasons
