"""#2534 decision 1 (approved 2026-10-07): an outage must not be credited as a recovery.

capability:vision pressure 0.85 wins resource_pressure in the BEFORE tick; the
eye then goes dark, apply_diffusion drops vision's pressure as unmeasured, and
resource_pressure falls to the next measured capability. On main that drop was
credited ("pressure_delta:resource_pressure:-0.5...") because
channel_write_backed() only checks the AFTER winner, which is genuinely
measured. These fail on main.
"""
from __future__ import annotations

from datetime import datetime, timezone

from orion.feedback.builder import _gate_positive_delta_channels, build_feedback_frame
from orion.feedback.outcome_resolution import resolve_action_outcomes
from orion.feedback.policy import load_feedback_policy
from orion.field.credit_integrity import (
    BEFORE_WINNER_UNMEASURED,
    before_winner_went_unmeasured,
    dimension_winner_holder,
)
from orion.schemas.action_prediction import ExpectedEffectV1
from orion.schemas.execution_dispatch_frame import ExecutionDispatchCandidateV1, ExecutionDispatchFrameV1
from orion.schemas.field_state import FieldEdgeV1, FieldStateV1
from pathlib import Path

NOW = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)
REPO = Path(__file__).resolve().parents[1]
POLICY = load_feedback_policy(REPO / "config" / "feedback" / "feedback_policy.v1.yaml")
EDGES = [
    FieldEdgeV1(
        source_id="node:substrate.vision_organ",
        target_id="capability:vision",
        edge_type="node_capability",
        weight=0.85,
        channel_map={"vision_frame_staleness": "pressure"},
    ),
    FieldEdgeV1(
        source_id="node:athena",
        target_id="capability:storage",
        edge_type="node_capability",
        weight=0.85,
        channel_map={"disk_pressure": "pressure"},
    ),
]


def _field(tick_id: str, *, vision: float | None, storage: float) -> FieldStateV1:
    caps = {"capability:storage": {"pressure": storage}}
    prov = {"capability:storage": {"pressure": "node:athena"}}
    nodes = {"node:athena": {"disk_pressure": storage / 0.85}}
    if vision is not None:
        caps["capability:vision"] = {"pressure": vision}
        prov["capability:vision"] = {"pressure": "node:substrate.vision_organ"}
        nodes["node:substrate.vision_organ"] = {"vision_frame_staleness": vision / 0.85}
    else:
        caps["capability:vision"] = {}
        prov["capability:vision"] = {}
    return FieldStateV1(
        generated_at=NOW,
        tick_id=tick_id,
        node_vectors=nodes,
        node_vector_updated_at={n: {c: NOW for c in v} for n, v in nodes.items()},
        capability_vectors=caps,
        capability_provenance=prov,
        edges=EDGES,
    )


BEFORE = _field("before", vision=0.85, storage=0.3)
AFTER_DARK = _field("after", vision=None, storage=0.3)
AFTER_RECOVERED = _field("after", vision=0.1, storage=0.3)


def test_winner_holder_is_the_capability_vector_not_the_feeding_node() -> None:
    assert dimension_winner_holder(BEFORE, "resource_pressure") == ("capability", "capability:vision", "pressure")


def test_dark_before_winner_is_detected_but_a_real_recovery_is_not() -> None:
    assert before_winner_went_unmeasured(BEFORE, AFTER_DARK, "resource_pressure") is True
    assert before_winner_went_unmeasured(BEFORE, AFTER_RECOVERED, "resource_pressure") is False
    assert before_winner_went_unmeasured(None, AFTER_DARK, "resource_pressure") is False


def test_reseeded_key_without_provenance_still_reads_unmeasured() -> None:
    """reconcile re-seeds the key at 0.0 every tick; an edge-fed channel with no
    provenance is unmeasured even when the key is present."""
    after = AFTER_DARK.model_copy(deep=True)
    after.capability_vectors["capability:vision"]["pressure"] = 0.0
    assert before_winner_went_unmeasured(BEFORE, after, "resource_pressure") is True


def test_gate_withholds_the_outage_and_credits_the_recovery() -> None:
    gated, backed, withheld = _gate_positive_delta_channels(
        AFTER_DARK, {"resource_pressure": "decrease"}, max_staleness_seconds=120.0, field_before=BEFORE
    )
    assert "resource_pressure" not in gated
    assert backed["resource_pressure"] is False
    assert withheld == [f"withheld:resource_pressure:{BEFORE_WINNER_UNMEASURED}"]

    gated, _backed, withheld = _gate_positive_delta_channels(
        AFTER_RECOVERED, {"resource_pressure": "decrease"}, max_staleness_seconds=120.0, field_before=BEFORE
    )
    assert gated == {"resource_pressure": "decrease"}
    assert withheld == []


def _dispatch(candidates=()) -> ExecutionDispatchFrameV1:
    return ExecutionDispatchFrameV1(
        frame_id="execution.dispatch.frame:t",
        source_policy_frame_id="policy.frame:t",
        source_proposal_frame_id="proposal.frame:t",
        source_field_tick_id="before",
        generated_at=NOW,
        execution_dispatch_policy_id="execution_dispatch_policy.v1",
        dispatch_mode="dispatch_read_only",
        dispatch_attempted=bool(candidates),
        dispatched_candidates=list(candidates),
        dispatch_count=len(candidates),
    )


def test_feedback_frame_does_not_credit_an_outage_as_recovery() -> None:
    frame = build_feedback_frame(
        dispatch_frame=_dispatch(),
        policy_frame=None,
        proposal_frame=None,
        field_before=BEFORE,
        field_after=AFTER_DARK,
        cortex_results=None,
        policy=POLICY,
        now=NOW,
    )
    assert not any(e.startswith("pressure_delta:resource_pressure") for e in frame.positive_evidence)
    assert f"withheld:resource_pressure:{BEFORE_WINNER_UNMEASURED}" in frame.withheld_evidence


def test_outcome_resolution_skips_a_claim_whose_before_winner_went_dark() -> None:
    cand = ExecutionDispatchCandidateV1(
        dispatch_id="d1",
        source_decision_id="decision:d1",
        source_proposal_id="proposal:d1",
        dispatch_status="dispatched",
        dispatch_mode="dispatch_read_only",
        dispatch_kind="inspect",
        target_id="capability:vision",
        target_kind="capability",
        risk_score=0.05,
        confidence_score=0.9,
        dispatched_at=NOW,
        result_ref="result:d1",
        expected_effect=ExpectedEffectV1(
            signal_id="resource_pressure",
            direction="decrease",
            predicted_delta=-0.1,
            predictor_variance=0.25,
            predictor_n=0,
            cold_start=True,
        ),
    )
    res = resolve_action_outcomes(
        dispatch_frame=_dispatch([cand]),
        feedback_frame_id="feedback.frame:t",
        field_before=BEFORE,
        field_after=AFTER_DARK,
        now=NOW,
    )
    assert res.records == []
    assert res.skipped == {"d1": f"{BEFORE_WINNER_UNMEASURED}:resource_pressure"}

    res = resolve_action_outcomes(
        dispatch_frame=_dispatch([cand]),
        feedback_frame_id="feedback.frame:t",
        field_before=BEFORE,
        field_after=AFTER_RECOVERED,
        now=NOW,
    )
    assert len(res.records) == 1
