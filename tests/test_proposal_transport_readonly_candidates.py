from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from orion.proposals.builder import build_proposal_frame
from orion.proposals.policy import load_proposal_policy
from orion.proposals.templates import FORBIDDEN_TRANSPORT_PROPOSAL_KEYS, TRANSPORT_PROPOSAL_TEMPLATE_KEYS
from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1

REPO_ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 5, 25, 23, 30, 10, tzinfo=timezone.utc)


def _self_state() -> SelfStateV1:
    dims = {
        dim: SelfStateDimensionV1(dimension_id=dim, score=0.8, confidence=0.7)
        for dim in (
            "field_intensity",
            "coherence",
            "uncertainty",
            "agency_readiness",
            "resource_pressure",
            "execution_pressure",
            "reasoning_pressure",
            "reliability_pressure",
            "continuity_pressure",
            "introspection_pressure",
            "social_pressure",
            "policy_pressure",
        )
    }
    dims["reliability_pressure"] = SelfStateDimensionV1(
        dimension_id="reliability_pressure", score=0.9, confidence=0.8
    )
    return SelfStateV1(
        self_state_id="self.state:transport:test",
        generated_at=NOW,
        source_field_tick_id="tick",
        source_field_generated_at=NOW,
        source_attention_frame_id="attention.frame:test",
        source_attention_generated_at=NOW,
        overall_condition="loaded",
        overall_intensity=0.6,
        overall_confidence=0.7,
        dimensions=dims,
        dominant_attention_targets=["capability:transport"],
        dominant_field_channels={"contract_pressure": 1.0},
        summary_labels=["transport_contract_drift"],
    )


def test_transport_readonly_templates_present() -> None:
    policy = load_proposal_policy(REPO_ROOT / "config" / "proposals" / "proposal_policy.v1.yaml")
    frame = build_proposal_frame(self_state=_self_state(), attention=None, field=None, policy=policy, now=NOW)
    keys = {c.proposal_id.split(":")[1] for c in frame.candidates}
    assert TRANSPORT_PROPOSAL_TEMPLATE_KEYS & keys
    assert not (FORBIDDEN_TRANSPORT_PROPOSAL_KEYS & keys)
    transport_candidates = [
        c for c in frame.candidates if c.proposal_id.split(":")[1] in TRANSPORT_PROPOSAL_TEMPLATE_KEYS
    ]
    assert all(c.required_policy_gate == "read_only" for c in transport_candidates)
