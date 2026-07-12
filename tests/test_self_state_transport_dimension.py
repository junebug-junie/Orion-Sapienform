from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from orion.attention.field_attention.builder import build_attention_frame
from orion.attention.field_attention.policy import load_attention_policy
from orion.self_state.builder import build_self_state
from orion.self_state.policy import load_self_state_policy
from orion.self_state.scoring import weighted_overall_intensity

REPO_ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 5, 25, 23, 30, 10, tzinfo=timezone.utc)


def _fixtures():
    from orion.schemas.field_state import FieldStateV1

    field = FieldStateV1(
        tick_id="tick_self_transport",
        generated_at=NOW,
        topology_id="orion_field_topology",
        topology_version="v1",
        node_vectors={
            "node:athena": {
                "bus_health": 1.0,
                "delivery_confidence": 1.0,
                "contract_pressure": 1.0,
                "transport_pressure": 0.0,
            }
        },
        capability_vectors={
            "capability:transport": {
                "contract_pressure": 1.0,
                "confidence": 1.0,
                "available_capacity": 1.0,
            }
        },
        edges=[],
        recent_perturbations=[],
    )
    attn_policy = load_attention_policy(REPO_ROOT / "config" / "attention" / "field_attention_policy.v1.yaml")
    attention = build_attention_frame(field=field, policy=attn_policy, now=NOW)
    self_policy = load_self_state_policy(REPO_ROOT / "config" / "self_state" / "self_state_policy.v1.yaml")
    return field, attention, self_policy


def test_flag_false_no_transport_dimension() -> None:
    field, attention, policy = _fixtures()
    state = build_self_state(field=field, attention=attention, policy=policy, enable_transport_influence=False)
    assert "transport_integrity" not in state.dimensions


def test_flag_true_catalog_drift_transport_integrity() -> None:
    field, attention, policy = _fixtures()
    state = build_self_state(field=field, attention=attention, policy=policy, enable_transport_influence=True)
    assert "transport_integrity" in state.dimensions
    assert state.dimensions["transport_integrity"].score == 0.5
    assert "transport_contract_drift" in state.summary_labels


def test_transport_integrity_has_policy_weight() -> None:
    _, _, policy = _fixtures()
    assert "transport_integrity" in policy.dimension_weights
    assert policy.dimension_weights["transport_integrity"] > 0.0


def test_transport_integrity_contributes_to_overall_intensity_when_enabled() -> None:
    # Phase 1 (2026-07-12): transport_integrity previously had no
    # dimension_weights entry, and even when one existed, the builder folded
    # its score into dimension_scores *after* overall_intensity was already
    # computed — so it never actually moved overall_intensity either way.
    # Both bugs are fixed: it now has a policy weight and is computed before
    # overall_intensity when the flag is on.
    field, attention, policy = _fixtures()
    state_off = build_self_state(
        field=field, attention=attention, policy=policy, enable_transport_influence=False
    )
    state_on = build_self_state(
        field=field, attention=attention, policy=policy, enable_transport_influence=True
    )

    # Recompute the weighted average by hand from the actual per-dimension
    # scores in each state, proving overall_intensity is a real function of
    # what's in `dimensions` (not just that some downstream number happens
    # to differ).
    dim_scores_on = {dim_id: d.score for dim_id, d in state_on.dimensions.items()}
    assert "transport_integrity" in dim_scores_on
    expected_on = weighted_overall_intensity(dim_scores_on, policy)
    assert abs(state_on.overall_intensity - expected_on) < 1e-9

    dim_scores_off = {dim_id: d.score for dim_id, d in state_off.dimensions.items()}
    assert "transport_integrity" not in dim_scores_off
    expected_off = weighted_overall_intensity(dim_scores_off, policy)
    assert abs(state_off.overall_intensity - expected_off) < 1e-9

    # transport_integrity's score in this fixture (0.5) differs from the
    # weighted average of the other dimensions, so folding it in must move
    # overall_intensity.
    assert state_on.overall_intensity != state_off.overall_intensity
