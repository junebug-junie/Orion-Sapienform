from pathlib import Path

from orion.proposals.policy import load_proposal_policy

REPO = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO / "config" / "proposals" / "proposal_policy.v1.yaml"


def test_loads_yaml() -> None:
    policy = load_proposal_policy(POLICY_PATH)
    assert policy.schema_version == "proposal_policy.v1"


def test_policy_id() -> None:
    policy = load_proposal_policy(POLICY_PATH)
    assert policy.policy_id == "proposal_policy.v1"


def test_inspect_template_exists() -> None:
    policy = load_proposal_policy(POLICY_PATH)
    assert "inspect_execution_pressure" in policy.proposal_templates


def test_policy_review_requires_operator_review() -> None:
    policy = load_proposal_policy(POLICY_PATH)
    tmpl = policy.proposal_templates["request_policy_review_for_action"]
    assert tmpl.required_policy_gate == "operator_review"


def test_dimension_weights_include_execution_pressure() -> None:
    policy = load_proposal_policy(POLICY_PATH)
    assert "execution_pressure" in policy.dimension_weights


def test_inspect_node_resource_pressure_template_targets_node() -> None:
    policy = load_proposal_policy(POLICY_PATH)
    tmpl = policy.proposal_templates["inspect_node_resource_pressure"]
    assert tmpl.kind == "inspect"
    assert tmpl.target_kind == "node"
    assert tmpl.target_id == "node:atlas"
    assert "resource_pressure" in tmpl.dimensions


def test_inspect_field_topology_catalog_template_targets_field() -> None:
    policy = load_proposal_policy(POLICY_PATH)
    tmpl = policy.proposal_templates["inspect_field_topology_catalog"]
    assert tmpl.kind == "inspect"
    assert tmpl.target_kind == "field"
    assert tmpl.target_id == "config/field/orion_field_topology.v1.yaml"
    # 2026-07-30: field_intensity removed -- never produced by
    # field_pressures(), was this template's only (dead) dimension. Honestly
    # empty now rather than implying a scored-but-fake dimension.
    assert tmpl.dimensions == {}


def test_inspect_attended_target_template_has_attention_binding() -> None:
    policy = load_proposal_policy(POLICY_PATH)
    tmpl = policy.proposal_templates["inspect_attended_target"]
    assert tmpl.kind == "inspect"
    assert tmpl.target_kind == "capability"
    assert tmpl.target_id == "capability:orchestration"
    assert tmpl.target_binding == "attention.dominant_targets[0]"
    assert tmpl.required_policy_gate == "read_only"
    # 2026-07-30: field_intensity removed -- see
    # test_inspect_field_topology_catalog_template_targets_field's comment.
    assert tmpl.dimensions == {}


def test_other_templates_have_no_target_binding() -> None:
    policy = load_proposal_policy(POLICY_PATH)
    for key, tmpl in policy.proposal_templates.items():
        if key == "inspect_attended_target":
            continue
        assert tmpl.target_binding is None


def test_deviation_pressure_dimension_weight_declared() -> None:
    policy = load_proposal_policy(POLICY_PATH)
    assert "deviation_pressure" in policy.dimension_weights


def test_tension_driven_templates_score_on_deviation_pressure() -> None:
    policy = load_proposal_policy(POLICY_PATH)
    for key in ("observe_tension_via_camera", "prune_dangling_images", "prune_stopped_containers"):
        assert "deviation_pressure" in policy.proposal_templates[key].dimensions


def test_mutating_tension_templates_use_read_only_policy_gate_like_prune_build_cache() -> None:
    """Safety comes from execution_dispatch_policy.v1.yaml's three gates, not
    this field -- same precedent as prune_build_cache's own comment."""
    policy = load_proposal_policy(POLICY_PATH)
    for key in ("prune_dangling_images", "prune_stopped_containers"):
        tmpl = policy.proposal_templates[key]
        assert tmpl.kind == "maintain"
        assert tmpl.required_policy_gate == "read_only"
        assert tmpl.reversibility == 1.0


def test_observe_tension_via_camera_is_read_only() -> None:
    policy = load_proposal_policy(POLICY_PATH)
    tmpl = policy.proposal_templates["observe_tension_via_camera"]
    assert tmpl.kind == "inspect"
    assert tmpl.required_policy_gate == "read_only"
