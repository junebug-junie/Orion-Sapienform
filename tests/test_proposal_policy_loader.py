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
    assert "field_intensity" in tmpl.dimensions
