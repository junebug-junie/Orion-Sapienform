from pathlib import Path

import pytest
from pydantic import ValidationError

from orion.policy.policy import SubstratePolicyV1, load_substrate_policy

REPO = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO / "config" / "policy" / "substrate_policy.v1.yaml"


def test_loads_yaml() -> None:
    policy = load_substrate_policy(POLICY_PATH)
    assert policy.schema_version == "substrate_policy.v1"


def test_policy_id() -> None:
    policy = load_substrate_policy(POLICY_PATH)
    assert policy.policy_id == "substrate_policy.v1"


def test_execution_without_operator_disabled() -> None:
    policy = load_substrate_policy(POLICY_PATH)
    assert policy.autonomy.allow_execution_without_operator is False


def test_prepare_action_requires_review() -> None:
    policy = load_substrate_policy(POLICY_PATH)
    rule = policy.proposal_kind_rules["prepare_action"]
    assert rule.default_decision == "requires_operator_review"
    assert rule.max_autonomy_tier == "operator_review"


def test_inspect_read_only() -> None:
    policy = load_substrate_policy(POLICY_PATH)
    rule = policy.proposal_kind_rules["inspect"]
    assert rule.default_decision == "approved_read_only"
    assert rule.allowed_scope == "inspect_only"


def test_hard_blocks_include_cortex_exec() -> None:
    policy = load_substrate_policy(POLICY_PATH)
    assert "cortex_exec_direct_call" in policy.hard_blocks


def test_nested_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        SubstratePolicyV1.model_validate(
            {
                "schema_version": "substrate_policy.v1",
                "policy_id": "substrate_policy.v1",
                "autonomy": {"allow_executon_without_operator": False},
            }
        )
