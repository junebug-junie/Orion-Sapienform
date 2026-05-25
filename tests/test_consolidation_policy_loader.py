from pathlib import Path

from orion.consolidation.policy import ConsolidationPolicyV1, load_consolidation_policy

REPO = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO / "config" / "consolidation" / "consolidation_policy.v1.yaml"


def test_loads_yaml() -> None:
    policy = load_consolidation_policy(POLICY_PATH)
    assert policy.schema_version == "consolidation_policy.v1"
    assert policy.policy_id == "consolidation_policy.v1"


def test_window_config() -> None:
    policy = load_consolidation_policy(POLICY_PATH)
    assert policy.window.lookback_minutes == 60
    assert policy.window.min_support_count == 3


def test_motif_rules_present() -> None:
    policy = load_consolidation_policy(POLICY_PATH)
    labels = {r.label for r in policy.motif_rules.values()}
    assert "loaded_but_reliable" in labels
    assert "dry_run_feedback_loop" in labels
    assert "stable_after_dry_run" in labels


def test_tracked_dimensions() -> None:
    policy = load_consolidation_policy(POLICY_PATH)
    assert "execution_pressure" in policy.tracked_self_dimensions


def test_tensor_config() -> None:
    policy = load_consolidation_policy(POLICY_PATH)
    assert policy.tensor.enabled is True
    assert policy.tensor.max_coordinates == 200
    assert policy.tensor_axes["field_attention_self"] == [
        "time_bucket",
        "self_condition",
        "attention_target",
        "dimension",
    ]
    assert policy.tensor_axes["policy_dispatch_feedback"] == [
        "proposal_kind",
        "policy_decision",
        "dispatch_status",
        "feedback_outcome",
    ]
    assert policy.tensor_axes["motif_condition_outcome"] == [
        "motif",
        "self_condition",
        "outcome_status",
    ]
