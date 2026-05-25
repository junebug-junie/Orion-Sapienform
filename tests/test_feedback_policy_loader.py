from pathlib import Path

from orion.feedback.policy import load_feedback_policy

REPO = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO / "config" / "feedback" / "feedback_policy.v1.yaml"


def test_loads_yaml() -> None:
    policy = load_feedback_policy(POLICY_PATH)
    assert policy.schema_version == "feedback_policy.v1"
    assert policy.policy_id == "feedback_policy.v1"


def test_windows_defaults() -> None:
    policy = load_feedback_policy(POLICY_PATH)
    assert policy.windows.field_after_window_sec == 30
    assert policy.windows.result_wait_window_sec == 30


def test_scoring_keys() -> None:
    policy = load_feedback_policy(POLICY_PATH)
    assert policy.scoring.dry_run_score == 0.50
    assert policy.scoring.completed_score == 0.85
    assert policy.scoring.absent_score == 0.15


def test_pressure_channels() -> None:
    policy = load_feedback_policy(POLICY_PATH)
    assert "execution_pressure" in policy.pressure_channels
    assert policy.positive_delta_channels["agency_readiness"] == "increase"


def test_absence_rules() -> None:
    policy = load_feedback_policy(POLICY_PATH)
    assert policy.absence_rules["dry_run_needs_no_cortex_result"] is True
    assert policy.absence_rules["dispatch_read_only_requires_result"] is True
