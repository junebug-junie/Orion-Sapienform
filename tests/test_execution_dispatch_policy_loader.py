from pathlib import Path

from orion.execution_dispatch.policy import load_execution_dispatch_policy

REPO = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO / "config" / "execution_dispatch" / "execution_dispatch_policy.v1.yaml"


def test_loads_yaml() -> None:
    policy = load_execution_dispatch_policy(POLICY_PATH)
    assert policy.schema_version == "execution_dispatch_policy.v1"


def test_default_mode_dry_run() -> None:
    policy = load_execution_dispatch_policy(POLICY_PATH)
    assert policy.mode.default_dispatch_mode == "dry_run"


def test_allow_dispatch_read_only_false() -> None:
    policy = load_execution_dispatch_policy(POLICY_PATH)
    assert policy.mode.allow_dispatch_read_only is False


def test_allow_mutating_dispatch_false() -> None:
    policy = load_execution_dispatch_policy(POLICY_PATH)
    assert policy.mode.allow_mutating_dispatch is False


def test_routes_for_inspect_summarize_observe() -> None:
    policy = load_execution_dispatch_policy(POLICY_PATH)
    assert "inspect" in policy.proposal_kind_to_cortex
    assert "summarize" in policy.proposal_kind_to_cortex
    assert "observe" in policy.proposal_kind_to_cortex
    assert policy.proposal_kind_to_cortex["inspect"].cortex_verb == "substrate.inspect"


def test_hard_blocks_include_destructive_classes() -> None:
    policy = load_execution_dispatch_policy(POLICY_PATH)
    for token in (
        "destructive_action",
        "file_write",
        "network_call",
        "service_restart",
        "settings_mutation",
        "approved_for_execution",
        "prepare_action",
    ):
        assert token in policy.hard_blocks
