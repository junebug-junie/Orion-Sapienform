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


def test_allow_dispatch_read_only_true() -> None:
    # P1 of the motor-nerve spec: this gate is open, but EXECUTION_DISPATCH_MODE
    # still defaults to dry_run (test_default_mode_dry_run above), so real
    # sends require both gates opened, not this one alone.
    policy = load_execution_dispatch_policy(POLICY_PATH)
    assert policy.mode.allow_dispatch_read_only is True


def test_allow_mutating_dispatch_true_and_the_loader_carries_it() -> None:
    """Was `..._false`. Flipped 2026-08-12 by explicit operator decision, so
    this asserts the new intended position rather than being deleted.

    What the loader guarantees is only that the flag round-trips; what keeps
    the flag narrow lives in `tests/test_maintenance_dispatch_gating.py`
    (exactly one `maintenance_bounded` route) and in the policy's own
    `default_dispatch_mode: dry_run`, which still means an env-unset
    deployment cannot mutate.
    """
    policy = load_execution_dispatch_policy(POLICY_PATH)
    assert policy.mode.allow_mutating_dispatch is True
    assert policy.mode.default_dispatch_mode == "dry_run"


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
