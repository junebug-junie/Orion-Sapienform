from __future__ import annotations

from orion.cognition.quality_evaluator import (
    detect_unverified_install_commands,
    should_block_unsupported_specifics,
    should_rewrite_for_instructional,
)


def test_pip_install_blocked_without_verified_finding() -> None:
    bad, reason = detect_unverified_install_commands("Run `pip install foobar` now.", set())
    assert bad is True
    assert reason == "unsupported_specific_pip_install"


def test_pip_allowed_when_verified_token_contains_command() -> None:
    ok, _ = detect_unverified_install_commands(
        "Run pip install foobar",
        {"verified:pip install foobar"},
    )
    assert ok is False


def test_should_block_respects_allow_unverified() -> None:
    blocked, _ = should_block_unsupported_specifics(
        "pip install x",
        findings_bundle={"findings": [], "missing_evidence": []},
        answer_contract={"allow_unverified_specifics": True},
    )
    assert blocked is False


def test_should_rewrite_instructional_on_unverified_pip_with_contract() -> None:
    should, reason = should_rewrite_for_instructional(
        "pip install numpy",
        "implementation_guide",
        request_text="debug my repo",
        grounding_mode="orion_repo_architecture",
        findings_bundle={"findings": []},
        answer_contract={"requires_repo_grounding": True},
    )
    assert should is True
    assert "pip" in reason
