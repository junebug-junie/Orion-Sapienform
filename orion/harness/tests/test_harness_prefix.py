from __future__ import annotations

from pathlib import Path

import pytest

from orion.harness.operator_brief import HARNESS_MOTOR_MAX_READ_LINES, is_relational_motor_stance
from orion.harness.prefix import compile_harness_prefix, harness_motor_instruction
from orion.harness.tests.fixtures import make_thought
from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.harness_finalize import HarnessRepairOverlayV1
from orion.schemas.thought import StanceHarnessSliceV1


def test_compile_harness_prefix_includes_stance_slice() -> None:
    thought = make_thought(imperative="Inspect the module.", tone="direct")
    prompt = compile_harness_prefix(
        thought,
        repair_overlay=HarnessRepairOverlayV1(),
        user_message="how does coalition work?",
    )
    assert "Task mode: direct_response" in prompt
    assert "Answer strategy: direct" in prompt
    assert "how does coalition work?" in prompt


def test_compile_harness_prefix_imperative_first_unified_brief() -> None:
    thought = make_thought(
        imperative="Search orion/thought for stance_react; cite file paths.",
        tone="direct",
    )
    prompt = compile_harness_prefix(
        thought,
        repair_overlay=HarnessRepairOverlayV1(),
        user_message="implement a function",
    )
    assert "Tools are available from the start" in prompt
    assert "Search orion/thought for stance_react" in prompt
    assert "Imperative: Search orion/thought" in prompt
    assert "Requires repo grounding" not in prompt
    assert "Answer contract:" not in prompt
    assert "Orion harness motor — repo/technical turn" not in prompt  # old gated brief
    assert "prefer rg/Grep" in prompt  # unified brief includes tool guidance
    assert f"longer than {HARNESS_MOTOR_MAX_READ_LINES} lines" in prompt


def test_compile_harness_prefix_ignores_answer_contract() -> None:
    thought = make_thought()
    contract = AnswerContract(
        request_kind="repo_technical",
        requires_repo_grounding=True,
    )
    prompt = compile_harness_prefix(
        thought,
        repair_overlay=HarnessRepairOverlayV1(),
        answer_contract=contract,
    )
    assert "Requires repo grounding" not in prompt
    assert "Answer contract:" not in prompt


def test_compile_harness_prefix_includes_github_repo_when_mcp_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HARNESS_FCC_MCP_ENABLED", "true")
    monkeypatch.setenv("ORION_GITHUB_OWNER", "junebug-junie")
    monkeypatch.setenv("ORION_GITHUB_REPO", "Orion-Sapienform")
    thought = make_thought(imperative="List latest PR title.")
    prompt = compile_harness_prefix(
        thought,
        repair_overlay=HarnessRepairOverlayV1(),
    )
    assert "owner='junebug-junie'" in prompt
    assert "repo='Orion-Sapienform'" in prompt
    assert "perPage=1" in prompt
    assert "get_pull_request" in prompt
    assert "search_pull_requests" in prompt


def test_compile_harness_prefix_resolves_github_repo_from_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path("/mnt/scripts/Orion-Sapienform")
    if not (repo_root / ".git").is_dir():
        pytest.skip("workspace git metadata unavailable")
    monkeypatch.setenv("HARNESS_FCC_MCP_ENABLED", "true")
    monkeypatch.delenv("ORION_GITHUB_OWNER", raising=False)
    monkeypatch.delenv("ORION_GITHUB_REPO", raising=False)
    monkeypatch.delenv("HARNESS_FCC_WORKSPACE", raising=False)
    thought = make_thought(imperative="List latest PR title.")
    prompt = compile_harness_prefix(
        thought,
        repair_overlay=HarnessRepairOverlayV1(),
        workspace=str(repo_root),
    )
    assert "owner='junebug-junie'" in prompt
    assert "repo='Orion-Sapienform'" in prompt


def test_harness_motor_instruction_relational_discourages_tools() -> None:
    thought = make_thought(
        imperative="Stay present; acknowledge status — no task tracking.",
        stance_harness_slice=StanceHarnessSliceV1(
            task_mode="reflective_dialogue",
            conversation_frame="reflective",
            interaction_regime="minimal",
            answer_strategy="companion_presence",
        ),
    )
    assert is_relational_motor_stance(thought) is True
    instruction = harness_motor_instruction(thought=thought, answer_contract=None)
    assert "do NOT use GitHub MCP" in instruction
    assert "Execute your imperative" in instruction


def test_harness_motor_instruction_imperative_forward() -> None:
    thought = make_thought(imperative="Inspect docker logs for orion-hub.")
    instruction = harness_motor_instruction(thought=thought, answer_contract=None)
    assert "Execute your imperative" in instruction
    assert f"over {HARNESS_MOTOR_MAX_READ_LINES} lines" in instruction
    assert "rg/Grep" in instruction
