from __future__ import annotations

from pathlib import Path

import pytest

from orion.harness.operator_brief import HARNESS_MOTOR_MAX_READ_LINES, is_relational_motor_stance
from orion.harness.prefix import compile_harness_prefix, harness_motor_instruction
from orion.harness.tests.fixtures import make_grounding_capsule, make_thought
from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.harness_finalize import HarnessRepairOverlayV1
from orion.schemas.thought import AutonomySliceV1, StanceHarnessSliceV1


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


def test_compile_harness_prefix_includes_prior_tool_fetch_line() -> None:
    thought = make_thought(imperative="Inspect the module.", tone="direct")
    prompt = compile_harness_prefix(
        thought,
        repair_overlay=HarnessRepairOverlayV1(),
        user_message="what did you fetch?",
        prior_tool_fetch_names=["mcp__github__get_file_contents", "WebFetch"],
    )
    assert "Last turn you fetched content via tool: mcp__github__get_file_contents, WebFetch" in prompt


def test_compile_harness_prefix_omits_prior_tool_fetch_line_when_none() -> None:
    thought = make_thought(imperative="Inspect the module.", tone="direct")
    prompt = compile_harness_prefix(
        thought,
        repair_overlay=HarnessRepairOverlayV1(),
        user_message="hello",
    )
    assert "Last turn you fetched content via tool" not in prompt


def test_compile_harness_prefix_includes_autonomy_slice_recent_actions() -> None:
    """Regression for the stance_react dispatch-evidence patch: the FCC
    motor's own prefix must render recent_actions directly, not just leave
    it to the upstream stance LLM's imperative/tone. A prior version of
    _format_autonomy_slice only emitted dominant_drive/active_tensions/
    pressure_trend and silently dropped recent_actions."""
    thought = make_thought(
        imperative="Inspect the module.",
        tone="direct",
        autonomy_slice=AutonomySliceV1(
            dominant_drive="coherence",
            recent_actions=["inspect: checked substrate graph health"],
        ),
    )
    prompt = compile_harness_prefix(
        thought,
        repair_overlay=HarnessRepairOverlayV1(),
        user_message="what have you been doing?",
    )
    assert "Recent actions: inspect: checked substrate graph health" in prompt


def test_compile_harness_prefix_includes_context_provenance_when_capsule_has_it() -> None:
    """The motor has its own tool access (e.g. GitHub MCP file reads) -- the
    provenance legend has to reach this literal prompt, not just live on an
    unread GroundingCapsuleV1 field, or a tool-fetched read can still get
    narrated as live substrate computation. See
    project_orion_substrate_bridge_confabulation for the incident this closes.
    """
    capsule = make_grounding_capsule(
        context_provenance={
            "self_state": "live_runtime_projection",
            "attention_broadcast": "live_runtime_projection",
            "recall_bundle": "memory_recall",
        }
    )
    thought = make_thought(imperative="Inspect the module.", tone="direct", grounding_capsule=capsule)
    prompt = compile_harness_prefix(
        thought,
        repair_overlay=HarnessRepairOverlayV1(),
        user_message="what's live right now?",
    )
    assert "CONTEXT PROVENANCE" in prompt
    assert "live now: attention_broadcast, self_state" in prompt
    assert "retrieved memory: recall_bundle" in prompt


def test_compile_harness_prefix_omits_context_provenance_when_capsule_empty() -> None:
    capsule = make_grounding_capsule(context_provenance={})
    thought = make_thought(imperative="Inspect the module.", tone="direct", grounding_capsule=capsule)
    prompt = compile_harness_prefix(
        thought,
        repair_overlay=HarnessRepairOverlayV1(),
        user_message="hi",
    )
    assert "CONTEXT PROVENANCE" not in prompt


def test_compile_harness_prefix_omits_recent_actions_line_when_empty() -> None:
    thought = make_thought(
        imperative="Inspect the module.",
        tone="direct",
        autonomy_slice=AutonomySliceV1(dominant_drive="coherence"),
    )
    prompt = compile_harness_prefix(
        thought,
        repair_overlay=HarnessRepairOverlayV1(),
        user_message="what have you been doing?",
    )
    assert "Recent actions:" not in prompt


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


def test_compile_harness_prefix_includes_self_index_briefs_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HARNESS_FCC_MCP_ENABLED", "true")
    monkeypatch.setenv("ORION_GITHUB_OWNER", "junebug-junie")
    monkeypatch.setenv("ORION_GITHUB_REPO", "Orion-Sapienform")
    monkeypatch.setenv("HARNESS_FCC_GITNEXUS_ENABLED", "true")
    monkeypatch.setenv("HARNESS_FCC_CONTEXT_MODE_ENABLED", "true")
    thought = make_thought(imperative="Trace the unified turn.")
    prompt = compile_harness_prefix(
        thought,
        repair_overlay=HarnessRepairOverlayV1(),
    )
    assert "GitNexus code-graph MCP is available" in prompt
    assert "call it before falling back to raw source search" in prompt
    assert "derived cache, never authority" in prompt
    assert "Always read the GitNexus status/context resource before any" in prompt
    assert "do not skip the status check, without" in prompt
    assert "Context Mode MCP is available" in prompt
    assert "ctx_search" in prompt


def test_compile_harness_prefix_self_index_briefs_gated_on_master_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the master MCP flag no MCP config is rendered, so the prompt
    must not advertise GitNexus/Context Mode tools that don't exist."""
    monkeypatch.delenv("HARNESS_FCC_MCP_ENABLED", raising=False)
    monkeypatch.setenv("HARNESS_FCC_GITNEXUS_ENABLED", "true")
    monkeypatch.setenv("HARNESS_FCC_CONTEXT_MODE_ENABLED", "true")
    thought = make_thought()
    prompt = compile_harness_prefix(
        thought,
        repair_overlay=HarnessRepairOverlayV1(),
    )
    assert "GitNexus" not in prompt
    assert "Context Mode MCP" not in prompt


def test_compile_harness_prefix_includes_context_mode_brief_in_hook_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hook mode (plugin-owned MCP server) advertises the same ctx_* brief,
    even with the standalone context-mode flag off."""
    monkeypatch.setenv("HARNESS_FCC_MCP_ENABLED", "true")
    monkeypatch.setenv("ORION_GITHUB_OWNER", "junebug-junie")
    monkeypatch.setenv("ORION_GITHUB_REPO", "Orion-Sapienform")
    monkeypatch.delenv("HARNESS_FCC_GITNEXUS_ENABLED", raising=False)
    monkeypatch.delenv("HARNESS_FCC_CONTEXT_MODE_ENABLED", raising=False)
    monkeypatch.setenv("HARNESS_FCC_CONTEXT_MODE_HOOKS_ENABLED", "true")
    thought = make_thought(imperative="Trace the unified turn.")
    prompt = compile_harness_prefix(
        thought,
        repair_overlay=HarnessRepairOverlayV1(),
    )
    assert "Context Mode MCP is available" in prompt
    assert "ctx_search" in prompt
    assert "GitNexus" not in prompt


def test_compile_harness_prefix_omits_context_mode_brief_when_both_modes_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HARNESS_FCC_MCP_ENABLED", "true")
    monkeypatch.setenv("ORION_GITHUB_OWNER", "junebug-junie")
    monkeypatch.setenv("ORION_GITHUB_REPO", "Orion-Sapienform")
    monkeypatch.delenv("HARNESS_FCC_GITNEXUS_ENABLED", raising=False)
    monkeypatch.delenv("HARNESS_FCC_CONTEXT_MODE_ENABLED", raising=False)
    monkeypatch.delenv("HARNESS_FCC_CONTEXT_MODE_HOOKS_ENABLED", raising=False)
    thought = make_thought()
    prompt = compile_harness_prefix(
        thought,
        repair_overlay=HarnessRepairOverlayV1(),
    )
    assert "Context Mode MCP" not in prompt


def test_compile_harness_prefix_hook_mode_brief_gated_on_master_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HARNESS_FCC_MCP_ENABLED", raising=False)
    monkeypatch.delenv("HARNESS_FCC_GITNEXUS_ENABLED", raising=False)
    monkeypatch.delenv("HARNESS_FCC_CONTEXT_MODE_ENABLED", raising=False)
    monkeypatch.setenv("HARNESS_FCC_CONTEXT_MODE_HOOKS_ENABLED", "true")
    thought = make_thought()
    prompt = compile_harness_prefix(
        thought,
        repair_overlay=HarnessRepairOverlayV1(),
    )
    assert "Context Mode MCP" not in prompt


def test_compile_harness_prefix_omits_self_index_briefs_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HARNESS_FCC_GITNEXUS_ENABLED", raising=False)
    monkeypatch.delenv("HARNESS_FCC_CONTEXT_MODE_ENABLED", raising=False)
    monkeypatch.delenv("HARNESS_FCC_CONTEXT_MODE_HOOKS_ENABLED", raising=False)
    thought = make_thought()
    prompt = compile_harness_prefix(
        thought,
        repair_overlay=HarnessRepairOverlayV1(),
    )
    assert "GitNexus" not in prompt
    assert "Context Mode MCP" not in prompt


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
