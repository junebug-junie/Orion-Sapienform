from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, patch

import pytest

from orion.harness.runner import HarnessMotorResult, HarnessRunner, build_harness_prompt
from orion.harness.tests.fixtures import make_thought
from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.context_exec import ContextExecPermissionV1
from orion.schemas.harness_finalize import GrammarReceiptV1, HarnessRunRequestV1


async def _mock_fcc_runner(**_: Any) -> AsyncIterator[dict[str, Any]]:
    yield {
        "type": "step",
        "step": {
            "type": "assistant",
            "raw": {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "tool_use", "name": "Read", "input": {}},
                    ]
                },
            },
        },
    }
    yield {"type": "final", "llm_response": "internal draft answer", "metadata": {"exit_code": 0}}


@pytest.mark.asyncio
async def test_harness_runner_surfaces_fcc_error_code() -> None:
    async def _error_runner(**_: Any) -> AsyncIterator[dict[str, Any]]:
        yield {
            "type": "error",
            "error": "fcc turn timed out after 120.0s",
            "error_code": "fcc_timeout",
        }

    request = HarnessRunRequestV1(
        correlation_id="c-timeout",
        thought_event=make_thought(),
        user_message="hello",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )
    runner = HarnessRunner(AsyncMock(), fcc_runner=_error_runner)
    result = await runner.run(request)

    assert result.draft_text == ""
    assert result.compliance_verdict == "failed"
    assert result.grounding_status == "fcc_timeout"


@pytest.mark.asyncio
async def test_harness_runner_collects_grammar_receipts_and_draft() -> None:
    thought = make_thought()
    request = HarnessRunRequestV1(
        correlation_id="c-runner",
        thought_event=thought,
        user_message="hello",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )
    bus = AsyncMock()
    runner = HarnessRunner(bus, fcc_runner=_mock_fcc_runner)

    result = await runner.run(request)

    assert result.draft_text == "internal draft answer"
    assert result.step_count == 1
    assert len(result.grammar_receipts) == 1
    assert result.grammar_receipts[0].tool_name == "Read"
    assert result.draft_molecule is not None
    assert result.draft_molecule.draft_text == "internal draft answer"
    assert result.compliance_verdict == "completed"


def _step_with_tool_result(*, is_error: bool, text: str) -> dict[str, Any]:
    return _step_with_tool_results([(is_error, text)])


def _step_with_tool_results(blocks: list[tuple[bool, str]]) -> dict[str, Any]:
    return {
        "type": "user",
        "raw": {
            "type": "user",
            "message": {
                "content": [
                    {"type": "tool_result", "is_error": is_error, "content": text}
                    for is_error, text in blocks
                ]
            },
        },
    }


@pytest.mark.asyncio
async def test_harness_runner_tool_failure_streak_counts_repeated_error_kind() -> None:
    """The real substrate for a stuck-loop signal: repeated is_error tool_result blocks
    inside otherwise-normal steps (NOT the once-per-turn fcc-subprocess `error` event)."""

    async def _repeated_denial_runner(**_: Any) -> AsyncIterator[dict[str, Any]]:
        for _ in range(3):
            yield {
                "type": "step",
                "step": _step_with_tool_result(
                    is_error=True, text="Permission to use Bash has been denied"
                ),
            }
        yield {"type": "final", "llm_response": "gave up", "metadata": {"exit_code": 0}}

    request = HarnessRunRequestV1(
        correlation_id="c-streak",
        thought_event=make_thought(),
        user_message="hello",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )
    result = await HarnessRunner(AsyncMock(), fcc_runner=_repeated_denial_runner).run(request)
    assert result.tool_failure_streak_max == 3


@pytest.mark.asyncio
async def test_harness_runner_tool_failure_streak_counts_multiple_errors_within_one_step() -> None:
    """Parallel tool calls in a single assistant turn can produce multiple is_error
    tool_result blocks in ONE step's content array -- the streak must count each,
    not just the once-per-step case the other tests exercise."""

    async def _parallel_denial_runner(**_: Any) -> AsyncIterator[dict[str, Any]]:
        yield {
            "type": "step",
            "step": _step_with_tool_results(
                [
                    (True, "Permission to use Bash has been denied"),
                    (True, "Permission to use Write has been denied"),
                ]
            ),
        }
        yield {"type": "final", "llm_response": "gave up", "metadata": {"exit_code": 0}}

    request = HarnessRunRequestV1(
        correlation_id="c-streak-parallel",
        thought_event=make_thought(),
        user_message="hello",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )
    result = await HarnessRunner(AsyncMock(), fcc_runner=_parallel_denial_runner).run(request)
    assert result.tool_failure_streak_max == 2


@pytest.mark.asyncio
async def test_harness_runner_tool_failure_streak_resets_on_different_error_kind() -> None:
    async def _alternating_runner(**_: Any) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "step", "step": _step_with_tool_result(is_error=True, text="Permission denied")}
        yield {"type": "step", "step": _step_with_tool_result(is_error=True, text="Timeout waiting for response")}
        yield {"type": "step", "step": _step_with_tool_result(is_error=True, text="Permission denied")}
        yield {"type": "final", "llm_response": "gave up", "metadata": {"exit_code": 0}}

    request = HarnessRunRequestV1(
        correlation_id="c-streak-reset",
        thought_event=make_thought(),
        user_message="hello",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )
    result = await HarnessRunner(AsyncMock(), fcc_runner=_alternating_runner).run(request)
    assert result.tool_failure_streak_max == 1


@pytest.mark.asyncio
async def test_harness_runner_accumulates_step_char_stats() -> None:
    async def _verbose_runner(**_: Any) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "step", "step": _step_with_tool_result(is_error=False, text="x" * 100)}
        yield {"type": "step", "step": _step_with_tool_result(is_error=False, text="y" * 300)}
        yield {"type": "final", "llm_response": "done", "metadata": {"exit_code": 0}}

    request = HarnessRunRequestV1(
        correlation_id="c-verbose",
        thought_event=make_thought(),
        user_message="hello",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )
    result = await HarnessRunner(AsyncMock(), fcc_runner=_verbose_runner).run(request)
    assert result.step_char_sum == 400
    assert result.step_char_max == 300


@pytest.mark.asyncio
async def test_harness_runner_error_path_does_not_use_tool_result_streak() -> None:
    """The once-per-turn fcc-subprocess `error` branch is structurally separate from
    the per-step tool_result is_error streak -- confirms they don't get conflated."""

    async def _subprocess_error_runner(**_: Any) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "step", "step": _step_with_tool_result(is_error=False, text="ok")}
        yield {"type": "error", "error": "fcc crashed", "error_code": "fcc_crash"}

    request = HarnessRunRequestV1(
        correlation_id="c-subprocess-error",
        thought_event=make_thought(),
        user_message="hello",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )
    result = await HarnessRunner(AsyncMock(), fcc_runner=_subprocess_error_runner).run(request)
    assert result.tool_failure_streak_max == 0
    assert result.compliance_verdict == "failed"


def _result_step(*, output_tokens: int) -> dict[str, Any]:
    return {
        "type": "result",
        "raw": {
            "type": "result",
            "usage": {"input_tokens": 10, "output_tokens": output_tokens},
            "result": "done",
        },
    }


@pytest.mark.asyncio
async def test_harness_runner_captures_real_output_tokens_from_result_event() -> None:
    async def _tokened_runner(**_: Any) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "step", "step": _step_with_tool_result(is_error=False, text="ok")}
        yield {"type": "step", "step": _result_step(output_tokens=42)}
        yield {"type": "final", "llm_response": "done", "metadata": {"exit_code": 0}}

    request = HarnessRunRequestV1(
        correlation_id="c-tokens",
        thought_event=make_thought(),
        user_message="hello",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )
    result = await HarnessRunner(AsyncMock(), fcc_runner=_tokened_runner).run(request)
    assert result.reasoning_output_tokens == 42


@pytest.mark.asyncio
async def test_harness_runner_ignores_non_result_step_for_output_tokens() -> None:
    """A "step"-type event (not "result") must never be mistaken for the CLI's own
    end-of-turn usage summary, even if it happened to carry a similarly-shaped dict."""

    async def _decoy_runner(**_: Any) -> AsyncIterator[dict[str, Any]]:
        yield {
            "type": "step",
            "step": {
                "type": "assistant",
                "raw": {"type": "assistant", "usage": {"output_tokens": 999}},
            },
        }
        yield {"type": "final", "llm_response": "done", "metadata": {"exit_code": 0}}

    request = HarnessRunRequestV1(
        correlation_id="c-decoy",
        thought_event=make_thought(),
        user_message="hello",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )
    result = await HarnessRunner(AsyncMock(), fcc_runner=_decoy_runner).run(request)
    assert result.reasoning_output_tokens == 0


def _tool_use_step(tool_name: str) -> dict[str, Any]:
    return {
        "type": "assistant",
        "raw": {
            "type": "assistant",
            "message": {"content": [{"type": "tool_use", "name": tool_name, "input": {}}]},
        },
    }


@pytest.mark.asyncio
async def test_harness_runner_classifies_step_tool_kinds() -> None:
    async def _mixed_runner(**_: Any) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "step", "step": _tool_use_step("Read")}
        yield {"type": "step", "step": _tool_use_step("mcp__gitnexus__cypher")}
        yield {"type": "step", "step": _tool_use_step("Bash")}
        yield {"type": "step", "step": _tool_use_step("SomeUnlistedMcpTool")}
        yield {"type": "final", "llm_response": "done", "metadata": {"exit_code": 0}}

    request = HarnessRunRequestV1(
        correlation_id="c-mixed",
        thought_event=make_thought(),
        user_message="hello",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )
    result = await HarnessRunner(AsyncMock(), fcc_runner=_mixed_runner).run(request)
    assert result.context_gathering_step_count == 2
    assert result.execution_step_count == 1


async def _mock_fcc_runner_fetch_then_confabulate(**_: Any) -> AsyncIterator[dict[str, Any]]:
    yield {
        "type": "step",
        "step": {
            "type": "assistant",
            "raw": {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "tool_use", "name": "mcp__github__get_file_contents", "input": {}},
                    ]
                },
            },
        },
    }
    yield {
        "type": "final",
        "llm_response": "This is computing in the background right now.",
        "metadata": {"exit_code": 0},
    }


@pytest.mark.asyncio
async def test_harness_runner_flags_tool_provenance_mismatch() -> None:
    thought = make_thought()
    request = HarnessRunRequestV1(
        correlation_id="c-mismatch",
        thought_event=thought,
        user_message="what's live right now?",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )
    bus = AsyncMock()
    runner = HarnessRunner(bus, fcc_runner=_mock_fcc_runner_fetch_then_confabulate)

    result = await runner.run(request)

    assert result.tool_provenance_audit is not None
    assert "mcp__github__get_file_contents" in result.tool_provenance_audit
    assert result.draft_molecule is not None
    assert result.draft_molecule.tool_provenance_audit == result.tool_provenance_audit
    # Not a motor failure -- grounding_status/compliance_verdict untouched.
    assert result.compliance_verdict == "completed"
    assert result.grounding_status == "grounded"


@pytest.mark.asyncio
async def test_harness_runner_publishes_last_tool_fetch_on_fetch_tool_use() -> None:
    thought = make_thought(session_id="sess-cross-turn")
    request = HarnessRunRequestV1(
        correlation_id="c-cross-turn",
        thought_event=thought,
        user_message="what's live right now?",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )
    bus = AsyncMock()
    runner = HarnessRunner(bus, fcc_runner=_mock_fcc_runner_fetch_then_confabulate)
    publish_mock = AsyncMock()

    with patch("orion.harness.runner.publish_last_tool_fetch", publish_mock):
        await runner.run(request)

    publish_mock.assert_awaited_once_with(
        bus,
        session_id="sess-cross-turn",
        correlation_id="c-cross-turn",
        tool_names=["mcp__github__get_file_contents"],
    )


@pytest.mark.asyncio
async def test_harness_runner_does_not_publish_last_tool_fetch_when_no_fetch_tool() -> None:
    thought = make_thought()
    request = HarnessRunRequestV1(
        correlation_id="c-no-fetch",
        thought_event=thought,
        user_message="hello",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )
    bus = AsyncMock()
    runner = HarnessRunner(bus, fcc_runner=_mock_fcc_runner)
    publish_mock = AsyncMock()

    with patch("orion.harness.runner.publish_last_tool_fetch", publish_mock):
        await runner.run(request)

    publish_mock.assert_not_awaited()


async def _mock_fcc_runner_fetch_then_error_with_partial(**_: Any) -> AsyncIterator[dict[str, Any]]:
    yield {
        "type": "step",
        "step": {
            "type": "assistant",
            "raw": {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "tool_use", "name": "mcp__github__get_file_contents", "input": {}},
                    ]
                },
            },
        },
    }
    yield {
        "type": "error",
        "llm_response": "partial text salvaged before the timeout",
        "error_code": "fcc_timeout",
        "error": "fcc turn timed out after 120.0s",
    }


@pytest.mark.asyncio
async def test_harness_runner_does_not_publish_last_tool_fetch_on_partial_draft_error() -> None:
    """Regression: a turn that used a fetch tool but then errored/timed out
    with only a partial draft must NOT record a tool-fetch for continuity --
    the `if not draft_text` early return doesn't catch this case (draft_text
    is non-empty), so the exclusion has to be explicit."""
    thought = make_thought(session_id="sess-partial-error")
    request = HarnessRunRequestV1(
        correlation_id="c-partial-error",
        thought_event=thought,
        user_message="what's in the file?",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )
    bus = AsyncMock()
    runner = HarnessRunner(bus, fcc_runner=_mock_fcc_runner_fetch_then_error_with_partial)
    publish_mock = AsyncMock()

    with patch("orion.harness.runner.publish_last_tool_fetch", publish_mock):
        result = await runner.run(request)

    assert result.draft_text  # partial draft is non-empty, confirming this exercises the right branch
    assert result.compliance_verdict == "partial"
    publish_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_harness_runner_no_tool_provenance_mismatch_on_normal_turn() -> None:
    thought = make_thought()
    request = HarnessRunRequestV1(
        correlation_id="c-normal",
        thought_event=thought,
        user_message="hello",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )
    bus = AsyncMock()
    runner = HarnessRunner(bus, fcc_runner=_mock_fcc_runner)

    result = await runner.run(request)

    assert result.tool_provenance_audit is None
    assert result.draft_molecule.tool_provenance_audit is None


@pytest.mark.asyncio
async def test_harness_runner_publishes_lifecycle_grammar_on_motor_run() -> None:
    thought = make_thought()
    request = HarnessRunRequestV1(
        correlation_id="c-lifecycle",
        thought_event=thought,
        user_message="hello",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )
    bus = AsyncMock()
    runner = HarnessRunner(bus, fcc_runner=_mock_fcc_runner, node_name="test-node")
    lifecycle_publish = AsyncMock()

    with patch(
        "orion.harness.runner.publish_harness_lifecycle_grammar",
        lifecycle_publish,
    ):
        result = await runner.run(request)

    assert result.step_count > 0
    lifecycle_publish.assert_awaited_once()
    events = lifecycle_publish.await_args.kwargs["events"]
    roles = {e.atom.semantic_role for e in events if e.atom}
    assert "exec_request_received" in roles
    assert "exec_plan_started" in roles
    assert "exec_step_started" in roles
    assert "exec_step_completed" in roles
    assert "exec_result_assembled" in roles
    assembled = next(
        e for e in events if e.atom and e.atom.semantic_role == "exec_result_assembled"
    )
    assert "reasoning_present=True" in assembled.atom.summary
    assert "thinking_source=harness_fcc" in assembled.atom.summary


@pytest.mark.asyncio
async def test_harness_runner_uses_compile_harness_prefix() -> None:
    thought = make_thought(imperative="Check logs first.", tone="direct")
    captured: dict[str, str] = {}

    async def _capture_prompt(*, prompt: str, **__: Any) -> AsyncIterator[dict[str, Any]]:
        captured["prompt"] = prompt
        yield {"type": "final", "llm_response": "done", "metadata": {"exit_code": 0}}

    request = HarnessRunRequestV1(
        correlation_id="c-prefix",
        thought_event=thought,
        user_message="what broke?",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )
    runner = HarnessRunner(AsyncMock(), fcc_runner=_capture_prompt)
    await runner.run(request)

    prompt = captured["prompt"]
    assert "Imperative: Check logs first." in prompt
    assert "Tone: direct" in prompt
    assert "what broke?" in prompt
    assert "Execute your imperative" in prompt


@pytest.mark.asyncio
async def test_harness_runner_reads_prior_tool_fetch_keyed_by_session_id() -> None:
    """Cross-turn continuity: run() must call read_last_tool_fetch with
    thought.session_id -- the same value the write side (publish_last_tool_fetch)
    keys by -- not any cross-service correlation id."""
    thought = make_thought(session_id="sess-continuity")
    request = HarnessRunRequestV1(
        correlation_id="c-continuity",
        thought_event=thought,
        user_message="what did you find?",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )
    read_mock = AsyncMock(return_value=None)
    runner = HarnessRunner(AsyncMock(), fcc_runner=_mock_fcc_runner)

    with patch("orion.harness.runner.read_last_tool_fetch", read_mock):
        await runner.run(request)

    read_mock.assert_awaited_once_with(runner.bus, session_id="sess-continuity")


@pytest.mark.asyncio
async def test_harness_runner_renders_prior_tool_fetch_in_compiled_prompt() -> None:
    thought = make_thought(session_id="sess-continuity-2")
    captured: dict[str, str] = {}

    async def _capture_prompt(*, prompt: str, **__: Any) -> AsyncIterator[dict[str, Any]]:
        captured["prompt"] = prompt
        yield {"type": "final", "llm_response": "done", "metadata": {"exit_code": 0}}

    request = HarnessRunRequestV1(
        correlation_id="c-continuity-2",
        thought_event=thought,
        user_message="what did you find?",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )
    read_mock = AsyncMock(
        return_value={"tool_names": ["mcp__github__get_file_contents"], "correlation_id": "c-prev", "at": "x"}
    )
    runner = HarnessRunner(AsyncMock(), fcc_runner=_capture_prompt)

    with patch("orion.harness.runner.read_last_tool_fetch", read_mock):
        await runner.run(request)

    assert "Last turn you fetched content via tool: mcp__github__get_file_contents" in captured["prompt"]


@pytest.mark.asyncio
async def test_harness_runner_omits_prior_tool_fetch_line_when_none_found() -> None:
    thought = make_thought(session_id="sess-fresh")
    captured: dict[str, str] = {}

    async def _capture_prompt(*, prompt: str, **__: Any) -> AsyncIterator[dict[str, Any]]:
        captured["prompt"] = prompt
        yield {"type": "final", "llm_response": "done", "metadata": {"exit_code": 0}}

    request = HarnessRunRequestV1(
        correlation_id="c-fresh",
        thought_event=thought,
        user_message="hello",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )
    runner = HarnessRunner(AsyncMock(), fcc_runner=_capture_prompt)

    with patch("orion.harness.runner.read_last_tool_fetch", AsyncMock(return_value=None)):
        await runner.run(request)

    assert "Last turn you fetched content via tool" not in captured["prompt"]
