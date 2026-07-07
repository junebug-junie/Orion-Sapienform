"""Regression: harness step summaries must carry tool_use names and tool_result bodies.

Root cause of the "orion dropped the PR title" bug (corr 9eb363c9): the FCC motor
drove list_pull_requests correctly (draft_len=70, grounded), but
`summarize_harness_step` collapsed the tool call to "[N] assistant" and the tool
result (the PR JSON) to "[N] user". The finalize/reflect pass then saw no evidence
of world-contact and ruled the grounded draft misaligned, so voice-finalize
confabulated "no GitHub MCP call succeeded".
"""

from __future__ import annotations

from orion.harness.fcc_motor import summarize_harness_step


def _step(raw: dict) -> dict:
    return {"type": raw.get("type"), "raw": raw}


def test_summarize_tool_use_carries_tool_name_and_args() -> None:
    step = _step(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "mcp__github__list_pull_requests",
                        "input": {"owner": "junebug-junie", "repo": "some-repo"},
                    }
                ]
            },
        }
    )
    summary = summarize_harness_step(step, index=1)
    assert "mcp__github__list_pull_requests" in summary
    assert "owner=junebug-junie" in summary


def test_summarize_tool_result_carries_body() -> None:
    body = '[{"number":854,"title":"fix(action-outcome): reconcile index ownership"}]'
    step = _step(
        {
            "type": "user",
            "message": {
                "content": [
                    {"type": "tool_result", "content": body, "tool_use_id": "t1"}
                ]
            },
        }
    )
    summary = summarize_harness_step(step, index=2)
    assert "tool_result" in summary
    assert "854" in summary
    assert "fix(action-outcome)" in summary


def test_summarize_tool_result_list_content_and_error_flag() -> None:
    step = _step(
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "is_error": True,
                        "content": [{"type": "text", "text": "boom"}],
                    }
                ]
            },
        }
    )
    summary = summarize_harness_step(step, index=3)
    assert "tool_result [error]" in summary
    assert "boom" in summary


def test_summarize_system_subtype() -> None:
    assert summarize_harness_step(
        _step({"type": "system", "subtype": "init"}), index=0
    ) == "[0] system init"
    assert summarize_harness_step(_step({"type": "system"}), index=0) == "[0] system"


def test_summarize_assistant_text_preserved() -> None:
    step = _step(
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "Checking repo."}]}}
    )
    assert summarize_harness_step(step, index=5) == "[5] assistant: Checking repo."


def test_pr_title_stream_receipts_carry_tool_evidence() -> None:
    """Replay the failing turn's stream: tool evidence must survive into summaries."""
    stream = [
        _step({"type": "system", "subtype": "init"}),
        _step(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "mcp__github__list_pull_requests",
                            "input": {"owner": "junebug-junie"},
                        }
                    ]
                },
            }
        ),
        _step(
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "content": '[{"number":854,"title":"fix(action-outcome): reconcile index ownership"}]',
                        }
                    ]
                },
            }
        ),
        _step({"type": "system", "subtype": "status"}),
        _step(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "fix(action-outcome): reconcile index ownership"}
                    ]
                },
            }
        ),
        _step(
            {"type": "result", "result": "fix(action-outcome): reconcile index ownership"}
        ),
    ]
    summaries = [summarize_harness_step(s, index=i) for i, s in enumerate(stream)]
    joined = "\n".join(summaries)
    # Tool call name survives.
    assert "mcp__github__list_pull_requests" in joined
    # The PR JSON (number + title) survives — the reflect pass can now see grounding.
    assert "854" in joined
    assert joined.count("fix(action-outcome)") >= 2
