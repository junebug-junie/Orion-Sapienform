from __future__ import annotations

from orion.harness.evals.stance_scope_live_eval import (
    HUNT_TRIPWIRE,
    parse_tool_steps,
    score_run,
)

CORR = "f924c7b9-1c82-40d8-a6a2-5acb2edffbb3"
OTHER = "00000000-0000-0000-0000-000000000000"


def _line(corr: str, step: int, tool: str) -> str:
    return (
        "[ORION-HARNESS-GOV] 2026-09-28 23:36:39,000 - INFO - orion.harness.grammar_publish - "
        f"harness_grammar_step_published corr={corr} channel=orion:grammar:event "
        f"step={step} tool={tool} event_id=e-{step}"
    )


def test_parse_tool_steps_keeps_only_this_turn_in_step_order_without_none() -> None:
    log = "\n".join(
        [
            _line(CORR, 9, "mcp__orion-introspect__reading_results"),
            _line(OTHER, 3, "Bash"),
            _line(CORR, 7, "ToolSearch"),
            _line(CORR, 8, "none"),
            _line(CORR, 24, "Agent"),
            "unrelated line",
        ]
    )
    assert parse_tool_steps(log, CORR) == [
        "ToolSearch",
        "mcp__orion-introspect__reading_results",
        "Agent",
    ]


def test_score_run_flags_the_incident_shape_as_a_hunt() -> None:
    tools = ["ToolSearch"] + ["mcp__orion-introspect__reading_results"] * 6 + ["Agent"] + ["Bash"] * 19 + ["Read"] * 5
    result = score_run(tools, finished=False, reply_text="")
    assert result["introspect_calls"] == 6
    assert result["discovery_calls"] == 1
    assert result["other_tool_calls"] == 25
    assert result["hunt"] is True
    assert result["passed"] is False


def test_score_run_passes_a_focused_finished_turn() -> None:
    tools = ["ToolSearch", "mcp__orion-introspect__reading_results", "mcp__orion-introspect__reading_results"]
    result = score_run(tools, finished=True, reply_text="I read three GPU pieces.")
    assert result["hunt"] is False
    assert result["passed"] is True


def test_score_run_fails_when_the_tool_was_never_called() -> None:
    result = score_run(["Bash"], finished=True, reply_text="From memory, GPUs are fast.")
    assert result["passed"] is False


def test_hunt_tripwire_is_exclusive() -> None:
    assert score_run(["Bash"] * HUNT_TRIPWIRE, finished=True, reply_text="x")["hunt"] is False
    assert score_run(["Bash"] * (HUNT_TRIPWIRE + 1), finished=True, reply_text="x")["hunt"] is True
