from __future__ import annotations

import json

import pytest

from orion.fcc.context_budget import (
    CONTEXT_PRESSURE_NUDGE,
    annotate_harness_step,
    apply_context_overflow_hint,
    build_context_pressure_step,
    context_risk_level,
    is_context_overflow_text,
    measure_step_payload_chars,
)
from orion.fcc.mcp_stdio_proxy import _maybe_truncate_line


def test_is_context_overflow_detects_prompt_too_long() -> None:
    assert is_context_overflow_text("Prompt is too long") is True
    assert is_context_overflow_text("exceed_context_size_error") is True


def test_apply_context_overflow_hint_appends_operator_guidance() -> None:
    out = apply_context_overflow_hint("Prompt is too long", n_ctx=65536)
    assert "get_pull_request" in out
    assert "perPage=1" in out


def test_measure_tool_result_step_chars() -> None:
    step = {
        "type": "user",
        "raw": {
            "type": "user",
            "message": {
                "content": [
                    {"type": "tool_result", "content": "x" * 20_000},
                ]
            },
        },
    }
    assert measure_step_payload_chars(step) == 20_000
    assert context_risk_level(accumulated_chars=40_000, step_chars=20_000) == "critical"


def test_annotate_harness_step_adds_context_obs() -> None:
    step = {"type": "assistant", "raw": {"type": "assistant", "message": {"content": []}}}
    annotated = annotate_harness_step(step, accumulated_chars=50_000)
    assert "context_obs" in annotated
    assert annotated["context_obs"]["risk"] in {"ok", "warn", "critical"}


def test_build_context_pressure_step_carries_nudge() -> None:
    step = build_context_pressure_step(fill_pct=72)
    assert step["raw"]["message"] == CONTEXT_PRESSURE_NUDGE
    assert step["raw"]["context_fill_pct"] == 72


def test_mcp_proxy_truncates_large_tool_text() -> None:
    huge = "a" * 20_000
    line = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [{"type": "text", "text": huge}],
            },
        }
    ) + "\n"
    out = _maybe_truncate_line(line, max_chars=12_000)
    parsed = json.loads(out.strip())
    text = parsed["result"]["content"][0]["text"]
    assert len(text) < len(huge)
    assert "orion-fcc-mcp-proxy" in text
