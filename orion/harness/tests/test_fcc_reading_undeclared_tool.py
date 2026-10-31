"""A reading turn that ends on a call to a tool it was never given is resumed once (2026-10-31).

Live: Qwen3.8-Flash-Next read WebFetch's "Binary content ... also saved to /root/.claude/.../tool-results/
webfetch-....pdf", planned to pdftotext it, and wrote a Bash call. Reading turns only have WebFetch and
WebSearch, so Strata left the call as text, the CLI ended the turn on it, and the reading failed with
"Could not parse JSON object from LLM text: '<tool_call>...'" (4 of 57 readings).
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, List

import pytest

from orion.harness import fcc_motor as motor
from orion.harness.tests.test_fcc_motor_cancel import _FakeProc

BASH_CALL = ("<tool_call>\n<function=Bash>\n<parameter=command>\nls -la /root/.claude/projects/x/tool-results/\n"
             "</parameter>\n</function>\n</tool_call>")
ANSWER = '{"claim": "the paper reports excess oxygen", "could_not_read": "the PDF body"}'


def test_detector_names_only_undeclared_tools():
    tools = motor.READING_ONLY_TOOLS
    assert motor.undeclared_tool_call(BASH_CALL, tools) == "Bash"
    assert motor.undeclared_tool_call('<tool_call>{"name": "Read", "arguments": {}}', tools) == "Read"
    assert motor.undeclared_tool_call("<tool_call>\n<function=WebFetch>\n", tools) is None
    assert motor.undeclared_tool_call(ANSWER, tools) is None
    assert motor.undeclared_tool_call(None, tools) is None


def test_reading_argv_declares_exactly_the_reading_tools():
    argv = motor.build_claude_argv(claude_bin="claude", model_id="m", prompt="p", correlation_id="c",
                                   reading_only=True)
    assert argv[argv.index("--tools") + 1] == "WebFetch,WebSearch"
    resumed = motor.build_claude_argv(claude_bin="claude", model_id="m", prompt="note", correlation_id="c",
                                      reading_only=True, resume_session_id="s1")
    assert resumed[:5] == ["claude", "-p", "note", "--resume", "s1"]


def _result(text: str, session: str) -> str:
    return json.dumps({"type": "result", "result": text, "session_id": session})


async def _run(monkeypatch, outputs: List[str], *, reading_only: bool = True):
    spawned: List[tuple] = []

    async def fake_exec(*args: Any, **kwargs: Any) -> _FakeProc:
        spawned.append(args)
        return _FakeProc([outputs[len(spawned) - 1]], hang=False, returncode=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(motor, "_preflight_fcc_server", lambda *a, **k: None)
    monkeypatch.setattr(motor, "load_fcc_env", lambda _p: {"MODEL_CHAT": "x"})
    monkeypatch.setattr(motor, "label_to_claude_model_id", lambda *_a, **_k: "llamacpp/chat")
    monkeypatch.setattr(motor, "_maybe_render_mcp_config", lambda **_k: None)
    frames = [f async for f in motor.run_fcc_turn(
        prompt="read the source", correlation_id="corr-undeclared", workspace="/tmp",
        fcc_server_url="http://127.0.0.1:8082", auth_token="tok", claude_bin="claude", timeout_sec=5.0,
        reading_only=reading_only)]
    return frames, spawned


@pytest.mark.asyncio
async def test_undeclared_tool_call_is_resumed_once_with_the_note(monkeypatch):
    frames, spawned = await _run(monkeypatch, [_result(BASH_CALL, "s1"), _result(ANSWER, "s1")])
    finals = [f for f in frames if f["type"] == "final"]
    assert len(finals) == 1 and finals[0]["llm_response"] == ANSWER   # the bad final is never yielded
    assert finals[0]["metadata"]["fcc_undeclared_tool_resume"] == {"tool": "Bash", "session_id": "s1"}
    assert len(spawned) == 2
    second = list(spawned[1])
    assert second[second.index("--resume") + 1] == "s1"
    assert "Only WebFetch and WebSearch" in second[second.index("-p") + 1]
    assert second[second.index("--tools") + 1] == "WebFetch,WebSearch"


@pytest.mark.asyncio
async def test_a_second_undeclared_call_is_final_no_loop(monkeypatch):
    frames, spawned = await _run(monkeypatch, [_result(BASH_CALL, "s1"), _result(BASH_CALL, "s1")])
    assert len(spawned) == 2
    assert [f["llm_response"] for f in frames if f["type"] == "final"] == [BASH_CALL]


@pytest.mark.asyncio
async def test_only_reading_turns_are_resumed(monkeypatch):
    frames, spawned = await _run(monkeypatch, [_result(BASH_CALL, "s1")], reading_only=False)
    assert len(spawned) == 1
    assert [f["llm_response"] for f in frames if f["type"] == "final"] == [BASH_CALL]


@pytest.mark.asyncio
async def test_a_good_reading_answer_is_untouched(monkeypatch):
    frames, spawned = await _run(monkeypatch, [_result(ANSWER, "s1")])
    final = [f for f in frames if f["type"] == "final"][0]
    assert len(spawned) == 1 and final["llm_response"] == ANSWER
    assert "fcc_undeclared_tool_resume" not in final["metadata"]
