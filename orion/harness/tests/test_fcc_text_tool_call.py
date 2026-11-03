"""An ordinary turn whose answer is a tool call written as text is resumed once, then cut short (2026-11-03).

Live 2026-10-31..11-03 (Strata, Qwen3.8-Flash-Next, `tool_call_recovery` on): 23 turns ended on a
`<tool_call><function=mcp__plugin_context-mode_context-mode__ctx_*>` block written as text. FCC runs the
CLI with ENABLE_TOOL_SEARCH=true, so an MCP tool's schema is only sent to the engine after ToolSearch
loads it; the model called ctx_* tools without loading them, Strata reads a call only for a declared
tool (its chat-engine `tool_calls_unread` counter read exactly 23), and the CLI ended the turn on the
text. Finalize then "repaired" the raw call into a summary of work that never ran (4bbaaf2a).

The drafts in strata_text_tool_call_drafts.json are four of those 23, copied from harness_turn_trace.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List
from unittest.mock import AsyncMock

import pytest

from orion.fcc.self_index_brief import context_mode_brief_lines
from orion.harness import fcc_motor as motor
from orion.harness.cut_short import CUT_SHORT_MARKER, TEXT_TOOL_CALL_CODE, is_cut_short_code
from orion.harness.runner import HarnessRunner
from orion.harness.tests.fixtures import make_thought
from orion.harness.tests.test_fcc_motor_cancel import _FakeProc
from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.context_exec import ContextExecPermissionV1
from orion.schemas.harness_finalize import HarnessRunRequestV1

from orion.harness import session_resume as _sr

_REAL_FIND_TRANSCRIPT = _sr.find_transcript
LIVE = json.loads((Path(__file__).parent / "strata_text_tool_call_drafts.json").read_text())
CTX_FILE = next(r["draft_text"] for r in LIVE if r["correlation_id"].startswith("4bbaaf2a"))
CTX_FILE_TOOL = "mcp__plugin_context-mode_context-mode__ctx_execute_file"


def test_detector_reads_every_live_draft():
    names = [motor.text_tool_call(r["draft_text"]) for r in LIVE]
    assert names == [
        CTX_FILE_TOOL,
        "mcp__plugin_context-mode_context-mode__ctx_batch_execute",
        "mcp__plugin_context-mode_context-mode__ctx_search",
        "mcp__plugin_context-mode_context-mode__ctx_execute",
    ]
    # Prose before the call still counts; prose after it, or a quote, does not.
    assert motor.text_tool_call("Now the look.\n\n" + CTX_FILE) == CTX_FILE_TOOL
    assert motor.text_tool_call(CTX_FILE + "\nand then I found X") is None
    assert motor.text_tool_call('{"claim": "Qwen emits <tool_call> blocks"}') is None
    assert motor.text_tool_call("") is None


def test_note_tells_the_model_to_load_an_mcp_tool_first():
    note = motor.text_tool_call_note(CTX_FILE_TOOL)
    assert f'"select:{CTX_FILE_TOOL}"' in note and "ToolSearch" in note and "did not run" in note
    other = motor.text_tool_call_note("Bash")
    assert "did not run" in other and "select:" not in other


def test_brief_says_ctx_tools_must_be_loaded_first(monkeypatch):
    monkeypatch.setenv("HARNESS_FCC_CONTEXT_MODE_HOOKS_ENABLED", "1")
    line = context_mode_brief_lines()[0]
    assert 'ToolSearch with "select:mcp__plugin_context-mode_context-mode__ctx_batch_execute,' in line


def _result(text: str, session: str) -> str:
    return json.dumps({"type": "result", "result": text, "session_id": session})


def _patch_common(monkeypatch) -> List[tuple]:
    monkeypatch.setattr(motor, "_preflight_fcc_server", lambda *a, **k: None)
    monkeypatch.setattr(motor, "load_fcc_env", lambda _p: {"MODEL_CHAT": "x"})
    monkeypatch.setattr(motor, "label_to_claude_model_id", lambda *_a, **_k: "llamacpp/chat")
    monkeypatch.setattr(motor, "_maybe_render_mcp_config", lambda **_k: None)
    from orion.harness import session_resume

    monkeypatch.setattr(session_resume, "find_transcript", lambda _home, _sid: Path(__file__))
    return []


async def _spawn_run(monkeypatch, outputs: List[str], **kwargs) -> tuple[list, list]:
    spawned = _patch_common(monkeypatch)

    async def fake_exec(*args: Any, **_kw: Any) -> _FakeProc:
        spawned.append(args)
        return _FakeProc([outputs[len(spawned) - 1]], hang=False, returncode=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    frames = [f async for f in motor.run_fcc_turn(
        prompt="what are your capacities", correlation_id="corr-text-call", workspace="/tmp",
        fcc_server_url="http://127.0.0.1:8082", auth_token="tok", claude_bin="claude",
        timeout_sec=kwargs.pop("timeout_sec", 300.0), **kwargs)]
    return frames, spawned


@pytest.mark.asyncio
async def test_live_draft_is_resumed_once_with_the_load_it_first_note(monkeypatch):
    frames, spawned = await _spawn_run(monkeypatch, [_result(CTX_FILE, "s1"), _result("TOP KEYS: flows", "s1")])
    finals = [f for f in frames if f["type"] == "final"]
    assert [f["llm_response"] for f in finals] == ["TOP KEYS: flows"]  # the raw call is never yielded
    record = finals[0]["metadata"]["fcc_text_tool_call_resume"]
    assert record["tool"] == CTX_FILE_TOOL and record["session_id"] == "s1"
    second = list(spawned[1])
    assert second[second.index("--resume") + 1] == "s1"
    assert f"select:{CTX_FILE_TOOL}" in second[second.index("-p") + 1]
    assert "--tools" not in second  # an ordinary turn keeps its full tool set


@pytest.mark.asyncio
async def test_a_second_text_call_is_final_no_loop(monkeypatch):
    frames, spawned = await _spawn_run(monkeypatch, [_result(CTX_FILE, "s1"), _result(CTX_FILE, "s1")])
    assert len(spawned) == 2
    assert [f["llm_response"] for f in frames if f["type"] == "final"] == [CTX_FILE]


@pytest.mark.asyncio
async def test_a_plain_answer_is_untouched(monkeypatch):
    frames, spawned = await _spawn_run(monkeypatch, [_result("All four capacities are mapped.", "s1")])
    final = [f for f in frames if f["type"] == "final"][0]
    assert len(spawned) == 1 and "fcc_text_tool_call_resume" not in final["metadata"]


@pytest.mark.asyncio
async def test_no_resume_when_the_deadline_is_nearly_spent(monkeypatch):
    frames, spawned = await _spawn_run(monkeypatch, [_result(CTX_FILE, "s1")], timeout_sec=5.0)
    assert len(spawned) == 1
    assert [f["llm_response"] for f in frames if f["type"] == "final"] == [CTX_FILE]


@pytest.mark.asyncio
async def test_no_resume_without_a_transcript_keeps_the_held_final(monkeypatch):
    """`claude --resume` would fail "No conversation found" and replace a usable held final."""
    from orion.harness import session_resume

    real_patch = _patch_common

    def patch_no_transcript(mp):
        out = real_patch(mp)
        mp.setattr(session_resume, "find_transcript", lambda _h, _s: None)
        return out

    # _spawn_run calls _patch_common; this test needs the lookup to find nothing.
    monkeypatch.setitem(globals(), "_patch_common", patch_no_transcript)
    frames, spawned = await _spawn_run(monkeypatch, [_result(CTX_FILE, "s1")])
    assert len(spawned) == 1
    final = [f for f in frames if f["type"] == "final"][0]
    assert final["llm_response"] == CTX_FILE
    assert final["metadata"]["fcc_text_tool_call_resume_skipped"]["reason"] == "no_transcript"


@pytest.mark.asyncio
async def test_a_resume_that_fails_on_its_own_keeps_the_held_final(monkeypatch):
    spawned = _patch_common(monkeypatch)

    async def fake_exec(*args: Any, **_kw: Any) -> _FakeProc:
        spawned.append(args)
        if len(spawned) == 1:
            return _FakeProc([_result(CTX_FILE, "s1")], hang=False, returncode=0)
        return _FakeProc([], hang=False, returncode=1)   # "No conversation found"

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    frames = [f async for f in motor.run_fcc_turn(
        prompt="p", correlation_id="corr-resume-fail", workspace="/tmp", fcc_server_url="http://127.0.0.1:8082",
        auth_token="tok", claude_bin="claude", timeout_sec=300.0)]
    assert len(spawned) == 2
    assert [f["type"] for f in frames if f["type"] in ("final", "error")] == ["final"]
    final = [f for f in frames if f["type"] == "final"][0]
    assert final["llm_response"] == CTX_FILE
    assert final["metadata"]["fcc_text_tool_call_resume_skipped"]["reason"] == "resume_failed"


@pytest.mark.asyncio
async def test_a_killed_resume_of_a_durable_run_stays_resumable(monkeypatch, tmp_path):
    """The session end is recorded on the resume's frame, not the held one: a SIGKILLed resume
    reads `interrupted`, so the durable retry picks the same session back up."""
    from orion.harness import session_resume as sr

    spawned = _patch_common(monkeypatch)
    # The real lookup: this test puts the transcript on disk.
    monkeypatch.setattr(sr, "find_transcript", _REAL_FIND_TRANSCRIPT)
    monkeypatch.setenv("HARNESS_FCC_CLAUDE_CONFIG_DIR", str(tmp_path))
    sid = sr.session_id_for("run-text-call")
    (tmp_path / "projects" / "p").mkdir(parents=True)
    (tmp_path / "projects" / "p" / f"{sid}.jsonl").write_text("{}\n")

    async def fake_exec(*args: Any, **_kw: Any) -> _FakeProc:
        spawned.append(args)
        if len(spawned) == 1:
            return _FakeProc([_result(CTX_FILE, sid)], hang=False, returncode=0)
        return _FakeProc([json.dumps({"type": "system", "subtype": "init", "session_id": sid})],
                         hang=False, returncode=-9)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    frames = [f async for f in motor.run_fcc_turn(
        prompt="p", correlation_id="corr-durable", workspace="/tmp", fcc_server_url="http://127.0.0.1:8082",
        auth_token="tok", claude_bin="claude", timeout_sec=300.0, resume_key="run-text-call")]
    assert len(spawned) == 2
    ends = [f for f in frames if f["type"] in ("final", "error")]
    assert len(ends) == 1 and ends[0]["type"] == "error"
    assert ends[0]["metadata"]["session_resume"]["ended"] == sr.STATE_INTERRUPTED
    assert sr.plan_attempt("run-text-call").resume is True


class _FakeWarmTurn:
    """Enough of fcc_warm_pool.WarmTurn for one turn: lines out, then EOF at the result."""

    def __init__(self, lines: List[str]) -> None:
        self._lines = [ln.encode() + b"\n" for ln in lines]
        self.prompts: List[str] = []
        self.released = False
        self.died_unexpectedly = False
        self.result_seen = False
        self.killable = self

    async def send_prompt(self, prompt: str) -> None:
        self.prompts.append(prompt)

    async def readline(self) -> bytes:
        if not self._lines:
            return b""
        line = self._lines.pop(0)
        self.result_seen = self.result_seen or b'"result"' in line
        return line

    def kill(self) -> None:
        pass

    async def wait(self) -> int:
        return 0

    def stderr_snippet(self) -> str:
        return ""

    async def release(self) -> None:
        self.released = True


@pytest.mark.asyncio
async def test_a_warm_chat_turn_is_resumed_by_a_spawn_after_the_slot_is_released(monkeypatch):
    """4bbaaf2a was a warm chat reply: the slot is released, its transcript is resumed by a spawn."""
    spawned = _patch_common(monkeypatch)
    warm = _FakeWarmTurn([_result(CTX_FILE, "warm-s1")])

    async def fake_exec(*args: Any, **_kw: Any) -> _FakeProc:
        spawned.append(args)
        assert warm.released, "resume must not start while the warm slot is still held"
        return _FakeProc([_result("TOP KEYS: flows", "warm-s1")], hang=False, returncode=0)

    async def fake_acquire(**_kw: Any):
        return warm, None

    mcp_calls: List[dict] = []
    env_calls: List[dict] = []
    real_env = motor._build_subprocess_env

    def fake_mcp(**kw: Any):
        mcp_calls.append(kw)
        return None

    def spy_env(**kw: Any):
        env_calls.append(kw)
        return real_env(**kw)

    monkeypatch.setattr(motor, "_maybe_render_mcp_config", fake_mcp)
    monkeypatch.setattr(motor, "_build_subprocess_env", spy_env)

    import orion.harness.fcc_warm_pool as wp

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(wp, "get_warm_pool", lambda: object())
    monkeypatch.setattr(motor, "_acquire_warm_turn", fake_acquire)
    frames = [f async for f in motor.run_fcc_turn(
        prompt="what are your capacities", correlation_id="corr-warm-text-call", workspace="/tmp",
        fcc_server_url="http://127.0.0.1:8082", auth_token="tok", claude_bin="claude", timeout_sec=300.0,
        chat_reply=True, reading_binding="RB")]
    assert mcp_calls == [{"correlation_id": "corr-warm-text-call", "reading_binding": "RB"}]
    assert env_calls[-1]["chat_reply"] is True and env_calls[-1]["correlation_id"] == "corr-warm-text-call"
    assert warm.prompts == ["what are your capacities"]
    assert len(spawned) == 1
    argv = list(spawned[0])
    assert argv[argv.index("--resume") + 1] == "warm-s1"
    finals = [f for f in frames if f["type"] == "final"]
    assert [f["llm_response"] for f in finals] == ["TOP KEYS: flows"]
    assert finals[0]["metadata"]["fcc_text_tool_call_resume"]["tool"] == CTX_FILE_TOOL
    assert "--append-system-prompt" not in argv  # the warm slot never had the anchor


# ---- runner: a text call that survives the resume never reaches repair -------------------------


def _step(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "step", "step": {"type": str(raw.get("type")), "raw": raw}}


def _request(corr: str) -> HarnessRunRequestV1:
    return HarnessRunRequestV1(
        correlation_id=corr,
        thought_event=make_thought(),
        user_message="what are your capacities",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )


def _bash_round(tool_id: str, result: str) -> List[Dict[str, Any]]:
    return [
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": tool_id, "name": "Bash", "input": {"command": "python -P -m orion.dig capacities"}}]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": tool_id, "content": result}]}},
    ]


@pytest.mark.asyncio
async def test_runner_turns_a_surviving_text_call_into_a_cut_short_draft():
    async def _runner(**_: Any) -> AsyncIterator[Dict[str, Any]]:
        for raw in _bash_round("t1", "S1 operations: camera:cam0, camera:doorbell"):
            yield _step(raw)
        yield _step({"type": "assistant", "message": {"content": [
            {"type": "text", "text": "Now the look.\n\n" + CTX_FILE}]}})
        yield {"type": "final", "llm_response": "Now the look.\n\n" + CTX_FILE, "metadata": {"exit_code": 0}}

    result = await HarnessRunner(AsyncMock(), fcc_runner=_runner).run(_request("c-text-call"))
    assert result.draft_text.startswith(CUT_SHORT_MARKER)
    assert "came back as plain text and never ran" in result.draft_text
    assert "camera:doorbell" in result.draft_text  # the real finding stays
    assert "<tool_call>" not in result.draft_text and "ctx_execute_file" not in result.draft_text
    assert result.cut_short_reason == TEXT_TOOL_CALL_CODE  # finalize skips response repair
    assert result.grounding_status == TEXT_TOOL_CALL_CODE and result.compliance_verdict == "partial"
    assert "Now the look." in result.draft_text  # prose before the call stays
    assert is_cut_short_code(TEXT_TOOL_CALL_CODE)


@pytest.mark.asyncio
async def test_runner_fails_a_text_call_with_nothing_recorded():
    async def _runner(**_: Any) -> AsyncIterator[Dict[str, Any]]:
        # Live shape: the CLI streams the text call as an assistant step before the final.
        yield _step({"type": "assistant", "message": {"content": [{"type": "text", "text": CTX_FILE}]}})
        yield {"type": "final", "llm_response": CTX_FILE, "metadata": {"exit_code": 0}}

    result = await HarnessRunner(AsyncMock(), fcc_runner=_runner).run(_request("c-text-call-empty"))
    assert result.draft_text == ""
    assert result.compliance_verdict == "failed"
    assert result.grounding_status == TEXT_TOOL_CALL_CODE


@pytest.mark.asyncio
async def test_runner_leaves_reading_turns_and_plain_answers_alone():
    async def _runner(**_: Any) -> AsyncIterator[Dict[str, Any]]:
        yield {"type": "final", "llm_response": CTX_FILE, "metadata": {"exit_code": 0}}

    reading = _request("c-reading").model_copy(update={"reading_only": True})
    result = await HarnessRunner(AsyncMock(), fcc_runner=_runner).run(reading)
    assert result.cut_short_reason is None and result.draft_text == CTX_FILE

    async def _plain(**_: Any) -> AsyncIterator[Dict[str, Any]]:
        yield {"type": "final", "llm_response": "Four capacities.", "metadata": {"exit_code": 0}}

    result = await HarnessRunner(AsyncMock(), fcc_runner=_plain).run(_request("c-plain"))
    assert result.cut_short_reason is None and result.draft_text == "Four capacities."
