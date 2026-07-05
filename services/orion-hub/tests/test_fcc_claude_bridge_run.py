from __future__ import annotations

import asyncio
from typing import Any, List

import pytest

from scripts import fcc_claude_bridge as bridge


class _FakeStream:
    def __init__(self, lines: List[bytes]) -> None:
        self._lines = list(lines)
        self._idx = 0

    async def readline(self) -> bytes:
        if self._idx >= len(self._lines):
            return b""
        line = self._lines[self._idx]
        self._idx += 1
        await asyncio.sleep(0)
        return line


class _FakeProc:
    def __init__(self, stdout_lines: List[str], returncode: int = 0) -> None:
        self.stdout = _FakeStream([ln.encode("utf-8") + b"\n" for ln in stdout_lines])
        self.returncode = returncode
        self._terminated = False

    def kill(self) -> None:
        self._terminated = True

    async def wait(self) -> int:
        return self.returncode


@pytest.mark.asyncio
async def test_run_turn_yields_steps_and_final(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = [
        '{"type":"assistant","message":{"content":[{"type":"text","text":"Hi"}]}}',
        '{"type":"result","result":"Done.","session_id":"s1","duration_ms":50}',
    ]

    async def fake_exec(*args: Any, **kwargs: Any) -> _FakeProc:
        return _FakeProc(lines)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(bridge, "_preflight_fcc_server", lambda *a, **k: None)

    events = []
    async for ev in bridge.run_turn(
        prompt="hello",
        fcc_model_label="MODEL_HAIKU",
        correlation_id="corr-1",
        workspace="/tmp",
        fcc_server_url="http://127.0.0.1:8082",
        auth_token="tok",
        claude_bin="claude",
        timeout_sec=30.0,
    ):
        events.append(ev)

    kinds = [e["type"] for e in events]
    assert kinds.count("step") >= 1
    assert kinds[-1] == "final"
    assert events[-1]["llm_response"] == "Done."
    assert events[-1]["metadata"]["claude_session_id"] == "s1"


@pytest.mark.asyncio
async def test_run_turn_passes_stream_read_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_exec(*args: Any, **kwargs: Any) -> _FakeProc:
        captured.update(kwargs)
        return _FakeProc([])

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(bridge, "_preflight_fcc_server", lambda *a, **k: None)

    async for _ in bridge.run_turn(
        prompt="hello",
        fcc_model_label="MODEL_HAIKU",
        correlation_id="corr-limit",
        workspace="/tmp",
        fcc_server_url="http://127.0.0.1:8082",
        auth_token="tok",
        claude_bin="claude",
        timeout_sec=30.0,
    ):
        pass

    assert captured.get("limit") == bridge.DEFAULT_STREAM_READ_LIMIT


@pytest.mark.asyncio
async def test_run_turn_limit_overrun_returns_partial(monkeypatch: pytest.MonkeyPatch) -> None:
    class _OverrunStream:
        async def readline(self) -> bytes:
            raise asyncio.LimitOverrunError("Separator is not found, and chunk exceed the limit", 70000)

    class _OverrunProc:
        stdout = _OverrunStream()

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return -9

    async def fake_exec(*args: Any, **kwargs: Any) -> _OverrunProc:
        return _OverrunProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(bridge, "_preflight_fcc_server", lambda *a, **k: None)

    events = []
    async for ev in bridge.run_turn(
        prompt="big file",
        fcc_model_label="MODEL_HAIKU",
        correlation_id="corr-overrun",
        workspace="/tmp",
        fcc_server_url="http://127.0.0.1:8082",
        auth_token="tok",
        claude_bin="claude",
        timeout_sec=30.0,
    ):
        events.append(ev)

    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert events[0]["error_code"] == "fcc_claude_stream_line_limit"


@pytest.mark.asyncio
async def test_build_subprocess_env_sets_claude_context_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import settings as hub_settings

    monkeypatch.setattr(hub_settings.settings, "HUB_AGENT_CLAUDE_MAX_CONTEXT_TOKENS", 65536, raising=False)
    monkeypatch.setattr(hub_settings.settings, "HUB_AGENT_CLAUDE_FILE_READ_MAX_TOKENS", 8192, raising=False)
    env = bridge._build_subprocess_env(fcc_server_url="http://127.0.0.1:8082", auth_token="tok")
    assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "65536"
    assert env["CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS"] == "8192"


def test_is_context_overflow_text_detects_llamacpp_error() -> None:
    sample = 'exceed_context_size_error n_ctx":65536'
    assert bridge.is_context_overflow_text(sample) is True
    assert bridge.is_context_overflow_text("all good") is False


@pytest.mark.asyncio
async def test_cancel_turn_sigterms_active(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = _FakeProc([], returncode=-15)

    async def fake_exec(*args: Any, **kwargs: Any) -> _FakeProc:
        bridge._register_process("corr-x", proc)  # type: ignore[attr-defined]
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    bridge._register_process("corr-x", proc)  # type: ignore[arg-type]
    assert await bridge.cancel_turn("corr-x") is True
    assert proc._terminated is True
