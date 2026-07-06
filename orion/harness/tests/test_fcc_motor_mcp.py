from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, List

import pytest

from orion.harness import fcc_motor as motor


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
        self.stderr = _FakeStream([])
        self.returncode = returncode

    def kill(self) -> None:
        pass

    async def wait(self) -> int:
        return self.returncode


def _fake_fcc_env(_path: Any) -> dict[str, str]:
    return {"MODEL_HAIKU": "claude-haiku-test"}


@pytest.mark.asyncio
async def test_run_fcc_turn_adds_mcp_config_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_argv: list = []

    async def fake_exec(*args: Any, **kwargs: Any) -> _FakeProc:
        captured_argv.extend(args)
        return _FakeProc([
            '{"type":"result","result":"Done.","session_id":"s1"}',
        ])

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(motor, "_preflight_fcc_server", lambda *a, **k: None)
    monkeypatch.setattr(motor, "load_fcc_env", _fake_fcc_env)
    monkeypatch.setattr(motor, "_maybe_render_mcp_config", lambda **k: Path("/tmp/fake-mcp.json"))
    monkeypatch.setenv("HARNESS_FCC_MCP_ENABLED", "true")

    async for _ in motor.run_fcc_turn(
        prompt="hello",
        fcc_model_label="MODEL_HAIKU",
        correlation_id="corr-mcp",
        workspace="/tmp",
        fcc_server_url="http://127.0.0.1:8082",
        auth_token="tok",
        claude_bin="claude",
        timeout_sec=30.0,
    ):
        pass

    assert "--mcp-config" in captured_argv
    assert "/tmp/fake-mcp.json" in [str(x) for x in captured_argv]
    assert "--allowedTools" in captured_argv
    idx = captured_argv.index("--allowedTools")
    assert captured_argv[idx + 1] == "mcp__*"


@pytest.mark.asyncio
async def test_run_fcc_turn_omits_mcp_config_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_argv: list = []

    async def fake_exec(*args: Any, **kwargs: Any) -> _FakeProc:
        captured_argv.extend(args)
        return _FakeProc([
            '{"type":"result","result":"Done.","session_id":"s1"}',
        ])

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(motor, "_preflight_fcc_server", lambda *a, **k: None)
    monkeypatch.setattr(motor, "load_fcc_env", _fake_fcc_env)
    monkeypatch.delenv("HARNESS_FCC_MCP_ENABLED", raising=False)

    async for _ in motor.run_fcc_turn(
        prompt="hello",
        fcc_model_label="MODEL_HAIKU",
        correlation_id="corr-no-mcp",
        workspace="/tmp",
        fcc_server_url="http://127.0.0.1:8082",
        auth_token="tok",
        claude_bin="claude",
        timeout_sec=30.0,
    ):
        pass

    assert "--mcp-config" not in captured_argv
    assert "--allowedTools" not in captured_argv


@pytest.mark.asyncio
async def test_run_fcc_turn_surfaces_mcp_preflight_error_code(monkeypatch: pytest.MonkeyPatch) -> None:
    from orion.fcc.mcp_config import McpPreflightError

    def _raise_preflight(**_kwargs: Any) -> Path:
        raise McpPreflightError(error_code="fcc_mcp_github_missing", message="Missing GITHUB_PAT in FCC env")

    async def fake_exec(*args: Any, **kwargs: Any) -> _FakeProc:
        raise AssertionError("should not spawn")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(motor, "_preflight_fcc_server", lambda *a, **k: None)
    monkeypatch.setattr(motor, "load_fcc_env", _fake_fcc_env)
    monkeypatch.setattr(motor, "_maybe_render_mcp_config", _raise_preflight)
    monkeypatch.setenv("HARNESS_FCC_MCP_ENABLED", "true")

    events = []
    async for ev in motor.run_fcc_turn(
        prompt="hello",
        fcc_model_label="MODEL_HAIKU",
        correlation_id="corr-preflight",
        workspace="/tmp",
        fcc_server_url="http://127.0.0.1:8082",
        auth_token="tok",
        claude_bin="claude",
        timeout_sec=30.0,
    ):
        events.append(ev)

    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert events[0]["error_code"] == "fcc_mcp_github_missing"
