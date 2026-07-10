from __future__ import annotations

import asyncio
from typing import Any, List
from unittest.mock import AsyncMock

import pytest

from orion.harness import fcc_motor as motor


class _FakeStream:
    def __init__(self, lines: List[bytes], *, hang: bool = False) -> None:
        self._lines = list(lines)
        self._idx = 0
        self._hang = hang

    async def readline(self) -> bytes:
        if self._hang:
            await asyncio.sleep(3600)
            return b""
        if self._idx >= len(self._lines):
            return b""
        line = self._lines[self._idx]
        self._idx += 1
        await asyncio.sleep(0)
        return line

    async def read(self) -> bytes:
        return b""


class _FakeProc:
    def __init__(self, stdout_lines: List[str], *, hang: bool = False, returncode: int = -15) -> None:
        self.stdout = _FakeStream(
            [ln.encode("utf-8") + b"\n" for ln in stdout_lines],
            hang=hang,
        )
        self.stderr = _FakeStream([])
        self.returncode = returncode
        self.killed = False

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        return self.returncode


def test_cancel_fcc_turn_kills_registered_process() -> None:
    proc = _FakeProc([], hang=True)
    motor._register_process("corr-cancel-1", proc)  # type: ignore[arg-type]
    try:
        assert motor.cancel_fcc_turn("corr-cancel-1") is True
        assert proc.killed is True
        # Second cancel arms pending (no live proc) — still True.
        assert motor.cancel_fcc_turn("corr-cancel-1") is True
        assert "corr-cancel-1" in motor._PENDING_CANCEL
    finally:
        motor._unregister_process("corr-cancel-1")


def test_cancel_before_register_kills_on_spawn() -> None:
    assert motor.cancel_fcc_turn("corr-pending-1") is True
    assert "corr-pending-1" in motor._PENDING_CANCEL
    proc = _FakeProc([], hang=True)
    motor._register_process("corr-pending-1", proc)  # type: ignore[arg-type]
    assert proc.killed is True
    assert "corr-pending-1" not in motor._PENDING_CANCEL
    assert "corr-pending-1" not in {t["correlation_id"] for t in motor.active_fcc_turns()}


@pytest.mark.asyncio
async def test_run_fcc_turn_registers_and_unregisters_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc = _FakeProc(
        ['{"type":"result","result":"ok","session_id":"s1"}'],
        hang=False,
        returncode=0,
    )

    async def fake_exec(*args: Any, **kwargs: Any) -> _FakeProc:
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(motor, "_preflight_fcc_server", lambda *a, **k: None)
    monkeypatch.setattr(motor, "load_fcc_env", lambda _p: {"MODEL_CHAT": "x"})
    monkeypatch.setattr(motor, "label_to_claude_model_id", lambda *_a, **_k: "llamacpp/chat")
    monkeypatch.setattr(motor, "_maybe_render_mcp_config", lambda **_k: None)

    events = []
    async for ev in motor.run_fcc_turn(
        prompt="hi",
        correlation_id="corr-reg-1",
        workspace="/tmp",
        fcc_server_url="http://127.0.0.1:8082",
        auth_token="tok",
        claude_bin="claude",
        timeout_sec=5.0,
    ):
        # Mid-turn the process must be registered.
        if not events:
            assert "corr-reg-1" in {t["correlation_id"] for t in motor.active_fcc_turns()}
        events.append(ev)

    assert events[-1]["type"] == "final"
    assert "corr-reg-1" not in {t["correlation_id"] for t in motor.active_fcc_turns()}
