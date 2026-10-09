"""fcc_motor installs the repeat-failing-call breaker and logs when it fires."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from orion.fcc.repeat_failure_breaker import BLOCK_MARKER
from orion.harness import fcc_motor as motor
from orion.harness.tests.test_fcc_motor_mcp import _FakeProc, _fake_fcc_env


async def _run(monkeypatch: pytest.MonkeyPatch, lines: list[str], *, reading_only: bool = False) -> list:
    captured: list = []

    async def fake_exec(*args: Any, **kwargs: Any) -> _FakeProc:
        captured.extend(args)
        return _FakeProc(lines)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(motor, "_preflight_fcc_server", lambda *a, **k: None)
    monkeypatch.setattr(motor, "load_fcc_env", _fake_fcc_env)
    monkeypatch.setattr(motor, "_maybe_render_mcp_config", lambda **k: None)
    async for _ in motor.run_fcc_turn(
        prompt="hello",
        fcc_model_label="MODEL_HAIKU",
        correlation_id="corr-breaker",
        workspace="/tmp",
        fcc_server_url="http://127.0.0.1:8082",
        auth_token="tok",
        claude_bin="claude",
        timeout_sec=30.0,
        reading_only=reading_only,
    ):
        pass
    return captured


def _hook_command(argv: list) -> str:
    settings = json.loads(argv[argv.index("--settings") + 1])
    (entry,) = settings["hooks"]["PreToolUse"]
    assert entry["matcher"] == "*"
    return entry["hooks"][0]["command"]


_DONE = ['{"type":"result","result":"Done.","session_id":"s1"}']


@pytest.mark.asyncio
@pytest.mark.parametrize("reading_only", [False, True])
async def test_breaker_hook_installed_with_default_threshold(monkeypatch, reading_only) -> None:
    monkeypatch.delenv("HARNESS_FCC_REPEAT_FAILURE_THRESHOLD", raising=False)
    argv = await _run(monkeypatch, _DONE, reading_only=reading_only)
    cmd = _hook_command(argv)
    assert "repeat_failure_breaker.py" in cmd and cmd.endswith("--threshold 3")
    assert Path(cmd.split()[1]).is_file()  # the script path the hook runs exists


@pytest.mark.asyncio
async def test_breaker_threshold_from_env(monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_FCC_REPEAT_FAILURE_THRESHOLD", "5")
    assert _hook_command(await _run(monkeypatch, _DONE)).endswith("--threshold 5")


@pytest.mark.asyncio
async def test_breaker_disabled_at_zero(monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_FCC_REPEAT_FAILURE_THRESHOLD", "0")
    assert "--settings" not in await _run(monkeypatch, _DONE)


def test_invalid_threshold_falls_back_to_default(monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_FCC_REPEAT_FAILURE_THRESHOLD", "lots")
    assert motor.repeat_failure_threshold() == 3


@pytest.mark.asyncio
async def test_breaker_fire_is_logged_with_corr_id(monkeypatch, caplog) -> None:
    blocked = {
        "type": "user",
        "message": {"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": "t9", "is_error": True,
            "content": f"PreToolUse:mcp__firecrawl__firecrawl_scrape hook error: {BLOCK_MARKER} Blocked: ...",
        }]},
    }
    plain_error = {
        "type": "user",
        "message": {"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": "t8", "is_error": True, "content": "DNS resolution failed",
        }]},
    }
    with caplog.at_level(logging.WARNING, logger=motor.logger.name):
        await _run(monkeypatch, [json.dumps(plain_error), json.dumps(blocked), *_DONE])
    fired = [r.getMessage() for r in caplog.records if "fcc_repeat_failure_breaker_fired" in r.getMessage()]
    assert len(fired) == 1
    assert "corr=corr-breaker" in fired[0]
