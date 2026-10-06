"""L7 (spec 2026-10-06-unified-turn-latency-design.md): Claude Code auto-memory
is keyed by working directory and every FCC turn shares the sandbox checkout,
so chat replies were loading notes investigation runs wrote. Chat-reply turns
get CLAUDE_CODE_DISABLE_AUTO_MEMORY=1; investigation turns do not."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from orion.harness import runner as runner_mod
from orion.harness.fcc_motor import _build_subprocess_env, chat_auto_memory_disabled
from orion.harness.runner import HarnessRunner, is_chat_reply_request
from orion.harness.tests.fixtures import make_thought
from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.context_exec import ContextExecPermissionV1
from orion.schemas.harness_finalize import HarnessRunRequestV1

VAR = "CLAUDE_CODE_DISABLE_AUTO_MEMORY"
FLAG = "HARNESS_FCC_CHAT_DISABLE_AUTO_MEMORY"


def _env(chat_reply: bool):
    return _build_subprocess_env(fcc_server_url="http://fcc:8082", auth_token="t", chat_reply=chat_reply)


def _request(origin):
    return HarnessRunRequestV1(
        correlation_id="c-1", thought_event=make_thought(), user_message="hi",
        permissions=ContextExecPermissionV1(), answer_contract=AnswerContract(),
        utterance_origin=origin,
    )


def test_chat_reply_turn_disables_auto_memory(monkeypatch):
    monkeypatch.setenv(FLAG, "true")
    assert _env(True)[VAR] == "1"


def test_investigation_turn_keeps_auto_memory(monkeypatch):
    monkeypatch.setenv(FLAG, "true")
    assert VAR not in _env(False)


def test_flag_off_leaves_both_alone(monkeypatch):
    monkeypatch.setenv(FLAG, "false")
    assert VAR not in _env(True)
    assert VAR not in _env(False)


def test_flag_ships_on_when_unset(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert chat_auto_memory_disabled() is True
    assert _env(True)[VAR] == "1"


def test_container_wide_value_does_not_leak_into_investigation_turns(monkeypatch):
    monkeypatch.setenv(FLAG, "true")
    monkeypatch.setenv(VAR, "1")
    assert VAR not in _env(False)


@pytest.mark.parametrize(
    "origin, expected",
    [("juniper", True), (" Juniper ", True), ("orion", False), (None, False), ("", False)],
)
def test_only_juniper_origin_is_a_chat_reply(origin, expected):
    assert is_chat_reply_request(_request(origin)) is expected


def test_request_without_origin_keeps_old_wire_shape():
    req = _request(None)
    dumped = req.model_dump(mode="json")
    dumped.pop("utterance_origin")
    assert HarnessRunRequestV1.model_validate(dumped).utterance_origin is None


async def _capture_runner(request):
    captured = {}

    async def motor(**kwargs):
        captured.update(kwargs)
        yield {"type": "error", "error": "synthetic stop", "error_code": "test"}

    await HarnessRunner(AsyncMock(), fcc_runner=motor, fcc_timeout_sec=999).run(request)
    return captured


@pytest.mark.asyncio
async def test_runner_marks_chat_reply_for_juniper_turn():
    assert (await _capture_runner(_request("juniper")))["chat_reply"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ["orion", None])
async def test_runner_does_not_mark_investigation_or_other_turns(origin):
    assert "chat_reply" not in await _capture_runner(_request(origin))


@pytest.mark.asyncio
async def test_default_fcc_runner_forwards_chat_reply_to_the_motor(monkeypatch):
    seen = {}

    async def fake_run_fcc_turn(**kwargs):
        seen.update(kwargs)
        yield {"type": "final"}

    monkeypatch.setattr(runner_mod, "run_fcc_turn", fake_run_fcc_turn)
    monkeypatch.setattr(runner_mod, "load_fcc_env", lambda _p: {})
    async for _ in runner_mod.default_fcc_runner(prompt="p", correlation_id="c", chat_reply=True):
        pass
    assert seen["chat_reply"] is True
