"""Regression: Juniper's Hub chat turns never reached the warm pool (2026-10-06).

Live: corr d1c5272c-... and 7059ac4b-... logged ``mode=spawn`` with no
``fcc_warm_pool_fallback``. ``chat_reply`` was True (auto-memory stayed off),
but Hub's Unified Chat path always attaches a ``ReadingToolBindingV1``
(orion/hub/turn_orchestrator.py, reading_context="unified_chat") and the motor
refused the warm path for any turn with a reading binding. These tests drive
the request the way Hub builds it, through the governor's runner, into the
real motor and a real warm pool.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock

import pytest

from orion.fcc import mcp_config as mcp_mod
from orion.harness import fcc_motor as motor
from orion.harness.runner import HarnessRunner
from orion.harness.tests.fixtures import make_thought
from orion.harness.tests.test_fcc_warm_pool import (  # noqa: F401 -- fixtures
    _start_pool,
    _terminal,
    _turn,
    env_setup,
    pool_cleanup,
    stub,
)
from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.context_exec import ContextExecPermissionV1
from orion.schemas.harness_finalize import HarnessRunRequestV1
from orion.schemas.reading import ReadingToolBindingV1


def _hub_chat_request(corr: str) -> HarnessRunRequestV1:
    """The fields Hub's execute_unified_turn sets for a typed chat message."""
    binding = ReadingToolBindingV1(
        invocation_context="unified_chat", parent_run_id=corr, parent_trace_id=corr,
    )
    return HarnessRunRequestV1(
        correlation_id=corr, thought_event=make_thought(), user_message="hello Orion",
        permissions=ContextExecPermissionV1(), answer_contract=AnswerContract(),
        reading_binding=binding, utterance_origin="juniper",
    )


async def _governor_motor_kwargs(request: HarnessRunRequestV1) -> dict:
    """What the governor's runner hands the FCC motor for this request (wire round trip first)."""
    wire = HarnessRunRequestV1.model_validate_json(request.model_dump_json())
    captured: dict = {}

    async def capture(**kwargs):
        captured.update(kwargs)
        yield {"type": "error", "error": "captured", "error_code": "test"}

    await HarnessRunner(AsyncMock(), fcc_runner=capture, fcc_timeout_sec=999).run(wire)
    return captured


@pytest.mark.asyncio
async def test_hub_chat_turn_takes_the_warm_path(env_setup, stub, pool_cleanup, caplog):
    caplog.set_level(logging.INFO)
    pool = await _start_pool(env_setup, stub)
    kwargs = await _governor_motor_kwargs(_hub_chat_request("corr-hub-1"))
    assert kwargs["chat_reply"] is True
    assert kwargs["reading_binding"] is not None  # Hub always binds reading tools

    frame = _terminal(await _turn(
        env_setup, stub, "hello Orion", "corr-hub-1",
        chat_reply=kwargs["chat_reply"], reading_binding=kwargs["reading_binding"],
    ))
    assert frame["type"] == "final", frame
    assert frame["metadata"]["fcc_spawn_mode"] == "warm"
    assert pool.counters["hit"] == 1
    lines = [r.getMessage() for r in caplog.records if "fcc_warm_path_decision" in r.getMessage()]
    assert lines == [
        "fcc_warm_path_decision corr=corr-hub-1 decision=try_warm chat_reply=True "
        "reading_only=False has_reading_binding=True"
    ]


@pytest.mark.asyncio
async def test_decision_line_names_why_a_turn_spawned(env_setup, stub, pool_cleanup, caplog):
    caplog.set_level(logging.INFO)
    await _start_pool(env_setup, stub)
    await _turn(env_setup, stub, "curious", "corr-inv", chat_reply=False)
    msgs = [r.getMessage() for r in caplog.records if "fcc_warm_path_decision" in r.getMessage()]
    assert any("corr=corr-inv decision=spawn:not_chat_reply" in m for m in msgs)


@pytest.fixture
def mcp_on(env_setup, monkeypatch):
    """MCP enabled with the reading/introspect servers, external tool checks stubbed."""
    fcc_env = env_setup["tmp"] / "fcc.env"
    fcc_env.write_text(fcc_env.read_text() + "GITHUB_PAT=x\nFIRECRAWL_API_KEY=y\n", encoding="utf-8")
    monkeypatch.setenv("HARNESS_FCC_MCP_ENABLED", "true")
    monkeypatch.setenv("HARNESS_FCC_INTROSPECT_ENABLED", "true")
    monkeypatch.setenv("ORION_BUS_URL", "redis://127.0.0.1:1/0")
    for key in ("HARNESS_AITOWN_ENABLED", "HARNESS_FCC_CONTEXT_MODE_ENABLED",
                "HARNESS_FCC_CONTEXT_MODE_HOOKS_ENABLED", "HARNESS_FCC_GITNEXUS_ENABLED"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(mcp_mod, "_require_tool", lambda *a, **k: None)
    monkeypatch.setattr(mcp_mod, "_TMP_ROOT", env_setup["tmp"])
    return env_setup


def _binding(corr: str) -> ReadingToolBindingV1:
    return ReadingToolBindingV1(invocation_context="unified_chat", parent_run_id=corr, parent_trace_id=corr)


@pytest.mark.asyncio
async def test_warm_reading_tools_follow_the_current_turn(mcp_on, stub, pool_cleanup):
    pool = await _start_pool(mcp_on, stub)
    slot = pool._slots[0]
    assert slot.reading_tools is True
    cfg = json.loads(slot.mcp_config_path.read_text())
    assert cfg["mcpServers"]["orion-reading"]["env"]["ORION_READING_BINDING_FILE"] == str(slot.reading_binding_file)
    assert cfg["mcpServers"]["orion-introspect"]["env"]["ORION_INTROSPECT_BINDING_FILE"] == str(slot.introspect_binding_file)
    assert "ORION_READING_BINDING" not in cfg["mcpServers"]["orion-reading"]["env"]

    seen = []
    for corr in ("corr-r1", "corr-r2"):
        frame = _terminal(await _turn(mcp_on, stub, "READBINDING", corr, reading_binding=_binding(corr)))
        assert frame["metadata"]["fcc_spawn_mode"] == "warm", frame
        seen.append(json.loads(frame["llm_response"].removeprefix("BINDING ")))
    assert [s["mode"] for s in seen] == ["file", "file"]
    assert [s["binding"]["parent_run_id"] for s in seen] == ["corr-r1", "corr-r2"]
    # Between turns nothing is bound: a stray tool call cannot act for the last turn.
    assert slot.reading_binding_file.read_text() == ""
    assert slot.introspect_binding_file.read_text() == ""


def test_binding_file_reads_fail_closed_when_unbound(tmp_path):
    from orion.fcc.turn_binding_file import NoTurnBoundError, read_binding_file, write_binding_file

    path = tmp_path / "b.json"
    with pytest.raises(NoTurnBoundError):
        read_binding_file(path, ReadingToolBindingV1)
    write_binding_file(path, _binding("c-1"))
    assert read_binding_file(path, ReadingToolBindingV1).parent_run_id == "c-1"
    write_binding_file(path, None)
    with pytest.raises(NoTurnBoundError):
        read_binding_file(path, ReadingToolBindingV1)


def test_render_refuses_both_binding_forms(tmp_path):
    with pytest.raises(ValueError):
        mcp_mod.render_mcp_config(
            correlation_id="c", fcc_env={}, reading_binding=_binding("c"), reading_binding_file=tmp_path / "f",
        )
