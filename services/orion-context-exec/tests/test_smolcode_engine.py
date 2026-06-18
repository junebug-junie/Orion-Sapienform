"""Tests for SmolagentsCodeEngine."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

CTX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (str(REPO_ROOT), str(CTX_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _make_request(text: str = "explain how grounded mode works"):
    from orion.schemas.context_exec import (
        ContextExecRequestV1,
        context_exec_permissions_for_llm_profile,
    )
    return ContextExecRequestV1(
        text=text,
        mode="general_investigation",
        permissions=context_exec_permissions_for_llm_profile("agent"),
        llm_profile="agent",
    )


def _make_runtime() -> MagicMock:
    runtime = MagicMock()
    runtime.request = MagicMock()
    runtime.request.permissions.read_repo = True
    runtime.request.permissions.read_recall = True
    runtime.llm_route = "agent"
    runtime.repo_grep.return_value = []
    runtime.repo_read.return_value = None
    runtime.recall_query = AsyncMock(return_value={"hits": []})
    runtime.llm_chat = AsyncMock(return_value={"ok": True, "content": "agent response"})
    return runtime


@pytest.mark.asyncio
async def test_smolcode_engine_calls_agent_run():
    from app.smolcode_engine import SmolagentsCodeEngine

    runtime = _make_runtime()
    request = _make_request("explain how grounded mode works")

    with patch("app.smolcode_engine.CodeAgent") as mock_cls:
        agent_inst = MagicMock()
        agent_inst.run.return_value = "the final answer"
        mock_cls.return_value = agent_inst

        engine = SmolagentsCodeEngine()
        result = await engine.run(request, None, organ_runtime=runtime)

    assert result["engine"] == "smolcode"
    assert result["summary"] == "the final answer"
    agent_inst.run.assert_called_once_with("explain how grounded mode works")


@pytest.mark.asyncio
async def test_smolcode_engine_graceful_on_agent_error():
    from app.smolcode_engine import SmolagentsCodeEngine

    runtime = _make_runtime()
    request = _make_request("test error path")

    with patch("app.smolcode_engine.CodeAgent") as mock_cls:
        agent_inst = MagicMock()
        agent_inst.run.side_effect = RuntimeError("model unavailable")
        mock_cls.return_value = agent_inst

        engine = SmolagentsCodeEngine()
        result = await engine.run(request, None, organ_runtime=runtime)

    assert result["engine"] == "smolcode"
    assert "error" in result
    assert "model unavailable" in result["error"]


@pytest.mark.asyncio
async def test_smolcode_engine_requires_organ_runtime():
    from app.smolcode_engine import SmolagentsCodeEngine

    request = _make_request()
    engine = SmolagentsCodeEngine()
    result = await engine.run(request, None, organ_runtime=None)

    assert result["engine"] == "smolcode"
    assert "error" in result


@pytest.mark.asyncio
async def test_smolcode_model_calls_agent_lane():
    from app.smolcode_engine import OrionSmolagentsModel

    runtime = _make_runtime()
    loop = asyncio.get_running_loop()
    model = OrionSmolagentsModel(runtime, loop)

    messages = [
        {"role": "system", "content": "You are a codebase agent."},
        {"role": "user", "content": "find the bug"},
    ]

    # model.__call__ uses run_coroutine_threadsafe so it must be called from a thread
    result = await loop.run_in_executor(None, model, messages)

    assert result.content == "agent response"
    runtime.llm_chat.assert_awaited_once()
    _, kwargs = runtime.llm_chat.call_args
    assert kwargs.get("route") == "agent"


def test_build_engine_smolcode():
    from app.rlm_engine import build_engine
    from app.smolcode_engine import SmolagentsCodeEngine

    engine = build_engine("smolcode")
    assert isinstance(engine, SmolagentsCodeEngine)
    assert engine.engine_name == "smolcode"
