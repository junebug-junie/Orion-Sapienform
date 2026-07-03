"""Tests for agent_repl semantic tool telemetry (PR 789)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

CTX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (str(REPO_ROOT), str(CTX_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.agent_repl_telemetry import detect_semantic_tools_from_code


def test_detect_semantic_tools_direct_call() -> None:
    code = 'files = repo_find_files("*.py", path="services/orion-context-exec/app")'
    assert detect_semantic_tools_from_code(code) == ["repo_find_files"]


def test_detect_semantic_tools_multiple_calls_in_order() -> None:
    code = '''
files = repo_find_files("*.py")
outline = repo_outline(files.splitlines()[0])
chunk = repo_read_range("services/orion-context-exec/app/runner.py", 1, 20)
'''
    assert detect_semantic_tools_from_code(code) == [
        "repo_find_files",
        "repo_outline",
        "repo_read_range",
    ]


def test_detect_semantic_tools_dedupes_within_step() -> None:
    code = '''
repo_outline("a.py")
repo_outline("b.py")
'''
    assert detect_semantic_tools_from_code(code) == ["repo_outline"]


def test_detect_semantic_tools_no_known_tools() -> None:
    code = "x = 1 + 2"
    assert detect_semantic_tools_from_code(code) == []


def test_detect_semantic_tools_syntax_error_fallback() -> None:
    code = 'repo_outline("x.py"'
    assert detect_semantic_tools_from_code(code) == ["repo_outline"]


def test_detect_semantic_tools_empty_code() -> None:
    assert detect_semantic_tools_from_code("") == []
    assert detect_semantic_tools_from_code("   ") == []


def test_detect_semantic_tools_attribute_call() -> None:
    code = 'result = module.repo_outline("runner.py")'
    assert detect_semantic_tools_from_code(code) == ["repo_outline"]


def test_detect_semantic_tools_source_order_in_comprehension() -> None:
    code = '[repo_outline(p) for p in repo_find_files("*")]'
    assert detect_semantic_tools_from_code(code) == ["repo_outline", "repo_find_files"]


def test_detect_semantic_tools_unparseable_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.agent_repl_telemetry as telemetry

    def boom(_code: str):
        raise ValueError("parse exploded")

    monkeypatch.setattr(telemetry, "_detect_via_ast", boom)
    monkeypatch.setattr(telemetry, "_detect_via_regex", boom)
    assert detect_semantic_tools_from_code("repo_list()") == []


@pytest.mark.asyncio
async def test_run_agent_repl_single_semantic_tool_in_verb_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import runner as runner_mod
    from app.events import ContextExecEventEmitter
    from app.runner import ContextExecRunner
    from app.rlm_engine import RLMEngine
    from orion.schemas.context_exec import (
        ContextExecRequestV1,
        context_exec_permissions_for_llm_profile,
    )

    live_steps: list[dict] = []
    orig_agent_step = ContextExecEventEmitter.agent_step

    async def capture_agent_step(self, **kwargs):
        live_steps.append(kwargs)
        return await orig_agent_step(self, **kwargs)

    monkeypatch.setattr(ContextExecEventEmitter, "agent_step", capture_agent_step)

    class StubEngine(RLMEngine):
        engine_name = "smolcode"

        async def run(self, request, namespace, *, organ_runtime=None,
                      step_callbacks=None, max_steps=None, per_step_timeout=None,
                      workspace_info=None, workspace=None, **_kwargs):
            if step_callbacks:
                class _Step:
                    step_number = 0
                    is_final_answer = False
                    model_output = "find files"
                    code_action = 'files = repo_find_files("*.py", path="services/orion-context-exec/app")'
                    observations = "runner.py\nsmolcode_engine.py"
                    error = None

                    class timing:
                        duration = 0.3

                step_callbacks[0](_Step())
            return {"summary": "done", "engine": "smolcode", "mode": request.mode}

    r = ContextExecRunner(engine=StubEngine())

    async def fake_resolve(profile):
        from app.llm_profile_resolver import LLMProfileSelection
        return LLMProfileSelection(requested=profile, selected="agent", route_used="agent")

    monkeypatch.setattr(runner_mod, "resolve_llm_profile", fake_resolve)
    monkeypatch.setattr(runner_mod.settings, "orion_bus_enabled", False)

    req = ContextExecRequestV1(
        text="inspect context-exec",
        mode="agent_repl",
        permissions=context_exec_permissions_for_llm_profile("agent"),
        llm_profile="agent",
    )
    run = await r.run(req)
    assert run.verb_trace[0].callable == "repo_find_files"
    assert run.runtime_debug["agent_repl_tool_counts"] == {"repo_find_files": 1}
    assert run.runtime_debug["agent_repl_semantic_tool_detected"] is True
    assert live_steps[0]["tool_id"] == "repo_find_files"


@pytest.mark.asyncio
async def test_run_agent_repl_no_semantic_tool_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import runner as runner_mod
    from app.runner import ContextExecRunner
    from app.rlm_engine import RLMEngine
    from orion.schemas.context_exec import (
        ContextExecRequestV1,
        context_exec_permissions_for_llm_profile,
    )

    class StubEngine(RLMEngine):
        engine_name = "smolcode"

        async def run(self, request, namespace, *, organ_runtime=None,
                      step_callbacks=None, max_steps=None, per_step_timeout=None,
                      workspace_info=None, workspace=None):
            if step_callbacks:
                class _Step:
                    step_number = 0
                    is_final_answer = False
                    model_output = "compute"
                    code_action = "x = 1 + 2"
                    observations = "3"
                    error = None

                    class timing:
                        duration = 0.1

                step_callbacks[0](_Step())
            return {"summary": "done", "engine": "smolcode", "mode": request.mode}

    r = ContextExecRunner(engine=StubEngine())

    async def fake_resolve(profile):
        from app.llm_profile_resolver import LLMProfileSelection
        return LLMProfileSelection(requested=profile, selected="agent", route_used="agent")

    monkeypatch.setattr(runner_mod, "resolve_llm_profile", fake_resolve)

    req = ContextExecRequestV1(
        text="noop",
        mode="agent_repl",
        permissions=context_exec_permissions_for_llm_profile("agent"),
        llm_profile="agent",
    )
    run = await r.run(req)
    assert run.verb_trace[0].callable == "python_interpreter"
    assert run.runtime_debug["agent_repl_tool_counts"] == {}
    assert run.runtime_debug["agent_repl_semantic_tool_detected"] is False


@pytest.mark.asyncio
async def test_run_agent_repl_multiple_tools_keeps_python_interpreter(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import runner as runner_mod
    from app.runner import ContextExecRunner
    from app.rlm_engine import RLMEngine
    from orion.schemas.context_exec import (
        ContextExecRequestV1,
        context_exec_permissions_for_llm_profile,
    )

    class StubEngine(RLMEngine):
        engine_name = "smolcode"

        async def run(self, request, namespace, *, organ_runtime=None,
                      step_callbacks=None, max_steps=None, per_step_timeout=None,
                      workspace_info=None, workspace=None):
            if step_callbacks:
                class _Step:
                    step_number = 0
                    is_final_answer = False
                    model_output = "look around"
                    code_action = '''
files = repo_find_files("*.py")
outline = repo_outline(files.splitlines()[0])
'''
                    observations = "ok"
                    error = None

                    class timing:
                        duration = 0.4

                step_callbacks[0](_Step())
            return {"summary": "done", "engine": "smolcode", "mode": request.mode}

    r = ContextExecRunner(engine=StubEngine())

    async def fake_resolve(profile):
        from app.llm_profile_resolver import LLMProfileSelection
        return LLMProfileSelection(requested=profile, selected="agent", route_used="agent")

    monkeypatch.setattr(runner_mod, "resolve_llm_profile", fake_resolve)

    req = ContextExecRequestV1(
        text="multi tool",
        mode="agent_repl",
        permissions=context_exec_permissions_for_llm_profile("agent"),
        llm_profile="agent",
    )
    run = await r.run(req)
    assert run.verb_trace[0].callable == "python_interpreter"
    assert run.runtime_debug["agent_repl_tool_counts"] == {
        "repo_find_files": 1,
        "repo_outline": 1,
    }
    assert run.runtime_debug["agent_repl_semantic_tool_detected"] is True


@pytest.mark.asyncio
async def test_run_agent_repl_counts_across_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import runner as runner_mod
    from app.runner import ContextExecRunner
    from app.rlm_engine import RLMEngine
    from orion.schemas.context_exec import (
        ContextExecRequestV1,
        context_exec_permissions_for_llm_profile,
    )

    class StubEngine(RLMEngine):
        engine_name = "smolcode"

        async def run(self, request, namespace, *, organ_runtime=None,
                      step_callbacks=None, max_steps=None, per_step_timeout=None,
                      workspace_info=None, workspace=None):
            if step_callbacks:
                for i, code in enumerate([
                    'repo_find_files("*.py")',
                    'repo_find_files("*.md")',
                ]):
                    class _Step:
                        pass

                    step = _Step()
                    step.step_number = i
                    step.is_final_answer = False
                    step.model_output = "step"
                    step.code_action = code
                    step.observations = "ok"
                    step.error = None

                    class timing:
                        duration = 0.1

                    step.timing = timing()
                    step_callbacks[0](step)
            return {"summary": "done", "engine": "smolcode", "mode": request.mode}

    r = ContextExecRunner(engine=StubEngine())

    async def fake_resolve(profile):
        from app.llm_profile_resolver import LLMProfileSelection
        return LLMProfileSelection(requested=profile, selected="agent", route_used="agent")

    monkeypatch.setattr(runner_mod, "resolve_llm_profile", fake_resolve)

    req = ContextExecRequestV1(
        text="count steps",
        mode="agent_repl",
        permissions=context_exec_permissions_for_llm_profile("agent"),
        llm_profile="agent",
    )
    run = await r.run(req)
    assert run.runtime_debug["agent_repl_tool_counts"] == {"repo_find_files": 2}
