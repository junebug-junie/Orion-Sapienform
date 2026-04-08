from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import asyncio

import pytest

from orion.core.bus.bus_schemas import LLMMessage
from orion.schemas.agents.schemas import ContextBlock, Goal, PlannerRequest, ToolDef


def _load_planner_api_module():
    root = Path(__file__).resolve().parents[1]
    app_dir = root / "services" / "orion-planner-react" / "app"

    pkg = types.ModuleType("orion_planner_react")
    pkg.__path__ = [str(app_dir.parent)]
    subpkg = types.ModuleType("orion_planner_react.app")
    subpkg.__path__ = [str(app_dir)]
    sys.modules.setdefault("orion_planner_react", pkg)
    sys.modules.setdefault("orion_planner_react.app", subpkg)

    settings_spec = importlib.util.spec_from_file_location("orion_planner_react.app.settings", app_dir / "settings.py")
    settings_mod = importlib.util.module_from_spec(settings_spec)
    assert settings_spec and settings_spec.loader
    settings_spec.loader.exec_module(settings_mod)
    sys.modules["orion_planner_react.app.settings"] = settings_mod

    api_spec = importlib.util.spec_from_file_location("orion_planner_react.app.api", app_dir / "api.py")
    api_mod = importlib.util.module_from_spec(api_spec)
    assert api_spec and api_spec.loader
    api_spec.loader.exec_module(api_mod)
    return api_mod


api = _load_planner_api_module()


def _patch_bus(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_connect(self) -> None:  # noqa: ANN001
        return None

    async def fake_close(self) -> None:  # noqa: ANN001
        return None

    async def fake_rpc(self, *args, **kwargs):  # noqa: ANN001
        return {"data": b""}

    monkeypatch.setattr(api.OrionBusAsync, "connect", fake_connect)
    monkeypatch.setattr(api.OrionBusAsync, "close", fake_close)
    monkeypatch.setattr(api.OrionBusAsync, "rpc_request", fake_rpc)


def test_delegate_mode_repairs_missing_action_with_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_call_planner_llm(**_: object) -> tuple[dict, dict]:
        return {"thought": "x", "finish": False, "action": None, "final_answer": None}, {"salvage_succeeded": False, "raw_snippet": ""}

    monkeypatch.setattr(api, "_call_planner_llm", fake_call_planner_llm)
    _patch_bus(monkeypatch)

    req = PlannerRequest(
        goal=Goal(description="Help me decide next step"),
        context=ContextBlock(conversation_history=[LLMMessage(role="user", content="Need help now")]),
        toolset=[
            ToolDef(tool_id="triage", description="triage", input_schema={"properties": {"text": {"type": "string"}}}),
            ToolDef(tool_id="plan_action", description="plan", input_schema={"properties": {"goal": {"type": "string"}}}),
            ToolDef(tool_id="analyze_text", description="analyze", input_schema={"properties": {"text": {"type": "string"}}}),
        ],
        preferences={"delegate_tool_execution": True},
    )

    res = asyncio.run(api.run_react_loop(req))

    assert res.stop_reason == "delegate"
    assert res.trace
    assert res.trace[-1].action is not None
    assert res.trace[-1].action["tool_id"] in {"triage", "plan_action", "analyze_text"}


def test_delegate_mode_falls_back_when_planner_call_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_call_planner_llm(**_: object) -> tuple[dict, dict]:
        raise api.PlannerParseError("Planner LLM returned non-JSON: '{\"thought\":\"x\"'")

    monkeypatch.setattr(api, "_call_planner_llm", fake_call_planner_llm)
    _patch_bus(monkeypatch)

    req = PlannerRequest(
        goal=Goal(description="Triage this"),
        context=ContextBlock(conversation_history=[LLMMessage(role="user", content="Need triage")]),
        toolset=[ToolDef(tool_id="triage", description="triage", input_schema={})],
        preferences={"delegate_tool_execution": True},
    )

    res = asyncio.run(api.run_react_loop(req))

    assert res.status == "ok"
    assert res.stop_reason == "delegate"
    assert res.trace[-1].action is not None
    assert res.trace[-1].action["tool_id"] == "triage"


def test_finish_true_with_string_final_answer_is_accepted_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_call_planner_llm(**_: object) -> tuple[dict, dict]:
        return {"thought": "done", "finish": True, "action": None, "final_answer": "## Overview\nAll set."}, {"salvage_succeeded": False, "raw_snippet": ""}

    monkeypatch.setattr(api, "_call_planner_llm", fake_call_planner_llm)
    _patch_bus(monkeypatch)

    req = PlannerRequest(
        request_id="req-string",
        goal=Goal(description="Summarize"),
        context=ContextBlock(),
        toolset=[],
    )

    res = asyncio.run(api.run_react_loop(req))

    assert res.status == "ok"
    assert res.stop_reason == "final_answer"
    assert res.final_answer is not None
    assert res.final_answer.content == "## Overview\nAll set."
    assert res.final_answer.structured == {}


@pytest.mark.parametrize(
    ("raw_text", "expected_snippet"),
    [
        (
            'bash\n{"thought":"done","finish":true,"action":null,"final_answer":"## Deployment Instructions\\nShip it."}',
            "## Deployment Instructions",
        ),
        (
            'plaintext\n```json\n{"thought":"done","finish":true,"action":null,"final_answer":"Plain text final answer"}\n```',
            "Plain text final answer",
        ),
        (
            'Lead text before payload {"thought":"done","finish":true,"action":null,"final_answer":"Embedded JSON recovered"} trailing note',
            "Embedded JSON recovered",
        ),
    ],
)
def test_salvage_layer_recovers_common_wrappers(raw_text: str, expected_snippet: str) -> None:
    parsed, meta = api._parse_planner_response_text(raw_text)

    assert meta["salvage_succeeded"] is True
    assert meta["parse_mode"] == "normalized_from_jsonish"
    assert parsed["finish"] is True
    assert expected_snippet in parsed["final_answer"]


def test_salvage_prefers_outer_balanced_json_over_inner_tool_content() -> None:
    raw_text = """
{
  "thought": "done",
  "finish": true,
  "action": null,
  "final_answer": {
    "content": "Use this final answer.",
    "tool_preview": "```python\\nprint({'nested': 'object'})\\n```"
  }
}
""".strip()

    parsed, meta = api._parse_planner_response_text(raw_text)

    assert parsed["finish"] is True
    assert parsed["final_answer"]["content"] == "Use this final answer."
    assert meta["salvage_succeeded"] is False


def test_finish_true_with_dict_final_answer_is_normalized(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    async def fake_call_planner_llm(**_: object) -> tuple[dict, dict]:
        return {
            "thought": "done",
            "finish": True,
            "action": None,
            "final_answer": {
                "overview": "Top-level summary",
                "next_steps": ["Ship patch", "Monitor logs"],
            },
        }, {"salvage_succeeded": False, "raw_snippet": ""}

    monkeypatch.setattr(api, "_call_planner_llm", fake_call_planner_llm)
    _patch_bus(monkeypatch)

    req = PlannerRequest(
        request_id="req-dict",
        goal=Goal(description="Summarize"),
        context=ContextBlock(),
        toolset=[],
    )

    with caplog.at_level("INFO", logger="planner-react.api"):
        res = asyncio.run(api.run_react_loop(req))

    assert res.final_answer is not None
    assert "### Overview" in res.final_answer.content
    assert "Top-level summary" in res.final_answer.content
    assert "Ship patch" in res.final_answer.content
    assert res.final_answer.structured["overview"] == "Top-level summary"
    assert "failure_category=normalization_applied" in caplog.text
    assert "corr_id=req-dict" in caplog.text
    assert "final_answer_type=dict" in caplog.text


def test_finish_true_with_content_dict_final_answer_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_call_planner_llm(**_: object) -> tuple[dict, dict]:
        return {
            "thought": "done",
            "finish": True,
            "action": None,
            "final_answer": {"content": "Structured content"},
        }, {"salvage_succeeded": False, "raw_snippet": ""}

    monkeypatch.setattr(api, "_call_planner_llm", fake_call_planner_llm)
    _patch_bus(monkeypatch)

    req = PlannerRequest(
        request_id="req-content-dict",
        goal=Goal(description="Summarize"),
        context=ContextBlock(),
        toolset=[],
    )

    res = asyncio.run(api.run_react_loop(req))

    assert res.final_answer is not None
    assert res.final_answer.content == "Structured content"
    assert res.final_answer.structured["content"] == "Structured content"


def test_call_planner_llm_routes_requests_to_agent_lane() -> None:
    captured: dict[str, object] = {}

    class FakeCodec:
        def decode(self, _data: bytes):
            payload = {"content": "{\"thought\":\"done\",\"finish\":true,\"action\":null,\"final_answer\":{\"content\":\"ok\"}}"}
            envelope = types.SimpleNamespace(payload=payload)
            return types.SimpleNamespace(ok=True, envelope=envelope, error=None)

    class FakeBus:
        def __init__(self) -> None:
            self.codec = FakeCodec()

        async def rpc_request(self, channel, env, *, reply_channel, timeout_sec):  # noqa: ANN001
            captured["channel"] = channel
            captured["env"] = env
            captured["reply_channel"] = reply_channel
            captured["timeout_sec"] = timeout_sec
            return {"data": b"ok"}

    result, _meta = asyncio.run(
        api._call_planner_llm(
            FakeBus(),
            goal=Goal(description="Plan the safest Atlas rollout"),
            toolset=[],
            context=ContextBlock(conversation_history=[LLMMessage(role="user", content="Plan the rollout")]),
            prior_trace=[],
            limits=types.SimpleNamespace(timeout_seconds=5),
        )
    )

    assert result["finish"] is True
    assert captured["env"].payload["route"] == "agent"


def test_finish_true_with_text_dict_final_answer_is_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_call_planner_llm(**_: object) -> tuple[dict, dict]:
        return {
            "thought": "done",
            "finish": True,
            "action": None,
            "final_answer": {"text": "Use the text field"},
        }, {"salvage_succeeded": False, "raw_snippet": ""}

    monkeypatch.setattr(api, "_call_planner_llm", fake_call_planner_llm)
    _patch_bus(monkeypatch)

    req = PlannerRequest(
        request_id="req-text-dict",
        goal=Goal(description="Summarize"),
        context=ContextBlock(),
        toolset=[],
    )

    res = asyncio.run(api.run_react_loop(req))

    assert res.final_answer is not None
    assert res.final_answer.content == "Use the text field"
    assert res.final_answer.structured["text"] == "Use the text field"


def test_finish_true_with_list_final_answer_is_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_call_planner_llm(**_: object) -> tuple[dict, dict]:
        return {
            "thought": "done",
            "finish": True,
            "action": None,
            "final_answer": ["First item", {"details": "Second item"}],
        }, {"salvage_succeeded": False, "raw_snippet": ""}

    monkeypatch.setattr(api, "_call_planner_llm", fake_call_planner_llm)
    _patch_bus(monkeypatch)

    req = PlannerRequest(
        request_id="req-list",
        goal=Goal(description="Summarize"),
        context=ContextBlock(),
        toolset=[],
    )

    res = asyncio.run(api.run_react_loop(req))

    assert res.final_answer is not None
    assert "- First item" in res.final_answer.content
    assert "### Details" in res.final_answer.content
    assert res.final_answer.structured["items"][0] == "First item"


def test_malformed_response_triggers_repair_path(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    async def fake_call_planner_llm(**_: object) -> tuple[dict, dict]:
        calls["count"] += 1
        if calls["count"] == 1:
            return {"thought": "bad", "finish": False, "action": None, "final_answer": {"unexpected": True}}, {"salvage_succeeded": False, "raw_snippet": ""}
        return {"thought": "repaired", "finish": True, "action": None, "final_answer": "Recovered"}, {"salvage_succeeded": False, "raw_snippet": ""}

    monkeypatch.setattr(api, "_call_planner_llm", fake_call_planner_llm)
    _patch_bus(monkeypatch)

    req = PlannerRequest(
        request_id="req-repair",
        goal=Goal(description="Repair me"),
        context=ContextBlock(),
        toolset=[],
    )

    res = asyncio.run(api.run_react_loop(req))

    assert calls["count"] == 2
    assert res.final_answer is not None
    assert res.final_answer.content == "Recovered"


def test_repair_path_accepts_finish_true_without_action(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    async def fake_call_planner_llm(**_: object) -> tuple[dict, dict]:
        calls["count"] += 1
        if calls["count"] == 1:
            return {"thought": "bad", "finish": False, "action": None, "final_answer": None}, {"salvage_succeeded": False, "raw_snippet": ""}
        return {"finish": True, "final_answer": "## Deployment Instructions\nProceed."}, {"salvage_succeeded": False, "raw_snippet": ""}

    monkeypatch.setattr(api, "_call_planner_llm", fake_call_planner_llm)
    _patch_bus(monkeypatch)

    req = PlannerRequest(
        request_id="req-repair-finish",
        goal=Goal(description="Repair me"),
        context=ContextBlock(),
        toolset=[],
    )

    res = asyncio.run(api.run_react_loop(req))

    assert calls["count"] == 2
    assert res.final_answer is not None
    assert res.final_answer.content == "## Deployment Instructions\nProceed."


def test_repair_path_accepts_finish_true_content_object_without_action(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    async def fake_call_planner_llm(**_: object) -> tuple[dict, dict]:
        calls["count"] += 1
        if calls["count"] == 1:
            return {"thought": "bad", "finish": False, "action": None, "final_answer": None}, {"salvage_succeeded": False, "raw_snippet": ""}
        return {
            "finish": True,
            "action": None,
            "final_answer": {"content": "### Fixed\nShip the response."},
        }, {"salvage_succeeded": False, "raw_snippet": ""}

    monkeypatch.setattr(api, "_call_planner_llm", fake_call_planner_llm)
    _patch_bus(monkeypatch)

    req = PlannerRequest(
        request_id="req-repair-content-object",
        goal=Goal(description="Repair me"),
        context=ContextBlock(),
        toolset=[],
    )

    res = asyncio.run(api.run_react_loop(req))

    assert calls["count"] == 2
    assert res.final_answer is not None
    assert res.final_answer.content == "### Fixed\nShip the response."


def test_raw_code_block_final_answer_is_salvaged() -> None:
    parsed, meta = api._parse_planner_response_text("```bash\npython bot_bridge.py --discord\n```")

    assert parsed["finish"] is True
    assert parsed["final_answer"]["content"] == "python bot_bridge.py --discord"
    assert meta["parse_mode"] == "salvaged_from_code_block"
    assert meta["salvage_source"] == "bash"


def test_mixed_text_final_answer_is_salvaged() -> None:
    raw = "## Orion Discord Integration\n1. Add a Discord bridge service.\n2. Route messages through Orch and Exec."
    parsed, meta = api._parse_planner_response_text(raw)

    assert parsed["finish"] is True
    assert "Discord bridge service" in parsed["final_answer"]["content"]
    assert meta["parse_mode"] == "salvaged_final_answer_from_mixed_text"


def test_finish_true_with_final_answer_content_and_no_action_normalizes_cleanly() -> None:
    normalized = api._validate_or_normalize_planner_step(
        {"thought": "done", "finish": True, "action": None, "final_answer": {"content": "Ship it"}}
    )

    assert normalized["finish"] is True
    assert normalized["final_answer"] == "Ship it"
    assert normalized["action"] is None


def test_finish_false_without_action_fails_cleanly() -> None:
    with pytest.raises(api.PlannerSchemaError, match="finish=false requires an action"):
        api._validate_or_normalize_planner_step({"thought": "need tool", "finish": False, "action": None, "final_answer": None})


def test_unrecoverable_garbage_still_falls_back() -> None:
    with pytest.raises(api.PlannerParseError):
        api._parse_planner_response_text("total nonsense with no json object anywhere")
