from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest

from orion.schemas.agents.schemas import ContextBlock, Goal, Limits, PlannerRequest, Preferences, ToolDef


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


@pytest.fixture()
def semantic_toolset() -> list[ToolDef]:
    return [
        ToolDef(tool_id="housekeep_runtime", description="runtime housekeeping", input_schema={}),
        ToolDef(tool_id="assess_storage_health", description="storage check", input_schema={}),
        ToolDef(tool_id="assess_mesh_presence", description="mesh check", input_schema={}),
        ToolDef(tool_id="summarize_recent_changes", description="change summary", input_schema={}),
    ]


def test_finish_true_accepts_thought_when_final_answer_missing(semantic_toolset):
    step = api._validate_or_normalize_planner_step(
        {
            "thought": "Here is a concise answer about self-improvement habits.",
            "finish": True,
            "action": None,
            "final_answer": None,
        },
        toolset=semantic_toolset,
        from_salvage=False,
    )
    assert "self-improvement" in step["final_answer"]


def test_finish_true_accepts_summary_key_in_final_answer_dict(semantic_toolset):
    step = api._validate_or_normalize_planner_step(
        {
            "thought": "done",
            "finish": True,
            "action": None,
            "final_answer": {"summary": "Focus on sleep, movement, and one keystone habit."},
        },
        toolset=semantic_toolset,
        from_salvage=False,
    )
    assert "keystone" in step["final_answer"]


@pytest.fixture()
def semantic_toolset_with_answer_direct(semantic_toolset) -> list[ToolDef]:
    return list(semantic_toolset) + [
        ToolDef(tool_id="answer_direct", description="Answer the user directly", input_schema={}),
    ]


@pytest.fixture()
def semantic_toolset_with_write_guide_and_answer_direct(semantic_toolset) -> list[ToolDef]:
    return list(semantic_toolset) + [
        ToolDef(tool_id="write_guide", description="Structured implementation guide", input_schema={}),
        ToolDef(tool_id="answer_direct", description="Answer the user directly", input_schema={}),
    ]


def test_finish_true_meta_planner_thought_coerces_to_write_guide_for_implementation_guide(
    semantic_toolset_with_write_guide_and_answer_direct,
):
    step = api._validate_or_normalize_planner_step(
        {
            "thought": (
                "The user is asking how to become a better version of themselves. This is a self-improvement "
                "question, which requires a guide or tutorial. Since the output mode is implementation_guide, "
                "I should use write_guide to provide structured steps."
            ),
            "finish": True,
            "action": None,
            "final_answer": None,
        },
        toolset=semantic_toolset_with_write_guide_and_answer_direct,
        from_salvage=False,
        planning_goal_text="how do I become a better version of myself?",
        output_mode="implementation_guide",
    )
    assert step["finish"] is False
    assert step["action"]["tool_id"] == "write_guide"
    assert "better version" in (step["action"]["input"].get("request") or "").lower()
    assert "better version" in (step["action"]["input"].get("text") or "").lower()


def test_finish_true_meta_planner_thought_falls_back_to_answer_direct_without_write_guide(
    semantic_toolset_with_answer_direct,
):
    step = api._validate_or_normalize_planner_step(
        {
            "thought": (
                "The user is asking how to become a better version of themselves. This is a self-improvement "
                "question, which requires a guide or tutorial. Since the output mode is implementation_guide, "
                "I should use write_guide to provide structured steps."
            ),
            "finish": True,
            "action": None,
            "final_answer": None,
        },
        toolset=semantic_toolset_with_answer_direct,
        from_salvage=False,
        planning_goal_text="how do I become a better version of myself?",
        output_mode="implementation_guide",
    )
    assert step["finish"] is False
    assert step["action"]["tool_id"] == "answer_direct"
    assert "better version" in step["action"]["input"]["text"].lower()


def test_finish_false_missing_action_salvages_to_answer_direct_when_offered(semantic_toolset_with_answer_direct):
    step = api._validate_or_normalize_planner_step(
        {
            "thought": "I will answer next.",
            "finish": False,
            "action": None,
            "final_answer": None,
        },
        toolset=semantic_toolset_with_answer_direct,
        from_salvage=False,
        planning_goal_text="how do I eat a sandwich in space?",
        output_mode="implementation_guide",
    )
    assert step["finish"] is False
    assert step["action"]["tool_id"] == "answer_direct"
    assert "sandwich" in step["action"]["input"]["text"].lower()


def test_clean_parse_accepts_valid_tool(semantic_toolset):
    step = api._validate_or_normalize_planner_step(
        {
            "thought": "Need runtime check",
            "finish": False,
            "action": {"tool_id": "housekeep_runtime", "input": {"text": "dry-run cleanup"}},
            "final_answer": None,
        },
        toolset=semantic_toolset,
        from_salvage=False,
    )
    assert step["action"]["tool_id"] == "housekeep_runtime"


def test_clean_parse_rejects_unknown_tool(semantic_toolset):
    with pytest.raises(api.PlannerContractViolation) as exc:
        api._validate_or_normalize_planner_step(
            {
                "thought": "run shell",
                "finish": False,
                "action": {"name": "RunCommand", "input": {"text": "docker ps"}},
                "final_answer": None,
            },
            toolset=semantic_toolset,
            from_salvage=False,
        )
    assert exc.value.failure_category == "invalid_action_not_in_toolset"


def test_salvage_accepts_repaired_valid_tool(semantic_toolset):
    step = api._validate_or_normalize_planner_step(
        {
            "thought": "normalized",
            "finish": False,
            "action": {"name": "housekeep_runtime", "input": "dry-run please"},
            "final_answer": None,
        },
        toolset=semantic_toolset,
        from_salvage=True,
    )
    assert step["action"]["tool_id"] == "housekeep_runtime"


def test_salvage_rejects_runcommand_with_destructive_input(semantic_toolset):
    with pytest.raises(api.PlannerContractViolation) as exc:
        api._validate_or_normalize_planner_step(
            {
                "thought": "normalized from bad json",
                "finish": False,
                "action": {
                    "name": "RunCommand",
                    "input": "docker ps -a --filter 'status=exited' --format '{{.ID}}' | xargs -r docker rm",
                },
                "final_answer": None,
            },
            toolset=semantic_toolset,
            from_salvage=True,
        )
    assert exc.value.failure_category == "salvage_out_of_contract"


def test_action_id_not_in_toolset_rejected(semantic_toolset):
    with pytest.raises(api.PlannerContractViolation) as exc:
        api._validate_or_normalize_planner_step(
            {
                "thought": "bad action_id",
                "finish": False,
                "action": {"action_id": "RunCommand", "input": {"text": "noop"}},
                "final_answer": None,
            },
            toolset=semantic_toolset,
            from_salvage=False,
        )
    assert exc.value.failure_category == "invalid_action_id_not_in_toolset"


def test_action_name_id_mismatch_rejected(semantic_toolset):
    with pytest.raises(api.PlannerContractViolation) as exc:
        api._validate_or_normalize_planner_step(
            {
                "thought": "mismatch",
                "finish": False,
                "action": {"name": "housekeep_runtime", "action_id": "assess_mesh_presence", "input": {"text": "x"}},
                "final_answer": None,
            },
            toolset=semantic_toolset,
            from_salvage=False,
        )
    assert exc.value.failure_category == "action_name_id_mismatch"


def test_code_fence_shell_output_does_not_become_action(semantic_toolset):
    parsed, meta = api._parse_planner_response_text("""```bash\ndocker rm -f abc\n```""")
    assert parsed["finish"] is True
    assert parsed["action"] is None
    assert meta["final_answer_salvaged"] is True


class _FakeBus:
    async def connect(self):
        return None

    async def close(self):
        return None


async def _run_with_invalid_action(semantic_toolset):
    req = PlannerRequest(
        request_id="req-1",
        goal=Goal(description="dry-run cleanup of stopped containers"),
        context=ContextBlock(conversation_history=[{"role": "user", "content": "dry-run cleanup of stopped containers"}]),
        toolset=semantic_toolset,
        limits=Limits(max_steps=1, timeout_seconds=5),
        preferences=Preferences(plan_only=True, delegate_tool_execution=True),
    )
    return await api.run_react_loop(req)


def test_regression_runcommand_blocked_before_delegate(monkeypatch, semantic_toolset):
    async def _fake_call_planner_llm(**_kwargs):
        return (
            {
                "thought": "I should run command",
                "finish": False,
                "action": {
                    "name": "RunCommand",
                    "input": "docker ps -a --filter 'status=exited' --format '{{.ID}}' | xargs -r docker rm",
                },
                "final_answer": None,
            },
            {"salvage_succeeded": True, "salvage_source": "normalized_from_jsonish", "parse_mode": "normalized_from_jsonish"},
        )

    called = {"cortex": 0}

    async def _fake_call_cortex_verb(**_kwargs):
        called["cortex"] += 1
        return {"ok": True}

    monkeypatch.setattr(api, "OrionBusAsync", lambda **_kwargs: _FakeBus())
    monkeypatch.setattr(api, "_call_planner_llm", _fake_call_planner_llm)
    monkeypatch.setattr(api, "_call_cortex_verb", _fake_call_cortex_verb)

    response = asyncio.run(_run_with_invalid_action(semantic_toolset))

    assert response.stop_reason == "final_answer"
    assert response.trace and response.trace[0].action is None
    assert "planner contract violation" in response.final_answer.content.lower()
    assert called["cortex"] == 0


def test_dry_run_non_regression_valid_semantic_tool(monkeypatch, semantic_toolset):
    async def _fake_call_planner_llm(**_kwargs):
        return (
            {
                "thought": "Use semantic housekeeping tool in dry-run mode",
                "finish": False,
                "action": {"tool_id": "housekeep_runtime", "input": {"text": "dry-run cleanup of stopped containers"}},
                "final_answer": None,
            },
            {"salvage_succeeded": False, "salvage_source": "none", "parse_mode": "raw_parse_success"},
        )

    monkeypatch.setattr(api, "OrionBusAsync", lambda **_kwargs: _FakeBus())
    monkeypatch.setattr(api, "_call_planner_llm", _fake_call_planner_llm)

    response = asyncio.run(_run_with_invalid_action(semantic_toolset))

    assert response.stop_reason == "delegate"
    assert response.trace and response.trace[0].action
    assert response.trace[0].action["tool_id"] == "housekeep_runtime"


def test_parse_planner_raw_decode_after_invalid_apostrophe_escape_and_trailing_text():
    raw = (
        '{"thought": "Orion'
        + "\\'"
        + 's improvement", "finish": true, "action": null, "final_answer": {"content": "## Guide\\nok"}}'
        "\n\n(Additional analysis omitted.)"
    )
    parsed, meta = api._parse_planner_response_text(raw)
    assert meta["parse_mode"] in ("raw_decode_leading_object", "normalized_from_jsonish")
    assert parsed["thought"] == "Orion's improvement"
    assert parsed["finish"] is True
    assert parsed["final_answer"]["content"] == "## Guide\nok"
