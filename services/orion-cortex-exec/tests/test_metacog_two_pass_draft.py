from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from orion.schemas.metacog_patches import MetacogDraftTextPatchV1

ROOT = Path(__file__).resolve().parents[3]
EXEC_ROOT = Path(__file__).resolve().parents[1]


def _load_executor_module():
    app_dir = EXEC_ROOT / "app"
    executor_path = app_dir / "executor.py"
    package_name = "orion_cortex_exec_two_pass"
    app_package_name = f"{package_name}.app"
    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [str(app_dir.parent)]
        sys.modules[package_name] = pkg
    if app_package_name not in sys.modules:
        pkg = types.ModuleType(app_package_name)
        pkg.__path__ = [str(app_dir)]
        sys.modules[app_package_name] = pkg
    spec = importlib.util.spec_from_file_location(f"{app_package_name}.executor", executor_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_metacog_uncertainty_probe_messages_use_patch_fields():
    executor_module = _load_executor_module()
    patch = MetacogDraftTextPatchV1(
        summary="steady focus",
        mantra="hold the line",
        what_changed={"summary": "clarity up", "evidence": ["cue"]},
    )
    messages = executor_module._metacog_uncertainty_probe_messages(patch)
    assert messages[0]["role"] == "system"
    assert "summary" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "summary=steady focus" in messages[1]["content"]
    assert "mantra=hold the line" in messages[1]["content"]
    assert "what_changed=clarity up" in messages[1]["content"]


def test_metacog_uncertainty_probe_messages_truncate_long_fields():
    executor_module = _load_executor_module()
    patch = MetacogDraftTextPatchV1(summary="x" * 800)
    messages = executor_module._metacog_uncertainty_probe_messages(patch)
    assert len(messages[0]["content"]) <= 512
    assert len(messages[1]["content"]) <= 512
    assert messages[1]["content"].endswith("...")


def test_should_run_metacog_uncertainty_probe_respects_settings(monkeypatch):
    executor_module = _load_executor_module()
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_return_logprobs", False)
    assert executor_module._should_run_metacog_uncertainty_probe() is False
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_return_logprobs", True)
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_logprob_probe_mode", "")
    assert executor_module._should_run_metacog_uncertainty_probe() is False
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_logprob_probe_mode", "native_completion")
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_uncertainty_probe_enabled", False)
    assert executor_module._should_run_metacog_uncertainty_probe() is False
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_uncertainty_probe_enabled", True)
    assert executor_module._should_run_metacog_uncertainty_probe() is True


import asyncio
import json
from types import SimpleNamespace

from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.cortex.schemas import ExecutionStep

_VALID_DRAFT_JSON = json.dumps(
    {
        "summary": "steady",
        "mantra": "breathe",
    }
)


def _load_template(name: str) -> str:
    return (ROOT / "orion" / "cognition" / "prompts" / name).read_text(encoding="utf-8")


def _draft_ctx() -> dict:
    return {
        "trigger": {"trigger_kind": "baseline", "reason": "test", "pressure": 0.1, "zen_state": "zen"},
        "trigger_kind": "baseline",
        "context_summary": "unit test",
        "spark_state_json": "{}",
        "turn_effect_json": "{}",
        "recent_turn_effect_alerts_json": "[]",
        "turn_effect_policy_json": "{}",
        "turn_effect_explanations_json": "{}",
        "biometrics_json": "{}",
        "metacog_biometrics_cue": '{"status":"fresh","constraint":"NONE"}',
    }


def _fake_llm_response(*, content: str = "", meta: dict | None = None):
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(meta=meta or {}, choices=[choice])


def test_metacog_draft_pass1_excludes_logprob_flags(monkeypatch):
    executor_module = _load_executor_module()
    captured: list[dict] = []

    class FakeLLMClient:
        def __init__(self, bus):
            self.bus = bus

        async def chat(self, **kwargs):
            req = kwargs["req"]
            captured.append(dict(req.options or {}))
            if len(captured) == 1:
                return _fake_llm_response(content=_VALID_DRAFT_JSON)
            return _fake_llm_response(
                meta={
                    "llm_uncertainty": {
                        "schema_version": "v1",
                        "available": True,
                        "source": "llamacpp_native_completion",
                    }
                }
            )

    monkeypatch.setattr(executor_module, "LLMGatewayClient", FakeLLMClient)
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_return_logprobs", True)
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_logprob_probe_mode", "native_completion")
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_uncertainty_probe_enabled", True)

    template = _load_template("log_orion_metacognition_draft.j2")
    step = ExecutionStep(
        verb_name="log_orion_metacognition",
        step_name="draft_entry",
        order=0,
        services=["MetacogDraftService"],
        prompt_template=template,
    )
    source = ServiceRef(name="test", node="test", version="1.0")
    ctx = _draft_ctx()

    result = asyncio.run(
        executor_module.call_step_services(
            bus=object(),
            source=source,
            step=step,
            ctx=ctx,
            correlation_id="corr-two-pass-options",
        )
    )

    assert result.status == "success"
    assert len(captured) == 2
    assert captured[0]["response_format"] == {"type": "json_object"}
    assert "return_logprobs" not in captured[0]
    assert "logprob_probe_mode" not in captured[0]
    assert captured[1]["return_logprobs"] is True
    assert captured[1]["logprob_probe_mode"] == "native_completion"
    assert captured[1]["max_tokens"] == 128
    assert "response_format" not in captured[1]
    telemetry = ctx["collapse_entry"]["state_snapshot"]["telemetry"]
    assert telemetry["metacog_draft_mode"] == "llm"
    assert telemetry["llm_uncertainty"]["source"] == "llamacpp_native_completion"


def test_metacog_probe_failure_does_not_force_fallback(monkeypatch):
    executor_module = _load_executor_module()

    class FakeLLMClient:
        def __init__(self, bus):
            self.bus = bus

        async def chat(self, **kwargs):
            req = kwargs["req"]
            opts = req.options or {}
            if opts.get("response_format"):
                return _fake_llm_response(content=_VALID_DRAFT_JSON)
            raise TimeoutError("probe timeout")

    monkeypatch.setattr(executor_module, "LLMGatewayClient", FakeLLMClient)
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_return_logprobs", True)
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_logprob_probe_mode", "native_completion")
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_uncertainty_probe_enabled", True)

    template = _load_template("log_orion_metacognition_draft.j2")
    step = ExecutionStep(
        verb_name="log_orion_metacognition",
        step_name="draft_entry",
        order=0,
        services=["MetacogDraftService"],
        prompt_template=template,
    )
    source = ServiceRef(name="test", node="test", version="1.0")
    ctx = _draft_ctx()

    result = asyncio.run(
        executor_module.call_step_services(
            bus=object(),
            source=source,
            step=step,
            ctx=ctx,
            correlation_id="corr-probe-fail",
        )
    )

    assert result.status == "success"
    telemetry = ctx["collapse_entry"]["state_snapshot"]["telemetry"]
    assert telemetry["metacog_draft_mode"] == "llm"
    assert "llm_uncertainty" not in telemetry


def test_is_metacog_draft_example_echo_matches_case_and_whitespace_insensitive():
    executor_module = _load_executor_module()
    patch = MetacogDraftTextPatchV1(
        summary="  " + executor_module._METACOG_DRAFT_EXAMPLE_SUMMARY.upper() + "  ",
        mantra=executor_module._METACOG_DRAFT_EXAMPLE_MANTRA.upper(),
    )
    assert executor_module._is_metacog_draft_example_echo(patch) is True


def test_is_metacog_draft_example_echo_false_for_real_content():
    executor_module = _load_executor_module()
    patch = MetacogDraftTextPatchV1(summary="steady", mantra="breathe")
    assert executor_module._is_metacog_draft_example_echo(patch) is False


def test_is_metacog_draft_example_echo_false_when_only_one_field_matches():
    executor_module = _load_executor_module()
    patch = MetacogDraftTextPatchV1(
        summary=executor_module._METACOG_DRAFT_EXAMPLE_SUMMARY,
        mantra="a genuinely different mantra this time",
    )
    assert executor_module._is_metacog_draft_example_echo(patch) is False


def test_is_metacog_draft_example_echo_false_when_fields_missing():
    executor_module = _load_executor_module()
    patch = MetacogDraftTextPatchV1(summary=None, mantra=None)
    assert executor_module._is_metacog_draft_example_echo(patch) is False


def test_is_metacog_draft_example_echo_catches_light_reword():
    """A model that reworks the example instead of copying it verbatim is still an
    echo, not a real answer -- this is the failure mode an exact-string check alone
    would miss."""
    executor_module = _load_executor_module()
    patch = MetacogDraftTextPatchV1(
        summary="Clarity band ran high while novelty stayed low -- calm, not sedated.",
        mantra="Note the calm; don't confuse it with silence.",
    )
    assert executor_module._is_metacog_draft_example_echo(patch) is True


def test_is_metacog_draft_example_echo_false_for_distinct_calm_narration():
    """Real content that happens to share the same generally-calm register as the
    example, but is not a paraphrase of it, must not be rejected."""
    executor_module = _load_executor_module()
    patch = MetacogDraftTextPatchV1(
        summary="Energy band dropped after the last turn; coherence held steady through the lull.",
        mantra="Let the quiet do its work.",
    )
    assert executor_module._is_metacog_draft_example_echo(patch) is False


def test_metacog_draft_prompt_example_matches_echo_guard_constants():
    """Guards against orion/cognition/prompts/log_orion_metacognition_draft.j2's
    <example_json> drifting from the anti-echo constants in executor.py -- if
    someone edits the example wording without updating the guard, the guard
    silently stops protecting the new example text."""
    executor_module = _load_executor_module()
    template = _load_template("log_orion_metacognition_draft.j2")
    assert executor_module._METACOG_DRAFT_EXAMPLE_SUMMARY in template
    assert executor_module._METACOG_DRAFT_EXAMPLE_MANTRA in template


def test_metacog_draft_rejects_exact_example_echo(monkeypatch):
    """End-to-end: an LLM response that echoes the prompt's own example verbatim
    must be recorded as a fallback, not silently published as a real draft.
    Confirmed live 2026-09-03: this exact echo was 68% of all orion_metacog rows."""
    executor_module = _load_executor_module()
    echoed_json = json.dumps(
        {
            "summary": executor_module._METACOG_DRAFT_EXAMPLE_SUMMARY,
            "mantra": executor_module._METACOG_DRAFT_EXAMPLE_MANTRA,
        }
    )

    class FakeLLMClient:
        def __init__(self, bus):
            self.bus = bus

        async def chat(self, **kwargs):
            return _fake_llm_response(content=echoed_json)

    monkeypatch.setattr(executor_module, "LLMGatewayClient", FakeLLMClient)

    template = _load_template("log_orion_metacognition_draft.j2")
    step = ExecutionStep(
        verb_name="log_orion_metacognition",
        step_name="draft_entry",
        order=0,
        services=["MetacogDraftService"],
        prompt_template=template,
    )
    source = ServiceRef(name="test", node="test", version="1.0")
    ctx = _draft_ctx()

    result = asyncio.run(
        executor_module.call_step_services(
            bus=object(),
            source=source,
            step=step,
            ctx=ctx,
            correlation_id="corr-example-echo",
        )
    )

    assert result.status == "success"
    telemetry = ctx["collapse_entry"]["state_snapshot"]["telemetry"]
    assert telemetry["metacog_draft_mode"] == "fallback"
    assert telemetry["metacog_draft_fallback_reason"] == "example_echo"
    # The fallback entry must not silently carry the echoed example forward.
    assert ctx["collapse_entry"]["summary"] != executor_module._METACOG_DRAFT_EXAMPLE_SUMMARY
