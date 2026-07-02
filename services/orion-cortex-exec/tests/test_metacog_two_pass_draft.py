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
        type="flow",
        emergent_entity="Atlas",
        summary="steady focus",
        resonance_signature="flow: Atlas | Δ:low | →maintain",
    )
    messages = executor_module._metacog_uncertainty_probe_messages(patch)
    assert messages[0]["role"] == "system"
    assert "resonance_signature" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "type=flow" in messages[1]["content"]
    assert "entity=Atlas" in messages[1]["content"]
    assert "reference_signature=" in messages[1]["content"]


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
        "type": "flow",
        "emergent_entity": "Atlas",
        "summary": "steady",
        "mantra": "breathe",
        "resonance_signature": "flow: Atlas | Δ:low | →maintain",
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
        "spark_phi_narrative": "",
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
