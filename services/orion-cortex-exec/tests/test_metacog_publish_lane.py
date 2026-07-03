from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

ROOT = Path(__file__).resolve().parents[3]
EXEC_ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(EXEC_ROOT) not in sys.path:
    sys.path.append(str(EXEC_ROOT))

from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.collapse_mirror import CollapseMirrorEntryV2
from orion.schemas.cortex.schemas import ExecutionStep


def _load_executor_module():
    app_dir = EXEC_ROOT / "app"
    executor_path = app_dir / "executor.py"
    package_name = "orion_cortex_exec_lane"
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


def _load_template(name: str) -> str:
    return (ROOT / "orion" / "cognition" / "prompts" / name).read_text(encoding="utf-8")


def _draft_ctx(*, spark_blob: str = "{}") -> dict:
    return {
        "trigger": {"trigger_kind": "baseline", "reason": "test", "pressure": 0.1, "zen_state": "zen"},
        "trigger_kind": "baseline",
        "context_summary": "unit test context",
        "spark_state_json": spark_blob,
        "turn_effect_json": "{}",
        "recent_turn_effect_alerts_json": "[]",
        "turn_effect_policy_json": "{}",
        "turn_effect_explanations_json": "{}",
        "biometrics_json": "{}",
        "metacog_biometrics_cue": '{"status":"fresh","constraint":"NONE"}',
        "spark_phi_narrative": "",
    }


def test_metacog_biometrics_cue_draft_compact():
    executor_module = _load_executor_module()
    ctx = {
        "biometrics": {
            "status": "fresh",
            "freshness_s": 12.0,
            "constraint": "NONE",
            "cluster": {
                "composite": {"strain": 0.42, "homeostasis": 0.71, "stability": 0.88},
            },
            "nodes": {},
        }
    }
    cue = executor_module._metacog_biometrics_cue(ctx, phase="draft")
    assert len(cue) <= 350
    parsed = json.loads(cue)
    assert parsed["status"] == "fresh"
    assert parsed["strain"] == 0.42
    assert parsed["homeostasis"] == 0.71
    assert parsed["stability"] == 0.88
    assert parsed["freshness_s"] == 12


def test_metacog_biometrics_cue_enrich_includes_node_lines():
    executor_module = _load_executor_module()
    ctx = {
        "biometrics": {
            "status": "fresh",
            "constraint": "GPU_MEM",
            "cluster": {
                "composite": {"strain": 0.62, "homeostasis": 0.5, "stability": 0.44},
            },
            "nodes": {
                "atlas": {
                    "status": "OK",
                    "summary": {"composites": {"strain": 0.71}, "pressures": {"gpu": 0.82}},
                },
                "athena": {"status": "OK", "summary": {}},
            },
        }
    }
    cue = executor_module._metacog_biometrics_cue(ctx, phase="enrich")
    assert len(cue) <= 600
    parsed = json.loads(cue)
    assert "cluster" in parsed
    assert isinstance(parsed.get("nodes"), list)
    assert len(parsed["nodes"]) <= 4
    assert any("atlas" in line for line in parsed["nodes"])


def test_enrich_prompt_uses_enrich_biometrics_cue(monkeypatch):
    executor_module = _load_executor_module()
    captured_prompts: list[str] = []

    class FakeLLMClient:
        def __init__(self, bus):
            self.bus = bus

        async def chat(self, **kwargs):
            req = kwargs.get("req")
            messages = getattr(req, "messages", []) or []
            if messages:
                msg = messages[0]
                content = getattr(msg, "content", None)
                if content is None and isinstance(msg, dict):
                    content = msg.get("content")
                captured_prompts.append(str(content or ""))
            return {}

    monkeypatch.setattr(executor_module, "LLMGatewayClient", FakeLLMClient)
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_enrich_prompt_max_chars", 50000)
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_enrich_worker_ctx_char_budget", 50000)

    biometrics = {
        "status": "fresh",
        "constraint": "GPU_MEM",
        "cluster": {
            "composite": {"strain": 0.62, "homeostasis": 0.5, "stability": 0.44},
        },
        "nodes": {
            "atlas": {
                "status": "OK",
                "summary": {"composites": {"strain": 0.71}, "pressures": {"gpu": 0.82}},
            },
        },
    }
    ctx = _draft_ctx()
    ctx["biometrics"] = biometrics
    ctx["metacog_biometrics_cue"] = executor_module._metacog_biometrics_cue(
        {"biometrics": biometrics}, phase="draft"
    )
    ctx["metacog_biometrics_cue_enrich"] = executor_module._metacog_biometrics_cue(
        {"biometrics": biometrics}, phase="enrich"
    )

    draft_parsed = json.loads(ctx["metacog_biometrics_cue"])
    enrich_parsed = json.loads(ctx["metacog_biometrics_cue_enrich"])
    assert "nodes" not in draft_parsed
    assert any("atlas" in line for line in enrich_parsed["nodes"])

    draft_entry = CollapseMirrorEntryV2(
        event_id="evt-enrich-cue",
        id="evt-enrich-cue",
        trigger="dense",
        observer="orion",
        observer_state=["zen"],
        type="flow",
        emergent_entity="Test",
        summary="Test summary",
        mantra="Test mantra",
        field_resonance="Test resonance",
        resonance_signature="Test sig",
        source_service="metacog",
    ).model_dump(mode="json")
    draft_entry["state_snapshot"] = {"telemetry": {"metacog_draft_mode": "llm"}}
    ctx["collapse_entry"] = draft_entry
    ctx["collapse_json"] = json.dumps(draft_entry)

    template = _load_template("log_orion_metacognition_enrich.j2")
    step = ExecutionStep(
        verb_name="log_orion_metacognition",
        step_name="enrich_entry",
        order=1,
        services=["MetacogEnrichService"],
        prompt_template=template,
    )
    source = ServiceRef(name="test", node="test", version="1.0")

    result = asyncio.run(
        executor_module.call_step_services(
            bus=object(),
            source=source,
            step=step,
            ctx=ctx,
            correlation_id="corr-enrich-cue-swap",
        )
    )

    assert result.status == "success"
    assert captured_prompts
    prompt = captured_prompts[0]
    assert "atlas: strain=0.71 gpu=0.82" in prompt
    assert '"nodes"' in prompt


def test_metacog_biometrics_cue_missing_biometrics():
    executor_module = _load_executor_module()
    cue = executor_module._metacog_biometrics_cue({}, phase="draft")
    parsed = json.loads(cue)
    assert parsed["status"] == "missing"


def test_metacog_context_service_sets_biometrics_cue_from_cluster(monkeypatch):
    executor_module = _load_executor_module()
    from orion.schemas.telemetry.biometrics import BiometricsClusterV1

    cluster = BiometricsClusterV1(
        composites={"strain": 0.55, "homeostasis": 0.66, "stability": 0.77},
        constraint="NONE",
    )
    biometrics_context = executor_module._default_biometrics_context(
        status="fresh", reason="state_service"
    )
    biometrics_context["cluster"] = cluster.model_dump(mode="json")
    ctx = {"biometrics": biometrics_context}
    ctx["metacog_biometrics_cue"] = executor_module._metacog_biometrics_cue(ctx, phase="draft")

    parsed = json.loads(ctx["metacog_biometrics_cue"])
    assert parsed["strain"] == 0.55
    assert parsed["homeostasis"] == 0.66
    assert parsed["stability"] == 0.77


def test_metacog_format_node_cue_line_skips_bad_numeric_fields():
    executor_module = _load_executor_module()
    line = executor_module._metacog_format_node_cue_line(
        "atlas",
        {
            "status": "OK",
            "summary": {
                "composites": {"strain": "not-a-number"},
                "pressures": {"gpu": "bad"},
            },
        },
    )
    assert line == "atlas: ok"
    assert "strain=" not in line
    assert "gpu=" not in line


def test_metacog_biometrics_cue_enrich_overflow_falls_back_to_cluster_only(monkeypatch):
    executor_module = _load_executor_module()
    monkeypatch.setattr(executor_module, "_METACOG_BIOMETRICS_CUE_ENRICH_MAX_CHARS", 120)
    ctx = {
        "biometrics": {
            "status": "fresh",
            "constraint": "GPU_MEM",
            "cluster": {
                "composite": {"strain": 0.62, "homeostasis": 0.5, "stability": 0.44},
            },
            "nodes": {
                "atlas": {
                    "status": "OK",
                    "summary": {"composites": {"strain": 0.71}, "pressures": {"gpu": 0.82}},
                },
                "athena": {"status": "OK", "summary": {}},
            },
        }
    }
    cue = executor_module._metacog_biometrics_cue(ctx, phase="enrich")
    parsed = json.loads(cue)
    assert "cluster" in parsed
    assert "nodes" not in parsed
    assert len(cue) <= 120


def test_metacog_draft_prompt_under_slim_budget():
    executor_module = _load_executor_module()
    template = _load_template("log_orion_metacognition_draft.j2")
    ctx = _draft_ctx()
    ctx["metacog_biometrics_cue"] = executor_module._metacog_biometrics_cue(
        {
            "biometrics": {
                "status": "fresh",
                "freshness_s": 12,
                "constraint": "NONE",
                "cluster": {"composite": {"strain": 0.42, "homeostasis": 0.71, "stability": 0.88}},
            }
        },
        phase="draft",
    )
    prompt = executor_module._render_prompt(template, ctx)
    assert len(ctx["metacog_biometrics_cue"]) <= 350
    assert len(prompt) <= 6500


def test_metacog_draft_section_keys_cover_template_fields():
    executor_module = _load_executor_module()
    keys = executor_module._METACOG_DRAFT_CTX_LEN_KEYS
    assert "biometrics_json" not in keys
    assert "metacog_biometrics_cue" in keys
    assert "spark_phi_narrative" in keys

    template = _load_template("log_orion_metacognition_draft.j2")
    for key in keys:
        assert f"{{{{ {key} }}}}" in template or f"{{{{ {key}|" in template


def test_metacog_enrich_section_keys_cover_template_fields():
    executor_module = _load_executor_module()
    keys = executor_module._METACOG_ENRICH_CTX_LEN_KEYS
    assert "biometrics_json" not in keys
    assert "metacog_biometrics_cue" in keys
    assert "spark_phi_narrative" in keys

    template = _load_template("log_orion_metacognition_enrich.j2")
    for key in keys:
        assert f"{{{{ {key} }}}}" in template or f"{{{{ {key}|" in template


def test_oversized_draft_prompt_skips_llm_with_budget_fallback(monkeypatch):
    executor_module = _load_executor_module()
    calls: list[str] = []

    class FakeLLMClient:
        def __init__(self, bus):
            self.bus = bus

        async def chat(self, **kwargs):
            req = kwargs.get("req")
            calls.append(getattr(req, "raw_user_text", "draft"))
            return {}

    monkeypatch.setattr(executor_module, "LLMGatewayClient", FakeLLMClient)
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_draft_prompt_max_chars", 200)

    template = _load_template("log_orion_metacognition_draft.j2")
    ctx = _draft_ctx(spark_blob="X" * 5000)
    step = ExecutionStep(
        verb_name="log_orion_metacognition",
        step_name="draft_entry",
        order=0,
        services=["MetacogDraftService"],
        prompt_template=template,
    )
    source = ServiceRef(name="test", node="test", version="1.0")

    result = asyncio.run(
        executor_module.call_step_services(
            bus=object(),
            source=source,
            step=step,
            ctx=ctx,
            correlation_id="corr-draft-budget",
        )
    )

    assert result.status == "success"
    assert calls == []
    draft_result = result.result["MetacogDraftService"]
    assert draft_result["ok"] is True
    assert draft_result["fallback_reason"] == "prompt_budget_exceeded"
    telemetry = ctx["collapse_entry"]["state_snapshot"]["telemetry"]
    assert telemetry["metacog_draft_mode"] == "fallback"
    assert telemetry["metacog_draft_fallback_reason"] == "prompt_budget_exceeded"
    assert telemetry["metacog_prompt_chars"] > 200


def test_oversized_enrich_prompt_skips_llm_with_budget_fallback(monkeypatch):
    executor_module = _load_executor_module()
    calls: list[str] = []

    class FakeLLMClient:
        def __init__(self, bus):
            self.bus = bus

        async def chat(self, **kwargs):
            calls.append("enrich")
            return {}

    monkeypatch.setattr(executor_module, "LLMGatewayClient", FakeLLMClient)
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_enrich_prompt_max_chars", 200)

    draft_entry = CollapseMirrorEntryV2(
        event_id="evt-1",
        id="evt-1",
        trigger="dense",
        observer="orion",
        observer_state=["zen"],
        type="flow",
        emergent_entity="Test",
        summary="Test summary",
        mantra="Test mantra",
        field_resonance="Test resonance",
        resonance_signature="Test sig",
        source_service="metacog",
    ).model_dump(mode="json")
    draft_entry["state_snapshot"] = {"telemetry": {"metacog_draft_mode": "llm"}}

    template = _load_template("log_orion_metacognition_enrich.j2")
    ctx = _draft_ctx(spark_blob="Y" * 5000)
    ctx["collapse_entry"] = draft_entry
    ctx["collapse_json"] = __import__("json").dumps(draft_entry)

    step = ExecutionStep(
        verb_name="log_orion_metacognition",
        step_name="enrich_entry",
        order=1,
        services=["MetacogEnrichService"],
        prompt_template=template,
    )
    source = ServiceRef(name="test", node="test", version="1.0")

    result = asyncio.run(
        executor_module.call_step_services(
            bus=object(),
            source=source,
            step=step,
            ctx=ctx,
            correlation_id="corr-enrich-budget",
        )
    )

    assert result.status == "success"
    assert calls == []
    enrich_result = result.result["MetacogEnrichService"]
    assert enrich_result["ok"] is True
    assert enrich_result["fallback_reason"] == "prompt_budget_exceeded"
    telemetry = ctx["final_entry"]["state_snapshot"]["telemetry"]
    assert telemetry["metacog_enrich_fallback_reason"] == "prompt_budget_exceeded"


def test_draft_trims_biometrics_cue_before_ctx_overflow_fallback(monkeypatch):
    executor_module = _load_executor_module()
    calls: list[str] = []

    class FakeLLMClient:
        def __init__(self, bus):
            self.bus = bus

        async def chat(self, **kwargs):
            calls.append("draft")
            return {}

    monkeypatch.setattr(executor_module, "LLMGatewayClient", FakeLLMClient)
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_draft_prompt_max_chars", 50000)
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_draft_worker_ctx_char_budget", 8000)

    template = _load_template("log_orion_metacognition_draft.j2")
    ctx = _draft_ctx(spark_blob="{}")
    ctx["metacog_biometrics_cue"] = json.dumps({"status": "fresh", "blob": "x" * 5000})
    ctx["spark_state_json"] = "{}"

    step = ExecutionStep(
        verb_name="log_orion_metacognition",
        step_name="draft_entry",
        order=0,
        services=["MetacogDraftService"],
        prompt_template=template,
    )
    source = ServiceRef(name="test", node="test", version="1.0")

    result = asyncio.run(
        executor_module.call_step_services(
            bus=object(), source=source, step=step, ctx=ctx, correlation_id="corr-draft-trim",
        )
    )

    assert result.status == "success"
    assert json.loads(ctx["metacog_biometrics_cue"])["status"] == "trimmed"
    assert calls == ["draft"]


def test_draft_ctx_overflow_after_cue_and_spark_trim(monkeypatch):
    executor_module = _load_executor_module()
    calls: list[str] = []

    class FakeLLMClient:
        def __init__(self, bus):
            self.bus = bus

        async def chat(self, **kwargs):
            calls.append("draft")
            return {}

    monkeypatch.setattr(executor_module, "LLMGatewayClient", FakeLLMClient)
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_draft_prompt_max_chars", 50000)
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_draft_worker_ctx_char_budget", 500)

    template = _load_template("log_orion_metacognition_draft.j2")
    ctx = _draft_ctx(spark_blob="Z" * 8000)
    ctx["metacog_biometrics_cue"] = json.dumps({"status": "fresh", "strain": 0.5})

    step = ExecutionStep(
        verb_name="log_orion_metacognition",
        step_name="draft_entry",
        order=0,
        services=["MetacogDraftService"],
        prompt_template=template,
    )
    source = ServiceRef(name="test", node="test", version="1.0")

    result = asyncio.run(
        executor_module.call_step_services(
            bus=object(), source=source, step=step, ctx=ctx, correlation_id="corr-draft-overflow",
        )
    )

    assert result.status == "success"
    assert calls == []
    draft_result = result.result["MetacogDraftService"]
    assert draft_result.get("fallback_reason") == "prompt_context_overflow"


def test_enrich_trims_biometrics_before_ctx_overflow_fallback(monkeypatch):
    executor_module = _load_executor_module()
    calls: list[str] = []

    class FakeLLMClient:
        def __init__(self, bus):
            self.bus = bus

        async def chat(self, **kwargs):
            calls.append("enrich")
            return {}

    monkeypatch.setattr(executor_module, "LLMGatewayClient", FakeLLMClient)
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_enrich_prompt_max_chars", 50000)
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_enrich_worker_ctx_char_budget", 8000)

    draft_entry = CollapseMirrorEntryV2(
        event_id="evt-trim",
        id="evt-trim",
        trigger="dense",
        observer="orion",
        observer_state=["zen"],
        type="flow",
        emergent_entity="Test",
        summary="Test summary",
        mantra="Test mantra",
        field_resonance="Test resonance",
        resonance_signature="Test sig",
        source_service="metacog",
    ).model_dump(mode="json")
    draft_entry["state_snapshot"] = {"telemetry": {"metacog_draft_mode": "llm"}}

    template = _load_template("log_orion_metacognition_enrich.j2")
    ctx = _draft_ctx(spark_blob="{}")
    ctx["metacog_biometrics_cue"] = json.dumps({"status": "fresh", "blob": "x" * 5000})
    ctx["collapse_entry"] = draft_entry
    ctx["collapse_json"] = json.dumps(draft_entry)

    step = ExecutionStep(
        verb_name="log_orion_metacognition",
        step_name="enrich_entry",
        order=1,
        services=["MetacogEnrichService"],
        prompt_template=template,
    )
    source = ServiceRef(name="test", node="test", version="1.0")

    result = asyncio.run(
        executor_module.call_step_services(
            bus=object(),
            source=source,
            step=step,
            ctx=ctx,
            correlation_id="corr-enrich-trim",
        )
    )

    assert result.status == "success"
    assert json.loads(ctx["metacog_biometrics_cue"])["status"] == "trimmed"
    enrich_result = result.result["MetacogEnrichService"]
    assert enrich_result["ok"] is True
    assert enrich_result.get("fallback_reason") != "prompt_context_overflow"
    assert calls == ["enrich"]


def test_enrich_ctx_overflow_after_biometrics_trim(monkeypatch):
    executor_module = _load_executor_module()
    calls: list[str] = []

    class FakeLLMClient:
        def __init__(self, bus):
            self.bus = bus

        async def chat(self, **kwargs):
            calls.append("enrich")
            return {}

    monkeypatch.setattr(executor_module, "LLMGatewayClient", FakeLLMClient)
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_enrich_prompt_max_chars", 50000)
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_enrich_worker_ctx_char_budget", 1000)

    draft_entry = CollapseMirrorEntryV2(
        event_id="evt-overflow",
        id="evt-overflow",
        trigger="dense",
        observer="orion",
        observer_state=["zen"],
        type="flow",
        emergent_entity="Test",
        summary="Test summary",
        mantra="Test mantra",
        field_resonance="Test resonance",
        resonance_signature="Test sig",
        source_service="metacog",
    ).model_dump(mode="json")
    draft_entry["state_snapshot"] = {"telemetry": {"metacog_draft_mode": "llm"}}

    template = _load_template("log_orion_metacognition_enrich.j2")
    ctx = _draft_ctx(spark_blob="Z" * 8000)
    ctx["metacog_biometrics_cue"] = json.dumps({"status": "fresh", "strain": 0.5})
    ctx["collapse_entry"] = draft_entry
    ctx["collapse_json"] = json.dumps(draft_entry)

    step = ExecutionStep(
        verb_name="log_orion_metacognition",
        step_name="enrich_entry",
        order=1,
        services=["MetacogEnrichService"],
        prompt_template=template,
    )
    source = ServiceRef(name="test", node="test", version="1.0")

    result = asyncio.run(
        executor_module.call_step_services(
            bus=object(),
            source=source,
            step=step,
            ctx=ctx,
            correlation_id="corr-enrich-overflow",
        )
    )

    assert result.status == "success"
    assert calls == []
    enrich_result = result.result["MetacogEnrichService"]
    assert enrich_result["ok"] is True
    assert enrich_result["fallback_reason"] == "prompt_context_overflow"


def test_firebreak_skip_includes_fallback_reason_and_diagnostics():
    executor_module = _load_executor_module()
    mock_bus = MagicMock()
    mock_bus.publish = AsyncMock()

    corr_id = str(uuid4())
    ctx = {
        "trigger": {"trigger_kind": "baseline"},
        "metacog_draft_prompt_chars": 9000,
        "metacog_draft_section_sizes": {"spark_state_json": 7000, "context_summary": 120},
        "final_entry": {
            "id": "123",
            "state_snapshot": {
                "telemetry": {
                    "metacog_draft_mode": "fallback",
                    "metacog_draft_fallback_reason": "prompt_budget_exceeded",
                    "metacog_prompt_chars": 9000,
                    "metacog_prompt_section_sizes": {"spark_state_json": 7000},
                }
            },
        },
    }

    step = ExecutionStep(
        step_name="publish",
        verb_name="log_orion_metacognition",
        services=["MetacogPublishService"],
        order=1,
    )
    source = ServiceRef(name="test", node="test", version="1.0")

    result = asyncio.run(
        executor_module.call_step_services(
            bus=mock_bus,
            source=source,
            step=step,
            ctx=ctx,
            correlation_id=corr_id,
        )
    )

    publish = result.result["MetacogPublishService"]
    assert publish["skipped"] is True
    assert publish["reason"] == "firebreak_baseline_fallback"
    assert publish["fallback_reason"] == "prompt_budget_exceeded"
    assert publish["prompt_chars"] == 9000
    assert publish["largest_sections"]["spark_state_json"] == 7000
    mock_bus.publish.assert_called_once()


def test_manual_dense_fallback_still_publishes():
    executor_module = _load_executor_module()
    mock_bus = MagicMock()
    mock_bus.publish = AsyncMock()

    valid_entry = CollapseMirrorEntryV2(
        event_id="evt-dense",
        id="evt-dense",
        trigger="dense",
        observer="orion",
        observer_state=["zen"],
        type="flow",
        emergent_entity="Test",
        summary="Test summary",
        mantra="Test mantra",
        field_resonance="Test resonance",
        resonance_signature="Test sig",
        source_service="metacog",
    ).model_dump(mode="json")
    valid_entry["state_snapshot"] = {
        "telemetry": {
            "metacog_draft_mode": "fallback",
            "metacog_draft_fallback_reason": "json_parse_failed",
        }
    }

    ctx = {
        "trigger": {"trigger_kind": "dense"},
        "final_entry": valid_entry,
    }

    step = ExecutionStep(
        step_name="publish",
        verb_name="log_orion_metacognition",
        services=["MetacogPublishService"],
        order=1,
    )
    source = ServiceRef(name="test", node="test", version="1.0")

    result = asyncio.run(
        executor_module.call_step_services(
            bus=mock_bus,
            source=source,
            step=step,
            ctx=ctx,
            correlation_id=str(uuid4()),
        )
    )

    assert result.status == "success"
    publish = result.result["MetacogPublishService"]
    assert "skipped" not in publish
    mock_bus.publish.assert_called()


def test_log_orion_metacognition_recall_disabled_by_verb_default():
    from orion.cognition.plan_loader import build_plan_for_verb
    from app.recall_utils import delivery_safe_recall_decision

    plan = build_plan_for_verb("log_orion_metacognition", mode="brain")
    recall_cfg: dict = {}
    if str(plan.metadata.get("recall_enabled_default") or "").lower() == "false":
        recall_cfg["enabled"] = False
    decision = delivery_safe_recall_decision(recall_cfg, plan.steps, plan_verb_name=plan.verb_name)
    assert str(plan.metadata.get("recall_enabled_default") or "").lower() == "false"
    assert decision["run_recall"] is False
