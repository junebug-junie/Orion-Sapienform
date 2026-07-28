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
    cue = executor_module._metacog_biometrics_cue(ctx)
    assert len(cue) <= 350
    parsed = json.loads(cue)
    assert parsed["status"] == "fresh"
    assert parsed["strain"] == 0.42
    assert parsed["homeostasis"] == 0.71
    assert parsed["stability"] == 0.88
    assert parsed["freshness_s"] == 12


def test_metacog_biometrics_cue_missing_biometrics():
    executor_module = _load_executor_module()
    cue = executor_module._metacog_biometrics_cue({})
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
    ctx["metacog_biometrics_cue"] = executor_module._metacog_biometrics_cue(ctx)

    parsed = json.loads(ctx["metacog_biometrics_cue"])
    assert parsed["strain"] == 0.55
    assert parsed["homeostasis"] == 0.66
    assert parsed["stability"] == 0.77


def test_metacog_biometrics_cue_draft_uses_age_ms_when_freshness_missing():
    executor_module = _load_executor_module()
    ctx = {
        "biometrics": {
            "status": "fresh",
            "age_ms": 12500,
            "constraint": "NONE",
            "cluster": {"composite": {"strain": 0.42, "homeostasis": 0.71, "stability": 0.88}},
        }
    }
    cue = executor_module._metacog_biometrics_cue(ctx)
    parsed = json.loads(cue)
    assert parsed["freshness_s"] == 12


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
        }
    )
    prompt = executor_module._render_prompt(template, ctx)
    assert len(ctx["metacog_biometrics_cue"]) <= 350
    assert len(prompt) <= 6500


def test_metacog_draft_prompt_live_anatomy_fits_worker_budget():
    """Replay spec §2 section sizes (minus removed biometrics_json blob)."""
    executor_module = _load_executor_module()
    template = _load_template("log_orion_metacognition_draft.j2")
    ctx = _draft_ctx()
    ctx["context_summary"] = "T" * 1183
    ctx["spark_state_json"] = "S" * 634
    ctx["turn_effect_json"] = "E" * 60
    ctx["recent_turn_effect_alerts_json"] = "[]"
    ctx["turn_effect_policy_json"] = "{}"
    ctx["turn_effect_explanations_json"] = "{}"
    ctx["metacog_biometrics_cue"] = executor_module._metacog_biometrics_cue(
        {
            "biometrics": {
                "status": "fresh",
                "age_ms": 12000,
                "constraint": "NONE",
                "cluster": {"composite": {"strain": 0.42, "homeostasis": 0.71, "stability": 0.88}},
            }
        }
    )
    slim_prompt = executor_module._render_prompt(template, ctx)
    fat_ctx = dict(ctx)
    fat_ctx["metacog_biometrics_cue"] = "{}"
    fat_ctx["biometrics_json"] = json.dumps({"blob": "x" * 3823})
    # Template no longer references biometrics_json; savings vs legacy is cue vs blob size.
    assert len(ctx["metacog_biometrics_cue"]) <= 350
    assert len(slim_prompt) <= int(executor_module.settings.cortex_metacog_draft_worker_ctx_char_budget)
    assert len(ctx["metacog_biometrics_cue"]) < len(fat_ctx["biometrics_json"])


def test_metacog_draft_section_keys_cover_template_fields():
    executor_module = _load_executor_module()
    keys = executor_module._METACOG_DRAFT_CTX_LEN_KEYS
    assert "biometrics_json" not in keys
    assert "metacog_biometrics_cue" in keys
    assert "spark_phi_narrative" not in keys

    template = _load_template("log_orion_metacognition_draft.j2")
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
    assert ctx["metacog_ctx_trim_applied"] == ["biometrics_cue"]
    assert ctx["metacog_draft_prompt_chars"] <= 8000
    telemetry = ctx["collapse_entry"]["state_snapshot"]["telemetry"]
    assert telemetry["metacog_ctx_trim_applied"] == ["biometrics_cue"]
    assert telemetry["metacog_biometrics_cue_chars"] == len('{"status":"trimmed"}')
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


def test_firebreak_skip_includes_fallback_reason_and_diagnostics():
    executor_module = _load_executor_module()
    mock_bus = MagicMock()
    mock_bus.publish = AsyncMock()

    corr_id = str(uuid4())
    ctx = {
        "trigger": {"trigger_kind": "baseline"},
        "metacog_draft_prompt_chars": 9000,
        "metacog_draft_section_sizes": {"spark_state_json": 7000, "context_summary": 120},
        "collapse_entry": {
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
        "collapse_entry": valid_entry,
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


def test_publish_builds_metacog_entry_from_real_artifacts_no_self_report():
    """Real-artifact model: causal_density is scored purely from
    substrate_eventfulness/repair_pressure/turn_effect -- no numeric_sisters
    self-report, no self_state blend. Publishes MetacogEntryV1 to
    channel_metacog_sql_write, not CollapseMirrorEntryV2 to channel_collapse_sql_write."""
    executor_module = _load_executor_module()
    mock_bus = MagicMock()
    mock_bus.publish = AsyncMock()

    valid_entry = CollapseMirrorEntryV2(
        event_id="evt-substrate",
        id="evt-substrate",
        trigger="dense",
        observer="Orion",
        observer_state=["strained"],
        type="flow",
        emergent_entity="Substrate Pulse",
        summary="Test summary",
        mantra="Test mantra",
        field_resonance="Test resonance",
        resonance_signature="Test sig",
        source_service="metacog",
        tag_scores={"shift": 0.2},
    ).model_dump(mode="json")
    valid_entry["state_snapshot"] = {
        "telemetry": {"metacog_draft_mode": "llm"},
    }

    ctx = {
        "trigger": {"trigger_kind": "dense", "reason": "substrate_eventfulness:0.60"},
        "trigger_kind": "dense",
        "substrate_eventfulness_score": 0.6,
        "substrate_eventfulness_reasons": ["execution_pressure_spike"],
        "collapse_entry": valid_entry,
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
    assert publish.get("published") is True
    assert publish["channel"] == executor_module.settings.channel_metacog_sql_write
    mock_bus.publish.assert_called_once()
    channel_arg = mock_bus.publish.call_args[0][0]
    assert channel_arg == executor_module.settings.channel_metacog_sql_write
    envelope = mock_bus.publish.call_args[0][1]
    assert envelope.kind == "metacog.entry.v1"
    payload = envelope.payload
    assert "numeric_sisters" not in payload
    assert "state_snapshot" not in payload
    assert payload["trigger_kind"] == "dense"
    assert payload["trigger_reason"] == "substrate_eventfulness:0.60"
    assert payload["state"]["substrate_eventfulness_score"] == 0.6
    assert payload["state"]["substrate_eventfulness_reasons"] == ["execution_pressure_spike"]
    assert payload["causal_density"]["score"] == pytest.approx(0.6)
    assert payload["is_causally_dense"] is True
    assert payload["snapshot_kind"] == "confirmed_dense"
    # Topology (repurposed field_resonance): mechanically names which real
    # artifacts are present, not a hardcoded/omitted field.
    assert payload["touches"] == ["substrate"]
    # Severity (repurposed observer_state): no failed steps, no llm_uncertainty
    # signal in this ctx -> nominal, not a silent default masking real signal.
    assert payload["severity"] == "nominal"
    # Provenance: dynamic per trigger_kind and per touches, not a hardcoded
    # constant with impacts always [].
    assert payload["provenance"]["source"] == "cortex_exec.metacog_pipeline.dense"
    assert payload["provenance"]["impacts"] == ["execution_trajectory"]
    assert isinstance(payload["what_changed"]["evidence"], list)


def test_publish_severity_and_touches_reflect_failures_and_repair_pressure():
    """Same lane, adversarial ctx: a prior failed step plus real repair_pressure
    evidence should show up as non-nominal severity and multi-item touches --
    not silently dropped the way the first pass's hardcoded provenance was."""
    executor_module = _load_executor_module()
    mock_bus = MagicMock()
    mock_bus.publish = AsyncMock()

    valid_entry = CollapseMirrorEntryV2(
        event_id="evt-relational",
        id="evt-relational",
        trigger="relational",
        observer="Orion",
        observer_state=["strained"],
        type="flow",
        emergent_entity="Relational Pulse",
        summary="Test summary",
        mantra="Test mantra",
        field_resonance="Test resonance",
        resonance_signature="Test sig",
        source_service="metacog",
    ).model_dump(mode="json")
    valid_entry["state_snapshot"] = {"telemetry": {"metacog_draft_mode": "llm"}}

    ctx = {
        "trigger": {"trigger_kind": "relational", "reason": "relational_shift:repair:confidence=0.90"},
        "trigger_kind": "relational",
        "metadata": {
            "substrate_effect_summary": {
                "level": 0.9,
                "level_label": "HIGH",
                "confidence": 0.9,
                "evidence": [{"evidence_kind": "trust_rupture", "score": 0.8, "confidence": 0.9}],
                "behavior_applied": "acknowledge_and_repair",
            }
        },
        "collapse_entry": valid_entry,
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
    envelope = mock_bus.publish.call_args[0][1]
    payload = envelope.payload
    assert payload["touches"] == ["relational"]
    assert payload["provenance"]["source"] == "cortex_exec.metacog_pipeline.relational"
    assert payload["provenance"]["impacts"] == ["relationship_thread"]
    assert payload["state"]["repair_pressure"]["level"] == pytest.approx(0.9)
    assert payload["state"]["repair_pressure"]["evidence"][0]["evidence_kind"] == "trust_rupture"


def test_publish_output_unaffected_by_enrich_removal_end_to_end(monkeypatch):
    """Regression for the 2026-07-28 Enrich removal: run the real single-pass
    Draft -> Publish pipeline (no Enrich step at all, matching the trimmed
    verb yaml) and confirm MetacogPublishService still builds the same real
    MetacogEntryV1 fields (severity/touches/causal_density/provenance/state),
    sourced entirely from ctx artifacts plus Draft's leaner patch -- not from
    anything Enrich used to produce."""
    executor_module = _load_executor_module()
    mock_bus = MagicMock()
    mock_bus.publish = AsyncMock()

    class FakeLLMClient:
        def __init__(self, bus):
            self.bus = bus

        async def chat(self, **kwargs):
            return {
                "summary": "Steady coherence, slight clarity uptick.",
                "mantra": "Hold the signal.",
                "what_changed": {
                    "summary": "clarity up",
                    "evidence": ["spark clarity band high"],
                },
                "tags_suggested": ["mode:mirror"],
            }

    monkeypatch.setattr(executor_module, "LLMGatewayClient", FakeLLMClient)

    template = _load_template("log_orion_metacognition_draft.j2")
    ctx = _draft_ctx()
    ctx["trigger"] = {"trigger_kind": "dense", "reason": "substrate_eventfulness:0.60", "pressure": 0.6, "zen_state": "not_zen"}
    ctx["trigger_kind"] = "dense"
    ctx["substrate_eventfulness_score"] = 0.6
    ctx["substrate_eventfulness_reasons"] = ["execution_pressure_spike"]

    source = ServiceRef(name="test", node="test", version="1.0")

    draft_step = ExecutionStep(
        verb_name="log_orion_metacognition",
        step_name="draft_entry",
        order=0,
        services=["MetacogDraftService"],
        prompt_template=template,
    )
    draft_result = asyncio.run(
        executor_module.call_step_services(
            bus=object(),
            source=source,
            step=draft_step,
            ctx=ctx,
            correlation_id=str(uuid4()),
        )
    )
    assert draft_result.status == "success"
    assert ctx["collapse_entry"]["summary"] == "Steady coherence, slight clarity uptick."
    # Enrich no longer runs -- nothing ever sets ctx["final_entry"].
    assert "final_entry" not in ctx

    publish_step = ExecutionStep(
        step_name="publish",
        verb_name="log_orion_metacognition",
        services=["MetacogPublishService"],
        order=1,
    )
    publish_result = asyncio.run(
        executor_module.call_step_services(
            bus=mock_bus,
            source=source,
            step=publish_step,
            ctx=ctx,
            correlation_id=str(uuid4()),
        )
    )

    assert publish_result.status == "success"
    publish = publish_result.result["MetacogPublishService"]
    assert publish.get("published") is True
    mock_bus.publish.assert_called_once()
    envelope = mock_bus.publish.call_args[0][1]
    payload = envelope.payload

    # Same real fields as the pre-Enrich-removal contract, still sourced from
    # real ctx artifacts, not from anything an Enrich step would have produced.
    assert payload["summary"] == "Steady coherence, slight clarity uptick."
    assert payload["mantra"] == "Hold the signal."
    assert "numeric_sisters" not in payload
    assert payload["trigger_kind"] == "dense"
    assert payload["state"]["substrate_eventfulness_score"] == 0.6
    assert payload["causal_density"]["score"] == pytest.approx(0.6)
    assert payload["is_causally_dense"] is True
    assert payload["snapshot_kind"] == "confirmed_dense"
    assert payload["touches"] == ["substrate"]
    assert payload["severity"] == "nominal"
    assert payload["provenance"]["source"] == "cortex_exec.metacog_pipeline.dense"
    assert payload["provenance"]["impacts"] == ["execution_trajectory"]
    assert isinstance(payload["what_changed"]["evidence"], list)


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
