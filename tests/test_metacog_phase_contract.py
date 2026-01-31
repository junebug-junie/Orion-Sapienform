import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest

from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.cortex.schemas import ExecutionStep
from orion.schemas.metacog_patches import MetacogDraftTextPatchV1, MetacogEnrichScorePatchV1


def _load_executor_module():
    repo_root = Path(__file__).resolve().parents[1]
    app_dir = repo_root / "services" / "orion-cortex-exec" / "app"
    executor_path = app_dir / "executor.py"
    package_name = "orion_cortex_exec"
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


def _base_entry():
    return {
        "event_id": "collapse_1",
        "id": "collapse_1",
        "observer": "orion",
        "trigger": "baseline",
        "observer_state": ["idle"],
        "field_resonance": "calm",
        "type": "flow",
        "emergent_entity": "Test",
        "summary": "summary",
        "mantra": "mantra",
        "timestamp": None,
        "environment": None,
        "snapshot_kind": "baseline",
        "what_changed_summary": "delta",
        "what_changed": {"summary": "delta", "evidence": ["cue"]},
        "state_snapshot": {"telemetry": {"turn_effect": {"user": {"valence": 0.1}}}},
        "tags": ["flow"],
        "numeric_sisters": {},
        "causal_density": {},
        "is_causally_dense": False,
        "epistemic_status": "observed",
        "visibility": "internal",
        "redaction_level": "low",
        "source_service": "metacog",
        "source_node": None,
    }


def test_draft_patch_rejects_unknown_keys():
    with pytest.raises(Exception):
        MetacogDraftTextPatchV1.model_validate({"foo": "bar"})


def test_draft_patch_rejects_score_fields():
    with pytest.raises(Exception):
        MetacogDraftTextPatchV1.model_validate({"tag_scores": {"x": 1.0}})


def test_enrich_patch_rejects_text_fields():
    with pytest.raises(Exception):
        MetacogEnrichScorePatchV1.model_validate({"summary": "nope"})


def test_system_owned_fields_not_overwritten():
    executor_module = _load_executor_module()
    entry = _base_entry()
    ctx = {"trigger_correlation_id": "corr-1", "metacog_entry_id": "collapse_1"}
    patch = MetacogDraftTextPatchV1(summary="new summary", tags_suggested=["tag"])
    executor_module._apply_draft_patch(entry, patch)
    updated = executor_module._apply_metacog_system_fields(entry, ctx)
    assert updated["id"] == "collapse_1"
    telemetry = updated["state_snapshot"]["telemetry"]
    assert telemetry["turn_effect"]["user"]["valence"] == 0.1


def test_draft_patch_strips_unknown_keys_and_applies():
    executor_module = _load_executor_module()
    raw = {
        "summary": "updated summary",
        "observer": "orion",
        "source_service": "metacog",
        "visibility": "internal",
        "epistemic_status": "observed",
    }
    filtered, stripped = executor_module._sanitize_patch_payload(
        raw,
        model=MetacogDraftTextPatchV1,
    )
    assert "summary" in filtered
    assert "observer" in stripped
    patch = MetacogDraftTextPatchV1.model_validate(filtered)
    entry = _base_entry()
    executor_module._apply_draft_patch(entry, patch)
    assert entry["summary"] == "updated summary"


def test_trigger_string_and_type_present_when_patch_omits():
    executor_module = _load_executor_module()
    ctx = {"trigger_kind": "heartbeat", "trigger": {"trigger_kind": "heartbeat"}}
    base_entry = executor_module._fallback_metacog_draft(ctx).model_dump(mode="json")
    base_entry = executor_module._apply_metacog_system_fields(base_entry, ctx)
    entry = executor_module.normalize_collapse_entry(base_entry)
    assert isinstance(entry.trigger, str)
    assert entry.trigger
    assert entry.type


def test_fail_fast_skips_enrich_when_draft_fails(monkeypatch):
    executor_module = _load_executor_module()
    calls: list[str | None] = []

    class FakeLLMClient:
        def __init__(self, bus):
            self.bus = bus

        async def chat(self, **kwargs):
            req = kwargs.get("req")
            calls.append(getattr(req, "raw_user_text", None))
            return {}

    monkeypatch.setattr(executor_module, "LLMGatewayClient", FakeLLMClient)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(executor_module, "find_collapse_entry", _boom)

    step = ExecutionStep(
        verb_name="metacog",
        step_name="metacog",
        order=0,
        services=["MetacogDraftService", "MetacogEnrichService", "MetacogPublishService"],
        prompt_template="",
    )
    ctx = {"raw_user_text": "test", "trigger_kind": "heartbeat", "trigger": {"trigger_kind": "heartbeat"}}
    source = ServiceRef(name="test")

    result = asyncio.run(
        executor_module.call_step_services(
            bus=object(),
            source=source,
            step=step,
            ctx=ctx,
            correlation_id="corr-1",
        )
    )

    assert result.status == "fail"
    assert calls == ["test"]


def test_metacog_draft_telemetry_records_fallback():
    executor_module = _load_executor_module()
    entry = _base_entry()
    executor_module._set_metacog_draft_telemetry(
        entry,
        mode="fallback",
        rejected_keys=["observer"],
        error="no_json",
        raw_trigger_null=True,
    )
    telemetry = entry["state_snapshot"]["telemetry"]
    assert telemetry["metacog_draft_mode"] == "fallback"
    assert telemetry["metacog_draft_rejected_keys"] == ["observer"]
    assert telemetry["metacog_draft_error"] == "no_json"
    assert telemetry["metacog_draft_raw_trigger_null"] is True


def test_metacog_draft_telemetry_records_llm_success():
    executor_module = _load_executor_module()
    entry = _base_entry()
    executor_module._set_metacog_draft_telemetry(
        entry,
        mode="llm",
        rejected_keys=[],
        error=None,
        raw_trigger_null=False,
    )
    telemetry = entry["state_snapshot"]["telemetry"]
    assert telemetry["metacog_draft_mode"] == "llm"
    assert telemetry["metacog_draft_rejected_keys"] == []


def test_metacog_draft_overrides_fallback_summary_with_patch():
    executor_module = _load_executor_module()
    ctx = {"trigger_kind": "heartbeat", "trigger": {"trigger_kind": "heartbeat"}}
    base_entry = executor_module._fallback_metacog_draft(ctx).model_dump(mode="json")
    base_entry = executor_module._apply_metacog_system_fields(base_entry, ctx)
    patch = MetacogDraftTextPatchV1(what_changed={"summary": "clarity↑, overload↓"})
    executor_module._apply_draft_patch(base_entry, patch)
    executor_module._postprocess_metacog_draft_summary(base_entry, draft_mode="llm")
    entry = executor_module.normalize_collapse_entry(base_entry)
    assert entry.what_changed_summary == "clarity↑, overload↓"


def test_metacog_draft_keeps_fallback_summary_when_fallback():
    executor_module = _load_executor_module()
    ctx = {"trigger_kind": "heartbeat", "trigger": {"trigger_kind": "heartbeat"}}
    base_entry = executor_module._fallback_metacog_draft(ctx).model_dump(mode="json")
    base_entry = executor_module._apply_metacog_system_fields(base_entry, ctx)
    executor_module._postprocess_metacog_draft_summary(base_entry, draft_mode="fallback")
    entry = executor_module.normalize_collapse_entry(base_entry)
    assert entry.what_changed_summary == "fallback_generated"


def test_metacog_patch_sanitizer_handles_nested_keys_and_aliases():
    executor_module = _load_executor_module()
    raw = {
        "numeric_ sisters": {
            "risk_ score": 0.3,
            "constraints": {"severity_ score": 0.1, "extra_field": "drop"},
        },
        "summary": "drop",
    }
    sanitized, stripped = executor_module._sanitize_patch_payload(
        raw,
        model=MetacogEnrichScorePatchV1,
    )
    assert "summary" in stripped
    assert "numeric_sisters.constraints.extra_field" in stripped
    patch = MetacogEnrichScorePatchV1.model_validate(sanitized)
    assert patch.numeric_sisters
    assert patch.numeric_sisters.risk_score == 0.3
    assert patch.numeric_sisters.constraints.severity_score == 0.1


def test_metacog_patch_sanitizer_parses_code_fences():
    executor_module = _load_executor_module()
    raw = """```json
    {"summary": "delta", "what_changed": {"summary": "shift", "evidence": ["cue"]}}
    ```"""
    sanitized, stripped = executor_module._sanitize_patch_payload(
        raw,
        model=MetacogDraftTextPatchV1,
    )
    assert stripped == []
    patch = MetacogDraftTextPatchV1.model_validate(sanitized)
    assert patch.summary == "delta"
