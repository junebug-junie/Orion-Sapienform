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
    filtered, stripped = executor_module._filter_patch_dict(
        raw,
        executor_module._METACOG_DRAFT_ALLOWED_KEYS,
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
