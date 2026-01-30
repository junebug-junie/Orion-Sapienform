import importlib.util
import sys
import types
from pathlib import Path

import pytest

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
