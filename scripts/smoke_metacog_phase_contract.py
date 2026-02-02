import importlib
import importlib.util
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


def _ensure_platform_system():
    module = sys.modules.get("platform")
    if module is None:
        module = types.ModuleType("platform")
        sys.modules["platform"] = module
    if not hasattr(module, "system"):
        try:
            std_platform = importlib.import_module("platform")
            module.system = std_platform.system
        except Exception:
            module.system = lambda: "unknown"


def _load_executor_module():
    app_dir = REPO_ROOT / "services" / "orion-cortex-exec" / "app"
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


def main() -> None:
    _ensure_platform_system()
    executor_module = _load_executor_module()

    base = {
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

    try:
        draft_patch = executor_module.MetacogDraftTextPatchV1.model_validate(
            {"summary": "new summary", "tags_suggested": ["tag"], "tag_scores": {"bad": 1.0}}
        )
    except Exception:
        draft_patch = executor_module.MetacogDraftTextPatchV1()

    executor_module._apply_draft_patch(base, draft_patch)
    updated = executor_module._apply_metacog_system_fields(
        base,
        {"trigger_correlation_id": "corr-1", "metacog_entry_id": "collapse_1"},
    )
    assert updated["id"] == "collapse_1"

    try:
        enrich_patch = executor_module.MetacogEnrichScorePatchV1.model_validate(
            {"tag_scores": {"x": 1.0}, "summary": "nope"}
        )
    except Exception:
        enrich_patch = executor_module.MetacogEnrichScorePatchV1()

    executor_module._apply_enrich_patch(updated, enrich_patch)
    assert "summary" in updated
    print("ok")


if __name__ == "__main__":
    main()
