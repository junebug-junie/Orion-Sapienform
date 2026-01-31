import importlib.util
import json
import sys
import types
from pathlib import Path

from orion.schemas.metacog_patches import (
    MetacogDraftTextPatchV1,
    MetacogDraftWhatChangedV1,
    MetacogEnrichScorePatchV1,
    MetacogNumericSistersV1,
)


def _allowed_keys(model_cls):
    return set(model_cls.model_fields.keys())


def _read_template(name: str) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / "orion" / "cognition" / "prompts" / name).read_text()


def _extract_example_json(text: str) -> str:
    start_tag = "<example_json>"
    end_tag = "</example_json>"
    start = text.find(start_tag)
    end = text.find(end_tag)
    assert start != -1, "example_json start tag missing"
    assert end != -1, "example_json end tag missing"
    return text[start + len(start_tag):end].strip()


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


def test_draft_prompt_requires_patch_only():
    text = _read_template("log_orion_metacognition_draft.j2")
    example = _extract_example_json(text)
    assert "FULL CollapseMirrorEntryV2" not in text
    example_payload = json.loads(example)
    assert set(example_payload.keys()) <= _allowed_keys(MetacogDraftTextPatchV1)
    what_changed = example_payload.get("what_changed") or {}
    assert set(what_changed.keys()) <= _allowed_keys(MetacogDraftWhatChangedV1)


def test_enrich_prompt_requires_patch_only():
    text = _read_template("log_orion_metacognition_enrich.j2")
    example = _extract_example_json(text)
    assert "FULL CollapseMirrorEntryV2" not in text
    example_payload = json.loads(example)
    assert set(example_payload.keys()) <= _allowed_keys(MetacogEnrichScorePatchV1)
    numeric_sisters = example_payload.get("numeric_sisters") or {}
    assert set(numeric_sisters.keys()) <= _allowed_keys(MetacogNumericSistersV1)


def test_metacog_wrapper_instruction_is_patch_only():
    executor_module = _load_executor_module()
    messages = executor_module._metacog_messages(
        "prompt",
        allowed_keys=_allowed_keys(MetacogDraftTextPatchV1),
        phase="draft",
    )
    instruction = messages[1]["content"]
    assert "CollapseMirrorEntryV2" not in instruction
    assert "MetacogDraftTextPatchV1" not in instruction
