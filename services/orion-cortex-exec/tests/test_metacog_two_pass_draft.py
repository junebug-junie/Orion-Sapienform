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
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_uncertainty_probe_enabled", False)
    assert executor_module._should_run_metacog_uncertainty_probe() is False
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_uncertainty_probe_enabled", True)
    assert executor_module._should_run_metacog_uncertainty_probe() is True
