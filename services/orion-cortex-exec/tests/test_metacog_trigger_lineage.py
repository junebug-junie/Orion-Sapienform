from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXEC_ROOT = Path(__file__).resolve().parents[1]


def _load_executor_module():
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    if str(EXEC_ROOT) not in sys.path:
        sys.path.append(str(EXEC_ROOT))
    app_dir = EXEC_ROOT / "app"
    executor_path = app_dir / "executor.py"
    package_name = "orion_cortex_exec_lineage"
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


def test_metacog_trigger_lineage_passes_baseline_from_trigger_payload():
    executor = _load_executor_module()
    ctx = {
        "trigger": {"trigger_kind": "baseline", "reason": "scheduled_check"},
        "correlation_id": "corr-1",
    }
    lineage = executor._metacog_trigger_lineage(ctx)
    assert lineage["trigger_kind"] == "baseline"


def test_metacog_trigger_lineage_passes_dense_from_trigger_payload():
    executor = _load_executor_module()
    ctx = {
        "trigger": {"trigger_kind": "dense", "reason": "substrate_eventfulness:0.60"},
    }
    lineage = executor._metacog_trigger_lineage(ctx)
    assert lineage["trigger_kind"] == "dense"


def test_metacog_trigger_lineage_chat_turn_overrides_trigger_kind():
    executor = _load_executor_module()
    ctx = {
        "trigger": {"trigger_kind": "baseline"},
        "chat_correlation_id": "chat-99",
    }
    lineage = executor._metacog_trigger_lineage(ctx)
    assert lineage["trigger_kind"] == "chat_turn"
