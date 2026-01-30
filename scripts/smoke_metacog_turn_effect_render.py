from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

scripts_dir = Path(__file__).resolve().parent
repo_root = scripts_dir.parent
if str(scripts_dir) in sys.path:
    sys.path.remove(str(scripts_dir))
sys.path.insert(0, str(repo_root))

from orion.schemas.collapse_mirror import normalize_collapse_entry


def _load_executor_module():
    executor_path = repo_root / "services" / "orion-cortex-exec" / "app" / "executor.py"
    package_name = "orion_cortex_exec"
    app_package_name = f"{package_name}.app"
    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [str(executor_path.parent.parent)]
        sys.modules[package_name] = pkg
    if app_package_name not in sys.modules:
        pkg = types.ModuleType(app_package_name)
        pkg.__path__ = [str(executor_path.parent)]
        sys.modules[app_package_name] = pkg
    spec = importlib.util.spec_from_file_location(f"{app_package_name}.executor", executor_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    executor_module = _load_executor_module()
    turn_effect = {
        "user": {"valence": 0.1, "energy": -0.1},
        "assistant": {"coherence": 0.2},
    }
    summary = executor_module.summarize_turn_effect(turn_effect)
    parsed = {
        "observer": "orion",
        "trigger": "baseline",
        "observer_state": ["idle"],
        "type": "flow",
        "emergent_entity": "Self",
        "summary": "ok",
        "mantra": "om",
        "state_snapshot": {"telemetry": {}},
    }
    telemetry = executor_module._merge_telemetry_system_owned(parsed["state_snapshot"]["telemetry"], None)
    telemetry["turn_effect"] = turn_effect
    telemetry["turn_effect_summary"] = summary
    parsed["state_snapshot"]["telemetry"] = telemetry
    entry = normalize_collapse_entry(parsed)
    assert isinstance(entry.state_snapshot.telemetry, dict)
    assert isinstance(entry.state_snapshot.telemetry.get("turn_effect"), dict)
    assert isinstance(entry.state_snapshot.telemetry.get("turn_effect_summary"), str)
    assert entry.state_snapshot.telemetry["turn_effect_summary"]
    print("ok")


if __name__ == "__main__":
    main()
