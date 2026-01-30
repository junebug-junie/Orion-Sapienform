import importlib
import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "orion" / "schemas" / "collapse_mirror.py"


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


def _load_module():
    spec = importlib.util.spec_from_file_location("collapse_mirror", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    _ensure_platform_system()
    mod = _load_module()
    entry = mod.normalize_collapse_entry(
        {
            "observer": "orion",
            "trigger": "baseline",
            "observer_state": ["idle"],
            "type": "flow",
            "emergent_entity": "Test",
            "summary": "ok",
            "mantra": "keep",
            "state_snapshot": {
                "telemetry": {
                    "gpu_ mem": 0.5,
                    "gpu_ util": 0.9,
                    "phi_ hint": {"valence": 0.2, "energy": 0.4},
                }
            },
        }
    )
    telemetry = entry.state_snapshot.telemetry
    print("telemetry_keys", sorted(telemetry.keys()))
    print("phi_hint", telemetry.get("phi_hint"))
    assert "gpu_mem" in telemetry
    assert "gpu_util" in telemetry
    assert "phi_hint" in telemetry


if __name__ == "__main__":
    main()
