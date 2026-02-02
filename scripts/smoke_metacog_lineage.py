import importlib
import importlib.util
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


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


if __name__ == "__main__":
    _ensure_platform_system()

    executor_module = _load_executor_module()
    ctx = {
        "trigger_correlation_id": "corr-123",
        "trigger_trace_id": "trace-123",
    }
    entry = executor_module._fallback_metacog_draft(ctx)
    print(f"metacog_entry_id={entry.id}")
    print(f"trigger_correlation_id={ctx.get('trigger_correlation_id')}")
    assert entry.id != ctx.get("trigger_correlation_id"), "metacog id must differ from trigger correlation"
    print("ok")
