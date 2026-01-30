from __future__ import annotations

import sys
from pathlib import Path
import importlib.util
import types

scripts_dir = Path(__file__).resolve().parent
repo_root = scripts_dir.parent
if str(scripts_dir) in sys.path:
    sys.path.remove(str(scripts_dir))
sys.path.insert(0, str(repo_root))

from orion.schemas.telemetry.cognition_trace import CognitionTracePayload


def _load_worker_module():
    worker_path = repo_root / "services" / "orion-spark-introspector" / "app" / "worker.py"
    package_name = "orion_spark_introspector"
    app_package_name = f"{package_name}.app"
    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [str(worker_path.parent.parent)]
        sys.modules[package_name] = pkg
    if app_package_name not in sys.modules:
        pkg = types.ModuleType(app_package_name)
        pkg.__path__ = [str(worker_path.parent)]
        sys.modules[app_package_name] = pkg
    spec = importlib.util.spec_from_file_location(f"{app_package_name}.worker", worker_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    worker_module = _load_worker_module()
    trace = CognitionTracePayload(
        mode="heartbeat",
        verb="equilibrium_heartbeat",
        metadata={"heartbeat": True},
        timestamp=0.0,
    )
    assert worker_module._is_heartbeat_trace(trace) is True
    print("ok")


if __name__ == "__main__":
    main()
