import importlib.util
import sys
import types
from pathlib import Path


def _load_worker_module():
    repo_root = Path(__file__).resolve().parents[1]
    app_dir = repo_root / "services" / "orion-spark-introspector" / "app"
    worker_path = app_dir / "worker.py"
    package_name = "orion_spark_introspector"
    app_package_name = f"{package_name}.app"
    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [str(app_dir.parent)]
        sys.modules[package_name] = pkg
    if app_package_name not in sys.modules:
        pkg = types.ModuleType(app_package_name)
        pkg.__path__ = [str(app_dir)]
        sys.modules[app_package_name] = pkg
    spec = importlib.util.spec_from_file_location(f"{app_package_name}.worker", worker_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_publishable_channel_guard_blocks_wildcards():
    worker_module = _load_worker_module()
    assert worker_module._is_publishable_channel("orion:spark:introspect:candidate*") is False
    assert worker_module._is_publishable_channel("orion:spark:introspect:candidate") is True


def test_append_turn_effect_metadata_includes_effect():
    worker_module = _load_worker_module()
    meta = {}
    spark_meta = {
        "phi_before": {"valence": 0.2},
        "phi_after": {"valence": 0.4},
    }
    worker_module._append_turn_effect_metadata(meta, spark_meta)
    assert meta["turn_effect"]["user"]["valence"] == 0.2
    assert meta["turn_effect_summary"]
