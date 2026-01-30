import importlib.util
import sys
import types
from pathlib import Path


def _load_fusion_module():
    repo_root = Path(__file__).resolve().parents[1]
    app_dir = repo_root / "services" / "orion-recall" / "app"
    fusion_path = app_dir / "fusion.py"
    package_name = "orion_recall"
    app_package_name = f"{package_name}.app"
    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [str(app_dir.parent)]
        sys.modules[package_name] = pkg
    if app_package_name not in sys.modules:
        pkg = types.ModuleType(app_package_name)
        pkg.__path__ = [str(app_dir)]
        sys.modules[app_package_name] = pkg
    spec = importlib.util.spec_from_file_location(f"{app_package_name}.fusion", fusion_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_turn_effect_boost_ranks_high_delta():
    fusion = _load_fusion_module()
    profile = {
        "profile": "reflect.alerts.v1",
        "backend_weights": {"sql_timeline": 1.0},
        "sql": {"enable_turn_effect_boost": True, "turn_effect_min_abs_delta": 0.2, "turn_effect_boost_weight": 0.5},
        "filters": {},
        "max_per_source": 10,
        "max_total_items": 10,
    }
    candidates = [
        {"id": "low", "source": "sql_timeline", "text": "low", "score": 0.1, "turn_effect_delta": 0.1},
        {"id": "high", "source": "sql_timeline", "text": "high", "score": 0.1, "turn_effect_delta": 0.4},
    ]
    bundle, _ = fusion.fuse_candidates(candidates=candidates, profile=profile, query_text="")
    assert bundle.items[0].id == "high"


def test_tag_prefix_boost_ranks_alert_tag():
    fusion = _load_fusion_module()
    profile = {
        "profile": "reflect.alerts.v1",
        "backend_weights": {"sql_timeline": 1.0},
        "filters": {"prefer_tags": ["metacog.alert."]},
        "max_per_source": 10,
        "max_total_items": 10,
    }
    candidates = [
        {"id": "plain", "source": "sql_timeline", "text": "plain", "score": 0.2, "tags": []},
        {"id": "alert", "source": "sql_timeline", "text": "alert", "score": 0.2, "tags": ["metacog.alert.sev.warn"]},
    ]
    bundle, _ = fusion.fuse_candidates(candidates=candidates, profile=profile, query_text="")
    assert bundle.items[0].id == "alert"
