import importlib.util
import sys
import types
from pathlib import Path


def _load_core_event_cache():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "services" / "orion-cortex-exec" / "app" / "core_event_cache.py"
    package_name = "orion_cortex_exec"
    app_package_name = f"{package_name}.app"
    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [str(module_path.parent.parent)]
        sys.modules[package_name] = pkg
    if app_package_name not in sys.modules:
        pkg = types.ModuleType(app_package_name)
        pkg.__path__ = [str(module_path.parent)]
        sys.modules[app_package_name] = pkg
    spec = importlib.util.spec_from_file_location(f"{app_package_name}.core_event_cache", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def test_core_event_cache_filters_turn_effect_alerts():
    module = _load_core_event_cache()
    cache = module.CoreEventCache(maxlen=5)
    cache.append(
        {
            "event": "notify",
            "payload": {
                "event_type": "turn_effect_alert",
                "title": "Turn effect alert: coherence_drop",
                "body": "summary",
                "severity": "warn",
                "correlation_id": "corr-1",
                "trace_id": "trace-1",
                "metadata": {"rule": "coherence_drop", "value": -0.3, "threshold": -0.25},
            },
        }
    )
    cache.append({"event": "notify", "payload": {"event_type": "other_event"}})
    alerts = cache.get_recent_turn_effect_alerts(5)
    assert len(alerts) == 1
    assert alerts[0]["rule"] == "coherence_drop"
    assert alerts[0]["value"] == -0.3
    assert alerts[0]["corr_id"] == "corr-1"
    assert alerts[0]["trace_id"] == "trace-1"


def test_format_recent_turn_effect_alerts_summary():
    module = _load_core_event_cache()
    summary = module.format_recent_turn_effect_alerts(
        [{"rule": "novelty_spike", "value": 0.42, "severity": "info", "corr_id": "c1", "trace_id": "t1"}]
    )
    assert summary == "Recent Alerts: 1 (last: novelty_spike 0.420 info corr=c1 trace=t1)"


def test_system_alert_tags_merge():
    executor_module = _load_executor_module()
    system_tags = executor_module._alert_tags_from_recent_alerts(
        [{"rule": "coherence_drop", "severity": "error"}, {"rule": "valence_drop", "severity": "warn"}]
    )
    assert system_tags == [
        "metacog.alert.coherence_drop",
        "metacog.alert.sev.error",
        "metacog.alert.sev.warn",
        "metacog.alert.valence_drop",
    ]
    merged = executor_module._merge_system_tags(["user_tag"], system_tags)
    assert merged[:1] == ["user_tag"]
    assert all(tag in merged for tag in system_tags)
