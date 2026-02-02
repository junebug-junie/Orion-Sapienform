import importlib.util
import logging
import sys
import types
from pathlib import Path

from orion.schemas.telemetry.turn_effect import should_emit_turn_effect_alert


def _load_worker_module():
    repo_root = Path(__file__).resolve().parents[1]
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


def test_turn_effect_alert_audit_log(caplog):
    worker_module = _load_worker_module()
    caplog.set_level(logging.INFO, logger="orion-spark-introspector")
    worker_module._log_turn_effect_alert(
        rule="coherence_drop",
        value=-0.4,
        severity="warn",
        corr_id="corr-1",
        trace_id="trace-1",
        summary="turn: c-0.40",
        cooldown_sec=120,
    )
    assert "[turn_effect_alert] fired" in caplog.text
    assert "rule=coherence_drop" in caplog.text
    assert "value=-0.400" in caplog.text
    assert "corr_id=corr-1" in caplog.text
    assert "trace_id=trace-1" in caplog.text


def test_turn_effect_alert_cooldown_suppressed_log(caplog):
    worker_module = _load_worker_module()
    caplog.set_level(logging.DEBUG, logger="orion-spark-introspector")
    worker_module._log_turn_effect_alert_suppressed(key="abc", remaining=12.3)
    assert "[turn_effect_alert] suppressed cooldown" in caplog.text


def test_turn_effect_alert_cooldown_blocks_second_emit():
    now = 100.0
    assert should_emit_turn_effect_alert(None, now, 120) is True
    assert should_emit_turn_effect_alert(now, now + 1, 120) is False


def test_turn_effect_alert_heartbeat_block():
    worker_module = _load_worker_module()
    assert worker_module._is_turn_effect_alert_blocked("heartbeat", None) is True
    assert worker_module._is_turn_effect_alert_blocked("brain", "heartbeat") is True
    assert worker_module._is_turn_effect_alert_blocked("brain", None) is False


def test_turn_effect_alert_severity_mapping():
    worker_module = _load_worker_module()
    assert worker_module._alert_severity("coherence_drop", -0.6, -0.25) == "error"
    assert worker_module._alert_severity("valence_drop", -0.2, -0.25) == "info"
    assert worker_module._alert_severity("novelty_spike", 0.8, 0.35) == "warn"


def test_turn_effect_alert_dedupe_window():
    worker_module = _load_worker_module()
    now = 100.0
    assert worker_module._is_dedupe_suppressed(None, now, 10) is False
    assert worker_module._is_dedupe_suppressed(now, now + 1, 10) is True
