from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


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


def test_format_pad_frame_summary_prefers_signal_lists():
    executor = _load_executor_module()

    summary = executor._format_pad_frame_summary(
        {
            "frame": {
                "ts_ms": 1710000000000,
                "window_ms": 5000,
                "summary": {
                    "top_signals": ["user urgency", "planner conflict", "memory recall"],
                    "active_tasks": ["metacog_log", "recall"],
                    "risk_flags": ["overload_risk"],
                },
                "salient_event_ids": ["evt-1", "evt-2"],
            }
        }
    )

    assert "signals=user urgency; planner conflict; memory recall" in summary
    assert "tasks=metacog_log; recall" in summary
    assert "risks=overload_risk" in summary
    assert "salient_events=2" in summary


def test_format_pad_stats_summary_highlights_counters_and_last_frame():
    executor = _load_executor_module()

    summary = executor._format_pad_stats_summary(
        {
            "stats": {
                "ingested": 14,
                "dropped_total": 2,
                "frames_built": 3,
                "rpc_requests": 9,
                "rpc_errors": 1,
                "queue_depth": 4,
                "last_salience": 0.8123,
                "last_frame_ts_ms": 1710000000000,
            }
        }
    )

    assert "ingested=14" in summary
    assert "dropped_total=2" in summary
    assert "frames_built=3" in summary
    assert "rpc_requests=9" in summary
    assert "rpc_errors=1" in summary
    assert "queue_depth=4" in summary
    assert "last_salience=0.81" in summary
    assert "last_frame=2024-03-09T16:00:00+00:00" in summary
