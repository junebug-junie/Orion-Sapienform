"""Tier 1 patch: renders the real log_orion_metacognition_draft.j2 template
with a ctx containing `recent_trend_signals_json` and asserts the new RECENT
TREND SIGNALS section shows up with the real value -- a render-shape test,
not just an assertion that the ctx key was set.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXEC_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / "orion" / "cognition" / "prompts" / "log_orion_metacognition_draft.j2"


def _load_executor_module():
    app_dir = EXEC_ROOT / "app"
    executor_path = app_dir / "executor.py"
    package_name = "orion_cortex_exec_trend_cue_render"
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


def _base_ctx() -> dict:
    return {
        "trigger": {
            "trigger_kind": "baseline",
            "reason": "scheduled_check",
            "pressure": 0.2,
            "zen_state": "calm",
        },
    }


def test_recent_trend_signals_section_renders_real_data():
    executor = _load_executor_module()
    cue = {
        "prediction_error": {
            "node:substrate.execution": 0.297,
            "node:substrate.biometrics": 0.155,
        },
        "biometrics_induction": {
            "athena": {
                "channel": "cpu",
                "trend": 0.505,
                "level": 0.31,
                "spike_rate": 0.0,
                "volatility": 0.063,
            },
        },
        "as_of": "2026-07-30T02:28:44+00:00",
    }
    ctx = _base_ctx()
    ctx["recent_trend_signals_json"] = json.dumps(cue, indent=2)

    prompt = executor._render_prompt(TEMPLATE_PATH.read_text(encoding="utf-8"), ctx)

    assert "RECENT TREND SIGNALS" in prompt
    assert '"node:substrate.execution": 0.297' in prompt
    assert '"channel": "cpu"' in prompt
    assert '"trend": 0.505' in prompt


def test_recent_trend_signals_section_renders_empty_object_when_absent():
    executor = _load_executor_module()
    ctx = _base_ctx()
    ctx["recent_trend_signals_json"] = "{}"

    prompt = executor._render_prompt(TEMPLATE_PATH.read_text(encoding="utf-8"), ctx)

    assert "RECENT TREND SIGNALS" in prompt
    assert "{}" in prompt


def test_template_does_not_crash_when_key_missing():
    """Matches the sibling recent_turn_effect_alerts_json block's own convention
    of having no defensive default in _render_prompt's `defaults` dict --
    Jinja's default Undefined renders as an empty string rather than raising."""
    executor = _load_executor_module()
    ctx = _base_ctx()

    prompt = executor._render_prompt(TEMPLATE_PATH.read_text(encoding="utf-8"), ctx)

    assert "RECENT TREND SIGNALS" in prompt
