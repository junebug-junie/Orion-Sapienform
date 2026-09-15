"""Keep harness outer waits and verb LLM budgets in lockstep.

Live 2026-09-15: curiosity finalize LLM calls took 264-295s while
FINALIZE_REFLECT_TIMEOUT_SEC was still 180s — harness abandoned mid-gen.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_harness_settings_cls():
    path = Path(__file__).resolve().parents[1] / "app" / "settings.py"
    spec = importlib.util.spec_from_file_location("harness_governor_settings", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.HarnessGovernorSettings


def _field_default(name: str) -> float:
    return float(_load_harness_settings_cls().model_fields[name].default)


def test_finalize_reflect_timeout_covers_verb_budget() -> None:
    reflect_sec = _field_default("finalize_reflect_timeout_sec")
    verb = yaml.safe_load(
        (_repo_root() / "orion/cognition/verbs/harness_finalize_reflect.yaml").read_text()
    )
    assert reflect_sec >= verb["timeout_ms"] / 1000.0
    assert reflect_sec >= 480.0


def test_response_repair_timeout_covers_verb_budget() -> None:
    repair_sec = _field_default("response_repair_timeout_sec")
    verb = yaml.safe_load(
        (_repo_root() / "orion/cognition/verbs/orion_response_repair.yaml").read_text()
    )
    assert repair_sec >= verb["timeout_ms"] / 1000.0
    assert repair_sec >= 540.0
