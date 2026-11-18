"""Relational names come from the local .env, never from code (privacy)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_guard_path = Path(__file__).resolve().parent / "_mind_import_guard.py"


def _engine():
    spec = importlib.util.spec_from_file_location("_mind_guard_lazy", _guard_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ensure_orion_mind_app()
    from app import engine

    return engine


def test_relational_names_come_from_env_not_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real names of people close to Juniper are personal data: they live in the
    local .env (ORION_MIND_RELATIONAL_NAMES), not in engine.py."""
    engine = _engine()
    frame = engine._deterministic_frame_from_user_text
    monkeypatch.delenv("ORION_MIND_RELATIONAL_NAMES", raising=False)
    assert frame("dinner with Robin")[0] == "mixed"
    monkeypatch.setenv("ORION_MIND_RELATIONAL_NAMES", " Robin , Pip,")
    assert frame("dinner with Robin")[0] == "playful_relational"
    assert frame("pip is home")[0] == "playful_relational"
    assert frame("robinson crusoe")[0] == "mixed"
