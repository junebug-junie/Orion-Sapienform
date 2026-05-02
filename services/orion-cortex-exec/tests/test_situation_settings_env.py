from __future__ import annotations

from app.settings import Settings


def test_cortex_situation_enabled_false_values(monkeypatch):
    for value in ("false", "False", "0", "no"):
        monkeypatch.setenv("ORION_SITUATION_ENABLED", value)
        settings = Settings()
        assert settings.orion_situation_enabled is False


def test_cortex_situation_enabled_true_values(monkeypatch):
    for value in ("true", "1"):
        monkeypatch.setenv("ORION_SITUATION_ENABLED", value)
        settings = Settings()
        assert settings.orion_situation_enabled is True


def test_cortex_situation_enabled_defaults_true(monkeypatch):
    monkeypatch.delenv("ORION_SITUATION_ENABLED", raising=False)
    settings = Settings()
    assert settings.orion_situation_enabled is True
