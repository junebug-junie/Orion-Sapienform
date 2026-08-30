from __future__ import annotations

from app.settings import Settings


def test_situation_enabled_false_values(monkeypatch):
    for value in ("false", "False", "0", "no"):
        monkeypatch.setenv("ORION_SITUATION_ENABLED", value)
        settings = Settings()
        assert settings.ORION_SITUATION_ENABLED is False


def test_situation_enabled_true_values(monkeypatch):
    for value in ("true", "1"):
        monkeypatch.setenv("ORION_SITUATION_ENABLED", value)
        settings = Settings()
        assert settings.ORION_SITUATION_ENABLED is True


def test_situation_enabled_defaults_true(monkeypatch):
    monkeypatch.delenv("ORION_SITUATION_ENABLED", raising=False)
    settings = Settings()
    assert settings.ORION_SITUATION_ENABLED is True


def test_situation_curiosity_and_reverie_default_true(monkeypatch):
    # Deliberate contrast with perception/lab above: ON by default per
    # Juniper's explicit request (2026-08-30), not opt-in.
    monkeypatch.delenv("ORION_SITUATION_CURIOSITY_ENABLED", raising=False)
    monkeypatch.delenv("ORION_SITUATION_REVERIE_ENABLED", raising=False)
    settings = Settings()
    assert settings.ORION_SITUATION_CURIOSITY_ENABLED is True
    assert settings.ORION_SITUATION_REVERIE_ENABLED is True


def test_situation_curiosity_and_reverie_can_be_disabled(monkeypatch):
    monkeypatch.setenv("ORION_SITUATION_CURIOSITY_ENABLED", "false")
    monkeypatch.setenv("ORION_SITUATION_REVERIE_ENABLED", "false")
    settings = Settings()
    assert settings.ORION_SITUATION_CURIOSITY_ENABLED is False
    assert settings.ORION_SITUATION_REVERIE_ENABLED is False
