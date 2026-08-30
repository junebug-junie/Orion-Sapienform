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


def test_situation_prompt_max_chars_defaults_to_7200(monkeypatch):
    # 2026-08-30, Juniper's explicit request: raised from 1200 (the field
    # previously did not exist on Hub's Settings at all -- the adapter
    # hardcoded 1200 with no env override, see
    # orion.situational.context.hub_settings_to_runtime_namespace).
    monkeypatch.delenv("ORION_SITUATION_PROMPT_MAX_CHARS", raising=False)
    settings = Settings()
    assert settings.ORION_SITUATION_PROMPT_MAX_CHARS == 7200


def test_situation_prompt_max_chars_is_configurable(monkeypatch):
    monkeypatch.setenv("ORION_SITUATION_PROMPT_MAX_CHARS", "9000")
    settings = Settings()
    assert settings.ORION_SITUATION_PROMPT_MAX_CHARS == 9000
