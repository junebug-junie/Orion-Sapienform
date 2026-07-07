from app.settings import Settings


def test_curiosity_defaults_off():
    s = Settings()
    assert s.world_pulse_curiosity_fetch_enabled is False
    assert s.world_pulse_curiosity_max_articles_per_section == 5
    assert s.world_pulse_curiosity_max_sections == 9


def test_curiosity_env_override(monkeypatch):
    monkeypatch.setenv("WORLD_PULSE_CURIOSITY_FETCH_ENABLED", "true")
    monkeypatch.setenv("WORLD_PULSE_CURIOSITY_MAX_ARTICLES_PER_SECTION", "3")
    s = Settings()
    assert s.world_pulse_curiosity_fetch_enabled is True
    assert s.world_pulse_curiosity_max_articles_per_section == 3
