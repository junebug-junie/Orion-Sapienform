import importlib


def test_salience_flags_default_off(monkeypatch):
    monkeypatch.delenv("ORION_ATTENTION_SALIENCE_V2_ENABLED", raising=False)
    monkeypatch.delenv("ORION_ATTENTION_HABITUATION_ENABLED", raising=False)
    import app.settings as s
    importlib.reload(s)
    assert s.settings.attention_salience_v2_enabled is False
    assert s.settings.attention_habituation_enabled is False
