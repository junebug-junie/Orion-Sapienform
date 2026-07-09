import importlib


def _reload_settings(monkeypatch):
    """Reload orion-thought settings hermetically.

    settings.py calls load_dotenv() at import time, which would pull the operator's
    local services/orion-thought/.env into these tests and make them machine-
    dependent. Neutralize it so we test field defaults / explicit env only.
    """
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    import app.settings as s

    importlib.reload(s)
    return s


def test_mind_enrichment_defaults_off(monkeypatch):
    for key in (
        "ORION_THOUGHT_MIND_ENRICHMENT_ENABLED",
        "ORION_THOUGHT_MIND_ARTIFACT_PUBLISH_ENABLED",
        "ORION_THOUGHT_MIND_TIMEOUT_SEC",
        "ORION_THOUGHT_MIND_WALL_MS",
        "ORION_THOUGHT_MIND_ROUTER_PROFILE",
        "ORION_THOUGHT_MIND_MAX_RESPONSE_BYTES",
        "ORION_THOUGHT_MIND_COLORING_MAX_ITEMS",
        "ORION_MIND_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)
    s = _reload_settings(monkeypatch)
    assert s.settings.mind_enrichment_enabled is False
    assert s.settings.mind_artifact_publish_enabled is False
    assert s.settings.mind_timeout_sec == 210.0
    assert s.settings.mind_wall_ms == 180000
    assert s.settings.mind_router_profile == "default"
    assert s.settings.mind_max_response_bytes == 2_000_000
    assert s.settings.mind_coloring_max_items == 3
    assert s.settings.mind_base_url == "http://orion-mind:6611"
    assert s.settings.channel_mind_artifact == "orion:mind:artifact"
    assert s.MIND_LLM_TIMEOUT_SEC_ASSUMED == 60.0
    assert s.MIND_ENRICHMENT_MIN_VIABLE_WALL_MS == 180000


def test_mind_enrichment_reads_env(monkeypatch):
    monkeypatch.setenv("ORION_THOUGHT_MIND_ENRICHMENT_ENABLED", "true")
    monkeypatch.setenv("ORION_THOUGHT_MIND_WALL_MS", "9000")
    s = _reload_settings(monkeypatch)
    assert s.settings.mind_enrichment_enabled is True
    assert s.settings.mind_wall_ms == 9000


def test_default_wall_is_viable_for_three_phase_synthesis(monkeypatch):
    """Regression for fix/mind-enrichment-wall-budget: the shipped default wall must
    fit 3 sequential LLM phases, else synthesis always degrades to contract_only.
    Asserts the invariant, not a magic literal, so lowering the wall below viability
    fails loudly."""
    for key in ("ORION_THOUGHT_MIND_WALL_MS", "ORION_THOUGHT_MIND_TIMEOUT_SEC"):
        monkeypatch.delenv(key, raising=False)
    s = _reload_settings(monkeypatch)
    assert s.settings.mind_wall_ms >= s.MIND_ENRICHMENT_MIN_VIABLE_WALL_MS
    # HTTP read timeout must exceed the internal wall so Mind's fail-open result
    # is returned rather than the client aborting first.
    assert s.settings.mind_timeout_sec * 1000.0 > s.settings.mind_wall_ms


def test_config_warnings_silent_when_disabled(monkeypatch):
    monkeypatch.delenv("ORION_THOUGHT_MIND_ENRICHMENT_ENABLED", raising=False)
    monkeypatch.setenv("ORION_THOUGHT_MIND_WALL_MS", "1000")
    s = _reload_settings(monkeypatch)
    assert s.settings.mind_enrichment_enabled is False
    assert s.mind_enrichment_config_warnings(s.settings) == []


def test_config_warns_on_sub_viable_wall(monkeypatch):
    monkeypatch.setenv("ORION_THOUGHT_MIND_ENRICHMENT_ENABLED", "true")
    monkeypatch.setenv("ORION_THOUGHT_MIND_WALL_MS", "12000")
    monkeypatch.setenv("ORION_THOUGHT_MIND_TIMEOUT_SEC", "15")
    s = _reload_settings(monkeypatch)
    warnings = s.mind_enrichment_config_warnings(s.settings)
    assert any("wall_too_small" in w for w in warnings)


def test_config_warns_on_http_timeout_not_above_wall(monkeypatch):
    monkeypatch.setenv("ORION_THOUGHT_MIND_ENRICHMENT_ENABLED", "true")
    monkeypatch.setenv("ORION_THOUGHT_MIND_WALL_MS", "180000")
    monkeypatch.setenv("ORION_THOUGHT_MIND_TIMEOUT_SEC", "30")
    s = _reload_settings(monkeypatch)
    warnings = s.mind_enrichment_config_warnings(s.settings)
    assert any("http_timeout_not_above_wall" in w for w in warnings)


def test_shipped_defaults_produce_no_warnings(monkeypatch):
    for key in ("ORION_THOUGHT_MIND_WALL_MS", "ORION_THOUGHT_MIND_TIMEOUT_SEC"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ORION_THOUGHT_MIND_ENRICHMENT_ENABLED", "true")
    s = _reload_settings(monkeypatch)
    assert s.mind_enrichment_config_warnings(s.settings) == []
