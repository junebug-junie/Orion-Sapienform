import importlib


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
    import app.settings as s
    importlib.reload(s)
    assert s.settings.mind_enrichment_enabled is False
    assert s.settings.mind_artifact_publish_enabled is False
    assert s.settings.mind_timeout_sec == 15.0
    assert s.settings.mind_wall_ms == 12000
    assert s.settings.mind_router_profile == "default"
    assert s.settings.mind_max_response_bytes == 2_000_000
    assert s.settings.mind_coloring_max_items == 3
    assert s.settings.mind_base_url == "http://orion-mind:6611"
    assert s.settings.channel_mind_artifact == "orion:mind:artifact"


def test_mind_enrichment_reads_env(monkeypatch):
    monkeypatch.setenv("ORION_THOUGHT_MIND_ENRICHMENT_ENABLED", "true")
    monkeypatch.setenv("ORION_THOUGHT_MIND_WALL_MS", "9000")
    import app.settings as s
    importlib.reload(s)
    assert s.settings.mind_enrichment_enabled is True
    assert s.settings.mind_wall_ms == 9000
