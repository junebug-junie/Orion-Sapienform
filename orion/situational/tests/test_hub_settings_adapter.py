"""orion-hub's `Settings` (UPPERCASE ORION_SITUATION_*/ORION_PRESENCE_* attrs)
adapted into the lowercase-attribute shape `settings_from_runtime` expects.

Regression context (2026-08-22): "Orion asked how my evening was going at
12:45pm" traced to orion-hub's unified-turn path never calling
`build_situation_for_ctx` at all. Passing orion-hub's real `Settings` object
straight into `settings_from_runtime` would silently miss every field
(`getattr` is case-sensitive: `orion_situation_enabled` !=
`ORION_SITUATION_ENABLED`) and fall back to hardcoded defaults with no
visible error -- exactly the kind of silent-fallback trap this adapter
exists to make explicit instead.
"""

from __future__ import annotations

from types import SimpleNamespace

from orion.situational.context import hub_settings_to_runtime_namespace, settings_from_runtime


def test_adapter_reads_hub_uppercase_attrs_not_defaults() -> None:
    hub_settings = SimpleNamespace(
        ORION_SITUATION_ENABLED=False,
        ORION_SITUATION_TTL_SECONDS=60,
        ORION_SITUATION_TIMEZONE="Europe/Berlin",
        ORION_PRESENCE_DEFAULT_REQUESTOR="Someone Else",
        ORION_PRESENCE_PERSIST_ALLOWED=True,
        HUB_LLM_GATEWAY_URL="http://127.0.0.1:9999",
    )
    ns = hub_settings_to_runtime_namespace(hub_settings)

    assert ns.orion_situation_enabled is False
    assert ns.orion_situation_ttl_seconds == 60
    assert ns.orion_situation_timezone == "Europe/Berlin"
    assert ns.orion_presence_default_requestor == "Someone Else"
    assert ns.orion_presence_persist_allowed is True
    assert ns.cortex_exec_llm_gateway_url == "http://127.0.0.1:9999"


def test_adapter_output_survives_settings_from_runtime_round_trip() -> None:
    """The whole point of this adapter: feeding its output into
    `settings_from_runtime` (as every real caller does) must preserve the
    hub-configured values, not silently reset them to code defaults."""
    hub_settings = SimpleNamespace(
        ORION_SITUATION_ENABLED=True,
        ORION_SITUATION_TTL_SECONDS=45,
        ORION_SITUATION_TIMEZONE="Pacific/Auckland",
        ORION_PRESENCE_DEFAULT_REQUESTOR="Juniper",
        ORION_PRESENCE_PERSIST_ALLOWED=False,
        HUB_LLM_GATEWAY_URL="http://127.0.0.1:8210",
    )
    cfg = settings_from_runtime(hub_settings_to_runtime_namespace(hub_settings))

    assert cfg.enabled is True
    assert cfg.ttl_seconds == 45
    assert cfg.timezone == "Pacific/Auckland"
    assert cfg.default_requestor == "Juniper"
    assert cfg.llm_gateway_base_url == "http://127.0.0.1:8210"


def test_adapter_turns_off_unwired_providers_explicitly() -> None:
    """Lab/perception are not yet configurable from orion-hub -- the adapter
    must turn them off explicitly rather than leave it to a
    missing-attribute default to silently decide."""
    cfg = settings_from_runtime(hub_settings_to_runtime_namespace(SimpleNamespace()))

    assert cfg.lab_enabled is False
    assert cfg.perception_enabled is False
    # Weather, the runtime (self-model) probe, and affect ARE wired --
    # weather reads orion-hub's own ORION_SITUATION_WEATHER_* fields, the
    # runtime probe reuses HUB_LLM_GATEWAY_URL (a host orion-hub already
    # calls today), and affect reads off the bus connection orion-hub
    # already holds (bind_juniper_affect_state_bus in scripts/main.py) --
    # none of the three costs a new, unvetted dependency, unlike
    # lab/perception's unwired DSN/HTTP needs.
    assert cfg.runtime_enabled is True
    assert cfg.affect_enabled is True
    assert cfg.affect_max_age_seconds == 300
    # Curiosity/reverie (2026-08-30) ARE wired -- default ON, per Juniper's
    # explicit request, unlike lab/perception above.
    assert cfg.curiosity_enabled is True
    assert cfg.reverie_enabled is True
    # Cabinet sensors (2026-08-31) ARE wired -- default ON, same "no
    # private-home content" reasoning, reusing Hub's existing
    # CABINET_SENSORS_PATH default rather than a new sensor-path key.
    assert cfg.cabinet_enabled is True
    assert cfg.cabinet_sensors_path == "/run/orion-sensors/latest.json"


def test_adapter_reads_hub_affect_overrides() -> None:
    hub_settings = SimpleNamespace(
        ORION_SITUATION_AFFECT_ENABLED=False,
        ORION_SITUATION_AFFECT_MAX_AGE_SECONDS=60,
    )
    cfg = settings_from_runtime(hub_settings_to_runtime_namespace(hub_settings))

    assert cfg.affect_enabled is False
    assert cfg.affect_max_age_seconds == 60


def test_adapter_reads_hub_weather_config() -> None:
    hub_settings = SimpleNamespace(
        ORION_SITUATION_WEATHER_ENABLED=True,
        ORION_SITUATION_WEATHER_PROVIDER="openmeteo",
        ORION_SITUATION_WEATHER_LAT=41.2230,
        ORION_SITUATION_WEATHER_LON=-111.9738,
        ORION_SITUATION_WEATHER_TTL_SECONDS=600,
    )
    cfg = settings_from_runtime(hub_settings_to_runtime_namespace(hub_settings))

    assert cfg.weather_enabled is True
    assert cfg.weather_provider == "openmeteo"
    assert cfg.weather_lat == 41.2230
    assert cfg.weather_lon == -111.9738
    assert cfg.weather_ttl_seconds == 600


def test_adapter_weather_disabled_when_hub_sets_it_off() -> None:
    hub_settings = SimpleNamespace(ORION_SITUATION_WEATHER_ENABLED=False)
    cfg = settings_from_runtime(hub_settings_to_runtime_namespace(hub_settings))

    assert cfg.weather_enabled is False


def test_adapter_reads_hub_curiosity_config_and_reuses_existing_graph_keys() -> None:
    """Reuses HUB_CURIOSITY_GRAPH_* (already asserted against
    orion_worldview by curiosity_investigation.py) rather than a second,
    parallel set of graph keys -- see hub_settings_to_runtime_namespace's
    own docstring."""
    hub_settings = SimpleNamespace(
        ORION_SITUATION_CURIOSITY_ENABLED=True,
        ORION_SITUATION_CURIOSITY_TTL_SECONDS=90,
        HUB_CURIOSITY_GRAPH_HOST="127.0.0.1",
        HUB_CURIOSITY_GRAPH_PORT=6380,
        HUB_CURIOSITY_GRAPH_OWN="orion_worldview",
    )
    cfg = settings_from_runtime(hub_settings_to_runtime_namespace(hub_settings))

    assert cfg.curiosity_enabled is True
    assert cfg.curiosity_ttl_seconds == 90
    assert cfg.curiosity_graph_host == "127.0.0.1"
    assert cfg.curiosity_graph_port == 6380
    assert cfg.curiosity_graph_name == "orion_worldview"


def test_adapter_curiosity_disabled_when_hub_sets_it_off() -> None:
    hub_settings = SimpleNamespace(ORION_SITUATION_CURIOSITY_ENABLED=False)
    cfg = settings_from_runtime(hub_settings_to_runtime_namespace(hub_settings))

    assert cfg.curiosity_enabled is False


def test_adapter_curiosity_unconfigured_when_hub_has_no_graph_host() -> None:
    """Adapter-level default when the caller supplies no
    HUB_CURIOSITY_GRAPH_HOST attribute at all (e.g. a caller other than the
    real `services/orion-hub/app/settings.py` `Settings` class, whose own
    `HUB_CURIOSITY_GRAPH_HOST` field defaults to a non-empty
    "127.0.0.1" -- a real Hub deployment is NOT "unconfigured" for
    curiosity by default; see `test_adapter_reads_hub_curiosity_config_and_
    reuses_existing_graph_keys` above for that case). What this test
    verifies: the adapter must not silently invent a graph host out of
    nowhere -- an absent attribute produces the real, distinct
    "unconfigured" state (`_build_curiosity_context`'s own branch), not an
    error and not a guessed default host."""
    cfg = settings_from_runtime(hub_settings_to_runtime_namespace(SimpleNamespace()))

    assert cfg.curiosity_enabled is True
    assert cfg.curiosity_graph_host == ""


def test_adapter_reads_hub_reverie_config() -> None:
    hub_settings = SimpleNamespace(
        ORION_SITUATION_REVERIE_ENABLED=True,
        ORION_SITUATION_REVERIE_TTL_SECONDS=90,
    )
    cfg = settings_from_runtime(hub_settings_to_runtime_namespace(hub_settings))

    assert cfg.reverie_enabled is True
    assert cfg.reverie_ttl_seconds == 90


def test_adapter_reverie_disabled_when_hub_sets_it_off() -> None:
    hub_settings = SimpleNamespace(ORION_SITUATION_REVERIE_ENABLED=False)
    cfg = settings_from_runtime(hub_settings_to_runtime_namespace(hub_settings))

    assert cfg.reverie_enabled is False


def test_adapter_reads_hub_prompt_max_chars_override() -> None:
    """2026-08-30: previously hardcoded to a bare 1200 with no env override
    at all -- this is the regression test for that gap."""
    hub_settings = SimpleNamespace(ORION_SITUATION_PROMPT_MAX_CHARS=9000)
    cfg = settings_from_runtime(hub_settings_to_runtime_namespace(hub_settings))

    assert cfg.prompt_max_chars == 9000


def test_adapter_prompt_max_chars_defaults_to_7200_when_hub_settings_missing() -> None:
    ns = hub_settings_to_runtime_namespace(SimpleNamespace())
    assert ns.orion_situation_prompt_max_chars == 7200

    cfg = settings_from_runtime(ns)
    assert cfg.prompt_max_chars == 7200


def test_adapter_falls_back_to_safe_defaults_when_hub_settings_missing_attrs() -> None:
    ns = hub_settings_to_runtime_namespace(SimpleNamespace())

    assert ns.orion_situation_enabled is True
    assert ns.orion_situation_timezone == "America/Denver"
    assert ns.orion_presence_default_requestor == "Juniper"
