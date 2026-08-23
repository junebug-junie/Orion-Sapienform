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
    """Weather/lab/perception are not yet configurable from orion-hub --
    the adapter must turn them off explicitly rather than leave it to a
    missing-attribute default to silently decide."""
    cfg = settings_from_runtime(hub_settings_to_runtime_namespace(SimpleNamespace()))

    assert cfg.weather_enabled is False
    assert cfg.lab_enabled is False
    assert cfg.perception_enabled is False
    # The runtime (self-model) probe IS wired -- it reuses HUB_LLM_GATEWAY_URL,
    # a host orion-hub already calls today, so it costs no new dependency.
    assert cfg.runtime_enabled is True


def test_adapter_falls_back_to_safe_defaults_when_hub_settings_missing_attrs() -> None:
    ns = hub_settings_to_runtime_namespace(SimpleNamespace())

    assert ns.orion_situation_enabled is True
    assert ns.orion_situation_timezone == "America/Denver"
    assert ns.orion_presence_default_requestor == "Juniper"
