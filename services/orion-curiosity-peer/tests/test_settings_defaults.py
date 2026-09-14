"""Settings defaults for the contractor peer kill switch.

Peer work is off unless this service's CURIOSITY_PEER_ENABLED is true.
Hub's HUB_CURIOSITY_CONTRACTOR_PEER_ENABLED is a separate kill switch that
gates enqueue at Hub time — this service does not read Hub's flag, but both
must be on for a live hire path. Typo alias CURIOUSITY_PEER_ENABLED is also
accepted and still defaults false.
"""

from __future__ import annotations

import pytest

from app.settings import Settings


def test_curiosity_peer_defaults_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "CURIOSITY_PEER_ENABLED",
        "CURIOUSITY_PEER_ENABLED",
        "CURSOR_API_KEY",
        "CURIOSITY_PEER_AGENT_BIN",
    ):
        monkeypatch.delenv(key, raising=False)

    settings = Settings(_env_file=None)
    assert settings.CURIOSITY_PEER_ENABLED is False
    assert "CURSOR_API_KEY" not in Settings.model_fields
    assert "CURIOSITY_PEER_AGENT_BIN" in Settings.model_fields
    assert settings.CURIOSITY_PEER_AGENT_BIN.endswith("cursor-agent")


def test_curiosity_peer_enabled_only_when_explicitly_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CURIOUSITY_PEER_ENABLED", raising=False)
    monkeypatch.setenv("CURIOSITY_PEER_ENABLED", "true")
    settings = Settings(_env_file=None)
    assert settings.CURIOSITY_PEER_ENABLED is True


def test_curiousity_typo_alias_enables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CURIOSITY_PEER_ENABLED", raising=False)
    monkeypatch.setenv("CURIOUSITY_PEER_ENABLED", "true")
    settings = Settings(_env_file=None)
    assert settings.CURIOSITY_PEER_ENABLED is True


def test_agent_bin_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "CURIOSITY_PEER_AGENT_BIN",
        "/opt/cursor-agent/versions/test/cursor-agent",
    )
    settings = Settings(_env_file=None)
    assert settings.CURIOSITY_PEER_AGENT_BIN == (
        "/opt/cursor-agent/versions/test/cursor-agent"
    )


def test_hub_enqueue_flag_is_separate_contract() -> None:
    """Document the dual kill switch: Hub flag is not this Settings field.

    Enqueue requires HUB_CURIOSITY_CONTRACTOR_PEER_ENABLED (Hub). Processing
    requires CURIOSITY_PEER_ENABLED (this service). Turning only one on is a
    no-op by design.
    """
    assert "HUB_CURIOSITY_CONTRACTOR_PEER_ENABLED" not in Settings.model_fields
    assert "CURIOSITY_PEER_ENABLED" in Settings.model_fields
