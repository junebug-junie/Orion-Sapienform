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
    ):
        monkeypatch.delenv(key, raising=False)

    settings = Settings(_env_file=None)
    assert settings.CURIOSITY_PEER_ENABLED is False
    assert settings.CURIOSITY_PEER_CLAUDE_PROJECTS_PATH.endswith("/.claude/projects")


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


def test_claude_meter_path_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CURIOSITY_PEER_CLAUDE_PROJECTS_PATH", "/home/x/.claude/projects")
    settings = Settings(_env_file=None)
    assert settings.CURIOSITY_PEER_CLAUDE_PROJECTS_PATH == "/home/x/.claude/projects"


def test_no_cursor_settings_remain() -> None:
    """Cursor retired (#3397): kill means kill, no dormant Cursor knobs."""
    for name in Settings.model_fields:
        assert "CURSOR" not in name
        assert name not in ("CURIOSITY_PEER_AGENT_BIN", "CURIOSITY_PEER_MODEL", "CURIOSITY_PEER_REPO_ROOT")


def test_hub_enqueue_flag_is_separate_contract() -> None:
    """Document the dual kill switch: Hub flag is not this Settings field.

    Enqueue requires HUB_CURIOSITY_CONTRACTOR_PEER_ENABLED (Hub). Processing
    requires CURIOSITY_PEER_ENABLED (this service). Turning only one on is a
    no-op by design.
    """
    assert "HUB_CURIOSITY_CONTRACTOR_PEER_ENABLED" not in Settings.model_fields
    assert "CURIOSITY_PEER_ENABLED" in Settings.model_fields
