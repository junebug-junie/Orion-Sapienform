from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.settings import Settings


def test_settings_ignore_empty_env(monkeypatch):
    """Docker compose may pass empty strings when .env keys are missing."""
    monkeypatch.setenv("EQUILIBRIUM_METACOG_COOLDOWN_SEC", "")
    monkeypatch.setenv("EQUILIBRIUM_TRANSPORT_BASELINE_ENABLE", "")
    monkeypatch.setenv("EQUILIBRIUM_TRANSPORT_BASELINE_MIN_EXCESS_MS", "")

    settings = Settings()

    assert settings.metacog_cooldown_sec == 30.0
    assert settings.transport_baseline_enable is True
    assert settings.transport_baseline_min_excess_ms == 250.0
