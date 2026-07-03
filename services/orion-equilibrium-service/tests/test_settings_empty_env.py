from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.settings import Settings


def test_settings_ignore_empty_substrate_env(monkeypatch):
    """Docker compose may pass empty strings when .env keys are missing."""
    monkeypatch.setenv("EQUILIBRIUM_METACOG_SUBSTRATE_TRIGGER_ENABLE", "")
    monkeypatch.setenv("EQUILIBRIUM_METACOG_SUBSTRATE_DENSE_THRESHOLD", "")
    monkeypatch.setenv("EQUILIBRIUM_METACOG_SUBSTRATE_PULSE_THRESHOLD", "")

    settings = Settings()

    assert settings.metacog_substrate_trigger_enable is True
    assert settings.metacog_substrate_dense_threshold == 0.55
    assert settings.metacog_substrate_pulse_threshold == 0.30
