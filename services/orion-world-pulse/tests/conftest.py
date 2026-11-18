from __future__ import annotations

import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


import pytest


@pytest.fixture(autouse=True)
def _no_source_health_db(monkeypatch):
    """Pipeline tests never read live world_pulse_run history, whatever .env says."""
    from app.settings import settings

    monkeypatch.setattr(settings, "world_pulse_source_health_db_url", "")
