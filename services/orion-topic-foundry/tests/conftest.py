"""Shared test guards for orion-topic-foundry."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_live_physiology_bid(monkeypatch):
    """Unit tests must never POST a shadow bid to the live body economy (Patch C3b): the client
    returns None without a call. Tests of the shadow seam stub app.services.physiology_shadow.bid."""
    monkeypatch.setenv("ORION_PHYSIOLOGY_BID_CLIENT_ENABLED", "false")
