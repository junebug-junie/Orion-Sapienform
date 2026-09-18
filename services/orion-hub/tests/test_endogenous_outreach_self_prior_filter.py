"""Outreach must not treat line=self priors as talkable world content (#2224)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from orion.curiosity.worldview import LIVE_NON_SELF_PRIORS_CYPHER


def test_fetch_open_prior_previews_queries_non_self_cypher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import endogenous_outreach as outreach_module

    queried: list[str] = []

    class _FakeReader:
        def query(self, cypher: str) -> list[dict]:
            queried.append(cypher)
            return [
                {
                    "prior_id": "world-1",
                    "claim": "Juniper prefers terse status updates.",
                    "confidence": 0.85,
                    "status": "open",
                    "times_tested": 1,
                    "formed_from": "",
                    "last_tested_at": "",
                }
            ]

    import app.settings as settings_mod
    import orion.curiosity.worldview as worldview_mod

    monkeypatch.setattr(
        settings_mod,
        "get_settings",
        lambda: SimpleNamespace(
            HUB_CURIOSITY_GRAPH_HOST="127.0.0.1",
            HUB_CURIOSITY_GRAPH_PORT=6380,
            HUB_CURIOSITY_GRAPH_OWN="orion_worldview",
            HUB_CURIOSITY_GRAPH_ORION_USER="orion_curiosity",
            HUB_CURIOSITY_GRAPH_ORION_PASSWORD="secret",
        ),
    )
    monkeypatch.setattr(worldview_mod, "WorldviewReader", lambda **kwargs: _FakeReader())

    previews = outreach_module._fetch_open_prior_previews()

    assert queried == [LIVE_NON_SELF_PRIORS_CYPHER]
    assert previews
    assert "Juniper prefers terse status updates." in previews[0]
