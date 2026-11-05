"""Hub /health carries the leisure timer's last tick (ladder retirement PR 1), so a silent hold is
visible without reading logs. Separate from test_retro_leisure_self_start.py: this one imports the
whole Hub app (scripts.main, scripts.api_routes), which the lane-C workflow does not install."""
from __future__ import annotations

import json

from scripts.retro_leisure_self_start import RetroLeisureSelfStart


def test_health_carries_the_timers_last_tick(monkeypatch) -> None:
    import scripts.main as hub_main
    from scripts import api_routes

    timer = RetroLeisureSelfStart(config_provider=lambda: None, pool_provider=lambda: None,
                                  runner_provider=lambda: None)
    timer.last = {"result": "refused", "refusal": "cooldown", "ticked_at": "x"}
    monkeypatch.setattr(hub_main, "retro_leisure_self_start", timer, raising=False)
    body = api_routes.health()
    block = body["retro_leisure_self_start"]
    assert block["last"]["refusal"] == "cooldown" and block["tick_sec"] == 300.0 and block["enabled"] is True
    json.dumps(body)


def test_health_before_startup_is_none(monkeypatch) -> None:
    import scripts.main as hub_main
    from scripts import api_routes

    monkeypatch.setattr(hub_main, "retro_leisure_self_start", None, raising=False)
    assert api_routes.health()["retro_leisure_self_start"] is None
