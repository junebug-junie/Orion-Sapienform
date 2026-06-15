"""Agent lane health when fake engine is configured."""

from __future__ import annotations

from app.agent_lane_health import AGENT_LANE_FAKE_WARNING, agent_lane_health_block


def test_fake_engine_marks_agent_lane_degraded(monkeypatch) -> None:
    from app.settings import settings as cfg

    monkeypatch.setattr(cfg, "rlm_engine", "fake")
    block = agent_lane_health_block(cfg)
    assert block["rlm_engine"] == "fake"
    assert block["agent_lane_degraded"] is True
    assert block["agent_lane_warning"] == AGENT_LANE_FAKE_WARNING


def test_alexzhang_engine_not_degraded(monkeypatch) -> None:
    from app.settings import settings as cfg

    monkeypatch.setattr(cfg, "rlm_engine", "alexzhang")
    block = agent_lane_health_block(cfg)
    assert block["agent_lane_degraded"] is False
    assert block["agent_lane_warning"] is None
