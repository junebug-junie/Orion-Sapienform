"""Cooldown-lane independence for the two generative trigger kinds.

Same pattern as test_transport_separate_cooldown.py / test_chat_turn_separate_
cooldown.py. `chat_turn` shipped the shared-lane bug once already (a burst of one
kind silently starved every *other* kind's fires, not just its own excess);
insight and flow get their own lanes from day one, and this file is the gate that
proves it stays true -- including that the two new lanes are independent of each
other, not just of the pre-existing ones.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.service import EquilibriumService, settings
from orion.schemas.telemetry.metacog_trigger import MetacogTriggerV1

_ALL_COOLDOWN_ATTRS = (
    "metacog_cooldown_sec",
    "metacog_chat_turn_cooldown_sec",
    "metacog_transport_cooldown_sec",
    "metacog_insight_cooldown_sec",
    "metacog_flow_cooldown_sec",
)


def _trigger(trigger_kind: str) -> MetacogTriggerV1:
    return MetacogTriggerV1(
        trigger_kind=trigger_kind,
        reason="test",
        zen_state="zen",
        pressure=0.1,
    )


def _service() -> EquilibriumService:
    svc = EquilibriumService()
    svc.bus = MagicMock()
    svc.bus.publish = AsyncMock()
    return svc


def _set_all_cooldowns(monkeypatch, value: float) -> None:
    for attr in _ALL_COOLDOWN_ATTRS:
        monkeypatch.setattr(settings, attr, value)


def test_insight_and_flow_are_registered_as_their_own_lanes() -> None:
    """Structural guard: if either kind ever falls out of the per-kind dict it
    silently reverts to sharing the global lane -- exactly chat_turn's bug."""
    mapping = EquilibriumService._PER_KIND_COOLDOWN_SETTINGS_ATTR
    assert mapping["insight"] == "metacog_insight_cooldown_sec"
    assert mapping["flow"] == "metacog_flow_cooldown_sec"


@pytest.mark.asyncio
async def test_insight_second_fire_within_its_own_cooldown_is_dropped(monkeypatch) -> None:
    _set_all_cooldowns(monkeypatch, 30.0)
    svc = _service()

    await svc._publish_metacog_trigger(_trigger("insight"))
    await svc._publish_metacog_trigger(_trigger("insight"))

    assert svc.bus.publish.call_count == 1


@pytest.mark.asyncio
async def test_flow_second_fire_within_its_own_cooldown_is_dropped(monkeypatch) -> None:
    _set_all_cooldowns(monkeypatch, 30.0)
    svc = _service()

    await svc._publish_metacog_trigger(_trigger("flow"))
    await svc._publish_metacog_trigger(_trigger("flow"))

    assert svc.bus.publish.call_count == 1


@pytest.mark.asyncio
async def test_insight_and_flow_do_not_share_a_lane_with_each_other(monkeypatch) -> None:
    """The two new kinds read the same underlying field, so a shared lane would
    be an easy mistake to make -- they must still fire independently."""
    _set_all_cooldowns(monkeypatch, 30.0)
    svc = _service()

    await svc._publish_metacog_trigger(_trigger("insight"))
    await svc._publish_metacog_trigger(_trigger("flow"))

    assert svc.bus.publish.call_count == 2


@pytest.mark.asyncio
async def test_five_lanes_fired_back_to_back_all_fire_independently(monkeypatch) -> None:
    """Full generalization: chat_turn, transport, insight, flow (own lanes) and
    relational (the shared/global lane) must not starve each other."""
    _set_all_cooldowns(monkeypatch, 30.0)
    svc = _service()

    for kind in ("chat_turn", "transport", "insight", "flow", "relational"):
        await svc._publish_metacog_trigger(_trigger(kind))

    assert svc.bus.publish.call_count == 5


@pytest.mark.asyncio
async def test_insight_burst_does_not_starve_any_other_lane(monkeypatch) -> None:
    _set_all_cooldowns(monkeypatch, 30.0)
    svc = _service()

    await svc._publish_metacog_trigger(_trigger("insight"))
    # Dropped by insight's own cooldown -- must not touch anyone else's timestamp.
    await svc._publish_metacog_trigger(_trigger("insight"))
    await svc._publish_metacog_trigger(_trigger("flow"))
    await svc._publish_metacog_trigger(_trigger("chat_turn"))
    await svc._publish_metacog_trigger(_trigger("transport"))
    await svc._publish_metacog_trigger(_trigger("relational"))

    assert svc.bus.publish.call_count == 5


@pytest.mark.asyncio
async def test_flow_burst_does_not_starve_any_other_lane(monkeypatch) -> None:
    _set_all_cooldowns(monkeypatch, 30.0)
    svc = _service()

    await svc._publish_metacog_trigger(_trigger("flow"))
    await svc._publish_metacog_trigger(_trigger("flow"))
    await svc._publish_metacog_trigger(_trigger("insight"))
    await svc._publish_metacog_trigger(_trigger("relational"))

    assert svc.bus.publish.call_count == 3


@pytest.mark.asyncio
async def test_each_new_lane_uses_its_own_setting_not_the_shared_one(monkeypatch) -> None:
    """A wide-open insight/flow cooldown must not be readable from the global
    setting, and vice versa."""
    monkeypatch.setattr(settings, "metacog_cooldown_sec", 0.0)
    monkeypatch.setattr(settings, "metacog_chat_turn_cooldown_sec", 0.0)
    monkeypatch.setattr(settings, "metacog_transport_cooldown_sec", 0.0)
    monkeypatch.setattr(settings, "metacog_insight_cooldown_sec", 9999.0)
    monkeypatch.setattr(settings, "metacog_flow_cooldown_sec", 9999.0)
    svc = _service()

    await svc._publish_metacog_trigger(_trigger("insight"))
    await svc._publish_metacog_trigger(_trigger("insight"))
    await svc._publish_metacog_trigger(_trigger("flow"))
    await svc._publish_metacog_trigger(_trigger("flow"))
    # Shared lane is wide open at 0.0, so both of these fire.
    await svc._publish_metacog_trigger(_trigger("relational"))
    await svc._publish_metacog_trigger(_trigger("relational"))

    # insight: 1, flow: 1, relational: 2
    assert svc.bus.publish.call_count == 4


@pytest.mark.asyncio
async def test_wide_open_shared_lane_does_not_gate_insight_or_flow(monkeypatch) -> None:
    monkeypatch.setattr(settings, "metacog_cooldown_sec", 9999.0)
    monkeypatch.setattr(settings, "metacog_chat_turn_cooldown_sec", 9999.0)
    monkeypatch.setattr(settings, "metacog_transport_cooldown_sec", 9999.0)
    monkeypatch.setattr(settings, "metacog_insight_cooldown_sec", 0.0)
    monkeypatch.setattr(settings, "metacog_flow_cooldown_sec", 0.0)
    svc = _service()

    await svc._publish_metacog_trigger(_trigger("insight"))
    await svc._publish_metacog_trigger(_trigger("insight"))
    await svc._publish_metacog_trigger(_trigger("flow"))
    await svc._publish_metacog_trigger(_trigger("flow"))
    # Global lane is clamped shut, so only the first relational fires.
    await svc._publish_metacog_trigger(_trigger("relational"))
    await svc._publish_metacog_trigger(_trigger("relational"))

    # insight: 2, flow: 2, relational: 1
    assert svc.bus.publish.call_count == 5
