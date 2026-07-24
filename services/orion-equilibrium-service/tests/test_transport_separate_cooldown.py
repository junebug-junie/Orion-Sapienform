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
    monkeypatch.setattr(settings, "metacog_cooldown_sec", value)
    monkeypatch.setattr(settings, "metacog_chat_turn_cooldown_sec", value)
    monkeypatch.setattr(settings, "metacog_transport_cooldown_sec", value)


@pytest.mark.asyncio
async def test_transport_second_fire_within_its_own_cooldown_is_dropped(monkeypatch):
    _set_all_cooldowns(monkeypatch, 30.0)
    svc = _service()

    await svc._publish_metacog_trigger(_trigger("transport"))
    await svc._publish_metacog_trigger(_trigger("transport"))

    assert svc.bus.publish.call_count == 1


@pytest.mark.asyncio
async def test_three_lanes_fired_back_to_back_all_fire_independently(monkeypatch):
    """The N-way generalization this cooldown refactor exists to prove: chat_turn
    (its own lane), transport (its own lane), and relational (the shared/global
    lane) firing back-to-back must not starve each other -- not just the 2-lane
    case test_chat_turn_separate_cooldown.py already covers."""
    _set_all_cooldowns(monkeypatch, 30.0)
    svc = _service()

    await svc._publish_metacog_trigger(_trigger("chat_turn"))
    await svc._publish_metacog_trigger(_trigger("transport"))
    await svc._publish_metacog_trigger(_trigger("relational"))

    assert svc.bus.publish.call_count == 3


@pytest.mark.asyncio
async def test_transport_second_fire_does_not_block_chat_turn_or_relational(monkeypatch):
    _set_all_cooldowns(monkeypatch, 30.0)
    svc = _service()

    await svc._publish_metacog_trigger(_trigger("transport"))
    # Within transport's own cooldown -- must be dropped, but must not touch the
    # other two lanes' own timestamps.
    await svc._publish_metacog_trigger(_trigger("transport"))
    await svc._publish_metacog_trigger(_trigger("chat_turn"))
    await svc._publish_metacog_trigger(_trigger("relational"))

    assert svc.bus.publish.call_count == 3


@pytest.mark.asyncio
async def test_transport_cooldown_uses_its_own_setting_not_the_shared_one(monkeypatch):
    monkeypatch.setattr(settings, "metacog_cooldown_sec", 0.0)
    monkeypatch.setattr(settings, "metacog_chat_turn_cooldown_sec", 0.0)
    monkeypatch.setattr(settings, "metacog_transport_cooldown_sec", 9999.0)
    svc = _service()

    await svc._publish_metacog_trigger(_trigger("transport"))
    await svc._publish_metacog_trigger(_trigger("transport"))

    assert svc.bus.publish.call_count == 1


@pytest.mark.asyncio
async def test_wide_open_transport_cooldown_does_not_affect_chat_turn_or_shared_lane(monkeypatch):
    monkeypatch.setattr(settings, "metacog_cooldown_sec", 9999.0)
    monkeypatch.setattr(settings, "metacog_chat_turn_cooldown_sec", 9999.0)
    monkeypatch.setattr(settings, "metacog_transport_cooldown_sec", 0.0)
    svc = _service()

    await svc._publish_metacog_trigger(_trigger("transport"))
    await svc._publish_metacog_trigger(_trigger("transport"))
    await svc._publish_metacog_trigger(_trigger("chat_turn"))
    await svc._publish_metacog_trigger(_trigger("chat_turn"))

    # transport: both fire (its own cooldown is 0). chat_turn: only the first
    # fires (its own cooldown is wide open at 9999s).
    assert svc.bus.publish.call_count == 3
