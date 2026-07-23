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


@pytest.mark.asyncio
async def test_chat_turn_second_fire_within_its_own_cooldown_is_dropped(monkeypatch):
    monkeypatch.setattr(settings, "metacog_chat_turn_cooldown_sec", 30.0)
    monkeypatch.setattr(settings, "metacog_cooldown_sec", 30.0)
    svc = _service()

    await svc._publish_metacog_trigger(_trigger("chat_turn"))
    await svc._publish_metacog_trigger(_trigger("chat_turn"))

    assert svc.bus.publish.call_count == 1


@pytest.mark.asyncio
async def test_chat_turn_fire_does_not_block_other_trigger_kinds(monkeypatch):
    """The bug this fix closes: a shared cooldown timestamp would let a
    chat_turn fire silently starve telemetry_anomaly/relational's own fires
    moments later. Separate lanes means both fire independently."""
    monkeypatch.setattr(settings, "metacog_chat_turn_cooldown_sec", 30.0)
    monkeypatch.setattr(settings, "metacog_cooldown_sec", 30.0)
    svc = _service()

    await svc._publish_metacog_trigger(_trigger("chat_turn"))
    await svc._publish_metacog_trigger(_trigger("telemetry_anomaly"))

    assert svc.bus.publish.call_count == 2


@pytest.mark.asyncio
async def test_other_trigger_kind_fire_does_not_block_chat_turn(monkeypatch):
    monkeypatch.setattr(settings, "metacog_chat_turn_cooldown_sec", 30.0)
    monkeypatch.setattr(settings, "metacog_cooldown_sec", 30.0)
    svc = _service()

    await svc._publish_metacog_trigger(_trigger("relational"))
    await svc._publish_metacog_trigger(_trigger("chat_turn"))

    assert svc.bus.publish.call_count == 2


@pytest.mark.asyncio
async def test_two_non_chat_turn_kinds_still_share_the_global_cooldown(monkeypatch):
    """Unchanged pre-existing behavior: baseline/manual/pulse/relational/
    telemetry_anomaly still share one cooldown lane with each other."""
    monkeypatch.setattr(settings, "metacog_chat_turn_cooldown_sec", 30.0)
    monkeypatch.setattr(settings, "metacog_cooldown_sec", 30.0)
    svc = _service()

    await svc._publish_metacog_trigger(_trigger("relational"))
    await svc._publish_metacog_trigger(_trigger("telemetry_anomaly"))

    assert svc.bus.publish.call_count == 1


@pytest.mark.asyncio
async def test_chat_turn_cooldown_uses_its_own_setting_not_the_shared_one(monkeypatch):
    """chat_turn's cooldown is independently tunable -- a wide-open shared
    cooldown must not let chat_turn fire back-to-back if its own setting is
    still in effect."""
    monkeypatch.setattr(settings, "metacog_cooldown_sec", 0.0)
    monkeypatch.setattr(settings, "metacog_chat_turn_cooldown_sec", 9999.0)
    svc = _service()

    await svc._publish_metacog_trigger(_trigger("chat_turn"))
    await svc._publish_metacog_trigger(_trigger("chat_turn"))

    assert svc.bus.publish.call_count == 1


@pytest.mark.asyncio
async def test_shared_cooldown_at_zero_does_not_affect_chat_turn_lane(monkeypatch):
    """Inverse of the above: a wide-open chat_turn cooldown must not let the
    shared-lane kinds fire back-to-back if the shared setting is still tight."""
    monkeypatch.setattr(settings, "metacog_cooldown_sec", 9999.0)
    monkeypatch.setattr(settings, "metacog_chat_turn_cooldown_sec", 0.0)
    svc = _service()

    await svc._publish_metacog_trigger(_trigger("relational"))
    await svc._publish_metacog_trigger(_trigger("relational"))

    assert svc.bus.publish.call_count == 1
