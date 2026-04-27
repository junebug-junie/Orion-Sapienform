from __future__ import annotations

from datetime import datetime, timezone

from app.routers import publish as publish_router
from app.state import RUN_RESULTS
from app.services import publish_hub as publish_hub_service
from app.services.publish_hub import publish_hub_message
from app.services.renderers import render_email_digest, render_hub_digest
from orion.schemas.world_pulse import DailyWorldPulseSectionsV1, DailyWorldPulseV1, WorldPulseRunResultV1, WorldPulseRunV1


class _FakeBus:
    published: list[tuple[str, object]] = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def connect(self) -> None:
        return None

    async def publish(self, channel, envelope) -> None:
        self.published.append((channel, envelope))

    async def close(self) -> None:
        return None


def _digest() -> DailyWorldPulseV1:
    now = datetime.now(timezone.utc)
    return DailyWorldPulseV1(
        run_id="r1",
        date=now.date().isoformat(),
        generated_at=now,
        title="Daily World Pulse",
        executive_summary="summary",
        sections=DailyWorldPulseSectionsV1(),
        orion_analysis_layer="analysis",
        created_at=now,
    )


def test_render_email_subject_format():
    digest = _digest()
    email = render_email_digest(digest, subject_prefix="Orion Daily World Pulse", to=[], from_email=None, dry_run=True)
    assert email.subject == f"Orion Daily World Pulse — {digest.date}"


def test_publish_hub_dry_run_returns_payload_preview(monkeypatch):
    digest = _digest()
    msg = render_hub_digest(digest)
    monkeypatch.setattr(publish_hub_service.settings, "world_pulse_dry_run", True)
    monkeypatch.setattr(publish_hub_service, "OrionBusAsync", _FakeBus)
    _FakeBus.published = []
    result = publish_hub_message(message=msg)
    assert result["ok"] is True
    assert result["status"] == "dry_run"
    assert result["would_publish"] is True
    assert result["channel"] == "orion:hub:messages:create"
    assert result["kind"] == "hub.messages.create.v1"
    assert result["payload_preview"]["message_id"] == msg.message_id
    assert not _FakeBus.published


def test_publish_hub_real_path_emits_bus_envelope(monkeypatch):
    digest = _digest()
    msg = render_hub_digest(digest)
    monkeypatch.setattr(publish_hub_service.settings, "world_pulse_dry_run", False)
    monkeypatch.setattr(publish_hub_service, "OrionBusAsync", _FakeBus)
    _FakeBus.published = []
    result = publish_hub_message(message=msg)
    assert result["ok"] is True
    assert result["status"] == "published"
    assert len(_FakeBus.published) == 1
    channel, envelope = _FakeBus.published[0]
    assert channel == "orion:hub:messages:create"
    assert envelope.kind == "hub.messages.create.v1"
    assert envelope.payload["message_id"] == msg.message_id
    assert envelope.payload["run_id"] == msg.run_id


def test_publish_hub_message_id_is_stable_for_retry(monkeypatch):
    digest = _digest()
    msg = render_hub_digest(digest)
    monkeypatch.setattr(publish_hub_service.settings, "world_pulse_dry_run", False)
    monkeypatch.setattr(publish_hub_service, "OrionBusAsync", _FakeBus)
    _FakeBus.published = []
    first = publish_hub_message(message=msg)
    second = publish_hub_message(message=msg)
    assert first["ok"] is True and second["ok"] is True
    assert len(_FakeBus.published) == 2
    assert _FakeBus.published[0][1].payload["message_id"] == _FakeBus.published[1][1].payload["message_id"]


def test_publish_hub_route_disabled_returns_hub_messages_disabled(monkeypatch):
    digest = _digest()
    run = WorldPulseRunV1(run_id="r1", date=digest.date, started_at=digest.created_at, requested_by="test", dry_run=True)
    RUN_RESULTS["r1"] = WorldPulseRunResultV1(run=run, digest=digest)
    monkeypatch.setattr(publish_router.settings, "world_pulse_hub_messages_enabled", False)
    response = publish_router.publish_hub("r1")
    assert response["status"] == "hub_messages_disabled"
    assert response["ok"] is False
    assert response["run_id"] == "r1"
    RUN_RESULTS.clear()
