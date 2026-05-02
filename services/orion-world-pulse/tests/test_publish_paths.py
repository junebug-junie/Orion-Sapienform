from __future__ import annotations

from datetime import datetime, timezone

from app.routers import publish as publish_router
from app.state import RUN_RESULTS
from app.services import publish_hub as publish_hub_service
from app.services.publish_hub import publish_hub_message
from app.services.renderers import render_email_digest, render_hub_digest
from orion.schemas.world_pulse import (
    DailyWorldPulseItemV1,
    DailyWorldPulseSectionsV1,
    DailyWorldPulseV1,
    SectionRollupV1,
    WorthReadingItemV1,
    WorthWatchingItemV1,
    WorldPulseRunResultV1,
    WorldPulseRunV1,
)


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
        items=[
            DailyWorldPulseItemV1(
                item_id="item-1",
                run_id="r1",
                title="Policy update",
                category="us_politics",
                summary="summary",
                why_it_matters="matters",
                what_changed="changed",
                orion_read="read",
                source_ids=["source-a"],
                article_ids=["article-a"],
                created_at=now,
            )
        ],
        things_worth_reading=[
            WorthReadingItemV1(
                reading_id="read-1",
                title="Worth reading",
                source_id="source-a",
                article_id="article-a",
                url="https://example.com/read",
                reason_selected="context",
                reading_type="analysis",
                trust_tier=1,
                category="us_politics",
                created_at=now,
            )
        ],
        things_worth_watching=[
            WorthWatchingItemV1(
                watch_id="watch-1",
                topic_id="topic-1",
                title="Watch this",
                reason="risk",
                watch_condition="if x",
                category="us_politics",
                created_at=now,
            )
        ],
        section_rollups=[
            SectionRollupV1(
                section="us_politics",
                status="covered",
                article_count=1,
                cluster_count=1,
                digest_item_count=1,
                summary="rollup",
                source_notes=["source stable"],
                confidence=0.8,
            )
        ],
        source_notes=["note-1"],
        coverage_status="complete",
        accepted_article_count=1,
        article_cluster_count=1,
        max_digest_items_total=12,
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
    result = publish_hub_message(message=msg, dry_run=True)
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
    result = publish_hub_message(message=msg, dry_run=False)
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
    first = publish_hub_message(message=msg, dry_run=False)
    second = publish_hub_message(message=msg, dry_run=False)
    assert first["ok"] is True and second["ok"] is True
    assert len(_FakeBus.published) == 2
    assert _FakeBus.published[0][1].payload["message_id"] == _FakeBus.published[1][1].payload["message_id"]


def test_publish_hub_uses_explicit_dry_run_over_global(monkeypatch):
    digest = _digest()
    msg = render_hub_digest(digest)
    monkeypatch.setattr(publish_hub_service.settings, "world_pulse_dry_run", True)
    monkeypatch.setattr(publish_hub_service, "OrionBusAsync", _FakeBus)
    _FakeBus.published = []
    result = publish_hub_message(message=msg, dry_run=False)
    assert result["ok"] is True
    assert result["status"] == "published"
    assert len(_FakeBus.published) == 1


def test_render_hub_digest_contains_structured_payload():
    digest = _digest()
    msg = render_hub_digest(digest)
    structured = msg.structured_payload
    assert structured["message_type"] == "daily_world_pulse"
    assert structured["run_id"] == digest.run_id
    assert structured["coverage_status"] == "complete"
    assert structured["accepted_article_count"] == 1
    assert len(structured["items"]) == 1
    assert len(structured["section_rollups"]) == 1
    assert len(structured["things_worth_reading"]) == 1
    assert len(structured["things_worth_watching"]) == 1


def test_render_hub_digest_backfills_aggregate_fields_when_missing():
    digest = _digest().model_copy(
        update={
            "accepted_article_count": 0,
            "article_cluster_count": 0,
            "max_digest_items_total": 0,
            "coverage_status": "empty",
            "section_coverage": {},
            "source_ids": [],
            "section_rollups": [],
        }
    )
    msg = render_hub_digest(digest)
    structured = msg.structured_payload
    assert structured["accepted_article_count"] == 1
    assert structured["article_cluster_count"] == 1
    assert structured["max_digest_items_total"] >= 1
    assert structured["coverage_status"] != "empty"
    assert "us_politics" in structured["section_coverage"]


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
