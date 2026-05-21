from __future__ import annotations

from datetime import datetime, timezone

from orion.journaler import (
    build_compose_request,
    build_world_pulse_reflective_trigger,
    cooldown_key_for_trigger,
    journal_mode_for_trigger,
)
from orion.schemas.world_pulse import (
    DailyWorldPulseItemV1,
    DailyWorldPulseSectionsV1,
    DailyWorldPulseV1,
    SectionRollupV1,
    WorldPulseRunResultV1,
    WorldPulseRunV1,
)


def _sample_run_result(*, dry_run: bool = False) -> WorldPulseRunResultV1:
    run_id = "wp-run-1"
    now = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)
    digest = DailyWorldPulseV1(
        run_id=run_id,
        date="2026-05-20",
        generated_at=now,
        title="Daily World Pulse",
        executive_summary="Several policy and infrastructure threads shifted today.",
        sections=DailyWorldPulseSectionsV1(),
        items=[
            DailyWorldPulseItemV1(
                item_id="item-1",
                run_id=run_id,
                title="Grid stress watch",
                category="infrastructure",
                summary="Regional operators flagged elevated load.",
                why_it_matters="May affect lab power planning.",
                what_changed="New advisory issued.",
                orion_read="Watch regional grid advisories.",
                created_at=now,
            )
        ],
        orion_analysis_layer="deterministic",
        section_rollups=[
            SectionRollupV1(section="infrastructure", status="covered", article_count=3, digest_item_count=1)
        ],
        created_at=now,
    )
    return WorldPulseRunResultV1(
        run=WorldPulseRunV1(
            run_id=run_id,
            date="2026-05-20",
            started_at=now,
            completed_at=now,
            status="completed",
            dry_run=dry_run,
        ),
        digest=digest,
    )


def test_world_pulse_reflective_trigger_maps_to_digest_mode() -> None:
    trigger = build_world_pulse_reflective_trigger(_sample_run_result())
    assert trigger.trigger_kind == "world_pulse_digest"
    assert trigger.source_kind == "world_pulse"
    assert trigger.source_ref == "wp-run-1"
    assert journal_mode_for_trigger(trigger) == "digest"
    assert "Grid stress watch" in (trigger.prompt_seed or "")
    assert "executive_summary" in (trigger.prompt_seed or "").lower() or "Several policy" in (trigger.summary or "")


def test_world_pulse_cooldown_key_uses_run_id() -> None:
    trigger = build_world_pulse_reflective_trigger(_sample_run_result())
    key = cooldown_key_for_trigger(trigger)
    assert key == "actions:journal:world_pulse_digest:world_pulse:wp-run-1"


def test_world_pulse_compose_request_carries_trigger_metadata() -> None:
    trigger = build_world_pulse_reflective_trigger(_sample_run_result())
    req = build_compose_request(trigger, session_id="orion_journal", user_id="juniper", trace_id="corr-wp-1")
    assert req.verb == "journal.compose"
    assert req.context.metadata["journal_mode"] == "digest"
    assert req.context.metadata["journal_trigger"]["trigger_kind"] == "world_pulse_digest"
