from __future__ import annotations

from datetime import datetime, timezone

from orion.journaler import (
    JournalEntryDraftV1,
    build_compose_request,
    build_world_pulse_reflective_trigger,
    cooldown_key_for_trigger,
    journal_mode_for_trigger,
    merge_world_pulse_curiosity_into_draft,
)
from orion.schemas.world_pulse import (
    CuriosityFindingV1,
    CuriosityFollowupV1,
    DailyWorldPulseItemV1,
    DailyWorldPulseSectionsV1,
    DailyWorldPulseV1,
    SectionRollupV1,
    WorldPulseRunResultV1,
    WorldPulseRunV1,
)


def _sample_run_result(
    *, dry_run: bool = False, curiosity_followups: list[CuriosityFollowupV1] | None = None
) -> WorldPulseRunResultV1:
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
        curiosity_followups=curiosity_followups or [],
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


def test_world_pulse_seed_folds_in_autonomy_gap_fill_when_present() -> None:
    # Consolidation: the "Orion went looking" autonomy gap-fill narrative is folded
    # into the single world-pulse journal seed so the standalone autonomy_episode
    # journal (and its duplicate email) can be retired.
    followups = [
        CuriosityFollowupV1(
            section="hardware_compute_gpu",
            driving_gap="missing",
            query="new datacenter GPU supply",
            articles=[
                CuriosityFindingV1(
                    url="https://ex/1",
                    title="Fab ramps output",
                    description="capacity note",
                    salience=0.71,
                )
            ],
            action_id="goal-afe6211f",
        )
    ]
    trigger = build_world_pulse_reflective_trigger(_sample_run_result(curiosity_followups=followups))
    seed = trigger.prompt_seed or ""
    assert "orion_went_looking" in seed
    assert "hardware compute gpu" in seed
    assert "new datacenter GPU supply" in seed
    assert "Fab ramps output" in seed
    assert "https://ex/1" in seed


def test_world_pulse_seed_marks_coverage_gaps_when_no_followups() -> None:
    result = _sample_run_result()
    result.digest.section_rollups.append(
        SectionRollupV1(section="hardware_compute_gpu", status="missing", article_count=0, digest_item_count=0)
    )
    seed = build_world_pulse_reflective_trigger(result).prompt_seed or ""
    assert "coverage_gaps" in seed
    assert "hardware_compute_gpu" in seed
    assert "orion_went_looking" not in seed


def test_world_pulse_seed_marks_empty_gap_fill_and_renders_salience() -> None:
    followups = [
        CuriosityFollowupV1(section="policy_regulation", driving_gap="no_articles", query="tariff rule", articles=[]),
        CuriosityFollowupV1(
            section="hardware_compute_gpu",
            driving_gap="missing",
            query="gpu supply",
            articles=[CuriosityFindingV1(url="https://ex/2", title="Fab note", salience=0.5)],
        ),
    ]
    seed = build_world_pulse_reflective_trigger(_sample_run_result(curiosity_followups=followups)).prompt_seed or ""
    assert "(looked, found nothing)" in seed
    assert "[0.50] Fab note" in seed


def test_world_pulse_seed_omits_gap_fill_section_when_no_followups() -> None:
    trigger = build_world_pulse_reflective_trigger(_sample_run_result())
    assert "orion_went_looking" not in (trigger.prompt_seed or "")


def test_world_pulse_compose_request_carries_trigger_metadata() -> None:
    trigger = build_world_pulse_reflective_trigger(_sample_run_result())
    req = build_compose_request(trigger, session_id="orion_journal", user_id="juniper", trace_id="corr-wp-1")
    assert req.verb == "journal.compose"
    assert req.context.metadata["journal_mode"] == "digest"
    assert req.context.metadata["journal_trigger"]["trigger_kind"] == "world_pulse_digest"


def test_merge_world_pulse_curiosity_appends_missing_urls() -> None:
    followups = [
        CuriosityFollowupV1(
            section="hardware_compute_gpu",
            driving_gap="missing",
            query="hardware compute gpu recent news coverage",
            articles=[
                CuriosityFindingV1(
                    url="https://nvidianews.nvidia.com/news/nvidia-blackwell",
                    title="NVIDIA Blackwell Platform",
                    salience=0.67,
                )
            ],
        )
    ]
    result = _sample_run_result(curiosity_followups=followups)
    vague = JournalEntryDraftV1(
        mode="digest",
        title="Journal Pass",
        body="Orion looked and found some NVIDIA-related content, but details were vague.",
    )
    merged = merge_world_pulse_curiosity_into_draft(vague, result)
    assert "https://nvidianews.nvidia.com/news/nvidia-blackwell" in merged.body
    assert "NVIDIA Blackwell Platform" in merged.body
    assert "## Orion went looking" in merged.body


def test_merge_world_pulse_curiosity_noop_when_urls_already_present() -> None:
    followups = [
        CuriosityFollowupV1(
            section="hardware_compute_gpu",
            driving_gap="missing",
            query="gpu",
            articles=[CuriosityFindingV1(url="https://ex/1", title="Fab", salience=0.5)],
        )
    ]
    result = _sample_run_result(curiosity_followups=followups)
    body = "Already included https://ex/1 in prose."
    draft = JournalEntryDraftV1(mode="digest", title="Journal Pass", body=body)
    merged = merge_world_pulse_curiosity_into_draft(draft, result)
    assert merged.body == body
