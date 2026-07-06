from datetime import datetime, timezone

from orion.autonomy.substrate_metabolism import metabolize_substrate_signals
from orion.schemas.world_pulse import (
    DailyWorldPulseSectionsV1,
    DailyWorldPulseV1,
    SectionRollupV1,
    WorldPulseRunResultV1,
    WorldPulseRunV1,
)


def _gpu_gap_result() -> WorldPulseRunResultV1:
    run_id = "wp-run-gap-gpu"
    now = datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc)
    digest = DailyWorldPulseV1(
        run_id=run_id,
        date="2026-07-06",
        generated_at=now,
        title="Daily World Pulse",
        executive_summary="Sparse GPU coverage.",
        sections=DailyWorldPulseSectionsV1(),
        items=[],
        orion_analysis_layer="deterministic",
        coverage_status="sparse",
        section_rollups=[
            SectionRollupV1(
                section="hardware_compute_gpu",
                status="missing",
                article_count=0,
                digest_item_count=0,
                confidence=0.35,
            ),
            SectionRollupV1(
                section="ai_technology",
                status="covered",
                article_count=2,
                digest_item_count=1,
                confidence=1.0,
            ),
        ],
        created_at=now,
    )
    return WorldPulseRunResultV1(
        run=WorldPulseRunV1(
            run_id=run_id,
            date="2026-07-06",
            started_at=now,
            completed_at=now,
            status="completed",
            dry_run=False,
        ),
        digest=digest,
    )


def test_metabolism_sparse_gpu_section_raises_predictive() -> None:
    result = metabolize_substrate_signals(world_pulse_result=_gpu_gap_result())
    assert result.drive_deltas.get("predictive", 0.0) > 0.0
    assert any(t.kind == "substrate.world_coverage_gap" for t in result.tensions)
    assert any(
        s.signal_type == "world_coverage_gap" and "hardware_compute_gpu" in s.focal_node_refs[0]
        for s in result.curiosity_signals
    )


def test_metabolism_skips_covered_sections() -> None:
    covered_only = _gpu_gap_result()
    covered_only.digest.section_rollups = [
        SectionRollupV1(section="hardware_compute_gpu", status="covered", digest_item_count=2)
    ]
    result = metabolize_substrate_signals(world_pulse_result=covered_only)
    assert result.drive_deltas == {}
    assert result.tensions == []
    assert result.curiosity_signals == []
