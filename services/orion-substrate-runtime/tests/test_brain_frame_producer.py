from __future__ import annotations

from datetime import datetime, timezone


def test_brain_frame_schema_roundtrips_and_defaults():
    from orion.schemas.brain_frame import (
        SUBSTRATE_BRAIN_FRAME_KIND,
        BrainRegionV1,
        SubstrateBrainFrameV1,
    )

    assert SUBSTRATE_BRAIN_FRAME_KIND == "substrate.brain_frame.v1"

    region = BrainRegionV1(
        dimension="node_kind",
        region_id="node_kind:tension",
        label="Tension",
        intensity=0.9,
        state="firing",
        node_count=3,
        as_of=datetime(2026, 7, 7, tzinfo=timezone.utc),
        stale=False,
    )
    frame = SubstrateBrainFrameV1(
        frame_id="abc123",
        generated_at=datetime(2026, 7, 7, tzinfo=timezone.utc),
        tick_seq=1,
        phase="warming",
        regions=[region],
    )
    dumped = frame.model_dump(mode="json")
    again = SubstrateBrainFrameV1.model_validate(dumped)
    assert again.phase == "warming"
    assert again.regions[0].state == "firing"
    assert again.spotlight is None
    assert again.nodes == [] and again.edges == []
    assert again.schema_version == "substrate.brain_frame.v1"
