from datetime import datetime, timezone

from orion.schemas.memory_consolidation import MemoryConsolidationWindowV1


def test_consolidation_status_accepts_skipped():
    w = MemoryConsolidationWindowV1(
        memory_window_id="win-1",
        status="consolidated",
        consolidation_status="skipped",
        created_at=datetime.now(timezone.utc),
    )
    assert w.consolidation_status == "skipped"
