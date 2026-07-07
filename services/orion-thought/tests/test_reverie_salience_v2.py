from datetime import datetime, timezone

from orion.schemas.attention_frame import (
    AttentionBroadcastProjectionV1,
    AttentionFrameV1,
    OpenLoopV1,
)
from app.reverie import derive_salience


def _broadcast(selected_id: str, loops: list[OpenLoopV1]) -> AttentionBroadcastProjectionV1:
    frame = AttentionFrameV1(generated_at=datetime.now(timezone.utc), open_loops=loops)
    return AttentionBroadcastProjectionV1(
        generated_at=frame.generated_at, frame=frame,
        selected_open_loop_id=selected_id, coalition_stability_score=0.3,
    )


def test_derive_salience_uses_combiner_when_v2_on(monkeypatch):
    monkeypatch.setenv("ORION_ATTENTION_SALIENCE_V2_ENABLED", "true")
    loop = OpenLoopV1(id="loop-a", description="a", salience=0.81)
    b = _broadcast("loop-a", [loop])
    assert derive_salience(b) == 0.81


def test_derive_salience_legacy_when_v2_off(monkeypatch):
    monkeypatch.delenv("ORION_ATTENTION_SALIENCE_V2_ENABLED", raising=False)
    loop = OpenLoopV1(id="loop-a", description="a", novelty=0.6, emotional_charge=0.2)
    b = _broadcast("loop-a", [loop])
    # Legacy: max of the seven constant fields → 0.6.
    assert derive_salience(b) == 0.6
