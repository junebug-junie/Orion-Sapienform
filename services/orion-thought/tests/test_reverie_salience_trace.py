from datetime import datetime, timezone

from orion.schemas.attention_frame import (
    AttentionBroadcastProjectionV1,
    AttentionFrameV1,
    OpenLoopV1,
)
from app.reverie import build_salience_trace


def _broadcast(loop: OpenLoopV1) -> AttentionBroadcastProjectionV1:
    frame = AttentionFrameV1(generated_at=datetime.now(timezone.utc), open_loops=[loop])
    return AttentionBroadcastProjectionV1(
        generated_at=frame.generated_at, frame=frame,
        selected_open_loop_id=loop.id, coalition_stability_score=0.3,
    )


def test_build_salience_trace_from_selected_loop():
    loop = OpenLoopV1(id="loop-a", description="a", salience=0.72,
                      salience_features={"evidence_strength": 0.8})
    trace = build_salience_trace(_broadcast(loop), correlation_id="corr-1")
    assert trace is not None
    assert trace.loop_id == "loop-a"
    assert trace.salience == 0.72
    assert trace.features == {"evidence_strength": 0.8}
    assert trace.scope == "reverie"


def test_build_salience_trace_none_without_selection():
    frame = AttentionFrameV1(generated_at=datetime.now(timezone.utc), open_loops=[])
    b = AttentionBroadcastProjectionV1(generated_at=frame.generated_at, frame=frame,
                                       selected_open_loop_id=None)
    assert build_salience_trace(b, correlation_id="c") is None
