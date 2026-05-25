from datetime import datetime, timezone
from pathlib import Path

from orion.consolidation.builder import build_consolidation_frame
from orion.consolidation.policy import load_consolidation_policy
from orion.consolidation.windows import ConsolidationWindowData, stable_consolidation_frame_id

REPO = Path(__file__).resolve().parents[1]
POLICY = load_consolidation_policy(REPO / "config" / "consolidation" / "consolidation_policy.v1.yaml")
NOW = datetime(2026, 5, 25, 15, 30, tzinfo=timezone.utc)
START = datetime(2026, 5, 25, 14, 30, tzinfo=timezone.utc)


def test_builder_emits_source_counts_and_stable_frame_id() -> None:
    window = ConsolidationWindowData(
        window_start=START,
        window_end=NOW,
        self_states=[],
        attention_frames=[],
        proposal_frames=[],
        policy_frames=[],
        dispatch_frames=[],
        feedback_frames=[],
    )
    frame = build_consolidation_frame(window=window, policy=POLICY, generated_at=NOW)
    assert frame.frame_id == stable_consolidation_frame_id(
        window_start=START, window_end=NOW, policy_id=POLICY.policy_id
    )
    assert frame.source_counts["self_state"] == 0
    assert frame.source_counts["feedback"] == 0
    assert frame.consolidation_policy_id == "consolidation_policy.v1"
