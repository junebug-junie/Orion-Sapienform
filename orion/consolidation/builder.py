from __future__ import annotations

from datetime import datetime, timezone

from orion.consolidation.expectation import build_expectations_from_motifs
from orion.consolidation.motif import detect_motifs
from orion.consolidation.policy import ConsolidationPolicyV1
from orion.consolidation.windows import ConsolidationWindowData, stable_consolidation_frame_id
from orion.schemas.consolidation_frame import ConsolidationFrameV1


def build_consolidation_frame(
    *,
    window: ConsolidationWindowData,
    policy: ConsolidationPolicyV1,
    generated_at: datetime | None = None,
) -> ConsolidationFrameV1:
    now = generated_at or datetime.now(timezone.utc)
    motifs = detect_motifs(window=window, policy=policy)
    expectations = build_expectations_from_motifs(
        motifs=motifs,
        feedback_frames=window.feedback_frames,
        policy=policy,
    )
    dominant = [
        m.label
        for m in sorted(motifs, key=lambda x: x.support_score, reverse=True)
        if m.support_score >= policy.motif_thresholds.dominant_motif_min_support
    ]
    return ConsolidationFrameV1(
        frame_id=stable_consolidation_frame_id(
            window_start=window.window_start,
            window_end=window.window_end,
            policy_id=policy.policy_id,
        ),
        generated_at=now,
        window_start=window.window_start,
        window_end=window.window_end,
        consolidation_policy_id=policy.policy_id,
        motif_observations=motifs,
        dominant_motifs=dominant,
        expectations=expectations,
        source_counts={
            "self_state": len(window.self_states),
            "attention": len(window.attention_frames),
            "proposal": len(window.proposal_frames),
            "policy": len(window.policy_frames),
            "dispatch": len(window.dispatch_frames),
            "feedback": len(window.feedback_frames),
        },
    )
