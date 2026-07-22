from __future__ import annotations

from datetime import datetime, timezone

from orion.consolidation.policy import ConsolidationPolicyV1
from orion.consolidation.windows import ConsolidationWindowData
from orion.schemas.consolidation_frame import ExpectationV1, MotifObservationV1, SparseTensorSliceV1
from orion.schemas.execution_dispatch_frame import ExecutionDispatchCandidateV1, ExecutionDispatchFrameV1
from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1
from orion.schemas.proposal_frame import ProposalFrameV1


def _time_bucket(dt: datetime) -> str:
    normalized = dt.astimezone(timezone.utc).replace(second=0, microsecond=0)
    return normalized.isoformat()


def _stable_tensor_id(*, tensor_kind: str, window: ConsolidationWindowData) -> str:
    ws = window.window_start.astimezone(timezone.utc).isoformat()
    we = window.window_end.astimezone(timezone.utc).isoformat()
    return f"tensor:{tensor_kind}:{ws}:{we}"


def _cap_slice(
    *,
    coordinates: list[dict[str, str]],
    values: list[float],
    evidence_refs: list[str],
    max_coordinates: int,
) -> tuple[list[dict[str, str]], list[float], list[str]]:
    if len(coordinates) <= max_coordinates:
        return coordinates, values, evidence_refs
    return coordinates[:max_coordinates], values[:max_coordinates], evidence_refs[:max_coordinates]


def _proposal_kind(proposal_frames: list[ProposalFrameV1], proposal_frame_id: str) -> str:
    frame = next((item for item in proposal_frames if item.frame_id == proposal_frame_id), None)
    if frame is None or not frame.candidates:
        return "unknown"
    return frame.candidates[0].proposal_kind


def _policy_decision(policy_frames: list[PolicyDecisionFrameV1], policy_frame_id: str) -> str:
    frame = next((item for item in policy_frames if item.frame_id == policy_frame_id), None)
    if frame is None or not frame.decisions:
        return "unknown"
    if frame.approved_decisions:
        return frame.approved_decisions[0].decision
    return frame.decisions[0].decision


def _feedback_outcome(
    *,
    dispatch_frame_id: str,
    feedback_by_dispatch: dict[str, str],
) -> str:
    return feedback_by_dispatch.get(dispatch_frame_id, "absent")


def _append_dispatch_path(
    *,
    coordinates: list[dict[str, str]],
    values: list[float],
    evidence_refs: list[str],
    candidate: ExecutionDispatchCandidateV1,
    dispatch_frame: ExecutionDispatchFrameV1,
    proposal_kind: str,
    policy_decision: str,
    feedback_outcome: str,
) -> None:
    coordinates.append(
        {
            "proposal_kind": proposal_kind,
            "policy_decision": policy_decision,
            "dispatch_status": candidate.dispatch_status,
            "feedback_outcome": feedback_outcome,
        }
    )
    values.append(1.0)
    evidence_refs.append(dispatch_frame.frame_id)


def _build_policy_dispatch_feedback_slice(
    *,
    window: ConsolidationWindowData,
    policy: ConsolidationPolicyV1,
) -> SparseTensorSliceV1:
    axes = list(policy.tensor_axes.get("policy_dispatch_feedback", []))
    feedback_by_dispatch = {
        frame.source_execution_dispatch_frame_id: frame.outcome_status
        for frame in window.feedback_frames
    }
    coordinates: list[dict[str, str]] = []
    values: list[float] = []
    evidence_refs: list[str] = []

    for dispatch_frame in window.dispatch_frames:
        proposal_kind = _proposal_kind(window.proposal_frames, dispatch_frame.source_proposal_frame_id)
        policy_decision = _policy_decision(window.policy_frames, dispatch_frame.source_policy_frame_id)
        feedback_outcome = _feedback_outcome(
            dispatch_frame_id=dispatch_frame.frame_id,
            feedback_by_dispatch=feedback_by_dispatch,
        )
        for bucket in (
            dispatch_frame.candidates,
            dispatch_frame.blocked_candidates,
            dispatch_frame.dispatched_candidates,
        ):
            for candidate in bucket:
                _append_dispatch_path(
                    coordinates=coordinates,
                    values=values,
                    evidence_refs=evidence_refs,
                    candidate=candidate,
                    dispatch_frame=dispatch_frame,
                    proposal_kind=proposal_kind,
                    policy_decision=policy_decision,
                    feedback_outcome=feedback_outcome,
                )

    coordinates, values, evidence_refs = _cap_slice(
        coordinates=coordinates,
        values=values,
        evidence_refs=evidence_refs,
        max_coordinates=policy.tensor.max_coordinates,
    )
    return SparseTensorSliceV1(
        tensor_id=_stable_tensor_id(tensor_kind="policy_dispatch_feedback", window=window),
        tensor_kind="policy_dispatch_feedback",
        axes=axes,
        coordinates=coordinates,
        values=values,
        evidence_refs=sorted(set(evidence_refs)),
    )


def _outcome_status_for_motif(
    *,
    motif: MotifObservationV1,
    feedback_by_id: dict[str, str],
) -> str:
    for frame_id in motif.evidence_frame_ids:
        if frame_id in feedback_by_id:
            return feedback_by_id[frame_id]
    if feedback_by_id:
        return next(iter(feedback_by_id.values()))
    return "unknown"


def _build_motif_condition_outcome_slice(
    *,
    window: ConsolidationWindowData,
    motifs: list[MotifObservationV1],
    policy: ConsolidationPolicyV1,
) -> SparseTensorSliceV1:
    axes = list(policy.tensor_axes.get("motif_condition_outcome", []))
    feedback_by_id = {frame.frame_id: frame.outcome_status for frame in window.feedback_frames}
    coordinates: list[dict[str, str]] = []
    values: list[float] = []
    evidence_refs: list[str] = []

    for motif in motifs:
        coordinates.append(
            {
                "motif": motif.label,
                # "unknown" always, 2026-07-22 SelfStateV1 burn: self_condition
                # had no other source than window.self_states, which no longer
                # exists (docs/superpowers/specs/2026-07-22-self-state-phi-
                # endo-origination-burn-spec.md). Coordinate kept (not removed)
                # so this tensor's axis shape stays stable for existing
                # consumers/analysis code; disclosed as always-unknown rather
                # than silently dropped.
                "self_condition": "unknown",
                "outcome_status": _outcome_status_for_motif(motif=motif, feedback_by_id=feedback_by_id),
            }
        )
        values.append(float(motif.support_score))
        evidence_refs.extend(motif.evidence_frame_ids)

    coordinates, values, evidence_refs = _cap_slice(
        coordinates=coordinates,
        values=values,
        evidence_refs=evidence_refs,
        max_coordinates=policy.tensor.max_coordinates,
    )
    return SparseTensorSliceV1(
        tensor_id=_stable_tensor_id(tensor_kind="motif_condition_outcome", window=window),
        tensor_kind="motif_condition_outcome",
        axes=axes,
        coordinates=coordinates,
        values=values,
        evidence_refs=sorted(set(evidence_refs)),
    )


def build_sparse_tensor_slices(
    *,
    window: ConsolidationWindowData,
    motifs: list[MotifObservationV1],
    expectations: list[ExpectationV1],
    policy: ConsolidationPolicyV1,
) -> list[SparseTensorSliceV1]:
    del expectations
    if not policy.tensor.enabled:
        return []
    # "field_attention_self" slice removed 2026-07-22, SelfStateV1 burn
    # (docs/superpowers/specs/2026-07-22-self-state-phi-endo-origination-burn-
    # spec.md): its only input was window.self_states, which no longer exists.
    return [
        _build_policy_dispatch_feedback_slice(window=window, policy=policy),
        _build_motif_condition_outcome_slice(window=window, motifs=motifs, policy=policy),
    ]
