from __future__ import annotations

from orion.schemas.field_attention_frame import FieldAttentionFrameV1, FieldAttentionTargetV1


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _find_prior_target(
    target_id: str,
    previous_frame: FieldAttentionFrameV1 | None,
) -> FieldAttentionTargetV1 | None:
    """Real search order for "did this target_id have a real entry in the
    previous frame" -- all five buckets a target can land in, not just the
    two "active" ones (`node_targets`/`capability_targets`). A target
    scored below `policy.thresholds.suppress_below` or between
    `suppress_below`/`min_salience` last tick lands in `suppressed_targets`
    only (see `build_attention_frame()`), not in `node_targets`/
    `capability_targets` -- a real prior observation, just not an "active"
    one. Any caller asking "does a real prior value exist for this target"
    (confidence claims, not just novelty diffing) must use the same search
    this function does, or it will under-report confidence for a target
    that was real but suppressed last tick (code review, 2026-07-30).
    """
    if previous_frame is None:
        return None
    for bucket in (
        previous_frame.dominant_targets,
        previous_frame.node_targets,
        previous_frame.capability_targets,
        previous_frame.system_targets,
        previous_frame.suppressed_targets,
    ):
        for t in bucket:
            if t.target_id == target_id:
                return t
    return None


def prior_salience_for_target(
    target_id: str,
    previous_frame: FieldAttentionFrameV1 | None,
) -> float:
    found = _find_prior_target(target_id, previous_frame)
    return found.salience_score if found is not None else 0.0


def target_had_real_prior_entry(
    target_id: str,
    previous_frame: FieldAttentionFrameV1 | None,
) -> bool:
    """Whether `target_id` had a real entry in ANY of the previous frame's
    five target buckets (dominant/node/capability/system/suppressed) --
    the same search `prior_salience_for_target()` uses. A caller reporting
    `confidence_score` for a novelty claim must ask this, not just whether
    `previous_frame is not None` or whether the target is in one particular
    "active" bucket -- both under-report confidence for a target that
    really was observed but landed in a bucket the caller didn't check
    (code review, 2026-07-30: `_novelty_targets()` originally checked only
    `node_targets`/`capability_targets`, missing `suppressed_targets`).
    """
    return _find_prior_target(target_id, previous_frame) is not None


def novelty_for_target(
    target_id: str,
    current_salience: float,
    previous_frame: FieldAttentionFrameV1 | None,
) -> float:
    if previous_frame is None:
        return 0.0
    prior = prior_salience_for_target(target_id, previous_frame)
    return clamp01(abs(current_salience - prior))
