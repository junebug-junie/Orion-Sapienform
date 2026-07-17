from __future__ import annotations

from datetime import datetime, timezone

from orion.schemas.field_state import FieldStateV1

NODE_DECAY_CHANNELS = {
    # hardware / biometrics
    "staleness",
    "cpu_pressure",
    "memory_pressure",
    "gpu_pressure",
    "thermal_pressure",
    "disk_pressure",
    # execution
    "execution_load",
    "execution_friction",
    "reasoning_load",
    "failure_pressure",
    "egress_confidence_deficit",
    "repair_pressure",
    "conversation_load",
    # transport
    "transport_pressure",
    "contract_pressure",
    "catalog_drift_pressure",
    "observer_failure_pressure",
    "reliability_pressure",
    "field_coherence_warning",
    "prediction_error",
}

CAPABILITY_DECAY_CHANNELS = {
    "pressure",
    "execution_pressure",
    "reasoning_pressure",
    "reliability_pressure",
    "transport_pressure",
    "contract_pressure",
}


def _aware_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def apply_decay(
    state: FieldStateV1,
    *,
    decay_rate: float,
    now: datetime,
    staleness_threshold_sec: float,
) -> None:
    # Decay/injection-interval mismatch fix (2026-07-17, docs/superpowers/specs/
    # 2026-07-17-field-digester-decay-hold-fix-design.md): hold each channel's
    # value by default; only decay once it has actually gone stale (no fresh
    # perturbation within staleness_threshold_sec). Previously this loop
    # decayed every NODE_DECAY_CHANNELS entry unconditionally every 2s tick
    # regardless of whether fresh data arrived that tick, producing a
    # mechanical sawtooth against the ~15-30s real biometrics publish cadence
    # (0.92^7 ~= 0.56 -- ~44% lost between two real publishes, then snapped
    # straight back up with zero memory of the decayed trajectory).
    now_aware = _aware_utc(now)
    for node_id, vec in state.node_vectors.items():
        updated_at = state.node_vector_updated_at.get(node_id, {})
        for ch in NODE_DECAY_CHANNELS:
            if ch not in vec:
                continue
            last_updated_at = updated_at.get(ch)
            # Missing (never perturbed, or persisted from before this fix):
            # safe default, decay as today. Present but still fresh (within
            # staleness_threshold_sec of its last real write): hold, skip
            # decay this tick. Otherwise (present and genuinely stale):
            # decay as today.
            is_fresh = last_updated_at is not None and (
                now_aware - _aware_utc(last_updated_at)
            ).total_seconds() < staleness_threshold_sec
            if is_fresh:
                continue
            vec[ch] = vec[ch] * decay_rate
    # NOTE (2026-07-12, found by code review on the diffusion memoryless-
    # recompute fix): this capability_vectors loop -- and its
    # "available_capacity" recompute below -- is currently dead weight for
    # every capability in the live topology (config/field/orion_field_topology.v1.yaml).
    # apply_diffusion() runs immediately after this in the same tick
    # (app/tensor/update_rules.py) and now unconditionally overwrites every
    # CAPABILITY_DECAY_CHANNELS entry that is ever a diffusion target for a
    # capability with a fresh, memoryless recompute -- discarding whatever
    # this loop just wrote moments earlier. Any CAPABILITY_DECAY_CHANNELS
    # entry that is NOT a diffusion target for a given capability is never
    # written by anything else either (perturbation.py only writes
    # node_vectors), so it stays at its reconcile-seeded 0.0 regardless of
    # decay. Left in place rather than removed -- this was load-bearing
    # under the OLD accumulating diffusion model (the only counterweight to
    # unbounded additive growth) and is a reasonable safety net if a future
    # change ever writes a capability channel directly, outside diffusion,
    # without its own decay story. Do not assume this loop is currently
    # bounding anything live; verify against apply_diffusion() first if this
    # matters for a future change.
    for vec in state.capability_vectors.values():
        for ch in CAPABILITY_DECAY_CHANNELS:
            if ch in vec:
                vec[ch] = vec[ch] * decay_rate
        if "available_capacity" in vec:
            vec["available_capacity"] = min(1.0, 1.0 - vec.get("pressure", 0.0))
