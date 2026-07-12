from __future__ import annotations

from orion.schemas.field_state import FieldStateV1


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def apply_diffusion(state: FieldStateV1, *, diffusion_rate: float) -> None:
    """Recompute every diffused capability channel fresh from this tick's node/
    capability state (2026-07-12 fix -- see
    docs/notes/2026-07-12-metrics-swamp-arsonist-review.md and the redesign
    plan's Phase 1 for the double-counting bug this is related to but
    distinct from).

    Previously this accumulated across ticks (`tgt[ch] = min(1.0, tgt.get(ch,
    0.0) + contribution)`), carrying forward whatever was already there. With
    `BIOMETRICS_FIELD_DECAY_RATE=0.92` (8% removed per tick) and multiple real
    edges/channels routinely contributing up to `weight * diffusion_rate`
    (commonly ~0.9) each tick, the additive carryover overwhelmed decay almost
    immediately and pinned capability channels like `capability:orchestration`'s
    "pressure" at exactly 1.0 permanently, regardless of current real load --
    confirmed live: `resource_pressure` had been dead-flat 1.0 for the entire
    observed history, including before this fix, unrelated to any actual
    hardware/transport state.

    Root cause was two compounding issues, both fixed here:
    1. Multiple source channels legitimately feed the same target channel by
       design (e.g. node:athena->capability:orchestration maps BOTH
       cpu_pressure AND transport_pressure onto "pressure" -- orchestration
       capability is meant to reflect either stressor). These were SUMMED
       into the target, even within a single tick, rather than combined via
       max() -- two moderately-stressed channels alone could push a target
       near ceiling in one tick. Now: whichever contributing channel is
       currently worse dominates that tick's reading for the shared target,
       matching the same principle already used everywhere else in this
       redesign (self-state's own channel merging, the Phase 1 double-count
       fix).
    2. The result then carried forward and accumulated further next tick.
       Now: fully memoryless per diffusion-target channel. Each tick
       recomputes every such channel from scratch from the current
       node_vectors/capability_vectors, the same way orion-self-state-
       runtime's build_self_state() is already memoryless. A capability's
       pressure now reflects current load, not an ever-ratcheting historical
       maximum. Channels no edge currently contributes to are explicitly
       zeroed (not left stale) -- but only the specific channels that are
       ever a diffusion target for that capability; everything else in the
       dict (e.g. a confidence/available_capacity baseline the caller already
       reconciled in) is left untouched, so a capability with no incoming
       "pressure" edge at all still keeps its sensible 1.0 defaults instead
       of losing the key entirely.

    capability_provenance (Phase 3) is updated to match: previously it
    already tracked "which edge source contributed the most THIS tick" (a
    memoryless computation) while the actual persisted VALUE was a multi-tick
    accumulated blend from whichever past ticks contributed -- an internal
    inconsistency (provenance and the score it was explaining described two
    different things). Both are now memoryless together, so provenance and
    score are always consistent: provenance names the source that actually
    produced the score displayed this tick, and is cleared (not left stale)
    when nobody contributes.

    Precedence between direct diffusion and the pressure-derived formula
    (2026-07-12 follow-up): a capability like capability:transport has BOTH a
    direct diffusion edge into "confidence"/"available_capacity"
    (delivery_confidence->confidence, bus_health->available_capacity) AND a
    "pressure" target, whose derived formula below would otherwise
    unconditionally overwrite them. Direct diffusion wins when it actually
    contributed a real (>0) value THIS tick (checked via best_source, not via
    static channel-map membership -- a capability can be configured with a
    direct confidence/available_capacity edge yet receive nothing from it in
    a given tick, e.g. its source is temporarily missing that field; in that
    case the derived formula must still run as a fallback instead of leaving
    the channel hard-floored at 0.0). Every other capability (no direct edge
    into either channel) is unaffected. This was previously masked because
    transport_pressure has been idle (0.0) in observed traffic, making both
    formulas agree by coincidence.
    """
    best_contribution: dict[tuple[str, str], float] = {}
    best_source: dict[tuple[str, str], str] = {}
    possible_targets: dict[str, set[str]] = {}

    for edge in state.edges:
        src = state.node_vectors.get(edge.source_id) or state.capability_vectors.get(edge.source_id, {})
        for src_ch, tgt_ch in edge.channel_map.items():
            possible_targets.setdefault(edge.target_id, set()).add(tgt_ch)
            src_val = float(src.get(src_ch, 0.0))
            contribution = _clamp01(src_val * edge.weight * diffusion_rate)
            key = (edge.target_id, tgt_ch)
            # Only a real (>0) contribution may claim provenance/win the
            # max -- a zero contribution (source has no value for its mapped
            # channel this tick) must not overwrite another edge's real one
            # just by being evaluated later in iteration order.
            if contribution > 0.0 and contribution >= best_contribution.get(key, 0.0):
                best_contribution[key] = contribution
                best_source[key] = edge.source_id

    for target_id, channels in possible_targets.items():
        tgt = state.capability_vectors.setdefault(target_id, {})
        provenance = state.capability_provenance.setdefault(target_id, {})
        for tgt_ch in channels:
            key = (target_id, tgt_ch)
            tgt[tgt_ch] = best_contribution.get(key, 0.0)
            if key in best_source:
                provenance[tgt_ch] = best_source[key]
            else:
                # Nobody contributed this tick -- clear stale provenance too,
                # not just reset the value, so the two never disagree about
                # what's currently true.
                provenance.pop(tgt_ch, None)

        if "pressure" in tgt:
            # Gate on best_source (a real >0 contribution THIS tick), not on
            # `channels` (every channel ever configured as a target for this
            # capability, whether or not it fired this tick) -- a capability
            # can have "available_capacity"/"confidence" as configured
            # targets yet receive no contribution some tick (e.g. its source
            # is temporarily missing that field), in which case the derived
            # fallback must still run instead of leaving the channel
            # hard-floored at 0.0 by the `best_contribution.get(key, 0.0)`
            # reset above.
            if (target_id, "available_capacity") not in best_source:
                tgt["available_capacity"] = max(0.0, 1.0 - tgt.get("pressure", 0.0))
            if (target_id, "confidence") not in best_source:
                tgt["confidence"] = max(0.0, 1.0 - 0.5 * tgt.get("pressure", 0.0))
