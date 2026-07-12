from __future__ import annotations

from orion.schemas.field_state import FieldStateV1


def apply_diffusion(state: FieldStateV1, *, diffusion_rate: float) -> None:
    # Phase 3 (2026-07-12): track, per (target_id, channel), the single
    # largest weighted contribution seen in this diffusion pass, so
    # state.capability_provenance can record which edge source "won" --
    # a per-tick proxy, not a historical ledger (see FieldStateV1's
    # capability_provenance docstring for why).
    _max_contribution: dict[tuple[str, str], float] = {}
    for edge in state.edges:
        src = state.node_vectors.get(edge.source_id) or state.capability_vectors.get(edge.source_id, {})
        tgt = state.capability_vectors.setdefault(edge.target_id, {})
        for src_ch, tgt_ch in edge.channel_map.items():
            src_val = float(src.get(src_ch, 0.0))
            contribution = src_val * edge.weight * diffusion_rate
            tgt[tgt_ch] = min(1.0, tgt.get(tgt_ch, 0.0) + contribution)
            key = (edge.target_id, tgt_ch)
            # contribution > 0.0 is required, not just >=: 0.0 is both the
            # "no contribution recorded yet this call" sentinel and a
            # legitimate zero-contribution value (a source with no value for
            # its mapped channel this tick). Without the strict >0 guard, a
            # genuinely-zero edge processed first in state.edges' order would
            # claim/overwrite provenance carried over from a real contributor
            # in a prior tick, even though it changed nothing (adding 0.0 to
            # tgt[tgt_ch] is a no-op) -- found by code review, reproduced.
            if contribution > 0.0 and contribution >= _max_contribution.get(key, 0.0):
                _max_contribution[key] = contribution
                state.capability_provenance.setdefault(edge.target_id, {})[tgt_ch] = edge.source_id
        if "available_capacity" in tgt:
            tgt["available_capacity"] = max(0.0, 1.0 - tgt.get("pressure", 0.0))
        if "confidence" in tgt:
            tgt["confidence"] = max(0.0, 1.0 - 0.5 * tgt.get("pressure", 0.0))
