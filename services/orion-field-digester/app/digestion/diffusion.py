from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from orion.schemas.field_state import FieldStateV1
from orion.substrate.field_topology_plasticity import edge_ref_for

logger = logging.getLogger(__name__)

# Same truthy-parsing convention used repo-wide for boolean env flags (e.g.
# orion/substrate/endogenous_curiosity.py, orion/substrate/attention/salience.py,
# orion/substrate/felt_state_reader.py) -- str(os.getenv(name, "false")).strip()
# .lower() in _TRUTHY. Matched here rather than invented fresh so this flag reads
# the same way every other on/off switch in the codebase does.
_TRUTHY = {"1", "true", "yes", "on"}

FIELD_PLASTICITY_ENABLED_ENV = "FIELD_PLASTICITY_ENABLED"

# Must resolve to the same file the hub's CAUSAL_GEOMETRY_PROPOSAL_STORE writes to
# (services/orion-hub/scripts/api_routes.py) -- an operator's HITL adopt() only ever
# reaches this process's diffusion tick if both processes point at the same sqlite
# file. Unset (the .env_example default) means "in-memory, this process only" -- the
# overlay will never see a cross-process adoption, matching pre-plasticity behavior.
FIELD_PLASTICITY_SQL_DB_PATH_ENV = "FIELD_PLASTICITY_SQL_DB_PATH"

_LEARNED_STORE = None  # lazily constructed once; see _get_learned_store()

# cap->cap edges only, per the Causal Geometry v1 design spec (Phase B) --
# node_capability/node_service/etc. edges never consult the learned overlay,
# even defensively (the read-path below gates on this every time, regardless
# of whether an overlay entry happens to exist for a non-cap-cap edge's
# (source_id, target_id) pair).
CAPABILITY_CAPABILITY_EDGE_TYPE = "capability_capability"


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _plasticity_enabled() -> bool:
    return str(os.getenv(FIELD_PLASTICITY_ENABLED_ENV, "false")).strip().lower() in _TRUTHY


def get_learned_store():
    """Lazily construct (once, not per-tick) the shared
    `FieldTopologyLearnedWeightsStore`, backed by `FIELD_PLASTICITY_SQL_DB_PATH` when
    set so adoptions made in the hub process are actually visible here.

    Constructing this once and caching it (rather than fresh per `apply_diffusion`
    call) avoids reconnecting + reloading the sqlite file on every diffusion tick for
    state that only changes on a rare HITL adopt/reject action.

    Public (not `_`-prefixed): `app/worker.py`'s causal-geometry producer loop
    calls this too, so the read side (this module) and the write side (the
    producer's `store.propose()`) share one store instance/sqlite connection
    within this process instead of each opening their own.
    """
    global _LEARNED_STORE
    if _LEARNED_STORE is None:
        from orion.substrate.field_topology_learned_store import (
            FieldTopologyLearnedWeightsStore,
        )

        sql_db_path = str(os.getenv(FIELD_PLASTICITY_SQL_DB_PATH_ENV, "")).strip() or None
        _LEARNED_STORE = FieldTopologyLearnedWeightsStore(sql_db_path=sql_db_path)
    return _LEARNED_STORE


def _load_learned_overlay() -> dict[str, float]:
    """Best-effort read of HITL-adopted cap->cap edge-weight overrides
    (`orion/substrate/field_topology_learned_store.py`'s
    `FieldTopologyLearnedWeightsStore.current_overlay()`).

    Never raises: construction of the store (e.g. a misconfigured sqlite
    path) or the `current_overlay()` call itself failing for any reason
    degrades to an empty overlay -- equivalent to no learned deltas being
    applied this tick, i.e. every cap->cap edge falls back to its raw
    designed weight, exactly as if `FIELD_PLASTICITY_ENABLED` were off.
    """
    try:
        return get_learned_store().current_overlay()
    except Exception as exc:  # pragma: no cover - defensive, see docstring above
        logger.warning("field_plasticity_overlay_load_failed: %s", exc, exc_info=True)
        return {}


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

    # Causal Geometry v1, Rung 3A: when FIELD_PLASTICITY_ENABLED is off (the
    # default), the store is never constructed and the overlay dict stays
    # empty -- zero extra work, and every edge below falls straight through
    # to its raw designed `edge.weight`, byte-identical to pre-plasticity
    # behavior. `edge.weight` itself is never mutated; the effective weight
    # is computed locally per-edge inside the loop.
    plasticity_enabled = _plasticity_enabled()
    learned_overlay: dict[str, float] = _load_learned_overlay() if plasticity_enabled else {}

    for edge in state.edges:
        src = state.node_vectors.get(edge.source_id) or state.capability_vectors.get(edge.source_id, {})
        effective_weight = edge.weight
        if (
            plasticity_enabled
            and learned_overlay
            and edge.edge_type == CAPABILITY_CAPABILITY_EDGE_TYPE
        ):
            edge_ref = edge_ref_for(edge.source_id, edge.target_id)
            delta = learned_overlay.get(edge_ref, 0.0)
            if delta:
                effective_weight = _clamp01(edge.weight + delta)
                edge.weight_source = "learned"
                edge.learned_at = datetime.now(timezone.utc)
                logger.info(
                    "field_plasticity_learned_weight_applied edge_ref=%s designed_weight=%.4f "
                    "delta=%.4f effective_weight=%.4f",
                    edge_ref,
                    edge.weight,
                    delta,
                    effective_weight,
                )
        for src_ch, tgt_ch in edge.channel_map.items():
            possible_targets.setdefault(edge.target_id, set()).add(tgt_ch)
            src_val = float(src.get(src_ch, 0.0))
            contribution = _clamp01(src_val * effective_weight * diffusion_rate)
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
