NODE_CHANNELS = [
    "availability",
    "staleness",
    "cpu_pressure",
    "memory_pressure",
    "gpu_pressure",
    "thermal_pressure",
    "disk_pressure",
    "expected_offline_suppression",
    "execution_load",
    "execution_friction",
    "reasoning_load",
    "failure_pressure",
    "egress_confidence_deficit",
    "repair_pressure",
    "harness_step_load",
    "tool_failure_streak_pressure",
    "avg_step_chars_pressure",
    "compliance_deficit",
    "turn_incompletion",
    "conversation_load",
    "transport_pressure",
    "contract_pressure",
    "catalog_drift_pressure",
    "delivery_confidence",
    "bus_health",
    "observer_failure_pressure",
    "field_coherence_warning",
    "prediction_error",
]
CAPABILITY_CHANNELS = [
    "pressure",
    "confidence",
    "available_capacity",
    "execution_pressure",
    "reasoning_pressure",
    "reliability_pressure",
    "transport_pressure",
    "contract_pressure",
]

DEFAULT_NODE_VECTOR = {ch: 0.0 for ch in NODE_CHANNELS}
DEFAULT_NODE_VECTOR["availability"] = 1.0
# Kept at 1.0 (see SINGLE_OBSERVER_NODE_CHANNELS below): this is the
# "presumed healthy until first real report" default for node:athena itself
# the very first time it reconciles, before its own bus-observer has ever
# reported. It's no longer reached by any other node -- see below.
DEFAULT_NODE_VECTOR["bus_health"] = 1.0
DEFAULT_NODE_VECTOR["delivery_confidence"] = 1.0

# Design correction (2026-07-22): bus_health/delivery_confidence were
# modeled as ordinary per-node NODE_CHANNELS, merged across every lattice
# node via min() (the now-deleted orion/self_state/scoring.py's
# HIGHER_IS_BETTER_CHANNELS handling -- worst node wins). But there is
# exactly one bus, it runs on
# athena, and only athena's bus-observer service (BUS_OBSERVER_NODE_ID)
# ever produces a real reading -- atlas/circe run llamacpp-host + biometrics
# only, prometheus likewise, none of them have any code path that could
# ever legitimately report bus health. They aren't "nodes with an
# unreported bus health," they have no standing to have an opinion on it at
# all. The 2026-07-17 fix (default 0.0 -> 1.0, see git history) treated
# this as a data problem -- pick a less-wrong default for the non-reporting
# nodes -- and it was live-verified to work for *newly* reconciled nodes.
# But reconcile's _ensure_node_vector() preserves any already-persisted
# value over the template default, so pre-fix-era stale 0.0 entries already
# sitting in atlas/circe/prometheus's node_vectors from before that patch
# deployed could never self-correct -- confirmed live 2026-07-22: athena's
# real, fresh transport_bus_reducer report was bus_health=1.0/
# delivery_confidence=1.0, but atlas's persisted entry was a still-0.0
# value from before the fix with node_vector_updated_at=None (never once
# perturbed), and the min()-merge let that meaningless stale reading
# permanently mask athena's real, healthy one -- feeding a false
# "unhealthy"/coherence-degrading signal into every SelfStateV1 coherence
# score (the now-deleted config/self_state/self_state_policy.v1.yaml mapped
# both channels to `coherence`) for as long as field-digester has been
# running this schema and SelfStateV1 existed (module removed 2026-07-22).
#
# The real fix: stop treating this as a 4-way merge at all.
# SINGLE_OBSERVER_NODE_CHANNELS below is consulted by reconcile.py's
# _ensure_node_vector() to (a) never seed these two channels onto any node
# but their owner, and (b) actively prune them from any other node's
# vector every reconcile tick -- self-healing, no manual data migration
# needed. The now-deleted orion/self_state/transport.py's
# transport_channel_hints() already read bus_health/delivery_confidence
# from node:athena directly (never merged across nodes) and was unaffected
# by this bug; this fix brings collect_field_channel_pressures()'s merge in
# line with that same single-observer assumption instead of leaving two
# different mental models of the same two channels live in the codebase at
# once.
#
# "node:athena" is hardcoded, same precedent the deleted
# orion/self_state/transport.py used -- not derived from BUS_OBSERVER_NODE_ID
# (services/
# orion-bus/app/settings.py, default "athena"). If that env var is ever
# changed on a real deployment, a genuinely different node would start
# producing real transport_bus deltas and this map would silently prune
# them every reconcile tick instead of masking a stale value -- a
# resurrection of this same bug class in a new shape. Pre-existing risk
# (not introduced by this fix), flagged here since this is now the second
# place carrying the assumption.
SINGLE_OBSERVER_NODE_CHANNELS: dict[str, str] = {
    "bus_health": "node:athena",
    "delivery_confidence": "node:athena",
}

DEFAULT_CAPABILITY_VECTOR = {ch: 0.0 for ch in CAPABILITY_CHANNELS}
DEFAULT_CAPABILITY_VECTOR["confidence"] = 1.0
DEFAULT_CAPABILITY_VECTOR["available_capacity"] = 1.0
