from __future__ import annotations

import re
from pathlib import Path

from orion.schemas.context_provenance import (
    CONTEXT_PROVENANCE_REGISTRY,
    PLUMBING_KEYS,
    classify,
)

_EXECUTOR_PY = Path(__file__).resolve().parents[3] / "services" / "orion-cortex-exec" / "app" / "executor.py"

# Manually-refreshed snapshot of every key executor.py's ctx dict carried for
# a live chat turn (captured from orion-athena-cortex-exec-chat container
# logs, "Context Keys available: [...]"). This is the primary gate for keys
# that never appear as a literal ctx["key"] = assignment in executor.py
# (e.g. self_state, attention_broadcast -- set via dict merges in helper
# modules). Refresh by re-capturing that log line when the ingress shape
# changes; test_static_ctx_assignments_covered below is the automated
# safety net for the common case this snapshot can't self-update against.
LIVE_CTX_KEY_SNAPSHOT = frozenset(
    {
        "user_message", "metadata", "lane", "user_id", "trigger_source", "trace_id",
        "parent_event_id", "correlation_id", "plan_metadata", "personality_file",
        "trigger_correlation_id", "trigger_trace_id", "recall", "plan_recall_profile",
        "debug", "_cortex_exec_grammar_collector", "_cortex_exec_grammar_request_recorded",
        "options", "verb", "orion_identity_summary", "juniper_relationship_summary",
        "response_policy_summary", "identity_kernel_source", "_run_scope_corr_id",
        "prior_step_results_by_corr", "prior_step_results", "recall_bundle", "memory_digest",
        "recall_fragments", "memory_used", "memory_bundle", "pcr_memory", "continuity_digest",
        "belief_digest", "situation_brief", "presence_context", "temporal_phase",
        "situation_affordances", "situation_prompt_fragment", "world_context_capsule_loaded",
        "capsule_age_hours", "capsule_filtered_reason", "stance_world_context_items_used",
        "politics_context_suppressed", "message_history", "pad_frame", "pad_frame_json",
        "pad_stats", "pad_stats_json", "biometrics", "biometrics_json", "phi_hint",
        "spark_phi_narrative", "embodiment_hint", "spark_embodiment_narrative",
        "spark_state_json", "turn_effect", "turn_effect_json", "turn_effect_evidence",
        "turn_effect_evidence_json", "recent_turn_effect_alerts_json", "system_alert_tags",
        "turn_effect_policy", "turn_effect_policy_json", "turn_effect_explanations",
        "turn_effect_explanations_json", "trigger", "trigger_kind", "context_summary",
        "metacog_biometrics_cue", "metacog_biometrics_cue_enrich", "self_state",
        "execution_trajectory_projection", "transport_bus_projection",
        "active_node_pressure_projection", "attention_broadcast", "episode_summary",
        "curiosity_signals", "metacog_substrate_cue", "substrate_eventfulness_score",
        "substrate_eventfulness_reasons",
    }
)


def test_registry_and_plumbing_have_no_overlap():
    assert not (set(CONTEXT_PROVENANCE_REGISTRY) & PLUMBING_KEYS)


def test_live_snapshot_fully_covered():
    covered = set(CONTEXT_PROVENANCE_REGISTRY) | PLUMBING_KEYS
    missing = LIVE_CTX_KEY_SNAPSHOT - covered
    assert not missing, f"unclassified ctx keys, add to registry or PLUMBING_KEYS: {sorted(missing)}"


def test_static_ctx_assignments_covered():
    """Automated safety net: any key executor.py assigns via a literal
    ctx["key"] = ... or ctx.setdefault("key", ...) must be classified. Doesn't
    catch keys set via dict merges in other modules (that's what
    LIVE_CTX_KEY_SNAPSHOT is for) but needs no hand-maintenance for the
    common case of a new direct assignment in executor.py itself.
    """
    text = _EXECUTOR_PY.read_text()
    assigned_keys = set(re.findall(r'ctx\[["\']([a-zA-Z0-9_]+)["\']\]\s*=', text))
    assigned_keys |= set(re.findall(r'ctx\.setdefault\(["\']([a-zA-Z0-9_]+)["\']', text))
    covered = set(CONTEXT_PROVENANCE_REGISTRY) | PLUMBING_KEYS
    missing = assigned_keys - covered
    assert not missing, f"executor.py assigns unclassified ctx keys: {sorted(missing)}"


def test_classify_returns_none_for_unknown_key():
    assert classify("some_key_nobody_registered") is None


def test_classify_returns_registered_kind():
    assert classify("self_state") == "live_runtime_projection"
    assert classify("recall_bundle") == "memory_recall"
    assert classify("orion_identity_summary") == "static_identity_config"
