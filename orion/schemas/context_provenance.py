"""Provenance classification for chat-turn context (``ctx``) keys.

``executor.py`` assembles ~80 keys into a single ``ctx`` dict over the course
of a turn — some are live substrate/biometric projections computed this tick,
some are memory retrieved from a recall backend, some are static identity
config, some are the user's own message. Nothing previously distinguished
these at the point Orion has to talk about them, which is how a GitHub file
fetch got narrated as live substrate computation "this turn" (see
docs/superpowers/pr-reports for the incident). This module is the registry
that makes that distinction inspectable and enforceable.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

ContextSourceKind = Literal[
    "live_runtime_projection",
    "derived_summary",
    "memory_recall",
    "static_identity_config",
    "user_input",
    "external_tool_fetch",
]


class ContextKeyProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    source_kind: ContextSourceKind
    description: str


def _entry(key: str, source_kind: ContextSourceKind, description: str) -> ContextKeyProvenance:
    return ContextKeyProvenance(key=key, source_kind=source_kind, description=description)


# Content-bearing ctx keys: anything Orion could plausibly narrate or draw a
# claim from. Every key here has a real producer somewhere in the substrate/
# cortex-exec pipeline (see docstrings on the source functions) — this
# registry only classifies, it does not compute.
CONTEXT_PROVENANCE_REGISTRY: dict[str, ContextKeyProvenance] = {
    entry.key: entry
    for entry in [
        # --- static identity / config: loaded once, stable across turns ---
        _entry("orion_identity_summary", "static_identity_config", "Identity kernel summary lines."),
        _entry("juniper_relationship_summary", "static_identity_config", "Relationship summary lines."),
        _entry("response_policy_summary", "static_identity_config", "Response policy summary lines."),
        _entry("identity_kernel_source", "static_identity_config", "Which identity kernel file was loaded."),
        _entry("personality_file", "static_identity_config", "Declared personality file for this turn."),
        # --- memory recall: retrieved from a persisted recall/PCR backend ---
        _entry("recall_bundle", "memory_recall", "Raw recall backend results."),
        _entry("recall_fragments", "memory_recall", "Selected recall fragments for the turn."),
        _entry("memory_used", "memory_recall", "Flag/summary of which memory was actually used."),
        _entry("memory_bundle", "memory_recall", "Bundled memory retrieval result."),
        _entry("pcr_memory", "memory_recall", "PCR (persistent chat recall) memory payload."),
        _entry("memory_digest", "memory_recall", "Compact digest of recalled memory."),
        _entry("continuity_digest", "memory_recall", "Digest of continuity-relevant recalled memory."),
        # --- derived summary: deterministic reduction of other context, not itself live ---
        _entry("belief_digest", "derived_summary", "Reduction of the unified relational belief graph."),
        _entry("situation_brief", "derived_summary", "Situation summary derived from ctx."),
        _entry("presence_context", "derived_summary", "Presence/availability framing derived from ctx."),
        _entry("temporal_phase", "derived_summary", "Time-of-day/temporal phase label."),
        _entry("situation_affordances", "derived_summary", "Derived list of situational affordances."),
        _entry("situation_prompt_fragment", "derived_summary", "Rendered situation prompt fragment."),
        _entry("world_context_capsule_loaded", "derived_summary", "Whether a world-context capsule was loaded."),
        _entry("capsule_age_hours", "derived_summary", "Age of the loaded world-context capsule."),
        _entry("capsule_filtered_reason", "derived_summary", "Why the world-context capsule was filtered."),
        _entry("stance_world_context_items_used", "derived_summary", "Count of world-context items folded into stance."),
        _entry("politics_context_suppressed", "derived_summary", "Whether politics-sensitive context was suppressed."),
        _entry("trigger", "derived_summary", "What triggered this turn (raw label)."),
        _entry("trigger_kind", "derived_summary", "Classified kind of turn trigger."),
        _entry("context_summary", "derived_summary", "General context summary derived for the turn."),
        _entry("episode_summary", "derived_summary", "Summary of the current episode."),
        _entry("chat_stance_brief", "derived_summary", "Parsed task_mode/conversation_frame stance brief."),
        _entry("prior_chat_stance_brief", "derived_summary", "Previous turn's stance brief."),
        _entry("prior_stance", "derived_summary", "Previous turn's full stance."),
        _entry("collapse_entry", "derived_summary", "Collapse-mirror entry for this turn."),
        _entry("collapse_json", "derived_summary", "JSON-rendered collapse-mirror entry."),
        _entry("final_entry", "derived_summary", "Final assembled turn entry."),
        _entry("speech_contract", "derived_summary", "Rendered per-turn speech contract."),
        _entry("world_context_capsule", "derived_summary", "Loaded world-context capsule content."),
        _entry("journal_pageindex_context", "memory_recall", "Journal page-index retrieval context."),
        # --- live runtime projection: computed this tick by a substrate/biometrics/turn-effect engine ---
        _entry("pad_frame", "live_runtime_projection", "Live pleasure-arousal-dominance affect frame."),
        _entry("pad_frame_json", "live_runtime_projection", "JSON-rendered PAD frame."),
        _entry("pad_stats", "live_runtime_projection", "Live PAD statistics."),
        _entry("pad_stats_json", "live_runtime_projection", "JSON-rendered PAD statistics."),
        _entry("biometrics", "live_runtime_projection", "Live biometrics substrate reading."),
        _entry("biometrics_json", "live_runtime_projection", "JSON-rendered biometrics reading."),
        _entry("phi_hint", "live_runtime_projection", "Live phi/tissue-viz hint."),
        _entry("spark_phi_narrative", "live_runtime_projection", "Narrative rendering of live spark/phi state."),
        _entry("embodiment_hint", "live_runtime_projection", "Live embodiment hint."),
        _entry("spark_embodiment_narrative", "live_runtime_projection", "Narrative rendering of live embodiment state."),
        _entry("spark_state_json", "live_runtime_projection", "JSON-rendered spark engine state."),
        _entry("turn_effect", "live_runtime_projection", "Live turn-effect engine output."),
        _entry("turn_effect_json", "live_runtime_projection", "JSON-rendered turn-effect output."),
        _entry("turn_effect_evidence", "live_runtime_projection", "Evidence backing the turn-effect output."),
        _entry("turn_effect_evidence_json", "live_runtime_projection", "JSON-rendered turn-effect evidence."),
        _entry("recent_turn_effect_alerts_json", "live_runtime_projection", "Recent turn-effect alerts."),
        _entry("system_alert_tags", "live_runtime_projection", "Live system alert tags."),
        _entry("turn_effect_policy", "live_runtime_projection", "Policy derived from live turn-effect state."),
        _entry("turn_effect_policy_json", "live_runtime_projection", "JSON-rendered turn-effect policy."),
        _entry("turn_effect_explanations", "live_runtime_projection", "Explanations for the live turn-effect output."),
        _entry("turn_effect_explanations_json", "live_runtime_projection", "JSON-rendered turn-effect explanations."),
        _entry("metacog_biometrics_cue", "live_runtime_projection", "Compact live biometrics cue for metacog prompts."),
        _entry("metacog_biometrics_cue_enrich", "live_runtime_projection", "Enriched live biometrics cue."),
        _entry("self_state", "live_runtime_projection", "Live self-state projection (SelfStateV1)."),
        _entry("execution_trajectory_projection", "live_runtime_projection", "Live execution-trajectory projection."),
        _entry("transport_bus_projection", "live_runtime_projection", "Live transport-bus projection."),
        _entry("active_node_pressure_projection", "live_runtime_projection", "Live substrate pressure-field projection."),
        _entry("attention_broadcast", "live_runtime_projection", "Live substrate attention-broadcast projection."),
        _entry("curiosity_signals", "live_runtime_projection", "Live curiosity/drive signals."),
        _entry("metacog_substrate_cue", "live_runtime_projection", "Compact live substrate cue for metacog prompts."),
        _entry("substrate_eventfulness_score", "live_runtime_projection", "Live substrate eventfulness score."),
        _entry("substrate_eventfulness_reasons", "live_runtime_projection", "Reasons behind the live eventfulness score."),
        # --- user input: this turn's message and directly-derived transcript ---
        _entry("user_message", "user_input", "The user's message this turn."),
        _entry("message_history", "user_input", "Conversation transcript."),
    ]
}

# Pure request/routing plumbing: correlation ids, trace ids, flags, and other
# bookkeeping that carries no cognitive content and that Orion would never
# legitimately narrate as perceived/recalled/computed signal. Exempted from
# the registry-coverage requirement rather than force-classified into one of
# the six meaningful kinds above.
PLUMBING_KEYS: frozenset[str] = frozenset(
    {
        "metadata",
        "lane",
        "user_id",
        "trigger_source",
        "trace_id",
        "parent_event_id",
        "correlation_id",
        "plan_metadata",
        "trigger_correlation_id",
        "trigger_trace_id",
        "recall",
        "plan_recall_profile",
        "debug",
        "_cortex_exec_grammar_collector",
        "_cortex_exec_grammar_request_recorded",
        "options",
        "verb",
        "_run_scope_corr_id",
        "prior_step_results_by_corr",
        "prior_step_results",
        "prior_step_results_by_service",
        # debug/telemetry about the prompt-construction process itself, not content:
        "chat_stance_debug",
        "metacog_ctx_trim_applied",
        "metacog_draft_prompt_chars",
        "metacog_draft_section_sizes",
        "metacog_enrich_prompt_chars",
        "metacog_enrich_section_sizes",
        "metacog_entry_id",
    }
)


def classify(key: str) -> ContextSourceKind | None:
    """Look up a ctx key's provenance. None means unclassified (neither a
    registered content key nor a known plumbing key) — callers should treat
    that as "needs a registry entry," not as a silent default."""
    entry = CONTEXT_PROVENANCE_REGISTRY.get(key)
    return entry.source_kind if entry else None
