from __future__ import annotations

import logging
from typing import Any, Dict, Mapping

logger = logging.getLogger(__name__)


def log_autonomy_payload_extraction(
    *,
    correlation_id: str,
    cortex_result: Any,
    payload: Dict[str, Any],
    source: str,
) -> None:
    """Diagnostic logging after autonomy extraction on Hub chat ingress paths."""
    metadata = getattr(cortex_result, "metadata", None)
    md_keys = sorted(metadata.keys()) if isinstance(metadata, Mapping) else []
    corr = str(correlation_id or "").strip() or "-"
    if not payload:
        logger.info(
            "hub_autonomy_extract_empty corr=%s source=%s cortex_metadata_keys=%s",
            corr,
            source,
            md_keys,
        )
        return
    summary = payload.get("autonomy_summary")
    debug = payload.get("autonomy_debug")
    logger.info(
        "hub_autonomy_extract_present corr=%s source=%s payload_keys=%s selected_subject=%s backend=%s summary_present=%s debug_present=%s state_preview_present=%s",
        corr,
        source,
        sorted(payload.keys()),
        payload.get("autonomy_selected_subject"),
        payload.get("autonomy_backend"),
        isinstance(summary, dict) and bool(summary),
        isinstance(debug, dict) and bool(debug),
        isinstance(payload.get("autonomy_state_preview"), dict) and bool(payload.get("autonomy_state_preview")),
    )


def extract_autonomy_payload(cortex_result: Any) -> Dict[str, Any]:
    metadata = getattr(cortex_result, "metadata", None)
    if not isinstance(metadata, dict):
        return {}
    payload: Dict[str, Any] = {}
    for key in (
        "autonomy_summary",
        "autonomy_debug",
        "autonomy_state_preview",
        "autonomy_execution_mode",
        "autonomy_goal_lineage",
        "autonomy_backend",
        "autonomy_selected_subject",
        "autonomy_repository_status",
        "mutation_cognition_context",
        "runtime_response_diagnostics",
        "chat_stance_debug",
        "turn_effect",
        "turn_effect_evidence",
        "turn_effect_status",
        "turn_effect_missing_reason",
        "situation_brief",
        "situation_prompt_fragment",
        "presence_context",
        "temporal_phase",
        "situation_affordances",
        "autonomy_state_v2_preview",
        "autonomy_state_delta",
    ):
        value = metadata.get(key)
        if value is not None:
            payload[key] = value
    return payload
