from __future__ import annotations

from typing import Any, Dict


def extract_autonomy_payload(cortex_result: Any) -> Dict[str, Any]:
    metadata = getattr(cortex_result, "metadata", None)
    if not isinstance(metadata, dict):
        return {}
    payload: Dict[str, Any] = {}
    for key in (
        "autonomy_summary",
        "autonomy_debug",
        "autonomy_state_preview",
        "autonomy_backend",
        "autonomy_selected_subject",
        "autonomy_repository_status",
        "chat_stance_debug",
        "turn_effect",
        "turn_effect_evidence",
        "turn_effect_status",
        "turn_effect_missing_reason",
    ):
        value = metadata.get(key)
        if value is not None:
            payload[key] = value
    return payload
