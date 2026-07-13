from __future__ import annotations

from typing import Any

from orion.journaler.schemas import JournalEntryWriteV1, JournalTriggerV1
from orion.schemas.chat_stance import ChatStanceBrief


_LIST_FIELDS = (
    "active_identity_facets",
    "active_growth_axes",
    "active_relationship_facets",
    "social_posture",
    "reflective_themes",
    "active_tensions",
    "dream_motifs",
    "response_hazards",
)


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _list_of_str(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    cleaned = [str(v).strip() for v in value if str(v).strip()]
    return cleaned or None


def build_journal_entry_index_payload(
    write: JournalEntryWriteV1,
    *,
    trigger: JournalTriggerV1 | None = None,
    chat_stance: ChatStanceBrief | None = None,
    stance_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a denormalized journal retrieval payload.

    This stays journal-specific and degrades gracefully when trigger/stance
    metadata is not present at write time.
    """

    stance_metadata = stance_metadata or {}
    payload: dict[str, Any] = {
        "entry_id": write.entry_id,
        "created_at": write.created_at,
        "author": write.author,
        "mode": write.mode,
        "title": write.title,
        "body": write.body,
        "source_kind": write.source_kind,
        "source_ref": write.source_ref,
        "correlation_id": write.correlation_id,
        "trigger_kind": (trigger.trigger_kind if trigger else None) or getattr(write, "trigger_kind", None),
        "trigger_summary": trigger.summary if trigger else None,
        "conversation_frame": None,
        "task_mode": None,
        "identity_salience": None,
        "answer_strategy": None,
        "stance_summary": None,
        "active_identity_facets": None,
        "active_growth_axes": None,
        "active_relationship_facets": None,
        "social_posture": None,
        "reflective_themes": None,
        "active_tensions": None,
        "dream_motifs": None,
        "response_hazards": None,
    }

    source = chat_stance.model_dump(mode="json") if chat_stance is not None else {}
    for field in ("conversation_frame", "task_mode", "identity_salience", "answer_strategy", "stance_summary"):
        payload[field] = _as_optional_str(source.get(field) if field in source else stance_metadata.get(field))

    for field in _LIST_FIELDS:
        if field in source:
            payload[field] = _list_of_str(source.get(field))
        else:
            payload[field] = _list_of_str(stance_metadata.get(field))

    unc = stance_metadata.get("llm_uncertainty")
    if not isinstance(unc, dict):
        unc = None
    payload["llm_uncertainty"] = unc
    payload["llm_mean_logprob"] = unc.get("mean_logprob") if isinstance(unc, dict) else None
    payload["llm_mean_top1_margin"] = unc.get("mean_top1_margin") if isinstance(unc, dict) else None
    payload["llm_unstable_span_count"] = unc.get("unstable_span_count") if isinstance(unc, dict) else None

    return payload
