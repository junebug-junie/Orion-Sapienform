"""Chat stance brief → chat_stance OrionSignalV1 (Milestone B3)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from orion.schemas.chat_stance import ChatStanceBrief
from orion.signals.adapters.base import OrionSignalAdapter
from orion.signals.models import OrionOrganRegistryEntry, OrionSignalV1
from orion.signals.normalization import NormalizationContext, clamp01
from orion.signals.registry import ORGAN_REGISTRY
from orion.signals.signal_ids import make_signal_id

_TASK_MODE_COHERENCE = {
    "direct_response": 0.85,
    "triage": 0.55,
    "technical_collaboration": 0.75,
    "identity_dialogue": 0.7,
    "reflective_dialogue": 0.72,
    "playful_exchange": 0.65,
    "mixed": 0.5,
}

_SALIENCE_VALENCE = {"low": 0.35, "medium": 0.55, "high": 0.75}

_FORBIDDEN_TEXT_KEYS = frozenset(
    {
        "stance_summary",
        "user_intent",
        "juniper_relevance",
        "self_relevance",
        "answer_strategy",
        "temporal_context",
        "audience_context",
        "environmental_context",
        "operational_context",
    }
)


def _extract_brief_dict(payload: dict) -> dict[str, Any] | None:
    for key in ("chat_stance_brief", "ChatStanceBrief"):
        raw = payload.get(key)
        if isinstance(raw, dict):
            return raw
    meta = payload.get("metadata")
    if isinstance(meta, dict):
        raw = meta.get("chat_stance_brief")
        if isinstance(raw, dict):
            return raw
    debug = payload.get("chat_stance_debug")
    if isinstance(debug, dict):
        contract = debug.get("final_prompt_contract") or {}
        if isinstance(contract, dict):
            raw = contract.get("chat_stance_brief")
            if isinstance(raw, dict):
                return raw
    return None


def _dimensions_from_brief(brief: ChatStanceBrief | dict[str, Any]) -> tuple[dict[str, float], float, list[str]]:
    notes: list[str] = []
    if isinstance(brief, ChatStanceBrief):
        task_mode = brief.task_mode
        salience = brief.identity_salience
        confidence = 0.9
    else:
        task_mode = str(brief.get("task_mode") or "mixed")
        salience = str(brief.get("identity_salience") or "medium")
        confidence = 0.55
        notes.append("partial chat_stance payload; schema validation skipped")

    coherence = _TASK_MODE_COHERENCE.get(task_mode, 0.5)
    valence = _SALIENCE_VALENCE.get(salience, 0.55)
    return (
        {"coherence": coherence, "valence": valence, "confidence": confidence},
        confidence,
        notes,
    )


class ChatStanceAdapter(OrionSignalAdapter):
    organ_id = "chat_stance"

    def can_handle(self, channel: str, payload: dict) -> bool:
        if _extract_brief_dict(payload) is not None:
            return True
        if "chat_stance" in channel:
            return True
        if "cognition:trace" in channel and payload.get("metadata", {}).get("chat_stance_debug_present"):
            return True
        if payload.get("chat_stance_debug"):
            return True
        return False

    def adapt(
        self,
        channel: str,
        payload: dict,
        registry: Dict[str, OrionOrganRegistryEntry],
        prior_signals: Dict[str, OrionSignalV1],
        norm_ctx: NormalizationContext,
    ) -> Optional[OrionSignalV1]:
        entry = registry.get(self.organ_id) or ORGAN_REGISTRY.get(self.organ_id)
        if entry is None:
            return None

        now = datetime.now(timezone.utc)
        brief_raw = _extract_brief_dict(payload)
        notes: list[str] = []

        if brief_raw is None:
            meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            if meta.get("chat_stance_debug_present"):
                dimensions = {"coherence": 0.5, "valence": 0.5, "confidence": 0.4}
                notes.append("stance_debug_only_no_brief")
            else:
                return None
        else:
            brief: ChatStanceBrief | dict[str, Any]
            try:
                brief = ChatStanceBrief.model_validate(brief_raw)
            except Exception:
                brief = brief_raw
            dimensions, _, notes = _dimensions_from_brief(brief)

        corr = (
            payload.get("correlation_id")
            or payload.get("_envelope_correlation_id")
            or (payload.get("metadata") or {}).get("root_correlation_id")
        )
        src_id = str(corr) if corr else None
        if src_id is None:
            verb = str(payload.get("verb") or payload.get("verb_name") or "chat")
            src_id = f"{verb}:{int(now.timestamp())}"
            notes.append("synthetic_correlation_id")

        causal_parents = [
            prior_signals[p].signal_id
            for p in (entry.causal_parent_organs or [])
            if p in prior_signals
        ]

        task_mode = ""
        if brief_raw and isinstance(brief_raw, dict):
            task_mode = str(brief_raw.get("task_mode") or "")
        summary = f"chat_stance task_mode={task_mode or 'unknown'}" if task_mode else "chat_stance"

        return OrionSignalV1(
            signal_id=make_signal_id(self.organ_id, src_id),
            organ_id=self.organ_id,
            organ_class=entry.organ_class,
            signal_kind="chat_stance",
            dimensions={k: clamp01(float(v)) for k, v in dimensions.items()},
            causal_parents=causal_parents,
            source_event_id=src_id,
            observed_at=now,
            emitted_at=now,
            summary=summary,
            notes=notes[:5],
        )
