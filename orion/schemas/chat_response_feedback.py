from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4
import hashlib
import json
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from orion.core.bus.bus_schemas import Envelope

class ChatResponseFeedbackV1(BaseModel):
    """Operator/user feedback event for a chat response."""

    model_config = ConfigDict(extra="forbid")

    feedback_id: str = Field(default_factory=lambda: f"chat-response-feedback-{uuid4()}")
    correlation_id: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    response_id: str | None = None
    rating: str = Field(default="neutral")
    feedback_text: str | None = None
    tags: list[str] = Field(default_factory=list, max_length=16)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


CHAT_RESPONSE_FEEDBACK_KIND = "chat.response.feedback.v1"
FeedbackValue = Literal["up", "down"]
MAX_FREE_TEXT_CHARS = 2000

THUMBS_UP_CATEGORY_LABELS: Dict[str, str] = {
    "helpful_actionable": "Helpful / actionable",
    "well_grounded": "Well grounded",
    "good_recall_continuity": "Good recall / continuity",
    "right_depth": "Right depth",
    "good_tone": "Good tone",
    "strong_implementation_detail": "Strong implementation detail",
    "good_structure_easy_to_use": "Good structure / easy to use",
    "good_judgment": "Good judgment",
    "other": "Other",
}

THUMBS_DOWN_CATEGORY_LABELS: Dict[str, str] = {
    "made_up_facts": "Made up facts",
    "fabricated_recall_memory": "Fabricated recall / memory",
    "missed_relevant_context": "Missed relevant context",
    "lost_conversation_continuity": "Lost conversation continuity",
    "contradicted_earlier_messages": "Contradicted earlier messages",
    "did_not_distinguish_fact_vs_inference": "Didn't distinguish fact vs inference",
    "overconfident_false_certainty": "Overconfident / false certainty",
    "too_surface_level": "Too surface-level",
    "too_abstract": "Too abstract",
    "not_actionable": "Not actionable",
    "incomplete_truncated": "Incomplete / truncated",
    "did_not_answer_directly": "Didn't answer directly",
    "missed_edge_cases": "Missed edge cases",
    "incorrect_tone": "Incorrect tone",
    "too_boilerplate_generic": "Too boilerplate / generic",
    "too_guarded_sanitized": "Too guarded / sanitized",
    "poor_attunement": "Poor attunement",
    "ignored_instructions": "Ignored instructions",
    "asked_unnecessary_follow_up": "Asked unnecessary follow-up",
    "poor_structure_hard_to_scan": "Poor structure / hard to scan",
    "wrong_tool_wrong_routing_wrong_mode": "Wrong tool / routing / mode",
    "should_have_probed_more_about_stated_topics": "Should have probed more",
    "other": "Other",
}

THUMBS_UP_CATEGORIES = set(THUMBS_UP_CATEGORY_LABELS.keys())
THUMBS_DOWN_CATEGORIES = set(THUMBS_DOWN_CATEGORY_LABELS.keys())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_feedback_category_options() -> Dict[str, List[Dict[str, str]]]:
    return {
        "up": [{"value": key, "label": label} for key, label in THUMBS_UP_CATEGORY_LABELS.items()],
        "down": [{"value": key, "label": label} for key, label in THUMBS_DOWN_CATEGORY_LABELS.items()],
    }


class ChatResponseFeedbackV1(BaseModel):
    """Append-only feedback event for a specific assistant response."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    feedback_id: str
    target_turn_id: Optional[str] = None
    target_message_id: Optional[str] = None
    target_correlation_id: Optional[str] = None
    target_key: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    feedback_value: FeedbackValue
    categories: List[str] = Field(default_factory=list, max_length=8)
    free_text: Optional[str] = Field(default=None, max_length=MAX_FREE_TEXT_CHARS)
    source: Optional[str] = None
    ui_context: Optional[Dict[str, Any]] = None
    submission_fingerprint: Optional[str] = None
    created_at: str = Field(default_factory=_now_iso)

    @field_validator(
        "feedback_id",
        "target_turn_id",
        "target_message_id",
        "target_correlation_id",
        "target_key",
        "session_id",
        "user_id",
        "source",
        mode="before",
    )
    @classmethod
    def _strip_blank_strings(cls, value: Any) -> Any:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("categories", mode="before")
    @classmethod
    def _normalize_categories(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("categories must be a list")
        normalized: List[str] = []
        for item in value:
            text = str(item or "").strip()
            if not text:
                continue
            normalized.append(text)
        return normalized

    @field_validator("free_text", mode="before")
    @classmethod
    def _normalize_free_text(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @model_validator(mode="after")
    def _validate_and_derive(self) -> "ChatResponseFeedbackV1":
        if not self.feedback_id:
            raise ValueError("feedback_id is required")

        if not (self.target_turn_id or self.target_message_id or self.target_correlation_id):
            raise ValueError("At least one target identifier is required")

        allowed = THUMBS_UP_CATEGORIES if self.feedback_value == "up" else THUMBS_DOWN_CATEGORIES
        unknown = [item for item in self.categories if item not in allowed]
        if unknown:
            raise ValueError(f"Unknown categories for feedback_value={self.feedback_value}: {unknown}")

        duplicate_categories = {item for item in self.categories if self.categories.count(item) > 1}
        if duplicate_categories:
            raise ValueError(f"Duplicate categories are not allowed: {sorted(duplicate_categories)}")

        if self.free_text and len(self.free_text) > MAX_FREE_TEXT_CHARS:
            raise ValueError(f"free_text exceeds max length {MAX_FREE_TEXT_CHARS}")

        self.target_key = self.target_key or "|".join(
            [
                self.target_turn_id or "",
                self.target_message_id or "",
                self.target_correlation_id or "",
                self.session_id or "",
                self.user_id or "",
            ]
        )

        fingerprint_material = {
            "target_key": self.target_key,
            "feedback_value": self.feedback_value,
            "categories": self.categories,
            "free_text": self.free_text or "",
            "source": self.source or "",
        }
        fingerprint_seed = json.dumps(fingerprint_material, sort_keys=True, separators=(",", ":"))
        self.submission_fingerprint = self.submission_fingerprint or hashlib.sha256(fingerprint_seed.encode("utf-8")).hexdigest()
        return self


class ChatResponseFeedbackEnvelope(Envelope[ChatResponseFeedbackV1]):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    kind: Literal[CHAT_RESPONSE_FEEDBACK_KIND] = Field(CHAT_RESPONSE_FEEDBACK_KIND)
    payload: ChatResponseFeedbackV1
