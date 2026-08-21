from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4
import hashlib
import json
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from orion.core.bus.bus_schemas import Envelope

# 2026-08-21: a SECOND `class ChatResponseFeedbackV1` used to sit here, with
# `rating`/`feedback_text`/`tags`/`metadata` fields. It was shadowed by the
# real one below -- same name, later definition wins -- so nothing could ever
# construct it and nothing ever did. Removed. Note for anyone verifying this:
# `inspect.getsourcelines()` reported the FIRST definition's line number even
# while the live class was the second, because it matches by name in the
# source; `model_fields` is the ground truth, not the reported line.


CHAT_RESPONSE_FEEDBACK_KIND = "chat.response.feedback.v1"
FeedbackValue = Literal["up", "down"]
MAX_FREE_TEXT_CHARS = 2000

# An artifact ref is `artifact:<kind>:<dispatch_id>` and the dispatch_id half
# is NOT decoration. Review finding: without it the ref is a free-form string
# that nothing validates and nothing can attribute, so a typo silently creates
# an orphan rating and a rating that cannot be attributed to an action teaches
# nothing about any action -- which is the entire purpose of the pipeline it
# feeds. Real dispatch_ids look like
# `dispatch:proposal:prune_stopped_containers:tick_fc7585176059:none:execution_dispatch_policy.v1`
# -- colon-heavy and far too long to retype, which is why scripts/
# rate_artifact.py BUILDS the ref rather than accepting one.
ARTIFACT_REF_PREFIX = "artifact:"
DISPATCH_ID_PREFIX = "dispatch:"


def build_artifact_ref(kind: str, dispatch_id: str) -> str:
    kind = (kind or "").strip()
    dispatch_id = (dispatch_id or "").strip()
    if not kind or ":" in kind:
        raise ValueError(f"artifact kind must be non-empty and colon-free, got {kind!r}")
    if not dispatch_id.startswith(DISPATCH_ID_PREFIX):
        raise ValueError(
            f"dispatch_id must start with {DISPATCH_ID_PREFIX!r}, got {dispatch_id!r}"
        )
    return f"{ARTIFACT_REF_PREFIX}{kind}:{dispatch_id}"


def parse_artifact_ref(ref: str) -> tuple[str, str]:
    """-> (kind, dispatch_id). Raises rather than guessing."""
    if not ref.startswith(ARTIFACT_REF_PREFIX):
        raise ValueError(f"artifact ref must start with {ARTIFACT_REF_PREFIX!r}: {ref!r}")
    remainder = ref[len(ARTIFACT_REF_PREFIX):]
    kind, sep, dispatch_id = remainder.partition(":")
    if not sep or not kind:
        raise ValueError(f"artifact ref must be artifact:<kind>:<dispatch_id>: {ref!r}")
    if not dispatch_id.startswith(DISPATCH_ID_PREFIX):
        raise ValueError(
            f"artifact ref must carry a dispatch id starting {DISPATCH_ID_PREFIX!r}: {ref!r}"
        )
    return kind, dispatch_id

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
    # 2026-08-21: the fourth target, and the first that is not chat-shaped.
    # An Orion-produced artifact -- a self-study journal entry, an affect or
    # co-creation report, a vision description, a pull request. Carries the
    # dispatch_id of the action that produced it, so a rating can be scored
    # back onto that action (orion/autonomy/rating.py). Without it this
    # channel can only ever grade conversation, which is why the only
    # non-biometric outcome Orion could have was unreachable.
    target_artifact_ref: Optional[str] = None
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
        "target_artifact_ref",
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

        if not (
            self.target_turn_id
            or self.target_message_id
            or self.target_correlation_id
            or self.target_artifact_ref
        ):
            raise ValueError("At least one target identifier is required")

        if self.target_artifact_ref:
            # Refuse an unattributable rating outright rather than storing an
            # orphan. parse_artifact_ref raises with the reason.
            parse_artifact_ref(self.target_artifact_ref)
            # ATTESTATION, NOT AUTHENTICATION -- say which, because the
            # difference is the patch's whole premise. Nothing on this host can
            # actually prove a rating came from a person: the Hub route has no
            # auth, and the Hub holds the docker socket, so any container is
            # root-equivalent (resolved 2026-08-14: "detect, do not pretend to
            # prevent"). Requiring user_id and source makes a self-rating
            # DETECTABLE -- an artifact rating with no rater on record is
            # rejected, and one claiming a rater is a claim that can be audited
            # against `source`. It does not make one impossible.
            if not self.user_id:
                raise ValueError(
                    "artifact ratings require user_id: an unattributed rating "
                    "cannot be distinguished from Orion rating itself"
                )

        allowed = THUMBS_UP_CATEGORIES if self.feedback_value == "up" else THUMBS_DOWN_CATEGORIES
        unknown = [item for item in self.categories if item not in allowed]
        if unknown:
            raise ValueError(f"Unknown categories for feedback_value={self.feedback_value}: {unknown}")

        duplicate_categories = {item for item in self.categories if self.categories.count(item) > 1}
        if duplicate_categories:
            raise ValueError(f"Duplicate categories are not allowed: {sorted(duplicate_categories)}")

        if self.free_text and len(self.free_text) > MAX_FREE_TEXT_CHARS:
            raise ValueError(f"free_text exceeds max length {MAX_FREE_TEXT_CHARS}")

        # The artifact component is appended ONLY when an artifact is the
        # target, so a chat-shaped key stays byte-identical to every key
        # already stored. That is not cosmetic: `submission_fingerprint`
        # below is a hash OF this string, and it is what stops the same
        # submission landing twice. Adding a component unconditionally would
        # have re-shaped every chat key from 5 parts to 6, changed its
        # fingerprint, and silently un-deduplicated resubmissions across the
        # deploy boundary.
        key_parts = [
            self.target_turn_id or "",
            self.target_message_id or "",
            self.target_correlation_id or "",
            self.session_id or "",
            self.user_id or "",
        ]
        if self.target_artifact_ref:
            # Prefixed, not raw. The join separator is an unescaped "|", so a
            # raw append lets a chat row whose user_id happens to contain
            # "|artifact:z" produce a byte-identical key -- and therefore a
            # byte-identical submission_fingerprint -- to a genuine artifact
            # rating. Found in review; the previous test compared a trivial
            # pair and asserted the property without testing the case that
            # breaks it.
            key_parts.append(f"artifact={self.target_artifact_ref}")
        self.target_key = self.target_key or "|".join(key_parts)

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
