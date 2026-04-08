from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


SocialGifDecisionKind = Literal["no_gif", "text_only", "text_plus_gif", "gif_only_disabled"]
SocialGifIntentKind = Literal[
    "celebrate",
    "laugh_with",
    "sympathetic_reaction",
    "dramatic_agreement",
    "soft_facepalm",
    "playful_confusion",
    "victory_lap",
]
SocialGifReactionClass = Literal[
    "celebrate",
    "laugh_with",
    "amused",
    "sympathetic",
    "disbelief",
    "frustration",
    "confusion",
    "dramatic_agreement",
    "soft_facepalm",
    "playful_confusion",
    "unknown",
]
SocialGifConfidenceLevel = Literal["none", "low", "medium"]
SocialGifAmbiguityLevel = Literal["low", "medium", "high"]
SocialGifCueDisposition = Literal["used", "softened", "ignored"]


class SocialGifObservedSignalV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    signal_id: str = Field(default_factory=lambda: f"social-gif-observed-{uuid4()}")
    platform: str
    room_id: str
    thread_key: Optional[str] = None
    sender_id: Optional[str] = None
    sender_name: Optional[str] = None
    media_present: bool = False
    provider: Optional[str] = None
    transport_source: Optional[str] = None
    provider_title: str = ""
    alt_text: str = ""
    query_text: str = ""
    tags: List[str] = Field(default_factory=list)
    filename: str = ""
    caption_text: str = ""
    surrounding_text: str = ""
    created_at: str = Field(default_factory=_utcnow_iso)
    rationale: str = ""
    reasons: List[str] = Field(default_factory=list)
    metadata: Dict[str, str] = Field(default_factory=dict)


class SocialGifProxyContextV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    context_id: str = Field(default_factory=lambda: f"social-gif-proxy-{uuid4()}")
    platform: str
    room_id: str
    thread_key: Optional[str] = None
    sender_id: Optional[str] = None
    sender_name: Optional[str] = None
    media_present: bool = False
    provider: Optional[str] = None
    transport_source: Optional[str] = None
    provider_title: str = ""
    alt_text: str = ""
    query_text: str = ""
    tags: List[str] = Field(default_factory=list)
    filename: str = ""
    caption_text: str = ""
    surrounding_text: str = ""
    thread_summary: str = ""
    reply_target_name: str = ""
    proxy_inputs_present: List[str] = Field(default_factory=list)
    proxy_text_fragments: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_utcnow_iso)
    rationale: str = ""
    reasons: List[str] = Field(default_factory=list)
    metadata: Dict[str, str] = Field(default_factory=dict)


class SocialGifInterpretationV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    interpretation_id: str = Field(default_factory=lambda: f"social-gif-interpretation-{uuid4()}")
    platform: str
    room_id: str
    thread_key: Optional[str] = None
    sender_id: Optional[str] = None
    sender_name: Optional[str] = None
    media_present: bool = False
    reaction_class: SocialGifReactionClass = "unknown"
    confidence_level: SocialGifConfidenceLevel = "none"
    ambiguity_level: SocialGifAmbiguityLevel = "high"
    cue_disposition: SocialGifCueDisposition = "ignored"
    rationale: str = ""
    reasons: List[str] = Field(default_factory=list)
    observed_signal_id: Optional[str] = None
    proxy_context_id: Optional[str] = None
    created_at: str = Field(default_factory=_utcnow_iso)
    metadata: Dict[str, str] = Field(default_factory=dict)


class SocialGifIntentV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    intent_id: str = Field(default_factory=lambda: f"social-gif-intent-{uuid4()}")
    platform: str
    room_id: str
    thread_key: Optional[str] = None
    intent_kind: SocialGifIntentKind
    gif_query: str
    provider_hint: str = "provider_neutral_reaction_gif"
    audience_scope: str = "peer"
    target_participant_id: Optional[str] = None
    target_participant_name: Optional[str] = None
    rationale: str = ""
    reasons: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_utcnow_iso)
    metadata: Dict[str, str] = Field(default_factory=dict)


class SocialGifUsageStateV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    usage_state_id: str = Field(default_factory=lambda: f"social-gif-usage-{uuid4()}")
    platform: str
    room_id: str
    thread_key: Optional[str] = None
    consecutive_gif_turns: int = Field(default=0, ge=0)
    turns_since_last_orion_gif: int = Field(default=999, ge=0)
    recent_gif_density: float = Field(default=0.0, ge=0.0, le=1.0)
    recent_gif_turn_count: int = Field(default=0, ge=0)
    recent_turn_window_size: int = Field(default=10, ge=1)
    orion_turn_count: int = Field(default=0, ge=0)
    recent_turn_was_gif: List[bool] = Field(default_factory=list)
    recent_intent_kinds: List[SocialGifIntentKind] = Field(default_factory=list)
    recent_target_participant_ids: List[str] = Field(default_factory=list)
    recent_target_participant_names: List[str] = Field(default_factory=list)
    last_intent_kind: Optional[SocialGifIntentKind] = None
    last_target_participant_id: Optional[str] = None
    last_target_participant_name: Optional[str] = None
    last_gif_at: Optional[str] = None
    updated_at: str = Field(default_factory=_utcnow_iso)
    metadata: Dict[str, str] = Field(default_factory=dict)


class SocialGifPolicyDecisionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    policy_id: str = Field(default_factory=lambda: f"social-gif-policy-{uuid4()}")
    platform: str
    room_id: str
    thread_key: Optional[str] = None
    gif_allowed: bool = False
    decision_kind: SocialGifDecisionKind = "text_only"
    intent_kind: Optional[SocialGifIntentKind] = None
    selected_intent: Optional[SocialGifIntentV1] = None
    rationale: str = ""
    reasons: List[str] = Field(default_factory=list)
    cooldown_active: bool = False
    consecutive_gif_turns: int = Field(default=0, ge=0)
    turns_since_last_orion_gif: int = Field(default=999, ge=0)
    recent_gif_density: float = Field(default=0.0, ge=0.0, le=1.0)
    audience_scope: str = "peer"
    target_participant_id: Optional[str] = None
    target_participant_name: Optional[str] = None
    created_at: str = Field(default_factory=_utcnow_iso)
    metadata: Dict[str, str] = Field(default_factory=dict)
