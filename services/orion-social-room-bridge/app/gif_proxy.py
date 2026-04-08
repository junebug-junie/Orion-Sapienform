from __future__ import annotations

import os
import re
from typing import Any

from orion.schemas.social_autonomy import SocialTurnPolicyDecisionV1
from orion.schemas.social_bridge import CallSyneRoomMessageV1
from orion.schemas.social_gif import (
    SocialGifConfidenceLevel,
    SocialGifInterpretationV1,
    SocialGifObservedSignalV1,
    SocialGifProxyContextV1,
    SocialGifReactionClass,
)


_BLOCKED_RE = re.compile(r"\b(private|sealed|blocked|secret|off[- ]record|journal|mirror)\b", re.IGNORECASE)
_URL_ONLY_RE = re.compile(r"^(https?://|www\.)", re.IGNORECASE)
_SEPARATOR_RE = re.compile(r"[\s,_|/\\\-]+")
_NON_WORD_RE = re.compile(r"[^a-z0-9\s]+")
_TITLE_KEYS = ("gif_title", "media_title", "title")
_ALT_KEYS = ("gif_alt_text", "media_alt_text", "alt_text")
_QUERY_KEYS = ("gif_query", "media_query", "search_query", "query")
_TAG_KEYS = ("gif_tags", "media_tags", "tags")
_FILENAME_KEYS = ("gif_filename", "media_filename", "filename")
_CAPTION_KEYS = ("gif_caption", "media_caption", "caption")
_PROVIDER_KEYS = ("gif_provider", "media_provider", "provider")

_REACTION_PATTERNS: dict[SocialGifReactionClass, tuple[re.Pattern[str], ...]] = {
    "celebrate": (
        re.compile(r"\b(celebrat|congrats|congratulations|victory|success|win|winner|nailed it|let'?s go|finally|confetti|party)\b", re.IGNORECASE),
    ),
    "laugh_with": (
        re.compile(r"\b(lol|lmao|haha|hehe|rofl|laughing with you|cracking up)\b", re.IGNORECASE),
    ),
    "amused": (
        re.compile(r"\b(amused|smirk|chuckle|giggle|funny|that got me)\b", re.IGNORECASE),
    ),
    "sympathetic": (
        re.compile(r"\b(sorry|sympathy|sympathetic|comfort|hug|there there|rough|oof|condolence)\b", re.IGNORECASE),
    ),
    "disbelief": (
        re.compile(r"\b(no way|cannot believe|can'?t believe|disbelief|stunned|are you serious|unbelievable)\b", re.IGNORECASE),
    ),
    "frustration": (
        re.compile(r"\b(frustrat|annoyed|irritated|angry|ugh|furious|fed up)\b", re.IGNORECASE),
    ),
    "confusion": (
        re.compile(r"\b(confus|wait what|what\?|huh|bewildered|unsure)\b", re.IGNORECASE),
    ),
    "dramatic_agreement": (
        re.compile(r"\b(exactly|same|totally|yes|absolutely|preach|this)\b", re.IGNORECASE),
    ),
    "soft_facepalm": (
        re.compile(r"\b(facepalm|eye roll|eyeroll|welp|yikes|cringe|whoops|awkward)\b", re.IGNORECASE),
    ),
    "playful_confusion": (
        re.compile(r"\b(wait what lol|chaos|what even|confused but in a funny way|playful confusion)\b", re.IGNORECASE),
    ),
    "unknown": (),
}
_SOURCE_WEIGHTS = {
    "provider_title": 2.0,
    "query_text": 2.2,
    "alt_text": 2.0,
    "caption_text": 1.7,
    "tags": 1.2,
    "filename": 0.6,
    "surrounding_text": 1.0,
    "thread_summary": 0.4,
}


def _boolish(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _candidate_value(message: CallSyneRoomMessageV1, *keys: str) -> Any:
    metadata = dict(message.metadata or {})
    raw_payload = dict(message.raw_payload or {})
    media_hint = dict(metadata.get("media_hint") or raw_payload.get("media_hint") or {})
    for key in keys:
        for container in (metadata, raw_payload, media_hint):
            value = container.get(key)
            if value is not None and value != "":
                return value
    return None


def _normalize_fragment(value: Any, *, allow_url_like: bool = False) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if _BLOCKED_RE.search(text):
        return ""
    if not allow_url_like and _URL_ONLY_RE.match(text):
        return ""
    return text[:180]


def _normalize_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = re.split(r"[,;|]", str(value or ""))
    tags = []
    for item in raw_items:
        text = _normalize_fragment(item)
        if text:
            tags.append(text[:48])
    return tags[:8]


def _normalize_filename(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    base = os.path.basename(text)
    if _URL_ONLY_RE.match(base):
        return ""
    stem, _ = os.path.splitext(base)
    safe = _normalize_fragment(stem)
    return safe[:64]


def extract_social_gif_observed_signal(message: CallSyneRoomMessageV1) -> SocialGifObservedSignalV1 | None:
    metadata = dict(message.metadata or {})
    raw_payload = dict(message.raw_payload or {})
    media_hint = dict(metadata.get("media_hint") or raw_payload.get("media_hint") or {})
    media_present = any(
        (
            _boolish(metadata.get("peer_used_gif")),
            _boolish(metadata.get("contains_gif")),
            _boolish(raw_payload.get("peer_used_gif")),
            _boolish(raw_payload.get("contains_gif")),
            str(media_hint.get("kind") or "").strip().lower() == "gif",
            any(_candidate_value(message, key) for key in _TITLE_KEYS + _ALT_KEYS + _QUERY_KEYS + _TAG_KEYS + _FILENAME_KEYS + _CAPTION_KEYS),
        )
    )
    if not media_present:
        return None

    provider = _normalize_fragment(_candidate_value(message, *_PROVIDER_KEYS))
    transport_source = _normalize_fragment(metadata.get("transport") or raw_payload.get("transport"))
    reasons = ["gif_media_present"]
    signal = SocialGifObservedSignalV1(
        platform=message.platform,
        room_id=message.room_id,
        thread_key=message.thread_id,
        sender_id=message.sender_id,
        sender_name=message.sender_name,
        media_present=True,
        provider=provider or None,
        transport_source=transport_source or None,
        provider_title=_normalize_fragment(_candidate_value(message, *_TITLE_KEYS)),
        alt_text=_normalize_fragment(_candidate_value(message, *_ALT_KEYS)),
        query_text=_normalize_fragment(_candidate_value(message, *_QUERY_KEYS)),
        tags=_normalize_tags(_candidate_value(message, *_TAG_KEYS)),
        filename=_normalize_filename(_candidate_value(message, *_FILENAME_KEYS)),
        caption_text=_normalize_fragment(_candidate_value(message, *_CAPTION_KEYS)),
        surrounding_text=_normalize_fragment(message.text),
        rationale="Observed GIF proxy inputs come only from transport metadata and adjacent text, not from visual inspection.",
        reasons=reasons,
        metadata={
            "media_hint_present": "true" if bool(media_hint) else "false",
            "raw_payload_has_media": "true" if bool(raw_payload.get("media_hint")) else "false",
        },
    )
    return signal


def build_social_gif_proxy_context(
    *,
    message: CallSyneRoomMessageV1,
    social_memory: dict[str, Any],
    observed_signal: SocialGifObservedSignalV1,
) -> SocialGifProxyContextV1:
    room = dict(social_memory.get("room") or {})
    fragments: list[str] = []
    present: list[str] = []

    def _push(label: str, value: str) -> None:
        if value:
            present.append(label)
            fragments.append(value)

    _push("title", observed_signal.provider_title)
    _push("query", observed_signal.query_text)
    _push("alt_text", observed_signal.alt_text)
    if observed_signal.tags:
        present.append("tags")
        fragments.extend(observed_signal.tags[:4])
    _push("filename", observed_signal.filename)
    _push("caption", observed_signal.caption_text)
    _push("surrounding_text", observed_signal.surrounding_text)
    thread_summary = _normalize_fragment(room.get("current_thread_summary"))
    _push("thread_summary", thread_summary)

    return SocialGifProxyContextV1(
        platform=observed_signal.platform,
        room_id=observed_signal.room_id,
        thread_key=observed_signal.thread_key,
        sender_id=observed_signal.sender_id,
        sender_name=observed_signal.sender_name,
        media_present=observed_signal.media_present,
        provider=observed_signal.provider,
        transport_source=observed_signal.transport_source,
        provider_title=observed_signal.provider_title,
        alt_text=observed_signal.alt_text,
        query_text=observed_signal.query_text,
        tags=list(observed_signal.tags),
        filename=observed_signal.filename,
        caption_text=observed_signal.caption_text,
        surrounding_text=observed_signal.surrounding_text,
        thread_summary=thread_summary,
        reply_target_name=str(message.target_participant_name or message.reply_to_sender_id or "").strip(),
        proxy_inputs_present=present,
        proxy_text_fragments=fragments[:8],
        rationale="Proxy context keeps only safe text hints around the GIF so Orion can treat them as a soft conversational cue.",
        reasons=["non_visual_gif_proxy", f"input_count={len(present)}"],
        metadata={
            "weak_proxy": "true" if len(present) <= 1 else "false",
        },
    )


def _score_reaction(proxy: SocialGifProxyContextV1) -> tuple[SocialGifReactionClass, float, float, list[str], bool]:
    scores: dict[SocialGifReactionClass, float] = {key: 0.0 for key in _REACTION_PATTERNS}
    reasons: list[str] = []
    only_filename = True
    for label, value in (
        ("provider_title", proxy.provider_title),
        ("query_text", proxy.query_text),
        ("alt_text", proxy.alt_text),
        ("caption_text", proxy.caption_text),
        ("filename", proxy.filename),
        ("surrounding_text", proxy.surrounding_text),
        ("thread_summary", proxy.thread_summary),
    ):
        text = str(value or "").strip()
        if not text:
            continue
        if label != "filename":
            only_filename = False
        weight = _SOURCE_WEIGHTS[label]
        for reaction_class, patterns in _REACTION_PATTERNS.items():
            if reaction_class == "unknown":
                continue
            if any(pattern.search(text) for pattern in patterns):
                scores[reaction_class] += weight
                reasons.append(f"{label}->{reaction_class}")
    if proxy.tags:
        only_filename = False
        for tag in proxy.tags:
            for reaction_class, patterns in _REACTION_PATTERNS.items():
                if reaction_class == "unknown":
                    continue
                if any(pattern.search(tag) for pattern in patterns):
                    scores[reaction_class] += _SOURCE_WEIGHTS["tags"]
                    reasons.append(f"tags->{reaction_class}")

    ranked = sorted(((score, key) for key, score in scores.items() if key != "unknown"), reverse=True)
    if not ranked or ranked[0][0] < 1.0:
        return "unknown", 0.0, 0.0, reasons, only_filename
    best_score, best_class = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else 0.0
    return best_class, best_score, second_score, reasons, only_filename


def _confidence_and_ambiguity(
    *,
    reaction_class: SocialGifReactionClass,
    best_score: float,
    second_score: float,
    proxy: SocialGifProxyContextV1,
    only_filename: bool,
) -> tuple[SocialGifConfidenceLevel, str]:
    if reaction_class == "unknown":
        return "low" if proxy.media_present else "none", "high"
    if only_filename:
        return "low", "high"
    if best_score >= 3.4 and (best_score - second_score) >= 1.0 and len(proxy.proxy_inputs_present) >= 2:
        return "medium", "low"
    if best_score >= 2.0 and (best_score - second_score) >= 0.5:
        return "low", "medium"
    return "low", "high"


def interpret_social_gif_proxy(
    *,
    message: CallSyneRoomMessageV1,
    turn_policy: SocialTurnPolicyDecisionV1,
    social_memory: dict[str, Any],
    observed_signal: SocialGifObservedSignalV1,
    proxy_context: SocialGifProxyContextV1,
) -> SocialGifInterpretationV1:
    reaction_class, best_score, second_score, reasons, only_filename = _score_reaction(proxy_context)
    confidence_level, ambiguity_level = _confidence_and_ambiguity(
        reaction_class=reaction_class,
        best_score=best_score,
        second_score=second_score,
        proxy=proxy_context,
        only_filename=only_filename,
    )
    room = dict(social_memory.get("room") or {})
    contested = bool(room.get("claim_divergence_signals")) or bool(turn_policy.epistemic_signal)
    repair_active = bool(turn_policy.repair_signal or turn_policy.repair_decision)
    clarification_active = turn_policy.decision == "ask_follow_up" or bool(room.get("clarifying_question"))
    ambiguity_high = bool(turn_policy.thread_routing and turn_policy.thread_routing.ambiguity_level == "high")

    cue_disposition = "used"
    disposition_reasons: list[str] = []
    if repair_active or contested:
        cue_disposition = "ignored"
        disposition_reasons.append("stronger_live_cues_override_gif_proxy")
    elif clarification_active or ambiguity_high or confidence_level != "medium":
        cue_disposition = "softened"
        disposition_reasons.append("gif_proxy_stays_soft_context")

    if reaction_class == "unknown":
        cue_disposition = "softened" if observed_signal.media_present else "ignored"
        disposition_reasons.append("meaning_unclear_from_metadata")

    confidence = confidence_level
    if cue_disposition == "ignored" and confidence_level == "medium":
        confidence = "low"

    return SocialGifInterpretationV1(
        platform=observed_signal.platform,
        room_id=observed_signal.room_id,
        thread_key=observed_signal.thread_key,
        sender_id=observed_signal.sender_id,
        sender_name=observed_signal.sender_name,
        media_present=observed_signal.media_present,
        reaction_class=reaction_class,
        confidence_level=confidence,
        ambiguity_level=ambiguity_level,  # type: ignore[arg-type]
        cue_disposition=cue_disposition,  # type: ignore[arg-type]
        rationale=(
            "Use GIF metadata and nearby text only as a soft cue; Orion cannot literally see the GIF and should stay uncertain when the proxy is weak."
        ),
        reasons=list(dict.fromkeys(reasons + disposition_reasons)),
        observed_signal_id=observed_signal.signal_id,
        proxy_context_id=proxy_context.context_id,
        metadata={
            "best_score": f"{best_score:.2f}",
            "second_score": f"{second_score:.2f}",
            "only_filename": "true" if only_filename else "false",
        },
    )
