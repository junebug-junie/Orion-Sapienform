"""Deterministic checks on what the episode distiller proposes (spec 2026-09-30, section 1 "validate").

The distiller decides what is worth remembering. This module never second-guesses that (no word
lists, no "memorability" rules). It only checks things code can check exactly:

* **Grounding.** Every quote must be a substring of the cited turn's FULL, untruncated field
  (prompt or response). Revision 1 of the spec withdrew a false "the compactor invented this" claim
  because a query cut the prompt at 160 characters; quotes are therefore checked against the whole
  text, never a preview.
* **Voice vs source.** ``juniper_said`` needs a verified quote from one of Juniper's prompts;
  ``worked_out_together`` needs one from a prompt and one from a response. A memory whose evidence
  does not support its voice is DOWNGRADED to the voice its evidence does support, and the
  downgrade is logged. It is rejected only when no quote verifies at all.
* **Source monitoring.** Nothing from an internal channel (reverie, curiosity, dream, ...) may be
  labelled as something Juniper said or worked out together; that is rejected, not downgraded.
* **Structure.** Workflow-command turns produce no memories; a statement of five words or fewer,
  or a duplicate statement within the episode, is rejected.
* **Stakes floor.** A high-stakes reason the model gives forces ``stakes=high`` (the model cannot
  lower it). An ``about_juniper`` statement with a content word found in none of its quotes is
  forced high (spec section 3 backstop). Orion's conclusions about its own machinery, about the
  relationship, or asking for direction become ``orion_self_conclusion`` and pending confirmation.

Rejected memories are returned (with their reason) for logging; they are never stored.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from orion.schemas.memory_episode import (
    DistilledMemoryV1,
    DistilledQuestionV1,
    DistillEvidenceV1,
    DistillReferentV1,
    EpisodeDistillationV1,
)

# Stable namespace so a replayed persist mints the same memory_id (idempotent writes).
MEMORY_ID_NAMESPACE = uuid.UUID("6f1c7b8e-2d0a-4c35-9a51-3e7d9b0c4a21")

REFERENT_KINDS = frozenset({"person", "event", "place", "service", "file", "pr", "concept", "project"})
SELF_MACHINERY_KINDS = frozenset({"service", "file", "pr", "concept"})
HIGH_STAKES_REASONS = frozenset(
    {"health", "family", "identity_conclusion_about_juniper", "relationship", "safety_location", "orion_self_conclusion"}
)
INTERNAL_CHANNELS = frozenset({"reverie", "curiosity", "dream", "journal", "topic_model"})
JUNIPER_VOICES = frozenset({"juniper_said", "worked_out_together"})
MIN_STATEMENT_WORDS = 6  # "0 statements of 5 words or fewer" (Stage 1 acceptance 7)

# Purpose -> (start strength, half-life days). Spec section 4.
STRENGTH_BY_PURPOSE: dict[str, tuple[float, Optional[float]]] = {
    "happened": (0.8, 14.0),
    "about_juniper": (0.9, 180.0),
    "orion_view": (0.8, 90.0),
    "follow_up": (1.0, None),
}

_WS = re.compile(r"\s+")
_WORD = re.compile(r"[a-z0-9][a-z0-9'\-]*")
_SLUG_BAD = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class EpisodeTurn:
    """One chat turn of the episode, with its FULL text."""

    label: str            # t1, t2, ... (what the prompt shows the distiller)
    correlation_id: str
    prompt: str
    response: str
    created_at: Optional[datetime] = None
    is_command: bool = False


@dataclass
class VerifiedEvidence:
    source_kind: str      # chat_prompt | chat_response
    source_id: str        # the turn's correlation_id
    quote: str
    verified: bool


@dataclass
class MemoryEvent:
    op: str
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidatedMemory:
    memory_id: str
    purpose: str
    voice: str
    channel: str
    statement: str
    occurred_at: Optional[datetime]
    stakes: str
    stakes_reason: Optional[str]
    confirmation_state: str
    strength: float
    half_life_days: Optional[float]
    due_after: Optional[datetime]
    expires_at: Optional[datetime]
    referents: list[tuple[str, str]]
    evidence: list[VerifiedEvidence]
    events: list[MemoryEvent]
    novel_words: list[str]


@dataclass
class ValidatedQuestion:
    question_id: str
    text: str
    kind: str
    scope: str
    answer_via: str
    referents: list[str]
    evidence: list[VerifiedEvidence]


@dataclass
class Rejection:
    kind: str             # memory | question
    index: int
    reason: str
    candidate: dict[str, Any]


@dataclass
class ValidationResult:
    memories: list[ValidatedMemory]
    questions: list[ValidatedQuestion]
    rejections: list[Rejection]

    @property
    def downgrades(self) -> int:
        return sum(1 for m in self.memories for e in m.events if e.op == "downgraded_voice")


def normalize_ws(text: str) -> str:
    return _WS.sub(" ", str(text or "")).strip()


def _clean_quote(quote: str) -> str:
    q = normalize_ws(quote)
    # Models often wrap a quote in quotation marks it did not copy from the source.
    while len(q) >= 2 and q[0] == q[-1] and q[0] in "\"'`":
        q = q[1:-1].strip()
    for left, right in (("“", "”"), ("‘", "’")):
        if q.startswith(left) and q.endswith(right):
            q = q[1:-1].strip()
    return q


def quote_in_text(quote: str, text: str) -> bool:
    """Exact substring of the FULL field, after collapsing whitespace runs on both sides."""
    q = _clean_quote(quote)
    return bool(q) and q in normalize_ws(text)


def normalize_referent_key(raw: str) -> Optional[str]:
    """``kind:slug`` with a known kind, or None. "Service: orion durable runs" -> "service:orion-durable-runs"."""
    if ":" not in str(raw or ""):
        return None
    kind, _, rest = str(raw).partition(":")
    kind = kind.strip().lower()
    slug = _SLUG_BAD.sub("-", rest.strip().lower()).strip("-")
    if kind not in REFERENT_KINDS or not slug:
        return None
    return f"{kind}:{slug}"


def _parse_dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _content_words(text: str) -> set[str]:
    from orion.memory.intake_junk import _STOPWORDS  # reused as-is (frozen; never extended here)

    return {w for w in _WORD.findall(str(text or "").lower()) if len(w) >= 3 and w not in _STOPWORDS}


def novel_content_words(statement: str, quotes: Iterable[str], referents: Iterable[DistillReferentV1]) -> list[str]:
    """Content words of the statement found in none of its quotes (nor its own referent names)."""
    allowed: set[str] = set()
    for q in quotes:
        allowed |= _content_words(q)
    for r in referents:
        allowed |= _content_words(r.key.replace(":", " ").replace("-", " "))
        for alias in r.aliases:
            allowed |= _content_words(alias)
    return sorted(_content_words(statement) - allowed)


def memory_id_for(episode_id: str, purpose: str, statement: str) -> str:
    return str(uuid.uuid5(MEMORY_ID_NAMESPACE, f"{episode_id}|{purpose}|{normalize_ws(statement).lower()}"))


def _verify(evidence: list[DistillEvidenceV1], by_label: dict[str, EpisodeTurn]) -> list[tuple[VerifiedEvidence, Optional[EpisodeTurn], str]]:
    out = []
    for ev in evidence:
        turn = by_label.get(str(ev.turn).strip())
        text = "" if turn is None else (turn.prompt if ev.field == "prompt" else turn.response)
        verified = turn is not None and quote_in_text(ev.quote, text)
        out.append(
            (
                VerifiedEvidence(
                    source_kind=f"chat_{ev.field}",
                    source_id=turn.correlation_id if turn else f"unknown:{ev.turn}",
                    quote=_clean_quote(ev.quote),
                    verified=verified,
                ),
                turn,
                ev.field,
            )
        )
    return out


def _supported_voice(voice: str, has_prompt: bool, has_response: bool) -> str:
    """The strongest voice the verified evidence supports, starting from the claimed one."""
    if voice == "worked_out_together":
        if has_prompt and has_response:
            return voice
        return "juniper_said" if has_prompt else "orion_thought"
    if voice == "juniper_said":
        return voice if has_prompt else "orion_thought"
    if voice in ("orion_read", "orion_self_knowledge"):
        # A chat episode has no reading or graphify source to cite; what Orion said in the turn is
        # its own thought, what Juniper said is hers.
        return "orion_thought" if has_response else "juniper_said"
    return voice  # orion_thought: any verified quote supports it


def validate_distillation(
    distillation: EpisodeDistillationV1,
    turns: list[EpisodeTurn],
    *,
    episode_id: str,
) -> ValidationResult:
    by_label = {t.label: t for t in turns}
    memories: list[ValidatedMemory] = []
    questions: list[ValidatedQuestion] = []
    rejections: list[Rejection] = []
    seen_statements: set[str] = set()

    for idx, cand in enumerate(distillation.memories):
        raw = cand.model_dump(mode="json")
        statement = normalize_ws(cand.statement)
        if len(statement.split()) < MIN_STATEMENT_WORDS:
            rejections.append(Rejection("memory", idx, "statement_too_short", raw))
            continue
        norm = statement.lower()
        if norm in seen_statements:
            rejections.append(Rejection("memory", idx, "duplicate_statement", raw))
            continue
        if cand.channel in INTERNAL_CHANNELS and cand.voice in JUNIPER_VOICES:
            rejections.append(Rejection("memory", idx, "internal_channel_labelled_as_juniper", raw))
            continue
        checked = _verify(cand.evidence, by_label)
        verified = [(ev, turn, fld) for ev, turn, fld in checked if ev.verified]
        if not verified:
            rejections.append(Rejection("memory", idx, "no_verified_quote", raw))
            continue
        if all(turn is not None and turn.is_command for _, turn, _ in verified):
            rejections.append(Rejection("memory", idx, "command_turn_only", raw))
            continue
        events: list[MemoryEvent] = []
        unverified = [ev for ev, _, _ in checked if not ev.verified]
        if unverified:
            events.append(
                MemoryEvent("evidence_dropped", "quote_not_in_full_turn_text", {"quotes": [e.quote for e in unverified]})
            )
        has_prompt = any(fld == "prompt" for _, _, fld in verified)
        has_response = any(fld == "response" for _, _, fld in verified)
        voice = _supported_voice(cand.voice, has_prompt, has_response)
        if voice != cand.voice:
            events.append(
                MemoryEvent("downgraded_voice", f"{cand.voice}_not_supported_by_evidence", {"from": cand.voice, "to": voice})
            )
        channel = cand.channel
        if voice in JUNIPER_VOICES and channel != "chat":
            channel = "chat"  # her words only ever arrive through chat in a chat episode

        referents: list[tuple[str, str]] = []
        for r in cand.referents:
            key = normalize_referent_key(r.key)
            if key is None:
                events.append(MemoryEvent("referent_dropped", "not_kind_slug", {"key": r.key}))
                continue
            if (key, r.role) not in referents:
                referents.append((key, str(r.role or "about")))

        stakes = cand.stakes
        stakes_reason = cand.stakes_reason
        if stakes_reason in HIGH_STAKES_REASONS:
            stakes = "high"
        novel: list[str] = []
        if cand.purpose == "about_juniper":
            novel = novel_content_words(statement, [ev.quote for ev, _, _ in verified], cand.referents)
            if novel:
                stakes = "high"
                stakes_reason = stakes_reason or "identity_conclusion_about_juniper"
                events.append(MemoryEvent("stakes_raised", "novel_content_words", {"words": novel}))
        if cand.purpose == "orion_view":
            kinds = {k.split(":", 1)[0] for k, _ in referents}
            if kinds & SELF_MACHINERY_KINDS or cand.asks_direction or stakes_reason == "relationship":
                stakes, stakes_reason = "high", "orion_self_conclusion"
        confirmation_state = "pending_confirmation" if stakes == "high" else "auto"
        strength, half_life = STRENGTH_BY_PURPOSE[cand.purpose]

        due_after = _parse_dt(cand.due_after) if cand.purpose == "follow_up" else None
        expires_at = _parse_dt(cand.expires_at) if cand.purpose == "follow_up" else None

        seen_statements.add(norm)
        memories.append(
            ValidatedMemory(
                memory_id=memory_id_for(episode_id, cand.purpose, statement),
                purpose=cand.purpose,
                voice=voice,
                channel=channel,
                statement=statement,
                occurred_at=_parse_dt(cand.occurred_at),
                stakes=stakes,
                stakes_reason=stakes_reason,
                confirmation_state=confirmation_state,
                strength=strength,
                half_life_days=half_life,
                due_after=due_after,
                expires_at=expires_at,
                referents=referents,
                evidence=[ev for ev, _, _ in checked],
                events=events,
                novel_words=novel,
            )
        )

    for idx, q in enumerate(distillation.questions):
        raw = q.model_dump(mode="json")
        text = normalize_ws(q.text)
        checked = _verify(q.evidence, by_label)
        if not text or not any(ev.verified for ev, _, _ in checked):
            rejections.append(Rejection("question", idx, "no_verified_quote" if text else "empty_question", raw))
            continue
        keys = [k for k in (normalize_referent_key(r.key) for r in q.referents) if k]
        questions.append(
            ValidatedQuestion(
                question_id=str(uuid.uuid5(MEMORY_ID_NAMESPACE, f"{episode_id}|question|{text.lower()}")),
                text=text,
                kind=q.kind,
                scope=q.scope,
                answer_via=q.answer_via,
                referents=keys,
                evidence=[ev for ev, _, _ in checked],
            )
        )

    return ValidationResult(memories=memories, questions=questions, rejections=rejections)


def coverage(result: ValidationResult, turns: list[EpisodeTurn]) -> dict[str, Any]:
    """Stage 1 acceptance 6: share of non-command turns cited by some kept memory."""
    content = [t for t in turns if not t.is_command]
    cited = {ev.source_id for m in result.memories for ev in m.evidence if ev.verified}
    hit = [t for t in content if t.correlation_id in cited]
    return {
        "content_turns": len(content),
        "cited_turns": len(hit),
        "coverage": round(len(hit) / len(content), 3) if content else None,
    }
