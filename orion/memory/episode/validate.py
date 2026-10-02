"""Deterministic checks on what the episode distiller proposes (spec 2026-09-30, section 1 "validate").

The distiller decides what is worth remembering. This module never second-guesses that (no word
lists, no "memorability" rules). It only checks things code can check exactly:

* **Grounding.** Every quote must be found in the cited turn's FULL, untruncated field (prompt or
  response), after folding typography on both sides (NFKC, curly quotes, dashes, case,
  whitespace), and must be at least 3 words (15 characters for scripts without spaces). Revision 1
  of the spec withdrew a false claim because a query cut a prompt at 160 characters.
* **Voice vs source, one direction only.** A memory may lose Juniper's voice when its evidence does
  not support it (worked_out_together -> juniper_said -> orion_thought), and the change is logged.
  A voice that is not Juniper's is NEVER moved into hers; orion_read / orion_self_knowledge become
  orion_thought (Orion's own reply quoted) or are rejected.
* **Source monitoring on the final voice and channel.** Juniper's voice only arrives through chat:
  a Juniper voice on an internal channel (reverie, curiosity, dream, ...) is rejected. The channel
  is never rewritten.
* **Structure.** Workflow-command turns produce no memories; a statement of five words or fewer,
  or a duplicate statement within the episode, is rejected.
* **Stakes are the distiller's own field.** The stakes policy is Juniper's open decision (review of
  2026-10-02 removed a word-list backstop); high stakes simply means pending confirmation.

Rejected memories are returned (with their reason) for logging; they are never stored.
"""

from __future__ import annotations

import re
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from orion.schemas.memory_episode import (
    DistilledMemoryV1,
    DistilledQuestionV1,
    DistillEvidenceV1,
    EpisodeDistillationV1,
)

# Stable namespace so a replayed persist mints the same memory_id (idempotent writes).
MEMORY_ID_NAMESPACE = uuid.UUID("6f1c7b8e-2d0a-4c35-9a51-3e7d9b0c4a21")

REFERENT_KINDS = frozenset({"person", "event", "place", "service", "file", "pr", "concept", "project"})
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


# Character folds applied to BOTH the quote and the source text before matching. Typography the
# model or the keyboard may change without changing the words: curly vs straight quotes, dash
# variants, non-breaking spaces, full-width forms (NFKC), and case.
_FOLD = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"', "\u2033": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2015": "-", "\u2212": "-",
})
MIN_QUOTE_WORDS = 3
MIN_QUOTE_CHARS_UNSPACED = 15  # scripts written without spaces (CJK, Thai, ...)


def fold_text(text: str) -> str:
    return normalize_ws(unicodedata.normalize("NFKC", str(text or "")).translate(_FOLD)).casefold()


def _clean_quote(quote: str) -> str:
    q = normalize_ws(quote)
    # Models often wrap a quote in quotation marks it did not copy from the source.
    for left, right in (("\u201c", "\u201d"), ("\u2018", "\u2019")):
        if q.startswith(left) and q.endswith(right):
            q = q[1:-1].strip()
    while len(q) >= 2 and q[0] == q[-1] and q[0] in "\"'`":
        q = q[1:-1].strip()
    return q


def quote_long_enough(quote: str) -> bool:
    """At least 3 words, or 15 characters of a script written without spaces. A one-letter quote
    ("I") is a substring of almost anything and proves nothing."""
    q = _clean_quote(quote)
    if len(q.split()) >= MIN_QUOTE_WORDS:
        return True
    unspaced = any(ord(c) >= 0x2E80 for c in q)
    return unspaced and len(q.replace(" ", "")) >= MIN_QUOTE_CHARS_UNSPACED


def quote_in_text(quote: str, text: str) -> bool:
    """Long-enough quote found in the FULL field after folding both sides (see fold_text)."""
    q = _clean_quote(quote)
    return bool(q) and quote_long_enough(q) and fold_text(q) in fold_text(text)


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


def _supported_voice(voice: str, has_prompt: bool, has_response: bool) -> Optional[str]:
    """The voice the verified evidence supports, moving only AWAY from Juniper's voice.

    Juniper's voices can lose attribution (worked_out_together -> juniper_said -> orion_thought);
    a voice that is not hers is never moved into hers. orion_read / orion_self_knowledge cannot be
    supported by a chat episode (no reading or graphify source): they become orion_thought when
    Orion's own reply is quoted, and None (reject) otherwise.
    """
    if voice == "worked_out_together":
        if has_prompt and has_response:
            return voice
        return "juniper_said" if has_prompt else "orion_thought"
    if voice == "juniper_said":
        return voice if has_prompt else "orion_thought"
    if voice in ("orion_read", "orion_self_knowledge"):
        return "orion_thought" if has_response else None
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
        if cand.voice in JUNIPER_VOICES and cand.channel != "chat":
            # The CLAIM itself confuses an internal thought with Juniper's words; its statement is
            # written that way too, so it is rejected, not relabelled.
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
        if voice is None:
            rejections.append(Rejection("memory", idx, "voice_unsupported_by_evidence", raw))
            continue
        channel = cand.channel  # never rewritten: an internal channel stays internal
        # Source monitoring on the FINAL voice and channel: Juniper's voice only arrives through chat.
        if voice in JUNIPER_VOICES and channel != "chat":
            rejections.append(Rejection("memory", idx, "internal_channel_labelled_as_juniper", raw))
            continue
        if voice != cand.voice:
            events.append(
                MemoryEvent("downgraded_voice", f"{cand.voice}_not_supported_by_evidence", {"from": cand.voice, "to": voice})
            )

        referents: list[tuple[str, str]] = []
        for r in cand.referents:
            key = normalize_referent_key(r.key)
            if key is None:
                events.append(MemoryEvent("referent_dropped", "not_kind_slug", {"key": r.key}))
                continue
            if (key, r.role) not in referents:
                referents.append((key, str(r.role or "about")))

        # Stakes are the distiller's own field. The stakes policy (floor, backstops, which
        # self-conclusions to ask about) is Juniper's open decision; the validator only checks
        # evidence, voice and channel.
        stakes = cand.stakes
        stakes_reason = cand.stakes_reason
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
