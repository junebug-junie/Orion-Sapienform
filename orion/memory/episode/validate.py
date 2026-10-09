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
* **Names must come from Juniper or the quotes.** A statement that names a known person, place,
  project or service Juniper never said in the episode, and that none of its own quotes contain,
  is kept but escalated to high stakes ("ungrounded_name") so Juniper is asked. A memory in
  Juniper's voice is grounded only by her own prompts; Orion's reply grounds only Orion's voice. Live case
  2026-10-06: "...lives in Ogden, Utah, not Chicago" took "Chicago" from Orion's own reply. The
  names come from referent keys (data), never from a word list.
* **End dates on any purpose need Juniper's words.** A non-follow_up memory keeps ``expires_at``
  only when ``until_quote`` is found in one of her prompts and the date is not before the episode.
* **Stakes are the distiller's own judgment.** Juniper decided the rubric on 2026-10-06 (see
  memory_episode_distill.j2). This module never reads the statement to judge stakes; it only checks
  that ``stakes`` and ``stakes_reason`` are present and agree with each other and with
  ``asks_direction``. A pair that does not agree is resolved toward high (ask Juniper), never toward
  low, and the change is logged. High stakes means pending confirmation.

Rejected memories are returned (with their reason) for logging; they are never stored.
"""

from __future__ import annotations

import re
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Optional, get_args
from zoneinfo import ZoneInfo

from orion.schemas.memory_episode import (
    HIGH_STAKES_REASONS,
    StakesReason,
    DistilledMemoryV1,
    DistilledQuestionV1,
    DistillEvidenceV1,
    EpisodeDistillationV1,
)

# Juniper's timezone: the prompt asks for datetimes "in her timezone", so a value the model
# writes without an offset is read there, never as UTC (six hours early in MDT).
DEFAULT_TZ = "America/Denver"

# Stable namespace so a replayed persist mints the same memory_id (idempotent writes).
MEMORY_ID_NAMESPACE = uuid.UUID("6f1c7b8e-2d0a-4c35-9a51-3e7d9b0c4a21")

REFERENT_KINDS = frozenset({"person", "event", "place", "service", "file", "pr", "concept", "project"})
INTERNAL_CHANNELS = frozenset({"reverie", "curiosity", "dream", "journal", "topic_model"})
JUNIPER_VOICES = frozenset({"juniper_said", "worked_out_together"})
VALID_STAKES_REASONS = frozenset(get_args(StakesReason))
MIN_STATEMENT_WORDS = 6  # "0 statements of 5 words or fewer" (Stage 1 acceptance 7)

# The two people in every chat episode: naming them is never an import from outside it.
PARTICIPANT_REFERENTS = frozenset({"person:juniper", "person:orion"})
# Kinds whose slug is a name someone says ("ogden", "hecate"). Event and concept slugs are
# descriptions the distiller mints ("austin-offsite"), not words anyone said, so they are skipped.
NAMED_REFERENT_KINDS = frozenset({"person", "place", "project", "service"})
MIN_NAME_CHARS = 3
# Stored as stakes_reason when a statement names something Juniper never said (validator label,
# like "unjudged"; the confirmation card explains it).
UNGROUNDED_NAME_STAKES_LABEL = "ungrounded_name"

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
    # The writer's own names for each referent key, verbatim, each with the distiller's
    # alias_kind (proper_name | descriptor). The referent store only checks they are
    # Juniper's words (alias_grounding_v1); it never classifies them by vocabulary.
    referent_aliases: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    # The distiller's alias_kind for each key's OWN name ("person:my-cousin" -> descriptor).
    referent_name_kinds: dict[str, str] = field(default_factory=dict)


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
    return dt if dt.tzinfo else dt.replace(tzinfo=ZoneInfo(DEFAULT_TZ))


def _contains_word(folded_text: str, folded_name: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(folded_name) + r"(?!\w)", folded_text) is not None


def ungrounded_names(statement: str, *, known_keys: Iterable[str], grounding_texts: Iterable[str]) -> list[str]:
    """Known referent keys whose name the statement uses but no grounding text contains.

    ``grounding_texts`` are Juniper's own prompts in the episode plus the memory's verified quotes.
    Names come from the keys' slugs ("place:the-wade" -> "the wade"), matched as whole words after
    the same folding the quote check uses. Participants and minted event/concept slugs are skipped.
    """
    grounds = [fold_text(t) for t in grounding_texts if t]
    return [
        key for key, name in _named_in(statement, known_keys)
        if not any(_contains_word(g, name) for g in grounds)
    ]


def referent_name(key: str) -> str:
    """The name a referent key's slug spells, folded ("place:the-wade" -> "the wade")."""
    return fold_text(key.partition(":")[2].replace("-", " "))


def _named_in(text: str, known_keys: Iterable[str]) -> list[tuple[str, str]]:
    folded = fold_text(text)
    out: list[tuple[str, str]] = []
    for key in sorted(set(known_keys)):
        kind = key.partition(":")[0]
        if key in PARTICIPANT_REFERENTS or kind not in NAMED_REFERENT_KINDS:
            continue
        name = referent_name(key)
        if len(name) >= MIN_NAME_CHARS and _contains_word(folded, name):
            out.append((key, name))
    return out


def named_referents(text: str, known_keys: Iterable[str]) -> list[str]:
    """Known person/place/project/service keys whose name appears in ``text`` as a whole word.
    The same matching ``ungrounded_names`` uses; the situation graph cues recall with it."""
    return [key for key, _ in _named_in(text, known_keys)]


def _until_quote_problem(quote: Optional[str], turns: list[EpisodeTurn]) -> Optional[str]:
    """None when ``quote`` is Juniper's own words from one of her prompts, else the drop reason."""
    if not quote:
        return "no_until_quote"
    if not quote_long_enough(quote):
        return "until_quote_too_short"
    if not any(not t.is_command and quote_in_text(quote, t.prompt) for t in turns):
        return "until_quote_not_in_juniper_prompt"
    return None


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


# Stored as stakes_reason when the validator escalates a memory the distiller did not categorize.
# A validator label, not a category the distiller may use: the row says WHY it is high.
UNJUDGED_STAKES_LABEL = "unjudged"
# The first distiller prompt that asks for a stakes category on every memory (v3, 2026-10-06).
STAKES_CATEGORY_REQUIRED_FROM = 3
_VERSION_NUM = re.compile(r"\.v(\d+)$")


def stakes_category_required(prompt_version: Optional[str]) -> bool:
    """True when the prompt that produced the answer asked for a category. Unknown -> True (strict)."""
    m = _VERSION_NUM.search(str(prompt_version or ""))
    return m is None or int(m.group(1)) >= STAKES_CATEGORY_REQUIRED_FROM


def resolve_stakes(
    stakes: str, stakes_reason: Optional[str], asks_direction: bool = False, *, category_required: bool = True
) -> tuple[str, Optional[str], Optional[MemoryEvent]]:
    """(stakes, stakes_reason, event or None). Presence and consistency only, never vocabulary.

    Consistent pairs pass unchanged: ``high`` with a high-stakes category, or ``low`` with "none"
    (and ``asks_direction`` false). Otherwise the memory only ever moves toward high, because high
    means "confirm with Juniper first" and a wrong low is stored as settled fact. Every move is
    logged, and an escalation without a category is stored with the label "unjudged":

    * low + a high-stakes category      -> high with that category (the category is the judgment)
    * asks_direction + low              -> high, "orion_asks_direction" (Juniper's rule)
    * high + "none" / missing / unknown -> high, "unjudged"
    * low + missing / unknown           -> high, "unjudged" (the distiller did not judge it)

    ``category_required=False`` is for answers to a prompt that never asked for a category (v1/v2,
    e.g. a checkpoint answered before v3 deployed): a missing category is then not a defect, so the
    distiller's own low/high stands (logged as ``stakes_uncategorized``), and the two escalations
    above that rest on a real signal (a high category, asks_direction) still apply.
    """
    reason = str(stakes_reason).strip().lower() if stakes_reason is not None else None
    known = reason in VALID_STAKES_REASONS
    detail = {"stakes": stakes, "stakes_reason": stakes_reason, "asks_direction": asks_direction}
    if stakes == "high":
        if known and reason in HIGH_STAKES_REASONS:
            return "high", reason, None
        if asks_direction:
            return "high", "orion_asks_direction", MemoryEvent("stakes_reason_set", "asks_direction", detail)
        if not category_required:
            return "high", None, MemoryEvent("stakes_uncategorized", "prompt_did_not_ask_for_category", detail)
        return "high", UNJUDGED_STAKES_LABEL, MemoryEvent("stakes_reason_missing", "high_without_category", detail)
    # stakes == "low"
    if known and reason in HIGH_STAKES_REASONS:
        return "high", reason, MemoryEvent("stakes_raised", "category_is_high_stakes", detail)
    if asks_direction:
        return "high", "orion_asks_direction", MemoryEvent("stakes_raised", "asks_direction", detail)
    if reason == "none":
        return "low", "none", None
    if not category_required:
        return "low", None, MemoryEvent("stakes_uncategorized", "prompt_did_not_ask_for_category", detail)
    return "high", UNJUDGED_STAKES_LABEL, MemoryEvent("stakes_raised", "stakes_reason_missing_or_unknown", detail)


def validate_distillation(
    distillation: EpisodeDistillationV1,
    turns: list[EpisodeTurn],
    *,
    episode_id: str,
    prompt_version: Optional[str] = None,
    known_referents: Iterable[str] = (),
) -> ValidationResult:
    """``prompt_version`` = the version of the template the answer was generated from (the durable
    graph stamps it at render time). None means the current prompt. ``known_referents`` = referent
    keys already in use (the prompt's candidate list); the keys this answer emits are added to it."""
    by_label = {t.label: t for t in turns}
    known_keys = {k for k in (normalize_referent_key(x) for x in known_referents) if k}
    for c in list(distillation.memories) + list(distillation.questions):
        known_keys.update(k for k in (normalize_referent_key(r.key) for r in c.referents) if k)
    juniper_prompts = [t.prompt for t in turns if not t.is_command]
    episode_start = min((t.created_at for t in turns if t.created_at), default=None)
    category_required = stakes_category_required(prompt_version)
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
        referent_aliases: dict[str, list[tuple[str, str]]] = {}
        referent_name_kinds: dict[str, str] = {}
        for r in cand.referents:
            key = normalize_referent_key(r.key)
            if key is None:
                events.append(MemoryEvent("referent_dropped", "not_kind_slug", {"key": r.key}))
                continue
            if (key, r.role) not in referents:
                referents.append((key, str(r.role or "about")))
            referent_name_kinds.setdefault(key, r.alias_kind)
            kept = referent_aliases.setdefault(key, [])
            for alias in r.aliases or []:
                text = normalize_ws(alias.text)
                if text and text not in {t for t, _k in kept}:
                    kept.append((text, alias.alias_kind))

        stakes, stakes_reason, stakes_event = resolve_stakes(
            cand.stakes, cand.stakes_reason, cand.asks_direction, category_required=category_required
        )
        if stakes_event is not None:
            events.append(stakes_event)
        # Source monitoring for names: a memory in Juniper's voice is grounded only by what she
        # wrote. Orion's own reply never grounds her claim (the live case quoted Orion's "Chicago").
        # A memory in Orion's voice may also name what Orion said in their quoted reply.
        own_reply_quotes = [] if voice in JUNIPER_VOICES else [ev.quote for ev, _, fld in verified if fld == "response"]
        imported = ungrounded_names(
            statement, known_keys=known_keys, grounding_texts=juniper_prompts + own_reply_quotes,
        )
        if imported:
            events.append(MemoryEvent("ungrounded_name", "statement_names_what_juniper_did_not_say",
                                      {"referents": imported, "stakes_before": stakes}))
            if stakes == "low":
                stakes, stakes_reason = "high", UNGROUNDED_NAME_STAKES_LABEL
        confirmation_state = "pending_confirmation" if stakes == "high" else "auto"
        strength, half_life = STRENGTH_BY_PURPOSE[cand.purpose]

        due_after = _parse_dt(cand.due_after) if cand.purpose == "follow_up" else None
        expires_at = _parse_dt(cand.expires_at)
        if cand.purpose != "follow_up" and expires_at is not None:
            # An end date on a fact must rest on Juniper's own words for that period.
            drop = _until_quote_problem(cand.until_quote, turns)
            if drop is None and episode_start is not None and expires_at <= episode_start:
                drop = "ends_before_episode"
            if drop:
                events.append(MemoryEvent("validity_dropped", drop,
                                          {"expires_at": cand.expires_at, "until_quote": cand.until_quote}))
                expires_at = None

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
                referent_aliases={k: v for k, v in referent_aliases.items() if v},
                referent_name_kinds=referent_name_kinds,
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
    """Stage 1 acceptance 6: share of Juniper's non-command turns cited by some kept memory.
    Turns Orion wrote on their own (no prompt) are context, not Juniper turns to cover."""
    content = [t for t in turns if not t.is_command and t.prompt.strip()]
    cited = {ev.source_id for m in result.memories for ev in m.evidence if ev.verified}
    hit = [t for t in content if t.correlation_id in cited]
    return {
        "content_turns": len(content),
        "cited_turns": len(hit),
        "coverage": round(len(hit) / len(content), 3) if content else None,
    }
