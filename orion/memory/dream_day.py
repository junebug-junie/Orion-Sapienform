"""The day dream's deterministic half: parse the turn's answer, check every ref, build the memories, grade.

Design: docs/superpowers/specs/2026-11-17-nightly-day-dream-design.md, Patch 2. Contract:
orion/schemas/dream_day.py. Caller: services/orion-durable-runs/app/dream_day_graph.py (``persist``). Pure:
the one DB read (resolving refs) is the caller's, through ``RESOLVE_SQL``.

Rules, all code:

* **Parse.** The reply's LAST fenced ```json block (else its last top-level ``{...}``) is the answer. No
  answer at all is a failed turn attempt (the caller raises). A malformed item is dropped and counted; it
  never sinks the whole answer.
* **Every ref must resolve.** An item (memory, plan item, the day paragraph) with a ref of an unsupported
  kind, or one that does not resolve, is dropped and counted. A memory and a ``new`` concern need at least
  one ref; a ``keep``/``drop`` needs a reason and a held ``concern_id``, and its refs (if any) must resolve.
* **At most 5 memories** (``DREAM_DAY_MAX_MEMORIES``) and **at most 1 new concern**: the rest are counted.
* **Memories go through orion.memory.episode.validate** (statement length, duplicates, voice/channel,
  stakes, names Juniper never said), with each resolved ref's own text as its evidence. Voice is
  ``orion_thought`` (Orion's own), channel ``dream``. Ids are deterministic from (day, index), so a replayed
  persist writes nothing twice.
* **Grading.** ``produced`` only when at least one memory was written; otherwise "no lasting trace".
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Iterable, Mapping, Optional
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from orion.memory.episode.validate import (
    MEMORY_ID_NAMESPACE, EpisodeTurn, ValidationResult, VerifiedEvidence, validate_distillation,
)
from orion.schemas.dream_day import (
    DREAM_DAY_MAX_MEMORIES, DREAM_DAY_MAX_NEW_CONCERNS, DREAM_DAY_PROMPT_VERSION, DREAM_DAY_PURPOSES,
    DREAM_DAY_TEXT_MAX, DreamDayAnswerV1, DreamDayMemoryV1, DreamDayParagraphV1, DreamDayPlanItemV1, split_ref,
)
from orion.schemas.memory_episode import DistilledMemoryV1, DistillEvidenceV1, EpisodeDistillationV1

ZONE = ZoneInfo("America/Denver")
NO_TRACE_NOTE = "no lasting trace"
# How much of a resolved record's text is kept as a memory's evidence quote.
EVIDENCE_CHARS = 500
EVIDENCE_SOURCE_KIND = "dream_ref"

_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)\n\s*```", re.S | re.I)

# One SELECT per ref kind: (ref id) -> a short text of the record. Every query reads ids only from the
# ``ids`` array parameter. ``chat`` and ``chat_turn`` both open a chat_history_log row by id or correlation id.
RESOLVE_SQL: dict[str, str] = {
    "experience": "SELECT event_id AS id, coalesce(source_kind, '') || ': ' || coalesce(label, '') "
                  "|| coalesce(' [' || verdict || ']', '') AS text FROM temporal_self_event WHERE event_id = ANY(%s)",
    "act": "SELECT act_id AS id, actor || ' ' || verb || ' ' || coalesce(object_ref, '') "
           "|| coalesce(': ' || outcome_note, '') AS text FROM grammar_acts WHERE act_id = ANY(%s)",
    "dream_cycle": "SELECT cycle_id AS id, 'sleep ' || coalesce(trigger, '') || ' ' || status || ', '"
                   " || coalesce(hypothesis_count, 0) || ' links: ' || coalesce(cycle_json->>'note', '') AS text "
                   "FROM dream_cycle WHERE cycle_id = ANY(%s)",
    "dream_hypothesis": "SELECT hypothesis_id AS id, coalesce(claim, '') || ' because ' || coalesce(why, '') AS text "
                        "FROM dream_hypothesis WHERE hypothesis_id = ANY(%s)",
    "harness_turn_trace": "SELECT correlation_id AS id, 'harness turn ' || coalesce(workflow, 'chat') || ' at ' "
                          "|| created_at::text AS text FROM harness_turn_trace WHERE correlation_id = ANY(%s)",
    "chat_turn": "SELECT x.id, x.text FROM (SELECT unnest(ARRAY[c.id, c.correlation_id]) AS id, "
                 "'Juniper: ' || left(coalesce(c.prompt, ''), 300) || ' / Orion: ' || left(coalesce(c.response, ''), 300) "
                 "AS text FROM chat_history_log c WHERE c.id = ANY(%s) OR c.correlation_id = ANY(%s)) x",
}
REF_KIND_ALIASES = {"chat": "chat_turn"}


def day_window(day: date) -> tuple[datetime, datetime]:
    """[start, end) of the Denver day ``day`` as aware UTC instants (DST-correct: built in Denver time)."""
    lo = datetime.combine(day, time(0), tzinfo=ZONE)
    hi = datetime.combine(day + timedelta(days=1), time(0), tzinfo=ZONE)
    return lo.astimezone(timezone.utc), hi.astimezone(timezone.utc)


def night_window(night: date, start_hour: int, end_hour: int) -> tuple[datetime, datetime]:
    """[start, end) of the run window on the Denver date ``night`` (it reads the day before)."""
    lo = datetime.combine(night, time(start_hour), tzinfo=ZONE)
    hi = datetime.combine(night, time(end_hour), tzinfo=ZONE)
    return lo.astimezone(timezone.utc), hi.astimezone(timezone.utc)


def canonical_ref(raw: Any) -> Optional[str]:
    """``kind:id`` with the kind's alias folded (``chat:x`` -> ``chat_turn:x``), or None for a non-ref."""
    parts = split_ref(raw)
    if parts is None:
        return None
    kind, rid = parts
    return f"{REF_KIND_ALIASES.get(kind, kind)}:{rid}"


# --------------------------------------------------------------------------- parse


def _json_candidates(text: str) -> list[str]:
    fenced = _FENCE_RE.findall(text or "")
    if fenced:
        return list(reversed(fenced))
    # The last top-level {...} span: scan from each '{' that opens at depth 0.
    out, depth, start = [], 0, None
    in_str, esc = False, False
    for i, ch in enumerate(text or ""):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"' and depth > 0:
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                out.append(text[start:i + 1])
    return list(reversed(out))


@dataclass
class ParsedAnswer:
    answer: DreamDayAnswerV1
    malformed: int = 0     # items whose shape was wrong (dropped before any ref check)


def parse_answer(text: str) -> Optional[ParsedAnswer]:
    """The answer the reply ends with, or None when it holds no JSON object with any of the three parts."""
    for raw in _json_candidates(text):
        try:
            obj = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(obj, dict) or not ({"memories", "plan", "day"} & set(obj)):
            continue
        malformed = 0
        memories, plan = [], []
        for item in obj.get("memories") or []:
            try:
                memories.append(DreamDayMemoryV1.model_validate(item))
            except ValidationError:
                malformed += 1
        for item in obj.get("plan") or []:
            try:
                plan.append(DreamDayPlanItemV1.model_validate(item))
            except ValidationError:
                malformed += 1
        day = None
        raw_day = obj.get("day")
        if isinstance(raw_day, str) and raw_day.strip():
            raw_day = {"text": raw_day, "refs": obj.get("day_refs") or []}
        if raw_day:
            try:
                day = DreamDayParagraphV1.model_validate(raw_day)
            except ValidationError:
                malformed += 1
        return ParsedAnswer(DreamDayAnswerV1(memories=memories, plan=plan, day=day), malformed)
    return None


def answer_refs(answer: DreamDayAnswerV1) -> list[str]:
    """Every distinct canonical ref the answer cites (the ones to resolve)."""
    refs: list[str] = []
    items = [*answer.memories, *answer.plan, *([answer.day] if answer.day else [])]
    for item in items:
        for r in item.refs:
            c = canonical_ref(r)
            if c and c not in refs:
                refs.append(c)
    return refs


def refs_by_kind(refs: Iterable[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for r in refs:
        kind, _, rid = r.partition(":")
        out.setdefault(kind, []).append(rid)
    return out


# --------------------------------------------------------------------------- the ref check


@dataclass
class CheckedMemory:
    statement: str
    purpose: str
    refs: list[str]


@dataclass
class RefCheck:
    memories: list[CheckedMemory] = field(default_factory=list)
    plan: list[dict[str, Any]] = field(default_factory=list)
    day_text: Optional[str] = None
    day_refs: list[str] = field(default_factory=list)
    day_cut: bool = False
    items_dropped: int = 0
    refs_dropped: int = 0
    memories_over_cap: int = 0
    plan_rejected: int = 0
    malformed: int = 0
    reasons: list[str] = field(default_factory=list)

    @property
    def kept(self) -> int:
        return sum(1 for p in self.plan if p["action"] == "keep")

    @property
    def dropped(self) -> int:
        return sum(1 for p in self.plan if p["action"] == "drop")

    @property
    def new(self) -> int:
        return sum(1 for p in self.plan if p["action"] == "new")


def _item_refs(raw_refs: Iterable[str], resolved: Mapping[str, str]) -> tuple[list[str], int]:
    """(canonical refs, how many did not resolve or are not a supported kind)."""
    good, bad = [], 0
    for r in raw_refs:
        c = canonical_ref(r)
        if c is None or c not in resolved:
            bad += 1
            continue
        if c not in good:
            good.append(c)
    return good, bad


def check_answer(parsed: ParsedAnswer, resolved: Mapping[str, str], held_concern_ids: Iterable[str]) -> RefCheck:
    """Drop every item with an unresolved ref (counted), cap memories and new concerns, keep the rest.
    ``resolved``: canonical ref -> the record's text, for the refs that resolved."""
    out = RefCheck(malformed=parsed.malformed)
    held = set(held_concern_ids)
    answer = parsed.answer

    for m in answer.memories:
        refs, bad = _item_refs(m.refs, resolved)
        if bad or not refs:
            out.items_dropped += 1
            out.refs_dropped += bad
            out.reasons.append("memory_unresolved_ref" if bad else "memory_without_ref")
            continue
        if len(out.memories) >= DREAM_DAY_MAX_MEMORIES:
            out.memories_over_cap += 1
            continue
        purpose = m.purpose if m.purpose in DREAM_DAY_PURPOSES else "happened"
        out.memories.append(CheckedMemory(statement=" ".join(m.statement.split()), purpose=purpose, refs=refs))

    seen_concerns: set[str] = set()
    for p in answer.plan:
        refs, bad = _item_refs(p.refs, resolved)
        if bad:
            out.items_dropped += 1
            out.refs_dropped += bad
            out.reasons.append("plan_unresolved_ref")
            continue
        if not p.reason.strip():
            out.plan_rejected += 1
            out.reasons.append("plan_without_reason")
            continue
        if p.action == "new":
            if not refs or not (p.gist or "").strip():
                out.plan_rejected += 1
                out.reasons.append("new_without_ref_or_gist")
                continue
            if out.new >= DREAM_DAY_MAX_NEW_CONCERNS:
                out.plan_rejected += 1
                out.reasons.append("second_new_concern")
                continue
            out.plan.append({"action": "new", "gist": " ".join(p.gist.split()), "reason": p.reason.strip(),
                             "refs": refs})
            continue
        cid = (p.concern_id or "").strip()
        if cid not in held or cid in seen_concerns:
            out.plan_rejected += 1
            out.reasons.append("plan_unknown_or_repeated_concern")
            continue
        seen_concerns.add(cid)
        out.plan.append({"action": p.action, "concern_id": cid, "reason": p.reason.strip(), "refs": refs})

    if answer.day is not None:
        refs, bad = _item_refs(answer.day.refs, resolved)
        if bad:
            out.items_dropped += 1
            out.refs_dropped += bad
            out.reasons.append("day_unresolved_ref")
        else:
            text = " ".join(answer.day.text.split())
            if len(text) > DREAM_DAY_TEXT_MAX:
                text, out.day_cut = text[:DREAM_DAY_TEXT_MAX - 1].rsplit(" ", 1)[0] + "…", True
            out.day_text, out.day_refs = text, refs
    return out


# --------------------------------------------------------------------------- memories


def dream_episode_id(day: date) -> str:
    return f"dream_day:{day.isoformat()}"


def dream_memory_id(day: date, index: int) -> str:
    """Deterministic from (day, index): a replayed persist of the same night mints the same ids."""
    return str(uuid.uuid5(MEMORY_ID_NAMESPACE, f"dream_day|{day.isoformat()}|{index}"))


def build_memories(check: RefCheck, resolved: Mapping[str, str], day: date, *,
                   known_referents: Iterable[str] = ()) -> ValidationResult:
    """The checked memories through orion.memory.episode.validate, with each ref's own text as its evidence.

    Each resolved ref is one source "turn" whose response is the record's text, so a quote is that text and
    the voice is Orion's own (``orion_thought``). After validation the ids become (day, index) ids and the
    evidence names the ref it came from (``dream_ref``, ``kind:id``), never a chat turn."""
    labels: dict[str, EpisodeTurn] = {}
    for ref in {r for m in check.memories for r in m.refs}:
        text = " ".join(str(resolved.get(ref) or "").split())[:EVIDENCE_CHARS] or ref
        labels[ref] = EpisodeTurn(label=ref, correlation_id=ref, prompt="", response=text)
    candidates = [
        DistilledMemoryV1(
            purpose=m.purpose, voice="orion_thought", channel="dream", statement=m.statement,
            occurred_at=datetime.combine(day, time(12), tzinfo=ZONE).isoformat(),
            stakes="low", stakes_reason="none",
            evidence=[DistillEvidenceV1(turn=r, field="response", quote=labels[r].response) for r in m.refs])
        for m in check.memories
    ]
    result = validate_distillation(EpisodeDistillationV1(memories=candidates), list(labels.values()),
                                   episode_id=dream_episode_id(day), prompt_version=DREAM_DAY_PROMPT_VERSION,
                                   known_referents=known_referents)
    for i, m in enumerate(result.memories):
        m.memory_id = dream_memory_id(day, i)
        m.evidence = [VerifiedEvidence(source_kind=EVIDENCE_SOURCE_KIND, source_id=ev.source_id, quote=ev.quote,
                                       verified=ev.verified) for ev in m.evidence]
    return result


def grade(memories_written: int) -> tuple[bool, str]:
    """(produced, note) for the dream_day row and the sleep act: a lasting trace is a written memory."""
    if memories_written >= 1:
        return True, f"{memories_written} memor{'y' if memories_written == 1 else 'ies'} written"
    return False, NO_TRACE_NOTE


def experience_cites(check: RefCheck, memory_refs: Iterable[str] = ()) -> list[str]:
    """Every ``experience:`` ref the kept items cite, in order (the now's cites; record_use keeps shown ones)."""
    refs = [*memory_refs, *(r for p in check.plan for r in p["refs"]), *check.day_refs]
    out: list[str] = []
    for r in refs:
        if r.startswith("experience:") and r not in out:
            out.append(r)
    return out
