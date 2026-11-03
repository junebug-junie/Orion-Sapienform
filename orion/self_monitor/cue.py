"""Orion's track record, as a short first-person cue for stance and reverie (spec step 3). Pure.

``project_track_record_cue(view, now=...)`` turns a ``SelfMonitorView`` (the reader contract in
``view.py``) into at most ``MAX_CHARS`` characters of plain sentences, one per notable faculty:

* only held statuses ``broken`` > ``unreliable`` > ``miscalibrated`` (never ``reliable``, never
  ``unknown``, never a candidate that has not held), then by evidence size (attempts or graded bets),
  then by how far off it is, then by faculty key, so the same state always renders the same text;
* every number comes from the faculty's record (counts, decay-weighted shares, mean stated
  confidence); no LLM, no identifiers in the text;
* when at least one sentence shows and room is left, one short clause counts the faculties that
  cannot be graded at all yet (structural reasons, not "too few bets");
* an unknown view (absent, unparseable, stale, future-stamped) renders "" with the view's reason as
  ``skip_reason``; nothing notable renders "" with ``nothing_notable``.

Faculty keys map to plain plural nouns with ``plain_name`` (a small table plus rules per key family;
anything unrecognised becomes its humanised words plus "bets").
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from orion.schemas.self_monitor import FacultyRecordV1, SelfTrackRecordCueV1
from orion.self_monitor.view import SelfMonitorView

MAX_CHARS = 300
NOTABLE: tuple[str, ...] = ("broken", "unreliable", "miscalibrated")
# Gradeability reasons that mean "cannot be graded as built", not "not enough bets this week".
STRUCTURAL_UNGRADEABLE = frozenset({
    "ungradeable:source_bug",
    "ungradeable:no_stated_confidence",
    "ungradeable:self_graded",
    "ungradeable:never_settled",
})

# Exact keys whose generic rule reads badly.
_EXACT: dict[str, str] = {
    "turn.self.steps": "guesses about how many steps my turns take",
    "turn.self.reply_length": "guesses about how long my replies run",
    "turn.self.asks_question": "guesses that I'll ask a question",
    "turn.self.fetches_outside": "guesses that I'll look something up outside",
    "vision.percept_expectation": "guesses about what I'll see next",
    "act.hub.world_pulse_read.look": "world-pulse reads",
    "act.topic_foundry.run.maintain": "topic-foundry maintenance runs",
    "act.aitown.session.play": "AI Town sessions",
}
# Act verbs -> the plural noun for one try of that act.
_VERB_NOUNS: dict[str, str] = {
    "make": "runs", "run": "runs", "predict": "predictions", "infer": "inferences", "look": "looks",
    "say": "messages", "ask": "questions", "play": "sessions", "sense": "checks",
    "maintain": "maintenance runs", "act": "acts",
}
# Doers that name the hosting service, not the faculty: dropped from the plain name.
_HOST_WORDS = frozenset({"actions", "hub", "durable", "thought"})
_TOKEN_WORDS: dict[str, str] = {"temp_c": "temperature", "temp": "temperature", "rh": "humidity"}


def _words(segment: str) -> str:
    seg = segment.split(":")[-1]
    if seg in _TOKEN_WORDS:
        return _TOKEN_WORDS[seg]
    for token, word in _TOKEN_WORDS.items():
        if seg.endswith("_" + token):
            seg = seg[: -len(token) - 1] + " " + word
            break
    return re.sub(r"[_\s]+", " ", seg).strip()


def _hyphen(segment: str) -> str:
    return _words(segment).replace(" ", "-")


def plain_name(faculty: str) -> str:
    """Faculty key -> a plain plural noun phrase that reads after "My ". Deterministic."""
    if faculty in _EXACT:
        return _EXACT[faculty]
    parts = [p for p in faculty.split(".") if p]
    head, rest = (parts[0], parts[1:]) if parts else ("", [])
    if head == "act" and rest:
        if rest[:2] == ["actions", "journal"] and len(rest) >= 4:
            return f"{_hyphen(rest[3])} journal runs"
        verb_at = next((i for i in range(len(rest) - 1, -1, -1) if rest[i] in _VERB_NOUNS), None)
        if verb_at is not None and verb_at >= len(rest) - 2:
            doer = [w for w in rest[:verb_at] if w not in _HOST_WORDS]
            name = rest[verb_at + 1:]
            words = " ".join(_words(w) for w in doer + name)
            noun = _VERB_NOUNS[rest[verb_at]]
            if doer == ["prime"] and not name:
                words = "Prime"
            if words.endswith(" " + noun.rstrip("s")) or words == noun.rstrip("s"):
                return f"{words}s"  # "session sessions" -> "sessions"
            return f"{words} {noun}".strip()
    if head == "forecast" and rest:
        return f"{' '.join(_words(w) for w in rest)} forecast ranges"
    if head == "turn" and rest:
        return f"guesses about my turns' {' '.join(_words(w) for w in rest if w != 'self')}"
    if head == "body" and len(rest) >= 2:
        return f"{_words(rest[0]).capitalize()} {_words(rest[-1])} moves"
    words = " ".join(_words(p) for p in parts if p != "act")
    return f"{words or 'unnamed'} bets"


def _pct(x: float) -> str:
    return f"{round(100 * x)}%"


def sentence(record: FacultyRecordV1) -> Optional[str]:
    """One plain sentence for a notable faculty from its own numbers, or None if they are missing."""
    name = plain_name(record.faculty)
    status = record.status
    if status == "broken":
        attempts = record.n_graded + record.n_failed_to_execute
        lo = record.execution_failure_lo
        if attempts and lo is not None and lo > 0.5:
            return f"My {name} failed to start {record.n_failed_to_execute} of {attempts} times this week."
        if record.streak_kind == "failed_to_execute" and record.streak_len:
            return f"My last {record.streak_len} {name} all failed to start."
        return None
    if record.reliability is None:
        return None
    works = _pct(record.reliability)
    if status == "unreliable":
        return f"My {name} work {works} of the time ({record.n_graded} tries this week)."
    if status == "miscalibrated" and record.stated_mean is not None:
        said = _pct(record.stated_mean)
        if record.claim_kind == "interval":
            return f"My {name} hold {works} of the time (I say {said})."
        if record.claim_kind == "success":
            return f"My {name} work {works} of the time (I expect {said})."
        return f"My {name} come true {works} of the time (I say {said})."
    return None


def _evidence(record: FacultyRecordV1) -> int:
    return record.n_graded + record.n_failed_to_execute


def _miss(record: FacultyRecordV1) -> float:
    """How far the record is from fine: tie-break among equal evidence (sibling turn self-bets share
    one bet count, and "6% vs 80%" says more than "11% vs 7%")."""
    if record.status == "broken":
        return record.execution_failure or 0.0
    if record.reliability is None:
        return 0.0
    if record.status == "miscalibrated" and record.stated_mean is not None:
        return abs(record.stated_mean - record.reliability)
    return 1.0 - record.reliability


def project_track_record_cue(view: SelfMonitorView, *, now: datetime,
                             max_chars: int = MAX_CHARS) -> SelfTrackRecordCueV1:
    """Never raises for a valid view; "" whenever nothing may be shown."""
    if not view.known:
        return SelfTrackRecordCueV1(rendered_at=now, skip_reason=view.reason, state_age_sec=view.age_sec)
    notable = sorted(
        (r for r in view.faculties.values() if r.status in NOTABLE),
        key=lambda r: (NOTABLE.index(r.status), -_evidence(r), -round(_miss(r), 6), r.faculty),
    )
    lines: list[str] = []
    shown: list[str] = []
    dropped = 0
    used = 0
    for record in notable:
        text = sentence(record)
        if text is None:
            continue
        extra = len(text) + (1 if lines else 0)
        if dropped or used + extra > max_chars:  # strict priority: never skip ahead to a shorter one
            dropped += 1
            continue
        lines.append(text)
        shown.append(record.faculty)
        used += extra
    if not lines:
        return SelfTrackRecordCueV1(rendered_at=now, skip_reason="nothing_notable", state_age_sec=view.age_sec)
    ungradeable = sum(1 for r in view.faculties.values()
                      if r.status == "unknown" and r.gradeability in STRUCTURAL_UNGRADEABLE)
    if ungradeable and not dropped:
        clause = f"I can't grade {ungradeable} other {'thing' if ungradeable == 1 else 'things'} I do yet."
        if used + 1 + len(clause) <= max_chars:
            lines.append(clause)
    return SelfTrackRecordCueV1(rendered_at=now, text=" ".join(lines), shown=shown, lines_dropped=dropped,
                                state_age_sec=view.age_sec)


__all__ = ["MAX_CHARS", "plain_name", "project_track_record_cue", "sentence"]
