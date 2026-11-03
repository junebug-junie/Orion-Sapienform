"""Orion's track record, as a short first-person cue for stance and reverie (spec step 3). Pure.

``project_track_record_cue(view, now=...)`` turns a ``SelfMonitorView`` (the reader contract in
``view.py``) into at most ``MAX_CHARS`` characters of plain sentences, one per notable faculty:

* only held statuses ``broken`` > ``unreliable`` > ``miscalibrated`` (never ``reliable``, never
  ``unknown``, never a candidate that has not held), then by evidence size as an order of magnitude
  (attempts or graded bets), then by how far off it is, then raw evidence, then faculty key, so the
  same state always renders the same text;
* every number comes from the faculty's record (counts, decay-weighted shares, mean stated
  confidence; miscalibration from ``calibrated_share``, the share over the same bets as
  ``stated_mean``); no LLM, no identifiers in the text;
* a held status whose window numbers no longer support it (hysteresis lag) renders no line;
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
# Doers that name the hosting service, not the faculty: dropped from the plain name unless nothing
# else is left ("act.hub.say" -> "hub messages", never a bare "messages").
_HOST_WORDS = frozenset({"actions", "hub", "durable", "thought"})
# Whole-segment and suffix tokens -> words; unit suffixes (_c, _w, ...) are dropped.
_TOKEN_WORDS: dict[str, str] = {"temp_c": "temperature", "temp": "temperature", "rh": "humidity"}
_UNIT_SUFFIX = re.compile(r"_(c|f|w|kw|kwh|pct|ms|s|sec)$")


def _words(segment: str) -> str:
    seg = segment.split(":")[-1]
    if seg in _TOKEN_WORDS:
        return _TOKEN_WORDS[seg]
    for token, word in _TOKEN_WORDS.items():
        if seg.endswith("_" + token):
            seg = seg[: -len(token) - 1] + " " + word
            break
    seg = _UNIT_SUFFIX.sub("", seg)
    return re.sub(r"[_\s-]+", " ", seg).strip()


def _hyphen(segment: str) -> str:
    return _words(segment).replace(" ", "-")


def _generic(parts: list[str]) -> str:
    words = " ".join(w for w in (_words(p) for p in parts if p != "act") if w)
    return f"{words or 'unnamed'} bets"


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
            doer = [w for w in rest[:verb_at] if w not in _HOST_WORDS] or rest[:verb_at]
            name = rest[verb_at + 1:]
            words = " ".join(w for w in (_words(x) for x in doer + name) if w)
            if not words:
                return _generic(parts)  # "act.make" -> "make bets", never a bare "runs"
            noun = _VERB_NOUNS[rest[verb_at]]
            if doer == ["prime"] and not name:
                words = "Prime"
            if words.endswith(" " + noun.rstrip("s")) or words == noun.rstrip("s"):
                return f"{words}s"  # "session sessions" -> "sessions"
            return f"{words} {noun}"
    if head == "forecast" and rest:
        words = " ".join(w for w in (_words(x) for x in rest) if w)
        if words:
            return f"{words} forecast ranges"
    if head == "turn":
        words = " ".join(w for w in (_words(x) for x in rest if x != "self") if w)
        if words:
            return f"guesses about my turns' {words}"
    if head == "body" and len(rest) >= 2:
        domain, skill = _words(rest[0]), _words(rest[-1])
        if domain and skill:
            return f"{domain} {skill} moves"
    return _generic(parts)


def _pct(x: float) -> str:
    """Whole percent, but never a false 0% or 100% for a share strictly between them."""
    p = round(100 * x)
    if 0.0 < x < 1.0:
        p = min(99, max(1, p))
    return f"{p}%"


def window_phrase(record: FacultyRecordV1) -> str:
    """The record's own window in words: "this week" only for a 7-day window."""
    days = round((record.window_end - record.window_start).total_seconds() / 86400.0, 1)
    if days == 7:
        return "this week"
    if days == 1:
        return "in the last day"
    return f"in the last {days:g} days"


def _calibration_off(record: FacultyRecordV1) -> bool:
    """Do this window's numbers still say miscalibrated? (The held status lags by hysteresis.)"""
    if record.candidate_status == "miscalibrated":
        return True
    s, lo, hi = record.stated_mean, record.calibrated_lo, record.calibrated_hi
    return None not in (s, lo, hi) and (s < lo or s > hi)


def sentence(record: FacultyRecordV1) -> Optional[str]:
    """One plain sentence for a notable faculty from its own numbers, or None when the numbers are
    missing or no longer support the held status (a held status can lag its window by hysteresis;
    the cue never prints numbers that contradict the claim)."""
    name = plain_name(record.faculty)
    status = record.status
    when = window_phrase(record)
    if status == "broken":
        attempts = record.n_graded + record.n_failed_to_execute
        lo = record.execution_failure_lo
        if attempts and lo is not None and lo > 0.5:
            return f"My {name} failed to start {record.n_failed_to_execute} of {attempts} times {when}."
        if record.streak_kind == "failed_to_execute" and record.streak_len:
            return f"My last {record.streak_len} {name} all failed to start."
        return None
    if status == "unreliable":
        if record.reliability is None or record.reliability_hi is None or record.reliability_hi >= 0.5:
            return None
        return f"My {name} work {_pct(record.reliability)} of the time ({record.n_graded} tries {when})."
    if status == "miscalibrated":
        share, stated = record.calibrated_share, record.stated_mean
        if share is None or stated is None or not _calibration_off(record):
            return None
        works, said = _pct(share), _pct(stated)
        if record.claim_kind == "interval":
            return f"My {name} hold {works} of the time (I say {said})."
        if record.claim_kind == "success":
            return f"My {name} work {works} of the time (I expect {said})."
        return f"My {name} come true {works} of the time (I say {said})."
    return None


def _evidence(record: FacultyRecordV1) -> int:
    return record.n_graded + record.n_failed_to_execute


def _evidence_tier(record: FacultyRecordV1) -> int:
    """Evidence by order of magnitude (1-9, tens, hundreds, thousands...). Ranking on the raw count
    let 4 extra bets put "51% vs 38%" ahead of "5% vs 80%" on live data (2026-11-03); within a tier
    the bigger miss leads, and the raw count only breaks a remaining tie."""
    n = _evidence(record)
    return len(str(n)) if n > 0 else 0


def _miss(record: FacultyRecordV1) -> float:
    """How far the record is from fine: tie-break among equal evidence (sibling turn self-bets share
    one bet count, and "6% vs 80%" says more than "11% vs 7%")."""
    if record.status == "broken":
        return record.execution_failure or 0.0
    if record.status == "miscalibrated":
        if record.calibrated_share is None or record.stated_mean is None:
            return 0.0
        return abs(record.stated_mean - record.calibrated_share)
    return 1.0 - (record.reliability if record.reliability is not None else 1.0)


def project_track_record_cue(view: SelfMonitorView, *, now: datetime,
                             max_chars: int = MAX_CHARS) -> SelfTrackRecordCueV1:
    """Never raises for a valid view; "" whenever nothing may be shown."""
    if not view.known:
        return SelfTrackRecordCueV1(rendered_at=now, skip_reason=view.reason, state_age_sec=view.age_sec)
    notable = sorted(
        (r for r in view.faculties.values() if r.status in NOTABLE),
        key=lambda r: (NOTABLE.index(r.status), -_evidence_tier(r), -round(_miss(r), 6), -_evidence(r),
                       r.faculty),
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


__all__ = ["MAX_CHARS", "plain_name", "project_track_record_cue", "sentence", "window_phrase"]
