"""Orion's track record, as a short first-person cue for stance and reverie (spec step 3). Pure.

``project_track_record_cue(view, now=...)`` turns a ``SelfMonitorView`` (the reader contract in
``view.py``) into at most ``MAX_CHARS`` characters of plain sentences, one per notable faculty:

* only held statuses ``broken`` > ``unreliable`` > ``miscalibrated`` (never ``reliable``, never
  ``unknown``, never a candidate that has not held), then by evidence size as an order of magnitude
  (attempts or graded bets), then by how far off it is, then raw evidence, then faculty key, so the
  same state always renders the same text;
* every number comes from the faculty's record (counts, decay-weighted shares, mean stated
  confidence; miscalibration from ``calibrated_share``, the share over the same bets as
  ``stated_mean``); no LLM;
* each faculty is named as ``names.named``: plain words first, the internal id after it in
  brackets (Juniper 2026-11-03: "use plain names ALONG with internal ids"); the truth eval removes
  the bracketed id before it reads numbers, and the cap still drops whole lines rather than cutting one;
* a held status whose window numbers no longer support it (hysteresis lag) renders no line;
* when at least one sentence shows and room is left, one short clause counts the faculties that
  cannot be graded at all yet (structural reasons, not "too few bets");
* an unknown view (absent, unparseable, stale, future-stamped) renders "" with the view's reason as
  ``skip_reason``; nothing notable renders "" with ``nothing_notable``.

Faculty keys map to plain plural nouns with ``orion.self_monitor.names.plain_name`` (shared with
E2, E3 and the Temporal Self faculty arcs; re-exported here).
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from orion.schemas.self_monitor import FacultyRecordV1, SelfTrackRecordCueV1
from orion.self_monitor.names import named, plain_name
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
    name = named(record.faculty)
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


__all__ = ["MAX_CHARS", "named", "plain_name", "project_track_record_cue", "sentence", "window_phrase"]
