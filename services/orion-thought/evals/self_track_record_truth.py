"""Truth check for the track-record cue: every number shown resolves to its faculty's record.

Independent of the renderer's templates: each shown sentence is matched to its faculty (``cue.shown``
order), and every integer in it must be one of that record's own numbers (counts, streak, or a share
rounded to a whole percent). Also: only held broken / unreliable / miscalibrated faculties, the cap
holds, each sentence names its own faculty's id in brackets (Juniper 2026-11-03: plain names along
with internal ids), and no other identifier appears in the text.

The bracketed id is removed from a sentence before its numbers are read, and a number that also
appears inside the faculty id (``gpu4`` in ``act.hub.gpu4_warm.make``, which the plain name repeats)
is allowed. Nothing is imported from the renderer's naming code: a digit the plain name invented
would still be caught, and a number forged anywhere else is still caught.
"""

from __future__ import annotations

import re

from orion.schemas.self_monitor import FacultyRecordV1, SelfTrackRecordCueV1
from orion.self_monitor.view import SelfMonitorView

NOTABLE = {"broken", "unreliable", "miscalibrated"}
_SENTENCE = re.compile(r"(?<=\.)\s+(?=[A-Z])")
_INT = re.compile(r"\d+")


def allowed_numbers(r: FacultyRecordV1) -> set[int]:
    nums = {r.n_failed_to_execute, r.n_graded + r.n_failed_to_execute, r.n_graded, r.streak_len}
    for share in (r.reliability, r.stated_mean, r.execution_failure, r.calibrated_share):
        if share is not None:
            p = round(100 * share)
            nums.add(min(99, max(1, p)) if 0.0 < share < 1.0 else p)  # the cue's no-false-0/100 clamp
    days = (r.window_end - r.window_start).total_seconds() / 86400.0
    nums.add(round(days))  # "in the last N days"
    return nums


def check_cue(cue: SelfTrackRecordCueV1, view: SelfMonitorView, *, max_chars: int = 300) -> list[str]:
    problems: list[str] = []
    if len(cue.text) > max_chars:
        problems.append(f"text is {len(cue.text)} chars > {max_chars}")
    if not cue.text:
        return problems
    sentences = _SENTENCE.split(cue.text)
    if len(sentences) < len(cue.shown):
        problems.append(f"{len(cue.shown)} faculties shown but only {len(sentences)} sentences")
    for faculty, line in zip(cue.shown, sentences):
        record = view.faculty(faculty)
        if record is None:
            problems.append(f"{faculty}: shown but not in the state")
            continue
        if record.status not in NOTABLE:
            problems.append(f"{faculty}: status {record.status} must never be cued")
        own_id = f"({faculty})"
        if own_id not in line:
            problems.append(f"{faculty}: sentence does not name its id {own_id}: {line!r}")
        prose = line.replace(own_id, "")
        in_id = {int(n) for n in _INT.findall(faculty)}
        stray = {int(n) for n in _INT.findall(prose)} - allowed_numbers(record) - in_id
        if stray:
            problems.append(f"{faculty}: numbers {sorted(stray)} not in its record: {line!r}")
    prose_text = cue.text
    for faculty in cue.shown:
        prose_text = prose_text.replace(f"({faculty})", "")
    for token in ("act.", "turn.self", "forecast.", "_"):
        if token in prose_text:
            problems.append(f"identifier fragment {token!r} in text outside a shown faculty's id")
    return problems
