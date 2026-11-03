"""Reduce settled bets into one ``FacultyRecordV1`` per faculty. Pure: no I/O, no clock.

Window and decay: bets settled in the last ``window_days`` count, each weighted
``0.5 ** (age / half_life)``, so an old failure fades and a healed faculty can recover (an all-time
total would be an absorbing state). Intervals are Wilson score intervals on the decay-weighted share
with Kish's effective sample size ``sum(w)^2 / sum(w^2)`` (equal weights give back n; ``sum(w)`` would
shrink the evidence by the decay itself and hide a real failure streak).

Candidate status, first match wins (``reasons`` names the rule that fired):

1. ``broken``: at least ``MIN_ATTEMPTS`` attempts and the failed-to-execute share's Wilson lower bound
   is above 0.5, or the latest ``BROKEN_STREAK`` attempts all failed to execute. Any claim kind.
2. Below ``MIN_GRADED`` graded bets: ``unknown`` (gradeability says why).
3. ``unreliable`` (success claims only): the upheld share's upper bound is below 0.5 -- the act fails
   more often than it works, with confidence.
4. ``miscalibrated``: the decay-weighted mean stated confidence lies outside the upheld share's
   interval (both over the same bets: graded ones that stated a confidence) by more than
   ``min(CALIBRATION_SLACK, 0.25 * min(p, 1 - p))`` -- an absolute 0.05 at p = 0.07 would forgive a
   ~70% relative miss. For intervals the upheld share is the realized coverage. Skipped when the
   stated confidence was formed at a coarser key shared with other faculties (``belief_key``): the
   act ledger keeps one posterior per verb|actor|signal, so judging one trigger name against it
   would blame that name for its siblings.
5. ``reliable``: success claims whose lower bound is at least 0.5; interval claims that pass rule 4;
   event claims that pass rule 4 AND have Brier skill above 0 (calibration-in-the-large is necessary,
   not sufficient: always saying the base rate passes it).
6. ``unknown``.

Held status (``advance``): a new candidate replaces the held status only after it has held for
``HOLD`` consecutive evaluations, so one bad hour does not flip a faculty (trailing-window
thresholds need hysteresis).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Optional, Sequence

from orion.prediction.grading import grade
from orion.schemas.self_monitor import FacultyRecordV1, SettledBetV1

WINDOW_DAYS = 7.0
HALF_LIFE_DAYS = 3.5
MIN_GRADED = 10
MIN_ATTEMPTS = 10
BROKEN_STREAK = 5
CALIBRATION_SLACK = 0.05
HOLD = 3
Z = 1.96

_GRADING_SETTLERS = ("world", "juniper")


@dataclass(frozen=True)
class Ungradeable:
    """A faculty the organ knows about but cannot grade from its own bets (see catalog.py)."""

    faculty: str
    source: str
    gradeability: str
    unlock: str


def wilson(share: float, n_eff: float) -> tuple[float, float]:
    if n_eff <= 0:
        return 0.0, 1.0
    denom = 1 + Z * Z / n_eff
    centre = (share + Z * Z / (2 * n_eff)) / denom
    half = Z * math.sqrt(share * (1 - share) / n_eff + Z * Z / (4 * n_eff * n_eff)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def kish(weights: Sequence[float]) -> float:
    total = sum(weights)
    squares = sum(w * w for w in weights)
    return (total * total / squares) if squares > 0 else 0.0


def _weight(bet: SettledBetV1, now: datetime) -> float:
    age_days = max(0.0, (now - bet.settled_at).total_seconds() / 86400.0)
    return 0.5 ** (age_days / HALF_LIFE_DAYS)


def _streak(bets: Sequence[SettledBetV1]) -> tuple[Optional[str], int, Optional[datetime]]:
    counted = [b for b in sorted(bets, key=lambda b: b.settled_at, reverse=True)
               if b.outcome in ("upheld", "refuted", "failed_to_execute")]
    if not counted:
        return None, 0, None
    kind = counted[0].outcome
    n = 0
    since = None
    for bet in counted:
        if bet.outcome != kind:
            break
        n += 1
        since = bet.settled_at
    return kind, n, since


def _slack(p: float) -> float:
    return min(CALIBRATION_SLACK, 0.25 * min(p, 1.0 - p))


def reduce_faculty(faculty: str, bets: Sequence[SettledBetV1], *, now: datetime,
                   window_days: float = WINDOW_DAYS, pooled_belief: Optional[str] = None) -> FacultyRecordV1:
    """``pooled_belief``: the belief key this faculty shares with other faculties, if any (set by
    ``reduce_bets``). Calibration is not judged then (rule 4)."""
    start = now - timedelta(days=window_days)
    bets = [b for b in bets if start <= b.settled_at <= now]
    graded = [b for b in bets if b.outcome in ("upheld", "refuted") and b.settled_by in _GRADING_SETTLERS]
    failed = [b for b in bets if b.outcome == "failed_to_execute"]
    expired = [b for b in bets if b.outcome == "expired_unsettled"]
    attempts = graded + failed
    kinds = sorted({b.claim_kind for b in bets})
    claim_kind = kinds[0] if len(kinds) == 1 else None

    wg = [_weight(b, now) for b in graded]
    wa = [_weight(b, now) for b in attempts]
    w_graded = sum(wg)
    w_upheld = sum(w for w, b in zip(wg, graded) if b.outcome == "upheld")
    w_attempts = sum(wa)
    w_failed = sum(w for w, b in zip(wa, attempts) if b.outcome == "failed_to_execute")

    reliability = lo = hi = None
    if w_graded > 0:
        reliability = w_upheld / w_graded
        lo, hi = wilson(reliability, kish(wg))
    exec_fail = exec_lo = None
    if w_attempts > 0:
        exec_fail = w_failed / w_attempts
        exec_lo = wilson(exec_fail, kish(wa))[0]

    stated = [b for b in graded if b.stated is not None]
    ws = [_weight(b, now) for b in stated]
    stated_mean = cal_rel = cal_lo = cal_hi = None
    if stated and sum(ws) > 0:
        stated_mean = sum(w * b.stated for w, b in zip(ws, stated)) / sum(ws)
        cal_rel = sum(w for w, b in zip(ws, stated) if b.outcome == "upheld") / sum(ws)
        cal_lo, cal_hi = wilson(cal_rel, kish(ws))
    brier = skill = None
    if stated and claim_kind in ("success", "event") and pooled_belief is None:
        g = grade([b.stated for b in stated], [b.outcome == "upheld" for b in stated])
        brier, skill = g["brier"], g["skill"]

    streak_kind, streak_len, streak_since = _streak(bets)

    if graded and len(graded) >= MIN_GRADED and stated:
        gradeability, unlock = "graded", None
    elif bets and all(b.settled_by == "self" for b in bets):
        gradeability, unlock = "ungradeable:self_graded", "settle against the world, not Orion's own re-judgment"
    elif graded and not stated:
        gradeability, unlock = "ungradeable:no_stated_confidence", "record a confidence before the outcome"
    elif bets and not graded and not failed:
        gradeability, unlock = "ungradeable:never_settled", "settle bets before they expire"
    else:
        gradeability, unlock = "ungradeable:too_few", f"needs {MIN_GRADED} graded bets in {window_days:g} days"

    reasons: list[str] = []
    candidate = "unknown"
    if len(attempts) >= MIN_ATTEMPTS and exec_lo is not None and exec_lo > 0.5:
        candidate = "broken"
        reasons.append(f"failed_to_execute {len(failed)}/{len(attempts)} (lower bound {exec_lo:.2f})")
    elif streak_kind == "failed_to_execute" and streak_len >= BROKEN_STREAK:
        candidate = "broken"
        reasons.append(f"last {streak_len} attempts failed to execute")
    elif gradeability != "graded":
        reasons.append(gradeability)
    elif None in (reliability, lo, hi, stated_mean, cal_rel, cal_lo, cal_hi):
        reasons.append("no graded evidence")  # unreachable while "graded" implies both sets non-empty
    else:
        what = "coverage" if claim_kind == "interval" else "happened"
        judged = pooled_belief is None
        slack = _slack(stated_mean)
        off = judged and (stated_mean < cal_lo - slack or stated_mean > cal_hi + slack)
        if claim_kind == "success" and hi < 0.5:
            candidate = "unreliable"
            reasons.append(f"works {reliability:.0%} (upper bound {hi:.2f} < 0.5)")
        elif off:
            candidate = "miscalibrated"
            reasons.append(f"stated {stated_mean:.2f}, {what} {cal_rel:.2f} [{cal_lo:.2f}, {cal_hi:.2f}]")
        elif claim_kind == "success":
            if lo >= 0.5:
                candidate = "reliable"
                reasons.append(f"works {reliability:.0%} (lower bound {lo:.2f})")
            else:
                reasons.append(f"works {reliability:.0%}, interval [{lo:.2f}, {hi:.2f}] spans 0.5")
        elif claim_kind == "event" and not (skill is not None and skill > 0):
            reasons.append(f"stated {stated_mean:.2f} matches {what} {cal_rel:.2f}, but no skill over the base rate")
        elif judged:
            candidate = "reliable"
            reasons.append(f"stated {stated_mean:.2f} matches {what} {cal_rel:.2f} [{cal_lo:.2f}, {cal_hi:.2f}]")
        if not judged:
            reasons.append(f"calibration not judged: confidence is pooled at {pooled_belief}")

    return FacultyRecordV1(
        faculty=faculty,
        sources=sorted({b.source for b in bets}),
        claim_kind=claim_kind,
        gradeability=gradeability,
        unlock=unlock,
        window_start=start,
        window_end=now,
        n_bets=len(bets),
        n_graded=len(graded),
        n_failed_to_execute=len(failed),
        n_expired=len(expired),
        weight_graded=w_graded,
        reliability=reliability,
        reliability_lo=lo,
        reliability_hi=hi,
        execution_failure=exec_fail,
        execution_failure_lo=exec_lo,
        stated_mean=stated_mean,
        calibrated_share=cal_rel,
        calibrated_lo=cal_lo,
        calibrated_hi=cal_hi,
        brier=brier,
        brier_skill=skill,
        streak_kind=streak_kind,
        streak_len=streak_len,
        streak_since=streak_since,
        candidate_status=candidate,
        status=candidate,
        status_since=None,
        candidate_held=1,
        reasons=reasons,
    )


def reduce_bets(bets: Iterable[SettledBetV1], *, now: datetime, window_days: float = WINDOW_DAYS,
                ungradeable: Iterable[Ungradeable] = ()) -> list[FacultyRecordV1]:
    """Group by faculty and reduce. Each bet_id counts once (first seen wins)."""
    seen: set[str] = set()
    by_faculty: dict[str, list[SettledBetV1]] = defaultdict(list)
    for bet in bets:
        if bet.bet_id in seen:
            continue
        seen.add(bet.bet_id)
        by_faculty[bet.faculty].append(bet)
    owners: dict[str, set[str]] = defaultdict(set)
    for faculty, fbets in by_faculty.items():
        for bet in fbets:
            if bet.belief_key:
                owners[bet.belief_key].add(faculty)

    def pooled(fbets: list[SettledBetV1]) -> Optional[str]:
        shared = sorted({b.belief_key for b in fbets if b.belief_key and len(owners[b.belief_key]) > 1})
        return shared[0] if shared else None

    records = [reduce_faculty(f, b, now=now, window_days=window_days, pooled_belief=pooled(b))
               for f, b in sorted(by_faculty.items())]
    known = {r.faculty for r in records}
    start = now - timedelta(days=window_days)
    for u in ungradeable:
        if u.faculty in known:
            continue
        records.append(FacultyRecordV1(
            faculty=u.faculty, sources=[u.source], gradeability=u.gradeability, unlock=u.unlock,
            window_start=start, window_end=now, n_bets=0, n_graded=0, n_failed_to_execute=0,
            n_expired=0, weight_graded=0.0, candidate_status="unknown", status="unknown",
            candidate_held=1, reasons=[u.gradeability],
        ))
    return records


def advance(previous: Optional[FacultyRecordV1], current: FacultyRecordV1, *, now: datetime,
            hold: int = HOLD) -> FacultyRecordV1:
    """Carry the held status forward. The first evaluation adopts its candidate (its window is
    already ``window_days`` of evidence); after that a different candidate must hold ``hold``
    consecutive evaluations before it becomes the status."""
    if previous is not None and previous.faculty != current.faculty:
        raise ValueError(f"advance: {previous.faculty!r} is not {current.faculty!r}")
    if previous is None or previous.status_since is None:  # no held status yet: a first evaluation
        return current.model_copy(update={"status": current.candidate_status, "status_since": now,
                                          "candidate_held": hold})
    held = previous.candidate_held + 1 if current.candidate_status == previous.candidate_status else 1
    status, since = previous.status, previous.status_since
    if current.candidate_status != status and held >= hold:
        status, since = current.candidate_status, now
    return current.model_copy(update={"status": status, "status_since": since, "candidate_held": held})
