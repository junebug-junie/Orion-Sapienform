"""Street rhythm surprise: how surprised the walkway rhythm forecast was by who did or didn't appear.

Spec: docs/superpowers/specs/2026-11-24-pe-retire-perception-and-alarms-design.md rev 4, item 5 and
retire step 6 (Juniper approved 2026-11-24: shadow until a 7-day calm check passes). Orion's one real
perception prediction error. The room camera has no forecast, so it has no counterpart here.

Source. The sql-writer rhythm learner (``services/orion-sql-writer/app/vision_rhythm.py``) writes an
expectation BEFORE its window opens: "a known subject (``individual:<id>`` or ``label:<label>``)
appears on the walkway between window_start and window_end, with probability ``confidence``". It
scores it after the window closes: ``met``, ``missed``, or ``unscorable`` (the camera did not watch
enough of the window). Rows live in ``vision_percept_expectation``; this module only reads them.

Score, one unit (nats), calm point set by math, on the RAW stated p (clamped to [0.02, 0.98]):

    surprisal  s   = -ln p(outcome)                 (p if met, 1 - p if missed)
    calm       H   = -p ln p - (1 - p) ln(1 - p)    (the mean of s if p is right)
    excess     s-H                                   (mean 0 under calibration)
    calm_var       = p (1 - p) logit(p)^2            (its variance under calibration)

No recalibration (design rev 4): a held-out logistic map made log loss worse by 0.0067 nats (SE 0.0050),
so the node scores the same number Orion reads. Rule kept: recalibrate only when held-out log loss
improves by more than 2 SE, and then show both numbers.

Refusals (reported as a rate, never as misses):

- ``unscorable``: the camera did not watch the window. Absence of evidence is not a miss (#3388).
- ``stale_key``: an ``individual:`` subject not seen for 7+ days before the window opened. A key that
  stopped appearing may be a departure or an identity re-key (#3388, scripts/reanchor_vision_individuals.py);
  either way the forecast about it is not news. The forecaster already refits on a rolling 28-day
  window (``VISION_RHYTHM_HISTORY_DAYS``), so no refit happens here.
- ``unknown_key``: an ``individual:`` subject with no sighting before the window at all (deleted or
  merged by a re-anchor). Same reason.

``label:`` subjects (``label:vehicle``) are a class, not an identity: they have no stale rule.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping

from orion.prediction import surprisal as sp

STREAM_ID = "walkway"
P_MIN, P_MAX = 0.02, 0.98
STALE_KEY_AFTER = timedelta(days=7)
# Only outcomes scored this recently are published: the node's fold window
# (orion/substrate/prediction_street_rhythm.py WINDOW). A backlog after an outage is never
# written as now.
PUBLISH_MAX_AGE = timedelta(minutes=30)
SCORED_STATUSES = ("met", "missed")
REFUSAL_REASONS = ("unscorable", "stale_key", "unknown_key")


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


def clamp_p(p: float) -> float:
    return sp.clip_p(p, P_MIN)   # [P_MIN, 1 - P_MIN] == [P_MIN, P_MAX] (tested)


@dataclass(frozen=True)
class OutcomeTerms:
    p: float
    happened: bool
    surprisal: float
    calm_surprisal: float
    excess: float
    calm_var: float


def outcome_terms(stated_p: float, happened: bool) -> OutcomeTerms:
    """The raw stated p's terms (the math: ``orion.prediction.surprisal.bet_terms``, clamp P_MIN)."""
    t = sp.bet_terms(stated_p, happened, P_MIN)
    return OutcomeTerms(p=t.q, happened=t.happened, surprisal=t.surprisal, calm_surprisal=t.entropy,
                        excess=t.excess, calm_var=t.calm_var)


def subject_kind(subject_key: str) -> str | None:
    head = str(subject_key or "").split(":", 1)[0]
    return head if head in ("individual", "label") else None


@dataclass(frozen=True)
class StreetRhythmOutcome:
    expectation_id: str
    subject_key: str
    subject_kind: str
    window_start: datetime
    window_end: datetime
    scored_at: datetime
    stated_p: float
    terms: OutcomeTerms
    key_age_days: float | None   # individual: window_start - first sighting; label: None

    def event(self) -> dict[str, Any]:
        """The ``prediction.street_rhythm_surprise.v1`` payload (orion/schemas/prediction.py)."""
        t = self.terms
        return {
            "expectation_id": self.expectation_id, "stream_id": STREAM_ID, "subject_key": self.subject_key,
            "subject_kind": self.subject_kind, "window_start": self.window_start,
            "window_end": self.window_end, "scored_at": self.scored_at, "stated_p": self.stated_p,
            "p": t.p, "happened": t.happened, "surprisal": t.surprisal, "calm_surprisal": t.calm_surprisal,
            "excess": t.excess, "calm_var": t.calm_var, "key_age_days": self.key_age_days,
        }


def score_row(row: Mapping[str, Any]) -> tuple[StreetRhythmOutcome | None, str]:
    """(outcome or None, reason) for one closed expectation row. ``row`` carries the
    vision_percept_expectation columns plus ``last_seen_before`` / ``first_seen_before`` (the
    subject's latest and earliest sighting start before window_start; None for a label subject or
    an individual never seen before the window)."""
    if str(row.get("stream_id") or "") != STREAM_ID:
        return None, "not_walkway"
    status = str(row.get("status") or "")
    if status == "unscorable":
        return None, "unscorable"
    if status not in SCORED_STATUSES:
        return None, "not_closed"
    key = str(row.get("subject_key") or "")
    kind = subject_kind(key)
    if kind is None:
        return None, "unknown_subject_kind"
    start = _aware(row["window_start"])
    key_age: float | None = None
    if kind == "individual":
        last_seen = row.get("last_seen_before")
        if last_seen is None:
            return None, "unknown_key"
        if start - _aware(last_seen) >= STALE_KEY_AFTER:
            return None, "stale_key"
        first_seen = row.get("first_seen_before")
        if first_seen is not None:
            key_age = max(0.0, (start - _aware(first_seen)).total_seconds() / 86400.0)
    p = row.get("confidence")
    if p is None or not math.isfinite(float(p)):
        return None, "no_stated_p"
    if row.get("scored_at") is None:
        return None, "not_closed"
    return StreetRhythmOutcome(
        expectation_id=str(row["expectation_id"]), subject_key=key, subject_kind=kind,
        window_start=start, window_end=_aware(row["window_end"]), scored_at=_aware(row["scored_at"]),
        stated_p=float(p), terms=outcome_terms(float(p), status == "met"), key_age_days=key_age,
    ), "scored"


class StreetRhythmLive:
    """One pass per tick: read the outcomes scored in the last ``PUBLISH_MAX_AGE``, score the new ones,
    queue their events for the bus. Read-only on the database. In-memory dedupe: a restart republishes
    the last 30 min once, which the substrate fold absorbs (it keys samples by expectation_id)."""

    def __init__(self, store: Any, *, clock=None):
        self._store = store
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._seen: dict[str, datetime] = {}
        self._pending: list[dict[str, Any]] = []
        self.counts: dict[str, int] = {}
        self.last: dict[str, Any] = {}

    def _bump(self, key: str, n: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + n

    def tick(self) -> int:
        now = _aware(self._clock())
        since = now - PUBLISH_MAX_AGE
        self._seen = {k: t for k, t in self._seen.items() if t >= since}
        rows = self._store.street_rhythm_outcomes(since)
        queued = 0
        for row in rows:
            eid = str(row.get("expectation_id") or "")
            if not eid or eid in self._seen:
                continue
            self._seen[eid] = _aware(row["scored_at"]) if row.get("scored_at") else now
            out, reason = score_row(row)
            self._bump(reason)
            if out is None:
                continue
            self._pending.append(out.event())
            queued += 1
        self.last = {"at": now.isoformat(), "rows": len(rows), "queued": queued}
        return queued

    def take_pending_events(self) -> list[dict[str, Any]]:
        out, self._pending = self._pending, []
        return out


# --- calm eval helpers (evals/run_street_rhythm_calm_eval.py) -----------------------------------------


def score_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[list[StreetRhythmOutcome], dict[str, int]]:
    outs: list[StreetRhythmOutcome] = []
    reasons: dict[str, int] = {}
    for r in rows:
        o, why = score_row(r)
        reasons[why] = reasons.get(why, 0) + 1
        if o is not None:
            outs.append(o)
    return outs, reasons


def quantile(values: list[float], q: float) -> float | None:
    """Linear-interpolated quantile; None for an empty list."""
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)
