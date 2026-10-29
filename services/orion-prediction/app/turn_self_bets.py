"""Turn self-bets: stance's private guesses about how this turn will go, settled against what it did.

Spec: docs/superpowers/specs/2026-10-23-unified-turn-self-prediction-design.md, A3 ("Settle").
A1 (live 2026-10-24) has stance write ``thought_decision.self_prediction`` on every unified turn. The
motor never sees it, so a hit is self-knowledge, not obedience. This module settles each bet from the
turn's own record (``harness_turn_trace.run_artifact``) and grades Orion's calibration.

Unlike action bets, the probability here is Orion's own stated number; the service does not compute
it. What the service owns is the settling rule and the grade.

Four faculties per turn, one row each (bet_id ``<correlation_id>:<faculty>``):

    turn.self.asks_question    p      hit when the reply's last 300 characters contain "?"
    turn.self.fetches_outside  p      hit when source_fetches is non-empty
    turn.self.steps            range  hit when steps_lo <= step_count <= steps_hi
    turn.self.reply_length     range  hit when reply_chars_lo <= len(final_text) <= reply_chars_hi

"?" is a crude stand-in for "asks a question". It is the first thing to replace, and it must not
grow into a word list.

Status: a probability bet records the event (``happened`` / ``not_happened``); it is never "broken" by a
low-p event that did not happen -- the Brier score grades the number. A range bet is ``upheld`` (the
outcome fell inside) or ``broken``. Void rows never count toward a grade:
    failed_to_execute       the turn ran but errored or was cut short (exit_code != 0, no final_text, or
                            the harness's cut-short notice in place of a reply)
    ungradeable_bad_record  the bet or the turn record is not a JSON object (one bad row must not stall
                            every later pass, so it settles void instead of raising)
    expired_unsettled       no turn record EXPIRY after the bet
    ungradeable_no_stated   stance stated nothing for this faculty (A1 never fills a value in)

Grades, per faculty and per caller (session_id prefix: world-pulse, curiosity, self, ...; a bare UUID
session is a chat session, ``chat``; no session id is ``unknown``):
    p faculties     Brier, and skill against the per-caller base rate. Callers differ a lot (curiosity
                    fetches, chat mostly does not), so a single base rate would credit Orion for knowing
                    who called. The per-caller rate is computed on the same rows (in-sample), which
                    makes it a slightly stricter bar, not a looser one.
    range faculties realized coverage vs the stated 0.8, median width, and the mean interval score
                    (Winkler, alpha = 1 - coverage): width plus a penalty for each miss, so an always-
                    right huge range does not score well.
"""
from __future__ import annotations

import logging
import re
import statistics
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable

from app.grading import grade
from orion.harness.cut_short import CUT_SHORT_MARKER

logger = logging.getLogger("orion-prediction.turn_self_bets")

EXPIRY = timedelta(hours=6)
LOOKBACK = timedelta(days=30)   # bets older than this are never settled (they predate A1 anyway)
PENDING_LIMIT = 2000            # bets read per settle pass
QUESTION_TAIL = 300
MIN_REPORTED = 50               # no grade below this many settled bets in a cell (= grading.MIN_GRADED)
VERDICT_ROWS = 20000
# The live grade reads bets placed in this window. Rolling, so a prompt change ages its old bets out:
# 2026-10-29 the stance prompt stopped counting repo files as "outside" and dropped a "3 to 7 steps"
# example (real turns: median 14), and bets placed before that grade a different question.
LIVE_WINDOW = timedelta(days=7)

P_FACULTIES = {"turn.self.asks_question": "asks_question_p", "turn.self.fetches_outside": "fetches_outside_p"}
RANGE_FACULTIES = {"turn.self.steps": ("steps_lo", "steps_hi"),
                   "turn.self.reply_length": ("reply_chars_lo", "reply_chars_hi")}
FACULTIES = tuple(P_FACULTIES) + tuple(RANGE_FACULTIES)
_CALLER = re.compile(r"^[a-z_]+")
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


def caller_of(session_id: str | None) -> str:
    """The turn's caller from its session id: a bare UUID is a chat session ('chat'), else the prefix
    ('orion_curiosity:...' -> 'orion_curiosity'); nothing usable -> 'unknown'. A UUID must never fall
    to the prefix rule: its first hex letters would split chat into dozens of one-session 'callers'."""
    sid = (session_id or "").strip()
    if _UUID.match(sid):
        return "chat"
    m = _CALLER.match(sid)
    prefix = m.group(0).rstrip("_") if m else ""
    return prefix or "unknown"


def _num(v) -> float | None:
    try:
        return None if v is None or isinstance(v, bool) else float(v)
    except (TypeError, ValueError):
        return None


def observe(artifact: dict | None) -> tuple[str | None, dict]:
    """(void reason or None, observed values) from a run_artifact. None artifact = no record yet."""
    if artifact is None:
        return "no_record", {}
    if not isinstance(artifact, dict):
        return "ungradeable_bad_record", {}
    text = artifact.get("final_text")
    code = artifact.get("exit_code")
    if (code not in (0, None) or not isinstance(text, str) or not text.strip()
            or text.lstrip().startswith(CUT_SHORT_MARKER)):
        return "failed_to_execute", {"exit_code": code}
    fetches = artifact.get("source_fetches")
    steps = _num(artifact.get("step_count"))
    return None, {
        "turn.self.asks_question": "?" in text[-QUESTION_TAIL:],
        "turn.self.fetches_outside": bool(fetches) if isinstance(fetches, list) else None,
        "turn.self.steps": steps,
        "turn.self.reply_length": float(len(text)),
    }


def settle(bet: dict, artifact: dict | None, now: datetime) -> list[dict] | None:
    """Rows for one turn's bet, or None while it may still settle (no record yet, not expired).

    ``bet``: correlation_id, session_id, created_at (aware), self_prediction (dict)."""
    pred = bet.get("self_prediction") or {}
    void, obs = observe(artifact)
    if not isinstance(pred, dict):
        pred, void = {}, "ungradeable_bad_record"
    if void == "no_record":
        if now - bet["created_at"] < EXPIRY:
            return None
        void = "expired_unsettled"
    coverage = _num(pred.get("coverage")) or 0.8
    base = {"correlation_id": bet["correlation_id"], "caller": caller_of(bet.get("session_id")),
            "placed_at": bet["created_at"], "settled_at": now, "coverage": coverage}
    rows = []
    for faculty in FACULTIES:
        row = dict(base, bet_id=f"{bet['correlation_id']}:{faculty}", faculty=faculty,
                   stated_p=None, stated_lo=None, stated_hi=None, observed=None, hit=None, void_reason=None)
        if faculty in P_FACULTIES:
            row["claim_kind"] = "probability"
            row["stated_p"] = _num(pred.get(P_FACULTIES[faculty]))
            stated = row["stated_p"] is not None
        else:
            row["claim_kind"] = "interval"
            lo_key, hi_key = RANGE_FACULTIES[faculty]
            row["stated_lo"], row["stated_hi"] = _num(pred.get(lo_key)), _num(pred.get(hi_key))
            stated = row["stated_lo"] is not None and row["stated_hi"] is not None
        value = obs.get(faculty)
        if void:
            row["void_reason"] = void
        elif not stated:
            row["void_reason"] = "ungradeable_no_stated"
        elif value is None:
            row["void_reason"] = "ungradeable_no_outcome"
        else:
            row["observed"] = float(value)
            row["hit"] = (bool(value) if row["claim_kind"] == "probability"
                          else row["stated_lo"] <= value <= row["stated_hi"])
        if row["void_reason"]:
            row["status"] = "void"
        elif row["claim_kind"] == "probability":
            row["status"] = "happened" if row["hit"] else "not_happened"
        else:
            row["status"] = "upheld" if row["hit"] else "broken"
        rows.append(row)
    return rows


# --- grading ---------------------------------------------------------------------------------------
def _grade_p(rows: list[dict]) -> dict:
    out = grade([r["stated_p"] for r in rows], [r["hit"] for r in rows])
    by_caller = defaultdict(list)
    for r in rows:
        by_caller[r["caller"]].append(r)
    # Reference: always forecasting the caller's own observed rate.
    ref = sum(sum((statistics.fmean([x["hit"] for x in g]) - x["hit"]) ** 2 for x in g) for g in by_caller.values())
    ref /= max(1, len(rows))
    out["brier_caller_base_rate"] = ref
    out["skill_vs_caller"] = (1 - out["brier"] / ref) if out["brier"] is not None and ref > 0 else None
    out["mean_stated_p"] = statistics.fmean(r["stated_p"] for r in rows) if rows else None
    return out


def _grade_range(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "enough": False}
    alpha = 1 - statistics.fmean(r["coverage"] for r in rows)
    scores = []
    for r in rows:
        lo, hi, x = r["stated_lo"], r["stated_hi"], r["observed"]
        s = hi - lo
        if x < lo:
            s += 2 / alpha * (lo - x)
        elif x > hi:
            s += 2 / alpha * (x - hi)
        scores.append(s)
    realized = statistics.fmean(1.0 if r["hit"] else 0.0 for r in rows)
    stated = statistics.fmean(r["coverage"] for r in rows)
    return {"n": len(rows), "enough": len(rows) >= MIN_REPORTED, "stated_coverage": stated,
            "realized_coverage": realized,
            "reads": ("too_narrow" if realized < stated - 0.1 else "too_wide" if realized > stated + 0.1
                      else "calibrated"),
            "median_width": statistics.median(r["stated_hi"] - r["stated_lo"] for r in rows),
            "median_observed": statistics.median(r["observed"] for r in rows),
            "interval_score": statistics.fmean(scores)}


def grade_faculty(faculty: str, rows: list[dict]) -> dict:
    graded = [r for r in rows if r["status"] != "void"]
    out = _grade_p(graded) if faculty in P_FACULTIES else _grade_range(graded)
    if out.get("n", 0) < MIN_REPORTED:
        # Below the floor a grade is noise: keep the count, drop the numbers.
        out = {"n": out.get("n", 0), "enough": False}
    return out


def live_summary(rows: Iterable[dict]) -> dict:
    """Per faculty: overall grade, per-caller grades, and the status counts (void reasons included)."""
    rows = list(rows)
    out: dict = {"n_rows": len(rows), "faculties": {}}
    for faculty in FACULTIES:
        fr = [r for r in rows if r["faculty"] == faculty]
        counts: dict[str, int] = defaultdict(int)
        for r in fr:
            counts[r["void_reason"] or r["status"]] += 1
        callers = sorted({r["caller"] for r in fr})
        out["faculties"][faculty] = {
            "counts": dict(counts),
            "overall": grade_faculty(faculty, fr),
            "by_caller": {c: grade_faculty(faculty, [r for r in fr if r["caller"] == c]) for c in callers},
        }
    return out


# --- service object --------------------------------------------------------------------------------
class TurnSelfBets:
    """Settle pass + live summary over a store (app/store.py PostgresStore or a fake)."""

    def __init__(self, store, clock: Callable[[], datetime] | None = None, *, verdict_ttl_sec: float = 60.0,
                 monotonic: Callable[[], float] = time.monotonic):
        self.store = store
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.verdict_ttl_sec = verdict_ttl_sec
        self._mono = monotonic
        self._live: tuple[float, dict] | None = None
        self.last_pass: dict = {}

    def settle_pending(self) -> dict:
        """One pass: read unsettled bets with their turn records, write rows for every bet that can settle."""
        now = self.clock()
        pending = self.store.pending_turn_self_bets(now - LOOKBACK, PENDING_LIMIT)
        rows, waiting = [], 0
        for bet in pending:
            settled = settle(bet, bet.get("run_artifact"), now)
            if settled is None:
                waiting += 1
            else:
                rows.extend(settled)
        written = self.store.insert_turn_self_bets(rows) if rows else 0
        self.last_pass = {"at": now, "read": len(pending), "settled_turns": len(rows) // len(FACULTIES),
                          "waiting": waiting, "rows_written": written}
        if written:
            logger.info("turn_self_bets_settled turns=%s rows=%s waiting=%s", len(rows) // len(FACULTIES),
                        written, waiting)
        return self.last_pass

    def live(self, force: bool = False, since: datetime | None = None) -> dict:
        """Grade of bets placed since ``since`` (default: the last LIVE_WINDOW). Cached for the default."""
        now = self._mono()
        if since is None and not force and self._live is not None and now - self._live[0] < self.verdict_ttl_sec:
            return self._live[1]
        start = since or self.clock() - LIVE_WINDOW
        out = live_summary(self.store.turn_self_bets(since=start, limit=VERDICT_ROWS))
        out.update(since=start, last_pass=self.last_pass)
        if since is None:
            self._live = (now, out)
        return out
