"""Turn self-bets: stance's private guesses about how this turn will go, settled against what it did.

Spec: docs/superpowers/specs/2026-10-23-unified-turn-self-prediction-design.md, A3 ("Settle").
A1 (live 2026-10-24) has stance write ``thought_decision.self_prediction`` on every unified turn. The
motor never sees it, so a hit is self-knowledge, not obedience. This module settles each bet from the
turn's own record (``harness_turn_trace.run_artifact``) and grades Orion's calibration.

Unlike action bets, the probability here is Orion's own stated number; the service does not compute
it. What the service owns is the settling rule and the grade.

Six faculties per turn, one row each (bet_id ``<correlation_id>:<faculty>``):

    turn.self.asks_question            p      hit when the reply's last 300 characters contain "?"
    turn.self.fetches_outside          p      hit when source_fetches is non-empty
    turn.self.steps                    range  hit when steps_lo <= step_count <= steps_hi        (bet v1)
    turn.self.reply_length             range  hit when reply_chars_lo <= len(final_text) <= reply_chars_hi (v1)
    turn.self.more_steps_than_usual    p      hit when step_count > reference                    (bet v2)
    turn.self.longer_reply_than_usual  p      hit when len(final_text) > reference               (bet v2)

v2 replaced the size ranges, right 1-8% of the time against a stated 80%, with "bigger than usual?"
(spec 2026-11-02-turn-self-bets-relative-size-design.md). ``reference`` is the median true size of the
same caller's executed turns placed in the REFERENCE_WINDOW strictly before this bet. Stance never sees
it, so a constant guess at the caller's own rate scores exactly 0 skill. ``observed`` holds what the
turn did whenever it executed, even when no bet was stated for that faculty: the size rows of every
executed turn are the reference history, whichever bet version placed them.

"?" is a crude stand-in for "asks a question". It is the first thing to replace, and it must not
grow into a word list.

Status: a probability bet records the event (``happened`` / ``not_happened``); it is never "broken" by a
low-p event that did not happen -- the Brier score grades the number. A range bet is ``upheld`` (the
outcome fell inside) or ``broken``. Void rows never count toward a grade:
    failed_to_execute       the turn ran but errored or was cut short (exit_code != 0, no final_text, or
                            the harness's cut-short notice in place of a reply). Includes the motor's own
                            kills (timeout, stall, context ceiling), a nonzero exit with no cancel, and
                            every old record with no ``stop_reason``.
    interrupted             the turn was stopped from outside before it finished: its record's
                            ``stop_reason`` is a known harness cancel (Juniper's stop, a client disconnect,
                            durable-runs stopping the attempt on a GPU recall, worker recovery; see
                            orion/harness/stop_reason.py). Not Orion failing, so not a failed turn.
    ungradeable_bad_record  the bet or the turn record is not a JSON object (one bad row must not stall
                            every later pass, so it settles void instead of raising)
    expired_unsettled       no turn record EXPIRY after the bet
    ungradeable_no_stated   stance stated nothing for this faculty (A1 never fills a value in; a v1 bet
                            states no relative faculty, a v2 bet no range)
    ungradeable_no_reference  fewer than MIN_REFERENCE earlier executed turns of this caller in the window

Grades, per faculty and per caller (session_id prefix: world-pulse, curiosity, self, ...; a bare UUID
session is a chat session, ``chat``; no session id is ``unknown``):
    p faculties     Brier, skill against the per-caller base rate, and the sorting score (``sort_auc``,
                    ``sort_p``: within-caller AUC and its one-sided p-value; see grading.sorting_score). Callers differ a lot (curiosity
                    fetches, chat mostly does not), so a single base rate would credit Orion for knowing
                    who called. The per-caller rate is computed on the same rows (in-sample), which
                    makes it a slightly stricter bar, not a looser one.
    range faculties realized coverage vs the stated 0.8, median width, and the mean interval score
                    (Winkler, alpha = 1 - coverage): width plus a penalty for each miss, so an always-
                    right huge range does not score well.
"""
from __future__ import annotations

import bisect
import logging
import re
import statistics
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable

from orion.prediction.grading import grade, sorting_score
from orion.harness.cut_short import CUT_SHORT_MARKER
from orion.harness.stop_reason import is_outside_stop

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

REFERENCE_WINDOW = timedelta(days=7)
MIN_REFERENCE = 20              # earlier executed turns of the caller needed before "usual" means anything

P_FACULTIES = {"turn.self.asks_question": "asks_question_p", "turn.self.fetches_outside": "fetches_outside_p"}
RANGE_FACULTIES = {"turn.self.steps": ("steps_lo", "steps_hi"),
                   "turn.self.reply_length": ("reply_chars_lo", "reply_chars_hi")}
# relative faculty -> (stated key, the size faculty whose observed value it compares to its reference)
RELATIVE_FACULTIES = {"turn.self.more_steps_than_usual": ("more_steps_than_usual_p", "turn.self.steps"),
                      "turn.self.longer_reply_than_usual": ("longer_reply_than_usual_p", "turn.self.reply_length")}
SIZE_FACULTIES = tuple(RANGE_FACULTIES)
PROBABILITY_GRADED = set(P_FACULTIES) | set(RELATIVE_FACULTIES)
FACULTIES = tuple(P_FACULTIES) + tuple(RANGE_FACULTIES) + tuple(RELATIVE_FACULTIES)
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
        # Only a known outside cancel is "interrupted"; a missing or unknown stop_reason
        # stays a failure (old records have none and are not backfilled).
        void = "interrupted" if is_outside_stop(artifact.get("stop_reason")) else "failed_to_execute"
        return void, {"exit_code": code}
    fetches = artifact.get("source_fetches")
    steps = _num(artifact.get("step_count"))
    return None, {
        "turn.self.asks_question": "?" in text[-QUESTION_TAIL:],
        "turn.self.fetches_outside": bool(fetches) if isinstance(fetches, list) else None,
        "turn.self.steps": steps,
        "turn.self.reply_length": float(len(text)),
    }


def settle(bet: dict, artifact: dict | None, now: datetime,
           references: dict[str, float | None] | None = None) -> list[dict] | None:
    """Rows for one turn's bet, or None while it may still settle (no record yet, not expired).

    ``bet``: correlation_id, session_id, created_at (aware), self_prediction (dict).
    ``references``: size faculty -> the caller's usual size before this bet (SizeHistory.reference);
    missing or None -> the relative faculties are void ``ungradeable_no_reference``."""
    references = references or {}
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
                   stated_p=None, stated_lo=None, stated_hi=None, observed=None, reference=None, hit=None,
                   void_reason=None)
        if faculty in RELATIVE_FACULTIES:
            key, size_faculty = RELATIVE_FACULTIES[faculty]
            row["claim_kind"] = "probability"
            row["stated_p"] = _num(pred.get(key))
            row["reference"] = references.get(size_faculty)
            stated = row["stated_p"] is not None
        elif faculty in P_FACULTIES:
            row["claim_kind"] = "probability"
            row["stated_p"] = _num(pred.get(P_FACULTIES[faculty]))
            stated = row["stated_p"] is not None
        else:
            row["claim_kind"] = "interval"
            lo_key, hi_key = RANGE_FACULTIES[faculty]
            row["stated_lo"], row["stated_hi"] = _num(pred.get(lo_key)), _num(pred.get(hi_key))
            stated = row["stated_lo"] is not None and row["stated_hi"] is not None
        value = obs.get(RELATIVE_FACULTIES[faculty][1] if faculty in RELATIVE_FACULTIES else faculty)
        if not void and value is not None:
            row["observed"] = float(value)   # what the turn did, recorded even when nothing was stated
        if void:
            # Voids every faculty of the turn, stated or not: the self-monitor reads "all rows
            # failed_to_execute" as one failed turn (orion/self_monitor/collect.py, turn.run, #3078)
            # and never as a failed guess; "interrupted" rows are no turn.run bet at all.
            row["void_reason"] = void
        elif not stated:
            row["void_reason"] = "ungradeable_no_stated"
        elif value is None:
            row["void_reason"] = "ungradeable_no_outcome"
        elif faculty in RELATIVE_FACULTIES and row["reference"] is None:
            row["void_reason"] = "ungradeable_no_reference"
        elif faculty in RELATIVE_FACULTIES:
            row["hit"] = value > row["reference"]   # strictly: a turn at the usual size is not bigger
        else:
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


class SizeHistory:
    """Executed turns' true sizes per (caller, size faculty), for the relative bets' reference."""

    def __init__(self, rows: Iterable[dict] = ()):
        self._by: dict[tuple[str, str], list[tuple[datetime, float]]] = defaultdict(list)
        for r in rows:
            self.add(r["caller"], r["faculty"], r["placed_at"], r["observed"])

    def add(self, caller: str, faculty: str, placed_at: datetime, observed: float | None) -> None:
        if faculty in SIZE_FACULTIES and observed is not None:
            bisect.insort(self._by[(caller, faculty)], (placed_at, float(observed)))   # kept in time order

    def reference(self, caller: str, faculty: str, before: datetime) -> float | None:
        """Median size of the caller's turns placed in [before - REFERENCE_WINDOW, before); the bet's
        own turn is never in it. None below MIN_REFERENCE turns."""
        rows = self._by.get((caller, faculty), [])
        lo = bisect.bisect_left(rows, (before - REFERENCE_WINDOW,))
        hi = bisect.bisect_left(rows, (before,))   # strictly before: a turn placed at `before` is excluded
        return statistics.median(x for _, x in rows[lo:hi]) if hi - lo >= MIN_REFERENCE else None


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
    # Sorting, within each caller: does stance give the turns where it happened the higher p?
    # Immune to a shifting base rate (11-03: replies doubled on 10-31 and the 7-day "usual" lagged).
    srt = sorting_score([r["stated_p"] for r in rows], [r["hit"] for r in rows], [r["caller"] for r in rows])
    out.update(sort_auc=srt["auc"], sort_pairs=srt["pairs"], sort_p=srt["p_value"])
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
    out = _grade_p(graded) if faculty in PROBABILITY_GRADED else _grade_range(graded)
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
                 monotonic: Callable[[], float] = time.monotonic, surprise_events_enabled: bool = False):
        self.store = store
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.verdict_ttl_sec = verdict_ttl_sec
        self._mono = monotonic
        self._live: tuple[float, dict] | None = None
        self.last_pass: dict = {}
        # Chat-turn surprise (app/turn_surprise.py): events built after each settle pass, published by
        # main.py. PREDICTION_TURN_SURPRISE_EVENTS_ENABLED.
        self.surprise_events_enabled = surprise_events_enabled
        self.pending_surprise: list = []
        self.surprise_counts: dict[str, int] = defaultdict(int)
        self.last_surprise_error: str | None = None

    def settle_pending(self) -> dict:
        """One pass: read unsettled bets with their turn records, write rows for every bet that can settle."""
        now = self.clock()
        pending = self.store.pending_turn_self_bets(now - LOOKBACK, PENDING_LIMIT)
        history = SizeHistory(self.store.turn_size_history(
            min((b["created_at"] for b in pending), default=now) - REFERENCE_WINDOW) if pending else ())
        rows, waiting = [], 0
        for bet in pending:   # oldest first, so a turn settled in this pass is history for later ones
            caller = caller_of(bet.get("session_id"))
            refs = {f: history.reference(caller, f, bet["created_at"]) for f in SIZE_FACULTIES}
            settled = settle(bet, bet.get("run_artifact"), now, refs)
            if settled is None:
                waiting += 1
                continue
            rows.extend(settled)
            for r in settled:
                history.add(r["caller"], r["faculty"], r["placed_at"], r["observed"])
        written = self.store.insert_turn_self_bets(rows) if rows else 0
        self.last_pass = {"at": now, "read": len(pending), "settled_turns": len(rows) // len(FACULTIES),
                          "waiting": waiting, "rows_written": written}
        if self.surprise_events_enabled and written:
            try:   # surprise never costs the settle pass (its rows are already written)
                self.pending_surprise.extend(self.surprise_events(rows, now))
                self.last_surprise_error = None
            except Exception as exc:  # noqa: BLE001
                self.last_surprise_error = f"{type(exc).__name__}: {exc}"[:300]
                logger.warning("turn_surprise_failed err=%s", exc)
        if written:
            logger.info("turn_self_bets_settled turns=%s rows=%s waiting=%s", len(rows) // len(FACULTIES),
                        written, waiting)
        return self.last_pass

    def surprise_events(self, rows: list[dict], now: datetime) -> list:
        """One ``PredictionTurnSurpriseV1`` per chat turn settled in this pass whose used bets carry
        information (app/turn_surprise.py). Calibration reads only bets already settled and placed
        strictly before the turn. A turn placed more than MAX_TURN_AGE ago is a backlog: counted,
        never published (the node would write an old turn as now)."""
        from app import turn_surprise as tsur
        from orion.schemas.prediction import PredictionTurnSurpriseBetV1, PredictionTurnSurpriseV1

        by_turn: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            if r["caller"] == tsur.SCOPE_CALLER:
                by_turn[r["correlation_id"]].append(r)
        if not by_turn:
            return []
        earliest = min(r["placed_at"] for trs in by_turn.values() for r in trs)
        history = tsur.CalibrationHistory(
            dict(h, stated_p=float(h["stated_p"]))
            for h in self.store.turn_surprise_calibration_rows(tsur.SCOPE_CALLER, tsur.USED_FACULTIES,
                                                                earliest - tsur.CAL_WINDOW))
        out = []
        for cid, trs in by_turn.items():
            placed = trs[0]["placed_at"]
            if now - placed > tsur.MAX_TURN_AGE:
                self.surprise_counts["stale_turn"] += 1
                continue
            scored, reason = tsur.score_turn(trs, history)
            self.surprise_counts[reason] += 1
            if scored is None:
                continue
            out.append(PredictionTurnSurpriseV1(
                correlation_id=cid, caller=tsur.SCOPE_CALLER, placed_at=placed, settled_at=now,
                bets=[PredictionTurnSurpriseBetV1(
                    faculty=b.faculty, stated_p=b.stated_p, calibrated_p=b.calibrated_p, happened=b.happened,
                    surprisal=b.surprisal, expected_surprisal=b.expected_surprisal, excess=b.excess,
                    calm_var=b.calm_var) for b in scored.bets],
                excess=scored.excess, calm_var=scored.calm_var, z=scored.z))
        return out

    def take_pending_surprise(self) -> list:
        pending, self.pending_surprise = self.pending_surprise, []
        return pending

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
