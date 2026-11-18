"""Turn surprise: how much a settled chat turn surprised Orion's own stance bets. Pure, stdlib only.

Spec: docs/superpowers/specs/2026-11-08-prediction-error-suite-rebuild-design.md ("Definition",
"Retire order" step 6). Successor of ``node:substrate.chat`` (an EWMA z of the change in two chat
pressure hints, negatives clipped: not a forecast).

Stance places private bets on every unified turn before the motor runs (``thought_decision.
self_prediction``); ``app/turn_self_bets.py`` settles them against the turn's own record. A bet is a
stated probability that something happens, stored before the outcome, settled when the turn ends:
the spec's "forecast first, clock-due outcome".

Which bets (``USED_FACULTIES``, chat turns only). Measured 2026-11-13 on every settled chat bet
(evals/run_turn_surprise_replay_eval.py), held out: each bet's log loss against the chat caller's
own base rate from the 7 days strictly before it.

    asks_question            gap +0.155 nats/bet  SE 0.035  P(better) 1.000   used
    more_steps_than_usual    gap +0.179           SE 0.031  P(better) 1.000   used
    fetches_outside          gap -0.027           SE 0.015  P(better) 0.037   not used (chat never fetches: 0 of 378)
    longer_reply_than_usual  gap -0.419           SE 0.349  P(better) 0.115   not used (worse than the base rate)

The split is not near any line: the two used kinds sit more than 4 SE above the base rate, the two
dropped ones below it. Across every caller (curiosity, world-pulse, self...) no kind beats the
per-caller base rate, so the node is chat-only. The list is fixed here; changing it is its own PR
with the eval re-run (spec rule 3: gates do not move quietly).

Recalibration. The stated numbers are not calibrated (more_steps: stated 0.17, happened 0.25), and
surprise measured against a miscalibrated p never rests at 0. So each stated p goes through a
logistic map ``q = sigmoid(a + b * logit(p))`` fitted on the same faculty's settled chat bets placed
in the ``CAL_WINDOW`` strictly before this turn (held out by time, never the turn itself), with a
ridge pull toward the identity map (``a = 0, b = 1``: the stated number is the prior, data moves it
only as far as it can). Fewer than ``MIN_CAL`` earlier bets: that faculty is skipped for this turn.

Surprise per bet, one unit (nats), calm point set by math:

    surprisal  s   = -ln q(outcome)                 (q if it happened, 1 - q if not)
    expected   H   = -q ln q - (1 - q) ln(1 - q)     (the mean of s if q is right)
    excess     s-H = -(y - q) * logit(q)             (mean 0 under calibration)
    calm_var       = q (1 - q) logit(q)^2            (its variance under calibration)

Per turn: ``z = sum(excess) / sqrt(sum(calm_var))`` over the used bets (the same z-of-summed-excess
the room/body nodes use, so one turn with two bets is one outcome, not two). A 50/50 bet has
calm_var 0 and excess 0: it carries no surprise either way; a turn whose bets are all 50/50 has no z
(no event). Rest: a turn that went the likely way reads z < 0, and the node stores
``clip(z / 3, 0, 1)``, so a calm turn rests at exactly 0. A turn where a 2 % event happened reads
z ~ 7 -> 1.0. Known edge: one bet near (not at) 0.5 is a standardised coin flip, |z| ~ 1 whichever
way it lands, so the node reads ~0.33 when the less likely side happens. Rare after recalibration
(the used kinds happen 13 % and 25 % of the time).
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Mapping, Sequence

USED_FACULTIES: tuple[str, ...] = ("turn.self.asks_question", "turn.self.more_steps_than_usual")
SCOPE_CALLER = "chat"
CAL_WINDOW = timedelta(days=7)
MIN_CAL = 20
P_CLIP = 0.01            # stance states 0.02 .. 0.98; clip only guards a stated 0 or 1
RIDGE = 1.0              # pull toward the identity map, in units of one bet's worth of evidence
# A turn placed longer ago than this when it settles is a backlog after an outage: published never,
# so a catch-up pass does not write old turns as now. Settle lag, chat, 7 d: p50 116 s, p99 897 s.
MAX_TURN_AGE = timedelta(minutes=60)


def _clip(p: float) -> float:
    return min(1.0 - P_CLIP, max(P_CLIP, float(p)))


def logit(p: float) -> float:
    q = _clip(p)
    return math.log(q / (1.0 - q))


def _sigmoid(x: float) -> float:
    x = max(-30.0, min(30.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def log_loss(p: float, hit: bool) -> float:
    q = _clip(p)
    return -math.log(q if hit else 1.0 - q)


def fit_calibration(history: Sequence[tuple[float, bool]], ridge: float = RIDGE) -> tuple[float, float]:
    """(a, b) of ``q = sigmoid(a + b * logit(p))`` by penalized maximum likelihood (Newton), ridge
    toward (0, 1). Empty history -> the identity map."""
    a, b = 0.0, 1.0
    xs = [logit(p) for p, _ in history]
    ys = [1.0 if h else 0.0 for _, h in history]
    for _ in range(100):
        ga, gb = -ridge * a, -ridge * (b - 1.0)
        haa, hab, hbb = ridge, 0.0, ridge
        for x, y in zip(xs, ys):
            q = _sigmoid(a + b * x)
            ga += y - q
            gb += (y - q) * x
            w = q * (1.0 - q)
            haa += w
            hab += w * x
            hbb += w * x * x
        det = haa * hbb - hab * hab
        if det <= 0:
            break
        da = (hbb * ga - hab * gb) / det
        db = (-hab * ga + haa * gb) / det
        a, b = a + da, b + db
        if abs(da) + abs(db) < 1e-10:
            break
    return a, b


def calibrated(p: float, ab: tuple[float, float]) -> float:
    return _sigmoid(ab[0] + ab[1] * logit(p))


@dataclass(frozen=True)
class BetTerms:
    faculty: str
    stated_p: float
    calibrated_p: float
    happened: bool
    surprisal: float
    expected_surprisal: float
    excess: float
    calm_var: float


def bet_terms(faculty: str, stated_p: float, q: float, happened: bool) -> BetTerms:
    q = _clip(q)
    lg = math.log(q / (1.0 - q))
    s = -math.log(q if happened else 1.0 - q)
    h = -q * math.log(q) - (1.0 - q) * math.log(1.0 - q)
    return BetTerms(faculty=faculty, stated_p=float(stated_p), calibrated_p=q, happened=bool(happened),
                    surprisal=s, expected_surprisal=h, excess=s - h, calm_var=q * (1.0 - q) * lg * lg)


@dataclass(frozen=True)
class TurnSurprise:
    correlation_id: str
    placed_at: datetime
    bets: tuple[BetTerms, ...]
    excess: float
    calm_var: float
    z: float


def turn_surprise(correlation_id: str, placed_at: datetime, bets: Sequence[BetTerms]) -> TurnSurprise | None:
    """One turn's joint surprise, or None when no used bet carries any (empty, or all 50/50)."""
    bets = tuple(bets)
    var = sum(b.calm_var for b in bets)
    if not bets or var <= 1e-12:
        return None
    total = sum(b.excess for b in bets)
    return TurnSurprise(correlation_id=correlation_id, placed_at=placed_at, bets=bets, excess=total,
                        calm_var=var, z=total / math.sqrt(var))


def node_value(z: float) -> float:
    """What the node stores: 3 sigma = 1.0, a calm (z <= 0) turn = exactly 0."""
    return max(0.0, min(1.0, z / 3.0))


class CalibrationHistory:
    """Settled, graded bets per used faculty (scope caller only), in time order, for the held-out map."""

    def __init__(self, rows: Iterable[Mapping] = ()):
        self._by: dict[str, list[tuple[datetime, float, bool, datetime | None]]] = defaultdict(list)
        for r in rows:
            self.add(r)

    def add(self, r: Mapping) -> None:
        if (r.get("faculty") in USED_FACULTIES and r.get("caller") == SCOPE_CALLER
                and r.get("status") in ("happened", "not_happened") and r.get("stated_p") is not None
                and r.get("hit") is not None):
            self._by[r["faculty"]].append((r["placed_at"], float(r["stated_p"]), bool(r["hit"]),
                                           r.get("settled_at")))
            self._by[r["faculty"]].sort(key=lambda t: t[0])

    def before(self, faculty: str, t: datetime, settled_by: datetime | None = None) -> list[tuple[float, bool]]:
        """Bets of this faculty placed in [t - CAL_WINDOW, t): strictly earlier turns only. With
        ``settled_by``, only bets already settled then (what the live service could have read)."""
        return [(p, h) for ts, p, h, st in self._by.get(faculty, [])
                if t - CAL_WINDOW <= ts < t and (settled_by is None or st is None or st <= settled_by)]


def score_turn(turn_rows: Sequence[Mapping], history: CalibrationHistory) -> tuple[TurnSurprise | None, str]:
    """(surprise or None, reason) for one settled turn's rows (all faculties of one correlation_id)."""
    if not turn_rows:
        return None, "no_rows"
    first = turn_rows[0]
    if first.get("caller") != SCOPE_CALLER:
        return None, "not_chat"
    terms = []
    no_cal = 0
    for r in turn_rows:
        if r.get("faculty") not in USED_FACULTIES or r.get("status") not in ("happened", "not_happened"):
            continue
        if r.get("stated_p") is None or r.get("hit") is None:
            continue
        hist = history.before(r["faculty"], r["placed_at"], r.get("settled_at"))
        if len(hist) < MIN_CAL:
            no_cal += 1
            continue
        q = calibrated(float(r["stated_p"]), fit_calibration(hist))
        terms.append(bet_terms(r["faculty"], float(r["stated_p"]), q, bool(r["hit"])))
    if not terms:
        return None, ("too_little_calibration_history" if no_cal else "no_graded_used_bet")
    out = turn_surprise(str(first["correlation_id"]), first["placed_at"], terms)
    return (out, "scored") if out is not None else (None, "no_information")


# --- held-out skill (the evidence behind USED_FACULTIES; evals/run_turn_surprise_replay_eval.py) -----
def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def heldout_skill(rows: Sequence[Mapping], faculty: str, caller: str | None = SCOPE_CALLER,
                  window: timedelta = CAL_WINDOW, min_prior: int = MIN_CAL,
                  recalibrate: bool = True) -> dict:
    """Per-bet log-loss gap of the (recalibrated) stated p over the caller's own trailing base rate.

    Positive gap = the bets know something the base rate does not. Both forecasts use only bets
    placed strictly before and already settled when this one settled (held out by time, as the live
    producer reads them). SE: the larger of the iid SE and a by-day cluster SE
    (bets in one day or session are not independent). ``prob_better`` = Phi(gap / SE): reported, no
    alpha line. ``caller=None`` pools every caller, each against its own base rate."""
    graded = sorted((r for r in rows if r.get("faculty") == faculty
                     and r.get("status") in ("happened", "not_happened") and r.get("stated_p") is not None
                     and (caller is None or r.get("caller") == caller)), key=lambda r: r["placed_at"])
    by_caller: dict[str, list[Mapping]] = defaultdict(list)
    diffs: list[tuple[object, float]] = []
    hits = []
    for r in graded:
        settled = r.get("settled_at")
        prior = [q for q in by_caller[r["caller"]] if r["placed_at"] - window <= q["placed_at"] < r["placed_at"]
                 and (settled is None or q.get("settled_at") is None or q["settled_at"] <= settled)]
        by_caller[r["caller"]].append(r)
        if len(prior) < min_prior:
            continue
        y = bool(r["hit"])
        base = (sum(1 for q in prior if q["hit"]) + 0.5) / (len(prior) + 1.0)
        p = float(r["stated_p"])
        if recalibrate:
            p = calibrated(p, fit_calibration([(float(q["stated_p"]), bool(q["hit"])) for q in prior]))
        diffs.append((r["placed_at"].date(), log_loss(base, y) - log_loss(p, y)))
        hits.append(y)
    n = len(diffs)
    if n < 2:
        return {"faculty": faculty, "n": n, "gap": None, "se": None, "prob_better": None}
    vals = [d for _, d in diffs]
    gap = statistics.fmean(vals)
    se_iid = statistics.stdev(vals) / math.sqrt(n)
    days: dict[object, float] = defaultdict(float)
    for day, d in diffs:
        days[day] += d - gap
    g = len(days)
    se_day = math.sqrt(sum(v * v for v in days.values()) * g / (g - 1)) / n if g > 1 else se_iid
    se = max(se_iid, se_day)
    return {"faculty": faculty, "n": n, "days": g, "gap": gap, "se_iid": se_iid, "se_day": se_day, "se": se,
            "prob_better": _phi(gap / se) if se > 0 else None, "rate": statistics.fmean(hits)}
